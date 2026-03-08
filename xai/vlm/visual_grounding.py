"""
Visual Grounding XAI for MedGemma (Vision-Language Model)

Implements Token-Specific CAM methods for visual grounding - shows which image regions
influenced the generation of specific tokens/labels in MedGemma's output.

Based on the HistoLens toolkit approach for MedGemma-4B:
- Computes gradient of specific token logit w.r.t. visual encoder (SigLIP) output
- Creates spatial attribution heatmap showing image regions that influenced that token
- Enables interactive "click on label → see evidence" workflow for clinical validation
- Supports multiple XAI methods: GradCAM, GradCAM++, HiResCAM, Guided GradCAM

Architecture:
MedGemma = SigLIP Encoder (vision) + Gemma Decoder (text)
Gradient Flow: Token Logit → Cross-Attention → Vision Embeddings → Spatial Heatmap

Key Improvements (HistoLens-inspired):
1. Skip CLS token (position 0) for spatial-only attribution
2. Support multiple CAM variants for different use cases
3. Reuse original generation outputs for consistent attribution
4. HiResCAM for pixel-precise localization in medical imaging
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image
from enum import Enum

logger = logging.getLogger(__name__)


class XAIMethod(Enum):
    """Available XAI methods for visual grounding"""
    GRADCAM = "gradcam"
    GRADCAM_PP = "gradcam++"
    HIRESCAM = "hirescam"
    GUIDED_GRADCAM = "guided_gradcam"


class BaseCAM:
    """Base class for CAM-based XAI methods (HistoLens-style)"""
    
    def __init__(self, model, target_layer, skip_cls_token: bool = True):
        """
        Initialize CAM generator.
        
        Args:
            model: The model to analyze
            target_layer: Target layer for gradient capture (vision encoder layer)
            skip_cls_token: Whether to skip CLS token (position 0) for spatial attribution
        """
        self.model = model
        self.target_layer = target_layer
        self.skip_cls_token = skip_cls_token
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        def forward_hook(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                self.activations = output[0]
            else:
                self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _get_spatial_features(self):
        """
        Get spatial features, optionally skipping CLS token.
        
        Returns:
            Tuple of (gradients, activations) with spatial dimensions only
        """
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients/Activations not captured. Did you call backward()?")
        
        grads = self.gradients
        acts = self.activations
        
        # Skip CLS token if present (position 0) - critical for spatial attribution
        # This is the key HistoLens insight: CLS token doesn't have spatial meaning
        if self.skip_cls_token and grads.shape[1] > 1:
            # Check if we have CLS token by looking at sequence length
            # SigLIP typically has grid_size^2 + 1 tokens (with CLS)
            seq_len = grads.shape[1]
            sqrt_check = int(np.sqrt(seq_len))
            if sqrt_check * sqrt_check != seq_len:
                # Not a perfect square, likely has CLS token
                grads = grads[:, 1:, :]  # Skip CLS
                acts = acts[:, 1:, :]    # Skip CLS
                logger.debug(f"Skipped CLS token: {seq_len} -> {grads.shape[1]} spatial tokens")
        
        return grads, acts
    
    def calculate_cam(self) -> np.ndarray:
        """Calculate CAM heatmap - to be implemented by subclasses"""
        raise NotImplementedError


class GradCAM(BaseCAM):
    """
    Standard GradCAM implementation.
    
    Computes: weights = mean(gradients) across spatial dim
              CAM = ReLU(sum(weights * activations))
    
    Good for: General overview of attention regions
    """
    
    def calculate_cam(self) -> np.ndarray:
        """Calculate GradCAM heatmap"""
        grads, acts = self._get_spatial_features()
        
        # GradCAM: compute weights as mean of gradients across patches
        weights = torch.mean(grads, dim=1)  # (batch, hidden_dim)
        
        # Weighted sum of activations
        cam = torch.einsum('bpd,bd->bp', acts, weights)  # (batch, num_patches)
        
        # ReLU and normalize
        cam = cam.squeeze().cpu().float().detach()
        cam = torch.clamp(cam, min=0)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.numpy()


class GradCAMPlusPlus(BaseCAM):
    """
    GradCAM++ implementation for better localization of multiple objects.
    
    Uses alpha-weighted positive gradients for more precise attribution.
    Reference: Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations"
    
    Good for: Multiple regions of interest, more precise boundaries
    """
    
    def calculate_cam(self) -> np.ndarray:
        """Calculate GradCAM++ heatmap"""
        grads, acts = self._get_spatial_features()
        
        # GradCAM++: use alpha-weighted positive gradients
        grads_pos = F.relu(grads)  # Only positive gradients
        
        # Compute alpha weights (importance of each gradient)
        alpha_num = grads_pos.pow(2)
        alpha_denom = 2 * grads_pos.pow(2) + torch.sum(acts * grads_pos.pow(3), dim=-1, keepdim=True)
        alpha = alpha_num / (alpha_denom + 1e-8)
        
        # Weighted gradients
        weights = torch.sum(alpha * grads_pos, dim=-1)  # (batch, num_patches)
        
        # Weighted sum with activations
        cam = torch.einsum('bp,bpd->bd', weights, acts).sum(dim=-1)
        
        # Handle shape issues
        if cam.dim() == 0:
            # Fallback to simpler computation if einsum collapses dimensions
            cam = (acts * grads_pos).sum(dim=-1).squeeze()
        
        cam = cam.cpu().float().detach().numpy()
        cam = np.maximum(cam, 0)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Ensure we return spatial map
        if cam.ndim == 0:
            cam = np.array([cam])
        
        return cam


class HiResCAM(BaseCAM):
    """
    HiResCAM implementation for pixel-precise localization.
    
    Uses element-wise multiplication instead of mean pooling for finer detail.
    Reference: Draelos & Carin, "Use HiResCAM instead of Grad-CAM"
    
    RECOMMENDED for medical imaging - preserves fine-grained spatial details.
    
    Good for: Precise tumor/lesion boundaries, small structures
    """
    
    def calculate_cam(self) -> np.ndarray:
        """Calculate HiResCAM heatmap - more precise than GradCAM"""
        grads, acts = self._get_spatial_features()
        
        # HiResCAM: element-wise multiplication, then sum over channels
        # This preserves more spatial detail than mean-pooled GradCAM
        cam = (acts * grads).sum(dim=-1)  # (batch, num_patches)
        
        cam = cam.squeeze().cpu().float().detach()
        cam = torch.clamp(cam, min=0)
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.numpy()


class GuidedBackprop:
    """
    Guided Backpropagation for pixel-level gradients.
    
    Modifies ReLU backward pass to only propagate positive gradients.
    Combined with CAM methods for Guided GradCAM.
    
    Good for: Fine-grained edge/texture attribution
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._override_relu()
    
    def _override_relu(self):
        """Override ReLU backward to only pass positive gradients"""
        def relu_backward_hook(module, grad_in, grad_out):
            return (F.relu(grad_in[0]),)
        
        # Apply to all ReLU modules in vision tower
        for module in self.model.vision_tower.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_full_backward_hook(relu_backward_hook))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def calculate_gradients(self, pixel_values: torch.Tensor) -> np.ndarray:
        """
        Get guided backprop gradients from pixel values.
        
        Args:
            pixel_values: Input image tensor with gradients enabled
            
        Returns:
            Gradient map as numpy array (H, W, C)
        """
        if pixel_values.grad is None:
            raise RuntimeError("Guided Backprop failed: pixel_values.grad is None. "
                             "Ensure pixel_values.requires_grad_(True) before forward pass.")
        
        grad = pixel_values.grad.cpu().numpy()[0]  # (C, H, W)
        return np.transpose(grad, (1, 2, 0))  # (H, W, C)


# Backward compatibility alias
GradCAMPP = GradCAMPlusPlus


def _find_grid_shape(num_patches: int) -> Tuple[int, int]:
    """Find the grid shape for reshaping patch tokens to 2D spatial map."""
    for h in range(int(np.sqrt(num_patches)), 0, -1):
        if num_patches % h == 0:
            return (h, num_patches // h)
    return (1, num_patches)


class MedGemmaVisualGrounding:
    """
    Computes visual grounding heatmaps for MedGemma token predictions.
    
    Shows which image regions contributed to specific generated tokens (labels).
    
    Key improvements (HistoLens-inspired):
    1. Reuses original generation outputs for consistent attribution
    2. Supports multiple XAI methods: GradCAM, GradCAM++, HiResCAM, Guided GradCAM
    3. Skips CLS token for spatial-only attribution
    4. Caches generation results to allow computing grounding for multiple labels efficiently
    """
    
    def __init__(self, model, processor, device: str = "cuda"):
        """
        Initialize visual grounding system.
        
        Args:
            model: MedGemma model instance (PaliGemmaForConditionalGeneration)
            processor: MedGemma processor instance
            device: Device for computation ("cuda" or "cpu")
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Cache for generation results (HistoLens approach: reuse for multiple labels)
        self._cached_inputs = None
        self._cached_generation = None
        self._cached_generated_text = None
        self._cached_prompt = None
        self._cached_image_hash = None
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Get a simple hash for caching purposes."""
        return f"{image.size}_{image.mode}_{hash(image.tobytes()[:1000])}"
    
    def compute_visual_grounding(
        self,
        image: Image.Image,
        prompt: str,
        target_token: str,
        generated_text: Optional[str] = None,
        xai_method: str = "hirescam",
        inputs_cache: Optional[Dict] = None,
        generation_cache: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding heatmap for a specific target token.
        
        IMPORTANT: For consistent attribution (HistoLens approach), provide the
        same inputs/generation used for label suggestion. This ensures the heatmap
        shows regions that actually influenced the specific label generation.
        
        Args:
            image: Input PIL image
            prompt: Text prompt used for generation
            target_token: Token/word to compute grounding for (e.g., "tumor")
            generated_text: Pre-generated text (optional, will generate if not provided)
            xai_method: XAI method to use ("gradcam", "gradcam++", "hirescam", "guided_gradcam")
            inputs_cache: Cached inputs from original generation (for consistency)
            generation_cache: Cached generation tensor from original generation
        
        Returns:
            Tuple of (heatmap, metadata):
                - heatmap: np.ndarray of shape (H, W) with spatial attribution
                - metadata: dict with statistics and information
        """
        try:
            # Check cache or use provided inputs
            image_hash = self._get_image_hash(image)
            use_cache = (
                inputs_cache is not None and generation_cache is not None
            ) or (
                self._cached_inputs is not None and 
                self._cached_prompt == prompt and
                self._cached_image_hash == image_hash
            )
            
            if inputs_cache is not None and generation_cache is not None:
                # Use provided cache (best practice - from label generation)
                inputs = inputs_cache
                generation = generation_cache
                generated_text_local = generated_text or self._decode_generation(inputs, generation)
                logger.info("Using provided inputs/generation cache for consistent attribution")
            elif use_cache and self._cached_inputs is not None:
                # Use internal cache
                inputs = self._cached_inputs
                generation = self._cached_generation
                generated_text_local = self._cached_generated_text
                logger.info("Using cached inputs/generation for consistent attribution")
            else:
                # Generate fresh (less ideal but necessary if no cache available)
                logger.info("Generating fresh text for visual grounding (cache miss)")
                generated_text_local, inputs, generation = self._generate_text_with_outputs(image, prompt)
                
                # Cache for future calls
                self._cached_inputs = inputs
                self._cached_generation = generation
                self._cached_generated_text = generated_text_local
                self._cached_prompt = prompt
                self._cached_image_hash = image_hash
            
            if generated_text_local is None or inputs is None or generation is None:
                return None, {"error": "Failed to generate text", "success": False}
            
            logger.info(f"Generated text: {generated_text_local[:150]}...")
            
            # Find target token in generated text
            token_info = self._find_token_in_text(target_token, generated_text_local)
            if token_info is None:
                logger.warning(f"Token '{target_token}' not found in generated text")
                return None, {
                    "error": f"Token '{target_token}' not found in generated text",
                    "generated_text": generated_text_local[:200],
                    "success": False
                }
            
            # Find token positions in the generation
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = generation[0][input_len:]
            full_generated_ids = torch.cat([inputs['input_ids'][0], generated_ids], dim=0)
            
            # Encode target text to token IDs
            target_ids = self.processor.tokenizer.encode(target_token, add_special_tokens=False)
            target_ids_tensor = torch.tensor(target_ids).to(self.device)
            
            logger.info(f"Target '{target_token}' -> {len(target_ids)} tokens")
            
            # Find token position in GENERATED part only
            found, start_idx, end_idx = self._find_token_position(
                full_generated_ids, target_ids_tensor, input_len, target_token
            )
            
            if not found:
                return None, {
                    "error": f"Token '{target_token}' not found in generated sequence",
                    "success": False,
                    "generated_text": generated_text_local[:200]
                }
            
            # Create labels tensor (only target token positions, rest are -100)
            labels = torch.full_like(full_generated_ids, -100)
            labels[start_idx:end_idx] = full_generated_ids[start_idx:end_idx]
            
            valid_labels = (labels != -100).sum().item()
            if valid_labels == 0:
                return None, {"error": "Label creation failed", "success": False}
            
            logger.info(f"Created labels with {valid_labels} valid positions")
            
            # Compute CAM using selected method
            return self._compute_cam_heatmap(
                inputs=inputs,
                full_generated_ids=full_generated_ids,
                labels=labels,
                image=image,
                target_token=target_token,
                target_ids=target_ids,
                start_idx=start_idx,
                end_idx=end_idx,
                xai_method=xai_method,
                generated_text=generated_text_local
            )
        
        except Exception as e:
            logger.error(f"Error computing visual grounding: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            try:
                self.model.eval()
            except:
                pass
            return None, {"error": str(e), "success": False}
    
    def _decode_generation(self, inputs, generation) -> str:
        """Decode generation tensor to text."""
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = generation[0][input_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)
    
    def _find_token_position(
        self,
        full_generated_ids: torch.Tensor,
        target_ids_tensor: torch.Tensor,
        input_len: int,
        target_token: str
    ) -> Tuple[bool, int, int]:
        """Find position of target token in generated sequence."""
        
        # Search for exact match in generated portion only
        for i in range(input_len, len(full_generated_ids) - len(target_ids_tensor) + 1):
            if torch.equal(full_generated_ids[i:i+len(target_ids_tensor)], target_ids_tensor):
                logger.info(f"Exact token match at positions {i}-{i+len(target_ids_tensor)}")
                return True, i, i + len(target_ids_tensor)
        
        # Fallback: substring search in generated part
        logger.info("Exact tokens not found, trying substring search...")
        target_lower = target_token.lower().strip()
        
        best_match_score = 0
        best_start = 0
        best_end = 0
        
        for i in range(input_len, len(full_generated_ids)):
            for length in range(1, min(10, len(full_generated_ids) - i + 1)):
                j = i + length
                subseq = full_generated_ids[i:j]
                decoded = self.processor.decode(subseq, skip_special_tokens=True).lower().strip()
                
                if len(decoded) > 0 and target_lower in decoded:
                    score = len(target_lower) / max(len(decoded), 1)
                    if decoded == target_lower:
                        score = 2.0
                    
                    if score > best_match_score:
                        best_match_score = score
                        best_start = i
                        best_end = j
        
        if best_match_score > 0:
            matched = self.processor.decode(full_generated_ids[best_start:best_end], skip_special_tokens=True)
            logger.info(f"Substring match (score={best_match_score:.2f}): '{matched}'")
            return True, best_start, best_end
        
        return False, 0, 0
    
    def _compute_cam_heatmap(
        self,
        inputs: Dict,
        full_generated_ids: torch.Tensor,
        labels: torch.Tensor,
        image: Image.Image,
        target_token: str,
        target_ids: List[int],
        start_idx: int,
        end_idx: int,
        xai_method: str,
        generated_text: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute CAM heatmap using the specified XAI method.
        
        Supports: gradcam, gradcam++, hirescam, guided_gradcam
        """
        # Get target layer (vision encoder last layer)
        target_layer = self.model.vision_tower.vision_model.encoder.layers[-1]
        
        # Select CAM class based on method
        xai_method_lower = xai_method.lower()
        if xai_method_lower == "gradcam++":
            cam_class = GradCAMPlusPlus
        elif xai_method_lower == "hirescam":
            cam_class = HiResCAM
        else:  # Default to GradCAM
            cam_class = GradCAM
        
        # Initialize CAM generator with CLS token skipping
        cam_generator = cam_class(self.model, target_layer, skip_cls_token=True)
        
        # Initialize Guided Backprop if needed
        gbp_generator = None
        if xai_method_lower == "guided_gradcam":
            gbp_generator = GuidedBackprop(self.model)
        
        try:
            # Prepare pixel values (enable gradients for guided backprop)
            pixel_values = inputs['pixel_values'].to(self.device)
            if xai_method_lower == "guided_gradcam":
                pixel_values = pixel_values.clone().requires_grad_(True)
            
            # Set model to train mode for gradient computation
            self.model.train()
            
            with torch.enable_grad():
                outputs = self.model(
                    input_ids=full_generated_ids.unsqueeze(0),
                    pixel_values=pixel_values,
                    labels=labels.unsqueeze(0)
                )
                loss = outputs.loss
            
            if torch.isnan(loss):
                logger.error("Loss is NaN!")
                return None, {"error": "NaN loss encountered", "success": False}
            
            logger.info(f"Loss: {loss.item():.4f}")
            
            # Backpropagate
            self.model.zero_grad()
            loss.backward(retain_graph=(gbp_generator is not None))
            
            # Compute CAM
            cam = cam_generator.calculate_cam()
            
            # Reshape to 2D grid
            h, w = _find_grid_shape(cam.shape[0])
            cam_2d = cam.reshape(h, w)
            
            # Compute guided gradients if requested
            saliency_map = None
            if gbp_generator is not None:
                try:
                    saliency_map = gbp_generator.calculate_gradients(pixel_values)
                    # Convert to grayscale magnitude
                    saliency_map = np.abs(saliency_map).max(axis=-1)
                    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
                except Exception as e:
                    logger.warning(f"Guided backprop failed: {e}, using CAM only")
            
            # Resize to image size
            cam_resized = cv2.resize(cam_2d, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
            
            # For Guided GradCAM, combine saliency and CAM
            if saliency_map is not None:
                # Resize saliency to match
                saliency_resized = cv2.resize(saliency_map, (image.width, image.height))
                cam_resized = cam_resized * saliency_resized
                # Re-normalize
                if cam_resized.max() > 0:
                    cam_resized = cam_resized / cam_resized.max()
            
            # Restore model to eval mode
            self.model.eval()
            
            # Compile metadata
            method_display = {
                "gradcam": "GradCAM",
                "gradcam++": "GradCAM++",
                "hirescam": "HiResCAM (recommended)",
                "guided_gradcam": "Guided GradCAM"
            }.get(xai_method_lower, xai_method)
            
            metadata = {
                "target_token": target_token,
                "token_id": target_ids[0] if len(target_ids) > 0 else -1,
                "generated_text": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                "token_position": f"{start_idx}-{end_idx}",
                "loss": float(loss.item()),
                "mean_attribution": float(cam_resized.mean()),
                "max_attribution": float(cam_resized.max()),
                "attribution_std": float(cam_resized.std()),
                "success": True,
                "method": method_display,
                "xai_method": xai_method_lower,
                "skip_cls_token": True
            }
            
            logger.info(f"Visual grounding ({method_display}) for '{target_token}': "
                       f"mean={metadata['mean_attribution']:.3f}, max={metadata['max_attribution']:.3f}")
            
            return cam_resized, metadata
        
        finally:
            # Always cleanup
            cam_generator.remove_hooks()
            if gbp_generator is not None:
                gbp_generator.remove_hooks()
            self.model.eval()
    
    def _generate_text_with_outputs(self, image: Image.Image, prompt: str, max_new_tokens: int = 100) -> Tuple[str, dict, torch.Tensor]:
        """Generate text using MedGemma model and return inputs and generation."""
        try:
            # Apply chat template with image token (MedGemma format)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs with the formatted text
            inputs = self.processor(
                text=input_text,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1
                )
            
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = generation[0][input_len:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            self.last_generated_text = generated_text
            self.last_token_ids = generated_ids
            
            logger.info(f"Generated text for visual grounding: {generated_text[:100]}...")
            return generated_text, inputs, generation
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return "", None, None
    
    def _generate_text(self, image: Image.Image, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text using MedGemma model with proper chat template."""
        try:
            # Apply chat template with image token (MedGemma format)
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs with the formatted text
            inputs = self.processor(
                text=input_text,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1
                )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            self.last_generated_text = generated_text
            self.last_token_ids = generated_ids
            
            logger.info(f"Generated text for visual grounding: {generated_text[:100]}...")
            return generated_text
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return ""
    
    def _find_token_in_text(self, target_token: str, generated_text: str) -> Optional[Dict[str, Any]]:
        """Find token position in generated text."""
        target_lower = target_token.lower().strip()
        text_lower = generated_text.lower()
        
        if target_lower in text_lower:
            start_pos = text_lower.index(target_lower)
            return {
                "token": target_token,
                "start": start_pos,
                "end": start_pos + len(target_token),
                "found": True
            }
        
        return None
    
    def _get_token_id(self, token: str) -> Optional[int]:
        """Get token ID from vocabulary."""
        try:
            token_ids = self.processor.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 0:
                return token_ids[0]
            return None
        except Exception as e:
            logger.warning(f"Could not get token ID for '{token}': {e}")
            return None
    
    def _get_token_position(
        self,
        input_ids: torch.Tensor,
        target_token: str,
        token_info: Dict[str, Any]
    ) -> Optional[int]:
        """
        Find position of target token in input sequence.
        This is approximate - we look for the token ID in the sequence.
        """
        try:
            token_id = self._get_token_id(target_token)
            if token_id is None:
                return None
            
            # Find first occurrence of this token_id in the sequence
            input_ids_list = input_ids[0].tolist()
            if token_id in input_ids_list:
                return input_ids_list.index(token_id)
            
            return None
        
        except Exception as e:
            logger.warning(f"Could not find token position: {e}")
            return None


def create_visual_grounding_overlay(
    base_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Create visual grounding overlay with heatmap on base image.
    
    Args:
        base_image: Original image as RGB numpy array
        heatmap: Attribution heatmap (H, W) normalized to [0, 1]
        alpha: Blending factor (default: 0.5)
        colormap: Matplotlib colormap name (default: "jet")
    
    Returns:
        RGB image with heatmap overlay
    """
    import cv2
    import matplotlib.pyplot as plt
    
    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Ensure base image is uint8 RGB
    if base_image.dtype != np.uint8:
        base_image = ((base_image - base_image.min()) / (base_image.max() - base_image.min()) * 255).astype(np.uint8)
    
    if len(base_image.shape) == 2:
        base_image = np.stack([base_image] * 3, axis=-1)
    
    # Resize heatmap to match base image if needed
    if heatmap_colored.shape[:2] != base_image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (base_image.shape[1], base_image.shape[0]))
    
    # Blend
    result = cv2.addWeighted(base_image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return result
