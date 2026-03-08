"""
Visual Grounding XAI for MedGemma-4B (Gemma3 VLM)

This module implements two principled approaches for VLM explainability:

Strategy 1: Input × Gradient (RECOMMENDED)
==========================================
The standard VLM XAI approach. Computes gradients of the target word's logit
with respect to the projected image tokens entering the LLM.

Formula: Score = ImageTokens × ∂(Logit_target)/∂(ImageTokens)

This directly answers: "Which image regions CAUSED this word to be generated?"

Strategy 2: Decoder Attention (SDPA Workaround)  
================================================
Extracts attention weights from the LLM decoder layers to see which image tokens
the model "looked at" when generating the target word.

Requires disabling FlashAttention/SDPA temporarily.

MedGemma Architecture:
----------------------
- Vision Tower: SiglipVisionModel (896×896 image → 64×64 patches = 4096 tokens)
- Projector: AvgPool2d(4,4) + Linear (4096 → 256 tokens with 2560 dim)
- Language Model: Gemma3TextModel (decoder)

Final image token grid: 16×16 = 256 tokens entering LLM
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class VLMVisualGrounding:
    """
    Proper VLM Visual Grounding using Input×Gradient and Decoder Attention.
    
    This is the correct approach for VLMs - we analyze what happens in the DECODER
    (which generates text), not just the ENCODER (which extracts features).
    """
    
    def __init__(self, model, processor, device: str = "cuda"):
        """
        Initialize VLM Visual Grounding.
        
        Args:
            model: MedGemma model (Gemma3ForConditionalGeneration)
            processor: MedGemma processor  
            device: Device for computation
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Model architecture info
        self._init_model_config()
        
        # Cache for fast repeated XAI
        self._cache = {
            'inputs': None,
            'prompt': None,
            'image_hash': None,
            'generated_text': None,
            'image_token_embeds': None,  # Cached projected image tokens
            'generation_ids': None
        }
        
        # Hook storage for gradient computation
        self._hooks = []
        self._activations = {}
        self._gradients = {}
        
        logger.info(f"VLM Visual Grounding initialized: {self.num_image_tokens} image tokens "
                   f"({self.grid_size}×{self.grid_size} grid)")
    
    def _init_model_config(self):
        """Extract model configuration for proper XAI."""
        try:
            # Get vision config
            vision_cfg = self.model.config.vision_config
            self.image_size = vision_cfg.image_size  # 896
            self.patch_size = vision_cfg.patch_size  # 14
            self.vision_hidden_size = vision_cfg.hidden_size  # 1152
            
            # SigLIP output grid
            self.vision_grid = self.image_size // self.patch_size  # 64
            self.vision_patches = self.vision_grid ** 2  # 4096
            
            # After AvgPool2d(4,4) in projector
            self.pool_kernel = 4
            self.grid_size = self.vision_grid // self.pool_kernel  # 16
            self.num_image_tokens = self.grid_size ** 2  # 256
            
            # LLM hidden size (for projector output)
            self.llm_hidden_size = self.model.config.text_config.hidden_size  # 2560
            
            logger.info(f"Model config: image={self.image_size}, patches={self.vision_patches}, "
                       f"after_pool={self.num_image_tokens}, llm_dim={self.llm_hidden_size}")
                       
        except Exception as e:
            logger.warning(f"Could not read full model config: {e}. Using defaults.")
            self.image_size = 896
            self.patch_size = 14
            self.vision_grid = 64
            self.vision_patches = 4096
            self.pool_kernel = 4
            self.grid_size = 16
            self.num_image_tokens = 256
            self.llm_hidden_size = 2560
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Get a hash for caching."""
        return f"{image.size}_{image.mode}_{hash(image.tobytes()[:1000])}"
    
    def cache_generation_outputs(
        self,
        image: Image.Image,
        prompt: str,
        generated_text: str,
        inputs: Dict[str, torch.Tensor],
        generation_ids: torch.Tensor
    ):
        """
        Cache generation outputs for fast repeated XAI.
        
        Call this after generating labels to enable fast XAI on those labels.
        """
        self._cache['inputs'] = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}
        self._cache['prompt'] = prompt
        self._cache['image_hash'] = self._get_image_hash(image)
        self._cache['generated_text'] = generated_text
        self._cache['generation_ids'] = generation_ids.clone() if generation_ids is not None else None
        self._cache['preprocessed_image'] = image
        
        logger.info(f"Cached generation for XAI: '{generated_text[:50]}...'")
    
    def clear_cache(self):
        """Clear the cache."""
        for key in self._cache:
            self._cache[key] = None
        self._clear_hooks()
        logger.info("Cleared VLM visual grounding cache")
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}
        self._gradients = {}
    
    # =========================================================================
    # Strategy 1: Input × Gradient (The "Real" Solution)
    # =========================================================================
    
    def compute_input_gradient(
        self,
        image: Image.Image,
        prompt: str,
        target_label: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding using Input × Gradient method.
        
        This is the STANDARD approach for VLM XAI:
        1. Hook the projector output (image tokens entering LLM)
        2. Forward pass to get logits for target word
        3. Backward pass to compute gradients w.r.t. image tokens
        4. Score = ImageTokens × Gradient
        
        Args:
            image: Input image (PIL)
            prompt: Text prompt
            target_label: The label/word to explain (e.g., "tumor")
            
        Returns:
            Tuple of (heatmap, metadata)
        """
        self._clear_hooks()
        
        try:
            # Prepare inputs
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            input_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=input_text, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Find where the projector outputs image tokens
            # In Gemma3, the flow is: vision_tower → multi_modal_projector → language_model
            projector = self.model.model.multi_modal_projector
            
            # Storage for hooked values
            image_token_embeds = None
            
            def projector_hook(module, input, output):
                nonlocal image_token_embeds
                # Output is (batch, num_image_tokens, llm_hidden_dim)
                # Shape: (1, 256, 2560)
                image_token_embeds = output
                # Enable gradients for this tensor
                if not output.requires_grad:
                    output.requires_grad_(True)
                return output
            
            # Register hook on projector
            # The projector's forward() returns the projected image tokens
            hook = projector.register_forward_hook(projector_hook)
            self._hooks.append(hook)
            
            # Get target token ID(s)
            target_ids = self.processor.tokenizer.encode(
                target_label, add_special_tokens=False
            )
            if not target_ids:
                return None, {'success': False, 'error': f'Could not tokenize "{target_label}"'}
            
            logger.info(f"Target '{target_label}' tokenized to IDs: {target_ids}")
            
            # Forward pass with gradient tracking
            self.model.eval()
            
            # We need to generate up to and including the target token
            # Then compute gradient of target token logit w.r.t. image tokens
            
            with torch.enable_grad():
                # First, do a forward pass to get the vocabulary logits
                # We'll use teacher forcing approach: feed the full prompt + partial generation
                # and look at logits for the target token
                
                outputs = self.model(
                    **inputs,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True
                )
                
                # Logits shape: (batch, seq_len, vocab_size)
                logits = outputs.logits
                
                # The logits at position -1 predict the next token
                # We want the logit for our target token
                next_token_logits = logits[0, -1, :]  # (vocab_size,)
                
                # Get the logit for the first token of our target
                target_token_id = target_ids[0]
                target_logit = next_token_logits[target_token_id]
                
                # Now we need to get gradients w.r.t. the image token embeddings
                # The issue: image_token_embeds was created during forward pass
                # and may not be in the computation graph properly
                
                # Alternative: Use the input embeddings directly
                # Get the embeddings before they enter the LLM
                
                if image_token_embeds is None:
                    self._clear_hooks()
                    return None, {'success': False, 'error': 'Could not capture image token embeddings'}
                
                # Make sure we can compute gradients
                if not image_token_embeds.requires_grad:
                    logger.warning("Image token embeds don't require grad, enabling...")
                    image_token_embeds = image_token_embeds.detach().requires_grad_(True)
                    
                    # Need to re-run forward with grad-enabled embeds
                    # This is tricky - let's use an alternative approach
            
            # Alternative: Hook into the input of the language model
            # and compute gradient there
            return self._compute_input_gradient_v2(image, prompt, target_label, inputs)
            
        except Exception as e:
            logger.error(f"Error in input×gradient: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
        finally:
            self._clear_hooks()
    
    def _compute_input_gradient_v2(
        self,
        image: Image.Image,
        prompt: str,
        target_label: str,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Input × Gradient using a cleaner approach.
        
        We create a custom forward that lets us:
        1. Extract image token embeddings with gradients enabled
        2. Compute target logit
        3. Backpropagate to get gradients
        """
        self._clear_hooks()
        
        try:
            # Get target token ID
            target_ids = self.processor.tokenizer.encode(target_label, add_special_tokens=False)
            if not target_ids:
                return None, {'success': False, 'error': f'Could not tokenize "{target_label}"'}
            
            target_token_id = target_ids[0]
            
            # Extract image features manually so we can control gradients
            with torch.no_grad():
                # Vision tower processes the image
                pixel_values = inputs['pixel_values']
                vision_outputs = self.model.model.vision_tower(pixel_values)
                
                # Get the hidden states (before pooling in projector)
                image_features = vision_outputs.last_hidden_state  # (1, 4096, 1152)
            
            # Now process through projector with gradients enabled
            image_features = image_features.requires_grad_(True)
            
            # Apply projector manually
            projector = self.model.model.multi_modal_projector
            
            # Projector typically does: reshape → pool → linear projection
            # Let's trace the exact operations
            batch_size = image_features.shape[0]
            
            # Reshape for pooling: (B, H*W, C) → (B, C, H, W)
            h = w = self.vision_grid  # 64
            image_features_spatial = image_features.transpose(1, 2).view(batch_size, -1, h, w)
            
            # Apply pooling
            pooled = projector.avg_pool(image_features_spatial)  # (B, C, 16, 16)
            
            # Flatten spatial dims
            pooled_flat = pooled.flatten(2).transpose(1, 2)  # (B, 256, 1152)
            
            # Apply normalization if present
            if hasattr(projector, 'mm_soft_emb_norm'):
                pooled_flat = projector.mm_soft_emb_norm(pooled_flat)
            
            # Apply linear projection via weight matrix
            # mm_input_projection_weight: (1152, 2560) - need to check transpose
            if hasattr(projector, 'mm_input_projection_weight'):
                weight = projector.mm_input_projection_weight
                # (B, 256, 1152) @ (1152, 2560) = (B, 256, 2560)
                # But weight is (1152, 2560), so we need (B, 256, 1152) @ (1152, 2560)
                if weight.shape[0] == self.vision_hidden_size:
                    image_embeds = torch.matmul(pooled_flat.float(), weight.float())
                else:
                    image_embeds = torch.matmul(pooled_flat.float(), weight.T.float())
            else:
                # Fallback - just use pooled features
                image_embeds = pooled_flat
            
            # image_embeds: (B, 256, 2560) - the projected image tokens
            
            # Now we need to feed this into the language model with text tokens
            # Get text embeddings
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Get the text embedding layer
            embed_tokens = self.model.model.language_model.get_input_embeddings()
            text_embeds = embed_tokens(input_ids)  # (B, seq_len, 2560)
            
            # Find where image tokens should go
            # In Gemma3, the image tokens replace a special image token in the input
            # Let's check for image token position
            
            # For simplicity, assume image tokens are prepended
            # Concatenate: [image_embeds, text_embeds]
            # But this might not match the actual model behavior
            
            # Actually, let's use a gradient-based approach that works with the existing forward pass
            # by hooking into the embedding layer
            
            return self._compute_gradient_via_embedding_hook(image, prompt, target_label, inputs)
            
        except Exception as e:
            logger.error(f"Error in input×gradient v2: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
    
    def _compute_gradient_via_embedding_hook(
        self,
        image: Image.Image,
        prompt: str,
        target_label: str,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute gradients by hooking the embedding layer.
        
        This approach:
        1. Makes pixel_values require gradients (the IMAGE input)
        2. Runs forward pass - gradients flow through vision tower → projector → LLM
        3. Backprops from target logit
        4. Aggregates gradients at the projected image token level
        
        Key insight: We need gradients w.r.t. the IMAGE, not the text embeddings.
        """
        self._clear_hooks()
        
        try:
            # Get target token ID
            target_ids = self.processor.tokenizer.encode(target_label, add_special_tokens=False)
            if not target_ids:
                return None, {'success': False, 'error': f'Could not tokenize "{target_label}"'}
            
            target_token_id = target_ids[0]
            logger.info(f"Computing gradient for token '{target_label}' (ID: {target_token_id})")
            
            # Storage for projected image embeddings and their gradients
            projected_embeds = [None]
            projected_grads = [None]
            
            # Hook the projector output to capture image token embeddings
            projector = self.model.model.multi_modal_projector
            
            def projector_forward_hook(module, input, output):
                # Output: projected image embeddings (B, num_image_tokens, hidden_dim)
                projected_embeds[0] = output
            
            def projector_backward_hook(module, grad_input, grad_output):
                # grad_output: gradient w.r.t. projector output
                if isinstance(grad_output, tuple):
                    projected_grads[0] = grad_output[0]
                else:
                    projected_grads[0] = grad_output
            
            # Register hooks on projector
            fwd_hook = projector.register_forward_hook(projector_forward_hook)
            bwd_hook = projector.register_full_backward_hook(projector_backward_hook)
            self._hooks.extend([fwd_hook, bwd_hook])
            
            # CRITICAL: Make pixel_values require gradients
            # This creates a computation graph from input image → vision tower → projector → LLM → logits
            self.model.eval()
            
            # Clone inputs and enable gradients on pixel_values
            pixel_values = inputs['pixel_values'].clone().requires_grad_(True)
            grad_inputs = {k: v for k, v in inputs.items()}
            grad_inputs['pixel_values'] = pixel_values
            
            # Forward pass WITH gradient tracking
            outputs = self.model(
                **grad_inputs,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=True
            )
            
            # Get logit for target token
            logits = outputs.logits  # (B, seq_len, vocab_size)
            target_logit = logits[0, -1, target_token_id]
            
            logger.info(f"Target logit value: {target_logit.item():.4f}")
            logger.info(f"Target logit requires_grad: {target_logit.requires_grad}")
            
            # Check if we have a gradient path
            if not target_logit.requires_grad:
                self._clear_hooks()
                return None, {'success': False, 'error': 'No gradient path from image to logits'}
            
            # Backward pass
            target_logit.backward()
            
            # Check if we captured the projector output and gradients
            if projected_embeds[0] is None:
                self._clear_hooks()
                return None, {'success': False, 'error': 'Could not capture projected embeddings'}
            
            if projected_grads[0] is None:
                self._clear_hooks()
                return None, {'success': False, 'error': 'Could not capture gradients'}
            
            embeds = projected_embeds[0]
            grads = projected_grads[0]
            
            logger.info(f"Projected embeddings shape: {embeds.shape}")
            logger.info(f"Projected gradients shape: {grads.shape}")
            
            # Compute input × gradient for image tokens
            # embeds and grads shape: (B, num_image_tokens, hidden_dim)
            # Convert to float32 for computation (BFloat16 causes issues)
            input_grad = embeds.float() * grads.float()  # Element-wise product
            
            # Sum over hidden dimension to get per-token attribution
            token_attribution = input_grad.sum(dim=-1).abs()  # (B, num_image_tokens)
            token_attribution = token_attribution[0]  # (num_image_tokens,)
            
            num_tokens = token_attribution.shape[0]
            logger.info(f"Image token attribution shape: {num_tokens}, "
                       f"mean: {token_attribution.mean():.6f}, max: {token_attribution.max():.6f}")
            
            # Reshape to spatial grid and normalize
            # Convert to float32 for numpy (BFloat16 not supported)
            heatmap = self._attribution_to_spatial(
                token_attribution.detach().cpu().float().numpy()
            )
            
            return heatmap, {
                'success': True,
                'method': 'input_gradient',
                'target_token': target_label,
                'target_logit': float(target_logit.item()),
                'num_image_tokens': num_tokens,
                'mean_attribution': float(heatmap.mean()),
                'max_attribution': float(heatmap.max())
            }
            
        except Exception as e:
            logger.error(f"Error in gradient via embedding hook: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
        finally:
            self._clear_hooks()
    
    # =========================================================================
    # Strategy 2: Decoder Attention (SDPA Workaround)
    # =========================================================================
    
    def compute_decoder_attention(
        self,
        image: Image.Image,
        prompt: str,
        target_label: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding using decoder attention weights.
        
        This extracts attention from the LLM decoder to see which image tokens
        the model attended to when generating the output.
        
        Uses SDPA workaround: temporarily disables FlashAttention to get weights.
        
        Args:
            image: Input image (PIL)
            prompt: Text prompt
            target_label: The label to explain (for logging only)
            
        Returns:
            Tuple of (heatmap, metadata)
        """
        try:
            # Prepare inputs
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            input_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=input_text, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # CRITICAL: Change attention implementation to "eager" temporarily
            # The torch.backends.cuda.sdp_kernel() context manager doesn't work
            # because HuggingFace checks config._attn_implementation BEFORE running attention
            original_attn_impl = getattr(self.model.config, '_attn_implementation', None)
            original_text_attn = getattr(self.model.config.text_config, '_attn_implementation', None)
            
            try:
                # Set to eager attention (allows output_attentions=True)
                self.model.config._attn_implementation = "eager"
                if hasattr(self.model.config, 'text_config'):
                    self.model.config.text_config._attn_implementation = "eager"
                
                # Also need to update each layer's attention implementation
                # This is necessary because some models cache the implementation
                for layer in self.model.model.language_model.layers:
                    if hasattr(layer, 'self_attn'):
                        if hasattr(layer.self_attn, 'config'):
                            layer.self_attn.config._attn_implementation = "eager"
                
                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True
                    )
            finally:
                # Restore original attention implementation
                if original_attn_impl is not None:
                    self.model.config._attn_implementation = original_attn_impl
                if original_text_attn is not None and hasattr(self.model.config, 'text_config'):
                    self.model.config.text_config._attn_implementation = original_text_attn
            
            # Extract attention from decoder layers
            attentions = outputs.attentions
            if attentions is None or len(attentions) == 0:
                return None, {'success': False, 'error': 'No attention weights available'}
            
            logger.info(f"Got {len(attentions)} attention layers")
            
            # Use last few layers (they capture high-level semantics)
            num_layers = len(attentions)
            layers_to_use = attentions[num_layers // 2:]  # Last half
            
            # Each attention: (batch, num_heads, seq_len, seq_len)
            attn_stack = []
            for attn in layers_to_use:
                if attn is not None:
                    attn_stack.append(attn)
            
            if not attn_stack:
                return None, {'success': False, 'error': 'No valid attention layers'}
            
            # Average over layers and heads
            attn_avg = torch.stack(attn_stack).mean(dim=0)  # (B, H, S, S)
            attn_avg = attn_avg.mean(dim=1)  # (B, S, S) - average over heads
            
            # Find image token positions (same logic as gradient method)
            input_ids = inputs['input_ids'][0]
            
            if 'token_type_ids' in inputs:
                token_types = inputs['token_type_ids'][0]
                image_mask = (token_types == 1)
                image_positions = torch.where(image_mask)[0]
            else:
                IMAGE_PLACEHOLDER_ID = 262144
                image_mask = (input_ids == IMAGE_PLACEHOLDER_ID)
                image_positions = torch.where(image_mask)[0]
            
            if len(image_positions) == 0:
                logger.warning("No image tokens found, using first positions")
                start_pos, end_pos = 0, self.num_image_tokens
            else:
                start_pos = image_positions[0].item()
                end_pos = image_positions[-1].item() + 1
                logger.info(f"Image tokens at positions {start_pos}-{end_pos}")
            
            # Get attention from last token to image tokens
            last_token_attn = attn_avg[0, -1, start_pos:end_pos]  # (num_image_tokens,)
            
            logger.info(f"Attention to image tokens: mean={last_token_attn.mean():.6f}, "
                       f"max={last_token_attn.max():.6f}")
            
            # Reshape to spatial grid
            heatmap = self._attribution_to_spatial(
                last_token_attn.detach().cpu().numpy()
            )
            
            return heatmap, {
                'success': True,
                'method': 'decoder_attention',
                'target_token': target_label,
                'num_layers_used': len(layers_to_use),
                'mean_attribution': float(heatmap.mean()),
                'max_attribution': float(heatmap.max())
            }
            
        except Exception as e:
            logger.error(f"Error in decoder attention: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def compute_visual_grounding(
        self,
        image: Image.Image,
        prompt: str,
        target_label: str,
        method: str = "input_gradient"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding for a target label.
        
        Args:
            image: Input image (PIL)
            prompt: Text prompt
            target_label: The label/word to explain
            method: XAI method:
                - "input_gradient" (RECOMMENDED): Input × Gradient on projected tokens
                - "decoder_attention": Decoder attention weights (SDPA workaround)
                
        Returns:
            Tuple of (heatmap, metadata)
        """
        logger.info(f"Computing VLM visual grounding: method={method}, target='{target_label}'")
        
        if method == "input_gradient":
            return self.compute_input_gradient(image, prompt, target_label)
        elif method == "decoder_attention":
            return self.compute_decoder_attention(image, prompt, target_label)
        else:
            return None, {'success': False, 'error': f'Unknown method: {method}'}
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _attribution_to_spatial(self, attribution: np.ndarray) -> np.ndarray:
        """
        Convert 1D attribution vector to 2D spatial heatmap.
        
        Args:
            attribution: 1D array of shape (num_image_tokens,)
            
        Returns:
            2D heatmap normalized to [0, 1]
        """
        # Ensure we have the right number of values
        expected_tokens = self.num_image_tokens  # 256
        if len(attribution) != expected_tokens:
            logger.warning(f"Attribution length {len(attribution)} != expected {expected_tokens}")
            # Pad or truncate
            if len(attribution) < expected_tokens:
                attribution = np.pad(attribution, (0, expected_tokens - len(attribution)))
            else:
                attribution = attribution[:expected_tokens]
        
        # Reshape to grid
        heatmap = attribution.reshape(self.grid_size, self.grid_size)
        
        # Normalize to [0, 1]
        hmap_min = heatmap.min()
        hmap_max = heatmap.max()
        if hmap_max > hmap_min:
            heatmap = (heatmap - hmap_min) / (hmap_max - hmap_min)
        else:
            heatmap = np.zeros_like(heatmap)
        
        # Upsample to reasonable size for overlay (e.g., 224×224)
        target_size = 224
        heatmap_upsampled = cv2.resize(
            heatmap.astype(np.float32),
            (target_size, target_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Ensure values are still in [0, 1] after interpolation
        heatmap_upsampled = np.clip(heatmap_upsampled, 0, 1)
        
        return heatmap_upsampled
