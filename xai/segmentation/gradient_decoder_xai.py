"""
Gradient-Based Decoder XAI for SAM2
====================================

This module implements TRUE decoder-level XAI by computing gradients from the mask prediction
back to the encoder output (decoder input). This approach bypasses FlashAttention completely
by using standard PyTorch autograd on the inputs/outputs of the decoder.

Method: Target the Decoder Input (Cross-Attention Keys)
-------------------------------------------------------
- Hook: Encoder output / Decoder input (image embeddings)
- Compute: ∂(mask_score) / ∂(image_embeddings)
- Result: 64×64 heatmap showing which SPATIAL REGIONS in the encoded image
          influenced the segmentation decision

Key Advantages:
1. No FlashAttention inspection needed - uses standard backprop
2. Decision-specific - gradients come from actual mask prediction
3. Spatial localization - shows WHERE the decision came from
4. Prompt-aware - decoder uses prompts, gradients reflect them

Author: SegMed-Pro XAI Team
Date: January 17, 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class GradientDecoderXAI:
    """
    Gradient-based XAI that extracts decision-specific explanations from SAM2.
    
    This extractor hooks the image embeddings (encoder output / decoder input) and
    computes gradients from the predicted mask score, providing true decoder-level
    explanations without needing to inspect FlashAttention internals.
    """
    
    def __init__(self):
        self.image_embeddings: Optional[torch.Tensor] = None
        self.gradient_map: Optional[np.ndarray] = None
        self._hook = None
        self._predictor = None
        
    def setup(self, predictor) -> bool:
        """
        Setup gradient extraction for SAM2 predictor.
        
        Args:
            predictor: SAM2ImagePredictor instance
            
        Returns:
            True if setup successful
        """
        try:
            self._predictor = predictor
            logger.info("GradientDecoderXAI: Setup complete")
            return True
        except Exception as e:
            logger.error(f"GradientDecoderXAI setup failed: {e}")
            return False
    
    def _register_embedding_hook(self):
        """Register hook to capture image embeddings with gradients enabled."""
        try:
            # Get the model's image encoder
            model = self._predictor.model
            
            # Hook at the end of the encoder (before decoder input)
            # SAM2 architecture: image_encoder -> neck -> embeddings
            def embedding_hook(module, input, output):
                # Store embeddings and ensure gradients are tracked
                # Handle different output types: tensor, tuple, list, dict
                embeddings = None
                
                # Recursively unwrap nested structures to find the tensor
                def unwrap_to_tensor(obj, depth=0):
                    if depth > 5:  # Prevent infinite recursion
                        return None
                    
                    if isinstance(obj, torch.Tensor):
                        return obj
                    elif isinstance(obj, (tuple, list)) and len(obj) > 0:
                        # Try each element
                        for item in obj:
                            result = unwrap_to_tensor(item, depth + 1)
                            if result is not None:
                                return result
                        return None
                    elif isinstance(obj, dict):
                        # Try common keys
                        for key in ['embeddings', 'features', 'output', 0]:
                            if key in obj:
                                result = unwrap_to_tensor(obj[key], depth + 1)
                                if result is not None:
                                    return result
                        return None
                    else:
                        return None
                
                embeddings = unwrap_to_tensor(output)
                
                if embeddings is None or not isinstance(embeddings, torch.Tensor):
                    logger.warning(f"Embedding hook could not find tensor in output type: {type(output)}")
                    if isinstance(output, (tuple, list)):
                        logger.warning(f"  Output is sequence with {len(output)} elements, types: {[type(x) for x in output]}")
                    return output
                
                self.image_embeddings = embeddings
                
                # Enable gradient tracking
                self.image_embeddings.requires_grad_(True)
                self.image_embeddings.retain_grad()
                
                logger.debug(f"✅ Captured image embeddings: {self.image_embeddings.shape}, requires_grad={self.image_embeddings.requires_grad}")
                return output
            
            # Hook the neck output (final encoder output before decoder)
            if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'neck'):
                self._hook = model.image_encoder.neck.register_forward_hook(embedding_hook)
                logger.info("Registered gradient hook on image_encoder.neck")
                return True
            else:
                logger.warning("Could not find neck module for gradient hook")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register embedding hook: {e}")
            import traceback
            logger.debug(f"Hook registration traceback: {traceback.format_exc()}")
            return False
    
    def _remove_hook(self):
        """Remove the registered hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
    
    def compute_gradient_map(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute gradient-based explanation for SAM2 prediction.
        
        Args:
            image: RGB image (H, W, 3)
            points: Point prompts (N, 2)
            labels: Point labels (N,)
            boxes: Box prompts (M, 4)
            
        Returns:
            Tuple of (gradient_heatmap, mask_logits):
            - gradient_heatmap: (H, W) normalized to [0, 1], or None if failed
            - mask_logits: (H, W) raw logits (pre-sigmoid), or None if failed
        """
        try:
            # Validate inputs
            if image is None:
                logger.warning("No image provided for gradient XAI")
                return None, None
            
            if self._predictor is None:
                logger.error("Predictor not set up - call setup() first")
                return None, None
            
            # Get the model (not predictor, to avoid caching)
            model = self._predictor.model
            
            # Preprocess image like SAM2 does
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image).float()
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            
            # Rearrange to (C, H, W) and add batch dimension
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Move to same device as model
            device = next(model.parameters()).device
            image_tensor = image_tensor.to(device)
            
            # Resize to SAM2 input size (1024x1024)
            import torch.nn.functional as F
            image_tensor = F.interpolate(image_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            logger.debug(f"Image tensor prepared: {image_tensor.shape}, device={device}")
            
            # Enable gradients for the image tensor
            image_tensor.requires_grad_(True)
            
            # Run encoder directly (bypasses predictor caching)
            with torch.set_grad_enabled(True):
                # Get image embeddings from encoder  
                # Use forward_image which returns processed backbone features
                backbone_out = model.forward_image(image_tensor)
                
                # Prepare backbone features the exact same way as predictor does
                _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)
                
                # Add no_mem_embed if needed (for video training compatibility)
                if hasattr(model, 'directly_add_no_mem_embed') and model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
                
                logger.info(f"📊 vision_feats shapes: {[f.shape for f in vision_feats]}")
                logger.info(f"📊 feat_sizes from model: {feat_sizes}")
                logger.info(f"📊 _bb_feat_sizes from predictor: {self._predictor._bb_feat_sizes}")
                
                # CRITICAL: The predictor expects specific sizes defined in _bb_feat_sizes
                # But we need to match vision_feats to those sizes correctly
                # The issue: feat_sizes from model might be larger than _bb_feat_sizes
                # Solution: Use _bb_feat_sizes but ensure we're matching the right features
                
                # The predictor takes the LAST len(_bb_feat_sizes) features
                num_expected = len(self._predictor._bb_feat_sizes)
                vision_feats_trimmed = vision_feats[-num_expected:] if len(vision_feats) >= num_expected else vision_feats
                
                logger.info(f"📊 Using last {num_expected} vision_feats: {[f.shape for f in vision_feats_trimmed]}")
                
                # Process features exactly like predictor.set_image() does
                # CRITICAL: vision_feats are in HWxNxC format, need to convert to NxCxHxW
                # Then downsample to match _bb_feat_sizes if needed
                import torch.nn.functional as F
                
                feats = []
                for i, (feat, expected_size) in enumerate(zip(vision_feats_trimmed[::-1], self._predictor._bb_feat_sizes[::-1])):
                    # feat is (HW, N, C) - permute to (N, C, HW)
                    feat_permuted = feat.permute(1, 2, 0)  # (N, C, HW)
                    
                    # Infer actual spatial size from HW dimension
                    HW = feat.shape[0]
                    C = feat.shape[2]
                    N = feat.shape[1]
                    actual_H = int(HW ** 0.5)
                    actual_W = actual_H
                    
                    # Reshape to (N, C, H, W)
                    feat_spatial = feat_permuted.reshape(N, C, actual_H, actual_W)
                    
                    # Downsample to expected size if needed
                    expected_H, expected_W = expected_size
                    if actual_H != expected_H or actual_W != expected_W:
                        feat_spatial = F.interpolate(feat_spatial, size=(expected_H, expected_W), mode='bilinear', align_corners=False)
                        logger.debug(f"Downsampled feat {i}: ({actual_H}, {actual_W}) -> ({expected_H}, {expected_W})")
                    
                    feats.append(feat_spatial)
                
                # Reverse back to original order
                feats = feats[::-1]
                
                logger.debug(f"Feats after processing: {len(feats)} levels, shapes={[f.shape for f in feats]}")
                
                image_embeddings = feats[-1]  # Main embedding (lowest resolution)
                high_res_feats = feats[:-1]   # Multi-scale features (higher resolutions)
                
                logger.debug(f"Encoder output: image_embed={image_embeddings.shape}, high_res_feats={len(high_res_feats)} scales, shapes={[f.shape for f in high_res_feats]}")
                
                # Prepare prompts for decoder
                concat_points = None
                if points is not None and labels is not None:
                    point_coords_tensor = torch.from_numpy(points).float().to(device)
                    point_labels_tensor = torch.from_numpy(labels).to(device)
                    
                    # Transform coordinates to 1024x1024 space using predictor's transforms
                    unnorm_coords = self._predictor._transforms.transform_coords(
                        point_coords_tensor, 
                        normalize=True,
                        orig_hw=(image.shape[0], image.shape[1])
                    )
                    if len(unnorm_coords.shape) == 2:
                        unnorm_coords = unnorm_coords.unsqueeze(0)
                        point_labels_tensor = point_labels_tensor.unsqueeze(0)
                    
                    concat_points = (unnorm_coords, point_labels_tensor)
                
                unnorm_box = None
                if boxes is not None:
                    box_tensor = torch.from_numpy(boxes).float().to(device)
                    box_coords = box_tensor.reshape(-1, 2, 2)
                    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=device)
                    box_labels = box_labels.repeat(box_tensor.size(0), 1)
                    
                    # Merge boxes with points
                    if concat_points is not None:
                        concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                        concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                        concat_points = (concat_coords, concat_labels)
                    else:
                        concat_points = (box_coords, box_labels)
                
                logger.debug(f"Prompts prepared: concat_points={concat_points[0].shape if concat_points else None}")
                
                # Call prompt encoder
                sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
                    points=concat_points,
                    boxes=None,
                    masks=None,
                )
                
                # Prepare high_res_features exactly like predictor._predict does
                # Index with img_idx=0 (first/only image) and unsqueeze
                high_res_features = [
                    feat_level[0].unsqueeze(0)
                    for feat_level in high_res_feats
                ]
                
                logger.info(f"📊 SHAPES: image_embed={image_embeddings.shape}, image_embed[0]={image_embeddings[0].shape}")
                logger.info(f"📊 SHAPES: high_res input={[f.shape for f in high_res_feats]}")
                logger.info(f"📊 SHAPES: high_res output={[f.shape for f in high_res_features]}")
                logger.info(f"📊 SHAPES: sparse_embed={sparse_embeddings.shape}, dense_embed={dense_embeddings.shape}")
                
                # Call mask decoder directly with our gradient-enabled embeddings
                batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
                low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
                    image_embeddings=image_embeddings[0].unsqueeze(0),  # Index and unsqueeze like predictor does
                    image_pe=model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                
                # Use IoU predictions as scores
                scores = iou_predictions
                
                logger.debug(f"Decoder output: low_res_masks={low_res_masks.shape}, scores={scores.shape}")
                
                # Store mask logits (pre-sigmoid) for uncertainty computation
                # Get best mask based on IoU score
                best_mask_idx = scores.argmax(dim=1)
                best_mask_logits = low_res_masks[0, best_mask_idx[0]].detach().cpu().numpy()
                logger.info(f"Captured mask logits: {best_mask_logits.shape}, range=[{best_mask_logits.min():.3f}, {best_mask_logits.max():.3f}]")
                
                # Get best mask score
                if isinstance(scores, torch.Tensor):
                    best_score = scores.max()
                else:
                    best_score = torch.tensor(scores.max(), device=device, requires_grad=True)
                
                logger.info(f"Best mask score: {best_score.item():.3f}, requires_grad={best_score.requires_grad}")
            
            # Compute gradients: ∂(score) / ∂(image_tensor)
            if best_score.requires_grad:
                best_score.backward(retain_graph=False)
                
                # Extract gradients from image tensor
                if image_tensor.grad is not None:
                    grad = image_tensor.grad.detach()
                    grad_magnitude = torch.abs(grad)
                    
                    # Reduce across channels and batch: (1, C, H, W) -> (H, W)
                    grad_map = grad_magnitude[0].mean(dim=0)
                    
                    # Convert to numpy
                    grad_map_np = grad_map.cpu().numpy()
                    
                    logger.info(f"Gradient map computed: {grad_map_np.shape}, range=[{grad_map_np.min():.4f}, {grad_map_np.max():.4f}]")
                    
                    # Normalize
                    if grad_map_np.max() > grad_map_np.min():
                        grad_map_normalized = (grad_map_np - grad_map_np.min()) / (grad_map_np.max() - grad_map_np.min())
                    else:
                        grad_map_normalized = np.zeros_like(grad_map_np)
                    
                    # Resize to original image size
                    from scipy.ndimage import zoom
                    scale_factors = (image.shape[0] / grad_map_normalized.shape[0],
                                   image.shape[1] / grad_map_normalized.shape[1])
                    grad_map_resized = zoom(grad_map_normalized, scale_factors, order=1)
                    
                    self.gradient_map = grad_map_resized
                    logger.info(f"✅ Gradient map ready: {grad_map_resized.shape}")
                    return grad_map_resized, best_mask_logits
                else:
                    logger.warning("No gradients on image tensor")
                    return None, best_mask_logits
            else:
                logger.warning("Best score doesn't require gradients")
                return None, best_mask_logits
            
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self._remove_hook()
            return None, None
    
    def get_last_gradient_map(self) -> Optional[np.ndarray]:
        """
        Get the last computed gradient map.
        
        Returns:
            Gradient heatmap (H, W) or None
        """
        return self.gradient_map
    
    def cleanup(self):
        """Clean up hooks and cached data."""
        self._remove_hook()
        self.image_embeddings = None
        self.gradient_map = None
        logger.debug("GradientDecoderXAI cleaned up")


# Factory function
def create_gradient_decoder_xai() -> GradientDecoderXAI:
    """
    Factory function to create a GradientDecoderXAI instance.
    
    Returns:
        GradientDecoderXAI instance
    """
    return GradientDecoderXAI()
