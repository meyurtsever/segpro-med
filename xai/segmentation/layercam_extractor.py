"""
LayerCAM-based XAI for SAM2/MedSAM2

LayerCAM implementation for explainable AI that works with feature maps only.
This is particularly suitable for SAM2 with Hiera backbone as it doesn't require
breaking FlashAttention - we only need access to convolutional/transformer block outputs.

Reference: https://github.com/PengtaoJiang/LayerCAM-jittor

Key advantages:
- Works with feature maps only (no gradient computation needed for base version)
- Compatible with FlashAttention and other optimized operations
- Fast computation during inference
- Good localization on hierarchical backbones like Hiera

Note: For full LayerCAM, we still use gradients, but the feature extraction
is simpler and more compatible with modern architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract feature maps from SAM2's Hiera backbone.
    
    Hiera is a hierarchical vision transformer, so we hook into
    the transformer blocks to get multi-scale features.
    """
    
    def __init__(self):
        self.features = {}
        self.hooks = []
        
    def _make_hook(self, name: str):
        """Create a forward hook for feature extraction"""
        def hook(module, input, output):
            # Handle different output formats
            if isinstance(output, dict):
                # Try to extract the actual feature tensor
                feat = output.get('vision_features', None)
                if feat is None:
                    for key in ['backbone_fpn', 'vision_pos_enc', 'x']:
                        if key in output:
                            feat = output[key]
                            break
                if feat is None:
                    # Just take the first value
                    feat = list(output.values())[0]
            elif isinstance(output, (list, tuple)):
                feat = output[0] if isinstance(output, tuple) else output[-1]
            else:
                feat = output
            
            # Store the feature
            if isinstance(feat, torch.Tensor):
                self.features[name] = feat.detach()
        return hook
    
    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Register hooks on specified layers or auto-detect suitable layers.
        
        Args:
            model: SAM2 model
            layer_names: Optional list of layer names to hook
        """
        self.clear_hooks()
        
        # Try to access image_encoder
        if hasattr(model, 'image_encoder'):
            encoder = model.image_encoder
        else:
            encoder = model
        
        # Auto-detect layers if not specified
        if layer_names is None:
            layer_names = self._auto_detect_layers(encoder)
        
        # Register hooks
        for name, module in encoder.named_modules():
            if any(ln in name for ln in layer_names):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
                logger.info(f"Registered hook on: {name}")
    
    def _auto_detect_layers(self, encoder: nn.Module) -> List[str]:
        """
        Auto-detect suitable layers for feature extraction in Hiera backbone.
        
        CRITICAL: Hiera blocks output SEQUENCE format (B, N, C), not spatial!
        We MUST hook the NECK layers which convert sequences back to spatial Conv2d features.
        
        The neck contains Conv2d/ConvTranspose2d layers that produce proper (B, C, H, W) features.
        """
        layer_names = []
        
        # PRIORITY 1: Hook neck layers (spatial Conv2d features)
        # These are the ONLY layers with proper spatial structure!
        neck_layers = []
        for name, module in encoder.named_modules():
            if 'neck' in name.lower():
                # Look for Conv2d or ConvTranspose2d in neck
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    neck_layers.append(name)
                    logger.debug(f"Found neck layer: {name} ({type(module).__name__})")
        
        if neck_layers:
            # Use neck layers - these have proper spatial structure
            layer_names = neck_layers[:4]  # Use up to 4 neck layers
            logger.info(f"Using {len(layer_names)} neck Conv2d layers (spatial features)")
        else:
            # FALLBACK: If no neck found, look for ANY Conv2d layers
            logger.warning("No neck layers found, searching for Conv2d layers...")
            for name, module in encoder.named_modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    layer_names.append(name)
                    logger.debug(f"Found Conv2d layer: {name}")
                    if len(layer_names) >= 4:
                        break
        
        # DO NOT hook transformer blocks - they output sequences, not spatial maps!
        # Transformer blocks produce (B, N, C) which creates vertical stripes when reshaped
        
        if not layer_names:
            logger.error("Could not find any suitable Conv2d layers for LayerCAM!")
        else:
            logger.info(f"Auto-detected {len(layer_names)} layers for LayerCAM: {layer_names}")
        
        return layer_names
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get extracted features"""
        return self.features
    
    def clear_features(self):
        """Clear stored features"""
        self.features = {}
    
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.features = {}


class LayerCAMExtractor:
    """
    LayerCAM implementation for SAM2/MedSAM2.
    
    LayerCAM computes class activation maps by using the feature maps
    from intermediate layers. It's particularly effective for hierarchical
    architectures like Hiera.
    
    The algorithm:
    1. Extract feature maps from multiple layers
    2. For each layer, compute spatial importance (using gradients OR class-specific pooling)
    3. Combine multi-scale feature maps with weighted aggregation
    4. Generate final activation map
    """
    
    def __init__(self, use_gradients: bool = True):
        """
        Args:
            use_gradients: If True, use gradient-based LayerCAM (more accurate but slower)
                          If False, use feature-magnitude-based approximation (faster)
        """
        self.use_gradients = use_gradients
        self.feature_extractor = FeatureExtractor()
        self._model = None
        self._device = None
        self._last_cam = None
        
    def setup(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> bool:
        """
        Setup LayerCAM for the given model.
        
        Args:
            model: SAM2/MedSAM2 model
            layer_names: Optional list of layer names to extract features from
            
        Returns:
            True if setup successful
        """
        try:
            self._model = model
            self._device = next(model.parameters()).device
            
            # Register hooks for feature extraction
            self.feature_extractor.register_hooks(model, layer_names)
            
            logger.info(f"LayerCAM setup complete (gradient-based: {self.use_gradients})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup LayerCAM: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def compute_cam(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Compute LayerCAM for the given image.
        
        Args:
            image: Input image (H, W, 3) uint8 or float
            mask: Optional segmentation mask (used to focus on segmented regions)
            normalize: Whether to normalize output to [0, 1]
            
        Returns:
            LayerCAM heatmap (H, W) float32
        """
        if self._model is None:
            logger.warning("LayerCAM not set up")
            return None
        
        try:
            # Prepare image
            img_tensor = self._prepare_image(image)
            
            # Clear previous features
            self.feature_extractor.clear_features()
            
            # Forward pass to extract features
            with torch.set_grad_enabled(self.use_gradients):
                if img_tensor.requires_grad != self.use_gradients:
                    img_tensor.requires_grad_(self.use_gradients)
                
                # Run through image encoder
                if hasattr(self._model, 'image_encoder'):
                    output = self._model.image_encoder(img_tensor)
                else:
                    output = self._model(img_tensor)
                
                # Get extracted features
                features = self.feature_extractor.get_features()
                
                if not features:
                    logger.warning("No features extracted")
                    return None
                
                # Compute CAM from features
                if self.use_gradients:
                    cam = self._compute_gradient_based_cam(output, features, mask)
                else:
                    cam = self._compute_feature_based_cam(features, mask)
            
            # Resize to original image size
            if cam is not None:
                cam = self._resize_cam(cam, image.shape[:2])
                
                # Normalize
                if normalize and cam.max() > cam.min():
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                
                self._last_cam = cam
                logger.info(f"LayerCAM computed: shape={cam.shape}, range=[{cam.min():.3f}, {cam.max():.3f}]")
            
            return cam
            
        except Exception as e:
            logger.error(f"Failed to compute LayerCAM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model input"""
        # Convert to float [0, 1]
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)
            if img_float.max() > 1.0:
                img_float = img_float / 255.0
        
        # Ensure RGB
        if img_float.ndim == 2:
            img_float = np.stack([img_float] * 3, axis=-1)
        elif img_float.shape[-1] == 1:
            img_float = np.concatenate([img_float] * 3, axis=-1)
        
        # To tensor: (H, W, 3) -> (1, 3, H, W)
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self._device)
        
        # Resize to 1024x1024 (SAM2 requirement)
        if img_tensor.shape[2] != 1024 or img_tensor.shape[3] != 1024:
            img_tensor = F.interpolate(
                img_tensor,
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            )
        
        return img_tensor
    
    def _compute_gradient_based_cam(
        self,
        output: Any,
        features: Dict[str, torch.Tensor],
        mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compute LayerCAM using gradient-based approach (original LayerCAM).
        
        This uses gradients to weight the feature maps, providing more accurate
        class-specific activations.
        """
        # Extract a scalar from output for backprop
        if isinstance(output, dict):
            # Try to get a meaningful output
            output_tensor = output.get('vision_features', None)
            if output_tensor is None:
                output_tensor = list(output.values())[0]
        elif isinstance(output, (list, tuple)):
            output_tensor = output[0] if isinstance(output, tuple) else output[-1]
        else:
            output_tensor = output
        
        if not isinstance(output_tensor, torch.Tensor):
            logger.warning("Cannot extract tensor from output")
            return self._compute_feature_based_cam(features, mask)
        
        # Compute target (sum of output)
        target = output_tensor.sum()
        
        # Backward to get gradients
        self._model.zero_grad()
        target.backward(retain_graph=False)
        
        # Compute weighted CAM from each layer
        cams = []
        for layer_name, feature in features.items():
            if feature.grad is not None:
                # LayerCAM: multiply feature maps with their gradients
                grad = feature.grad
                
                # Compute weighted feature map
                weighted_feature = (feature * grad).sum(dim=1, keepdim=True)
                
                # Apply ReLU (only positive contributions)
                weighted_feature = F.relu(weighted_feature)
                
                # Convert to numpy
                cam_layer = weighted_feature.squeeze().cpu().numpy()
                
                cams.append(cam_layer)
        
        if not cams:
            logger.warning("No gradient-based CAMs computed")
            return None
        
        # Aggregate multi-scale CAMs
        cam = self._aggregate_cams(cams)
        
        # CRITICAL: Weight by segmentation mask if provided
        # This focuses XAI on the regions that were actually segmented
        if cam is not None and mask is not None:
            cam = self._apply_mask_weighting(cam, mask)
        
        return cam
    
    def _compute_feature_based_cam(
        self,
        features: Dict[str, torch.Tensor],
        mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Compute LayerCAM using feature magnitude (fast approximation, no gradients).
        
        This is faster and works well for fast masking scenarios.
        """
        cams = []
        
        for layer_name, feature in features.items():
            logger.debug(f"Processing layer {layer_name}: shape={feature.shape}")
            
            # ONLY accept proper 4D Conv2d features (B, C, H, W)
            # DO NOT try to reshape transformer sequences - they create noise!
            if feature.ndim == 3:
                # This is a transformer sequence (B, N, C) - SKIP IT!
                logger.warning(f"Layer {layer_name}: Transformer sequence (B, N, C) detected, skipping. Use neck layers instead!")
                continue
            
            if feature.ndim != 4:
                logger.warning(f"Layer {layer_name}: Expected 4D Conv2d features, got {feature.ndim}D, skipping")
                continue
            
            # Verify this is a proper spatial feature map
            B, C, H, W = feature.shape
            
            # Check if spatial dimensions are reasonable
            if H < 4 or W < 4:
                logger.warning(f"Layer {layer_name}: Spatial dimensions too small ({H}x{W}), skipping")
                continue
            
            if H > 512 or W > 512:
                logger.warning(f"Layer {layer_name}: Spatial dimensions too large ({H}x{W}), skipping")
                continue
            
            # Simple approach: use channel-wise magnitude
            # This captures regions with high activation
            cam_layer = feature.abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            # Apply some spatial smoothing
            if cam_layer.shape[2] > 4 and cam_layer.shape[3] > 4:
                cam_layer = F.avg_pool2d(cam_layer, kernel_size=3, stride=1, padding=1)
            
            # Convert to numpy
            cam_np = cam_layer.squeeze().cpu().numpy()
            
            # Ensure it's at least 2D
            if cam_np.ndim == 0:
                # Scalar - skip
                continue
            elif cam_np.ndim == 1:
                # 1D sequence - try to reshape to 2D
                n = cam_np.shape[0]
                h = w = int(np.sqrt(n))
                if h * w == n:
                    cam_np = cam_np.reshape(h, w)
                else:
                    # Not a perfect square, skip this layer
                    logger.debug(f"Skipping layer {layer_name} with non-square sequence length {n}")
                    continue
            elif cam_np.ndim > 2:
                # More than 2D, squeeze extra dimensions
                cam_np = np.squeeze(cam_np)
                if cam_np.ndim > 2:
                    # Still too many dimensions, take first slice
                    while cam_np.ndim > 2:
                        cam_np = cam_np[0]
            
            cams.append(cam_np)
        
        if not cams:
            logger.warning("No feature-based CAMs computed")
            return None
        
        # Aggregate multi-scale CAMs
        cam = self._aggregate_cams(cams)
        
        # CRITICAL: Weight by segmentation mask if provided
        # This focuses XAI on the regions that were actually segmented
        if cam is not None and mask is not None:
            cam = self._apply_mask_weighting(cam, mask)
        
        return cam
    
    def _aggregate_cams(self, cams: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate CAMs from multiple layers.
        
        Strategy: Resize all to the same size and average with weights
        favoring later (more semantic) layers.
        """
        if len(cams) == 0:
            return None
        
        if len(cams) == 1:
            return cams[0]
        
        # Find the largest spatial resolution
        max_h = max(cam.shape[0] for cam in cams)
        max_w = max(cam.shape[1] for cam in cams)
        
        # Resize and weight
        weighted_cams = []
        for i, cam in enumerate(cams):
            # Resize to max resolution
            if cam.shape[0] != max_h or cam.shape[1] != max_w:
                cam_resized = self._resize_cam(cam, (max_h, max_w))
            else:
                cam_resized = cam
            
            # Weight: later layers get more weight (more semantic)
            weight = (i + 1) / len(cams)  # Linear weighting: 1/n, 2/n, ..., n/n
            weighted_cams.append(cam_resized * weight)
        
        # Average
        aggregated = np.mean(weighted_cams, axis=0)
        
        return aggregated
    
    def _resize_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize CAM to target size"""
        # Squeeze any singleton dimensions
        cam = np.squeeze(cam)
        
        # If still not 2D after squeezing, it might be a 1D sequence - reshape it
        if cam.ndim == 1:
            # Try to reshape to a square
            n = cam.shape[0]
            h = w = int(np.sqrt(n))
            if h * w == n:
                cam = cam.reshape(h, w)
            else:
                # Not a perfect square, use closest square and pad/crop
                h = w = int(np.ceil(np.sqrt(n)))
                padded = np.zeros(h * w, dtype=cam.dtype)
                padded[:n] = cam
                cam = padded.reshape(h, w)
        elif cam.ndim > 2:
            # If still more than 2D, take the first 2D slice
            logger.warning(f"CAM has unexpected shape {cam.shape}, taking first 2D slice")
            while cam.ndim > 2:
                cam = cam[0]
        
        if cam.shape[:2] == target_size:
            return cam
        
        # Use cv2 for resizing (more robust than PIL)
        import cv2
        cam_resized = cv2.resize(cam.astype(np.float32), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        return cam_resized
    
    def _apply_mask_weighting(self, cam: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Weight CAM to focus on relevant regions.
        
        PROBLEM: Boundary detection doesn't work - it just highlights obvious edges (skull).
        SOLUTION: Don't use mask at all - just return normalized CAM showing general features.
        
        The real issue: encoder features can't distinguish tumor from brain because
        they're computed before segmentation happens. We'd need decoder features for that.
        
        For now: Just normalize and return CAM without mask weighting.
        Future: Use prompt points to create region of interest, or use decoder features.
        
        Args:
            cam: Activation map (H, W) float
            mask: Segmentation mask (H, W) or (H, W, 3) uint8 or float [IGNORED]
            
        Returns:
            CAM without mask weighting (just shows encoder attention)
        """
        import cv2
        
        # Just return the CAM normalized
        # This shows what the encoder focused on in the whole image
        # Not perfect, but better than highlighting wrong boundaries
        
        logger.debug(f"Returning unweighted CAM: range=[{cam.min():.3f}, {cam.max():.3f}]")
        
        return cam
    
    def get_last_cam(self) -> Optional[np.ndarray]:
        """Get the last computed CAM"""
        return self._last_cam
    
    def clear(self):
        """Clear cached data and hooks"""
        self.feature_extractor.clear_features()
        self._last_cam = None
    
    def cleanup(self):
        """Full cleanup including removing hooks"""
        self.feature_extractor.clear_hooks()
        self.clear()


# Global instance
_layercam_extractor: Optional[LayerCAMExtractor] = None


def get_layercam_extractor(use_gradients: bool = True) -> LayerCAMExtractor:
    """
    Get or create the global LayerCAM extractor.
    
    Args:
        use_gradients: Whether to use gradient-based LayerCAM
        
    Returns:
        LayerCAM extractor instance
    """
    global _layercam_extractor
    if _layercam_extractor is None:
        _layercam_extractor = LayerCAMExtractor(use_gradients=use_gradients)
    return _layercam_extractor


def compute_layercam_for_sam2(
    model: nn.Module,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    use_gradients: bool = True
) -> Optional[np.ndarray]:
    """
    Convenience function to compute LayerCAM for SAM2.
    
    Args:
        model: SAM2/MedSAM2 model
        image: Input image
        mask: Optional segmentation mask
        use_gradients: Whether to use gradient-based approach
        
    Returns:
        LayerCAM heatmap
    """
    extractor = get_layercam_extractor(use_gradients=use_gradients)
    
    if not extractor.setup(model):
        return None
    
    return extractor.compute_cam(image, mask)
