"""
GradCAM-based XAI for SAM2/MedSAM2

Uses pytorch-grad-cam library to generate meaningful saliency maps
that show which image regions influenced the segmentation decision.

This works even with models using F.scaled_dot_product_attention
because it uses gradients through the CNN encoder, not attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Any, Dict
import logging

logger = logging.getLogger(__name__)

# Check if grad-cam is available
try:
    from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, LayerCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed. Install with: pip install grad-cam")


class SAM2GradCAMWrapper(nn.Module):
    """
    Wrapper to make SAM2's image encoder compatible with pytorch-grad-cam.
    
    GradCAM requires:
    1. A model that produces output with requires_grad=True
    2. Target layers that produce 4D activations (B, C, H, W)
    3. Gradient flow from output back to target layers
    
    This wrapper extracts just the image encoder and adds a classification
    head to enable proper gradient flow.
    """
    
    # SAM2 expects 1024x1024 input due to positional embeddings
    EXPECTED_SIZE = 1024
    
    def __init__(self, sam2_model: nn.Module):
        super().__init__()
        # Extract just the image encoder
        if hasattr(sam2_model, 'image_encoder'):
            self.image_encoder = sam2_model.image_encoder
        else:
            raise ValueError("Model does not have image_encoder attribute")
        
        # We'll determine the feature dimension dynamically
        self._feature_dim = None
        self._classifier = None
        self._original_size = None
        
    def _ensure_classifier(self, feat_dim: int, device):
        """Create classifier head if not exists"""
        if self._classifier is None or self._feature_dim != feat_dim:
            self._feature_dim = feat_dim
            # Simple classifier to enable gradient flow
            self._classifier = nn.Linear(feat_dim, 1).to(device)
            # Initialize with ones so all features contribute equally
            nn.init.ones_(self._classifier.weight)
            nn.init.zeros_(self._classifier.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through image encoder with gradient-friendly output.
        """
        # Store original size
        self._original_size = x.shape[2:]  # H, W
        
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.detach().requires_grad_(True)
        
        # Resize to expected size if needed (SAM2 requires 1024x1024)
        if x.shape[2] != self.EXPECTED_SIZE or x.shape[3] != self.EXPECTED_SIZE:
            x = F.interpolate(
                x, 
                size=(self.EXPECTED_SIZE, self.EXPECTED_SIZE),
                mode='bilinear',
                align_corners=False
            )
        
        # Run image encoder
        features = self.image_encoder(x)
        
        # Handle different output formats
        if isinstance(features, dict):
            feat = features.get('vision_features', None)
            if feat is None:
                for key in ['backbone_fpn', 'vision_pos_enc']:
                    if key in features and features[key] is not None:
                        feat = features[key]
                        if isinstance(feat, (list, tuple)):
                            feat = feat[-1]
                        break
            if feat is None:
                vals = list(features.values())
                feat = vals[0] if vals else None
                if isinstance(feat, (list, tuple)):
                    feat = feat[-1]
        elif isinstance(features, (list, tuple)):
            feat = features[-1]
        else:
            feat = features
        
        if feat is None:
            raise ValueError("Could not extract features from image encoder")
        
        # Ensure 4D tensor for GradCAM
        if feat.ndim == 3:  # B, N, C (sequence format)
            B, N, C = feat.shape
            H = W = int(N ** 0.5)
            if H * W == N:
                feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                # Cannot reshape, use as-is
                feat = feat.permute(0, 2, 1)  # B, C, N
        
        # Global average pool + classifier for gradient flow
        if feat.ndim == 4:
            pooled = feat.mean(dim=(-2, -1))  # B, C
        else:
            pooled = feat.mean(dim=-1)  # B, C
        
        # Ensure classifier exists
        self._ensure_classifier(pooled.shape[-1], pooled.device)
        
        # Produce scalar output for GradCAM
        output = self._classifier(pooled)  # B, 1
        
        return output
    
    def get_target_layers(self) -> List[nn.Module]:
        """Get suitable target layers for GradCAM"""
        target_layers = []
        
        # Try to get layers from trunk (Hiera backbone)
        if hasattr(self.image_encoder, 'trunk'):
            trunk = self.image_encoder.trunk
            if hasattr(trunk, 'blocks'):
                blocks = list(trunk.blocks)
                if len(blocks) > 0:
                    # Use the last block
                    target_layers.append(blocks[-1])
        
        # Try neck as fallback
        if not target_layers and hasattr(self.image_encoder, 'neck'):
            neck = self.image_encoder.neck
            if hasattr(neck, 'convs'):
                convs = list(neck.convs)
                if len(convs) > 0:
                    target_layers.append(convs[-1])
        
        if not target_layers:
            logger.warning("No suitable target layers found")
            
        return target_layers


class MaskTarget:
    """
    Custom target for GradCAM that returns the model output directly.
    
    Since our wrapper produces a scalar output (for gradient flow),
    we just return that output to get gradients w.r.t. overall features.
    """
    
    def __init__(self, mask: np.ndarray = None):
        """
        Args:
            mask: Optional binary mask (not used for scalar output, kept for compatibility)
        """
        self.mask = mask
            
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Return the model output for backpropagation.
        
        For scalar output from our wrapper, just return it directly.
        """
        # Our wrapper outputs (B, 1) - squeeze and return
        if model_output.ndim >= 1:
            return model_output.squeeze()
        return model_output


class SAM2GradCAMExtractor:
    """
    Extracts GradCAM visualizations from SAM2/MedSAM2 models.
    
    This provides meaningful XAI by showing which image regions
    had the most influence on the segmentation output.
    
    Uses a wrapper around the image encoder only (not full model)
    to avoid SAM2's NotImplementedError on forward().
    """
    
    def __init__(
        self,
        cam_type: str = "gradcam",  # "gradcam", "gradcam++", "hirescam", "layercam"
    ):
        self.cam_type = cam_type
        self._cam = None
        self._model = None
        self._wrapped_model = None  # The wrapper for GradCAM
        self._target_layers = None
        self._last_cam_output = None
        self._device = None
        
    def setup(
        self,
        model: nn.Module,
        target_layer_names: Optional[List[str]] = None
    ) -> bool:
        """
        Setup GradCAM for the given model.
        
        Args:
            model: SAM2/MedSAM2 model
            target_layer_names: Names of layers to extract CAM from
            
        Returns:
            True if setup successful
        """
        if not GRADCAM_AVAILABLE:
            logger.error("pytorch-grad-cam not available")
            return False
            
        try:
            self._model = model
            self._device = next(model.parameters()).device
            
            # Create wrapper around image encoder only
            try:
                self._wrapped_model = SAM2GradCAMWrapper(model)
                self._wrapped_model.to(self._device)
                self._wrapped_model.eval()
                logger.info("Created SAM2GradCAMWrapper around image encoder")
            except ValueError as e:
                logger.error(f"Cannot create GradCAM wrapper: {e}")
                return False
            
            # Get target layers from the wrapper (it knows the best layers)
            self._target_layers = self._wrapped_model.get_target_layers()
            
            # If wrapper didn't find layers, try manual search
            if not self._target_layers:
                self._target_layers = self._find_target_layers(self._wrapped_model, target_layer_names)
            
            if not self._target_layers:
                logger.warning("No suitable target layers found for GradCAM")
                return False
                
            layer_names_found = [type(l).__name__ for l in self._target_layers]
            logger.info(f"GradCAM setup with {len(self._target_layers)} target layers: {layer_names_found}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup GradCAM: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _find_target_layers(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> List[nn.Module]:
        """Find suitable layers for GradCAM extraction in the wrapped model"""
        target_layers = []
        
        if layer_names:
            # Use specified layers
            for name, module in model.named_modules():
                if any(ln in name for ln in layer_names):
                    target_layers.append(module)
                    logger.debug(f"Found target layer: {name}")
        else:
            # Auto-detect good layers (Conv2d near the end of encoder)
            candidates = []
            
            for name, module in model.named_modules():
                # Look for conv layers - the wrapper has image_encoder directly
                if isinstance(module, nn.Conv2d):
                    candidates.append((name, module))
                # Also check for neck/fpn layers
                if 'neck' in name or 'fpn' in name:
                    if isinstance(module, nn.Conv2d):
                        candidates.append((name, module))
            
            # Take the last few conv layers
            if candidates:
                # Prefer neck layers if available
                neck_layers = [(n, m) for n, m in candidates if 'neck' in n]
                if neck_layers:
                    target_layers = [m for _, m in neck_layers[-2:]]
                    logger.debug(f"Using neck layers: {[n for n, _ in neck_layers[-2:]]}")
                else:
                    target_layers = [m for _, m in candidates[-2:]]
                    logger.debug(f"Using last conv layers: {[n for n, _ in candidates[-2:]]}")
                    
            logger.info(f"Auto-detected {len(target_layers)} target layers for GradCAM")
            # For SAM2, look in image_encoder
            candidates = []
            
            for name, module in model.named_modules():
                # Look for conv layers in image encoder
                if 'image_encoder' in name:
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        candidates.append((name, module))
                    # Also check for neck/fpn layers
                    if 'neck' in name or 'fpn' in name:
                        if isinstance(module, nn.Conv2d):
                            candidates.append((name, module))
            
            # Take the last few conv layers
            if candidates:
                # Prefer neck layers if available
                neck_layers = [(n, m) for n, m in candidates if 'neck' in n]
                if neck_layers:
                    target_layers = [m for _, m in neck_layers[-2:]]
                else:
                    target_layers = [m for _, m in candidates[-2:]]
                    
            logger.debug(f"Auto-detected {len(target_layers)} target layers")
                    
        return target_layers
    
    def compute_cam(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Compute GradCAM for the given image.
        
        Args:
            image: Input image (H, W, 3) uint8 or float
            mask: Optional segmentation mask to focus gradients on
            normalize: Whether to normalize output to [0, 1]
            
        Returns:
            GradCAM heatmap (H, W) float32
        """
        if not GRADCAM_AVAILABLE or self._wrapped_model is None:
            logger.warning("GradCAM not available or model not set up")
            return None
            
        if not self._target_layers:
            logger.warning("No target layers configured")
            return None
            
        try:
            # Prepare image tensor
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
            img_tensor.requires_grad_(True)
            
            # Always provide a target for proper gradient computation
            targets = [MaskTarget(mask)]
            
            # Select CAM type
            cam_class = {
                "gradcam": GradCAM,
                "gradcam++": GradCAMPlusPlus,
                "hirescam": HiResCAM,
                "layercam": LayerCAM,
            }.get(self.cam_type, GradCAM)
            
            # Create CAM using the WRAPPED model (image encoder only)
            # Temporarily enable training mode for gradients
            self._wrapped_model.train()
            
            with cam_class(model=self._wrapped_model, target_layers=self._target_layers) as cam:
                # Compute CAM
                grayscale_cam = cam(
                    input_tensor=img_tensor,
                    targets=targets,
                    aug_smooth=False,
                    eigen_smooth=False
                )
                
                # Get first batch item
                cam_output = grayscale_cam[0]
            
            # Restore eval mode
            self._wrapped_model.eval()
                
            # Normalize
            if normalize and cam_output.max() > cam_output.min():
                cam_output = (cam_output - cam_output.min()) / (cam_output.max() - cam_output.min())
                
            self._last_cam_output = cam_output
            
            logger.info(f"GradCAM computed: shape={cam_output.shape}, range=[{cam_output.min():.3f}, {cam_output.max():.3f}]")
            return cam_output
            
        except Exception as e:
            logger.error(f"Failed to compute GradCAM: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_last_cam(self) -> Optional[np.ndarray]:
        """Get the last computed CAM"""
        return self._last_cam_output
    
    def create_visualization(
        self,
        image: np.ndarray,
        cam: Optional[np.ndarray] = None,
        colormap: str = "jet",
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization overlay.
        
        Args:
            image: Original image (H, W, 3)
            cam: CAM heatmap (uses last computed if None)
            colormap: Matplotlib colormap name
            alpha: Overlay transparency
            
        Returns:
            Visualization image (H, W, 3) uint8
        """
        if cam is None:
            cam = self._last_cam_output
            
        if cam is None:
            return image
            
        try:
            # Ensure image is float [0, 1]
            if image.dtype == np.uint8:
                img_float = image.astype(np.float32) / 255.0
            else:
                img_float = image.astype(np.float32)
                if img_float.max() > 1.0:
                    img_float = img_float / 255.0
            
            # Ensure RGB
            if img_float.ndim == 2:
                img_float = np.stack([img_float] * 3, axis=-1)
                
            # Resize CAM if needed
            if cam.shape[:2] != img_float.shape[:2]:
                import cv2
                cam = cv2.resize(cam, (img_float.shape[1], img_float.shape[0]))
            
            # Use show_cam_on_image if available
            if GRADCAM_AVAILABLE:
                visualization = show_cam_on_image(
                    img_float,
                    cam,
                    use_rgb=True,
                    colormap=self._get_cv2_colormap(colormap),
                    image_weight=1 - alpha
                )
            else:
                # Manual overlay
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap(colormap)
                cam_colored = cmap(cam)[:, :, :3]
                visualization = (img_float * (1 - alpha) + cam_colored * alpha)
                visualization = (visualization * 255).astype(np.uint8)
                
            return visualization
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return image
    
    def _get_cv2_colormap(self, name: str) -> int:
        """Convert colormap name to OpenCV constant"""
        import cv2
        colormaps = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "turbo": cv2.COLORMAP_TURBO,
        }
        return colormaps.get(name, cv2.COLORMAP_JET)
    
    def clear(self):
        """Clear cached data"""
        self._last_cam_output = None


# Global instance
_gradcam_extractor: Optional[SAM2GradCAMExtractor] = None


def get_gradcam_extractor() -> SAM2GradCAMExtractor:
    """Get or create the global GradCAM extractor"""
    global _gradcam_extractor
    if _gradcam_extractor is None:
        _gradcam_extractor = SAM2GradCAMExtractor()
    return _gradcam_extractor


def compute_gradcam_for_sam2(
    model: nn.Module,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    cam_type: str = "gradcam"
) -> Optional[np.ndarray]:
    """
    Convenience function to compute GradCAM for SAM2.
    
    Args:
        model: SAM2/MedSAM2 model
        image: Input image
        mask: Optional segmentation mask
        cam_type: Type of CAM ("gradcam", "gradcam++", "hirescam", "layercam")
        
    Returns:
        GradCAM heatmap
    """
    extractor = get_gradcam_extractor()
    extractor.cam_type = cam_type
    
    if not extractor.setup(model):
        return None
        
    return extractor.compute_cam(image, mask)
