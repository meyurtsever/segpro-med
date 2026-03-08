"""
Uncertainty Map Generation for SAM2 Segmentation

Computes per-pixel confidence/uncertainty from mask logits.
NO FlashAttention or gradient access required!
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class UncertaintyMapGenerator:
    """
    Generate per-pixel confidence/uncertainty maps from SAM2 logits
    
    NO FlashAttention or gradient access required!
    Uses only the mask logits (pre-sigmoid scores)
    """
    
    def __init__(self, uncertainty_threshold: float = 0.3):
        """
        Args:
            uncertainty_threshold: Threshold for "uncertain" classification
                                  (0.3 means 30% uncertainty or more is "high")
        """
        self.uncertainty_threshold = uncertainty_threshold
    
    def compute_from_logits(
        self, 
        mask_logits: np.ndarray,
        method: str = "margin"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute uncertainty map from mask logits
        
        Args:
            mask_logits: Raw SAM2 output [H, W] BEFORE sigmoid
            method: "entropy", "margin", or "variance"
                   - "margin": Distance from decision boundary (most intuitive)
                   - "entropy": Information-theoretic uncertainty
                   - "variance": Probability variance p*(1-p)
            
        Returns:
            uncertainty_map: [H, W] values in [0, 1] (0=certain, 1=uncertain)
            stats: Dictionary with statistics
        """
        # Convert logits to probabilities
        prob = self._sigmoid(mask_logits)
        
        if method == "entropy":
            # Entropy-based uncertainty
            # H(p) = -p*log(p) - (1-p)*log(1-p)
            eps = 1e-10
            entropy = -(
                prob * np.log(prob + eps) + 
                (1 - prob) * np.log(1 - prob + eps)
            )
            # Normalize to [0, 1] (max entropy = log(2) for binary classification)
            uncertainty = entropy / np.log(2)
            
        elif method == "margin":
            # Distance from decision boundary (0.5)
            # Close to 0.5 = high uncertainty
            # Far from 0.5 (close to 0 or 1) = low uncertainty
            margin = np.abs(prob - 0.5)
            uncertainty = 1 - 2 * margin  # Transform to [0, 1]
            
        elif method == "variance":
            # For single prediction, use probability variance
            # High variance in p*(1-p) means uncertain
            variance = prob * (1 - prob)
            uncertainty = variance / 0.25  # Max variance = 0.25 at p=0.5
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'entropy', 'margin', or 'variance'")
        
        # Compute statistics
        stats = {
            'mean_uncertainty': float(uncertainty.mean()),
            'max_uncertainty': float(uncertainty.max()),
            'min_uncertainty': float(uncertainty.min()),
            'uncertain_pixels_ratio': float((uncertainty > self.uncertainty_threshold).sum() / uncertainty.size),
            'mean_confidence': float(1 - uncertainty.mean()),
            'high_confidence_pixels': int((uncertainty < 0.2).sum()),
            'uncertain_pixels': int((uncertainty > self.uncertainty_threshold).sum())
        }
        
        logger.info(f"Uncertainty computation complete: mean={stats['mean_uncertainty']:.3f}, "
                   f"uncertain_pixels={stats['uncertain_pixels_ratio']*100:.1f}%")
        
        return uncertainty, stats
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function"""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -20, 20)
        return 1 / (1 + np.exp(-x_clipped))
    
    def create_colored_overlay(
        self,
        uncertainty_map: np.ndarray,
        colormap: str = "RdYlGn_r"
    ) -> np.ndarray:
        """
        Create colored heatmap overlay
        
        Args:
            uncertainty_map: [H, W] uncertainty values in [0, 1]
            colormap: Matplotlib colormap name
                     "RdYlGn_r" = Red (uncertain) -> Yellow -> Green (certain)
            
        Returns:
            colored_map: [H, W, 3] RGB image (uint8)
        """
        import matplotlib.cm as cm
        
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Apply colormap (returns RGBA)
        colored = cmap(uncertainty_map)[:, :, :3]  # Drop alpha channel
        
        # Convert to uint8
        colored_uint8 = (colored * 255).astype(np.uint8)
        
        logger.debug(f"Created colored overlay with shape {colored_uint8.shape}")
        
        return colored_uint8
    
    def blend_with_image(
        self,
        base_image: np.ndarray,
        uncertainty_map: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "RdYlGn_r",
        mask: Optional[np.ndarray] = None,
        dilation_pixels: int = 15
    ) -> np.ndarray:
        """
        Blend uncertainty overlay with base image
        
        Args:
            base_image: Original image [H, W] or [H, W, 3]
            uncertainty_map: Uncertainty values [H, W]
            alpha: Overlay transparency (0=invisible, 1=opaque)
            colormap: Colormap for uncertainty visualization
            mask: Optional binary mask [H, W] to restrict uncertainty display
                  If provided, uncertainty is only shown within mask + dilation
            dilation_pixels: Number of pixels to dilate mask (to include boundary region)
            
        Returns:
            blended: [H, W, 3] RGB image with overlay
        """
        # Ensure base image is RGB
        if len(base_image.shape) == 2:
            base_rgb = np.stack([base_image] * 3, axis=-1)
        else:
            base_rgb = base_image.copy()
        
        # Ensure uint8
        if base_rgb.dtype != np.uint8:
            base_rgb = (base_rgb * 255).astype(np.uint8)
        
        # Create colored uncertainty overlay
        uncertainty_colored = self.create_colored_overlay(uncertainty_map, colormap)
        
        # Apply mask if provided (only show uncertainty in segmented regions + margin)
        if mask is not None:
            import cv2
            
            # Dilate mask to include boundary region
            if dilation_pixels > 0:
                kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
                mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            else:
                mask_dilated = mask.astype(np.uint8)
            
            # Create 3-channel mask
            mask_3ch = np.stack([mask_dilated] * 3, axis=-1).astype(bool)
            
            # Blend only within masked region
            blended = base_rgb.copy()
            blended[mask_3ch] = ((1 - alpha) * base_rgb[mask_3ch] + alpha * uncertainty_colored[mask_3ch]).astype(np.uint8)
            
            logger.debug(f"Applied masked uncertainty overlay: {mask_dilated.sum()} pixels affected")
        else:
            # Blend everywhere
            blended = ((1 - alpha) * base_rgb + alpha * uncertainty_colored).astype(np.uint8)
        
        logger.debug(f"Blended uncertainty overlay with alpha={alpha}")
        
        return blended
