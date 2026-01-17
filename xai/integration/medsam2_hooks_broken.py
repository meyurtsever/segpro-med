"""
MedSAM2 XAI Integration Hooks

This module provides the integration layer between the XAI feature
extraction and the existing MedSAM2 handler code.

It supports multiple visualization approaches:
1. LayerCAM (PRIMARY) - uses feature maps from Hiera backbone, fast and effective
2. GradCAM-based (COMMENTED) - uses gradients through CNN encoder
3. Mask-confidence based (fallback) - uses segmentation output

XAI computation is ALWAYS enabled by default.
The checkbox only controls VISIBILITY of the overlay.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import logging

# Import XAI components
from ..config import xai_config, get_xai_config
from ..segmentation.mask_confidence import (
    MaskConfidenceExtractor, 
    get_mask_confidence_extractor,
    create_focus_map
)

# LayerCAM - PRIMARY XAI METHOD
from ..segmentation.layercam_extractor import (
    LayerCAMExtractor,
    get_layercam_extractor
)

# GradCAM - COMMENTED OUT FOR NOW
# from ..segmentation.gradcam_extractor import (
#     SAM2GradCAMExtractor,
#     get_gradcam_extractor,
#     GRADCAM_AVAILABLE
# )
# GRADCAM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class XAIState:
    """State container for XAI during inference"""
    
    # LayerCAM extraction (PRIMARY METHOD)
    layercam_extractor: Optional[LayerCAMExtractor] = None
    last_layercam: Optional[np.ndarray] = None
    
    # GradCAM extraction (COMMENTED OUT)
    # gradcam_extractor: Optional[SAM2GradCAMExtractor] = None
    # last_gradcam: Optional[np.ndarray] = None
    
    # Mask confidence extraction (fallback)
    mask_confidence_extractor: Optional[MaskConfidenceExtractor] = None
    last_mask: Optional[np.ndarray] = None
    last_image: Optional[np.ndarray] = None
    
    # Configuration - XAI always computes, checkbox just controls display
    show_overlay: bool = False  # Checkbox controls this
    
    # Cached results for instant toggling
    cached_overlay: Optional[np.ndarray] = None
    cached_base_image: Optional[np.ndarray] = None
    last_image_size: Tuple[int, int] = (512, 512)
    
    # Which XAI method produced the result
    xai_source: str = "none"


class XAIMedSAM2Integration:
    """
    Integration class for XAI features with MedSAM2.
    
    XAI computation is ALWAYS enabled by default.
    The checkbox only controls VISIBILITY of the overlay.
    
    Primary method: LayerCAM (fast, works with feature maps only)
    
    Usage:
        xai = get_xai_integration()
        xai.prepare_for_inference(model)
        # ... run inference ...
        xai.capture_xai_data(image, mask)
        display_image = xai.get_display_image(base_image, show_overlay=checkbox_value)
    """
    
    def __init__(self, use_fast_layercam: bool = True):
        """
        Initialize XAI integration.
        
        Args:
            use_fast_layercam: If True, use feature-based LayerCAM (no gradients, faster)
                              If False, use gradient-based LayerCAM (more accurate)
        """
        self.state = XAIState()
        self._model_ref = None
        self.use_fast_layercam = use_fast_layercam
        
        # Initialize extractors
        self.state.mask_confidence_extractor = get_mask_confidence_extractor()
        
        # Initialize LayerCAM (PRIMARY METHOD)
        self.state.layercam_extractor = get_layercam_extractor(use_gradients=not use_fast_layercam)
        
        # GradCAM - COMMENTED OUT
        # if GRADCAM_AVAILABLE:
        #     self.state.gradcam_extractor = get_gradcam_extractor()
        
        logger.info(f"XAIMedSAM2Integration initialized (LayerCAM - Fast mode: {use_fast_layercam})")
    
    def prepare_for_inference(
        self, LayerCAM (PRIMARY METHOD)
        if self.state.layercam_extractor is not None:
            try:
                success = self.state.layercam_extractor.setup(model)
                if success:
                    mode = "fast (feature-based)" if self.use_fast_layercam else "gradient-based"
                    logger.info(f"XAI: LayerCAM ready ({mode})")
                    return True
            except Exception as e:
                logger.warning(f"LayerCAM setup failed: {e}")
        
        # GradCAM - COMMENTED OUT
        # if GRADCAM_AVAILABLE and self.state.gradcam_extractor is not None:
        #     try:
        #         success = self.state.gradcam_extractor.setup(model)
        #         if success:
        #             logger.info("XAI: GradCAM ready")
        #             return True
        #     except Exception as e:
        #     model: The MedSAM2 model
            show_overlay: Whether to show overlay (checkbox state)
            
        Returns:
            True if preparation successful
        """
        self.state.show_overlay = show_overlay
        self._model_ref = model
        
        # Setup GradCAM if available
        if GRADCAM_AVAILABLE and self.state.gradcam_extractor is not None:
            try:
                success = self.state.gradcam_extractor.setup(model)
                if success:
                    logger.info("XAI: GradCAM ready")
                    return True
            except Exception as e:
                logger.warning(f"GradCAM setup failed: {e}")
        
        logger.info("XAI: Using mask confidence fallback")
        return True
    
    def capture_xai_data(
        self, 
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        prompts: Optional[List[Tuple[int, int]]] = None
    ) -> None:
        """
        Capture XAI data after inference.
        
        Args:
            image: Original input image
            mask: Segmentation mask
            prompts: Point prompts used (for visualization)
        """
        # Store data
        self.state.last_image = image.copy() if image is not None else None
        self.state.last_mask = mask
        if image is not None:
            self.state.last_image_size = image.shape[:2]
        
        # Set prompts for mask confidence
        if prompts and self.state.mask_confidence_extractor:
            self.state.mask_confidence_extractor.set_prompts(prompts, None)
        
        # Try LayerCAM first (PRIMARY METHOD)
        if (self.state.layercam_extractor is not None and 
            image is not None and 
            self._model_ref is not None):
            try:
                layercam = self.state.layercam_extractor.compute_cam(
                    image=image,
                    mask=mask,
                    normalize=True
                )
                if layercam is not None and layercam.max() > 0:
                    self.state.last_layercam = layercam
                    self.state.xai_source = "layercam"
                    logger.info(f"XAI: LayerCAM computed, range=[{layercam.min():.3f}, {layercam.max():.3f}]")
                    return
            except Exception as e:
                logger.warning(f"LayerCAM failed: {e}")
        
        # GradCAM - COMMENTED OUT
        # if (GRADCAM_AVAILABLE and 
        #     self.state.gradcam_extractor is not None and 
        #     image is not None and 
        #     self._model_ref is not None):
        #     try:
        #         gradcam = self.state.gradcam_extractor.compute_cam(
        #             image=image,
        #             mask=mask,
        #             normalize=True
        #         )
        #         if gradcam is not None and gradcam.max() > 0:
        #             self.state.last_gradcam = gradcam
        #             self.state.xai_source = "gradcam"
        #             logger.info(f"XAI: GradCAM computed, range=[{gradcam.min():.3f}, {gradcam.max():.3f}]")
        #             return
        #     except Exception as e:
        #         gradcam = self.state.gradcam_extractor.compute_cam(
                    image=image,
                    mask=mask,
                    normalize=True
                )
                if gradcam is not None and gradcam.max() > 0:
                    self.state.last_gradcam = gradcam
                    self.state.xai_source = "gradcam"
                    logger.info(f"XAI: GradCAM computed, range=[{gradcam.min():.3f}, {gradcam.max():.3f}]")
                    return
            except Exception as e:
                logger.warning(f"GradCAM failed: {e}")
        
        # Fallback to mask confidence
        if mask is not None and self.state.mask_confidence_extractor is not None:
            self.state.mask_confidence_extractor.capture_outputs(mask, None)
            self.state.xai_source = "mask_confidence"
            logger.info("XAI: Using mask confidence visualization")
    
    def get_display_image(
        self,
        base_image: np.ndarray,
        show_overlay: Optional[bool] = None
    ) -> np.ndarray:
        """
        Get display image with or without XAI overlay.
        
        Args:
            base_image: Base image
            show_overlay: Whether to show overlay (uses state if None)
            
        Returns:
            Image with or without XAI overlay
        """
        # Cache the base image
        self.state.cached_base_image = base_image.copy()
        
        # Check if we should show overlay
        show =LayerCAM first (PRIMARY METHOD)
        if self.state.last_layercam is not None:
            heatmap = self.state.last_layercam
        # Try GradCAM (COMMENTED OUT)
        # elif self.state.last_gradcam is not None:
        # if not show:
            return base_image
        
        return self._apply_overlay(base_image)
    
    def _apply_overlay(self, base_image: np.ndarray) -> np.ndarray:
        """Apply XAI heatmap overlay to image."""
        import cv2
        
        config = get_xai_config()
        colormap = config.attention_colormap
        alpha = config.heatmap_alpha
        
        # Get visualization map
        heatmap = None
        
        # Try GradCAM
        if self.state.last_gradcam is not None:
            heatmap = self.state.last_gradcam
        # Fallback to mask confidence
        elif self.state.mask_confidence_extractor is not None:
            heatmap = self.state.mask_confidence_extractor.get_best_visualization()
        
        if heatmap is None:
            logger.debug("No XAI heatmap available")
            return base_image
        
        # Ensure base image is uint8 RGB
        if base_image.dtype != np.uint8:
            if base_image.max() <= 1.0:
                base_image = (base_image * 255).astype(np.uint8)
            else:
                base_image = base_image.astype(np.uint8)
        
        if base_image.ndim == 2:
            base_image = np.stack([base_image] * 3, axis=-1)
        
        # Resize heatmap if needed
        if heatmap.shape[:2] != base_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (base_image.shape[1], base_image.shape[0]))
        
        # Apply colormap
        heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
        colormap_cv = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "inferno": cv2.COLORMAP_INFERNO,
        }.get(colormap, cv2.COLORMAP_JET)
        
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap_cv)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        result = cv2.addWeighted(base_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Cache
        self.state.cached_overlay = result
        
        logger.info(f"XAI overlay applied (source: {self.state.xai_source})")
        return result
    
    def toggle_overlay(self, show: bool) -> Optional[np.ndarray]:
        """
        Toggle overlay visibility. Returns cached image for instant toggle.
        
        Args:
            show: Whether to show overlay
            
        Returns:
            Cached image (overlay or base) or None if not available
        """
        self.state.show_overlay = show
        logger.info(f"XAI overlay: {'SHOW' if show else 'HIDE'}")
        
        if show:
            # Return cached overlay, or compute it
            if self.state.cached_overlay is not None:
                return self.state.cached_overlay
            elif self.state.cached_base_image is not None:
                return self._apply_overlay(self.state.cached_base_image)
        else:
            return self.state.cached_base_image
        
        return None
    
    def has_xai_data(self) -> bool:
        """Check if XAI data is available."""
        return (
            self.state.last_layercam is not None or
            # self.state.last_gradcam is not None or  # COMMENTED OUT
            (self.state.mask_confidence_extractor is not None and 
             self.state.mask_confidence_extractor.get_best_visualization() is not None)
        )
    
    def is_overlay_visible(self) -> bool:
        """Check if overlay is shown."""
        return self.state.show_overlay
    
    def get_xai_source(self) -> str:
        """Get the source of current XAI visualization."""
        return self.state.xai_source
    
    def cleanup(self):
        """Clean up XAI resources."""
        self.state.last_layercam = None
        # self.state.last_gradcam = None  # COMMENTED OUT
        self.state.last_mask = None
        self.state.last_image = None
        self.state.cached_overlay = None
        self.state.cached_base_image = None
        self.state.xai_source = "none"
        
        # Clean up LayerCAM
        if self.state.layercam_extractor:
            self.state.layercam_extractor.clear()
        
        # GradCAM - COMMENTED OUT
        # if self.state.gradcam_extractor:
        #     self.state.gradcam_extractor.clear()
        
        logger.debug("XAI resources cleaned up")
    
    # Legacy compatibility methods
    def is_enabled(self) -> bool:
        """Legacy: Check if XAI is enabled (always True now)."""
        return True
    
    def toggle(self, enabled: bool):
        """Legacy: Toggle XAI (now just toggles visibility)."""
        self.toggle_overlay(enabled)


# Global singleton instance
_xai_integration_instance: Optional[XAIMedSAM2Integration] = None


def get_xai_integration() -> XAIMedSAM2Integration:
    """Get the global XAI integration instance."""
    global _xai_integration_instance
    if _xai_integration_instance is None:
        _xai_integration_instance = XAIMedSAM2Integration()
    return _xai_integration_instance


def initialize_xai_for_medsam2(
    model: torch.nn.Module = None,
    enabled: bool = True
) -> XAIMedSAM2Integration:
    """
    Initialize XAI integration for a MedSAM2 model.
    
    Legacy compatibility function. XAI is always enabled now,
    the 'enabled' parameter just controls overlay visibility.
    
    Args:
        model: MedSAM2 model (optional, for GradCAM setup)
        enabled: Whether to show overlay (default True)
        
    Returns:
        Configured XAI integration
    """
    integration = get_xai_integration()
    integration.toggle_overlay(enabled)
    return integration


# Exported flag
XAI_AVAILABLE = True
