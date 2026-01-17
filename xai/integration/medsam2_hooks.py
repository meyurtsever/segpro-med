"""
MedSAM2 XAI Integration Hooks

This module provides the integration layer between the XAI feature
extraction and the existing MedSAM2 handler code.

Primary and only method: Gradient Decoder XAI
- Computes gradients from IoU score back to vision features (decoder input)
- Works reliably with SAM2's architecture (avoids FlashAttention issues)
- Supports both point and box prompts
- Provides decoder-level explanations without encoder gradient requirements

XAI computation is ALWAYS enabled by default.
The checkbox only controls VISIBILITY of the overlay.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import logging

# Import XAI components
from ..config import xai_config, get_xai_config

# Gradient-based Decoder XAI - PRIMARY AND ONLY METHOD
from ..segmentation.gradient_decoder_xai import (
    GradientDecoderXAI,
    create_gradient_decoder_xai
)

logger = logging.getLogger(__name__)


@dataclass
class XAIState:
    """State container for XAI during inference"""
    
    # Gradient-based Decoder XAI (PRIMARY AND ONLY METHOD)
    gradient_xai: Optional[GradientDecoderXAI] = None
    last_gradient_map: Optional[np.ndarray] = None
    
    # Configuration - XAI always computes, checkbox just controls display
    show_overlay: bool = False  # Checkbox controls this
    
    # Cached results for instant toggling
    cached_overlay: Optional[np.ndarray] = None
    cached_base_image: Optional[np.ndarray] = None
    last_image_size: Tuple[int, int] = (512, 512)
    
    # XAI source (always "gradient_decoder" when available)
    xai_source: str = "none"


class XAIMedSAM2Integration:
    """
    Integration class for XAI features with MedSAM2.
    
    XAI computation is ALWAYS enabled by default.
    The checkbox only controls VISIBILITY of the overlay.
    
    Primary and only method: Gradient Decoder XAI
    - Computes gradients from IoU score back to vision features
    - Works reliably with SAM2's architecture (avoids FlashAttention issues)
    - Provides decoder-level explanations for both point and box prompts
    
    Usage:
        xai = get_xai_integration()
        xai.prepare_for_inference(model, predictor)
        # ... run inference ...
        xai.capture_xai_data(image, mask, boxes=box_coords)
        display_image = xai.get_display_image(base_image, show_overlay=checkbox_value)
    """
    
    def __init__(self):
        """Initialize XAI integration with Gradient Decoder XAI."""
        self.state = XAIState()
        self._model_ref = None
        self._predictor_ref = None
        
        # Initialize gradient decoder XAI (only method)
        self.state.gradient_xai = create_gradient_decoder_xai()
        
        logger.info("XAIMedSAM2Integration initialized (Gradient Decoder XAI only)")
    
    def prepare_for_inference(
        self, 
        model: torch.nn.Module,
        predictor=None,
        show_overlay: bool = False
    ) -> bool:
        """
        Prepare XAI for inference. Always enabled, show_overlay controls display.
        
        Args:
            model: The MedSAM2 model
            predictor: SAM2ImagePredictor instance (required for gradient XAI)
            show_overlay: Whether to show overlay (checkbox state)
            
        Returns:
            True if preparation successful
        """
        self.state.show_overlay = show_overlay
        self._model_ref = model
        self._predictor_ref = predictor
        
        # Initialize Gradient Decoder XAI (ONLY METHOD)
        if predictor is not None and self.state.gradient_xai is not None:
            try:
                logger.info("XAI: Setting up Gradient Decoder XAI...")
                success = self.state.gradient_xai.setup(predictor)
                if success:
                    logger.info("XAI: Gradient Decoder XAI ready - decoder-level explanations enabled")
                    return True
                else:
                    logger.error("XAI: Gradient Decoder XAI setup failed")
                    return False
            except Exception as e:
                logger.error(f"XAI: Gradient XAI setup failed: {e}")
                return False
        else:
            logger.error("XAI: Predictor required for Gradient Decoder XAI")
            return False
    
    def capture_xai_data(
        self, 
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        prompts: Optional[List[Tuple[int, int]]] = None,
        boxes: Optional[np.ndarray] = None
    ) -> None:
        """
        Capture XAI data after inference.
        
        Args:
            image: Original input image
            mask: Segmentation mask (not used in gradient XAI, kept for compatibility)
            prompts: Point prompts used (for gradient computation)
            boxes: Box prompts (M, 4) in XYXY format - for gradient XAI
        """
        logger.info(f"🔥 capture_xai_data CALLED: image={image is not None}, mask={mask is not None}, prompts={prompts}, boxes={boxes}")
        
        # Gradient-based Decoder XAI (ONLY METHOD)
        # Computes gradients from IoU score back to vision features (decoder input)
        # Provides true decoder-level explanations without FlashAttention issues
        # Supports both point prompts and box prompts
        logger.info(f"XAI capture_xai_data: gradient_xai={self.state.gradient_xai is not None}, "
                   f"image={image is not None}, prompts={prompts}, boxes={boxes}")
        
        if self.state.gradient_xai is None:
            logger.error("XAI: Gradient decoder not initialized")
            return
        
        if image is None:
            logger.warning("XAI: No image provided")
            return
        
        if prompts is None and boxes is None:
            logger.warning("XAI: No prompts or boxes provided")
            return
        
        try:
            logger.info("XAI: Computing gradient-based decoder explanation...")
            
            # Convert prompts to numpy arrays for SAM2
            points_np = None
            labels_np = None
            boxes_np = None
            
            if prompts is not None and len(prompts) > 0:
                points_np = np.array(prompts, dtype=np.float32)
                labels_np = np.ones(len(prompts), dtype=np.int32)  # All foreground points
                logger.info(f"XAI: Converted {len(prompts)} point prompts to numpy")
            
            if boxes is not None and len(boxes) > 0:
                boxes_np = np.array(boxes, dtype=np.float32)
                logger.info(f"XAI: Converted {len(boxes)} box prompts to numpy: {boxes_np}")
            
            # Compute gradient map
            gradient_map = self.state.gradient_xai.compute_gradient_map(
                image=image,
                points=points_np,
                labels=labels_np,
                boxes=boxes_np
            )
            
            if gradient_map is not None and gradient_map.max() > 0:
                self.state.last_gradient_map = gradient_map
                self.state.xai_source = "gradient_decoder"
                logger.info(f"✅ XAI: Gradient decoder map computed successfully, "
                          f"range=[{gradient_map.min():.3f}, {gradient_map.max():.3f}]")
            else:
                logger.warning("⚠️ XAI: Gradient map is invalid or all zeros")
                self.state.last_gradient_map = None
                self.state.xai_source = "none"
                
        except Exception as e:
            logger.error(f"❌ XAI: Gradient XAI computation failed: {e}")
            import traceback
            logger.debug(f"Gradient XAI traceback: {traceback.format_exc()}")
            self.state.last_gradient_map = None
            self.state.xai_source = "none"
    
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
        show = show_overlay if show_overlay is not None else self.state.show_overlay
        
        logger.info(f"XAI get_display_image: show={show}, has_gradient_map={self.state.last_gradient_map is not None}")
        
        if not show:
            logger.info("XAI overlay disabled - returning base image")
            return base_image
        
        return self._apply_overlay(base_image)
    
    def _apply_overlay(self, base_image: np.ndarray) -> np.ndarray:
        """Apply XAI heatmap overlay to image."""
        import cv2
        
        config = get_xai_config()
        alpha = config.heatmap_alpha
        
        # Get gradient decoder XAI map (ONLY METHOD)
        heatmap = self.state.last_gradient_map
        
        if heatmap is None:
            logger.warning("No gradient XAI map available - cannot display overlay")
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
        
        # Create bright red heatmap with high contrast: black (0) → bright red (1)
        # Red colormap is optimal for medical imaging visualization
        heatmap_colored = np.zeros((heatmap_uint8.shape[0], heatmap_uint8.shape[1], 3), dtype=np.uint8)
        
        # Apply gamma correction to boost visibility of mid-range values
        gamma = 0.5  # Lower gamma = brighter mid-tones
        heatmap_boosted = np.power(heatmap_uint8 / 255.0, gamma) * 255.0
        heatmap_boosted = heatmap_boosted.astype(np.uint8)
        
        heatmap_colored[:, :, 0] = heatmap_boosted  # Red channel only (FULL INTENSITY)
        logger.debug(f"Applied RED colormap for gradient decoder XAI (gamma={gamma})")
        
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
        return self.state.last_gradient_map is not None
    
    def is_overlay_visible(self) -> bool:
        """Check if overlay is shown."""
        return self.state.show_overlay
    
    def get_xai_source(self) -> str:
        """Get the source of current XAI visualization."""
        return self.state.xai_source
    
    def cleanup(self):
        """Clean up XAI resources."""
        self.state.last_gradient_map = None
        self.state.cached_overlay = None
        self.state.cached_base_image = None
        self.state.xai_source = "none"
        
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
        model: MedSAM2 model (optional)
        enabled: Whether to show overlay (default True)
        
    Returns:
        Configured XAI integration
    """
    integration = get_xai_integration()
    integration.toggle_overlay(enabled)
    return integration


# Exported flag
XAI_AVAILABLE = True
