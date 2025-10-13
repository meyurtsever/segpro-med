"""
Professional Viewport System for High-Resolution Medical Images

This module provides a viewport-based approach for displaying high-resolution images
(like mammography 3518×2800) in a fixed-size display window while maintaining
full image quality and accurate coordinates.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ImageViewport:
    """
    Professional viewport system for high-resolution medical images.
    
    Features:
    - Maintains original image resolution (no quality loss)
    - Fixed viewport dimensions for consistent UI layout
    - Smooth zoom/pan operations 
    - Accurate coordinate mapping between viewport and original image
    - Supports mouse wheel zoom and pan operations
    """
    
    def __init__(self, viewport_width: int = 1200, viewport_height: int = 800):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        # Viewport state
        self.original_image: Optional[np.ndarray] = None
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0  # Pan offset in original image coordinates
        self.pan_y: float = 0.0
        
        # Image properties
        self.original_width: int = 0
        self.original_height: int = 0
        
        # Viewport bounds in original image coordinates
        self.view_left: float = 0.0
        self.view_top: float = 0.0
        self.view_right: float = 0.0
        self.view_bottom: float = 0.0
        
    def set_image(self, image: np.ndarray) -> None:
        """Set the original high-resolution image with proper aspect ratio handling."""
        self.original_image = image.copy()
        self.original_height, self.original_width = image.shape[:2]
        
        # Calculate aspect ratios
        original_aspect = self.original_width / self.original_height
        viewport_aspect = self.viewport_width / self.viewport_height
        
        # Calculate initial zoom to fit image in viewport while preserving aspect ratio
        if original_aspect > viewport_aspect:
            # Image is wider than viewport - fit to width
            self.zoom_level = self.viewport_width / self.original_width
        else:
            # Image is taller than viewport - fit to height  
            self.zoom_level = self.viewport_height / self.original_height
        
        # Center the image initially
        self._center_image()
        
        logger.info(f"Viewport initialized: {self.original_width}×{self.original_height} → viewport {self.viewport_width}×{self.viewport_height}")
        logger.info(f"Original aspect: {original_aspect:.3f}, Viewport aspect: {viewport_aspect:.3f}")
        logger.info(f"Initial zoom level: {self.zoom_level:.3f} (preserves aspect ratio)")
    
    def _center_image(self) -> None:
        """Center the image in the viewport."""
        # For medical imaging with fixed viewport size (image_annotator), we allow zoom < 1.0
        # to show the complete image while preserving aspect ratio. This is essential for
        # mammography images (3518×2800) displayed in image_annotator (1200×600).
        # The original data remains lossless - we're just showing a downscaled VIEW.
        # Users can zoom in to see original resolution pixels.
            
        # Calculate the visible area size in original coordinates
        visible_width = self.viewport_width / self.zoom_level
        visible_height = self.viewport_height / self.zoom_level
        
        # Center the viewport
        self.pan_x = (self.original_width - visible_width) / 2
        self.pan_y = (self.original_height - visible_height) / 2
        
        self._update_viewport_bounds()
    
    def _update_viewport_bounds(self) -> None:
        """Update viewport bounds in original image coordinates."""
        visible_width = self.viewport_width / self.zoom_level
        visible_height = self.viewport_height / self.zoom_level
        
        self.view_left = max(0, self.pan_x)
        self.view_top = max(0, self.pan_y)
        self.view_right = min(self.original_width, self.pan_x + visible_width)
        self.view_bottom = min(self.original_height, self.pan_y + visible_height)
        
        # Adjust pan if we've gone out of bounds
        if self.view_left <= 0:
            self.pan_x = 0
        if self.view_top <= 0:
            self.pan_y = 0
        if self.view_right >= self.original_width:
            self.pan_x = self.original_width - visible_width
        if self.view_bottom >= self.original_height:
            self.pan_y = self.original_height - visible_height
    
    def zoom(self, zoom_factor: float, center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """
        Zoom the viewport by the given factor with lossless quality preservation.
        
        Args:
            zoom_factor: Zoom multiplier (>1 = zoom in, <1 = zoom out)
            center_x: Zoom center in viewport coordinates (optional)
            center_y: Zoom center in viewport coordinates (optional)
        """
        old_zoom = self.zoom_level
        
        # MEDICAL IMAGING ZOOM LIMITS:
        # For mammography in fixed image_annotator size, we need to allow zoom < 1.0
        # to show the complete image. The original data quality is preserved - 
        # we're showing a downscaled view that users can zoom into for detail.
        
        # Calculate minimum zoom to fit entire image with aspect ratio preservation
        fit_width_zoom = self.viewport_width / self.original_width
        fit_height_zoom = self.viewport_height / self.original_height
        min_fit_zoom = min(fit_width_zoom, fit_height_zoom)
        
        min_zoom = min_fit_zoom  # Allow zoom below 1x to show complete image
        max_zoom = 20.0  # Allow up to 20x zoom for detailed mammography inspection
        
        # At min_fit_zoom: viewport shows complete image with aspect ratio preserved
        # At zoom 1.0: viewport shows native resolution pixels (may be cropped for large images)
        # At zoom 2.0: viewport shows upscaled view for detailed inspection
        
        self.zoom_level = max(min_zoom, min(max_zoom, self.zoom_level * zoom_factor))
        
        # If zoom center is specified, adjust pan to zoom towards that point
        if center_x is not None and center_y is not None:
            # Convert viewport coordinates to original image coordinates (using old zoom)
            old_visible_width = self.viewport_width / old_zoom
            old_visible_height = self.viewport_height / old_zoom
            
            # Calculate the original coordinate the user clicked on
            ratio_x = center_x / self.viewport_width
            ratio_y = center_y / self.viewport_height
            
            orig_click_x = self.view_left + ratio_x * (self.view_right - self.view_left)
            orig_click_y = self.view_top + ratio_y * (self.view_bottom - self.view_top)
            
            # Calculate new visible area size
            new_visible_width = self.viewport_width / self.zoom_level
            new_visible_height = self.viewport_height / self.zoom_level
            
            # Center the new view around the clicked point
            self.pan_x = orig_click_x - new_visible_width * ratio_x
            self.pan_y = orig_click_y - new_visible_height * ratio_y
        
        self._update_viewport_bounds()
        
        # Calculate quality metrics for zoom level
        visible_pixels_per_original = self.zoom_level
        native_resolution = visible_pixels_per_original >= 1.0
        
        logger.debug(f"Zoomed to {self.zoom_level:.3f}x")
        logger.debug(f"  Visible region: {self.view_right - self.view_left:.1f}×{self.view_bottom - self.view_top:.1f} pixels from original")
        logger.debug(f"  Quality: {'NATIVE/UPSCALE' if native_resolution else 'DOWNSCALE'}")
        logger.debug(f"  Pixels per original: {visible_pixels_per_original:.3f}x")
    
    def pan(self, delta_x: float, delta_y: float) -> None:
        """
        Pan the viewport by the given offset.
        
        Args:
            delta_x: Pan offset in viewport coordinates
            delta_y: Pan offset in viewport coordinates
        """
        # Convert viewport delta to original image coordinates
        orig_delta_x = delta_x / self.zoom_level
        orig_delta_y = delta_y / self.zoom_level
        
        self.pan_x -= orig_delta_x  # Negative because pan direction is opposite to drag
        self.pan_y -= orig_delta_y
        
        self._update_viewport_bounds()
    
    def _get_display_bounds(self):
        """Get the actual display bounds accounting for centering and aspect ratio."""
        # Calculate visible region size in original coordinates
        visible_width = (self.view_right - self.view_left)
        visible_height = (self.view_bottom - self.view_top)
        
        # Calculate aspect ratio
        visible_aspect = visible_width / visible_height if visible_height > 0 else 1.0
        viewport_aspect = self.viewport_width / self.viewport_height
        
        # Calculate actual display size and offset
        if visible_aspect > viewport_aspect:
            # Fit to viewport width
            display_width = self.viewport_width
            display_height = int(self.viewport_width / visible_aspect)
            display_x_offset = 0
            display_y_offset = (self.viewport_height - display_height) // 2
        else:
            # Fit to viewport height
            display_height = self.viewport_height
            display_width = int(self.viewport_height * visible_aspect)
            display_x_offset = (self.viewport_width - display_width) // 2
            display_y_offset = 0
            
        return display_x_offset, display_y_offset, display_width, display_height
    
    def viewport_to_original_x(self, viewport_x: float) -> float:
        """Convert viewport x coordinate to original image x coordinate with proper aspect ratio handling."""
        display_x_offset, _, display_width, _ = self._get_display_bounds()
        
        # Adjust for centering offset
        adjusted_x = viewport_x - display_x_offset
        
        # Ensure within display bounds
        if adjusted_x < 0 or display_width <= 0:
            return self.view_left
        if adjusted_x >= display_width:
            return self.view_right
            
        # Convert to original coordinates
        ratio = adjusted_x / display_width
        return self.view_left + ratio * (self.view_right - self.view_left)
    
    def viewport_to_original_y(self, viewport_y: float) -> float:
        """Convert viewport y coordinate to original image y coordinate with proper aspect ratio handling."""
        _, display_y_offset, _, display_height = self._get_display_bounds()
        
        # Adjust for centering offset
        adjusted_y = viewport_y - display_y_offset
        
        # Ensure within display bounds
        if adjusted_y < 0 or display_height <= 0:
            return self.view_top
        if adjusted_y >= display_height:
            return self.view_bottom
            
        # Convert to original coordinates
        ratio = adjusted_y / display_height
        return self.view_top + ratio * (self.view_bottom - self.view_top)
    
    def original_to_viewport_x(self, original_x: float) -> float:
        """Convert original image x coordinate to viewport x coordinate with proper aspect ratio handling."""
        display_x_offset, _, display_width, _ = self._get_display_bounds()
        
        # Convert to display coordinates
        if self.view_right <= self.view_left:
            return display_x_offset
            
        ratio = (original_x - self.view_left) / (self.view_right - self.view_left)
        ratio = max(0, min(1, ratio))  # Clamp to [0, 1]
        
        return display_x_offset + ratio * display_width
    
    def original_to_viewport_y(self, original_y: float) -> float:
        """Convert original image y coordinate to viewport y coordinate with proper aspect ratio handling."""
        _, display_y_offset, _, display_height = self._get_display_bounds()
        
        # Convert to display coordinates  
        if self.view_bottom <= self.view_top:
            return display_y_offset
            
        ratio = (original_y - self.view_top) / (self.view_bottom - self.view_top)
        ratio = max(0, min(1, ratio))  # Clamp to [0, 1]
        
        return display_y_offset + ratio * display_height
    
    def get_viewport_image(self) -> np.ndarray:
        """
        TRUE VIEWPORT APPROACH with aspect ratio preservation.
        The viewport shows a portion of the original image at native/upscaled resolution.
        We maintain aspect ratio by letterboxing when necessary.
        
        Returns:
            np.ndarray: Fixed-size viewport showing native resolution pixels with correct aspect ratio
        """
        if self.original_image is None:
            raise ValueError("No image set in viewport")
        
        # Update viewport bounds to get the visible region
        self._update_viewport_bounds()
        
        # Extract the visible region boundaries
        left = max(0, int(np.floor(self.view_left)))
        top = max(0, int(np.floor(self.view_top)))
        right = min(self.original_width, int(np.ceil(self.view_right)))
        bottom = min(self.original_height, int(np.ceil(self.view_bottom)))
        
        # Extract the region from original image
        if right <= left or bottom <= top:
            return np.zeros((self.viewport_height, self.viewport_width, 3), dtype=np.uint8)
            
        extracted_region = self.original_image[top:bottom, left:right]
        
        if extracted_region.size == 0:
            return np.zeros((self.viewport_height, self.viewport_width, 3), dtype=np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(extracted_region.shape) == 2:
            extracted_region = cv2.cvtColor(extracted_region, cv2.COLOR_GRAY2RGB)
        
        # Get dimensions of extracted region
        region_height, region_width = extracted_region.shape[:2]
        
        # Calculate aspect ratios
        region_aspect = region_width / region_height
        viewport_aspect = self.viewport_width / self.viewport_height
        
        # Calculate target dimensions while preserving aspect ratio
        if region_aspect > viewport_aspect:
            # Region is wider - fit to viewport width
            target_width = self.viewport_width
            target_height = int(self.viewport_width / region_aspect)
        else:
            # Region is taller - fit to viewport height
            target_height = self.viewport_height
            target_width = int(self.viewport_height * region_aspect)
        
        # Ensure minimum size
        target_width = max(1, target_width)
        target_height = max(1, target_height)
        
        # Create viewport image with black background
        viewport_image = np.zeros((self.viewport_height, self.viewport_width, 3), dtype=np.uint8)
        
        # Determine if we can achieve perfect 1:1 mapping (BEST quality)
        scale_x = target_width / region_width
        scale_y = target_height / region_height
        scale = min(scale_x, scale_y)
        
        # PRIORITIZE NATIVE RESOLUTION: Check if we can use 1:1 mapping
        native_fit_width = min(region_width, self.viewport_width)
        native_fit_height = min(region_height, self.viewport_height)
        
        # Use native resolution when possible (zoom >= 1.0x and region fits in viewport)
        if (self.zoom_level >= 1.0 and 
            region_width <= self.viewport_width and 
            region_height <= self.viewport_height):
            # PERFECT 1:1 MAPPING - NO RESIZING (ABSOLUTE BEST QUALITY)
            resized_region = extracted_region
            actual_width, actual_height = region_width, region_height
            logger.debug(f"🎯 NATIVE RESOLUTION: Perfect 1:1 pixel mapping ({region_width}×{region_height}) - BEST QUALITY")
            
        elif abs(scale - 1.0) < 0.01:  # Very close to 1:1, avoid tiny scaling artifacts
            # Nearly perfect mapping - avoid unnecessary interpolation
            resized_region = extracted_region
            actual_width, actual_height = region_width, region_height
            logger.debug(f"🎯 NEAR-NATIVE: Scale {scale:.3f} too close to 1.0 - using native pixels")
            
        else:
            # Need to resize - use appropriate high-quality interpolation
            if scale > 1.0:
                # Upscaling - use high-quality interpolation for medical imaging
                # LANCZOS4 provides the best quality for upscaling medical images
                interpolation = cv2.INTER_LANCZOS4
                logger.debug(f"📈 UPSCALING: Using LANCZOS4 for scale {scale:.3f}x")
            else:
                # Downscaling - use INTER_AREA for high quality downsampling
                # This prevents aliasing and preserves image quality  
                interpolation = cv2.INTER_AREA
                logger.debug(f"📉 DOWNSCALING: Using INTER_AREA for scale {scale:.3f}x")
            
            resized_region = cv2.resize(extracted_region, (target_width, target_height), interpolation=interpolation)
            actual_width, actual_height = target_width, target_height
        
        # Center the region in the viewport (letterboxing if needed)
        start_x = (self.viewport_width - actual_width) // 2
        start_y = (self.viewport_height - actual_height) // 2
        
        viewport_image[start_y:start_y + actual_height, start_x:start_x + actual_width] = resized_region
        
        # Calculate final quality metrics for logging
        final_scale = min(actual_width / region_width, actual_height / region_height)
        if final_scale == 1.0:
            quality_status = "🎯 PERFECT 1:1"
        elif final_scale > 1.0:
            quality_status = "📈 UPSCALED"
        else:
            quality_status = "📉 DOWNSCALED"
        
        logger.debug(f"{quality_status}: {region_width}×{region_height} → {actual_width}×{actual_height} (final_scale={final_scale:.3f}, zoom={self.zoom_level:.3f})")
        
        return viewport_image
    
    def reset_view(self) -> None:
        """Reset to fit entire image with aspect ratio preserved."""
        # Calculate zoom to fit complete image with aspect ratio preservation
        fit_width_zoom = self.viewport_width / self.original_width
        fit_height_zoom = self.viewport_height / self.original_height
        self.zoom_level = min(fit_width_zoom, fit_height_zoom)
        self._center_image()
        logger.info(f"Viewport reset to fit zoom {self.zoom_level:.3f}x (complete image with aspect ratio preserved)")
    
    def get_viewport_info(self) -> Dict[str, Any]:
        """Get viewport information with accurate aspect ratio and quality metrics."""
        self._update_viewport_bounds()
        
        # Calculate visible region
        visible_width = self.view_right - self.view_left
        visible_height = self.view_bottom - self.view_top
        
        # Calculate aspect ratios
        visible_aspect = visible_width / visible_height if visible_height > 0 else 1.0
        viewport_aspect = self.viewport_width / self.viewport_height
        
        # Calculate how the visible region fits in viewport (same logic as get_viewport_image)
        if visible_aspect > viewport_aspect:
            # Visible region is wider - fit to viewport width
            target_width = self.viewport_width
            target_height = int(self.viewport_width / visible_aspect)
        else:
            # Visible region is taller - fit to viewport height
            target_height = self.viewport_height
            target_width = int(self.viewport_height * visible_aspect)
        
        # Calculate scale factor and quality ratio
        scale_x = target_width / visible_width if visible_width > 0 else 1.0
        scale_y = target_height / visible_height if visible_height > 0 else 1.0
        scale = min(scale_x, scale_y)
        
        # Quality ratio: original pixels per display pixel
        quality_ratio = 1.0 / scale if scale > 0 else 1.0
        
        # Lossless quality when scale >= 1.0 (no downscaling)
        lossless_quality = scale >= 1.0
        
        # Calculate display bounds for aspect ratio verification
        display_x_offset = (self.viewport_width - target_width) // 2
        display_y_offset = (self.viewport_height - target_height) // 2
        
        return {
            "original_size": (self.original_width, self.original_height),
            "viewport_size": (self.viewport_width, self.viewport_height),
            "zoom_level": self.zoom_level,
            "pan_offset": (self.pan_x, self.pan_y),
            "visible_bounds": (self.view_left, self.view_top, self.view_right, self.view_bottom),
            "visible_size": (visible_width, visible_height),
            "visible_aspect_ratio": visible_aspect,
            "display_bounds": (display_x_offset, display_y_offset, target_width, target_height),
            "scale_factor": scale,
            "quality_ratio": quality_ratio,
            "aspect_ratio_preserved": True,
            "lossless_quality": lossless_quality
        }
    
    def annotator_to_original_coords(self, annotator_x: float, annotator_y: float) -> Tuple[float, float]:
        """
        Convert image_annotator coordinates to original mammography image coordinates.
        
        CRITICAL for maintaining lossless data accuracy:
        This function ensures that any annotation made in the image_annotator component
        maps correctly to the original high-resolution mammography image without any
        loss of precision.
        
        Args:
            annotator_x: X coordinate from image_annotator (0 to viewport_width)
            annotator_y: Y coordinate from image_annotator (0 to viewport_height) 
            
        Returns:
            Tuple[float, float]: (original_x, original_y) in original image coordinates
        """
        # The image_annotator coordinates are identical to viewport coordinates
        # since we ensure the viewport dimensions match the annotator exactly
        return self.viewport_to_original_x(annotator_x), self.viewport_to_original_y(annotator_y)
    
    def original_to_annotator_coords(self, original_x: float, original_y: float) -> Tuple[float, float]:
        """
        Convert original mammography image coordinates to image_annotator coordinates.
        
        CRITICAL for maintaining lossless data accuracy:
        This function ensures that any data from the original high-resolution image
        displays correctly in the image_annotator component.
        
        Args:
            original_x: X coordinate in original image
            original_y: Y coordinate in original image
            
        Returns:
            Tuple[float, float]: (annotator_x, annotator_y) for image_annotator component
        """
        # The annotator coordinates are identical to viewport coordinates
        # since we ensure the viewport dimensions match the annotator exactly
        return self.original_to_viewport_x(original_x), self.original_to_viewport_y(original_y)
    
    def transform_annotation_box(self, box: dict) -> dict:
        """
        Transform an annotation box from image_annotator format to original image coordinates.
        
        Args:
            box: Annotation box from image_annotator with keys like 'xmin', 'ymin', 'xmax', 'ymax'
            
        Returns:
            dict: Transformed box in original image coordinates
        """
        if 'xmin' in box and 'ymin' in box and 'xmax' in box and 'ymax' in box:
            # Transform bounding box coordinates
            orig_xmin, orig_ymin = self.annotator_to_original_coords(box['xmin'], box['ymin'])
            orig_xmax, orig_ymax = self.annotator_to_original_coords(box['xmax'], box['ymax'])
            
            transformed_box = box.copy()
            transformed_box.update({
                'xmin': orig_xmin,
                'ymin': orig_ymin, 
                'xmax': orig_xmax,
                'ymax': orig_ymax
            })
            return transformed_box
        elif 'points' in box:
            # Transform polygon points
            transformed_points = []
            for point in box['points']:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    orig_x, orig_y = self.annotator_to_original_coords(point[0], point[1])
                    transformed_points.append([orig_x, orig_y])
            
            transformed_box = box.copy()
            transformed_box['points'] = transformed_points
            return transformed_box
        else:
            logger.warning(f"Unknown annotation box format: {box}")
            return box
    
    def transform_annotator_data(self, annotator_value: dict) -> dict:
        """
        Transform complete image_annotator data to original image coordinates.
        
        Args:
            annotator_value: Complete value from image_annotator component
                           Format: {"image": np.ndarray, "boxes": list, "orientation": int}
            
        Returns:
            dict: Transformed data with boxes in original image coordinates
        """
        if not annotator_value or 'boxes' not in annotator_value:
            return annotator_value
            
        transformed_value = annotator_value.copy()
        transformed_boxes = []
        
        for box in annotator_value['boxes']:
            transformed_box = self.transform_annotation_box(box)
            transformed_boxes.append(transformed_box)
            
        transformed_value['boxes'] = transformed_boxes
        
        logger.debug(f"Transformed {len(transformed_boxes)} annotation boxes to original coordinates")
        return transformed_value


def create_mammography_viewport(image: np.ndarray, viewport_width: int = 1200, viewport_height: int = 600) -> ImageViewport:
    """
    Create a professional viewport for mammography images.
    IMPORTANT: viewport_height=600 matches the image_annotator component height
    to ensure perfect coordinate mapping without any scaling issues.
    
    Args:
        image: Original high-resolution mammography image
        viewport_width: Fixed viewport width (matches image_annotator width=1200)
        viewport_height: Fixed viewport height (matches image_annotator height=600)
        
    Returns:
        ImageViewport: Configured viewport system
    """
    viewport = ImageViewport(viewport_width, viewport_height)
    viewport.set_image(image)
    return viewport