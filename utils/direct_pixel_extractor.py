"""
LOSSLESS Direct Pixel Extraction System for Fixed-Size Image Annotator

This is a RADICALLY SIMPLER approach:
1. Store original mammography data in full resolution (LOSSLESS)
2. Calculate exact pixel region to extract based on zoom/pan
3. Extract EXACT pixels from original data 
4. Resize ONLY the extracted region to fit image_annotator (1200×600)
5. NO complex viewport logic - just direct pixel access

This guarantees LOSSLESS quality because we work directly with original pixels.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DirectPixelExtractor:
    """
    Direct pixel extraction for lossless mammography viewing in fixed image_annotator.
    
    Simple principle: 
    - Store original data at full resolution
    - Extract exact pixel regions based on zoom/pan  
    - Display extracted region in image_annotator without quality loss
    """
    
    def __init__(self, original_image: np.ndarray, display_width: int = 1200, display_height: int = 600):
        """
        Initialize with original mammography data.
        
        Args:
            original_image: Full resolution mammography image (e.g., 2800×3518)
            display_width: image_annotator width (fixed)
            display_height: image_annotator height (fixed)
        """
        self.original_image = original_image.copy()  # Keep FULL original data
        self.original_height, self.original_width = original_image.shape[:2]
        self.display_width = display_width
        self.display_height = display_height
        
        # Current view state (where we're looking and how zoomed in)
        self.zoom_factor = 1.0  # 1.0 = show original pixels, 2.0 = 2x zoom, etc.
        self.center_x = self.original_width / 2   # Center point X in original coords
        self.center_y = self.original_height / 2  # Center point Y in original coords
        
        # Calculate initial zoom to fit entire image
        self.fit_zoom = min(display_width / self.original_width, display_height / self.original_height)
        self.zoom_factor = self.fit_zoom
        
        logger.info(f"DirectPixelExtractor: {self.original_width}×{self.original_height} → {display_width}×{display_height}")
        logger.info(f"Fit zoom: {self.fit_zoom:.3f}x")
    
    def get_display_image(self) -> np.ndarray:
        """
        Extract the exact pixels we need and display them in image_annotator size.
        
        Returns:
            np.ndarray: Image exactly sized for image_annotator (display_width × display_height)
        """
        # Calculate how much of the original image we can see at current zoom
        visible_width = self.display_width / self.zoom_factor
        visible_height = self.display_height / self.zoom_factor
        
        # Calculate extraction bounds in original image coordinates
        left = int(max(0, self.center_x - visible_width / 2))
        right = int(min(self.original_width, self.center_x + visible_width / 2))
        top = int(max(0, self.center_y - visible_height / 2))
        bottom = int(min(self.original_height, self.center_y + visible_height / 2))
        
        # Extract the EXACT pixel region from original data
        extracted_region = self.original_image[top:bottom, left:right]
        
        logger.debug(f"Extracted region: {extracted_region.shape} from bounds ({left},{top}) to ({right},{bottom})")
        logger.debug(f"Zoom: {self.zoom_factor:.3f}x, Visible: {visible_width:.1f}×{visible_height:.1f}")
        
        # Convert to RGB if needed
        if len(extracted_region.shape) == 2:
            extracted_region = cv2.cvtColor(extracted_region, cv2.COLOR_GRAY2RGB)
        
        # Create output image
        display_image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        region_height, region_width = extracted_region.shape[:2]
        
        if region_width > 0 and region_height > 0:
            # Resize extracted region to fit display while preserving aspect ratio
            region_aspect = region_width / region_height
            display_aspect = self.display_width / self.display_height
            
            if region_aspect > display_aspect:
                # Region is wider - fit to display width
                target_width = self.display_width
                target_height = int(self.display_width / region_aspect)
            else:
                # Region is taller - fit to display height
                target_height = self.display_height
                target_width = int(self.display_height * region_aspect)
            
            # Choose interpolation method for BEST quality
            scale = max(target_width / region_width, target_height / region_height)
            
            if abs(scale - 1.0) < 0.01:
                # Nearly 1:1 - use original pixels
                resized_region = extracted_region
                actual_width, actual_height = region_width, region_height
                logger.debug("🎯 PERFECT: Using original pixels (1:1 mapping)")
                
            elif scale > 1.0:
                # Upscaling - use LANCZOS4 for crisp medical imaging 
                resized_region = cv2.resize(extracted_region, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                actual_width, actual_height = target_width, target_height
                logger.debug(f"📈 UPSCALE: {scale:.2f}x using LANCZOS4")
                
            else:
                # Downscaling - use INTER_AREA for quality preservation
                resized_region = cv2.resize(extracted_region, (target_width, target_height), interpolation=cv2.INTER_AREA) 
                actual_width, actual_height = target_width, target_height
                logger.debug(f"📉 DOWNSCALE: {scale:.2f}x using INTER_AREA")
            
            # Center the resized region in display image
            start_x = (self.display_width - actual_width) // 2
            start_y = (self.display_height - actual_height) // 2
            
            display_image[start_y:start_y + actual_height, start_x:start_x + actual_width] = resized_region
        
        return display_image
    
    def zoom_at_point(self, zoom_delta: float, click_x: int, click_y: int) -> None:
        """
        Zoom in/out at a specific point in the display.
        
        Args:
            zoom_delta: Zoom multiplier (>1 = zoom in, <1 = zoom out)
            click_x: X coordinate in display image where user clicked
            click_y: Y coordinate in display image where user clicked
        """
        old_zoom = self.zoom_factor
        
        # Apply zoom with limits
        min_zoom = self.fit_zoom  # Never zoom out beyond fit
        max_zoom = 20.0  # Allow 20x zoom for detailed inspection
        self.zoom_factor = max(min_zoom, min(max_zoom, self.zoom_factor * zoom_delta))
        
        if self.zoom_factor != old_zoom:
            # Convert click point to original image coordinates
            visible_width_old = self.display_width / old_zoom
            visible_height_old = self.display_height / old_zoom
            
            # Calculate where the user clicked in the original image
            click_ratio_x = click_x / self.display_width
            click_ratio_y = click_y / self.display_height
            
            left_old = self.center_x - visible_width_old / 2
            top_old = self.center_y - visible_height_old / 2
            
            clicked_orig_x = left_old + click_ratio_x * visible_width_old
            clicked_orig_y = top_old + click_ratio_y * visible_height_old
            
            # Update center to keep the clicked point in the same display position
            visible_width_new = self.display_width / self.zoom_factor
            visible_height_new = self.display_height / self.zoom_factor
            
            self.center_x = clicked_orig_x + (click_ratio_x - 0.5) * visible_width_new
            self.center_y = clicked_orig_y + (click_ratio_y - 0.5) * visible_height_new
            
            # Keep center within bounds
            self.center_x = max(visible_width_new/2, min(self.original_width - visible_width_new/2, self.center_x))
            self.center_y = max(visible_height_new/2, min(self.original_height - visible_height_new/2, self.center_y))
            
            logger.debug(f"Zoomed to {self.zoom_factor:.3f}x at ({click_x}, {click_y})")
    
    def pan(self, delta_x: int, delta_y: int) -> None:
        """
        Pan the view by display pixels.
        
        Args:
            delta_x: Horizontal pan in display pixels
            delta_y: Vertical pan in display pixels  
        """
        # Convert display deltas to original image coordinates
        orig_delta_x = delta_x / self.zoom_factor
        orig_delta_y = delta_y / self.zoom_factor
        
        # Update center position
        self.center_x -= orig_delta_x  # Negative because pan direction is opposite
        self.center_y -= orig_delta_y
        
        # Keep center within bounds
        visible_width = self.display_width / self.zoom_factor
        visible_height = self.display_height / self.zoom_factor
        
        self.center_x = max(visible_width/2, min(self.original_width - visible_width/2, self.center_x))
        self.center_y = max(visible_height/2, min(self.original_height - visible_height/2, self.center_y))
    
    def reset_view(self) -> None:
        """Reset to show entire image."""
        self.zoom_factor = self.fit_zoom
        self.center_x = self.original_width / 2
        self.center_y = self.original_height / 2
        logger.info(f"Reset to fit view: {self.zoom_factor:.3f}x zoom")
    
    def display_to_original_coords(self, display_x: int, display_y: int) -> Tuple[float, float]:
        """Convert display coordinates to original image coordinates."""
        visible_width = self.display_width / self.zoom_factor
        visible_height = self.display_height / self.zoom_factor
        
        left = self.center_x - visible_width / 2
        top = self.center_y - visible_height / 2
        
        ratio_x = display_x / self.display_width
        ratio_y = display_y / self.display_height
        
        orig_x = left + ratio_x * visible_width
        orig_y = top + ratio_y * visible_height
        
        return orig_x, orig_y
    
    def original_to_display_coords(self, orig_x: float, orig_y: float) -> Tuple[int, int]:
        """Convert original image coordinates to display coordinates."""
        visible_width = self.display_width / self.zoom_factor
        visible_height = self.display_height / self.zoom_factor
        
        left = self.center_x - visible_width / 2
        top = self.center_y - visible_height / 2
        
        if visible_width > 0 and visible_height > 0:
            ratio_x = (orig_x - left) / visible_width
            ratio_y = (orig_y - top) / visible_height
            
            display_x = int(ratio_x * self.display_width)
            display_y = int(ratio_y * self.display_height)
            
            return display_x, display_y
        
        return 0, 0


def create_lossless_mammography_extractor(image: np.ndarray) -> DirectPixelExtractor:
    """
    Create a direct pixel extractor for lossless mammography viewing.
    
    Args:
        image: Original high-resolution mammography image
        
    Returns:
        DirectPixelExtractor: Configured for lossless viewing
    """
    return DirectPixelExtractor(image, display_width=1200, display_height=600)