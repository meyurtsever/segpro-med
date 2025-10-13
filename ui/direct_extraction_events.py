"""
UI Event Handlers for Direct Pixel Extractor Integration

This module provides the missing link between image_annotator UI events 
and the DirectPixelExtractor for lossless mammography viewing.
"""

import logging
from typing import Optional, Any
import gradio as gr

logger = logging.getLogger(__name__)


class DirectExtractionEventHandlers:
    """
    Handle UI events for DirectPixelExtractor integration.
    
    Connects image_annotator mouse/scroll events to the direct pixel extraction system
    for lossless mammography viewing with zoom/pan controls.
    """
    
    def __init__(self, state):
        self.state = state
        
    def handle_zoom_event(self, zoom_delta: float, center_x: int, center_y: int):
        """
        Handle zoom event from image_annotator.
        
        Args:
            zoom_delta: Zoom factor change (>1 = zoom in, <1 = zoom out)
            center_x: X coordinate where user zoomed
            center_y: Y coordinate where user zoomed
        
        Returns:
            tuple: (annotated_value, zoom_status) for image_display and zoom_status outputs
        """
        if (self.state.use_viewport_system and 
            hasattr(self.state, 'direct_pixel_extractor') and 
            self.state.direct_pixel_extractor is not None):
            
            try:
                # Apply zoom to direct pixel extractor
                self.state.direct_pixel_extractor.zoom_at_point(zoom_delta, center_x, center_y)
                
                # Get updated display image
                updated_image = self.state.direct_pixel_extractor.get_display_image()
                
                # Return updated image for image_annotator
                annotated_value = {
                    "image": updated_image,
                    "boxes": [],  # Preserve existing annotations if any
                    "orientation": 0
                }
                
                # Create zoom status message
                current_zoom = self.state.direct_pixel_extractor.zoom_factor
                zoom_status = f"Zoom: {current_zoom:.2f}x"
                
                logger.debug(f"Zoom applied: {zoom_delta:.2f}x at ({center_x}, {center_y}) → {current_zoom:.2f}x")
                return annotated_value, zoom_status
                
            except Exception as e:
                logger.error(f"Error handling zoom event: {e}")
                return None, f"Zoom Error: {str(e)}"
        
        # Return current state if no extractor
        return None, "No Direct Pixel Extractor active"
    
    def handle_pan_event(self, delta_x: int, delta_y: int):
        """
        Handle pan/drag event from image_annotator.
        
        Args:
            delta_x: Horizontal movement in pixels
            delta_y: Vertical movement in pixels
        """
        if (self.state.use_viewport_system and 
            hasattr(self.state, 'direct_pixel_extractor') and 
            self.state.direct_pixel_extractor is not None):
            
            try:
                # Apply pan to direct pixel extractor
                self.state.direct_pixel_extractor.pan(delta_x, delta_y)
                
                # Get updated display image
                updated_image = self.state.direct_pixel_extractor.get_display_image()
                
                # Return updated image for image_annotator
                annotated_value = {
                    "image": updated_image,
                    "boxes": [],  # Preserve existing annotations if any
                    "orientation": 0
                }
                
                logger.debug(f"Pan applied: ({delta_x}, {delta_y})")
                return annotated_value
                
            except Exception as e:
                logger.error(f"Error handling pan event: {e}")
        
        # Return current state if no extractor
        return None
    
    def handle_reset_view(self):
        """
        Handle reset view event (show complete image).
        
        Returns:
            tuple: (annotated_value, zoom_status) for image_display and zoom_status outputs
        """
        if (self.state.use_viewport_system and 
            hasattr(self.state, 'direct_pixel_extractor') and 
            self.state.direct_pixel_extractor is not None):
            
            try:
                # Reset view to show complete image
                self.state.direct_pixel_extractor.reset_view()
                
                # Get updated display image
                updated_image = self.state.direct_pixel_extractor.get_display_image()
                
                # Return updated image for image_annotator
                annotated_value = {
                    "image": updated_image,
                    "boxes": [],  # Preserve existing annotations if any
                    "orientation": 0
                }
                
                # Create zoom status message
                current_zoom = self.state.direct_pixel_extractor.zoom_factor
                zoom_status = f"Zoom: {current_zoom:.2f}x"
                
                logger.info(f"View reset to show complete mammography image - zoom: {current_zoom:.2f}x")
                return annotated_value, zoom_status
                
            except Exception as e:
                logger.error(f"Error handling reset view: {e}")
                return None, f"Reset Error: {str(e)}"
        
        # Return current state if no extractor
        return None, "No Direct Pixel Extractor active"


def create_zoom_handler(state):
    """
    Create a zoom event handler function for image_annotator.
    
    This can be connected to mouse wheel events or zoom buttons.
    """
    handlers = DirectExtractionEventHandlers(state)
    
    def zoom_in_handler():
        return handlers.handle_zoom_event(1.5, 600, 300)  # Zoom in at center
    
    def zoom_out_handler():
        return handlers.handle_zoom_event(0.67, 600, 300)  # Zoom out at center
    
    return zoom_in_handler, zoom_out_handler


def create_reset_handler(state):
    """
    Create a reset view handler for image_annotator.
    """
    handlers = DirectExtractionEventHandlers(state)
    return handlers.handle_reset_view