# filepath: c:\Users\Yurtsever\Downloads\segpro-med\ui\plotly_bridge_handler_extension.py
"""
Extension for the PlotlyBridgeHandler with improved coordinate handling.
"""
import json
import logging
from typing import Optional, Dict, Any, Tuple

from ui.plotly_bridge_handler import PlotlyBridgeHandler

logger = logging.getLogger(__name__)

class EnhancedPlotlyBridgeHandler(PlotlyBridgeHandler):
    """Enhanced handler for the PlotlyBridge component with improved coordinate handling"""
    
    def handle_hover_event(self, hover_data_str: str) -> str:
        """
        Handle hover event data with improved coordinate formatting.
        
        Args:
            hover_data_str: JSON string containing hover data
            
        Returns:
            str: Formatted coordinate text
        """
        print(f"EnhancedPlotlyBridgeHandler: Received hover data string: {hover_data_str}")
        
        if not hover_data_str or self.state.current_data is None:
            return "No data"
            
        try:
            # Parse the hover data
            hover_data = json.loads(hover_data_str)
            
            # Extract coordinates
            x = int(hover_data.get('x', 0))
            y = int(hover_data.get('y', 0))
            
            # Map 2D coordinates to 3D based on current view
            if self.state.current_view == 'axial':
                self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            elif self.state.current_view == 'sagittal':
                self.state.crosshair_position = (self.state.current_slice_idx, y, x)
            elif self.state.current_view == 'coronal':
                self.state.crosshair_position = (x, self.state.current_slice_idx, y)
            
            # Return the original hover coordinates directly, not the mapped ones
            # This ensures the coordinates display shows what the user is hovering over
            result = f"x: {x}, y: {y}, z: {self.state.current_slice_idx}"
            print(f"EnhancedPlotlyBridgeHandler: Hover result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"EnhancedPlotlyBridgeHandler: Error handling hover event: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    def handle_click_event(self, click_data_str: str) -> str:
        """
        Handle click event data with improved coordinate formatting.
        
        Args:
            click_data_str: JSON string containing click data
            
        Returns:
            str: Formatted coordinate text
        """
        print(f"EnhancedPlotlyBridgeHandler: Received click data string: {click_data_str}")
        
        if not click_data_str or self.state.current_data is None:
            return "No data"
            
        try:
            # Parse the click data
            click_data = json.loads(click_data_str)
            
            # Extract coordinates
            x = int(click_data.get('x', 0))
            y = int(click_data.get('y', 0))
            
            # Map 2D coordinates to 3D based on current view
            if self.state.current_view == 'axial':
                self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            elif self.state.current_view == 'sagittal':
                self.state.crosshair_position = (self.state.current_slice_idx, y, x)
            elif self.state.current_view == 'coronal':
                self.state.crosshair_position = (x, self.state.current_slice_idx, y)
            
            # Return the original click coordinates directly
            result = f"x: {x}, y: {y}, z: {self.state.current_slice_idx}"
            print(f"EnhancedPlotlyBridgeHandler: Click result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"EnhancedPlotlyBridgeHandler: Error handling click event: {str(e)}", exc_info=True)
            return f"Error: {str(e)}"
    
    def get_current_coordinates(self) -> str:
        """
        Get the current coordinates as a formatted string.
        
        Returns:
            str: Formatted X and Y coordinates
        """
        try:
            x, y, _ = self.state.crosshair_position
            return f"X: {x} Y: {y}"
        except:
            return "X: -- Y: --"