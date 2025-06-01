"""
PlotlyBridge handler for the SegMed-Pro application.
Handles the communication between Plotly hover events and Python.
"""

import logging
import json
from typing import Optional, Dict, Any
import numpy as np

from ui.state import AppState
from utils.visualization import display_slice, make_slice_figure, overlay_segmentation

logger = logging.getLogger(__name__)

class PlotlyBridgeHandler:
    """Handler for the PlotlyBridge component"""
    
    def __init__(self, state: AppState):
        self.state = state
        logger.info("PlotlyBridgeHandler initialized")
        print("PlotlyBridgeHandler initialized - Ready to receive events")
        
    def handle_hover_event(self, hover_data_str: str) -> Optional[Dict[str, Any]]:
        """
        Handle hover event data from the PlotlyBridge component.
        Updates the crosshair position based on the hover coordinates.
        
        Args:
            hover_data_str: JSON string containing hover data
            
        Returns:
            Optional[Dict[str, Any]]: Updated crosshair info or None if invalid data
        """
        print(f"PlotlyBridgeHandler: Received hover data string: {hover_data_str}")  # Console print for immediate feedback
        logger.info(f"PlotlyBridgeHandler: Received hover data string: {hover_data_str}")
        
        if not hover_data_str or self.state.current_data is None:
            logger.warning("PlotlyBridgeHandler: Empty hover data or no current data loaded")
            return None
            
        try:
            # Parse the hover data
            hover_data = json.loads(hover_data_str)
            
            # Extract coordinates
            x = int(hover_data.get('x', 0))
            y = int(hover_data.get('y', 0))
            
            logger.info(f"PlotlyBridgeHandler: Processed hover event at coordinates: ({x}, {y})")
            
            # Map 2D coordinates to 3D based on current view
            if self.state.current_view == 'axial':
                # Update x, y coordinates, keep z (slice) the same
                self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            elif self.state.current_view == 'sagittal':
                # Update y, z coordinates, keep x (slice) the same
                self.state.crosshair_position = (self.state.current_slice_idx, y, x)
            elif self.state.current_view == 'coronal':
                # Update x, z coordinates, keep y (slice) the same
                self.state.crosshair_position = (x, self.state.current_slice_idx, y)
            
            # Format crosshair text
            crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, z: {self.state.crosshair_position[2]}"
            
            logger.info(f"PlotlyBridgeHandler: Updated crosshair position to {self.state.crosshair_position}")
            
            return crosshair_text
            
        except Exception as e:
            logger.error(f"PlotlyBridgeHandler: Error handling hover event: {str(e)}", exc_info=True)
            return None
            
    def handle_click_event(self, click_data_str: str) -> Optional[Dict[str, Any]]:
        """
        Handle click event data from the PlotlyBridge component.
        Updates the crosshair position based on the click coordinates.
        
        Args:
            click_data_str: JSON string containing click data
            
        Returns:
            Optional[Dict[str, Any]]: Updated crosshair info or None if invalid data
        """
        print(f"PlotlyBridgeHandler: Received click data string: {click_data_str}")  # Console print for immediate feedback
        logger.info(f"PlotlyBridgeHandler: Received click data string: {click_data_str}")
        
        if not click_data_str or self.state.current_data is None:
            logger.warning("PlotlyBridgeHandler: Empty click data or no current data loaded")
            return None
            
        try:
            # Parse the click data
            click_data = json.loads(click_data_str)
            
            # Extract coordinates
            x = int(click_data.get('x', 0))
            y = int(click_data.get('y', 0))
            
            logger.info(f"PlotlyBridgeHandler: Processed click event at coordinates: ({x}, {y})")
            
            # Map 2D coordinates to 3D based on current view
            if self.state.current_view == 'axial':
                # Update x, y coordinates, keep z (slice) the same
                self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            elif self.state.current_view == 'sagittal':
                # Update y, z coordinates, keep x (slice) the same
                self.state.crosshair_position = (self.state.current_slice_idx, y, x)
            elif self.state.current_view == 'coronal':
                # Update x, z coordinates, keep y (slice) the same
                self.state.crosshair_position = (x, self.state.current_slice_idx, y)
            
            # Format crosshair text
            crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, z: {self.state.crosshair_position[2]}"
            
            logger.info(f"PlotlyBridgeHandler: Updated crosshair position to {self.state.crosshair_position}")
            
            return crosshair_text
            
        except Exception as e:
            logger.error(f"PlotlyBridgeHandler: Error handling click event: {str(e)}", exc_info=True)
            return None
            
    def update_crosshair_display(self) -> Optional[Dict[str, Any]]:
        """
        Update the image plot with the current crosshair position.
        
        Returns:
            Optional[Dict[str, Any]]: Updated image figure or None if error
        """
        print(f"Updating crosshair display at position: {self.state.crosshair_position}")
        logger.info(f"Updating crosshair display at position: {self.state.crosshair_position}")
        
        if self.state.current_data is None:
            logger.warning("Cannot update crosshair display: No data loaded")
            return None
        
        try:
            # Get current figure parameters
            current_view = self.state.current_view
            slice_idx = self.state.current_slice_idx
            
            # Create figure with crosshair overlay
            if hasattr(self.state, 'current_segmentation') and self.state.current_segmentation is not None:
                # If there's a segmentation, overlay it
                fig = overlay_segmentation(
                    self.state.current_data, 
                    self.state.current_segmentation,
                    view=current_view, 
                    slice_idx=slice_idx,
                    crosshair_pos=self.state.crosshair_position
                )
            else:
                # Otherwise, just show the image with crosshair
                fig = make_slice_figure(
                    self.state.current_data,
                    view=current_view, 
                    slice_idx=slice_idx,
                    crosshair_pos=self.state.crosshair_position
                )
            
            logger.info(f"Created updated figure with crosshair at {self.state.crosshair_position}")
            return fig
            
        except Exception as e:
            logger.error(f"Error updating crosshair display: {str(e)}", exc_info=True)
            return None