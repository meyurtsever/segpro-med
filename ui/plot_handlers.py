"""
SegMed-Pro Plot Handlers

This module contains event handlers specifically for gr.Plot components,
used in the viewer tab for annotation and segmentation functionality.
"""

import os
import logging
import gradio as gr
import numpy as np
from typing import Optional

from utils.dicom_utils import get_dicom_metadata
from utils.visualization import (display_slice, make_slice_figure, overlay_segmentation)
from utils.debug_utils import log_exception
from ui.state import AppState

logger = logging.getLogger(__name__)


class PlotViewerHandlers:
    """Handlers for viewer operations using gr.Plot component"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    @log_exception
    def update_slice_for_plot(self, slider_value, view_type):
        """Update the displayed slice for gr.Plot component (returns plotly figure)"""
        if self.state.current_data is None:
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), "0/0", "x: 0, y: 0, z: 0", {}, None, None
        
        # Update slice and view based on input
        self.state.current_slice_idx = int(slider_value)
        self.state.current_view = view_type.lower()
          # Generate the slice image
        try:
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                crosshair=None,
                add_orientation_marker=False
            )
              # Apply segmentation overlay if segmentation is loaded AND viewer tab should show it
            if (self.state.segmentation_loaded and self.state.segmentation_data is not None 
                and self.state.show_segmentation_in_viewer):
                try:
                    seg_slice = self.state.get_segmentation_slice(
                        self.state.current_view,
                        self.state.current_slice_idx
                    )
                    
                    if seg_slice is not None:
                        img = overlay_segmentation(
                            img, 
                            seg_slice, 
                            alpha=self.state.segmentation_alpha,
                            colormap=self.state.segmentation_colormap
                        )
                except Exception as e:
                    logger.error(f"Error applying segmentation overlay: {str(e)}")
            
            # Create plotly figure for gr.Plot
            fig = make_slice_figure(img, dragmode='pan')
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            # Update metadata
            metadata = {}
            if self.state.current_metadata:
                metadata.update(self.state.current_metadata)
                metadata.update({
                    'CurrentSlice': self.state.current_slice_idx + 1,
                    'TotalSlices': len(self.state.file_list) if self.state.file_list else 1,
                    'CurrentView': self.state.current_view
                })
            
            window_center = metadata.get('WindowCenter', 500)
            window_width = metadata.get('WindowWidth', 1000)
            if isinstance(window_center, list):
                window_center = window_center[0]
            if isinstance(window_width, list):
                window_width = window_width[0]
            return (fig, f"{self.state.current_slice_idx -1}/{self.state.get_max_slice_for_view(self.state.current_view) - 1}", 
                   crosshair_text, metadata, window_center, window_width)
        
        except Exception as e:
            logger.error(f"Error in update_slice_for_plot: {str(e)}")
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), "Error", f"Error: {str(e)}", {}, 500, 1000
    @log_exception  
    def change_view_for_plot(self, view):
        """Change the viewing orientation for gr.Plot component (returns plotly figure)"""
        if self.state.current_data is None:
            return gr.Slider(), "0/0", make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # Update current view
        self.state.current_view = view.lower()
        
        # Determine slice count and range based on view
        slider_min = 0
        slider_max = self.state.get_max_slice_for_view(self.state.current_view)
        
        if self.state.current_view == "axial":
            self.state.current_slice_idx = self.state.crosshair_position[2]
        elif self.state.current_view == "sagittal":
            self.state.current_slice_idx = self.state.crosshair_position[0]
        elif self.state.current_view == "coronal":
            self.state.current_slice_idx = self.state.crosshair_position[1]
          # Generate the slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # Apply segmentation overlay if loaded AND viewer tab should show it
        if (self.state.segmentation_loaded and self.state.segmentation_data is not None
            and self.state.show_segmentation_in_viewer):
            try:
                seg_slice = self.state.get_segmentation_slice(
                    self.state.current_view, 
                    self.state.current_slice_idx
                )
                
                if seg_slice is not None:
                    img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.state.segmentation_alpha,
                        colormap=self.state.segmentation_colormap
                    )
            except Exception as e:
                logger.error(f"Error applying segmentation overlay: {str(e)}")
        
        # Create plotly figure for gr.Plot
        fig = make_slice_figure(img, dragmode='pan')
        
        return (            gr.Slider(minimum=slider_min, maximum=slider_max, value=self.state.current_slice_idx), 
            f"{self.state.current_slice_idx}/{slider_max}", 
            fig
        )
    
    @log_exception
    def update_window_level_for_plot(self, level, width):
        """Apply window/level adjustments to the current image for gr.Plot component"""
        if self.state.current_data is None:
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8))
        
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=level,
            window_width=width,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # Apply segmentation overlay if loaded AND viewer tab should show it
        if (self.state.segmentation_loaded and self.state.segmentation_data is not None
            and self.state.show_segmentation_in_viewer):
            try:
                seg_slice = self.state.get_segmentation_slice(
                    self.state.current_view, 
                    self.state.current_slice_idx
                )
                
                if seg_slice is not None:
                    img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.state.segmentation_alpha,
                        colormap=self.state.segmentation_colormap
                    )
            except Exception as e:
                logger.error(f"Error applying segmentation overlay: {str(e)}")
        
        # Create plotly figure for gr.Plot
        fig = make_slice_figure(img, dragmode='pan')
        return fig
    
    @log_exception
    def select_file_from_browser_for_plot(self, selected_file):
        """When a file is selected from the browser dropdown for gr.Plot component"""
        if not selected_file or not self.state.file_list:
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), "0/0", "x: 0, y: 0, z: 0", {}, 0, None, None
        
        try:
            # Find the selected file in the file list
            selected_path = None
            for file_path in self.state.file_list:
                if os.path.basename(file_path) == selected_file:
                    selected_path = file_path
                    break
            
            if not selected_path:
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), "0/0", "x: 0, y: 0, z: 0", {"error": "File not found"}, 0, None, None
            
            # Load the selected DICOM file
            slice_idx = self.state.file_list.index(selected_path)
            self.state.current_slice_idx = slice_idx
            
            # Update crosshair z position
            x, y, _ = self.state.crosshair_position
            self.state.crosshair_position = (x, y, slice_idx)
            
            # Get metadata for this slice
            metadata = get_dicom_metadata(selected_path)
            
            # Extract window values
            window_center = None
            window_width = None
            
            if metadata and 'WindowCenter' in metadata:
                window_center = metadata['WindowCenter']
                if isinstance(window_center, list):
                    window_center = window_center[0]
            
            if metadata and 'WindowWidth' in metadata:
                window_width = metadata['WindowWidth']
                if isinstance(window_width, list):
                    window_width = window_width[0]            # Generate image at the new index
            img = display_slice(
                self.state.current_data, 
                slice_idx, 
                "axial",
                window_level=window_center,
                window_width=window_width,
                crosshair=None,
                add_orientation_marker=False
            )
            
            # Apply segmentation overlay if loaded AND viewer tab should show it
            if (self.state.segmentation_loaded and self.state.segmentation_data is not None
                and self.state.show_segmentation_in_viewer):
                try:
                    seg_slice = self.state.segmentation_data[slice_idx, :, :]
                    img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.state.segmentation_alpha,
                        colormap=self.state.segmentation_colormap
                    )
                except Exception as e:
                    logger.error(f"Error applying segmentation overlay: {str(e)}")
            
            # Create plotly figure for gr.Plot
            fig = make_slice_figure(img, dragmode='pan')
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            return (fig, f"{slice_idx}/{len(self.state.file_list)-1}", crosshair_text, 
                   metadata or {}, gr.Slider(value=slice_idx), window_center or 500, window_width or 1000)
            
        except Exception as e:
            logger.error(f"Error selecting file from browser: {str(e)}")
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), "Error", f"Error: {str(e)}", {}, 0, None, None

    def prev_slice(self, slider_value):
        """Go to previous slice by decrementing slider value"""
        current_value = int(slider_value)
        return max(0, current_value - 1)

    def next_slice(self, slider_value):
        """Go to next slice by incrementing slider value"""
        if self.state.current_data is None:
            return slider_value
        
        current_value = int(slider_value)
        max_value = self.state.get_max_slice_for_view(self.state.current_view)
        return min(current_value + 1, max_value)


class PlotToolHandlers:
    """Handlers for plot tool operations using gr.Plot component"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    def update_plot_tool_for_plot(self, tool_name):
        """Update plot tool for gr.Plot component (returns plotly figure with updated dragmode)"""
        if self.state.current_data is None:
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # Store the selected tool for reference
        self.state.current_plot_tool = tool_name        # Generate the current slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # Apply segmentation overlay if loaded AND viewer tab should show it
        if (self.state.segmentation_loaded and self.state.segmentation_data is not None
            and self.state.show_segmentation_in_viewer):
            try:
                seg_slice = self.state.get_segmentation_slice(
                    self.state.current_view, 
                    self.state.current_slice_idx
                )
                
                if seg_slice is not None:
                    img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.state.segmentation_alpha,
                        colormap=self.state.segmentation_colormap
                    )
            except Exception as e:
                logger.error(f"Error applying segmentation in plot tool: {str(e)}")
        
        # Map tool names to plotly dragmode
        tool_mapping = {
            "draw_circle": "drawcircle",
            "draw_rect": "drawrect", 
            "draw_line": "drawline",
            "draw_openpath": "drawopenpath",
            "draw_closedpath": "drawclosedpath",
            "erase_shape": "eraseshape",
            "pan": "pan",
            "zoom": "zoom",
            "reset": "pan"  # Reset just goes back to pan mode
        }
        
        dragmode = tool_mapping.get(tool_name, "pan")
        
        # Create plotly figure for gr.Plot with appropriate dragmode
        fig = make_slice_figure(img, dragmode=dragmode)
        return fig
