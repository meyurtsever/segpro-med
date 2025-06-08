"""
SegMed-Pro Image Handlers

This module contains event handlers specifically for gr.Image components,
used in the editor tab for MEDSAM2 AI annotation workflow.
"""

import os
import logging
import gradio as gr
import numpy as np
from typing import Optional

from utils.dicom_utils import get_dicom_metadata
from utils.visualization import (display_slice, overlay_segmentation, make_image_for_gradio)
from utils.debug_utils import log_exception
from ui.state import AppState

logger = logging.getLogger(__name__)


class ImageViewerHandlers:
    """Handlers for viewer operations using gr.Image component"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    @log_exception
    def update_slice(self, slider_value, view_type):
        """Update the displayed slice based on slider value and view"""
        if self.state.current_data is None:
            return None, "0/0", "x: 0, y: 0, z: 0", {}, None, None
        
        logger.info(f"Updating slice: value={slider_value}, view={view_type}")
        
        # Update the current slice index and view
        self.state.update_slice_position(int(slider_value), view_type)
        
        # Determine slice count based on view
        total_slices = self.state.get_max_slice_for_view(self.state.current_view) + 1
        
        # Initialize window center and width values
        window_center = None
        window_width = None
        metadata = self.state.current_metadata
        
        # Get per-slice metadata if available (for axial DICOM views)
        if (self.state.current_view == "axial" and 
            self.state.current_data_type == "dicom" and 
            self.state.file_list):
            if 0 <= self.state.current_slice_idx < len(self.state.file_list):
                try:
                    current_file = self.state.file_list[self.state.current_slice_idx]
                    logger.info(f"Loading metadata for file: {current_file}")
                    
                    # Get fresh metadata for the current slice
                    metadata = get_dicom_metadata(current_file)
                    
                    # Add series information to metadata
                    metadata['SeriesInfo'] = {
                        'CurrentSlice': self.state.current_slice_idx + 1,
                        'TotalSlices': len(self.state.file_list),
                        'CurrentFile': os.path.basename(current_file)
                    }
                    # Update the current metadata
                    self.state.current_metadata = metadata
                    logger.info(f"Updated metadata for slice {self.state.current_slice_idx}")
                except Exception as e:
                    logger.error(f"Error updating metadata: {str(e)}")
        
        # Extract WindowCenter and WindowWidth from metadata if available
        if metadata and 'WindowCenter' in metadata:
            window_center = metadata['WindowCenter']
            if isinstance(window_center, list):
                window_center = window_center[0]
            logger.info(f"Using WindowCenter from metadata: {window_center}")
        
        if metadata and 'WindowWidth' in metadata:
            window_width = metadata['WindowWidth']
            if isinstance(window_width, list):
                window_width = window_width[0]
            logger.info(f"Using WindowWidth from metadata: {window_width}")
        
        # Generate the slice image
        try:
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                window_level=window_center,
                window_width=window_width,
                crosshair=self.state.crosshair_position
            )
              # Apply segmentation overlay if segmentation is loaded AND editor tab should show it
            if (self.state.segmentation_loaded and self.state.segmentation_data is not None
                and self.state.show_segmentation_in_editor):
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
                        logger.info(f"Applied segmentation overlay to slice {self.state.current_slice_idx}")
                except Exception as e:
                    logger.error(f"Error applying segmentation overlay: {str(e)}")
            
            # Convert to PIL Image for gr.Image
            pil_image = make_image_for_gradio(img)
            logger.info(f"Generated slice image with shape {img.shape}")
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            # Default values for window center and width if not found
            if window_center is None:
                window_center = 500
            if window_width is None:
                window_width = 1000
            
            return pil_image, f"{self.state.current_slice_idx}/{total_slices-1}", crosshair_text, metadata, window_center, window_width
        except Exception as e:
            logger.error(f"Error generating slice image: {str(e)}")
            return None, f"Error: {str(e)}", "x: 0, y: 0, z: 0", {}, None, None
    
    def change_view(self, view):
        """Change the viewing orientation"""
        if self.state.current_data is None:
            return gr.Slider(), "0/0", None
        
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
            crosshair=self.state.crosshair_position
        )
        
        # Apply segmentation overlay if loaded AND editor tab should show it
        if (self.state.segmentation_loaded and self.state.segmentation_data is not None
            and self.state.show_segmentation_in_editor):
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
        
        # Convert to PIL Image for gr.Image
        pil_image = make_image_for_gradio(img)
        
        return (
            gr.Slider(minimum=slider_min, maximum=slider_max, value=self.state.current_slice_idx), 
            f"{self.state.current_slice_idx}/{slider_max}", 
            pil_image
        )
    
    def update_window_level(self, level, width):
        """Apply window/level adjustments to the current image"""
        if self.state.current_data is None:
            return None
          # Generate the slice image with window/level adjustments
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=level,
            window_width=width,
            crosshair=self.state.crosshair_position
        )
        
        # Apply segmentation overlay if loaded AND editor tab should show it
        if (self.state.segmentation_loaded and self.state.segmentation_data is not None
            and self.state.show_segmentation_in_editor):
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
        
        # Convert to PIL Image for gr.Image
        pil_image = make_image_for_gradio(img)
        return pil_image
    
    def select_file_from_browser(self, selected_file):
        """When a file is selected from the browser dropdown"""
        if not selected_file or not self.state.file_list:
            return None, "0/0", "x: 0, y: 0, z: 0", {}, 0, None, None
        
        try:
            # Find the selected file in the file list
            selected_path = None
            for file_path in self.state.file_list:
                if os.path.basename(file_path) == selected_file:
                    selected_path = file_path
                    break
            
            if not selected_path:
                return None, "0/0", "x: 0, y: 0, z: 0", {"error": "File not found"}, 0, None, None
            
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
                    window_width = window_width[0]
              # Generate image at the new index
            img = display_slice(
                self.state.current_data, 
                slice_idx, 
                "axial",
                window_level=window_center,
                window_width=window_width,
                crosshair=self.state.crosshair_position
            )
            
            # Apply segmentation overlay if loaded AND editor tab should show it
            if (self.state.segmentation_loaded and self.state.segmentation_data is not None
                and self.state.show_segmentation_in_editor):
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
            
            # Convert to PIL Image for gr.Image
            pil_image = make_image_for_gradio(img)
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            return (pil_image, f"{slice_idx}/{len(self.state.file_list)-1}", crosshair_text, 
                   metadata, slice_idx, window_center, window_width)
                   
        except Exception as e:
            return None, "0/0", "x: 0, y: 0, z: 0", {"error": str(e)}, 0, None, None

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


class ImagePlotToolHandlers:
    """Handlers for plot tool operations using gr.Image component (dummy implementation)"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    def update_plot_tool(self, tool_name):
        """Since we're using gr.Image now, this method just returns the current image"""
        if self.state.current_data is None:
            return None
        
        # Store the selected tool for reference (though not used with gr.Image)
        self.state.current_plot_tool = tool_name
        
        # Generate the current slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=self.state.crosshair_position
        )
          # Apply segmentation overlay if loaded AND editor tab should show it
        if (self.state.segmentation_loaded and self.state.segmentation_data is not None
            and self.state.show_segmentation_in_editor):
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
        
        # Convert to PIL Image for gr.Image
        pil_image = make_image_for_gradio(img)
        return pil_image
