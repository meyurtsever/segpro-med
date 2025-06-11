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
    
    def __init__(self, state: AppState, medsam2_handlers=None):
        self.state = state
        self.medsam2_handlers = medsam2_handlers
    
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
            )            # Apply segmentation overlay if segmentation is loaded AND editor tab should show it
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
            
            # Apply MEDSAM2 annotation overlay if available for current slice
            if (self.medsam2_handlers and 
                hasattr(self.medsam2_handlers, 'annotation_overlays') and
                self.state.current_slice_idx in self.medsam2_handlers.annotation_overlays):
                try:
                    overlay_data = self.medsam2_handlers.annotation_overlays[self.state.current_slice_idx]
                    mask_array = overlay_data['mask']
                    
                    img = overlay_segmentation(
                        img,
                        mask_array,
                        alpha=0.4,
                        colormap={1: [255, 0, 0]}  # Red overlay for MEDSAM2 annotations
                    )
                    logger.info(f"Applied MEDSAM2 annotation overlay to slice {self.state.current_slice_idx}")
                except Exception as e:
                    logger.error(f"Error applying MEDSAM2 annotation overlay: {str(e)}")
            
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
            
            return pil_image, f"{self.state.current_slice_idx + 1}/{total_slices}", crosshair_text, metadata, window_center, window_width
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
    
    @log_exception
    def update_slice_for_annotator(self, slider_value, view_type):
        """Update slice for image_annotator component - returns AnnotatedImageValue format"""
        if self.state.current_data is None:
            return None, "0/0", "x: 0, y: 0, z: 0", {}, None, None
        
        logger.info(f"Updating slice for annotator: value={slider_value}, view={view_type}")
        
        # Update the current slice index and view
        self.state.update_slice_position(int(slider_value), view_type)
        
        # Determine slice count based on view
        total_slices = self.state.get_max_slice_for_view(self.state.current_view) + 1
        
        # Get metadata and windowing parameters
        window_center = None
        window_width = None
        metadata = self.state.current_metadata
        
        # Get per-slice metadata if available (for axial DICOM views)
        if (self.state.current_view == "axial" and 
            self.state.current_data_type == "dicom" and 
            self.state.file_list):
            try:
                current_file_idx = min(self.state.current_slice_idx, len(self.state.file_list) - 1)
                file_path = self.state.file_list[current_file_idx]
                per_slice_metadata = get_dicom_metadata(file_path)
                
                if per_slice_metadata:
                    metadata.update(per_slice_metadata)
                    
                    # Get windowing information from per-slice metadata
                    if "window_center" in per_slice_metadata:
                        window_center = per_slice_metadata["window_center"]
                        if isinstance(window_center, list):
                            window_center = window_center[0]
                        logger.info(f"Using WindowCenter from per-slice metadata: {window_center}")
                    
                    if "window_width" in per_slice_metadata:
                        window_width = per_slice_metadata["window_width"]
                        if isinstance(window_width, list):
                            window_width = window_width[0]
                        logger.info(f"Using WindowWidth from per-slice metadata: {window_width}")
            except Exception as e:
                logger.error(f"Error getting per-slice metadata: {str(e)}")
        
        # Generate the slice image as numpy array
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
            
            # Apply MEDSAM2 annotation overlay if available for current slice
            if (self.medsam2_handlers and 
                hasattr(self.medsam2_handlers, 'annotation_overlays') and
                self.state.current_slice_idx in self.medsam2_handlers.annotation_overlays):
                try:
                    overlay_data = self.medsam2_handlers.annotation_overlays[self.state.current_slice_idx]
                    mask_array = overlay_data['mask']
                    
                    img = overlay_segmentation(
                        img,
                        mask_array,
                        alpha=0.4,
                        colormap={1: [255, 0, 0]}  # Red overlay for MEDSAM2 annotations
                    )
                    logger.info(f"Applied MEDSAM2 annotation overlay to slice {self.state.current_slice_idx}")
                except Exception as e:
                    logger.error(f"Error applying MEDSAM2 annotation overlay: {str(e)}")
            
            # Convert to format expected by image_annotator
            # Ensure it's RGB and uint8
            if len(img.shape) == 2:
                # Grayscale to RGB
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            
            # Create AnnotatedImageValue format
            annotated_value = {
                "image": img_rgb,
                "boxes": [],  # Start with empty boxes, will be populated by annotations
                "orientation": 0
            }
            
            logger.info(f"Generated annotated slice image with shape {img_rgb.shape}")
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            # Default values for window center and width if not found
            if window_center is None:
                window_center = 500
            if window_width is None:
                window_width = 1000
            
            return annotated_value, f"{self.state.current_slice_idx + 1}/{total_slices}", crosshair_text, metadata, window_center, window_width
        except Exception as e:
            logger.error(f"Error generating annotated slice image: {str(e)}")
            return None, f"Error: {str(e)}", "x: 0, y: 0, z: 0", {}, None, None

    def change_view_for_annotator(self, view):
        """Change the viewing orientation for image_annotator component"""
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
        
        # Generate the slice image for annotator
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
        
        # Convert to format expected by image_annotator
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": img_rgb,
            "boxes": [],
            "orientation": 0
        }
        
        return (
            gr.Slider(minimum=slider_min, maximum=slider_max, value=self.state.current_slice_idx), 
            f"{self.state.current_slice_idx}/{slider_max}",
            annotated_value
        )

    def update_window_level_for_annotator(self, window_level, window_width):
        """Update window level for image_annotator component"""
        if self.state.current_data is None:
            return None
        
        # Store the window level and width in state
        self.state.window_level = window_level
        self.state.window_width = window_width
        
        # Regenerate the current slice with new windowing
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=window_level,
            window_width=window_width,
            crosshair=self.state.crosshair_position
        )
        
        # Convert to format expected by image_annotator
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": img_rgb,
            "boxes": [],
            "orientation": 0
        }
        
        return annotated_value

    def select_file_from_browser_for_annotator(self, selected_file):
        """Handle file selection from browser for image_annotator component"""
        if not selected_file or self.state.current_data is None:
            return None, "0/0", "x: 0, y: 0, z: 0", {}, gr.Slider(), None, None
        
        # Find and load the selected file
        if self.state.file_list and selected_file in [os.path.basename(f) for f in self.state.file_list]:
            # Find the full path
            selected_file_path = None
            for file_path in self.state.file_list:
                if os.path.basename(file_path) == selected_file:
                    selected_file_path = file_path
                    break
            
            if selected_file_path:
                # Update current slice index based on file position
                self.state.current_slice_idx = self.state.file_list.index(selected_file_path)
                
                # Update crosshair position based on current view
                if self.state.current_view == "axial":
                    x, y, _ = self.state.crosshair_position
                    self.state.crosshair_position = (x, y, self.state.current_slice_idx)
                
                # Get metadata for this specific file
                metadata = get_dicom_metadata(selected_file_path)
                if not metadata:
                    metadata = self.state.current_metadata
                
                # Generate the slice image
                img = display_slice(
                    self.state.current_data, 
                    self.state.current_slice_idx, 
                    self.state.current_view,
                    crosshair=self.state.crosshair_position
                )
                
                # Convert to format expected by image_annotator
                if len(img.shape) == 2:
                    img_rgb = np.stack([img] * 3, axis=-1)
                else:
                    img_rgb = img
                
                if img_rgb.dtype != np.uint8:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                
                # Create AnnotatedImageValue format
                annotated_value = {
                    "image": img_rgb,
                    "boxes": [],
                    "orientation": 0
                }
                
                # Update crosshair info
                x, y, z = self.state.crosshair_position
                crosshair_text = f"x: {x}, y: {y}, z: {z}"
                
                # Get windowing information
                window_center = metadata.get("window_center", 500)
                window_width = metadata.get("window_width", 1000)
                if isinstance(window_center, list):
                    window_center = window_center[0]
                if isinstance(window_width, list):
                    window_width = window_width[0]
                
                # Update slider
                max_slices = self.state.get_max_slice_for_view(self.state.current_view)
                new_slider = gr.Slider(minimum=0, maximum=max_slices, value=self.state.current_slice_idx)
                
                return (annotated_value, 
                       f"{self.state.current_slice_idx + 1}/{max_slices + 1}", 
                       crosshair_text, 
                       metadata, 
                       new_slider, 
                       window_center, 
                       window_width)
        
        return None, "0/0", "x: 0, y: 0, z: 0", {}, gr.Slider(), None, None

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
