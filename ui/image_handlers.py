"""
SegMed-Pro Image Handlers

This module contains event handlers specifically for gr.Image components,
used in the editor tab for MEDSAM2 AI annotation workflow.
"""

import os
import logging
import gradio as gr
import numpy as np
from typing import Optional, List
import time

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
        # Store user annotations across all slices {slice_idx: [list_of_annotation_boxes]}
        self.user_annotations = {}
        # Store combined annotations (MEDSAM2 + user) for each slice
        self.combined_annotations = {}
    
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
        self.medsam2_handlers = None  # Will be set by the main app
        # Store user annotations across all slices {slice_idx: [list_of_annotation_boxes]}
        self.user_annotations = {}
        # Store combined annotations (MEDSAM2 + user) for each slice
        self.combined_annotations = {}
    
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
                    logger.error(f"Error applying segmentation overlay: {str(e)}")            # Convert to format expected by image_annotator
            # Ensure it's RGB and uint8
            if len(img.shape) == 2:
                # Grayscale to RGB
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)            # Check for MEDSAM2 annotation overlays and convert to polygon shapes
            annotation_shapes = []
            
            # Load MEDSAM2 annotations for current slice
            if (self.medsam2_handlers and 
                hasattr(self.medsam2_handlers, 'annotation_overlays') and
                self.state.current_slice_idx in self.medsam2_handlers.annotation_overlays):
                try:
                    from utils.visualization import create_annotation_boxes_from_mask
                    overlay_data = self.medsam2_handlers.annotation_overlays[self.state.current_slice_idx]
                    
                    # Handle both old format (single mask) and new format (multiple annotations)
                    if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                        # Old format - single annotation
                        mask_array = overlay_data['mask']
                        medsam2_shapes = create_annotation_boxes_from_mask(
                            mask_array, 
                            label="MEDSAM2 Annotation",
                            label_index=1
                        )
                        annotation_shapes.extend(medsam2_shapes)
                    else:
                        # New format - multiple annotations per slice
                        for annotation_id, annotation_data in overlay_data.items():
                            if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                                mask_array = annotation_data['mask']
                                shapes = create_annotation_boxes_from_mask(
                                    mask_array, 
                                    label=f"MEDSAM2 {annotation_id}",
                                    label_index=1
                                )
                                annotation_shapes.extend(shapes)
                    
                    logger.info(f"Loaded {len(annotation_shapes)} MEDSAM2 annotations for slice {self.state.current_slice_idx}")
                except Exception as e:
                    logger.error(f"Error converting MEDSAM2 annotation to polygon shapes: {str(e)}")
                    logger.error(f"Overlay data structure: {type(overlay_data)}")
                    if isinstance(overlay_data, dict):
                        logger.error(f"Overlay data keys: {list(overlay_data.keys())}")
            
            # Load user annotations for current slice
            if self.state.current_slice_idx in self.user_annotations:
                user_shapes = self.user_annotations[self.state.current_slice_idx]
                annotation_shapes.extend(user_shapes)
                logger.info(f"Loaded {len(user_shapes)} user annotations for slice {self.state.current_slice_idx}")
            
            # Store combined annotations for this slice
            self.combined_annotations[self.state.current_slice_idx] = annotation_shapes.copy()
            
            logger.info(f"Total annotations for slice {self.state.current_slice_idx}: {len(annotation_shapes)}")
            
            # Create AnnotatedImageValue format - display at full size
            annotated_value = {
                "image": img_rgb,
                "boxes": annotation_shapes,  # Include MEDSAM2 polygon shapes for interactive editing
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
    
    def handle_image_remove(self):
        """Handle image removal event from the image_annotator component.
        This is called when the X button (Remove Image) is clicked."""
        # Reset the current image but keep the data in state
        # Return an empty AnnotatedImageValue with a blank image
        logger.info("Image removal requested via X button")
        
        # Create a small blank/transparent image instead of None
        import numpy as np
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Small blank RGB image
        
        empty_annotated_value = {
            "image": blank_image,
            "boxes": [],
            "orientation": 0
        }
        
        return empty_annotated_value
    
    @log_exception
    def save_user_annotations(self, annotated_image_data):
        """Save user annotations from the image annotator component"""
        if annotated_image_data is None:
            return "No annotation data provided"
        
        try:
            current_slice = self.state.current_slice_idx
            
            # Extract boxes from the annotated image data
            if isinstance(annotated_image_data, dict) and 'boxes' in annotated_image_data:
                all_boxes = annotated_image_data['boxes']
                
                # Separate MEDSAM2 annotations from user annotations
                user_boxes = []
                for box in all_boxes:
                    # Check if it's a user annotation (not MEDSAM2)
                    if isinstance(box, dict) and 'label' in box:
                        label = box.get('label', '')
                        if not label.startswith('MEDSAM2'):
                            user_boxes.append(box)
                    else:
                        # If no label or not MEDSAM2, consider it user annotation
                        user_boxes.append(box)
                
                # Store user annotations for this slice
                self.user_annotations[current_slice] = user_boxes
                
                logger.info(f"Saved {len(user_boxes)} user annotations for slice {current_slice}")
                return f"Saved {len(user_boxes)} user annotations for slice {current_slice}"
            
            return "No annotation boxes found in data"
            
        except Exception as e:
            logger.error(f"Error saving user annotations: {str(e)}")
            return f"Error saving annotations: {str(e)}"
    
    @log_exception
    def get_annotation_summary(self):
        """Get a summary of all annotations across all slices"""
        try:
            summary = {}
            
            # Count MEDSAM2 annotations
            medsam2_count = 0
            if (self.medsam2_handlers and 
                hasattr(self.medsam2_handlers, 'annotation_overlays')):
                for slice_idx, overlay_data in self.medsam2_handlers.annotation_overlays.items():
                    if isinstance(overlay_data, dict):
                        if 'mask' in overlay_data:
                            # Old format - single annotation
                            medsam2_count += 1
                            summary[slice_idx] = summary.get(slice_idx, 0) + 1
                        else:
                            # New format - multiple annotations
                            count = len([k for k, v in overlay_data.items() 
                                       if isinstance(v, dict) and 'mask' in v])
                            medsam2_count += count
                            summary[slice_idx] = summary.get(slice_idx, 0) + count
            
            # Count user annotations
            user_count = 0
            for slice_idx, user_boxes in self.user_annotations.items():
                count = len(user_boxes)
                user_count += count
                summary[slice_idx] = summary.get(slice_idx, 0) + count
            
            logger.info(f"Annotation summary - MEDSAM2: {medsam2_count}, User: {user_count}, Total slices with annotations: {len(summary)}")
            return summary, medsam2_count, user_count
            
        except Exception as e:
            logger.error(f"Error getting annotation summary: {str(e)}")
            return {}, 0, 0

    def save_user_annotations(self, annotated_image_value, slice_idx: int = None):
        """Save user annotations from the ImageAnnotator component to persistent storage"""
        try:
            if slice_idx is None:
                slice_idx = self.state.current_slice_idx
            
            logger.info(f"Saving annotations for slice {slice_idx}")
            
            # Always clear existing user annotations for this slice first
            # This ensures deletions are properly handled
            if slice_idx in self.user_annotations:
                del self.user_annotations[slice_idx]
                logger.info(f"Cleared existing user annotations for slice {slice_idx}")
            
            if annotated_image_value is None:
                logger.info(f"No annotations to save for slice {slice_idx}")
                return
            
            # Extract annotations from the AnnotatedImageValue
            new_annotations = []
            
            # Handle different possible structures of annotated_image_value
            if hasattr(annotated_image_value, 'annotations'):
                annotations = annotated_image_value.annotations
            elif isinstance(annotated_image_value, dict) and 'boxes' in annotated_image_value:
                annotations = annotated_image_value['boxes']
            elif isinstance(annotated_image_value, dict) and 'annotations' in annotated_image_value:
                annotations = annotated_image_value['annotations']
            else:
                logger.warning(f"Unknown annotation format: {type(annotated_image_value)}")
                return
            
            if annotations:
                for annotation in annotations:
                    # Create a complete copy of the annotation with all properties
                    # This preserves shape, color, label, and any other modifications
                    user_annotation = {
                        'type': 'user_drawn',
                        'data': annotation,  # Store the complete annotation object
                        'timestamp': time.time(),
                        'slice_idx': slice_idx
                    }
                    new_annotations.append(user_annotation)
                
                # Store all new annotations for this slice
                self.user_annotations[slice_idx] = new_annotations
                logger.info(f"Saved {len(new_annotations)} user annotations for slice {slice_idx}")
                
                # Log the annotation details for debugging
                for i, ann in enumerate(new_annotations):
                    ann_data = ann['data']
                    if isinstance(ann_data, dict):
                        logger.info(f"  Annotation {i}: {ann_data.get('type', 'unknown')} - {ann_data.get('label', 'no label')}")
                    else:
                        logger.info(f"  Annotation {i}: {type(ann_data)}")
            else:
                logger.info(f"No annotations found in image value for slice {slice_idx}")
                
        except Exception as e:
            logger.error(f"Error saving user annotations for slice {slice_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_all_annotations_for_slice(self, slice_idx: int) -> List:
        """Get all annotations (MEDSAM2 + user) for a specific slice"""
        all_annotations = []
        
        try:
            # Add MEDSAM2 annotations
            medsam2_annotations = self.get_medsam2_annotations_for_slice(slice_idx)
            all_annotations.extend(medsam2_annotations)
              # Add user annotations
            if slice_idx in self.user_annotations:
                for user_annotation in self.user_annotations[slice_idx]:
                    all_annotations.append(user_annotation['data'])
            
            logger.info(f"Retrieved {len(all_annotations)} total annotations for slice {slice_idx}")
            return all_annotations
            
        except Exception as e:
            logger.error(f"Error getting all annotations for slice {slice_idx}: {e}")
            return []

    def on_annotation_change(self, annotated_image_value):
        """Handle annotation changes in the ImageAnnotator component"""
        try:
            current_slice = self.state.current_slice_idx
            logger.info(f"Processing annotation change for slice {current_slice}")
            
            # Save current annotations immediately - this handles deletions, edits, additions
            self.save_user_annotations(annotated_image_value, current_slice)
            
            # Log change details for debugging
            if annotated_image_value:
                if hasattr(annotated_image_value, 'annotations'):
                    current_count = len(annotated_image_value.annotations or [])
                elif isinstance(annotated_image_value, dict) and 'boxes' in annotated_image_value:
                    current_count = len(annotated_image_value['boxes'] or [])
                else:
                    current_count = 0
                    
                logger.info(f"Saved {current_count} annotations for slice {current_slice}")
            
            # Return the current value to maintain the interface
            # Don't reload - this prevents losing user edits in progress
            return annotated_image_value
            
        except Exception as e:
            logger.error(f"Error handling annotation change: {e}")
            return annotated_image_value
    
    def handle_annotator_slider_change(self, slider_value, current_annotated_value=None):
        """Handle slider changes in the annotator - save current annotations and load new slice"""
        try:
            # Save current annotations if provided
            if current_annotated_value is not None:
                self.save_user_annotations(current_annotated_value)
            
            # Update to new slice
            return self.update_slice_for_annotator(slider_value, self.state.current_view)
            
        except Exception as e:
            logger.error(f"Error handling annotator slider change: {e}")
            # Return current state if error occurs
            if current_annotated_value is not None:
                return current_annotated_value, f"{slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None
            else:
                return None, f"{slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None

    def handle_annotator_navigation(self, direction, current_slider_value, current_annotated_value=None):
        """Handle navigation buttons (prev/next) in the annotator"""
        try:
            # Save current annotations if provided
            if current_annotated_value is not None:
                self.save_user_annotations(current_annotated_value)
            
            # Calculate new slider value
            if direction == "next":
                new_value = self.next_slice(current_slider_value)
            elif direction == "prev":
                new_value = self.prev_slice(current_slider_value)
            else:
                new_value = current_slider_value
            
            # Update to new slice
            return self.update_slice_for_annotator(new_value, self.state.current_view), new_value
            
        except Exception as e:
            logger.error(f"Error handling annotator navigation: {e}")
            return (None, f"{current_slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None), current_slider_value

    def debug_annotation_state(self):
        """Debug method to show current annotation state"""
        try:
            logger.info("=== ANNOTATION DEBUG STATE ===")
            logger.info(f"Current slice: {self.state.current_slice_idx}")
            
            # Check MEDSAM2 annotations
            if (self.medsam2_handlers and 
                hasattr(self.medsam2_handlers, 'annotation_overlays')):
                medsam2_overlays = self.medsam2_handlers.annotation_overlays
                logger.info(f"MEDSAM2 overlays available for slices: {list(medsam2_overlays.keys())}")
                
                for slice_idx, overlay_data in medsam2_overlays.items():
                    if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                        logger.info(f"  Slice {slice_idx}: Single annotation (old format)")
                    else:
                        count = len([k for k, v in overlay_data.items() 
                                   if isinstance(v, dict) and 'mask' in v])
                        logger.info(f"  Slice {slice_idx}: {count} annotations (new format)")
            else:
                logger.info("No MEDSAM2 handler or overlays available")
            
            # Check user annotations
            logger.info(f"User annotations available for slices: {list(self.user_annotations.keys())}")
            for slice_idx, annotations in self.user_annotations.items():
                logger.info(f"  Slice {slice_idx}: {len(annotations)} user annotations")
            
            # Check combined annotations
            logger.info(f"Combined annotations available for slices: {list(self.combined_annotations.keys())}")
            for slice_idx, annotations in self.combined_annotations.items():
                logger.info(f"  Slice {slice_idx}: {len(annotations)} combined annotations")
            
            logger.info("=== END ANNOTATION DEBUG ===")
            
        except Exception as e:
            logger.error(f"Error in debug_annotation_state: {e}")

    def get_annotation_status_message(self):
        """Get a status message about current annotations"""
        try:
            total_slices_with_annotations = set()
            medsam2_count = 0
            user_count = 0
            
            # Count MEDSAM2 annotations
            if (self.medsam2_handlers and 
                hasattr(self.medsam2_handlers, 'annotation_overlays')):
                for slice_idx, overlay_data in self.medsam2_handlers.annotation_overlays.items():
                    total_slices_with_annotations.add(slice_idx)
                    if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                        medsam2_count += 1
                    else:
                        count = len([k for k, v in overlay_data.items() 
                                   if isinstance(v, dict) and 'mask' in v])
                        medsam2_count += count
            
            # Count user annotations
            for slice_idx, annotations in self.user_annotations.items():
                total_slices_with_annotations.add(slice_idx)
                user_count += len(annotations)
            
            message = f"Annotations: {len(total_slices_with_annotations)} slices, "
            message += f"{medsam2_count} MEDSAM2, {user_count} user-drawn"
            
            current_slice_annotations = 0
            if self.state.current_slice_idx in self.combined_annotations:
                current_slice_annotations = len(self.combined_annotations[self.state.current_slice_idx])
            
            message += f" | Current slice: {current_slice_annotations} annotations"
            
            return message
            
        except Exception as e:
            logger.error(f"Error getting annotation status: {e}")
            return f"Annotation status error: {str(e)}"
