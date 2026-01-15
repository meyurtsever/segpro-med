"""
SegMed-Pro Image Handlers

This module contains event handlers specifically for gr.Image components,
used in the editor tab for MEDSAM2 AI annotation workflow.
"""

import os
import logging
import gradio as gr
import numpy as np
from typing import Optional, List, Dict, Any
import time

from utils.dicom_utils import get_dicom_metadata
from utils.visualization import (display_slice, overlay_segmentation, make_image_for_gradio)
from utils.debug_utils import log_exception
from utils.metadata_sanitizer import sanitize_metadata_for_display
from utils.annotation_manager import get_annotation_manager
from utils.annotation_helpers import extract_coordinates, extract_bbox, calculate_area
from ui.state import AppState
from ui.export_handlers import ExportHandlers

# Behavioral analytics tracking
from analytics.tracking_integration import (
    track_annotation, track_annotation_edit, track_annotation_delete,
    track_slice_change
)

logger = logging.getLogger(__name__)


class ImageViewerHandlers:
    """Handlers for viewer operations using gr.Image component"""
    
    def __init__(self, state: AppState, medsam2_handlers=None, segmentation_handlers=None):
        self.state = state
        self.medsam2_handlers = medsam2_handlers
        self.segmentation_handlers = segmentation_handlers
        # Store user annotations across all slices {slice_idx: [list_of_annotation_boxes]}
        self.user_annotations = {}
        # Store combined annotations (MEDSAM2 + user + segmentation) for each slice
        self.combined_annotations = {}
        # Initialize export handlers
        self.export_handlers = ExportHandlers(state)
        # Initialize annotation manager for persistent storage
        self.annotation_manager = get_annotation_manager()
        # Current user ID (will be set when user logs in)
        self.current_user_id = None
    
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
        
        # Set default values if not found in metadata
        if window_center is None:
            window_center = 500
        if window_width is None:
            window_width = 1000
          # Generate the slice image
        try:
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                window_level=window_center,
                window_width=window_width,
                crosshair=None,
                add_orientation_marker=False            )
            
            # NOTE: Segmentation overlays are now handled via image_annotator shapes instead of image overlays
            
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
                    logger.error(f"Error applying MEDSAM2 annotation overlay: {str(e)}")            # Convert to format expected by image_annotator
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            
            # Get segmentation shapes for current slice if available
            segmentation_shapes = []
            if (self.segmentation_handlers and 
                self.state.segmentation_loaded and 
                self.state.segmentation_data is not None):
                segmentation_shapes = self.segmentation_handlers._get_shapes_for_current_slice()
            
            # Create AnnotatedImageValue format
            annotated_value = {
                "image": img_rgb,
                "boxes": segmentation_shapes,  # Include segmentation shapes
                "orientation": 0
            }
            
            logger.info(f"Generated annotated slice image with shape {img_rgb.shape} and {len(segmentation_shapes)} segmentation shapes")
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            # Sanitize metadata for display (removes PHI while keeping clinical data)
            display_metadata = sanitize_metadata_for_display(metadata)
            
            return annotated_value, f"{self.state.current_slice_idx + 1}/{total_slices}", crosshair_text, display_metadata, window_center, window_width
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
        # Ensure maximum > minimum to avoid log10(0) error
        slider_max = max(slider_max, 1)
        
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
            add_orientation_marker=False        )
        
        # NOTE: Segmentation overlays are now handled via image_annotator shapes instead of image overlays
        
        # Convert to format expected by image_annotator
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Get segmentation shapes for current slice if available
        segmentation_shapes = []
        if (self.segmentation_handlers and 
            self.state.segmentation_loaded and 
            self.state.segmentation_data is not None):
            segmentation_shapes = self.segmentation_handlers._get_shapes_for_current_slice()
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": img_rgb,
            "boxes": segmentation_shapes,
            "orientation": 0
        }
        
        return (
            gr.Slider(minimum=slider_min, maximum=slider_max, value=self.state.current_slice_idx), 
            f"{self.state.current_slice_idx}/{slider_max}", 
            annotated_value
        )
    
    def update_window_level(self, level, width):
        """Apply window/level adjustments to the current image"""
        if self.state.current_data is None:
            return None        # Generate the slice image with window/level adjustments
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=level,
            window_width=width,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # NOTE: Segmentation overlays are now handled via image_annotator shapes instead of image overlays
        
        # Convert to format expected by image_annotator
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Get segmentation shapes for current slice if available
        segmentation_shapes = []
        if (self.segmentation_handlers and 
            self.state.segmentation_loaded and 
            self.state.segmentation_data is not None):
            segmentation_shapes = self.segmentation_handlers._get_shapes_for_current_slice()
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": img_rgb,
            "boxes": segmentation_shapes,
            "orientation": 0
        }
        
        return annotated_value
    
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
            
            # Set default values if not found in metadata
            if window_center is None:
                window_center = 500
            if window_width is None:
                window_width = 1000
            
            # Generate image at the new index
            img = display_slice(
                self.state.current_data, 
                slice_idx, 
                "axial",
                window_level=window_center,
                window_width=window_width,
                crosshair=None,
                add_orientation_marker=False
            )
              # Apply segmentation overlay if loaded AND viewer should show it
            if (self.state.segmentation_loaded and self.state.segmentation_data is not None
                and self.state.show_segmentation):
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
            
            # Sanitize metadata for display (removes PHI while keeping clinical data)
            display_metadata = sanitize_metadata_for_display(metadata)
            
            return (annotated_value, f"{slice_idx}/{len(self.state.file_list)-1}", crosshair_text, 
                   display_metadata, slice_idx, window_center, window_width)
                   
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
    
    @log_exception
    def reset_view(self):
        """Reset the view to default state"""
        if self.state.current_data is None:
            return None
        
        # Reset to the first slice of current view
        self.state.update_slice_position(0, self.state.current_view)
        
        # Generate the slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=self.state.current_window_center, 
            window_width=self.state.current_window_width,
            crosshair=None,
            add_orientation_marker=False
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
        
        return annotated_value    @log_exception
    def toggle_annotations(self, show_annotations):
        """Toggle annotation visibility"""
        if self.state.current_data is None:
            return None
            
        # Generate the slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=self.state.current_window_center, 
            window_width=self.state.current_window_width,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # If annotations should be shown and we have segmentation data
        if show_annotations and hasattr(self.state, 'current_segmentation') and self.state.current_segmentation is not None:
            # Overlay segmentation if available
            seg_slice = self.state.get_current_segmentation_slice()
            if seg_slice is not None:
                img = overlay_segmentation(
                    img, 
                    seg_slice, 
                    opacity=getattr(self.state, 'segmentation_opacity', 0.5)
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
        
        return annotated_value    @log_exception
    def update_annotation_opacity(self, opacity):
        """Update annotation opacity"""
        if self.state.current_data is None:
            return None
        
        # Store the opacity setting
        self.state.segmentation_opacity = opacity
        
        # Generate the slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            window_level=self.state.current_window_center, 
            window_width=self.state.current_window_width,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # Overlay segmentation with new opacity if available
        if hasattr(self.state, 'current_segmentation') and self.state.current_segmentation is not None:
            seg_slice = self.state.get_current_segmentation_slice()
            if seg_slice is not None:
                img = overlay_segmentation(img, seg_slice, opacity=opacity)
        
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

    @log_exception
    def export_current_view(self):
        """Export the current view as an image file"""
        if self.state.current_data is None:
            logger.warning("No data loaded for export")
            return
        
        try:
            import os
            from datetime import datetime
            from PIL import Image
            
            # Get current slice data
            slice_data = self.state.get_current_slice()
            if slice_data is None:
                logger.warning("No current slice data for export")
                return
            
            # Create display image
            display_img = display_slice(
                slice_data, 
                self.state.current_window_center, 
                self.state.current_window_width
            )
            
            # Convert to PIL Image
            if isinstance(display_img, np.ndarray):
                # Ensure proper format for PIL
                if display_img.dtype != np.uint8:
                    display_img = (display_img * 255).astype(np.uint8)
                
                if len(display_img.shape) == 2:
                    pil_image = Image.fromarray(display_img, mode='L')
                else:
                    pil_image = Image.fromarray(display_img, mode='RGB')
                
                # Create export filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"exported_view_{self.state.current_view}_{self.state.current_slice_index}_{timestamp}.png"
                
                # Save to current directory or a designated export folder
                export_dir = "exports"
                if not os.path.exists(export_dir):
                    os.makedirs(export_dir)
                
                filepath = os.path.join(export_dir, filename)
                pil_image.save(filepath)
                
                logger.info(f"View exported to: {filepath}")
                
        except Exception as e:
            logger.error(f"Error exporting current view: {str(e)}")

    @log_exception
    def export_annotations(self):
        """Export annotations as a JSON file"""
        if not hasattr(self, 'user_annotations') or not self.user_annotations:
            logger.warning("No annotations to export")
            return
        
        try:
            import json
            import os
            from datetime import datetime
            
            # Prepare annotation data for export
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "view_type": self.state.current_view,
                    "total_slices": len(self.user_annotations),
                    "data_source": getattr(self.state, 'current_file_path', "unknown")
                },
                "annotations": self.user_annotations
            }
            
            # Create export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exported_annotations_{timestamp}.json"
            
            # Save to current directory or a designated export folder
            export_dir = "exports"
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            filepath = os.path.join(export_dir, filename)            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Annotations exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting annotations: {str(e)}")
    
    def on_annotation_change_viewer(self, annotated_value):
        """Handle when user modifies annotations in image_annotator (viewer tab specific)"""
        if annotated_value is None:
            return annotated_value
        
        # Extract the current shapes/boxes from the annotated value
        current_shapes = annotated_value.get("boxes", [])
        
        # Sort shapes by area (smallest first) to prevent occlusion
        if current_shapes:
            shapes_with_areas = []
            
            for shape in current_shapes:
                if 'points' in shape and len(shape['points']) >= 3:
                    # Calculate area using shoelace formula for polygons
                    points = shape['points']
                    area = 0
                    n = len(points)
                    for i in range(n):
                        j = (i + 1) % n
                        area += points[i]['x'] * points[j]['y']
                        area -= points[j]['x'] * points[i]['y']
                    area = abs(area) / 2.0
                    shapes_with_areas.append((area, shape))
                else:
                    # For non-polygon shapes, use bounding box area
                    if 'xmin' in shape and 'ymin' in shape and 'xmax' in shape and 'ymax' in shape:
                        area = (shape['xmax'] - shape['xmin']) * (shape['ymax'] - shape['ymin'])
                        shapes_with_areas.append((area, shape))
                    else:
                        # Default to area 0 if no area can be calculated
                        shapes_with_areas.append((0, shape))
            
            # Sort shapes by area (smallest first) to prevent occlusion
            shapes_with_areas.sort(key=lambda x: x[0])
            sorted_shapes = [shape for area, shape in shapes_with_areas]
            
            # Update the annotated_value with sorted shapes
            annotated_value = dict(annotated_value)  # Create a copy
            annotated_value["boxes"] = sorted_shapes
            
            logger.info(f"Sorted {len(sorted_shapes)} shapes by area (smallest first) in viewer")
        
        # Store the user-modified shapes for the current slice
        if self.segmentation_handlers:
            self.segmentation_handlers.update_user_shapes(annotated_value.get("boxes", []), self.state.current_slice_idx)
            logger.info(f"Updated user annotations for slice {self.state.current_slice_idx}: {len(annotated_value.get('boxes', []))} shapes")        
        return annotated_value
    
    def export_single_slice(self, export_format, include_overlays, output_dir, current_annotator_value=None):
        """Export the current slice in the selected format with optional overlays"""
        try:
            if self.state.current_data is None:
                return "No data loaded for export"
            
            success, message = self.export_handlers.export_single_slice(
                export_format=export_format,
                output_dir=output_dir,
                include_overlays=include_overlays,
                current_annotator_value=current_annotator_value
            )
            
            if success:
                return f"Success: {message}"
            else:
                return f"Error: {message}"
            
        except Exception as e:
            logger.error(f"Error in export_single_slice: {str(e)}")
            return f"Export failed: {str(e)}"
    
    def export_all_slices(self, export_format, include_overlays, output_dir, all_annotator_values=None):
        """Export all slices in the selected format with optional overlays"""
        try:
            if self.state.current_data is None:
                return "No data loaded for export"
            
            success, message = self.export_handlers.export_all_slices(
                export_format=export_format,
                output_dir=output_dir,
                include_overlays=include_overlays,
                all_annotator_values=all_annotator_values
            )
            
            if success:
                return f"Success: {message}"
            else:
                return f"Error: {message}"
            
        except Exception as e:
            logger.error(f"Error in export_all_slices: {str(e)}")
            return f"Export failed: {str(e)}"
        
    def handle_image_remove_viewer(self):
        """Handle image removal event from the image_annotator component.
        This is called when the X button (Remove Image) is clicked."""
        # Reset the current image but keep the data in state
        # Return an empty AnnotatedImageValue with a blank image
        logger.info("Image removal requested via X button")
        
        # Create a small blank/transparent image instead of None
        blank_image = np.zeros((10, 10, 3), dtype=np.uint8)  # Small blank RGB image
        
        empty_annotated_value = {
            "image": blank_image,
            "boxes": [],
            "orientation": 0
        }
        
        return empty_annotated_value
    
    # ========== Persistent Storage Methods ==========
    
    def set_current_user(self, user_id: str):
        """Set the current user ID for annotation management"""
        self.current_user_id = user_id
        logger.info(f"ImageViewerHandlers: Current user set to: {user_id}")
        
        # Load persistent annotations for the current user
        self.load_persistent_annotations()
    
    def load_persistent_annotations(self):
        """Load annotations from persistent storage for current study"""
        try:
            # Skip if no user is logged in
            if not self.current_user_id:
                logger.debug("ImageViewerHandlers: No user logged in, skipping annotation loading")
                return
            
            # Skip if no data is loaded
            if not self.state.current_directory:
                logger.debug("ImageViewerHandlers: No data loaded, skipping annotation loading")
                return
            
            # Load annotations from persistent storage
            annotation_data = self.annotation_manager.load_annotations(
                user_id=self.current_user_id,
                study_path=self.state.current_directory
            )
            
            if not annotation_data:
                logger.info("ImageViewerHandlers: No persistent annotations found for current study")
                return
            
            # Clear existing in-memory annotations
            self.user_annotations.clear()
            
            # Reconstruct user_annotations from persistent storage
            slice_annotations = annotation_data.get('slice_annotations', [])
            
            for ann_record in slice_annotations:
                slice_idx = ann_record.get('slice_idx')
                view_type = ann_record.get('view_type', 'axial')
                
                # Only load annotations for current view
                if view_type != self.state.current_view:
                    continue
                
                # Initialize slice annotations if needed
                if slice_idx not in self.user_annotations:
                    self.user_annotations[slice_idx] = []
                
                # Reconstruct annotation in the format used internally
                user_annotation = {
                    'type': ann_record.get('annotation_type', 'user_drawn'),
                    'data': ann_record.get('original_data', {}),
                    'timestamp': ann_record.get('timestamp', time.time()),
                    'slice_idx': slice_idx
                }
                
                self.user_annotations[slice_idx].append(user_annotation)
            
            logger.info(f"ImageViewerHandlers: Loaded {len(slice_annotations)} annotations from persistent storage "
                       f"for {len(self.user_annotations)} slices")
            
        except Exception as e:
            logger.error(f"ImageViewerHandlers: Error loading persistent annotations: {e}", exc_info=True)


class ImagePlotToolHandlers:
    """Handlers for plot tool operations using gr.Image component (dummy implementation)"""
    def __init__(self, state: AppState):
        self.state = state
        self.medsam2_handlers = None  # Will be set by the main app
        # Store user annotations across all slices {slice_idx: [list_of_annotation_boxes]}
        self.user_annotations = {}
        # Store combined annotations (MEDSAM2 + user) for each slice
        self.combined_annotations = {}        # Track if we're currently loading a slice to prevent saving during load
        self._loading_slice = False
        # Store fingerprints of loaded annotations to detect actual changes
        self._loaded_fingerprints = {}  # {slice_idx: annotation_fingerprint}        # Add a lock to prevent race conditions during fast navigation
        self._navigation_lock = False
        # Track the last slice we were saving to prevent conflicts
        self._last_saved_slice = None
        # Track last save time to prevent rapid saves
        self._last_save_time = 0
        # Initialize annotation manager for persistent storage
        self.annotation_manager = get_annotation_manager()
        # Current user ID (will be set when user logs in)
        self.current_user_id = None
    
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
            crosshair=None,
            add_orientation_marker=False
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
          # Set loading flag to prevent saving during slice load
        self._loading_slice = True
        
        # Add a small delay to help prevent race conditions during fast navigation
        time.sleep(0.05)  # 50ms delay
        
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
                    if "WindowCenter" in per_slice_metadata:
                        window_center = per_slice_metadata["WindowCenter"]
                        if isinstance(window_center, list):
                            window_center = window_center[0]
                        logger.info(f"Using WindowCenter from per-slice metadata: {window_center}")
                    
                    if "WindowWidth" in per_slice_metadata:
                        window_width = per_slice_metadata["WindowWidth"]
                        if isinstance(window_width, list):
                            window_width = window_width[0]
                        logger.info(f"Using WindowWidth from per-slice metadata: {window_width}")
            except Exception as e:
                logger.error(f"Error getting per-slice metadata: {str(e)}")
        
        # Set default values if not found in metadata
        if window_center is None:
            window_center = 500
        if window_width is None:
            window_width = 1000
          # Generate the slice image as numpy array
        try:
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                window_level=window_center,
                window_width=window_width,
                crosshair=None,
                add_orientation_marker=False
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
                img_rgb = (img_rgb * 255).astype(np.uint8)            # Check for annotation overlays and convert to polygon shapes
            annotation_shapes = []
              # Check if user has edited annotations for this slice
            has_user_annotations = self.state.current_slice_idx in self.user_annotations
            user_annotation_count = len(self.user_annotations.get(self.state.current_slice_idx, []))
            
            logger.info(f"Slice {self.state.current_slice_idx}: has_user_annotations={has_user_annotations}, count={user_annotation_count}")
            
            if has_user_annotations and user_annotation_count > 0:
                # Load user annotations (these replace MEDSAM2 annotations completely)
                user_annotation_objects = self.user_annotations[self.state.current_slice_idx]
                # Extract the actual annotation data from the stored user annotation objects
                for user_ann in user_annotation_objects:
                    if isinstance(user_ann, dict) and 'data' in user_ann:
                        annotation_shapes.append(user_ann['data'])
                        logger.debug(f"Loaded annotation: {user_ann['data'].get('type', 'unknown')} with label '{user_ann['data'].get('label', '')}'")
                    else:                        # For backward compatibility, if it's not in the expected format
                        annotation_shapes.append(user_ann)
                        logger.debug(f"Loaded annotation (legacy format): {type(user_ann)}")
                
                logger.info(f"Loaded {len(user_annotation_objects)} user annotations for slice {self.state.current_slice_idx} (replaces MEDSAM2)")
            elif has_user_annotations and user_annotation_count == 0:
                # User has explicitly deleted all annotations on this slice - show empty
                logger.info(f"Slice {self.state.current_slice_idx} has user deletions - showing empty slice")
            else:                # Load original MEDSAM2 annotations only if user hasn't edited them
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
                        
                        logger.info(f"Loaded {len(annotation_shapes)} original MEDSAM2 annotations for slice {self.state.current_slice_idx}")
                    except Exception as e:
                        logger.error(f"Error converting MEDSAM2 annotation to polygon shapes: {str(e)}")
                    logger.error(f"Overlay data structure: {type(overlay_data)}")
                    if isinstance(overlay_data, dict):
                        logger.error(f"Overlay data keys: {list(overlay_data.keys())}")            # Store combined annotations for this slice
            self.combined_annotations[self.state.current_slice_idx] = annotation_shapes.copy()
            
            logger.info(f"Total annotations for slice {self.state.current_slice_idx}: {len(annotation_shapes)}")
            
            # Create AnnotatedImageValue format - display at full size
            annotated_value = {
                "image": img_rgb,
                "boxes": annotation_shapes,  # Include MEDSAM2 polygon shapes for interactive editing
                "orientation": 0
            }
            
            # Store the fingerprint of what we just loaded to detect future user changes
            loaded_fingerprint = self._get_annotation_fingerprint(annotated_value)
            self._loaded_fingerprints[self.state.current_slice_idx] = loaded_fingerprint
            logger.info(f"Stored loaded fingerprint for slice {self.state.current_slice_idx}: '{loaded_fingerprint}'")
            
            logger.info(f"Generated annotated slice image with shape {img_rgb.shape}")
            
            # Update crosshair info
            x, y, z = self.state.crosshair_position
            crosshair_text = f"x: {x}, y: {y}, z: {z}"
            
            # Sanitize metadata for display (removes PHI while keeping clinical data)
            display_metadata = sanitize_metadata_for_display(metadata)
            
            # Clear loading flag before returning
            self._loading_slice = False
            return annotated_value, f"{self.state.current_slice_idx + 1}/{total_slices}", crosshair_text, display_metadata, window_center, window_width
        except Exception as e:
            logger.error(f"Error generating annotated slice image: {str(e)}")
            # Clear loading flag before returning
            self._loading_slice = False
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
        # Ensure maximum > minimum to avoid log10(0) error
        slider_max = max(slider_max, 1)
        
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
            crosshair=None,
            add_orientation_marker=False
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
            crosshair=None,
            add_orientation_marker=False
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
                
                # Extract window values
                window_center = metadata.get("WindowCenter", None)
                window_width = metadata.get("WindowWidth", None)
                if isinstance(window_center, list):
                    window_center = window_center[0]
                if isinstance(window_width, list):
                    window_width = window_width[0]
                
                # Set default values if not found in metadata
                if window_center is None:
                    window_center = 500
                if window_width is None:
                    window_width = 1000
                  # Generate the slice image
                img = display_slice(
                    self.state.current_data, 
                    self.state.current_slice_idx, 
                    self.state.current_view,
                    window_level=window_center,
                    window_width=window_width,
                    crosshair=None,
                    add_orientation_marker=False
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
                
                # Update slider - ensure maximum > minimum to avoid log10(0) error
                max_slices = self.state.get_max_slice_for_view(self.state.current_view)
                slider_max = max(max_slices, 1)  # Ensure maximum is at least 1
                new_slider = gr.Slider(minimum=0, maximum=slider_max, value=self.state.current_slice_idx, visible=(max_slices > 0))
                
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
        if self.state.current_data is None:            return slider_value        
        current_value = int(slider_value)
        max_value = self.state.get_max_slice_for_view(self.state.current_view)
        return min(current_value + 1, max_value)

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
            
            logger.info(f"Checking if annotations need saving for slice {slice_idx}")
            
            # Generate fingerprint of current annotation state
            current_fingerprint = self._get_annotation_fingerprint(annotated_image_value)
            
            # Check if we have a stored fingerprint for this slice (from when we loaded it)
            stored_fingerprint = self._loaded_fingerprints.get(slice_idx, None)
            
            logger.info(f"Slice {slice_idx}: current_fingerprint='{current_fingerprint}', stored_fingerprint='{stored_fingerprint}'")
            
            # Only save if the annotation has actually changed from what we loaded
            if current_fingerprint == stored_fingerprint:
                logger.info(f"No changes detected for slice {slice_idx} - skipping save")
                return
            
            logger.info(f"Changes detected - saving annotations for slice {slice_idx}")
            
            # Always clear existing user annotations for this slice first
            # This ensures deletions are properly handled
            if slice_idx in self.user_annotations:
                del self.user_annotations[slice_idx]
                logger.info(f"Cleared existing user annotations for slice {slice_idx}")
            
            if annotated_image_value is None:
                logger.info(f"No annotations to save for slice {slice_idx}")
                # Update stored fingerprint to reflect empty state
                self._loaded_fingerprints[slice_idx] = current_fingerprint
                # Save empty state to persistent storage
                self._save_to_persistent_storage()
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
                    # Skip incomplete freehand/polygon annotations (in-progress drawings)
                    ann_type = annotation.get('type', '')
                    ann_points = annotation.get('points', [])
                    
                    # Only save completed shapes:
                    # - For point-based shapes (freehand, polygon, polyline), require at least 3 points
                    # - For box/rect shapes, require minimum area (skip tiny in-progress boxes)
                    # - This filters out intermediate drawing states
                    if ann_type in ('freehand', 'polygon', 'polyline'):
                        if not ann_points or len(ann_points) < 3:
                            logger.debug(f"Skipping incomplete {ann_type} annotation with {len(ann_points)} points")
                            continue
                    elif ann_type in ('box', 'rect', 'rectangle'):
                        # Skip very small boxes that are likely in-progress drawings
                        xmin = annotation.get('xmin', 0)
                        ymin = annotation.get('ymin', 0)
                        xmax = annotation.get('xmax', 0)
                        ymax = annotation.get('ymax', 0)
                        width = abs(xmax - xmin)
                        height = abs(ymax - ymin)
                        area = width * height
                        
                        # Skip boxes smaller than 25 pixels (5x5) - likely accidental clicks or in-progress
                        if area < 25:
                            logger.debug(f"Skipping tiny {ann_type} annotation with area {area} (likely in-progress)")
                            continue
                    
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
                # Save empty list to indicate user explicitly deleted all annotations
                self.user_annotations[slice_idx] = []
                logger.info(f"Saved empty annotation list for slice {slice_idx} (user deleted all annotations)")
            
            # Update stored fingerprint to reflect the new saved state
            self._loaded_fingerprints[slice_idx] = current_fingerprint
            
            # Save to persistent storage
            self._save_to_persistent_storage()
            
        except Exception as e:
            logger.error(f"Error saving user annotations for slice {slice_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _save_to_persistent_storage(self):
        """Save all annotations to persistent storage using AnnotationManager"""
        import hashlib
        import json
        from datetime import datetime
        
        try:
            # Skip if no user is logged in
            if not self.current_user_id:
                logger.debug("No user logged in, skipping persistent storage")
                return
            
            # Skip if no data is loaded
            if not self.state.current_directory:
                logger.debug("No data loaded, skipping persistent storage")
                return
            
            # Load existing annotations to preserve annotations from other slices/views
            existing_data = self.annotation_manager.load_annotations(
                user_id=self.current_user_id,
                study_path=self.state.current_directory
            )
            
            if existing_data:
                logger.info(f"[ImagePlotToolHandlers] Loaded existing data with {len(existing_data.get('slice_annotations', []))} annotations")
            else:
                logger.info("[ImagePlotToolHandlers] No existing annotation data found")
            
            # Get the set of slice indices we're currently saving
            current_slice_indices = set(self.user_annotations.keys())
            logger.info(f"[ImagePlotToolHandlers] Current slice indices to save: {current_slice_indices}")
            
            # Filter out old annotations from slices we're updating
            # Keep annotations from other slices and other views
            preserved_annotations = []
            if existing_data and 'slice_annotations' in existing_data:
                for ann in existing_data['slice_annotations']:
                    slice_idx = ann.get('slice_idx')
                    view_type = ann.get('view_type', 'axial')
                    
                    # Keep annotation if it's from a different slice or different view
                    if slice_idx not in current_slice_indices or view_type != self.state.current_view:
                        preserved_annotations.append(ann)
                    else:
                        logger.debug(f"Removing old annotation {ann.get('annotation_id')} from slice {slice_idx} (will be replaced)")
            
            logger.info(f"Preserved {len(preserved_annotations)} annotations from other slices/views")
            
            # Prepare NEW slice annotations for slices in self.user_annotations
            new_slice_annotations = []
            
            for slice_idx, annotations in self.user_annotations.items():
                for ann in annotations:
                    # Extract the annotation data
                    ann_data = ann.get('data', {})
                    
                    # Generate stable annotation ID based on shape geometry
                    # This ensures the same shape keeps the same ID even if label/color changes
                    # Create geometry signature from points or bbox
                    geometry_data = {}
                    if 'points' in ann_data and ann_data['points']:
                        # Round points to avoid floating point differences
                        geometry_data['points'] = [
                            [round(p.get('x', 0), 1), round(p.get('y', 0), 1)] 
                            if isinstance(p, dict) else [round(p[0], 1), round(p[1], 1)]
                            for p in ann_data['points']
                        ]
                    elif 'xmin' in ann_data:
                        geometry_data['bbox'] = [
                            round(ann_data.get('xmin', 0), 1),
                            round(ann_data.get('ymin', 0), 1),
                            round(ann_data.get('xmax', 0), 1),
                            round(ann_data.get('ymax', 0), 1)
                        ]
                    
                    geometry_data['type'] = ann_data.get('type', 'unknown')
                    geometry_str = json.dumps(geometry_data, sort_keys=True)
                    geometry_hash = hashlib.md5(geometry_str.encode()).hexdigest()[:8]
                    stable_id = f"slice_{slice_idx}_{geometry_hash}"
                    
                    # Create a comprehensive annotation record
                    annotation_record = {
                        'slice_idx': slice_idx,
                        'view_type': self.state.current_view,
                        'annotation_id': stable_id,
                        'timestamp': ann.get('timestamp', time.time()),
                        'annotation_type': 'manual',  # Manual drawing
                        'shape_type': ann_data.get('type', 'unknown'),
                        'label': ann_data.get('label', ''),
                        'color': ann_data.get('color', None),
                        'coordinates': extract_coordinates(ann_data),
                        'bbox': extract_bbox(ann_data),
                        'area': calculate_area(ann_data),
                        'original_data': ann_data  # Store complete original data
                    }
                    
                    new_slice_annotations.append(annotation_record)
            
            # Combine preserved and new annotations
            all_slice_annotations = preserved_annotations + new_slice_annotations
            
            logger.info(f"Total annotations to save: {len(all_slice_annotations)} "
                       f"(preserved: {len(preserved_annotations)}, new: {len(new_slice_annotations)})")
            
            # Prepare study metadata
            study_metadata = {
                'data_type': self.state.current_data_type,
                'view': self.state.current_view,
                'shape': list(self.state.current_shape) if self.state.current_shape else None,
                'num_slices': len(self.state.file_list) if self.state.file_list else None,
            }
            
            # Add DICOM-specific metadata if available
            if self.state.current_metadata:
                study_metadata['modality'] = self.state.current_metadata.get('Modality', 'Unknown')
                study_metadata['series_description'] = self.state.current_metadata.get('SeriesDescription', '')
            
            # Manually build the complete annotation data structure to REPLACE (not merge)
            annotation_file = self.annotation_manager._get_annotation_file(
                self.current_user_id,
                self.state.current_directory
            )
            
            if existing_data:
                # Update existing record
                annotation_data = existing_data
                from datetime import datetime
                annotation_data['modified_at'] = datetime.now().isoformat()
                annotation_data['modification_count'] = annotation_data.get('modification_count', 0) + 1
            else:
                # Create new record
                from datetime import datetime
                annotation_data = {
                    'user_id': self.current_user_id,
                    'study_path': os.path.normpath(self.state.current_directory),
                    'study_hash': self.annotation_manager._get_study_hash(self.state.current_directory),
                    'created_at': datetime.now().isoformat(),
                    'modified_at': datetime.now().isoformat(),
                    'modification_count': 0,
                }
            
            # Replace slice_annotations completely (no merging by annotation_id)
            annotation_data['slice_annotations'] = all_slice_annotations
            annotation_data['annotation_type'] = 'manual'
            annotation_data['study_metadata'] = study_metadata
            
            # Write to file
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(all_slice_annotations)} total annotations to persistent storage "
                       f"({len(new_slice_annotations)} from current session)")
                
        except Exception as e:
            logger.error(f"Error saving to persistent storage: {e}", exc_info=True)
    
    def _extract_coordinates(self, ann_data: Dict[str, Any]) -> List:
        """Extract coordinates from annotation data"""
        try:
            shape_type = ann_data.get('type', '')
            
            if shape_type == 'polygon' and 'points' in ann_data:
                points = ann_data['points']
                if isinstance(points, list):
                    return [[p.get('x', p[0]) if isinstance(p, dict) else p[0], 
                            p.get('y', p[1]) if isinstance(p, dict) else p[1]] 
                            for p in points]
            elif shape_type == 'box' or shape_type == 'rect':
                if 'xmin' in ann_data and 'ymin' in ann_data:
                    return [
                        [ann_data['xmin'], ann_data['ymin']],
                        [ann_data['xmax'], ann_data['ymin']],
                        [ann_data['xmax'], ann_data['ymax']],
                        [ann_data['xmin'], ann_data['ymax']]
                    ]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            return []
    
    def _extract_bbox(self, ann_data: Dict[str, Any]) -> List:
        """Extract bounding box from annotation data"""
        try:
            if 'xmin' in ann_data:
                return [
                    ann_data.get('xmin', 0),
                    ann_data.get('ymin', 0),
                    ann_data.get('xmax', 0),
                    ann_data.get('ymax', 0)
                ]
            elif 'points' in ann_data:
                points = ann_data['points']
                if isinstance(points, list) and points:
                    xs = [p.get('x', p[0]) if isinstance(p, dict) else p[0] for p in points]
                    ys = [p.get('y', p[1]) if isinstance(p, dict) else p[1] for p in points]
                    return [min(xs), min(ys), max(xs), max(ys)]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting bbox: {e}")
            return []
    
    def _calculate_area(self, ann_data: Dict[str, Any]) -> float:
        """Calculate area of annotation"""
        try:
            shape_type = ann_data.get('type', '')
            
            if shape_type == 'polygon' and 'points' in ann_data:
                points = ann_data['points']
                if isinstance(points, list) and len(points) >= 3:
                    # Shoelace formula
                    area = 0
                    n = len(points)
                    for i in range(n):
                        j = (i + 1) % n
                        x1 = points[i].get('x', points[i][0]) if isinstance(points[i], dict) else points[i][0]
                        y1 = points[i].get('y', points[i][1]) if isinstance(points[i], dict) else points[i][1]
                        x2 = points[j].get('x', points[j][0]) if isinstance(points[j], dict) else points[j][0]
                        y2 = points[j].get('y', points[j][1]) if isinstance(points[j], dict) else points[j][1]
                        area += x1 * y2 - x2 * y1
                    return abs(area) / 2.0
            elif shape_type == 'box' or shape_type == 'rect':
                if 'xmin' in ann_data and 'ymin' in ann_data:
                    width = ann_data.get('xmax', 0) - ann_data.get('xmin', 0)
                    height = ann_data.get('ymax', 0) - ann_data.get('ymin', 0)
                    return abs(width * height)
            
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating area: {e}")
            return 0.0
    
    def load_persistent_annotations(self):
        """Load annotations from persistent storage for current study"""
        try:
            # Skip if no user is logged in
            if not self.current_user_id:
                logger.debug("No user logged in, skipping annotation loading")
                return
            
            # Skip if no data is loaded
            if not self.state.current_directory:
                logger.debug("No data loaded, skipping annotation loading")
                return
            
            # Load annotations from persistent storage
            annotation_data = self.annotation_manager.load_annotations(
                user_id=self.current_user_id,
                study_path=self.state.current_directory
            )
            
            if not annotation_data:
                logger.info("No persistent annotations found for current study")
                # Only clear memory if we're actually switching studies AND there's nothing to load
                # Don't clear if we're just loading the same study
                if self.user_annotations:
                    logger.debug("Keeping current in-memory annotations (no file to overwrite with)")
                return
            
            # We have persistent annotations - safe to clear memory and reload
            self.user_annotations.clear()
            self._loaded_fingerprints.clear()
            
            # Reconstruct user_annotations from persistent storage
            slice_annotations = annotation_data.get('slice_annotations', [])
            
            logger.info(f"Found {len(slice_annotations)} total annotations in storage for this study")
            
            for ann_record in slice_annotations:
                slice_idx = ann_record.get('slice_idx')
                view_type = ann_record.get('view_type', 'axial')
                
                # Only load annotations for current view
                if view_type != self.state.current_view:
                    logger.debug(f"Skipping annotation for slice {slice_idx} (view: {view_type}, current: {self.state.current_view})")
                    continue
                
                # Initialize slice annotations if needed
                if slice_idx not in self.user_annotations:
                    self.user_annotations[slice_idx] = []
                
                # Reconstruct annotation in the format used by save_user_annotations
                user_annotation = {
                    'type': ann_record.get('annotation_type', 'user_drawn'),
                    'data': ann_record.get('original_data', {}),
                    'timestamp': ann_record.get('timestamp', time.time()),
                    'slice_idx': slice_idx
                }
                
                self.user_annotations[slice_idx].append(user_annotation)
                logger.debug(f"Loaded annotation for slice {slice_idx}: {ann_record.get('shape_type')} with label '{ann_record.get('label')}'")
            
            # CRITICAL: Set fingerprints for loaded slices to prevent unnecessary re-saves
            # Generate fingerprint from the loaded annotation data in UI format
            for slice_idx, annotations in self.user_annotations.items():
                # Convert to UI format (boxes) for fingerprint generation
                ui_boxes = []
                for ann in annotations:
                    ann_data = ann.get('data', {})
                    if ann_data:
                        ui_boxes.append(ann_data)
                
                # Generate and store fingerprint
                ui_format = {'boxes': ui_boxes}
                fingerprint = self._get_annotation_fingerprint(ui_format)
                self._loaded_fingerprints[slice_idx] = fingerprint
                logger.info(f"Set fingerprint for loaded slice {slice_idx}: '{fingerprint}' ({len(ui_boxes)} annotations)")
            
            logger.info(f"Loaded {len(slice_annotations)} annotations from persistent storage "
                       f"for {len(self.user_annotations)} slices")
            
        except Exception as e:
            logger.error(f"Error loading persistent annotations: {e}", exc_info=True)
    
    def set_current_user(self, user_id: str):
        """Set the current user ID for annotation management"""
        self.current_user_id = user_id
        logger.info(f"Current user set to: {user_id}")
        
        # Load persistent annotations for the current user
        self.load_persistent_annotations()

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
    
    def on_annotation_change_editor_save(self, annotated_image_value):
        """Handle annotation changes in the ImageAnnotator component"""
        try:
            # Skip saving if we're currently loading a slice or navigating
            if self._loading_slice or self._navigation_lock:
                logger.info(f"Skipping annotation save during slice load or navigation (loading={self._loading_slice}, nav_lock={self._navigation_lock})")
                return None
            
            current_slice = self.state.current_slice_idx
            logger.info(f"Processing annotation change for slice {current_slice}")
            
            # Additional protection: don't save if this is the same slice we just processed
            if (self._last_saved_slice is not None and 
                current_slice != self._last_saved_slice and 
                time.time() - getattr(self, '_last_save_time', 0) < 0.5):
                logger.info(f"Skipping save - slice changed too quickly from {self._last_saved_slice} to {current_slice}")
                return None
            
            # Additional debug: log the source of the change
            if annotated_image_value:
                if isinstance(annotated_image_value, dict) and 'boxes' in annotated_image_value:
                    box_count = len(annotated_image_value.get('boxes', []))
                    logger.info(f"Annotation change contains {box_count} boxes")
            
            # Save current annotations immediately - this handles deletions, edits, additions
            self.save_user_annotations(annotated_image_value, current_slice)
            
            # Track the save operation
            self._last_saved_slice = current_slice
            self._last_save_time = time.time()
            
            # Log change details for debugging
            if annotated_image_value:
                if hasattr(annotated_image_value, 'annotations'):
                    current_count = len(annotated_image_value.annotations or [])
                elif isinstance(annotated_image_value, dict) and 'boxes' in annotated_image_value:
                    current_count = len(annotated_image_value['boxes'] or [])
                else:
                    current_count = 0
                    
                logger.info(f"Saved {current_count} annotations for slice {current_slice}")
                
                # Track annotation changes for behavioral analytics
                try:
                    # Get previous count FOR THIS SPECIFIC SLICE to detect additions/deletions
                    # This prevents false events when navigating between slices
                    slice_annotation_counts = getattr(self, '_slice_annotation_counts', {})
                    prev_count = slice_annotation_counts.get(current_slice, None)
                    
                    # Track creation time per slice to suppress immediate "edit" events after creation
                    # (Gradio fires multiple change events while drawing a shape)
                    slice_creation_times = getattr(self, '_slice_creation_times', {})
                    creation_cooldown_sec = 2.0  # Suppress edits within 2 seconds of creation
                    
                    # If we don't have a previous count for this slice, initialize it
                    # This happens on first visit to a slice - don't track as "created"
                    if prev_count is None:
                        slice_annotation_counts[current_slice] = current_count
                        self._slice_annotation_counts = slice_annotation_counts
                        logger.debug(f"Initialized annotation count for slice {current_slice}: {current_count}")
                    elif current_count > prev_count:
                        # New annotations added - FIRST CHANGE (shape drawn, no label yet)
                        logger.info(f"🔢 Count increased from {prev_count} to {current_count} - annotation added to boxes!")
                        
                        # Check if we have a tool timestamp
                        if self.state.last_tool_timestamp:
                            elapsed = int(time.time() * 1000) - self.state.last_tool_timestamp
                            logger.info(f"⏰ Time since tool selected: {elapsed}ms (waiting for modal confirmation)")
                        
                        boxes = annotated_image_value.get('boxes', []) if isinstance(annotated_image_value, dict) else []
                        
                        # Check if this is an AI-assisted annotation (has label already)
                        is_ai_annotation = False
                        if boxes and len(boxes) > 0:
                            last_box = boxes[-1]
                            label = last_box.get('label', '')
                            if label and ('MEDSAM2' in label.upper() or 'MEDSAM' in label.upper() or 
                                        'SAM2' in label.upper() or 'AUTO MASK' in label.upper()):
                                is_ai_annotation = True
                                logger.info(f"✅ AI-assisted annotation detected - tracking immediately")
                                
                                # IMMEDIATELY set annotation timestamp to filter automatic tool switches
                                self.state.last_annotation_timestamp = int(time.time() * 1000)
                                
                                # Track AI annotation immediately (no modal)
                                # Use last_tool_selected from state for accurate annotation_type
                                ann_type = self.state.last_tool_selected if self.state.last_tool_selected else last_box.get('type', 'unknown')
                                track_annotation(
                                    slice_idx=current_slice,
                                    annotation_type=ann_type,
                                    label=label,
                                    ai_assisted=True,
                                    duration_ms=None
                                )

                                # Reset tool timestamp
                                self.state.last_tool_selected = None
                                self.state.last_tool_timestamp = None
                        
                        if not is_ai_annotation:
                            # Manual annotation - store as PENDING (waiting for modal confirmation)
                            # IMMEDIATELY set annotation timestamp to filter automatic tool switches
                            self.state.last_annotation_timestamp = int(time.time() * 1000)
                            
                            # Create fingerprint of unlabeled annotation
                            unlabeled_fingerprint = self._get_annotation_fingerprint({'boxes': boxes})
                            
                            # CRITICAL: Store timestamp AND tool name in pending dict
                            # For box/circle: setDragMode() fires AFTER modal (in onModalNewChange)
                            # For polygon/freehand: setDragMode() fires BEFORE modal (in onPolygonFinishCreation)
                            # So we need to preserve both timestamp and tool_selected for all cases
                            # IMPORTANT: New annotations are inserted at index 0 (not at the end)
                            self.state.pending_modal_confirmations[current_slice] = {
                                'count': current_count,
                                'unlabeled_fingerprint': unlabeled_fingerprint,
                                'annotation_index': 0,  # NEW annotation is always at index 0
                                'timestamp': self.state.last_tool_timestamp,  # Preserve original tool selection timestamp
                                'tool_name': self.state.last_tool_selected  # Preserve tool name for accurate annotation_type
                            }
                            logger.info(f"📝 Pending manual annotation on slice {current_slice} - waiting for modal OK (tool: {self.state.last_tool_selected})")
                            
                            # DON'T reset state timestamp yet - polygon/freehand need it until modal shows
                            # It will be overwritten by setDragMode() but that's OK - we have it in pending dict
                        
                        # Update the slice count and record creation time
                        slice_annotation_counts[current_slice] = current_count
                        self._slice_annotation_counts = slice_annotation_counts
                        slice_creation_times[current_slice] = time.time()
                        self._slice_creation_times = slice_creation_times
                        
                    elif current_count < prev_count:
                        # Annotations deleted by user
                        for _ in range(prev_count - current_count):
                            track_annotation_delete(slice_idx=current_slice)
                        
                        # Update the slice count
                        slice_annotation_counts[current_slice] = current_count
                        self._slice_annotation_counts = slice_annotation_counts
                        
                    elif current_count > 0 and prev_count > 0:
                        # Same count - could be edit OR modal confirmation
                        
                        # Check if this is a pending modal confirmation (SECOND CHANGE)
                        if current_slice in self.state.pending_modal_confirmations:
                            pending = self.state.pending_modal_confirmations[current_slice]
                            
                            # Verify count matches
                            if pending['count'] == current_count:
                                boxes = annotated_image_value.get('boxes', []) if isinstance(annotated_image_value, dict) else []
                                current_fingerprint = self._get_annotation_fingerprint({'boxes': boxes})
                                
                                # Check if fingerprint changed (label was applied)
                                if current_fingerprint != pending['unlabeled_fingerprint']:
                                    logger.info(f"✅ Modal confirmed! Label applied to annotation on slice {current_slice}")
                                    
                                    # Find the annotation that matches the tool type we were tracking
                                    # For polygon/freehand, the annotation might have moved in the array
                                    annotation = None
                                    ann_idx = -1
                                    tool_name = pending.get('tool_name')
                                    
                                    # Search for the FIRST annotation that matches the tool type
                                    # (This will be the one we just created)
                                    for i, box in enumerate(boxes):
                                        if isinstance(box, dict) and box.get('type') == tool_name:
                                            annotation = box
                                            ann_idx = i
                                            break
                                    
                                    # Fallback: if we didn't find by type, use the stored index
                                    if annotation is None and boxes:
                                        ann_idx = pending.get('annotation_index', 0)
                                        if ann_idx < len(boxes):
                                            annotation = boxes[ann_idx]
                                            logger.info(f"⚠️ Using fallback index {ann_idx}")
                                    
                                    if annotation:
                                        # Extract label - ensure we get the actual string label, not color
                                        label = annotation.get('label', '')
                                        
                                        # If label is empty or is a color array/tuple, it means label wasn't set yet
                                        # This happens for polygon/freehand where modal fires before label is applied
                                        if not label or isinstance(label, (list, tuple)):
                                            # Don't track this annotation yet - wait for next change when label is actually set
                                            return None
                                        
                                        # Use tool_name from pending dict for accurate annotation_type
                                        # This ensures we track the correct shape type (box/circle/polygon/freehand)
                                        ann_type = pending.get('tool_name', annotation.get('type', 'unknown'))
                                        
                                        # Calculate duration NOW (includes drawing + modal time)
                                        # Use timestamp from pending dict (preserved from original tool selection)
                                        duration_ms = None
                                        original_timestamp = pending.get('timestamp')
                                        if original_timestamp:
                                            current_time_ms = int(time.time() * 1000)
                                            duration_ms = current_time_ms - original_timestamp
                                            logger.info(f"⏱️ Manual annotation complete - Total time: {duration_ms}ms (drawing + modal) - Type: {ann_type}")
                                        else:
                                            logger.warning("⚠️ No timestamp found in pending confirmation")
                                        
                                        # Track the completed annotation
                                        track_annotation(
                                            slice_idx=current_slice,
                                            annotation_type=ann_type,
                                            label=label if label else 'unlabeled',
                                            ai_assisted=False,
                                            duration_ms=duration_ms
                                        )
                                        
                                        # NOTE: Don't track tool_usage here - it's already tracked by tool_selected event handler
                                        # This was causing duplicate tool_usage events
                                    
                                    # Clear pending confirmation
                                    del self.state.pending_modal_confirmations[current_slice]
                        
                        # Only check for edits if this wasn't a modal confirmation
                        if current_slice not in self.state.pending_modal_confirmations or current_fingerprint == pending.get('unlabeled_fingerprint', ''):
                            # Regular edit detection (not a modal confirmation)
                            # BUT: Skip if within cooldown period after creation
                            # (Gradio fires multiple change events while drawing shapes)
                            last_creation_time = slice_creation_times.get(current_slice, 0)
                            time_since_creation = time.time() - last_creation_time
                            
                            # Also check if we're within 500ms of annotation being tracked (from last_annotation_timestamp)
                            # This filters out automatic "edit" events that fire immediately after annotation_created
                            time_since_annotation_ms = (int(time.time() * 1000) - self.state.last_annotation_timestamp) if hasattr(self.state, 'last_annotation_timestamp') else 999999
                            
                            if time_since_creation < creation_cooldown_sec:
                                # Still within cooldown - this is part of the drawing action, not a real edit
                                logger.debug(f"Skipping edit tracking - within {creation_cooldown_sec}s cooldown after creation")
                            elif time_since_annotation_ms < 1800:
                                # Within 1800ms of annotation_created event - this is an automatic edit, not user action
                                logger.debug(f"Skipping edit tracking - within {time_since_annotation_ms}ms of annotation_created (automatic component behavior)")
                            else:
                                # Outside cooldown - check if actually edited
                                slice_fingerprints = getattr(self, '_slice_fingerprints', {})
                                prev_fingerprint = slice_fingerprints.get(current_slice, '')
                                boxes = annotated_image_value.get('boxes', []) if isinstance(annotated_image_value, dict) else []
                                current_fingerprint = str(sorted([str(b.get('points', b.get('xmin', ''))) for b in boxes]))
                                if prev_fingerprint and prev_fingerprint != current_fingerprint:
                                    track_annotation_edit(slice_idx=current_slice)
                                slice_fingerprints[current_slice] = current_fingerprint
                                self._slice_fingerprints = slice_fingerprints
                except Exception as track_e:
                    logger.debug(f"Behavioral tracking skipped: {track_e}")
              # Don't return anything to avoid circular dependency with image_display.change
            return None
            
        except Exception as e:
            logger.error(f"Error handling annotation change: {e}")
            return None

    def handle_annotator_slider_change(self, slider_value, current_annotated_value=None):
        """Handle slider changes in the annotator - save current annotations and load new slice"""
        try:
            # Set navigation lock to prevent race conditions
            if self._navigation_lock:
                logger.info(f"Navigation already in progress, skipping slider change to {slider_value}")
                return None, f"{slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None
            
            self._navigation_lock = True
            logger.info(f"Starting slider navigation to slice {slider_value}")
            
            # Capture current slice index BEFORE any changes
            previous_slice_idx = self.state.current_slice_idx
            new_slice_idx = int(slider_value)
            
            # Save current annotations if provided - use the CURRENT slice before it changes
            if current_annotated_value is not None:
                logger.info(f"Saving annotations for previous slice {previous_slice_idx} before moving to slice {new_slice_idx}")
                self.save_user_annotations(current_annotated_value, previous_slice_idx)
            
            # Track slice navigation for behavioral analytics (only if actually changing slices)
            if previous_slice_idx != new_slice_idx:
                try:
                    track_slice_change(
                        from_slice=previous_slice_idx,
                        to_slice=new_slice_idx,
                        method="slider"
                    )
                except Exception as track_e:
                    logger.debug(f"Behavioral tracking skipped: {track_e}")
            
            # Update to new slice
            result = self.update_slice_for_annotator(slider_value, self.state.current_view)
            
            # Clear navigation lock
            self._navigation_lock = False
            logger.info(f"Completed slider navigation to slice {slider_value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling annotator slider change: {e}")
            # Clear navigation lock on error
            self._navigation_lock = False
            # Return current state if error occurs
            if current_annotated_value is not None:
                return current_annotated_value, f"{slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None
            else:
                return None, f"{slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None

    def handle_annotator_navigation(self, direction, current_slider_value, current_annotated_value=None):
        """Handle navigation buttons (prev/next) in the annotator"""
        try:
            # Set navigation lock to prevent race conditions during fast clicking
            if self._navigation_lock:
                logger.info(f"Navigation already in progress, skipping {direction} navigation")
                return (None, f"{current_slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None), current_slider_value
            
            self._navigation_lock = True
            
            # Capture previous slice BEFORE any changes
            previous_slice_idx = self.state.current_slice_idx
            
            logger.info(f"Starting {direction} navigation from slice {previous_slice_idx}")
            
            # Save current annotations if provided - use the CURRENT slice before it changes
            if current_annotated_value is not None:
                logger.info(f"Saving annotations for previous slice {previous_slice_idx} before navigation")
                self.save_user_annotations(current_annotated_value, previous_slice_idx)
            
            # Calculate new slider value
            if direction == "next":
                new_value = self.next_slice(current_slider_value)
            elif direction == "prev":
                new_value = self.prev_slice(current_slider_value)
            else:
                new_value = current_slider_value
            
            # Track slice navigation for behavioral analytics (only if actually changing slices)
            if previous_slice_idx != new_value:
                try:
                    track_slice_change(
                        from_slice=previous_slice_idx,
                        to_slice=new_value,
                        method="button"
                    )
                except Exception as track_e:
                    logger.debug(f"Behavioral tracking skipped: {track_e}")
            
            # Update to new slice
            result = self.update_slice_for_annotator(new_value, self.state.current_view), new_value
            
            # Clear navigation lock
            self._navigation_lock = False
            logger.info(f"Completed {direction} navigation to slice {new_value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling annotator navigation: {e}")
            # Clear navigation lock on error
            self._navigation_lock = False
            return (None, f"{current_slider_value}/0", "x: 0, y: 0, z: 0", {}, None, None), current_slider_value

    def debug_annotation_state(self):
        """Debug method to show current annotation state"""
        logger.info("=== ANNOTATION DEBUG STATE ===")
        logger.info(f"Current slice: {self.state.current_slice_idx}")
        logger.info(f"User annotations stored for slices: {list(self.user_annotations.keys())}")
        
        for slice_idx, annotations in self.user_annotations.items():
            logger.info(f"Slice {slice_idx}: {len(annotations)} annotations")
            for i, ann in enumerate(annotations):
                if isinstance(ann, dict):
                    ann_type = ann.get('type', 'unknown')
                    timestamp = ann.get('timestamp', 'no timestamp')
                    logger.info(f"  Annotation {i}: {ann_type} at {timestamp}")
                else:
                    logger.info(f"  Annotation {i}: {type(ann)}")
        
        logger.info("=== END ANNOTATION DEBUG ===")
        return f"Debug info logged. User annotations on {len(self.user_annotations)} slices."

    def debug_fingerprint_state(self):
        """Debug method to show current fingerprint state"""
        logger.info("=== FINGERPRINT DEBUG STATE ===")
        logger.info(f"Current slice: {self.state.current_slice_idx}")
        logger.info(f"Stored fingerprints for slices: {list(self._loaded_fingerprints.keys())}")
        
        for slice_idx, fingerprint in self._loaded_fingerprints.items():
            logger.info(f"Slice {slice_idx}: fingerprint='{fingerprint}'")
        
        logger.info("=== END FINGERPRINT DEBUG ===")
        return f"Debug info logged. Fingerprints stored for {len(self._loaded_fingerprints)} slices."

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
    
    def get_annotation_mode_for_slice(self, slice_idx: int) -> str:
        """Debug method to check annotation mode for a slice"""
        if slice_idx in self.user_annotations:
            return f"USER_EDITED ({len(self.user_annotations[slice_idx])} annotations)"
        elif (self.medsam2_handlers and 
              hasattr(self.medsam2_handlers, 'annotation_overlays') and
              slice_idx in self.medsam2_handlers.annotation_overlays):
            return "ORIGINAL_MEDSAM2"
        else:            return "NO_ANNOTATIONS"
    
    def _get_annotation_fingerprint(self, annotated_image_value):
        """Generate a fingerprint of annotation data to detect changes"""
        if not annotated_image_value:
            return "empty"
        
        try:
            if isinstance(annotated_image_value, dict) and 'boxes' in annotated_image_value:
                boxes = annotated_image_value.get('boxes', [])
                # Create a detailed fingerprint based on all annotation properties
                fingerprint_data = []
                for box in boxes:
                    if isinstance(box, dict):
                        # Extract key properties that uniquely identify the annotation
                        box_info = {
                            'type': box.get('type', ''),
                            'label': box.get('label', ''),
                            'color': box.get('color', ''),
                            'coordinates': str(box.get('coordinates', box.get('points', []))),
                        }                        # Add any other properties that might be relevant
                        if 'stroke' in box:
                            box_info['stroke'] = box['stroke']
                        if 'fill' in box:
                            box_info['fill'] = box['fill']
                        
                        fingerprint_data.append(str(sorted(box_info.items())))
                
                fingerprint = f"boxes:{len(boxes)}|" + "|".join(sorted(fingerprint_data))
                return fingerprint
            
            return f"unknown_format:{type(annotated_image_value).__name__}"
        except Exception as e:
            logger.error(f"Error creating annotation fingerprint: {e}")
            return f"error:{str(e)}"
    
    def clear_user_annotations_for_slice(self, slice_idx: int):
        """Clear user annotations for a specific slice and reset to MEDSAM2 state"""
        try:
            if slice_idx in self.user_annotations:
                del self.user_annotations[slice_idx]
                logger.info(f"Cleared user annotations for slice {slice_idx}")
            
            # Also clear the fingerprint so the slice can be reloaded fresh
            if slice_idx in self._loaded_fingerprints:
                del self._loaded_fingerprints[slice_idx]
                logger.info(f"Cleared fingerprint for slice {slice_idx}")
            
            return f"Cleared user annotations for slice {slice_idx}"
        except Exception as e:
            logger.error(f"Error clearing user annotations for slice {slice_idx}: {e}")
            return f"Error clearing annotations: {str(e)}"
    
    def reset_navigation_state(self):
        """Reset navigation locks if they get stuck - emergency method"""
        logger.info("Resetting navigation state - clearing locks")
        self._loading_slice = False
        self._navigation_lock = False
        self._last_save_time = 0
        return "Navigation state reset successfully"
    
    def get_navigation_status(self):
        """Get current navigation status for debugging"""
        return {
            "loading_slice": self._loading_slice,
            "navigation_lock": self._navigation_lock,
            "last_saved_slice": self._last_saved_slice,
            "last_save_time": self._last_save_time,
            "current_slice": self.state.current_slice_idx
        }

    def handle_image_remove(self):
        """Handle image removal event from the image_annotator component.
        This is called when the X button (Remove Image) is clicked."""
        # Reset the current image but keep the data in state
        # Return an empty AnnotatedImageValue with a blank image
        logger.info("Image removal requested via X button")
        
        # Create a small blank/transparent image instead of None
        blank_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Small blank RGB image
        
        empty_annotated_value = {
            "image": blank_image,
            "boxes": [],
            "orientation": 0
        }
        
        return empty_annotated_value
    
    # ========== Persistent Storage Methods (shared with ImageViewerHandlers) ==========
    
    def _save_to_persistent_storage(self):
        """Save all annotations to persistent storage using AnnotationManager"""
        import hashlib
        import json
        from datetime import datetime
        
        try:
            # Skip if no user is logged in
            if not self.current_user_id:
                logger.debug("No user logged in, skipping persistent storage")
                return
            
            # Skip if no data is loaded
            if not self.state.current_directory:
                logger.debug("No data loaded, skipping persistent storage")
                return
            
            # Load existing annotations to preserve annotations from other slices/views
            existing_data = self.annotation_manager.load_annotations(
                user_id=self.current_user_id,
                study_path=self.state.current_directory
            )
            
            if existing_data:
                logger.info(f"[ImageViewerHandlers] Loaded existing data with {len(existing_data.get('slice_annotations', []))} annotations")
            else:
                logger.info("[ImageViewerHandlers] No existing annotation data found")
            
            # Get the set of slice indices we're currently saving
            current_slice_indices = set(self.user_annotations.keys())
            logger.info(f"[ImageViewerHandlers] Current slice indices to save: {current_slice_indices}")
            
            # Filter out old annotations from slices we're updating
            # Keep annotations from other slices and other views
            preserved_annotations = []
            if existing_data and 'slice_annotations' in existing_data:
                for ann in existing_data['slice_annotations']:
                    slice_idx = ann.get('slice_idx')
                    view_type = ann.get('view_type', 'axial')
                    
                    # Keep annotation if it's from a different slice or different view
                    if slice_idx not in current_slice_indices or view_type != self.state.current_view:
                        preserved_annotations.append(ann)
                    else:
                        logger.debug(f"Removing old annotation {ann.get('annotation_id')} from slice {slice_idx} (will be replaced)")
            
            logger.info(f"Preserved {len(preserved_annotations)} annotations from other slices/views")
            
            # Prepare NEW slice annotations for slices in self.user_annotations
            new_slice_annotations = []
            
            logger.info(f"[ImageViewerHandlers] Processing {len(self.user_annotations)} slices from user_annotations")
            for slice_idx, annotations in self.user_annotations.items():
                logger.info(f"[ImageViewerHandlers]   Slice {slice_idx}: {len(annotations)} annotations, type: {type(annotations)}")
                for ann in annotations:
                    # Extract the annotation data
                    ann_data = ann.get('data', {})
                    
                    # Generate stable annotation ID based on shape geometry
                    # This ensures the same shape keeps the same ID even if label/color changes
                    # Create geometry signature from points or bbox
                    geometry_data = {}
                    if 'points' in ann_data and ann_data['points']:
                        # Round points to avoid floating point differences
                        geometry_data['points'] = [
                            [round(p.get('x', 0), 1), round(p.get('y', 0), 1)] 
                            if isinstance(p, dict) else [round(p[0], 1), round(p[1], 1)]
                            for p in ann_data['points']
                        ]
                    elif 'xmin' in ann_data:
                        geometry_data['bbox'] = [
                            round(ann_data.get('xmin', 0), 1),
                            round(ann_data.get('ymin', 0), 1),
                            round(ann_data.get('xmax', 0), 1),
                            round(ann_data.get('ymax', 0), 1)
                        ]
                    
                    geometry_data['type'] = ann_data.get('type', 'unknown')
                    geometry_str = json.dumps(geometry_data, sort_keys=True)
                    geometry_hash = hashlib.md5(geometry_str.encode()).hexdigest()[:8]
                    stable_id = f"slice_{slice_idx}_{geometry_hash}"
                    
                    # Create a comprehensive annotation record
                    annotation_record = {
                        'slice_idx': slice_idx,
                        'view_type': self.state.current_view,
                        'annotation_id': stable_id,
                        'timestamp': ann.get('timestamp', time.time()),
                        'annotation_type': 'manual',  # Manual drawing
                        'shape_type': ann_data.get('type', 'unknown'),
                        'label': ann_data.get('label', ''),
                        'color': ann_data.get('color', None),
                        'coordinates': extract_coordinates(ann_data),
                        'bbox': extract_bbox(ann_data),
                        'area': calculate_area(ann_data),
                        'original_data': ann_data  # Store complete original data
                    }
                    
                    new_slice_annotations.append(annotation_record)
            
            # Combine preserved and new annotations
            all_slice_annotations = preserved_annotations + new_slice_annotations
            
            logger.info(f"Total annotations to save: {len(all_slice_annotations)} "
                       f"(preserved: {len(preserved_annotations)}, new: {len(new_slice_annotations)})")
            
            # Prepare study metadata
            study_metadata = {
                'data_type': self.state.current_data_type,
                'view': self.state.current_view,
                'shape': list(self.state.current_shape) if self.state.current_shape else None,
                'num_slices': len(self.state.file_list) if self.state.file_list else None,
            }
            
            # Add DICOM-specific metadata if available
            if self.state.current_metadata:
                study_metadata['modality'] = self.state.current_metadata.get('Modality', 'Unknown')
                study_metadata['series_description'] = self.state.current_metadata.get('SeriesDescription', '')
            
            # Manually build the complete annotation data structure to REPLACE (not merge)
            annotation_file = self.annotation_manager._get_annotation_file(
                self.current_user_id,
                self.state.current_directory
            )
            
            if existing_data:
                # Update existing record
                annotation_data = existing_data
                from datetime import datetime
                annotation_data['modified_at'] = datetime.now().isoformat()
                annotation_data['modification_count'] = annotation_data.get('modification_count', 0) + 1
            else:
                # Create new record
                from datetime import datetime
                annotation_data = {
                    'user_id': self.current_user_id,
                    'study_path': os.path.normpath(self.state.current_directory),
                    'study_hash': self.annotation_manager._get_study_hash(self.state.current_directory),
                    'created_at': datetime.now().isoformat(),
                    'modified_at': datetime.now().isoformat(),
                    'modification_count': 0,
                }
            
            # Replace slice_annotations completely (no merging by annotation_id)
            annotation_data['slice_annotations'] = all_slice_annotations
            annotation_data['annotation_type'] = 'manual'
            annotation_data['study_metadata'] = study_metadata
            
            # Write to file
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(all_slice_annotations)} total annotations to persistent storage "
                       f"({len(new_slice_annotations)} from current session)")
                
        except Exception as e:
            logger.error(f"Error saving to persistent storage: {e}", exc_info=True)
    
    def _extract_coordinates(self, ann_data: Dict[str, Any]) -> List:
        """Extract coordinates from annotation data"""
        try:
            shape_type = ann_data.get('type', '')
            
            if shape_type == 'polygon' and 'points' in ann_data:
                points = ann_data['points']
                if isinstance(points, list):
                    return [[p.get('x', p[0]) if isinstance(p, dict) else p[0], 
                            p.get('y', p[1]) if isinstance(p, dict) else p[1]] 
                            for p in points]
            elif shape_type == 'box' or shape_type == 'rect':
                if 'xmin' in ann_data and 'ymin' in ann_data:
                    return [
                        [ann_data['xmin'], ann_data['ymin']],
                        [ann_data['xmax'], ann_data['ymin']],
                        [ann_data['xmax'], ann_data['ymax']],
                        [ann_data['xmin'], ann_data['ymax']]
                    ]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            return []
    
    def _extract_bbox(self, ann_data: Dict[str, Any]) -> List:
        """Extract bounding box from annotation data"""
        try:
            if 'xmin' in ann_data:
                return [
                    ann_data.get('xmin', 0),
                    ann_data.get('ymin', 0),
                    ann_data.get('xmax', 0),
                    ann_data.get('ymax', 0)
                ]
            elif 'points' in ann_data:
                points = ann_data['points']
                if isinstance(points, list) and points:
                    xs = [p.get('x', p[0]) if isinstance(p, dict) else p[0] for p in points]
                    ys = [p.get('y', p[1]) if isinstance(p, dict) else p[1] for p in points]
                    return [min(xs), min(ys), max(xs), max(ys)]
            
            return []
        except Exception as e:
            logger.error(f"Error extracting bbox: {e}")
            return []
    
    def _calculate_area(self, ann_data: Dict[str, Any]) -> float:
        """Calculate area of annotation"""
        try:
            shape_type = ann_data.get('type', '')
            
            if shape_type == 'polygon' and 'points' in ann_data:
                points = ann_data['points']
                if isinstance(points, list) and len(points) >= 3:
                    # Shoelace formula
                    area = 0
                    n = len(points)
                    for i in range(n):
                        j = (i + 1) % n
                        x1 = points[i].get('x', points[i][0]) if isinstance(points[i], dict) else points[i][0]
                        y1 = points[i].get('y', points[i][1]) if isinstance(points[i], dict) else points[i][1]
                        x2 = points[j].get('x', points[j][0]) if isinstance(points[j], dict) else points[j][0]
                        y2 = points[j].get('y', points[j][1]) if isinstance(points[j], dict) else points[j][1]
                        area += x1 * y2 - x2 * y1
                    return abs(area) / 2.0
            elif shape_type == 'box' or shape_type == 'rect':
                if 'xmin' in ann_data and 'ymin' in ann_data:
                    width = ann_data.get('xmax', 0) - ann_data.get('xmin', 0)
                    height = ann_data.get('ymax', 0) - ann_data.get('ymin', 0)
                    return abs(width * height)
            
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating area: {e}")
            return 0.0
    
    def load_persistent_annotations(self):
        """Load annotations from persistent storage for current study"""
        try:
            # Skip if no user is logged in
            if not self.current_user_id:
                logger.debug("No user logged in, skipping annotation loading")
                return
            
            # Skip if no data is loaded
            if not self.state.current_directory:
                logger.debug("No data loaded, skipping annotation loading")
                return
            
            # Load annotations from persistent storage
            annotation_data = self.annotation_manager.load_annotations(
                user_id=self.current_user_id,
                study_path=self.state.current_directory
            )
            
            if not annotation_data:
                logger.info("No persistent annotations found for current study")
                return
            
            # Clear existing in-memory annotations
            self.user_annotations.clear()
            self._loaded_fingerprints.clear()
            
            # Reconstruct user_annotations from persistent storage
            slice_annotations = annotation_data.get('slice_annotations', [])
            
            for ann_record in slice_annotations:
                slice_idx = ann_record.get('slice_idx')
                view_type = ann_record.get('view_type', 'axial')
                
                # Only load annotations for current view
                if view_type != self.state.current_view:
                    continue
                
                # Initialize slice annotations if needed
                if slice_idx not in self.user_annotations:
                    self.user_annotations[slice_idx] = []
                
                # Reconstruct annotation in the format used by save_user_annotations
                user_annotation = {
                    'type': ann_record.get('annotation_type', 'user_drawn'),
                    'data': ann_record.get('original_data', {}),
                    'timestamp': ann_record.get('timestamp', time.time()),
                    'slice_idx': slice_idx
                }
                
                self.user_annotations[slice_idx].append(user_annotation)
            
            logger.info(f"Loaded {len(slice_annotations)} annotations from persistent storage "
                       f"for {len(self.user_annotations)} slices")
            
        except Exception as e:
            logger.error(f"Error loading persistent annotations: {e}", exc_info=True)
    
    def set_current_user(self, user_id: str):
        """Set the current user ID for annotation management"""
        self.current_user_id = user_id
        logger.info(f"Current user set to: {user_id}")
        
        # Load persistent annotations for the current user
        self.load_persistent_annotations()
    
    

