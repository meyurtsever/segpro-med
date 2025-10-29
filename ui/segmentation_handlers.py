"""
SegMed-Pro Segmentation Event Handlers

This module contains all the segmentation-related event handling logic,
including loading, overlay management, and shape conversion.
"""

import os
import sys
import logging
import numpy as np
import nibabel as nib
import gradio as gr
from typing import Optional, Tuple

# Add utils to path for imports
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.visualization import (display_slice, overlay_segmentation, 
                                load_itk_snap_labels, make_image_for_gradio)
from utils.debug_utils import log_exception
from ui.state import AppState

logger = logging.getLogger(__name__)


class SegmentationHandlers:
    """Handlers for segmentation operations"""
    
    def __init__(self, state: AppState):
        self.state = state
        # Store segmentation shapes per slice for persistence across navigation        # Format: {slice_idx: [list_of_polygon_shapes]}
        self.segmentation_shapes = {}
        # Store user-modified shapes separately
        self.user_modified_shapes = {}
    
    def _generate_segmentation_shapes_for_slice(self, seg_slice, slice_idx):
        """Generate polygon shapes from segmentation mask for a specific slice"""
        from utils.visualization import create_annotation_boxes_from_mask
        
        shapes = []
        unique_labels = np.unique(seg_slice)
        
        # Collect all shapes with their areas for sorting
        shapes_with_areas = []
        
        for label_val in unique_labels:
            if label_val == 0:  # Skip background
                continue
                
            # Create binary mask for this label
            binary_mask = (seg_slice == label_val)            # Get color for this label from colormap
            color = self.state.segmentation_colormap.get(label_val, [255, 0, 0])  # Default to red
            logger.info(f"Label {label_val}: Using color {color} from colormap")
            
            # Get label name from labelmap
            label_name = self.state.segmentation_labelmap.get(label_val, f"Seg_Label_{label_val}")
            logger.info(f"Label {label_val}: Using label name '{label_name}' from labelmap")
            
            # Convert mask to polygon shapes
            label_shapes = create_annotation_boxes_from_mask(
                binary_mask, 
                label=label_name,  # Use actual label name from .label file
                label_index=label_val,
                color=tuple(color)  # Ensure color is a tuple
            )
            
            # Calculate area for each shape and store with the shape
            for shape in label_shapes:
                if 'points' in shape and len(shape['points']) >= 3:
                    # Calculate area using shoelace formula
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
        
        # Sort shapes by area (smallest first) to prevent occlusion
        shapes_with_areas.sort(key=lambda x: x[0])
        shapes = [shape for area, shape in shapes_with_areas]
        
        logger.info(f"Generated {len(shapes)} shapes for slice {slice_idx}, sorted by area (smallest first)")
        
        return shapes
    
    def _get_shapes_for_current_slice(self):
        """Get segmentation shapes for current slice, generating if needed"""
        current_idx = self.state.current_slice_idx
        
        # Check if user has modified shapes for this slice
        if current_idx in self.user_modified_shapes:
            return self.user_modified_shapes[current_idx]
        
        # Check if we have generated shapes for this slice
        if current_idx in self.segmentation_shapes:
            return self.segmentation_shapes[current_idx]
        
        # Generate shapes if segmentation is loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
            seg_slice = self.state.get_segmentation_slice(
                self.state.current_view, 
                current_idx
            )
            
            if seg_slice is not None:
                shapes = self._generate_segmentation_shapes_for_slice(seg_slice, current_idx)
                self.segmentation_shapes[current_idx] = shapes
                return shapes        
        return []
    
    def update_user_shapes(self, shapes, slice_idx=None):
        """Update user-modified shapes for a specific slice"""
        if slice_idx is None:
            slice_idx = self.state.current_slice_idx
        
        self.user_modified_shapes[slice_idx] = shapes
        logger.info(f"Updated user shapes for slice {slice_idx}: {len(shapes)} shapes")
    
    @log_exception
    def direct_load_segmentation(self, seg_file, label_file=None):
        """Direct method to load segmentation file with axis swapping for dimension mismatches"""
        # Helper function to create empty annotated image structure
        def create_empty_annotated_image():
            blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
            return {
                "image": blank_image,
                "boxes": [],
                "orientation": 0
            }
        
        if self.state.current_data is None:
            return "Please load a DICOM dataset first", create_empty_annotated_image()
        
        if seg_file is None:
            return "No segmentation file provided", create_empty_annotated_image()
        
        try:
            # Get file path from the uploaded file object
            seg_path = seg_file.name
            logger.info(f"Direct loading segmentation file: {seg_path}")
            
            # Load label file if provided
            if label_file is not None:
                label_path = label_file.name
                logger.info(f"Loading label file: {label_path}")
                custom_colormap, custom_labelmap = load_itk_snap_labels(label_path)
                
                if custom_colormap:
                    # Update the segmentation colormap with loaded values
                    self.state.segmentation_colormap = custom_colormap
                    logger.info(f"Loaded custom colormap with {len(custom_colormap)} entries")
                    # Log some example colors for debugging
                    example_colors = list(custom_colormap.items())[:5]
                    logger.info(f"Example colors: {example_colors}")
                
                if custom_labelmap:
                    # Update the segmentation labelmap with loaded values
                    self.state.segmentation_labelmap = custom_labelmap
                    logger.info(f"Loaded custom labelmap with {len(custom_labelmap)} entries")
                    # Log some example label names for debugging
                    example_labels = list(custom_labelmap.items())[:5]
                    logger.info(f"Example labels: {example_labels}")
            
            # Load the NIfTI file directly with nibabel
            nii_img = nib.load(seg_path)
            
            # Get data as array with explicit cast to int for segmentation labels
            seg_data_orig = np.asarray(nii_img.get_fdata()).astype(np.int32)
            logger.info(f"Original segmentation dimensions: {seg_data_orig.shape}")
            logger.info(f"DICOM data dimensions: {self.state.current_data.shape}")
            
            # Handle dimension mismatch
            if seg_data_orig.shape != self.state.current_data.shape:
                # Check if it's just an axis ordering issue
                dicom_dims = set(self.state.current_data.shape)
                seg_dims = set(seg_data_orig.shape)
                
                # For specific case: (400, 512, 20) vs (20, 512, 400)
                if dicom_dims == seg_dims:
                    logger.info("Dimensions match but in different order, performing axis swapping")
                    
                    # For this specific case, we need to swap axes 0 and 2
                    if (seg_data_orig.shape[0] == self.state.current_data.shape[2] and 
                        seg_data_orig.shape[2] == self.state.current_data.shape[0]):
                        # Swap axes 0 and 2
                        seg_data = np.transpose(seg_data_orig, (2, 1, 0))
                        logger.info(f"After swapping axes: segmentation shape {seg_data.shape}")
                    else:
                        # More general case - find the right permutation
                        perm = []
                        for i in range(3):
                            for j in range(3):
                                if self.state.current_data.shape[i] == seg_data_orig.shape[j]:
                                    perm.append(j)
                                    break
                        
                        if len(perm) == 3:
                            seg_data = np.transpose(seg_data_orig, perm)
                            logger.info(f"General axes permutation {perm}: segmentation shape {seg_data.shape}")
                        else:
                            # Fallback if we can't find a clean permutation
                            seg_data = seg_data_orig
                            logger.warning("Could not find a valid axis permutation. Using original data.")
                else:
                    logger.warning("Dimensions don't match exactly. Using original segmentation data.")
                    seg_data = seg_data_orig
            else:
                # Dimensions already match
                seg_data = seg_data_orig
            
            # Store the segmentation data
            self.state.segmentation_data = seg_data
            self.state.segmentation_loaded = True
              # Update display with segmentation overlay
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                crosshair=None,
                add_orientation_marker=False
            )
            
            # Get the corresponding segmentation slice
            seg_slice = self.state.get_segmentation_slice(
                self.state.current_view, 
                self.state.current_slice_idx            )
            
            if seg_slice is not None:
                # Clear any existing segmentation shapes (fresh load)
                self.segmentation_shapes = {}
                self.user_modified_shapes = {}
                
                # Generate shapes for the current slice
                shapes = self._generate_segmentation_shapes_for_slice(seg_slice, self.state.current_slice_idx)
                
                # Display the base image (without overlay since shapes will handle visualization)
                annotated_image = {
                    "image": make_image_for_gradio(img),
                    "boxes": shapes,  # Include segmentation shapes as annotation boxes
                    "orientation": 0
                }
                  # Check unique labels in segmentation
                unique_labels = np.unique(seg_data)
                label_str = ", ".join(map(str, unique_labels))
                
                # Create status message with label information
                status_msg = f"Segmentation loaded with axis reorientation. Found labels: {label_str}. Converted to {len(shapes)} editable shapes."
                
                # Add label file information if available
                if label_file is not None:
                    label_names = [self.state.segmentation_labelmap.get(label, f"Label_{label}") for label in unique_labels if label != 0]
                    if label_names:
                        status_msg += f" Label names: {', '.join(label_names)}."
                
                return status_msg, annotated_image
            else:
                return "Error: Could not extract segmentation slice", create_empty_annotated_image()
        
        except Exception as e:
            logger.error(f"Error in direct segmentation loading: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.state.segmentation_loaded = False
            self.state.segmentation_data = None
            return f"Error loading segmentation: {str(e)}", create_empty_annotated_image()
    
    @log_exception
    def update_segmentation_opacity(self, opacity):
        """Update the opacity/transparency of the segmentation shapes"""
        # Helper function to create empty annotated image structure
        def create_empty_annotated_image():
            blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
            return {
                "image": blank_image,
                "boxes": [],
                "orientation": 0
            }
        
        if not self.state.segmentation_loaded or self.state.segmentation_data is None:
            return "No segmentation loaded", create_empty_annotated_image()
        
        self.state.segmentation_alpha = opacity
        
        # Update the opacity of all existing segmentation shapes
        # This affects the display transparency in image_annotator
        
        # Re-generate the base image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=None,
            add_orientation_marker=False
        )
        
        # Get the current shapes for the slice
        shapes = self._get_shapes_for_current_slice()
        
        # Create annotated image value for image_annotator
        annotated_image = {
            "image": make_image_for_gradio(img),
            "boxes": shapes,  # Include current segmentation shapes
            "orientation": 0        }
        
        return f"Segmentation opacity updated to {opacity:.1f}", annotated_image
    
    @log_exception
    def clear_segmentation(self):
        """Clear the current segmentation shapes"""
        # Helper function to create empty annotated image structure
        def create_empty_annotated_image():
            blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
            return {
                "image": blank_image,
                "boxes": [],
                "orientation": 0
            }
        
        self.state.reset_segmentation()
        
        # Clear all stored segmentation shapes
        self.segmentation_shapes = {}
        self.user_modified_shapes = {}
        
        if self.state.current_data is None:
            return "No data loaded", create_empty_annotated_image()
        
        # Re-display image without segmentation
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=None,
            add_orientation_marker=False
        )        # Create annotated image value for image_annotator with no shapes
        annotated_image = {
            "image": make_image_for_gradio(img),
            "boxes": [],  # No annotation boxes after clearing
            "orientation": 0
        }
        
        return "Segmentation cleared", annotated_image
    
    @log_exception
    def load_label_file_and_update_annotator(self, label_file):
        """
        Load a .label file and return a new image_annotator with updated labels while preserving existing shapes
        
        Args:
            label_file: Gradio File object containing the .label file
            
        Returns:
            tuple: (status_message, new_image_annotator_with_updated_labels)
        """
        if label_file is None:
            return "No label file selected", self._get_current_annotator_state()
        
        try:
            from ui.viewer_tab import prepare_labels_for_annotator
            from utils.visualization import load_itk_snap_labels
            from gradio_image_annotation import image_annotator
            
            # Get file path from the uploaded file object
            label_path = label_file.name
            logger.info(f"Loading label file for annotator update: {label_path}")
            
            # Load the label file
            custom_colormap, custom_labelmap = load_itk_snap_labels(label_path)
            
            if custom_colormap and custom_labelmap:
                # Update the state with new labels
                self.state.segmentation_colormap.update(custom_colormap)
                self.state.segmentation_labelmap.update(custom_labelmap)
                
                # Clear existing shapes to force regeneration with new labels
                self.segmentation_shapes = {}
                self.user_modified_shapes = {}
                
                # Prepare labels for the image_annotator
                label_list, label_colors = prepare_labels_for_annotator(custom_labelmap, custom_colormap)
                
                logger.info(f"Updated state with {len(label_list)} labels")
                logger.info(f"Labels: {label_list}")
                
                # Get current annotator state (preserving current image and existing shapes)
                current_state = self._get_current_annotator_state()
                
                # Create new annotator with updated labels but preserve existing content
                new_annotator = self._create_annotator_with_new_labels(
                    label_list, label_colors, current_state
                )
                
                status_msg = f"Successfully loaded {len(label_list)} labels from file. "
                status_msg += f"Available labels: {', '.join(label_list)}. "
                status_msg += "The shape editor now uses these label names for new annotations."
                
                return status_msg, new_annotator
            else:
                return "Failed to parse label file", self._get_current_annotator_state()
                
        except Exception as e:
            logger.error(f"Error loading label file: {str(e)}")
            return f"Error loading label file: {str(e)}", self._get_current_annotator_state()
    
    def _get_current_annotator_state(self):
        """Get the current state of the image_annotator to preserve existing content"""
        if self.state.current_data is not None:
            from utils.visualization import display_slice, make_image_for_gradio
            
            # Get current image
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                crosshair=None,
                add_orientation_marker=False
            )
            
            # Get current shapes for the slice (includes both segmentation and user annotations)
            shapes = self._get_shapes_for_current_slice()
            
            # Also check if there are any user annotations stored for this slice
            current_idx = self.state.current_slice_idx
            slice_key = f"{self.state.current_view}_{current_idx}"
            
            # Get any saved annotations from state
            if hasattr(self.state, 'slice_annotations') and slice_key in self.state.slice_annotations:
                saved_annotations = self.state.slice_annotations[slice_key]
                if isinstance(saved_annotations, dict) and 'boxes' in saved_annotations:
                    # Merge segmentation shapes with user annotations
                    all_shapes = shapes + saved_annotations['boxes']
                else:
                    all_shapes = shapes
            else:
                all_shapes = shapes
            
            return {
                "image": make_image_for_gradio(img),
                "boxes": all_shapes,
                "orientation": 0
            }
        else:
            return None
    
    def _create_annotator_with_new_labels(self, label_list, label_colors, current_state):
        """Create a new image_annotator with updated labels while preserving current content"""
        from gradio_image_annotation import image_annotator
        
        return image_annotator(
            value=current_state,
            label="Medical Image Viewer", 
            label_list=label_list,
            label_colors=label_colors,
            box_min_size=10,
            handle_size=8,
            box_thickness=2,
            box_selected_thickness=3,
            boxes_alpha=0.7,
            height=600,
            width=1200,
            interactive=True,
            show_label=True,
            show_download_button=True,
            show_clear_button=True,
            show_remove_button=True,
            use_default_label=False,
            handles_cursor=True,
            image_type="numpy",
            single_box=False,
            disable_edit_boxes=False,            shape_creation_mode="drag",
        )
