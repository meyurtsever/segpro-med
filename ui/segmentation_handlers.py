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

from utils.visualization import (display_slice, make_slice_figure, overlay_segmentation, 
                                load_itk_snap_labels, segmentation_to_shapes)
from utils.debug_utils import log_exception
from ui.state import AppState

logger = logging.getLogger(__name__)


class SegmentationHandlers:
    """Handlers for segmentation operations"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    @log_exception
    def direct_load_segmentation(self, seg_file, label_file=None):
        """Direct method to load segmentation file with axis swapping for dimension mismatches"""
        if self.state.current_data is None:
            return "Please load a DICOM dataset first", None
        
        if seg_file is None:
            return "No segmentation file provided", None
        
        try:
            # Get file path from the uploaded file object
            seg_path = seg_file.name
            logger.info(f"Direct loading segmentation file: {seg_path}")
            
            # Load label file if provided
            if label_file is not None:
                label_path = label_file.name
                logger.info(f"Loading label file: {label_path}")
                custom_colormap = load_itk_snap_labels(label_path)
                
                if custom_colormap:
                    # Update the segmentation colormap with loaded values
                    self.state.segmentation_colormap = custom_colormap
                    logger.info(f"Loaded custom colormap with {len(custom_colormap)} entries")
            
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
                crosshair=self.state.crosshair_position
            )
            
            # Get the corresponding segmentation slice
            seg_slice = self.state.get_segmentation_slice(
                self.state.current_view, 
                self.state.current_slice_idx
            )
            
            if seg_slice is not None:                # Overlay segmentation on the image
                overlaid_img = overlay_segmentation(
                    img, 
                    seg_slice, 
                    alpha=self.state.segmentation_alpha,
                    colormap=self.state.segmentation_colormap
                )
                
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
                
                dragmode = tool_mapping.get(self.state.current_plot_tool, "pan")
                fig = make_slice_figure(overlaid_img, dragmode=dragmode)
                
                # Check unique labels in segmentation
                unique_labels = np.unique(seg_data)
                label_str = ", ".join(map(str, unique_labels))
                
                return f"Segmentation loaded with axis reorientation. Found labels: {label_str}", fig
            else:
                return "Error: Could not extract segmentation slice", None
        
        except Exception as e:
            logger.error(f"Error in direct segmentation loading: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.state.segmentation_loaded = False
            self.state.segmentation_data = None
            return f"Error loading segmentation: {str(e)}", None
    
    @log_exception
    def update_segmentation_opacity(self, opacity):
        """Update the opacity/transparency of the segmentation overlay"""
        if not self.state.segmentation_loaded or self.state.segmentation_data is None:
            return "No segmentation loaded", None
        
        self.state.segmentation_alpha = opacity
        
        # Re-generate the image with updated opacity
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=self.state.crosshair_position
        )
          # Get the corresponding segmentation slice
        seg_slice = self.state.get_segmentation_slice(
            self.state.current_view, 
            self.state.current_slice_idx
        )
        
        if seg_slice is not None:
            # Overlay segmentation on the image
            overlaid_img = overlay_segmentation(
                img, 
                seg_slice, 
                alpha=self.state.segmentation_alpha,
                colormap=self.state.segmentation_colormap
            )
            
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
            
            dragmode = tool_mapping.get(self.state.current_plot_tool, "pan")
            fig = make_slice_figure(overlaid_img, dragmode=dragmode)
            
            return f"Segmentation opacity updated to {opacity:.1f}", fig
        else:
            return "Error: Could not extract segmentation slice", None
    
    @log_exception
    def clear_segmentation(self):
        """Clear the current segmentation overlay"""
        self.state.reset_segmentation()
        
        if self.state.current_data is None:
            return "No data loaded", None
        
        # Re-display image without segmentation
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=self.state.crosshair_position
        )
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
        
        dragmode = tool_mapping.get(self.state.current_plot_tool, "pan")
        fig = make_slice_figure(img, dragmode=dragmode)
        return "Segmentation cleared", fig
    
    @log_exception
    def convert_segmentation_to_shapes(self):
        """Convert the current segmentation slice to editable Plotly shapes"""
        if not self.state.segmentation_loaded or self.state.segmentation_data is None:
            return "No segmentation loaded", None
        
        # Get the current slice of the segmentation
        seg_slice = self.state.get_segmentation_slice(
            self.state.current_view, 
            self.state.current_slice_idx
        )
        
        if seg_slice is None:
            return "Invalid view orientation or slice index", None
        
        try:
            # Import required libraries for contour finding
            try:
                from skimage import measure
            except ImportError:
                return "Error: scikit-image is required for contour detection. Please install with 'pip install scikit-image'", None
            
            # Display the current slice without segmentation overlay
            img = display_slice(
                self.state.current_data, 
                self.state.current_slice_idx, 
                self.state.current_view,
                crosshair=self.state.crosshair_position
            )
            
            # Create a figure with the current image, specifically set to editing mode
            fig = make_slice_figure(img, dragmode='drawclosedpath')
            
            # Convert segmentation to shapes
            shapes = segmentation_to_shapes(seg_slice)
            
            # Add shapes to the figure
            if shapes:
                fig.update_layout(
                    shapes=shapes,
                    # Ensure shape editing is enabled with the modebar
                    modebar=dict(
                        add=['drawclosedpath', 'eraseshape'],
                        remove=[]
                    ),
                    # Set the dragmode to modify for shape editing
                    dragmode='drawclosedpath',
                    # Enable selection of shapes including from inside
                    clickmode='event+select',
                    # Make sure all shapes can be selected
                    selectdirection='any'
                )
                
                # Directly configure JavaScript event handling through Plotly config
                fig.update_layout(hovermode='closest')
                
                # Update the config to make shapes easier to select
                for i, shape in enumerate(shapes):
                    if 'path' in shape:
                        # Ensure all path shapes have the properties needed for inside selection
                        shape['fillrule'] = 'evenodd'
                        shape['layer'] = 'above'
                
                return f"Converted {len(shapes)} shapes from segmentation. Click anywhere inside or on the edge of a shape to edit it.", fig
            else:
                return "No shapes found in segmentation", fig
        
        except ImportError:
            return "Error: scikit-image is required for contour detection. Please install with 'pip install scikit-image'", None
        except Exception as e:
            import traceback
            logger.error(f"Error converting segmentation to shapes: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error converting segmentation to shapes: {str(e)}", None
