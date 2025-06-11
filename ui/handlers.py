"""
SegMed-Pro Event Handlers

This module contains all the event handling logic for the SegMed-Pro application,
separated from the UI components for better maintainability.
"""

import os
import sys
import logging
import gradio as gr
import numpy as np
from typing import Optional, Tuple, Any, List

# Add utils to path for imports
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.dicom_utils import load_dicom_series, get_dicom_metadata
from utils.debug_utils import debug_dicom_loading
from utils.nifti_utils import load_nifti_file
from utils.visualization import (display_slice, make_slice_figure, overlay_segmentation, 
                                      make_image_for_gradio, segmentation_to_shapes)
from utils.conversion import dicom_to_nifti, nifti_to_mat, dicom_to_mat, nifti_to_png
from utils.debug_utils import log_exception
from ui.state import AppState

logger = logging.getLogger(__name__)


class DataLoadingHandlers:
    """Handlers for data loading operations"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    @log_exception
    def load_data(self, file_obj, directory):
        """Load DICOM data from file or directory"""
        logger.info(f"Loading data: file={file_obj}, directory={directory}")
        
        # Reset current data
        self.state.reset_data()
          # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
            elif ext in ['.nii', '.gz']:
                img3d, meta = load_nifti_file(path)
                self.state.current_data = img3d
                self.state.current_metadata = meta
                self.state.file_list = [path]
            elif ext == '.mat':
                # Placeholder for MAT loading
                self.state.current_data = None
                self.state.current_metadata = {}
                self.state.file_list = [path]
            else:
                return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", f"Unsupported file type: {ext}", 500, 1000
        
        self.state.current_data_type = "dicom"
        shape = self.state.current_data.shape
        self.state.current_shape = shape
        # Center crosshair
        self.state.crosshair_position = (shape[2]//2, shape[1]//2, shape[0]//2)
        
        # Prepare file names
        file_names = [os.path.basename(f) for f in self.state.file_list]
        
        # Slider range and visibility
        slider_min, slider_max = 0, shape[0] - 1
        self.state.current_slice_idx = 0
        visible_flag = slider_max > 0
        
        # Extract window values from metadata
        window_center = None
        window_width = None
        
        if self.state.current_metadata and 'WindowCenter' in self.state.current_metadata:
            window_center = self.state.current_metadata['WindowCenter']
            if isinstance(window_center, list):
                window_center = window_center[0]
            logger.info(f"Using WindowCenter from metadata: {window_center}")
        
        if self.state.current_metadata and 'WindowWidth' in self.state.current_metadata:
            window_width = self.state.current_metadata['WindowWidth']
            if isinstance(window_width, list):
                window_width = window_width[0]
            logger.info(f"Using WindowWidth from metadata: {window_width}")
          # Initial image
        img = display_slice(
            self.state.current_data, 
            0, 
            self.state.current_view.lower(), 
            window_level=window_center, 
            window_width=window_width,
            crosshair=self.state.crosshair_position
        )
        
        # Convert to PIL Image for gr.Image
        pil_image = make_image_for_gradio(img)
        crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, z: {self.state.crosshair_position[2]}"
        
        # Default window values
        window_level_value = window_center if window_center is not None else 500
        window_width_value = window_width if window_width is not None else 1000
        
        return (
            pil_image,
            gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None),
            self.state.current_metadata,
            gr.Slider(minimum=slider_min, maximum=slider_max, value=0, step=1, label="Slice Navigation", visible=visible_flag),
            f"0/{slider_max}",
            crosshair_text,
            f"DICOM data loaded: {len(self.state.file_list)} slice(s)",
            window_level_value,
            window_width_value
        )
    
    @log_exception
    def reset_directory(self):
        """Clear the directory textbox"""
        return ""
    
    @log_exception
    def debug_selected_file(self, file_obj, directory):
        """Debug the currently selected file"""
        file_path = None
        
        if file_obj is not None:
            file_path = file_obj.name
        elif directory and os.path.exists(directory) and not os.path.isdir(directory):
            file_path = directory
        
        if file_path and file_path.lower().endswith('.dcm'):
            logger.info(f"Debugging DICOM file: {file_path}")
            result = debug_dicom_loading(file_path)
            return f"Debug complete. Check log file for details: {os.path.abspath('utils/segmed_pro_debug.log')}"
        else:
            return "Please select a valid DICOM (.dcm) file to debug"
    
    @log_exception
    def load_data_for_plot(self, file_obj, directory):
        """Load DICOM data from file or directory and return plotly figure for gr.Plot component"""
        logger.info(f"Loading data for plot: file={file_obj}, directory={directory}")
        
        # Reset current data
        self.state.reset_data()
        
        # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
            elif ext in ['.nii', '.gz']:
                img3d, meta = load_nifti_file(path)
                self.state.current_data = img3d
                self.state.current_metadata = meta
                self.state.file_list = [path]
            elif ext == '.mat':
                # Placeholder for MAT loading
                self.state.current_data = None
                self.state.current_metadata = {}
                self.state.file_list = [path]
            else:
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", f"Unsupported file type: {ext}", 500, 1000

        self.state.current_data_type = "dicom"
        shape = self.state.current_data.shape
        self.state.current_shape = shape
        # Center crosshair
        self.state.crosshair_position = (shape[2]//2, shape[1]//2, shape[0]//2)
        
        # Prepare file names
        file_names = [os.path.basename(f) for f in self.state.file_list]
        
        # Slider range and visibility
        slider_min, slider_max = 0, shape[0] - 1
        self.state.current_slice_idx = 0
        visible_flag = slider_max > 0
        
        # Extract window values from metadata
        window_center = None
        window_width = None
        
        if self.state.current_metadata and 'WindowCenter' in self.state.current_metadata:
            window_center = self.state.current_metadata['WindowCenter']
            if isinstance(window_center, list):
                window_center = window_center[0]
            logger.info(f"Using WindowCenter from metadata: {window_center}")
        
        if self.state.current_metadata and 'WindowWidth' in self.state.current_metadata:
            window_width = self.state.current_metadata['WindowWidth']
            if isinstance(window_width, list):
                window_width = window_width[0]
            logger.info(f"Using WindowWidth from metadata: {window_width}")
        
        # Initial image
        img = display_slice(
            self.state.current_data, 
            0, 
            self.state.current_view.lower(), 
            window_level=window_center, 
            window_width=window_width,
            crosshair=self.state.crosshair_position
        )
        
        # Create plotly figure for gr.Plot component
        plotly_fig = make_slice_figure(img)
        crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, z: {self.state.crosshair_position[2]}"
        
        # Default window values
        window_level_value = window_center if window_center is not None else 500
        window_width_value = window_width if window_width is not None else 1000
        
        return (
            plotly_fig,
            gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None),
            self.state.current_metadata,
            gr.Slider(minimum=slider_min, maximum=slider_max, value=0, step=1, label="Slice Navigation", visible=visible_flag),
            f"0/{slider_max}",
            crosshair_text,
            f"DICOM data loaded: {len(self.state.file_list)} slice(s)",
            window_level_value,
            window_width_value
        )
    
    @log_exception
    def load_data_for_annotator(self, file_obj, directory):
        """Load DICOM data from file or directory and return AnnotatedImageValue format for image_annotator"""
        logger.info(f"Loading data for annotator: file={file_obj}, directory={directory}")
        
        # Reset current data
        self.state.reset_data()
        
        # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
            elif ext in ['.nii', '.gz']:
                img3d, meta = load_nifti_file(path)
                self.state.current_data = img3d
                self.state.current_metadata = meta
                self.state.file_list = [path]
            elif ext == '.mat':
                # Placeholder for MAT loading
                self.state.current_data = None
                self.state.current_metadata = {}
                self.state.file_list = [path]
            else:
                return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", f"Unsupported file type: {ext}", 500, 1000

        self.state.current_data_type = "dicom"
        shape = self.state.current_data.shape
        self.state.current_shape = shape
        # Center crosshair
        self.state.crosshair_position = (shape[2]//2, shape[1]//2, shape[0]//2)
        
        # Prepare file names
        file_names = [os.path.basename(f) for f in self.state.file_list]
        
        # Slider range and visibility
        slider_min, slider_max = 0, shape[0] - 1
        self.state.current_slice_idx = 0
        visible_flag = slider_max > 0
        
        # Extract window values from metadata
        window_center = None
        window_width = None
        
        if self.state.current_metadata and 'WindowCenter' in self.state.current_metadata:
            window_center = self.state.current_metadata['WindowCenter']
            if isinstance(window_center, list):
                window_center = window_center[0]
            logger.info(f"Using WindowCenter from metadata: {window_center}")
        
        if self.state.current_metadata and 'WindowWidth' in self.state.current_metadata:
            window_width = self.state.current_metadata['WindowWidth']
            if isinstance(window_width, list):
                window_width = window_width[0]
            logger.info(f"Using WindowWidth from metadata: {window_width}")
        
        # Initial image
        img = display_slice(
            self.state.current_data, 
            0, 
            self.state.current_view.lower(), 
            window_level=window_center, 
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
            "boxes": [],  # Start with empty boxes
            "orientation": 0
        }
        
        crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, z: {self.state.crosshair_position[2]}"
        
        # Default window values
        window_level_value = window_center if window_center is not None else 500
        window_width_value = window_width if window_width is not None else 1000
        
        return (
            annotated_value,
            gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None),
            self.state.current_metadata,
            gr.Slider(minimum=slider_min, maximum=slider_max, value=0, step=1, label="Slice Navigation", visible=visible_flag),
            f"0/{slider_max}",
            crosshair_text,
            f"DICOM data loaded: {len(self.state.file_list)} slice(s)",
            window_level_value,
            window_width_value
        )


class ViewerHandlers:
    """Handlers for viewer operations"""
    
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
              # Apply segmentation overlay if segmentation is loaded
            if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
          # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
          # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
              # Apply segmentation overlay if loaded
            if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
                crosshair=self.state.crosshair_position
            )
            
            # Apply segmentation overlay if segmentation is loaded
            if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
            return (fig, f"{self.state.current_slice_idx -1}/{self.state.get_max_slice_for_view(self.state.current_view) -1}", 
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
            crosshair=self.state.crosshair_position
        )
        
        # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
          # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
              # Apply segmentation overlay if loaded
            if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
                crosshair=self.state.crosshair_position
            )
            
            # Apply segmentation overlay if segmentation is loaded
            if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
            return (fig, f"{self.state.current_slice_idx -1}/{self.state.get_max_slice_for_view(self.state.current_view) -1}", 
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
            crosshair=self.state.crosshair_position
        )
        
        # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
        
        return (
            gr.Slider(minimum=slider_min, maximum=slider_max, value=self.state.current_slice_idx), 
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
            crosshair=self.state.crosshair_position
        )
        
        # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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
            
            # Apply segmentation overlay if loaded
            if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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


class PlotToolHandlers:
    """Handlers for plot tool operations"""
    
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
        
        # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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

    def update_plot_tool_for_plot(self, tool_name):
        """Update plot tool for gr.Plot component (returns plotly figure with updated dragmode)"""
        if self.state.current_data is None:
            return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # Store the selected tool for reference
        self.state.current_plot_tool = tool_name
        
        # Generate the current slice image
        img = display_slice(
            self.state.current_data, 
            self.state.current_slice_idx, 
            self.state.current_view,
            crosshair=self.state.crosshair_position
        )
        
        # Apply segmentation overlay if loaded
        if self.state.segmentation_loaded and self.state.segmentation_data is not None:
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


class ConversionHandlers:
    """Handlers for format conversion operations"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    def convert_files(self, file_obj, directory, conv_type, out_dir):
        """Convert files between formats"""
        try:
            input_path = None
            
            # Determine input source
            if file_obj is not None:
                input_path = file_obj.name
            elif directory and os.path.exists(directory):
                input_path = directory
            else:
                return "No valid input file or directory provided.", None
            
            # Ensure output directory exists
            if not out_dir:
                out_dir = os.path.dirname(input_path)
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
            # Determine output path
            base_name = os.path.basename(input_path)
            
            if conv_type == "DICOM to NIFTI":
                output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}.nii.gz")
                result_path = dicom_to_nifti(input_path, output_path)
            elif conv_type == "NIFTI to MAT":
                output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}.mat")
                result_path = nifti_to_mat(input_path, output_path)
            elif conv_type == "DICOM to MAT":
                output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}.mat")
                result_path = dicom_to_mat(input_path, output_path)
            elif conv_type == "NIFTI to PNG":
                output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}_png")
                result_path = nifti_to_png(input_path, output_path)
            
            # Create success message
            success_msg = f"Conversion successful: {conv_type}\nOutput: {result_path}"
            
            # Return output file if it's a single file
            output_file = None
            if os.path.isfile(result_path):
                output_file = result_path
            return success_msg, output_file
        
        except Exception as e:
            return f"Error during conversion: {str(e)}", None

    @log_exception
    def load_data_for_plot(self, file_obj, directory):
        """Load DICOM data from file or directory and return plotly figure for gr.Plot component"""
        logger.info(f"Loading data for plot: file={file_obj}, directory={directory}")
        
        # Reset current data
        self.state.reset_data()
        
        # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path)
            elif ext in ['.nii', '.gz']:
                img3d, meta = load_nifti_file(path)
                self.state.current_data = img3d
                self.state.current_metadata = meta
                self.state.file_list = [path]
            elif ext == '.mat':
                # Placeholder for MAT loading
                self.state.current_data = None
                self.state.current_metadata = {}
                self.state.file_list = [path]
            else:
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", f"Unsupported file type: {ext}", 500, 1000

        self.state.current_data_type = "dicom"
        shape = self.state.current_data.shape
        self.state.current_shape = shape
        # Center crosshair
        self.state.crosshair_position = (shape[2]//2, shape[1]//2, shape[0]//2)
        
        # Prepare file names
        file_names = [os.path.basename(f) for f in self.state.file_list]
        
        # Slider range and visibility
        slider_min, slider_max = 0, shape[0] - 1
        self.state.current_slice_idx = 0
        visible_flag = slider_max > 0
        
        # Extract window values from metadata
        window_center = None
        window_width = None
        
        if self.state.current_metadata and 'WindowCenter' in self.state.current_metadata:
            window_center = self.state.current_metadata['WindowCenter']
            if isinstance(window_center, list):
                window_center = window_center[0]
            logger.info(f"Using WindowCenter from metadata: {window_center}")
        
        if self.state.current_metadata and 'WindowWidth' in self.state.current_metadata:
            window_width = self.state.current_metadata['WindowWidth']
            if isinstance(window_width, list):
                window_width = window_width[0]
            logger.info(f"Using WindowWidth from metadata: {window_width}")
        
        # Initial image
        img = display_slice(
            self.state.current_data, 
            0, 
            self.state.current_view.lower(), 
            window_level=window_center, 
            window_width=window_width,
            crosshair=self.state.crosshair_position
        )
        
        # Create plotly figure for gr.Plot component
        plotly_fig = make_slice_figure(img)
        crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, z: {self.state.crosshair_position[2]}"
        
        # Default window values
        window_level_value = window_center if window_center is not None else 500
        window_width_value = window_width if window_width is not None else 1000
        
        return (
            plotly_fig,
            gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None),
            self.state.current_metadata,
            gr.Slider(minimum=slider_min, maximum=slider_max, value=0, step=1, label="Slice Navigation", visible=visible_flag),
            f"0/{slider_max}",
            crosshair_text,
            f"DICOM data loaded: {len(self.state.file_list)} slice(s)",
            window_level_value,
            window_width_value
        )
