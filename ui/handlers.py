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
import cv2
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


def setup_professional_viewport(img_rgb, data_type="unknown", state=None):
    """
    Setup professional viewport system for high-resolution images.
    
    Args:
        img_rgb: Input image as numpy array (H, W, 3)
        data_type: Type of medical data ("mammography", "brain_mri", etc.)
        state: AppState instance to store viewport
        
    Returns:
        tuple: (viewport_img_rgb, use_viewport_system)
    """
    try:
        height, width = img_rgb.shape[:2]
        
        # Determine if image needs DIRECT PIXEL EXTRACTION for lossless quality
        if data_type == "mammography" or width > 2000 or height > 2000:
            # High-resolution image - use DIRECT PIXEL EXTRACTION for guaranteed lossless quality
            logger.info(f"Setting up DIRECT PIXEL EXTRACTOR for {data_type} image ({width}×{height})")
            
            # Import direct pixel extractor
            from utils.direct_pixel_extractor import create_lossless_mammography_extractor
            
            # Create direct pixel extractor with original mammography data
            extractor = create_lossless_mammography_extractor(img_rgb)
            
            # Store extractor in state
            if state:
                state.direct_pixel_extractor = extractor
                state.use_viewport_system = True  # Keep flag for compatibility
                # Reset scaling factors since we're using direct extraction
                state.image_scale_x = 1.0
                state.image_scale_y = 1.0
            
            # Get initial display image (lossless quality, perfect fit)
            display_image = extractor.get_display_image()
            
            logger.info(f"DIRECT PIXEL EXTRACTION active: {width}×{height} → display {display_image.shape[1]}×{display_image.shape[0]} (LOSSLESS)")
            return display_image, True
        else:
            # Normal resolution image - no viewport needed
            logger.info(f"Normal resolution {data_type} image ({width}×{height}) - no special handling needed")
            if state:
                # Clear both viewport and direct pixel extractor
                state.image_viewport = None
                state.direct_pixel_extractor = None
                state.use_viewport_system = False
                state.image_scale_x = 1.0
                state.image_scale_y = 1.0
            return img_rgb, False
            
    except Exception as e:
        logger.error(f"Error setting up viewport: {e}")
        # Fallback to original image
        if state:
            # Clear both viewport and direct pixel extractor
            state.image_viewport = None
            state.direct_pixel_extractor = None
            state.use_viewport_system = False
        return img_rgb, False


class DataLoadingHandlers:
    """Handlers for data loading operations"""
    
    def __init__(self, state: AppState):
        self.state = state
    
    @log_exception
    def load_data(self, file_obj, directory, apply_deidentification=False):
        """Load DICOM data from file or directory"""
        logger.info(f"Loading data: file={file_obj}, directory={directory}, deidentification={apply_deidentification}")
        
        # Reset current data
        self.state.reset_data()
          # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
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
    def clear_data_and_display(self):
        """Clear all data and reset the image annotator display when file is cleared"""
        logger.info("Clearing data and display due to file input clear")
        
        # Reset state
        self.state.reset_data()
        
        # Create a simple black image for the annotator instead of None
        import numpy as np
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Return empty/default values for all components
        empty_annotated_value = {
            "image": black_image,  # Use black image instead of None
            "boxes": [],
            "orientation": 0
        }
        
        return (
            empty_annotated_value,  # image_display
            gr.Dropdown(choices=[], value=None),  # file_browser
            {},  # metadata_display
            gr.Slider(minimum=0, maximum=1, value=0, visible=False),  # slice_slider - fix: max > min to avoid math domain error
            "0/0",  # slice_text
            "x: 0, y: 0, z: 0",  # crosshair_info
            "No data loaded",  # error_display
            500,  # window_level
            1000,  # window_width
            gr.Radio(choices=["Axial", "Sagittal", "Coronal"], value="Axial", label="View Orientation")  # view_selector
        )
    
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
    def load_data_for_plot(self, file_obj, directory, apply_deidentification=False):
        """Load DICOM data from file or directory and return plotly figure for gr.Plot component"""
        logger.info(f"Loading data for plot: file={file_obj}, directory={directory}, deidentification={apply_deidentification}")
        
        # Reset current data
        self.state.reset_data()
        
        # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
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
    def load_data_for_annotator(self, file_obj, directory, apply_deidentification=False):
        """Load DICOM data from file or directory and return AnnotatedImageValue format for image_annotator"""
        logger.info(f"Loading data for annotator: file={file_obj}, directory={directory}, deidentification={apply_deidentification}")
        
        # Reset current data
        self.state.reset_data()
        
        # Determine input source
        if directory:
            path = directory
            self.state.current_directory = directory  # Store the directory path
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            # Store the directory of the single file
            self.state.current_directory = os.path.dirname(path)
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
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
        
        # Setup professional viewport system for high-resolution images
        # This maintains full quality while providing accurate coordinates
        viewport_img_rgb, use_viewport = setup_professional_viewport(
            img_rgb, 
            data_type=self.state.current_data_type,
            state=self.state
        )
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": viewport_img_rgb,
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
            window_width_value,
            # Return traditional brain MRI view orientation
            gr.Radio(choices=["Axial", "Sagittal", "Coronal"], value="Axial", label="View Orientation")
        )

    @log_exception
    def load_medical_data_adaptive(self, file_obj, directory, apply_deidentification=False):
        """
        Adaptive medical data loader that detects data type and loads accordingly
        Supports both brain MRI (sequential slices) and mammography (separate views)
        
        PRIORITY: Single file loading always takes precedence over directory path
        """
        from utils.dicom_utils import detect_medical_data_type, load_mammography_series
        
        logger.info(f"Loading medical data adaptively: file={file_obj}, directory={directory}")
        
        # Reset current data
        self.state.reset_data()
        
        # PRIORITY 1: Single file loading (if file is selected, ignore directory)
        if file_obj is not None and hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
            path = file_obj.name
            self.state.current_directory = os.path.dirname(path)
            logger.info(f"Single file mode: Loading from {path}")
            
            # For single file, detect type from the file's directory (for mammography context)
            detection_path = os.path.dirname(path)
            
            # Check if directory has DICOM files for mammography detection
            try:
                dicom_files_in_dir = [f for f in os.listdir(detection_path) if f.lower().endswith('.dcm')]
                if not dicom_files_in_dir:  # If no DICOM files in directory, use file itself
                    detection_path = path
            except (OSError, PermissionError):
                detection_path = path
                
        # PRIORITY 2: Directory loading (only if no file is selected)
        elif directory and directory.strip() and os.path.exists(directory):
            path = directory
            detection_path = directory
            self.state.current_directory = directory
            logger.info(f"Directory mode: Loading from {path}")
            
        else:
            logger.warning("No valid file or directory provided")
            return (None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No valid data source provided", 500, 1000,
                   gr.Radio(choices=["Axial", "Sagittal", "Coronal"], value="Axial", label="View Orientation"))
        
        # Detect data type from appropriate path
        data_type = detect_medical_data_type(detection_path)
        logger.info(f"Detected data type: {data_type}")
        
        try:
            if data_type == 'mammography':
                # For mammography, always load from directory (even if single file selected)
                # This ensures we get all views in the same folder
                mammography_dir = detection_path if os.path.isdir(detection_path) else os.path.dirname(detection_path)
                logger.info(f"Loading mammography from directory: {mammography_dir}")
                return self._load_mammography_data(mammography_dir, apply_deidentification)
            else:
                # For brain MRI/other data types - consistent loading behavior
                logger.info(f"Loading {data_type} data")
                if file_obj is not None and hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                    # Single file mode - load just this file, ignore directory parameter
                    logger.info(f"Single MR file loading: {file_obj.name}")
                    return self.load_data_for_annotator(file_obj, None, apply_deidentification)
                else:
                    # Directory mode - load all files in directory
                    logger.info(f"Directory MR loading: {directory}")
                    return self.load_data_for_annotator(None, directory, apply_deidentification)
                
        except Exception as e:
            logger.error(f"Error loading medical data: {str(e)}")
            return (None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", f"Error loading data: {str(e)}", 500, 1000, 
                   gr.Radio(choices=["Axial", "Sagittal", "Coronal"], value="Axial", label="View Orientation"))

    def _load_mammography_data(self, path, apply_deidentification=False):
        """
        Load mammography data with separate views
        """
        from utils.dicom_utils import load_mammography_series
        
        logger.info(f"Loading mammography data from: {path}")
        
        # Load mammography views
        view_data, metadata, view_files = load_mammography_series(path, apply_deidentification)
        
        # Store mammography data in state
        self.state.current_data = view_data  # Dict of view_name -> numpy array
        self.state.current_metadata = metadata
        self.state.file_list = list(view_files.values())
        self.state.current_data_type = "mammography"
        self.state.mammography_views = view_data
        self.state.mammography_view_files = view_files
        
        # Set current view to the first available view
        view_names = list(view_data.keys())
        self.state.current_mammography_view = view_names[0]
        current_view_data = view_data[view_names[0]]
        
        # Set shape for the current view
        self.state.current_shape = current_view_data.shape
        
        # Center crosshair for 2D mammography image
        self.state.crosshair_position = (current_view_data.shape[1]//2, current_view_data.shape[0]//2, 0)
        
        # Extract window values from metadata
        window_center = metadata.get('WindowCenter', 1000)
        window_width = metadata.get('WindowWidth', 2000)
        
        if isinstance(window_center, list):
            window_center = window_center[0]
        if isinstance(window_width, list):
            window_width = window_width[0]
        
        # Display the first view
        img = self._display_mammography_view(current_view_data, window_center, window_width)
        
        # Convert to format expected by image_annotator
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Setup professional viewport system for mammography (in _load_mammography_data)
        viewport_img_rgb, use_viewport = setup_professional_viewport(
            img_rgb, 
            data_type="mammography",
            state=self.state
        )
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": viewport_img_rgb,
            "boxes": [],  # Start with empty boxes
            "orientation": 0
        }
        
        # Create view selector dropdown instead of file browser
        crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, view: {self.state.current_mammography_view}"
        
        return (
            annotated_value,
            gr.Dropdown(choices=view_names, value=view_names[0], label="Mammography View"),
            self.state.current_metadata,
            gr.Slider(minimum=0, maximum=1, value=0, step=1, label="View Navigation", visible=False),  # Hide slice slider for mammography - fix math domain error
            f"View: {self.state.current_mammography_view}",
            crosshair_text,
            f"Mammography data loaded: {len(view_names)} view(s)",
            window_center if window_center is not None else 1000,
            window_width if window_width is not None else 2000,
            # Return view orientation update
            gr.Radio(choices=view_names, value=view_names[0], label="Mammography View")
        )
    
    def _display_mammography_view(self, view_data, window_center=None, window_width=None):
        """
        Display a mammography view with proper windowing
        """
        # Apply windowing
        if window_center is not None and window_width is not None:
            lower = window_center - window_width / 2
            upper = window_center + window_width / 2
            img = np.clip(view_data, lower, upper)
            img = (img - lower) / (upper - lower)
        else:
            # Auto-scale
            img = (view_data - np.min(view_data)) / (np.max(view_data) - np.min(view_data))
        
        return img

    @log_exception
    def switch_mammography_view(self, selected_view):
        """
        Switch to a different mammography view
        """
        if (self.state.current_data_type != "mammography" or 
            self.state.mammography_views is None or 
            selected_view not in self.state.mammography_views):
            logger.warning(f"Cannot switch to view {selected_view}")
            return None, "Invalid view", "Invalid view"
        
        # Update current view
        self.state.current_mammography_view = selected_view
        current_view_data = self.state.mammography_views[selected_view]
        
        # Update shape and crosshair
        self.state.current_shape = current_view_data.shape
        self.state.crosshair_position = (current_view_data.shape[1]//2, current_view_data.shape[0]//2, 0)
        
        # Get window values from metadata
        window_center = self.state.current_metadata.get('WindowCenter', 1000)
        window_width = self.state.current_metadata.get('WindowWidth', 2000)
        
        if isinstance(window_center, list):
            window_center = window_center[0]
        if isinstance(window_width, list):
            window_width = window_width[0]
        
        # Display the selected view
        img = self._display_mammography_view(current_view_data, window_center, window_width)
        
        # Convert to format expected by image_annotator
        if len(img.shape) == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        else:
            img_rgb = img
        
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Setup professional viewport system for mammography view switching
        viewport_img_rgb, use_viewport = setup_professional_viewport(
            img_rgb, 
            data_type="mammography",
            state=self.state
        )
        
        # Create AnnotatedImageValue format
        annotated_value = {
            "image": viewport_img_rgb,
            "boxes": [],  # Start with empty boxes
            "orientation": 0
        }
        
        crosshair_text = f"x: {self.state.crosshair_position[0]}, y: {self.state.crosshair_position[1]}, view: {selected_view}"
        status_text = f"Mammography view: {selected_view}"
        
        return annotated_value, status_text, crosshair_text

    @log_exception
    def update_view_orientation_for_data_type(self):
        """
        Update view orientation choices based on current data type
        Returns appropriate choices and default value for the view orientation radio
        """
        if self.state.current_data_type == "mammography":
            # Mammography views
            available_views = list(self.state.mammography_views.keys()) if self.state.mammography_views else []
            return gr.Radio(
                choices=available_views,
                value=available_views[0] if available_views else None,
                label="Mammography View"
            )
        else:
            # Traditional brain MRI orientations
            return gr.Radio(
                choices=["Axial", "Sagittal", "Coronal"],
                value="Axial",
                label="View Orientation"
            )

    @log_exception
    def handle_view_orientation_change(self, selected_orientation):
        """
        Handle view orientation change for both mammography and brain MRI
        """
        if self.state.current_data_type == "mammography":
            # Handle mammography view change
            if (self.state.mammography_views is None or 
                selected_orientation not in self.state.mammography_views):
                logger.warning(f"Mammography view {selected_orientation} not available")
                # Generate Gradio warning and return warning signal
                import gradio as gr
                gr.Warning(f"View '{selected_orientation}' is not present in the loaded mammography data.")
                return None, f"⚠️ View '{selected_orientation}' not found in current dataset", "warning_generated"
            
            # Switch to the selected mammography view
            result = self.switch_mammography_view(selected_orientation)
            if result is None:
                import gradio as gr
                gr.Warning(f"Failed to switch to view '{selected_orientation}'")
                return None, "Failed to switch view", "warning_generated"
                
            annotated_value, status_text, crosshair_text = result
            return annotated_value, status_text, None  # No warning
            
        else:
            # Handle traditional brain MRI orientation change
            # This would delegate to existing view orientation logic
            logger.info(f"Brain MRI orientation change: {selected_orientation}")
            # Return current state (no change implemented yet for brain MRI in this function)
            return None, f"Orientation changed to {selected_orientation}", None

    @log_exception
    def handle_browser_selection_adaptive(self, selected_item):
        """
        Handle browser/view selection that works for both brain MRI files and mammography views
        """
        if self.state.current_data_type == "mammography":
            # Handle mammography view switching
            return self.switch_mammography_view(selected_item)
        else:
            # Handle traditional file browser selection (delegate to existing handler)
            # This would need to be implemented based on existing file selection logic
            logger.info(f"Traditional file selection not implemented in adaptive handler: {selected_item}")
            return None, "File selection for MRI not implemented", "N/A"


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
    def load_data_for_plot(self, file_obj, directory, apply_deidentification=False):
        """Load DICOM data from file or directory and return plotly figure for gr.Plot component"""
        logger.info(f"Loading data for plot: file={file_obj}, directory={directory}, deidentification={apply_deidentification}")
        
        # Reset current data
        self.state.reset_data()
        
        # Determine input source
        if directory:
            path = directory
            self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
        else:
            # Load single file by extension
            path = file_obj.name if file_obj else None
            if not path or not os.path.exists(path):
                return make_slice_figure(np.zeros((100, 100, 3), dtype=np.uint8)), gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded", 500, 1000
            
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.dcm']:
                self.state.current_data, self.state.current_metadata, self.state.file_list = load_dicom_series(path, apply_deidentification=apply_deidentification)
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
