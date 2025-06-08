"""
SegMed-Pro Application State Management

This module provides a central application state that can be accessed by different
components of the application.
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

# Add utils to path for imports
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.dicom_utils import configure_dicom_handlers

logger = logging.getLogger(__name__)

class AppState:
    """Central application state for the SegMed-Pro application"""
    
    def __init__(self):
        """Initialize app state with default values"""
        # Set up DICOM handlers
        self.dicom_handlers = configure_dicom_handlers()
        logger.info(f"Initialized DICOM handlers: {len(self.dicom_handlers)} handlers available")
        
        # Core data state
        self.current_data: Optional[np.ndarray] = None
        self.current_data_type: Optional[str] = None  # "dicom" or "nifti"
        self.current_metadata: Optional[Dict[str, Any]] = None
        self.current_directory: Optional[str] = None
        self.file_list: List[str] = []
        self.current_slice_idx: int = 0
        self.current_view: str = "axial"  # axial, sagittal, coronal
        self.crosshair_position: Optional[Tuple[int, int, int]] = None
        self.current_shape: Tuple[int, int, int] = (0, 0, 0)
        
        # Plot tool state
        self.current_plot_tool: str = "drawclosedpath"
          # Segmentation state
        self.segmentation_data: Optional[np.ndarray] = None
        self.segmentation_loaded: bool = False
        self.segmentation_alpha: float = 0.5  # Default transparency
        self.segmentation_colormap: Dict[int, List[int]] = {
            0: [0, 0, 0],      # Background (transparent)
            1: [255, 0, 0],    # Label 1 (Red)
            2: [0, 255, 0],    # Label 2 (Green)
            3: [0, 0, 255],    # Label 3 (Blue)
            4: [255, 255, 0],  # Label 4 (Yellow)
            5: [0, 255, 255],  # Label 5 (Cyan)
            6: [255, 0, 255],  # Label 6 (Magenta)
            7: [255, 165, 0],  # Label 7 (Orange)
            8: [128, 0, 128]   # Label 8 (Purple)
        }
        
        # Tab-specific segmentation display flags (default false for editor tab)
        self.show_segmentation_in_viewer: bool = True   # Default true for viewer tab
        self.show_segmentation_in_editor: bool = False  # Default false for editor tab as requested
        
        # Custom components state
        self.custom_components_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "custom_components"
        )
        self.has_plot_tools = self._check_plot_tools()
    
    def _check_plot_tools(self) -> bool:
        """Check if custom plot tools are available"""
        if os.path.exists(self.custom_components_path):
            sys.path.append(self.custom_components_path)
            try:
                from custom_components.plot_tools.component import PlotTools
                return True
            except (ImportError, ModuleNotFoundError):
                return False
        return False
    
    def reset_data(self) -> None:
        """Reset all data-related state"""
        self.current_data = None
        self.current_data_type = None
        self.current_metadata = None
        self.current_directory = None
        self.file_list = []
        self.current_slice_idx = 0
        self.crosshair_position = None
        self.current_shape = (0, 0, 0)
    def reset_segmentation(self) -> None:
        """Reset segmentation-related state"""
        self.segmentation_data = None
        self.segmentation_loaded = False
        self.segmentation_alpha = 0.5
        # Reset display flags to defaults
        self.show_segmentation_in_viewer = True
        self.show_segmentation_in_editor = False
    
    def set_data(self, data: np.ndarray, data_type: str, metadata: Dict[str, Any], 
                 file_list: List[str]) -> None:
        """Set the main data and associated metadata"""
        self.current_data = data
        self.current_data_type = data_type
        self.current_metadata = metadata
        self.file_list = file_list
        self.current_shape = data.shape
        # Center crosshair
        self.crosshair_position = (
            data.shape[2] // 2, 
            data.shape[1] // 2, 
            data.shape[0] // 2
        )
        self.current_slice_idx = 0
        self.current_view = "axial"
    
    def update_slice_position(self, slice_idx: int, view: str) -> None:
        """Update current slice and view"""
        self.current_slice_idx = slice_idx
        self.current_view = view.lower()
        
        # Update crosshair position based on view
        if self.current_view == "axial":
            x, y, _ = self.crosshair_position
            self.crosshair_position = (x, y, slice_idx)
        elif self.current_view == "sagittal":
            _, y, z = self.crosshair_position
            self.crosshair_position = (slice_idx, y, z)
        elif self.current_view == "coronal":
            x, _, z = self.crosshair_position
            self.crosshair_position = (x, slice_idx, z)
    
    def get_max_slice_for_view(self, view: str) -> int:
        """Get maximum slice index for the given view"""
        if self.current_data is None:
            return 0
        
        view = view.lower()
        if view == "axial":
            return self.current_shape[0] - 1
        elif view == "sagittal":
            return self.current_shape[2] - 1
        elif view == "coronal":
            return self.current_shape[1] - 1
        return 0
    
    def get_segmentation_slice(self, view: str, slice_idx: int) -> Optional[np.ndarray]:
        """Get the segmentation slice for the current view and slice index"""
        if not self.segmentation_loaded or self.segmentation_data is None:
            return None
        
        view = view.lower()
        try:
            if view == 'axial':
                return self.segmentation_data[slice_idx, :, :]
            elif view == 'sagittal':
                return self.segmentation_data[:, :, slice_idx]
            elif view == 'coronal':
                return self.segmentation_data[:, slice_idx, :]
        except IndexError:
            logger.error(f"Index out of bounds for segmentation slice: {slice_idx}")
            return None
        
        return None

# Singleton instance
_app_state = None

def get_app_state():
    """Get the singleton app state instance"""
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state
