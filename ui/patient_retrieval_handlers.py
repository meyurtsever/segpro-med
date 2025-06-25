"""
Patient Retrieval Handlers

This module handles the patient search and selection functionality
for the Patient Retrieval System.
"""

import os
import gradio as gr
from typing import List, Tuple, Optional
from utils.patient_retrieval import patient_retrieval
from utils.debug_utils import logger


class PatientRetrievalHandlers:
    """Handlers for patient retrieval system functionality"""
    
    def __init__(self, app_state):
        """
        Initialize handlers with application state
        
        Args:
            app_state: Application state object
        """
        self.state = app_state
    
    def search_patients(self, query: str) -> dict:
        """
        Search for patients based on query string
        
        Args:
            query: Search query string
            
        Returns:
            Dropdown update with matching patients
        """
        try:
            if not query or len(query.strip()) < 1:
                # Return empty choices with empty value and allow custom value temporarily
                return gr.update(choices=[], value=None, allow_custom_value=True)
            
            # Get matching patients
            matches = patient_retrieval.search_patients(query.strip())
            
            # Return the matches for dropdown and clear selection
            return gr.update(choices=matches, value=None, allow_custom_value=False)
            
        except Exception as e:
            logger.error(f"Error searching patients: {e}")
            return gr.update(choices=[], value=None, allow_custom_value=True)
    
    def load_selected_patient(self, selected_patient: str) -> Tuple[
        Optional[dict],  # image_display
        List[str],       # file_browser choices
        dict,            # metadata_display
        dict,            # slice_slider
        str,             # slice_text
        str,             # crosshair_info
        str,             # error_display
        int,             # window_level
        int,             # window_width
        str              # segmentation_status
    ]:
        """
        Load the selected patient's DICOM data and segmentation
        
        Args:
            selected_patient: Selected patient display name
            
        Returns:
            Tuple of updated UI components
        """
        try:
            if not selected_patient:
                return (None, [], {}, gr.update(), "", "", "No patient selected", 500, 1000, "No segmentation loaded")
            
            # Get patient information
            patient_info = patient_retrieval.get_patient_info(selected_patient)
            if not patient_info:
                error_msg = f"Patient information not found for: {selected_patient}"
                logger.error(error_msg)
                return (None, [], {}, gr.update(), "", "", error_msg, 500, 1000, "No segmentation loaded")
            
            flair_path = patient_info['flair_path']
            segmentation_path = patient_info['segmentation_path']
            
            if not os.path.exists(flair_path):
                error_msg = f"FLAIR directory not found: {flair_path}"
                logger.error(error_msg)
                return (None, [], {}, gr.update(), "", "", error_msg, 500, 1000, "No segmentation loaded")
            
            # Import the data loading functionality
            from ui.handlers import DataLoadingHandlers
            
            # Create a temporary data loading handler instance
            data_handler = DataLoadingHandlers(self.state)
            
            # Load DICOM data using existing handler
            # We need to simulate loading from directory
            dummy_file = None  # No file input
            dir_input = flair_path  # Use FLAIR path as directory
            
            result = data_handler.load_data_for_annotator(dummy_file, dir_input)
            
            # Unpack the result from data loading
            (image_display, file_browser, metadata_display, slice_slider, 
             slice_text, crosshair_info, error_display, window_level, window_width) = result
            
            # Clear any existing annotations from image_annotator before loading segmentation
            if image_display is not None and hasattr(image_display, 'value'):
                # Clear annotations by updating the image_annotator value
                if isinstance(image_display.value, dict) and 'annotations' in image_display.value:
                    image_display.value['annotations'] = []
            
            # Handle segmentation loading - DO NOT AUTO-LOAD, provide status only
            segmentation_status = "No segmentation loaded"
            updated_image_display = image_display
            
            if segmentation_path and os.path.exists(segmentation_path):
                segmentation_status = "Segmentation available - click 'Load Segmentation' to view"
            else:
                segmentation_status = "No segmentation file found for this patient"
            
            # Update status with patient info
            success_msg = f"Loaded patient: {selected_patient}"
            if segmentation_path and os.path.exists(segmentation_path):
                success_msg += " (segmentation available)"
            else:
                success_msg += " (no segmentation)"
            
            logger.info(success_msg)
            
            return (
                updated_image_display,  # image_display
                file_browser,           # file_browser
                metadata_display,       # metadata_display
                slice_slider,           # slice_slider
                slice_text,             # slice_text
                crosshair_info,         # crosshair_info
                success_msg,            # error_display (status)
                window_level,           # window_level
                window_width,           # window_width
                segmentation_status     # segmentation_status
            )
            
        except Exception as e:
            error_msg = f"Failed to load patient {selected_patient}: {str(e)}"
            logger.error(error_msg)
            return (None, [], {}, gr.update(), "", "", error_msg, 500, 1000, "No segmentation loaded")
    
    def clear_search(self) -> Tuple[str, dict]:
        """
        Clear the patient search input and dropdown
        
        Returns:
            Tuple of (cleared_input, cleared_dropdown)
        """
        return "", gr.update(choices=[], value=None, allow_custom_value=True)
    
    def load_manual_segmentation(self, selected_patient: str) -> Tuple[str, Optional[dict]]:
        """
        Manually load segmentation for the currently selected patient
        
        Args:
            selected_patient: Selected patient display name
            
        Returns:
            Tuple of (status_message, updated_image_display)
        """
        try:
            if not selected_patient:
                return "No patient selected", None
                
            # Get patient information
            patient_info = patient_retrieval.get_patient_info(selected_patient)
            if not patient_info:
                return f"Patient information not found for: {selected_patient}", None
            
            segmentation_path = patient_info['segmentation_path']
            
            if not segmentation_path or not os.path.exists(segmentation_path):
                return f"No segmentation file found for patient: {selected_patient}", None
            
            # Import segmentation handlers
            from ui.segmentation_handlers import SegmentationHandlers
            
            # Create temporary segmentation handler
            seg_handler = SegmentationHandlers(self.state)
            
            # Create a dummy file object for the segmentation file
            class DummyFile:
                def __init__(self, path):
                    self.name = path
            
            dummy_seg_file = DummyFile(segmentation_path)
            dummy_label_file = None
            
            # Load segmentation
            seg_result = seg_handler.direct_load_segmentation(dummy_seg_file, dummy_label_file)
            status_message, updated_image_display = seg_result
            
            logger.info(f"Manually loaded segmentation from: {segmentation_path}")
            
            return updated_image_display, status_message
            
        except Exception as e:
            error_msg = f"Failed to load segmentation for {selected_patient}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def clear_manual_segmentation(self) -> Tuple[str, Optional[dict]]:
        """
        Clear the currently loaded segmentation overlay
        
        Returns:
            Tuple of (status_message, updated_image_display)
        """
        try:
            # Import segmentation handlers
            from ui.segmentation_handlers import SegmentationHandlers
            
            # Create temporary segmentation handler
            seg_handler = SegmentationHandlers(self.state)
            
            # Clear segmentation
            clear_result = seg_handler.clear_segmentation()
            status_message, updated_image_display = clear_result
            
            logger.info("Manually cleared segmentation overlay")
            
            return updated_image_display, status_message
            
        except Exception as e:
            error_msg = f"Failed to clear segmentation: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
