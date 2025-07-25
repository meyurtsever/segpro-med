"""
MedGemma-4B Handler for SegMed-Pro

This module provides functionality to integrate MedGemma-4B visual language model
for medical image analysis in the Editor tab using a persistent service for fast inference.
MedGemma-4B is Google's specialized medical model optimized for medical imaging tasks.
"""

import logging
import os
import tempfile
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# Add models directory to path for import
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
if models_dir not in sys.path:
    sys.path.append(models_dir)

# Add medgemma directory to path
medgemma_dir = os.path.join(models_dir, "medgemma")
if medgemma_dir not in sys.path:
    sys.path.append(medgemma_dir)

try:
    from medgemma.medgemma_service import get_service, cleanup_service, get_medical_prompt
except ImportError:
    # Fallback if service is not available
    logger = logging.getLogger(__name__)
    logger.warning("MedGemma service not available - check installation")
    get_service = None
    cleanup_service = None
    get_medical_prompt = None

logger = logging.getLogger(__name__)


class MedGemmaHandlers:
    """Handlers for MedGemma-4B visual language model operations"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self._service = None
        
    def _get_service(self):
        """Get or initialize the MedGemma service"""
        if self._service is None and get_service is not None:
            try:
                self._service = get_service(device="auto")
                logger.info("MedGemma service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MedGemma service: {e}")
                self._service = None
        return self._service
        
    def run_vlm_inference(self, image_annotator_value: Optional[dict], identify_anomalies: bool = True, describe_slice: bool = False) -> str:
        """
        Run MedGemma VLM inference on the current image from image_annotator
        
        Args:
            image_annotator_value: Value from the image_annotator component containing image data
            identify_anomalies: Whether to focus on identifying anomalies
            describe_slice: Whether to provide general description
            
        Returns:
            VLM analysis result as string
        """
        try:
            # Check if service is available
            service = self._get_service()
            if service is None:
                return "MedGemma service not available. Please check installation and try again."
            
            # Validate input
            if not image_annotator_value:
                return "No image data available. Please load a DICOM or medical image first."
            
            # Extract image from image_annotator
            image_data = self._extract_image_from_annotator(image_annotator_value)
            if image_data is None:
                return "Could not extract image data from the annotator. Please ensure an image is loaded."
            
            # Build prompt based on user selections
            prompt = self._build_prompt(identify_anomalies, describe_slice)
            logger.info(f"Running MedGemma inference with prompt type: {self._get_prompt_type(identify_anomalies, describe_slice)}")
            
            # Run inference
            try:
                response = service.generate_response(
                    image=image_data,
                    prompt=prompt,
                    max_new_tokens=256
                )
                
                if response and len(response.strip()) > 0:
                    return f"MedGemma-4B Analysis:\n\n{response}"
                else:
                    return "MedGemma generated an empty response. The image might not be suitable for analysis or the model encountered an issue."
                    
            except Exception as e:
                logger.error(f"MedGemma inference failed: {e}")
                return f"MedGemma inference failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in MedGemma VLM inference: {e}")
            return f"Error during MedGemma analysis: {str(e)}"
    
    def suggest_labels_for_annotations(self, image_annotator_value: Optional[dict]) -> str:
        """
        Use MedGemma to suggest labels for anatomical/pathological structures in the image
        
        Args:
            image_annotator_value: Value from the image_annotator component
            
        Returns:
            Suggested labels as a comma-separated string
        """
        try:
            # Check if service is available
            service = self._get_service()
            if service is None:
                return "MedGemma service not available for label suggestions."
            
            # Validate input
            if not image_annotator_value:
                return "No image data available for label suggestions."
            
            # Extract image from image_annotator
            image_data = self._extract_image_from_annotator(image_annotator_value)
            if image_data is None:
                return "Could not extract image for label suggestions."
            
            # Use label suggestion prompt
            if get_medical_prompt:
                prompt = get_medical_prompt("suggest_labels")
            else:
                prompt = ("This is a brain MRI slice. Please list the anatomical or pathological structures that are visible. "
                         "Return a comma-separated list of possible labels (e.g., eye, lvent, tvent, tumor, lesion, etc.).")
            
            logger.info("Running MedGemma label suggestion inference")
            
            # Run inference
            try:
                response = service.generate_response(
                    image=image_data,
                    prompt=prompt,
                    max_new_tokens=128  # Shorter for label suggestions
                )
                
                if response and len(response.strip()) > 0:
                    return f"MedGemma Label Suggestions:\n{response}"
                else:
                    return "MedGemma could not generate label suggestions for this image."
                    
            except Exception as e:
                logger.error(f"MedGemma label suggestion failed: {e}")
                return f"Label suggestion failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in MedGemma label suggestion: {e}")
            return f"Error during label suggestion: {str(e)}"
    
    def _extract_image_from_annotator(self, annotator_value: dict) -> Optional[np.ndarray]:
        """
        Extract image data from the image_annotator component value
        
        Args:
            annotator_value: Dictionary containing image_annotator data
            
        Returns:
            numpy array of image data or None if extraction fails
        """
        try:
            if not annotator_value:
                return None
            
            # Debug: Log what keys are available in the annotator value
            logger.debug(f"Annotator value keys: {list(annotator_value.keys()) if annotator_value else 'None'}")
            
            # Try to get the image from the annotator (standard key is "image")
            image_data = annotator_value.get("image")
            if image_data is None:
                logger.warning("No image found in annotator value")
                # Try alternative keys as fallback
                image_data = annotator_value.get("background")
                if image_data is None:
                    return None
                logger.info("Found image data using 'background' key as fallback")
            
            # Debug: Log image data type and shape if possible
            if isinstance(image_data, np.ndarray):
                logger.debug(f"Image data type: {image_data.dtype}, shape: {image_data.shape}")
            else:
                logger.debug(f"Image data type: {type(image_data)}")
            
            # Handle different image formats
            if isinstance(image_data, str):
                # If it's a file path, load the image
                if os.path.exists(image_data):
                    image = Image.open(image_data)
                    return np.array(image)
                else:
                    logger.warning(f"Image path does not exist: {image_data}")
                    return None
            elif isinstance(image_data, np.ndarray):
                # Direct numpy array - this is the expected format
                return image_data
            elif hasattr(image_data, 'array'):
                # PIL Image or similar
                return np.array(image_data)
            else:
                logger.warning(f"Unknown image format: {type(image_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting image from annotator: {e}")
            return None
    
    def _build_prompt(self, identify_anomalies: bool, describe_slice: bool) -> str:
        """
        Build the appropriate prompt based on user selections
        
        Args:
            identify_anomalies: Whether to focus on anomaly detection
            describe_slice: Whether to provide general description
            
        Returns:
            Formatted prompt string
        """
        if get_medical_prompt:
            if identify_anomalies and describe_slice:
                # Combined prompt
                return (get_medical_prompt("identify_anomalies") + " " +
                       "Additionally, " + get_medical_prompt("describe_slice").lower())
            elif identify_anomalies:
                return get_medical_prompt("identify_anomalies")
            elif describe_slice:
                return get_medical_prompt("describe_slice")
            else:
                # Default to description if neither is selected
                return get_medical_prompt("describe_slice")
        else:
            # Fallback prompts if service not available
            if identify_anomalies and describe_slice:
                return ("Analyze this brain MRI slice for any abnormalities or anomalies, "
                       "and also provide a general description of the anatomical structures visible.")
            elif identify_anomalies:
                return ("Analyze this brain MRI slice and identify any abnormalities, anomalies, "
                       "or pathological regions that may be present.")
            elif describe_slice:
                return ("Describe this brain MRI slice, identifying the anatomical structures "
                       "and overall characteristics visible in the image.")
            else:
                return "Analyze and describe this brain MRI slice."
    
    def _get_prompt_type(self, identify_anomalies: bool, describe_slice: bool) -> str:
        """Get a readable description of the prompt type"""
        if identify_anomalies and describe_slice:
            return "anomaly detection + description"
        elif identify_anomalies:
            return "anomaly detection"
        elif describe_slice:
            return "general description"
        else:
            return "default analysis"
    
    def cleanup(self):
        """Clean up MedGemma service resources"""
        try:
            if cleanup_service:
                cleanup_service()
                logger.info("MedGemma service cleaned up")
        except Exception as e:
            logger.error(f"Error during MedGemma cleanup: {e}")


# MRI Label Suggestion Prompt (for future use as specified in requirements)
MRI_LABEL_SUGGESTION_PROMPT = (
    "This is a brain MRI slice. Please list the anatomical or pathological structures that are visible. "
    "Return a comma-separated list of possible labels (e.g., eye, lvent, tvent, tumor, lesion, etc.)."
)
