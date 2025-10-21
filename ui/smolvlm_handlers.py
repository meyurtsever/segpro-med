"""
SmolVLM Handler for SegMed-Pro

This module provides functionality to integrate SmolVLM visual language model
for slice captioning in the Editor tab using a persistent service for fast inference.
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

from smolvlm.smolvlm_service import get_service, cleanup_service

logger = logging.getLogger(__name__)


class SmolVLMHandlers:
    """Handlers for SmolVLM visual language model operations"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self._service = None
        
    def _get_service(self):
        """Get or initialize the SmolVLM service"""
        if self._service is None:
            try:
                self._service = get_service(device="auto")
                logger.info("SmolVLM service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SmolVLM service: {e}")
                self._service = None
        return self._service
        
    def run_vlm_inference(self, image_annotator_value: Optional[dict], identify_anomalies: bool = True, describe_slice: bool = False, modality: str = "MRI") -> str:
        """
        Run VLM inference on the current image from image_annotator using persistent service
        
        Args:
            image_annotator_value: Current value from the image_annotator component
            identify_anomalies: Whether to use anomaly identification prompt
            describe_slice: Whether to use general description prompt
            modality: Imaging modality (MRI, MG for mammography, CT, etc.)
            
        Returns:
            str: VLM generated caption or error message
        """
        try:
            if image_annotator_value is None or "image" not in image_annotator_value:
                return "Error: No image available for VLM analysis"
            
            # Extract image from annotator value
            image_array = image_annotator_value["image"]
            if image_array is None:
                return "Error: Image data is None"
            
            # Determine the prompt based on checkbox selection and modality
            if modality == "MG":
                if identify_anomalies:
                    prompt = "Identify and describe any abnormal findings in this mammogram. Look for masses, calcifications, architectural distortions, or asymmetries. Use BI-RADS terminology."
                elif describe_slice:
                    prompt = "Describe this mammogram in detail, focusing on breast tissue composition, anatomical structures, and any notable features."
                else:
                    prompt = "Describe this mammogram in detail, focusing on breast tissue composition, anatomical structures, and any notable features."
            else:  # MRI or other
                if identify_anomalies:
                    prompt = "Identify and label abnormal regions in this brain MRI. Highlight any suspicious areas and suggest their likely pathology."
                elif describe_slice:
                    prompt = "Describe this medical image slice in detail, focusing on visible anatomical structures and any notable features."
                else:
                    prompt = "Describe this medical image slice in detail, focusing on visible anatomical structures and any notable features."
            
            logger.info(f"Running VLM inference with prompt for {modality}: {prompt[:50]}...")
            logger.info(f"Image shape: {image_array.shape}")
            
            # Convert numpy array to PIL Image
            if isinstance(image_array, np.ndarray):
                # Ensure the image is in the correct format (0-255, uint8)
                if image_array.dtype != np.uint8:
                    # Normalize to 0-255 if needed
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        image_array = image_array.astype(np.uint8)
                  # Convert to PIL Image
                if len(image_array.shape) == 3:
                    image = Image.fromarray(image_array, 'RGB')
                elif len(image_array.shape) == 2:
                    # Grayscale to RGB
                    image_rgb = np.stack([image_array] * 3, axis=-1)
                    image = Image.fromarray(image_rgb, 'RGB')
                else:
                    return "Error: Unsupported image shape for VLM processing"
            else:
                return "Error: Image data is not a numpy array"
              
            # Save image to temporary file
            temp_image_path = self._save_temp_image(image)
            if temp_image_path is None:
                return "Error: Failed to save temporary image"
            
            # Also save a persistent copy for debugging/reference - disabled after debugging
            # self._save_processed_image(image)
            
            try:
                # Get the persistent service
                service = self._get_service()
                if service is None:
                    return "Error: Failed to initialize SmolVLM service"
                
                # Run inference using the persistent service (much faster!)
                result = service.generate_caption(temp_image_path, prompt, max_tokens=75) # 100
                
                if result["success"]:
                    timings = result["timings"]
                    logger.info(f"VLM inference completed in {timings['total_time']:.2f}s "
                              f"(generation: {timings['text_generation']:.2f}s)")
                    return result["caption"]
                else:
                    logger.error(f"VLM service error: {result['error']}")
                    return f"Error: {result['error']}"
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            logger.error(f"Error in VLM inference: {e}")
            return f"Error: {str(e)}"
    
    def _save_processed_image(self, image: Image.Image) -> Optional[str]:
        """
        Save a persistent copy of the processed image for debugging/reference
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Path to saved image file or None if failed
        """
        try:
            # Create the filename in the current working directory
            output_path = "latest_processed_smolvlm.jpg"
            
            # Convert image to RGB if it's not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image as JPG for maximum compatibility
            image.save(output_path, 'JPEG', quality=95)
            logger.info(f"Saved processed SmolVLM image to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed SmolVLM image: {e}")
            return None

    def _save_temp_image(self, image: Image.Image) -> Optional[str]:
        """
        Save PIL Image to temporary file
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Path to temporary image file or None if failed
        """
        try:
            # Create temporary file with .jpg extension for better compatibility
            # SmolVLM supports JPG which is widely supported
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Convert image to RGB if it's not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
              # Save image as JPG for maximum compatibility
            image.save(temp_path, 'JPEG', quality=95)
            logger.info(f"Saved temporary image to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            return None
    
    def cleanup(self):
        """Clean up the VLM service resources"""
        if self._service is not None:
            try:
                cleanup_service()
                self._service = None
                logger.info("SmolVLM service cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up SmolVLM service: {e}")
