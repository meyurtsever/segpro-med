"""
Med-R1 Handler for SegMed-Pro

This module provides functionality to integrate Med-R1 visual language model
for medical image analysis in the Editor tab using a persistent service for fast inference.
Med-R1 is specifically optimized for medical imaging with MRI checkpoints and 384x384 image requirements.
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

# Add med-r1 directory to path
med_r1_dir = os.path.join(models_dir, "med-r1")
if med_r1_dir not in sys.path:
    sys.path.append(med_r1_dir)

try:
    from med_r1_service import get_service, cleanup_service
except ImportError:
    # Fallback if service is not available
    logger = logging.getLogger(__name__)
    logger.warning("Med-R1 service not available - check installation")
    get_service = None
    cleanup_service = None

logger = logging.getLogger(__name__)


class MedR1Handlers:
    """Handlers for Med-R1 visual language model operations"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self._service = None
        
    def _get_service(self):
        """Get or initialize the Med-R1 service"""
        if self._service is None and get_service is not None:
            try:
                # Use MRI checkpoint for brain imaging (if available)
                # You can modify this path to point to your local MRI checkpoint
                checkpoint_path = "yuxianglai117/Med-R1"  # Default HuggingFace model
                
                # Check if local MRI checkpoint exists
                local_mri_checkpoint = os.path.join(
                    os.path.dirname(__file__), "..", "models", "med-r1", "checkpoints", "MRI"
                )
                if os.path.exists(local_mri_checkpoint):
                    checkpoint_path = local_mri_checkpoint
                    logger.info(f"Using local MRI checkpoint: {checkpoint_path}")
                else:
                    logger.info(f"Using default HuggingFace checkpoint: {checkpoint_path}")
                
                self._service = get_service(device="auto", checkpoint_path=checkpoint_path)
                logger.info("Med-R1 service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Med-R1 service: {e}")
                self._service = None
        return self._service
        
    def run_med_r1_inference(self, image_annotator_value: Optional[dict], identify_anomalies: bool = True, describe_slice: bool = False) -> str:
        """
        Run Med-R1 VLM inference on the current image from image_annotator using persistent service
        
        Args:
            image_annotator_value: Current value from the image_annotator component
            identify_anomalies: Whether to use anomaly identification prompt
            describe_slice: Whether to use general description prompt
            
        Returns:
            str: Med-R1 generated caption or error message
        """
        try:
            if get_service is None:
                return "Error: Med-R1 service not available. Please check installation and requirements."
            
            if image_annotator_value is None or "image" not in image_annotator_value:
                return "Error: No image available for Med-R1 analysis"
            
            # Extract image from annotator value
            image_array = image_annotator_value["image"]
            if image_array is None:
                return "Error: Image data is None"
            
            # Determine the prompt based on checkbox selection
            if identify_anomalies:
                prompt = ("Analyze this brain MRI scan for any abnormal findings. "
                         "Identify any lesions, masses, hemorrhages, infarcts, or other pathological changes. "
                         "Describe the location, size, and characteristics of any abnormalities found. "
                         "If no abnormalities are detected, state that the scan appears normal.")
            elif describe_slice:
                prompt = ("Provide a detailed medical analysis of this brain MRI slice. "
                         "Describe the anatomical structures visible, imaging quality, "
                         "slice level, and any notable features or findings. "
                         "Include information about brain symmetry, ventricles, and tissue contrast.")
            else:
                # Fallback prompt if neither is selected
                prompt = ("Analyze this medical brain MRI image. "
                         "Describe the visible anatomical structures and any notable findings.")
            
            logger.info(f"Running Med-R1 inference with prompt: {prompt[:50]}...")
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
                    return "Error: Unsupported image shape for Med-R1 processing"
            else:
                return "Error: Image data is not a numpy array"
              
            # Save image to temporary file (Med-R1 needs file path)
            temp_image_path = self._save_temp_image(image)
            if temp_image_path is None:
                return "Error: Failed to save temporary image"
            
            # Also save a persistent copy for debugging/reference
            self._save_processed_image(image)
            
            try:
                # Get the persistent service
                service = self._get_service()
                if service is None:
                    return "Error: Failed to initialize Med-R1 service"
                
                # Run inference using the persistent service
                result = service.generate_caption(temp_image_path, prompt, max_tokens=256)
                
                if result["success"]:
                    timings = result["timings"]
                    logger.info(f"Med-R1 inference completed in {timings['total_time']:.2f}s "
                              f"(generation: {timings['text_generation']:.2f}s)")
                    
                    # Format the response nicely
                    caption = result["caption"]
                    
                    # Add timing information if in debug mode
                    if logger.isEnabledFor(logging.DEBUG):
                        caption += f"\n\n*Generated in {timings['total_time']:.2f}s*"
                    
                    return caption
                else:
                    logger.error(f"Med-R1 service error: {result['error']}")
                    return f"Error: {result['error']}"
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            logger.error(f"Error in Med-R1 inference: {e}")
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
            output_path = "latest_processed_med_r1.jpg"
            
            # Convert image to RGB if it's not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 384x384 as required by Med-R1
            image_resized = image.resize((384, 384), Image.Resampling.LANCZOS)
            
            # Save image as JPG for maximum compatibility
            image_resized.save(output_path, 'JPEG', quality=95)
            logger.info(f"Saved processed Med-R1 image (384x384) to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed Med-R1 image: {e}")
            return None

    def _save_temp_image(self, image: Image.Image) -> Optional[str]:
        """
        Save PIL Image to temporary file with Med-R1 requirements (384x384)
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Path to temporary image file or None if failed
        """
        try:
            # Create temporary file with .jpg extension for better compatibility
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Convert image to RGB if it's not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 384x384 as required by Med-R1
            image_resized = image.resize((384, 384), Image.Resampling.LANCZOS)
              
            # Save image as JPG for maximum compatibility
            image_resized.save(temp_path, 'JPEG', quality=95)
            logger.info(f"Saved temporary Med-R1 image (384x384) to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            return None
    
    def cleanup(self):
        """Clean up the Med-R1 service resources"""
        if self._service is not None and cleanup_service is not None:
            try:
                cleanup_service()
                self._service = None
                logger.info("Med-R1 service cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up Med-R1 service: {e}")

    def is_available(self) -> bool:
        """Check if Med-R1 service is available"""
        return get_service is not None
    
    def get_model_info(self) -> dict:
        """Get information about the Med-R1 model"""
        service = self._get_service()
        if service is not None:
            return {
                "model_name": "Med-R1 (Medical VLM)",
                "checkpoint": service.checkpoint_path,
                "device": service.device,
                "loaded": service.model_loaded,
                "image_size": "384x384",
                "modality": "MRI-optimized"
            }
        else:
            return {
                "model_name": "Med-R1 (Medical VLM)",
                "checkpoint": "Not available",
                "device": "N/A",
                "loaded": False,
                "image_size": "384x384",
                "modality": "MRI-optimized"
            }
