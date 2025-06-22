"""
SmolVLM Handler for SegMed-Pro

This module provides functionality to integrate SmolVLM visual language model
for slice captioning in the Editor tab.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class SmolVLMHandlers:
    """Handlers for SmolVLM visual language model operations"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self.smolvlm_path = os.path.join(os.getcwd(), "models", "smolvlm", "smolvlm_cli.py")
        
    def run_vlm_inference(self, image_annotator_value: Optional[dict]) -> str:
        """
        Run VLM inference on the current image from image_annotator
        
        Args:
            image_annotator_value: Current value from the image_annotator component
            
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
            
            logger.info(f"Running VLM inference on image with shape: {image_array.shape}")
            
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
            
            try:
                # Run SmolVLM CLI
                caption = self._run_smolvlm_cli(temp_image_path)
                return caption
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            logger.error(f"Error in VLM inference: {e}")
            return f"Error: {str(e)}"
    
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
    
    def _run_smolvlm_cli(self, image_path: str) -> str:
        """
        Run SmolVLM CLI with the given image path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: VLM generated caption
        """
        try:
            # Check if SmolVLM CLI exists
            if not os.path.exists(self.smolvlm_path):
                return f"Error: SmolVLM CLI not found at {self.smolvlm_path}"
            
            # Prepare command
            cmd = [
                "python", 
                self.smolvlm_path,
                image_path,
                "--message", "Describe this medical image slice in detail, focusing on visible anatomical structures and any notable features.",
                "--max-tokens", "256",
                "--device", "auto"
            ]
            
            logger.info(f"Running SmolVLM command: {' '.join(cmd)}")
            
            # Run the command and capture output
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,  # 2 minute timeout
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                # Parse the output to extract the actual response
                output_lines = result.stdout.strip().split('\n')
                
                # Look for the response between the "SMOLVLM RESPONSE:" markers
                in_response = False
                response_lines = []
                
                for line in output_lines:
                    if "SMOLVLM RESPONSE:" in line:
                        in_response = True
                        continue
                    elif in_response and "=" * 50 in line and len(response_lines) > 0:
                        break
                    elif in_response:
                        response_lines.append(line)
                
                if response_lines:
                    caption = '\n'.join(response_lines).strip()
                    logger.info(f"VLM generated caption: {caption[:100]}...")
                    return caption
                else:
                    # Fallback: return the last few non-empty lines
                    non_empty_lines = [line for line in output_lines if line.strip()]
                    if non_empty_lines:
                        return non_empty_lines[-1]
                    else:
                        return "VLM completed but no response found in output"
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"SmolVLM CLI failed with return code {result.returncode}: {error_msg}")
                return f"VLM Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            logger.error("SmolVLM CLI timed out")
            return "Error: VLM inference timed out (>120s)"
        except Exception as e:
            logger.error(f"Error running SmolVLM CLI: {e}")
            return f"Error running VLM: {str(e)}"
