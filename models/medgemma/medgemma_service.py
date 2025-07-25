"""
MedGemma-4B Service for SegMed-Pro

This module provides a persistent service for MedGemma-4B inference
to enable fast medical image analysis with the Google MedGemma model.
"""

import logging
import torch
import gc
from typing import Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

# Global service instance for persistence
_medgemma_service = None


class MedGemmaService:
    """Persistent service for MedGemma-4B inference"""
    
    def __init__(self, device: str = "auto"):
        """
        Initialize MedGemma-4B service
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.model_name = "google/medgemma-4b-it"
        self._initialize_model()
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _ensure_model_available(self):
        """Ensure MedGemma model is available locally, download if not"""
        try:
            # First, ensure we're authenticated with HuggingFace
            from hf_auth import login_to_huggingface, check_model_access, print_authentication_instructions
            
            # Try to authenticate
            auth_success = login_to_huggingface()
            if not auth_success:
                logger.error("HuggingFace authentication failed")
                print_authentication_instructions()
                raise RuntimeError("HuggingFace authentication required for MedGemma-4B")
            
            # Check if we have access to the model
            if not check_model_access(self.model_name):
                logger.error(f"No access to model {self.model_name}")
                print_authentication_instructions()
                raise RuntimeError(f"Access denied to {self.model_name}. Please request access and ensure proper authentication.")
            
            # Try to check if model exists by attempting to load config
            from transformers import AutoConfig
            
            try:
                # Try to load config to check if model exists locally
                config = AutoConfig.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=True  # Only check locally
                )
                logger.info("MedGemma model found locally")
                return
            except Exception:
                # Model not found locally, try to download
                logger.info("MedGemma model not found locally, attempting download...")
                
                # Try to import and run the download function
                try:
                    from download_medgemma import download_medgemma_model
                    success = download_medgemma_model(force_redownload=False)
                    if success:
                        logger.info("MedGemma model downloaded successfully")
                    else:
                        logger.warning("Failed to download MedGemma model, will try to load from HuggingFace Hub")
                except ImportError:
                    logger.warning("Download script not available, will try to load from HuggingFace Hub")
                except Exception as e:
                    logger.warning(f"Error during model download: {e}, will try to load from HuggingFace Hub")
                    
        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the MedGemma model and processor"""
        try:
            logger.info(f"Initializing MedGemma-4B on {self.device}")
            
            # Check if model exists locally, download if not
            self._ensure_model_available()
            
            # Import transformers components
            from transformers import (
                AutoProcessor, 
                AutoModelForImageTextToText,
                BitsAndBytesConfig
            )
            
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    logger.info("Using 4-bit quantization for GPU")
                except Exception as e:
                    logger.warning(f"Quantization not available: {e}")
                    quantization_config = None
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("Loaded MedGemma processor")
            
            # Load model with error handling and fallbacks
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                logger.info(f"MedGemma-4B loaded successfully on {self.device}")
                
                # Move to device if not using device_map
                if self.device != "cuda" or quantization_config is None:
                    self.model = self.model.to(self.device)
                
            except Exception as e:
                logger.error(f"Failed to load MedGemma model: {e}")
                # Fallback to CPU with minimal config
                logger.info("Attempting fallback to CPU...")
                self.device = "cpu"
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                }
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    **model_kwargs
                ).to("cpu")
                logger.info("MedGemma-4B loaded on CPU (fallback)")
            
            # Set model to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to initialize MedGemma service: {e}")
            raise RuntimeError(f"MedGemma initialization failed: {e}")
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Preprocess image for MedGemma
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert numpy array to PIL if needed
            if isinstance(image, np.ndarray):
                # Handle different image formats
                if image.dtype != np.uint8:
                    # Normalize to 0-255 range
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                
                # Convert to RGB if grayscale
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    image = np.repeat(image, 3, axis=2)
                
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to reasonable size for processing (MedGemma typically works well with 384x384 or 512x512)
            target_size = (512, 512)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def generate_response(self, image: Union[Image.Image, np.ndarray], prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate text response for an image using MedGemma
        
        Args:
            image: Input image (PIL Image or numpy array)
            prompt: Text prompt
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            if self.model is None or self.processor is None:
                raise RuntimeError("MedGemma service not properly initialized")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Prepare the message in chat format as required by MedGemma
            # Include system message to establish the model's role as an expert radiologist
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=input_text,
                images=processed_image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Extract the new tokens (response only)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating MedGemma response: {e}")
            return f"Error generating response: {str(e)}"
    
    def cleanup(self):
        """Clean up model resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("MedGemma service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during MedGemma cleanup: {e}")


def get_service(device: str = "auto") -> MedGemmaService:
    """
    Get or create the global MedGemma service instance
    
    Args:
        device: Device to use ("auto", "cuda", "cpu")
        
    Returns:
        MedGemmaService instance
    """
    global _medgemma_service
    
    try:
        if _medgemma_service is None:
            _medgemma_service = MedGemmaService(device=device)
            logger.info("Created new MedGemma service instance")
        return _medgemma_service
    except Exception as e:
        logger.error(f"Failed to get MedGemma service: {e}")
        raise


def cleanup_service():
    """Clean up the global MedGemma service"""
    global _medgemma_service
    
    if _medgemma_service is not None:
        _medgemma_service.cleanup()
        _medgemma_service = None
        logger.info("Global MedGemma service cleaned up")


# Medical imaging specific prompts for MedGemma
MEDICAL_PROMPTS = {
    "identify_anomalies": (
        "As a medical AI assistant, carefully analyze this brain MRI slice. "
        "Identify any abnormal findings, anomalies, or pathological regions. "
        "Describe the location, characteristics, and potential clinical significance of any abnormalities you observe. "
        "If no clear abnormalities are visible, state that the slice appears normal."
    ),
    "describe_slice": (
        "As a medical AI assistant, provide a detailed description of this brain MRI slice. "
        "Identify the anatomical structures visible, the imaging plane (axial, sagittal, or coronal), "
        "and describe the overall tissue contrast and image quality. "
        "Mention any notable anatomical landmarks or structures that are clearly visible."
    ),
    "suggest_labels": (
        "This is a brain MRI slice. Please list the anatomical or pathological structures that are visible. "
        "Return a comma-separated list of possible labels (e.g., eye, lvent, tvent, tumor, lesion, etc.)."
    )
}


def get_medical_prompt(prompt_type: str) -> str:
    """
    Get a medical-specific prompt for MedGemma
    
    Args:
        prompt_type: Type of prompt ("identify_anomalies", "describe_slice", "suggest_labels")
        
    Returns:
        Medical prompt string
    """
    return MEDICAL_PROMPTS.get(prompt_type, MEDICAL_PROMPTS["describe_slice"])
