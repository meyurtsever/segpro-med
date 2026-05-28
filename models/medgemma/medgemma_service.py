"""
MedGemma-4B Service for SegMed-Pro

This module provides a persistent service for MedGemma-4B inference
to enable fast medical image analysis with the Google MedGemma model.
"""

import logging
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import gc
import importlib.util
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
_medgemma_report_service = None
_medgemma15_service = None


class MedGemmaService:
    """Persistent service for MedGemma-4B inference"""
    
    def __init__(
        self,
        device: str = "auto",
        model_name: str = "google/medgemma-4b-it",
        local_model_dir: Optional[Union[str, Path]] = None,
        initialize_xai: bool = True,
        service_label: str = "MedGemma-4B",
    ):
        """
        Initialize MedGemma-4B service
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.local_model_dir = Path(local_model_dir) if local_model_dir else None
        self.initialize_xai = initialize_xai
        self.service_label = service_label
        self.using_device_map = False  # Track if using device_map for multi-GPU
        self.visual_grounding = None  # XAI visual grounding system
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
            if self.local_model_dir is not None:
                config_path = self.local_model_dir / "config.json"
                if config_path.exists():
                    logger.info("%s model found locally at %s", self.service_label, self.local_model_dir)
                    return
                raise RuntimeError(f"{self.service_label} local model missing config.json at {self.local_model_dir}")

            # First, ensure we're authenticated with HuggingFace
            try:
                from .hf_auth import login_to_huggingface, check_model_access, print_authentication_instructions
            except ImportError:
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
                    try:
                        from .download_medgemma import download_medgemma_model
                    except ImportError:
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
            logger.info(f"Initializing {self.service_label} on {self.device}")
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            
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
            if self.device == "cuda" and importlib.util.find_spec("bitsandbytes") is not None:
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
            elif self.device == "cuda":
                logger.info("bitsandbytes is not installed; using BF16 CUDA inference")
            
            # Load processor
            model_source = str(self.local_model_dir) if self.local_model_dir is not None else self.model_name

            self.processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True,
                use_fast=True,
            )
            logger.info("Loaded %s processor", self.service_label)
            
            # Track whether we're using device_map for multi-GPU distribution
            self.using_device_map = False
            
            # Load model with error handling and fallbacks
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
            }
            if self.device == "cuda":
                model_kwargs["attn_implementation"] = "sdpa"
            
            # Use device_map for multi-GPU setups only if we have quantization
            if self.device == "cuda" and quantization_config is not None:
                model_kwargs["device_map"] = "auto"
                model_kwargs["quantization_config"] = quantization_config
                self.using_device_map = True
                logger.info("Using device_map='auto' for multi-GPU distribution")
            
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    **model_kwargs
                )
                logger.info(f"{self.service_label} loaded successfully on {self.device}")
                
                # Move to device only if not using device_map
                if not self.using_device_map:
                    self.model = self.model.to(self.device)
                    logger.info(f"Model moved to {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load MedGemma model: {e}")
                # Fallback to CPU with minimal config
                logger.info("Attempting fallback to CPU...")
                self.device = "cpu"
                self.using_device_map = False
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                }
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_source,
                    **model_kwargs
                ).to("cpu")
                logger.info("%s loaded on CPU (fallback)", self.service_label)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Initialize visual grounding XAI systems (gradient-based and attention-based)
            try:
                if not self.initialize_xai:
                    self.visual_grounding = None
                    self.attention_grounding = None
                    self.vlm_grounding = None
                    logger.info("Visual grounding XAI disabled for %s", self.service_label)
                    return

                # Import visual grounding modules dynamically to avoid circular imports
                import sys
                import os
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                
                # Original gradient-based approach (fallback)
                from xai.vlm.visual_grounding import MedGemmaVisualGrounding
                self.visual_grounding = MedGemmaVisualGrounding(
                    model=self.model,
                    processor=self.processor,
                    device=self.device
                )
                logger.info("Gradient-based visual grounding XAI initialized (fallback)")
                
                # Fast attention-based approach (previous attempt)
                from xai.vlm.attention_visual_grounding import AttentionVisualGrounding
                self.attention_grounding = AttentionVisualGrounding(
                    model=self.model,
                    processor=self.processor,
                    device=self.device
                )
                logger.info("Attention-based visual grounding initialized")
                
                # NEW: Proper VLM XAI (Input×Gradient + Decoder Attention)
                from xai.vlm.vlm_visual_grounding import VLMVisualGrounding
                self.vlm_grounding = VLMVisualGrounding(
                    model=self.model,
                    processor=self.processor,
                    device=self.device
                )
                logger.info("VLM visual grounding XAI initialized (Input×Gradient + Decoder Attention)")
                
            except Exception as e:
                logger.warning(f"Could not initialize visual grounding XAI: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self.visual_grounding = None
                self.attention_grounding = None
                self.vlm_grounding = None
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.service_label} service: {e}")
            raise RuntimeError(f"{self.service_label} initialization failed: {e}")
    
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
            )
            
            # Handle device placement based on model configuration
            if self.using_device_map:
                # When using device_map='auto', the model is distributed across devices
                # We should let the model handle device placement internally
                # Only move inputs to the first device if explicitly needed
                try:
                    # Try to infer the device from the model's first parameter
                    first_device = next(self.model.parameters()).device
                    inputs = inputs.to(first_device)
                    logger.debug(f"Moved inputs to model's first device: {first_device}")
                except Exception as e:
                    logger.warning(f"Could not determine model device, keeping inputs on CPU: {e}")
                    # Keep inputs on CPU and let the model handle device placement
            else:
                # For single-device setups, move inputs to the model device
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,  # Enable dict output for caching
                    output_attentions=False,  # Attentions during generation are complex, use forward pass instead
                    output_hidden_states=False,
                )
            
            # Handle both dict and tensor outputs
            if hasattr(outputs, 'sequences'):
                generation = outputs.sequences
            else:
                generation = outputs
            
            # Extract the new tokens (response only)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = generation[0][input_length:]
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Cache generation outputs for attention-based XAI (FAST path)
            if hasattr(self, 'attention_grounding') and self.attention_grounding is not None:
                try:
                    self.attention_grounding.cache_generation_outputs(
                        image=processed_image,
                        prompt=prompt,
                        generated_text=response.strip(),
                        inputs=inputs,
                        generation=generation,
                        attentions=getattr(outputs, 'attentions', None),
                        hidden_states=getattr(outputs, 'hidden_states', None)
                    )
                    logger.info("Cached generation outputs for fast XAI")
                except Exception as e:
                    logger.warning(f"Could not cache generation outputs: {e}")
            
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
    
    def compute_visual_grounding(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        target_token: str,
        generated_text: Optional[str] = None,
        xai_method: str = "input_gradient"
    ) -> tuple:
        """
        Compute visual grounding heatmap for a specific token.
        
        Shows which image regions influenced the model to generate a specific token/label.
        
        Args:
            image: Input image (PIL Image or numpy array)
            prompt: Text prompt used for generation
            target_token: Token/word to visualize (e.g., "tumor", "lesion")
            generated_text: Pre-generated text (optional)
            xai_method: XAI method to use:
                - "input_gradient" (RECOMMENDED): Input×Gradient on projected image tokens
                - "decoder_attention": Decoder attention weights (SDPA workaround)
                - "attention": Legacy embedding similarity (fast but less accurate)
                - "hirescam", "gradcam", etc.: Legacy gradient-based (slow)
        
        Returns:
            Tuple of (heatmap, metadata):
                - heatmap: numpy array with spatial attribution (H, W), values in [0, 1]
                - metadata: dict with statistics and method information
        """
        try:
            # Preprocess image
            image_pil = self.preprocess_image(image)
            
            # Strategy 1: Input × Gradient (RECOMMENDED - proper VLM XAI)
            if xai_method == "input_gradient" and hasattr(self, 'vlm_grounding') and self.vlm_grounding is not None:
                logger.info("Using Input×Gradient XAI (proper VLM approach)")
                heatmap, metadata = self.vlm_grounding.compute_visual_grounding(
                    image=image_pil,
                    prompt=prompt,
                    target_label=target_token,
                    method="input_gradient"
                )
                if heatmap is not None and metadata.get('success'):
                    return heatmap, metadata
                else:
                    error_msg = metadata.get('error', 'Unknown error')
                    logger.info(f"Input×Gradient failed: {error_msg}, trying decoder attention...")
            
            # Strategy 2: Decoder Attention (SDPA workaround)
            if xai_method in ["decoder_attention", "input_gradient"] and hasattr(self, 'vlm_grounding') and self.vlm_grounding is not None:
                logger.info("Using Decoder Attention XAI (SDPA workaround)")
                heatmap, metadata = self.vlm_grounding.compute_visual_grounding(
                    image=image_pil,
                    prompt=prompt,
                    target_label=target_token,
                    method="decoder_attention"
                )
                if heatmap is not None and metadata.get('success'):
                    return heatmap, metadata
                else:
                    error_msg = metadata.get('error', 'Unknown error')
                    logger.info(f"Decoder attention failed: {error_msg}, trying legacy methods...")
            
            # Legacy: Attention-based (embedding similarity - fast but less principled)
            if xai_method == "attention" and hasattr(self, 'attention_grounding') and self.attention_grounding is not None:
                logger.info("Using legacy attention-based XAI (embedding similarity)")
                xai_image = self.attention_grounding._cache.get('preprocessed_image', image_pil)
                heatmap, metadata = self.attention_grounding.compute_visual_grounding(
                    image=xai_image,
                    prompt=prompt,
                    target_token=target_token
                )
                if heatmap is not None and metadata.get('success'):
                    metadata['method'] = 'embedding_similarity (legacy)'
                    return heatmap, metadata
            
            # Legacy: Gradient-based (slow but works)
            if self.visual_grounding is not None:
                logger.info(f"Using legacy gradient-based XAI: {xai_method}")
                heatmap, metadata = self.visual_grounding.compute_visual_grounding(
                    image=image_pil,
                    prompt=prompt,
                    target_token=target_token,
                    generated_text=generated_text,
                    xai_method=xai_method if xai_method in ["hirescam", "gradcam", "gradcam++", "guided_gradcam"] else "hirescam"
                )
                return heatmap, metadata
            
            logger.error("No visual grounding system available")
            return None, {"error": "Visual grounding not available", "success": False}
        
        except Exception as e:
            logger.error(f"Error computing visual grounding: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {"error": str(e), "success": False}

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
            models_dir = Path(__file__).resolve().parents[1]
            _medgemma_service = MedGemmaService(
                device=device,
                model_name="google/medgemma-4b-it",
                local_model_dir=models_dir / "medgemma",
                initialize_xai=True,
                service_label="MedGemma-4B",
            )
            logger.info("Created new MedGemma service instance")
        return _medgemma_service
    except Exception as e:
        logger.error(f"Failed to get MedGemma service: {e}")
        raise


def get_report_service(device: str = "auto") -> MedGemmaService:
    """Get or create a MedGemma-4B service optimized for report generation only."""
    global _medgemma_report_service

    try:
        if _medgemma_report_service is None:
            models_dir = Path(__file__).resolve().parents[1]
            _medgemma_report_service = MedGemmaService(
                device=device,
                model_name="google/medgemma-4b-it",
                local_model_dir=models_dir / "medgemma",
                initialize_xai=False,
                service_label="MedGemma-4B Report",
            )
            logger.info("Created new MedGemma report service instance")
        return _medgemma_report_service
    except Exception as e:
        logger.error(f"Failed to get MedGemma report service: {e}")
        raise


def get_medgemma15_service(device: str = "auto") -> MedGemmaService:
    """Get or create the local MedGemma 1.5 4B service instance."""
    global _medgemma15_service

    try:
        if _medgemma15_service is None:
            models_dir = Path(__file__).resolve().parents[1]
            _medgemma15_service = MedGemmaService(
                device=device,
                model_name="google/medgemma-1.5-4b-it",
                local_model_dir=models_dir / "medgemma-1.5-4b-it",
                initialize_xai=False,
                service_label="MedGemma-1.5-4B",
            )
            logger.info("Created new MedGemma 1.5 service instance")
        return _medgemma15_service
    except Exception as e:
        logger.error(f"Failed to get MedGemma 1.5 service: {e}")
        raise


def cleanup_service(keep: Optional[str] = None):
    """Clean up MedGemma services, optionally preserving one selected model family."""
    global _medgemma_service, _medgemma_report_service, _medgemma15_service
    
    if keep != "medgemma" and _medgemma_service is not None:
        _medgemma_service.cleanup()
        _medgemma_service = None
        logger.info("Global MedGemma service cleaned up")
    if keep != "medgemma" and _medgemma_report_service is not None:
        _medgemma_report_service.cleanup()
        _medgemma_report_service = None
        logger.info("Global MedGemma report service cleaned up")
    if keep != "medgemma-1.5" and _medgemma15_service is not None:
        _medgemma15_service.cleanup()
        _medgemma15_service = None
        logger.info("Global MedGemma 1.5 service cleaned up")


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
