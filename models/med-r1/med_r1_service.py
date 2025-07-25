"""
Med-R1 VLM Service for SegMed-Pro

This service provides a persistent Med-R1 Visual Language Model interface
for medical image analysis, specifically optimized for brain MRI scans.

Med-R1 is based on Qwen2VL architecture and fine-tuned for medical imaging tasks.
"""

import logging
import os
import sys
import tempfile
from typing import Optional, Dict, Any
import torch
from PIL import Image
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Global service instance
_service_instance = None

class MedR1Service:
    """Persistent Med-R1 VLM service for fast inference"""
    
    def __init__(self, device: str = "auto", checkpoint_path: str = None):
        """Initialize Med-R1 service
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
            checkpoint_path: Path to model checkpoint (local or HuggingFace)
        """
        self.device = self._resolve_device(device)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # Load model components
        self._load_model()
    
    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> str:
        """Resolve the checkpoint path to use local MRI checkpoint if available"""
        if checkpoint_path is not None:
            return checkpoint_path
            
        # Try to use local MRI checkpoint first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_mri_checkpoint = os.path.join(script_dir, "checkpoints", "MRI")
        
        logger.info(f"Checking for local MRI checkpoint at: {local_mri_checkpoint}")
        
        # Check for essential model files
        required_files = [
            "config.json",
            "tokenizer_config.json", 
            "tokenizer.json",
            "model.safetensors.index.json"
        ]
        
        # Check if directory exists and has required files
        if os.path.exists(local_mri_checkpoint):
            missing_files = []
            for file in required_files:
                file_path = os.path.join(local_mri_checkpoint, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
            
            if not missing_files:
                logger.info(f"✅ Using local MRI checkpoint: {local_mri_checkpoint}")
                return local_mri_checkpoint
            else:
                logger.warning(f"⚠️ Local MRI checkpoint incomplete. Missing files: {missing_files}")
                logger.info("📥 Falling back to HuggingFace checkpoint: yuxianglai117/Med-R1")
        else:
            logger.info(f"❌ Local MRI checkpoint directory not found at {local_mri_checkpoint}")
            logger.info("📥 Using HuggingFace checkpoint: yuxianglai117/Med-R1")
        
        # Fallback to HuggingFace
        return "yuxianglai117/Med-R1"
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load Med-R1 model and processor"""
        try:
            logger.info(f"Loading Med-R1 model from: {self.checkpoint_path}")
            logger.info(f"Using device: {self.device}")
            
            # Check if we need transformers and qwen_vl_utils
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                from qwen_vl_utils import process_vision_info
            except ImportError as e:
                logger.error(f"Required dependencies not available: {e}")
                logger.error("Please install: pip install transformers qwen-vl-utils")
                self.model_loaded = False
                return
            
            # Load processor first
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=True
            )
            
            # Fix: Ensure chat template is properly loaded
            if hasattr(self.processor, 'tokenizer'):
                # Check if tokenizer exists and try to load chat template
                try:
                    if not hasattr(self.processor.tokenizer, 'chat_template') or self.processor.tokenizer.chat_template is None:
                        logger.warning("Chat template not found, attempting to reload...")
                        self._load_chat_template_manually()
                    else:
                        logger.info("✅ Chat template already loaded")
                except Exception as e:
                    logger.warning(f"Error checking chat template: {e}")
            else:
                logger.warning("⚠️ Processor has no tokenizer attribute")
            
            # Load model with proper device handling (following official script)
            logger.info("Loading model...")
            if self.device == "cuda":
                # Load model on CUDA with official script parameters
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=torch.bfloat16,  # Official script uses bfloat16
                    attn_implementation="flash_attention_2" if self._has_flash_attention() else "eager",
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Load model on CPU
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.checkpoint_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True
                )
            
            self.model_loaded = True
            logger.info("✅ Med-R1 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Med-R1 model: {e}")
            logger.error("This might be due to:")
            logger.error("1. Missing model checkpoint")
            logger.error("2. Insufficient GPU memory")
            logger.error("3. Missing dependencies (transformers, qwen-vl-utils)")
            logger.error("4. PyTorch version incompatibility")
            self.model_loaded = False
    
    def _load_chat_template_manually(self):
        """Manually load chat template from tokenizer_config.json"""
        try:
            import json
            tokenizer_config_path = os.path.join(self.checkpoint_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'chat_template' in config:
                        self.processor.tokenizer.chat_template = config['chat_template']
                        logger.info("✅ Chat template loaded manually from tokenizer_config.json")
                        return True
                    else:
                        logger.warning("❌ No chat_template found in tokenizer_config.json")
            else:
                logger.warning(f"❌ tokenizer_config.json not found at {tokenizer_config_path}")
        except Exception as e:
            logger.warning(f"Failed to manually load chat template: {e}")
        return False
    
    def _construct_chat_manually(self, prompt: str) -> str:
        """Manually construct chat format for Qwen2VL when chat template is unavailable"""
        # Based on Qwen2VL chat format: https://github.com/QwenLM/Qwen2-VL
        text = "<|im_start|>system\nYou are a helpful assistant specialized in medical image analysis.<|im_end|>\n"
        text += f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text
    
    def _has_flash_attention(self) -> bool:
        """Check if flash attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _prepare_image(self, image_array: np.ndarray) -> Image.Image:
        """Prepare image for Med-R1 inference
        
        Med-R1 expects 384x384 resolution for optimal performance
        """
        # Convert numpy array to PIL Image
        if len(image_array.shape) == 2:
            # Grayscale to RGB
            image = Image.fromarray(image_array).convert('RGB')
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1:
                # Single channel to RGB
                image_array = np.repeat(image_array, 3, axis=2)
            image = Image.fromarray(image_array.astype(np.uint8))
        else:
            raise ValueError(f"Unsupported image shape: {image_array.shape}")
        
        # Resize to Med-R1 optimal resolution (384x384)
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
        
        return image
    
    def generate_caption(self, image_path: str, prompt: str, max_tokens: int = 512) -> dict:
        """Generate caption for medical image using Med-R1
        
        Args:
            image_path: Path to input image file
            prompt: Text prompt for analysis
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with success status, caption, error message, and timings
        """
        import time
        
        start_time = time.time()
        
        if not self.model_loaded:
            return {
                "success": False,
                "caption": "",
                "error": "Med-R1 model not loaded. Please check installation and dependencies.",
                "timings": {"total_time": 0, "text_generation": 0}
            }
        
        try:
            # Load and prepare image from file path
            try:
                image = Image.open(image_path).convert('RGB')
                # Resize to Med-R1 optimal resolution (384x384)
                image = image.resize((384, 384), Image.Resampling.LANCZOS)
            except Exception as e:
                return {
                    "success": False,
                    "caption": "",
                    "error": f"Failed to load image from {image_path}: {str(e)}",
                    "timings": {"total_time": time.time() - start_time, "text_generation": 0}
                }
            
            # Follow the official Med-R1 inference script pattern
            # from: https://github.com/Yuxiang-Lai117/Med-R1
            from qwen_vl_utils import process_vision_info
            
            # Create message format exactly as in official script
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Apply chat template with robust fallback handling
            text = None
            
            # Method 1: Try processor.apply_chat_template
            try:
                if hasattr(self.processor, 'apply_chat_template'):
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    logger.debug("✅ Used processor.apply_chat_template")
                else:
                    raise AttributeError("Processor has no apply_chat_template method")
            except Exception as e:
                logger.warning(f"Processor chat template failed: {e}")
                
            # Method 2: Try tokenizer.apply_chat_template if processor failed
            if text is None:
                try:
                    if hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'apply_chat_template'):
                        # Ensure chat template is loaded
                        if self.processor.tokenizer.chat_template is None:
                            logger.warning("Chat template is None, loading manually...")
                            self._load_chat_template_manually()
                        
                        text = self.processor.tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        logger.debug("✅ Used tokenizer.apply_chat_template")
                    else:
                        raise AttributeError("Tokenizer has no apply_chat_template method")
                except Exception as e:
                    logger.warning(f"Tokenizer chat template failed: {e}")
                    
            # Method 3: Manual construction as final fallback
            if text is None:
                logger.warning("All chat template methods failed, using manual construction")
                text = self._construct_chat_manually(prompt)
                logger.debug("✅ Used manual chat construction")
            
            # Process vision info (as in official script)
            image_inputs, video_inputs = process_vision_info([messages])
            
            # Process inputs (as in official script)
            inputs = self.processor(
                text=[text], 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Generate using official Med-R1 parameters
            generation_start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    use_cache=True,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Official script uses do_sample=False
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            generation_time = time.time() - generation_start
            
            # Decode response (as in official script)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
            response = self.processor.batch_decode(
                trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "caption": response.strip(),
                "error": "",
                "timings": {
                    "total_time": total_time,
                    "text_generation": generation_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error during Med-R1 inference: {e}")
            total_time = time.time() - start_time
            return {
                "success": False,
                "caption": "",
                "error": f"Failed to generate caption - {str(e)}",
                "timings": {
                    "total_time": total_time,
                    "text_generation": 0
                }
            }
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        logger.info("Med-R1 service cleaned up")


def get_service(device: str = "auto", checkpoint_path: str = None) -> MedR1Service:
    """Get or create Med-R1 service instance
    
    Args:
        device: Device to use ("auto", "cuda", "cpu")
        checkpoint_path: Path to model checkpoint (None for auto-detection)
        
    Returns:
        Med-R1 service instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = MedR1Service(device=device, checkpoint_path=checkpoint_path)
    
    return _service_instance


def cleanup_service():
    """Clean up the global service instance"""
    global _service_instance
    
    if _service_instance is not None:
        _service_instance.cleanup()
        _service_instance = None


def is_service_loaded() -> bool:
    """Check if service is loaded and ready"""
    global _service_instance
    return _service_instance is not None and _service_instance.model_loaded


# For backwards compatibility
def get_caption(image_array: np.ndarray, prompt: str, device: str = "auto") -> str:
    """Legacy function for getting captions from numpy arrays"""
    import tempfile
    import os
    
    # Save numpy array as temporary image file
    try:
        # Prepare image
        if len(image_array.shape) == 2:
            # Grayscale to RGB
            image = Image.fromarray(image_array).convert('RGB')
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1:
                # Single channel to RGB
                image_array = np.repeat(image_array, 3, axis=2)
            image = Image.fromarray(image_array.astype(np.uint8))
        else:
            return f"Error: Unsupported image shape: {image_array.shape}"
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Get service and generate caption
            service = get_service(device=device)
            result = service.generate_caption(temp_path, prompt)
            
            if result["success"]:
                return result["caption"]
            else:
                return f"Error: {result['error']}"
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        return f"Error: Failed to process image - {str(e)}"
