"""
SmolVLM Persistent Service for SegMed-Pro

This service keeps the SmolVLM model loaded in memory to avoid the expensive
model loading time on each inference request. It provides much faster response
times compared to the CLI approach.
"""

import torch
import time
import json
import logging
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmolVLMService:
    """Persistent SmolVLM service that keeps the model loaded"""
    
    def __init__(self, device="auto"):
        """Initialize the service with model loading"""
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.processor = None
        self.model = None
        self.model_loaded = False
        self.uses_device_map = False  # Track if we're using device_map
        
        logger.info(f"Initializing SmolVLM Service on device: {self.device}")
        self._ensure_model_available()
        self._load_model()

    def _ensure_model_available(self):
        """Auto-download SmolVLM if not already cached."""
        try:
            from download_smolvlm import ensure_smolvlm_available
            ensure_smolvlm_available()
        except Exception as e:
            logger.debug(f"Pre-download check skipped: {e} — will attempt loading directly.")

    def _load_model(self):
        """Load the SmolVLM model and processor"""
        try:
            start_time = time.time()
            logger.info("Loading SmolVLM model and processor...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
            
            # Load model with optimizations
            if self.device == "cuda":
                self.model = AutoModelForVision2Seq.from_pretrained(
                    "HuggingFaceTB/SmolVLM-Instruct",
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",
                    device_map="auto",
                )
                self.uses_device_map = True
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    "HuggingFaceTB/SmolVLM-Instruct",
                    torch_dtype=torch.float32,
                    _attn_implementation="eager",
                ).to(self.device)
                self.uses_device_map = False
            
            # Performance optimizations for CUDA
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                try:
                    self.model = torch.compile(self.model)
                    logger.info("✓ Model compiled for optimal performance")
                except Exception:
                    logger.info("✓ Model loaded (compilation not available)")
            
            load_time = time.time() - start_time
            logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load SmolVLM model: {e}")
            self.model_loaded = False
            raise
    
    def generate_caption(self, image_path: str, prompt: str, max_tokens: int = 100) -> dict:
        """
        Generate caption for an image with the given prompt
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            dict: Result containing caption and timing information
        """
        if not self.model_loaded:
            return {
                "success": False,
                "error": "Model not loaded",
                "caption": "",
                "timings": {}
            }
        
        try:
            total_start = time.time()
            
            # Load image
            image_start = time.time()
            if image_path.startswith(('http://', 'https://')):
                image = load_image(image_path)
            else:
                image_path_obj = Path(image_path)
                if not image_path_obj.exists():
                    return {
                        "success": False,
                        "error": f"Image file not found: {image_path}",
                        "caption": "",
                        "timings": {}
                    }
                image = Image.open(image_path_obj).convert('RGB')
            
            image_load_time = time.time() - image_start
            
            # Prepare inputs
            processing_start = time.time()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                },
            ]
            
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt_text, images=[image], return_tensors="pt")
            
            # Handle device placement based on whether we're using device_map
            if not self.uses_device_map:
                inputs = inputs.to(self.device)
            # If using device_map, don't move inputs - let the model handle device placement
            
            processing_time = time.time() - processing_start
            
            # Generate response
            generation_start = time.time()
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.autocast(device_type="cuda"):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=0.4,
                            top_p=0.8,
                            repetition_penalty=1.2,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.8,
                        repetition_penalty=1.2,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True,
                    )
                
                generated_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - total_start
            
            # Extract the response (remove the prompt part)
            full_response = generated_texts[0]
            # Find where the actual response starts (after the prompt)
            if "User:" in full_response and "Assistant:" in full_response:
                response_parts = full_response.split("Assistant:")
                if len(response_parts) > 1:
                    caption = response_parts[-1].strip()
                else:
                    caption = full_response.strip()
            else:
                caption = full_response.strip()
            
            return {
                "success": True,
                "error": "",
                "caption": caption,
                "timings": {
                    "image_loading": image_load_time,
                    "input_processing": processing_time,
                    "text_generation": generation_time,
                    "total_time": total_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return {
                "success": False,
                "error": str(e),
                "caption": "",
                "timings": {}
            }
    
    def cleanup(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SmolVLM Service cleaned up")


# Global service instance
_service_instance = None

def get_service(device="auto"):
    """Get or create the global service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = SmolVLMService(device=device)
    return _service_instance

def cleanup_service():
    """Cleanup the global service instance"""
    global _service_instance
    if _service_instance is not None:
        _service_instance.cleanup()
        _service_instance = None


if __name__ == "__main__":
    # Test the service
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="SmolVLM Service Test")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--message", "-m", default="Describe this image", help="Text prompt")
    parser.add_argument("--max-tokens", "-t", type=int, default=100, help="Max tokens")
    parser.add_argument("--device", "-d", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    try:
        service = get_service(args.device)
        result = service.generate_caption(args.image_path, args.message, args.max_tokens)
        
        if result["success"]:
            timings = result["timings"]
            print(f"Image loading: {timings['image_loading']:.2f}s")
            print(f"Processing: {timings['input_processing']:.2f}s")
            print(f"Generation: {timings['text_generation']:.2f}s")
            print(f"Total: {timings['total_time']:.2f}s")
            print("\nResponse:")
            print("=" * 50)
            print(result["caption"])
            print("=" * 50)
        else:
            print(f"Error: {result['error']}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cleanup_service()
