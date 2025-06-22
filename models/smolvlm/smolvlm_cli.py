import torch
import time
import argparse
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="SmolVLM Image Analysis")
    parser.add_argument("image_path", 
                       help="Path to the image file (local path or URL)")
    parser.add_argument("--message", "-m", 
                       default="Can you describe the image?",
                       help="Text message/question about the image (default: 'Can you describe the image?')")
    parser.add_argument("--device", "-d", 
                       choices=["auto", "cuda", "cpu"], 
                       default="auto",
                       help="Device to use (default: auto)")
    parser.add_argument("--max-tokens", "-t", 
                       type=int, 
                       default=512,
                       help="Maximum number of tokens to generate (default: 512)")
    
    args = parser.parse_args()
    
    # Start total timer
    total_start_time = time.time()
    
    # Device selection
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
        
    print(f"Using device: {DEVICE}")
    print(f"Image: {args.image_path}")
    print(f"Message: '{args.message}'")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 60)
    
    try:
        # Load image
        print("Loading image...")
        image_start_time = time.time()
        
        # Check if it's a local file or URL
        if args.image_path.startswith(('http://', 'https://')):
            image = load_image(args.image_path)
            print(f"✓ Image loaded from URL")
        else:
            image_path = Path(args.image_path)
            if not image_path.exists():
                print(f"Error: Image file '{args.image_path}' not found!")
                sys.exit(1)
            image = Image.open(image_path).convert('RGB')
            print(f"✓ Image loaded from local file")
            
        image_load_time = time.time() - image_start_time
        print(f"✓ Image loaded in {image_load_time:.2f} seconds")
        
        # Initialize processor and model
        print("Loading SmolVLM model and processor...")
        model_start_time = time.time()
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-Instruct",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            _attn_implementation="eager",  # Using eager attention for compatibility
            device_map="auto" if DEVICE == "cuda" else None,
        ).to(DEVICE)
        
        # Performance optimizations for CUDA
        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                model = torch.compile(model)
                print("✓ Model compiled for optimal performance")
            except Exception:
                print("✓ Model loaded (compilation not available)")
        
        model_load_time = time.time() - model_start_time
        print(f"✓ Model loaded in {model_load_time:.2f} seconds")
        
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.message}
                ]
            },
        ]
        
        # Prepare inputs
        print("Processing inputs...")
        processing_start_time = time.time()
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)
        processing_time = time.time() - processing_start_time
        print(f"✓ Inputs processed in {processing_time:.2f} seconds")
        
        # Generate outputs with optimized parameters
        print("Generating response...")
        generation_start_time = time.time()
        
        with torch.no_grad():
            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda"):
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.8,
                        repetition_penalty=1.2,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                    '''generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        #do_sample=True,
                        #temperature=0.4,
                        #top_p=0.8,
                        #repetition_penalty=1.2,
                        #pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True
                    )'''
            else:
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.8,
                    repetition_penalty=1.2,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True,
                )
            
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
        
        generation_time = time.time() - generation_start_time
        print(f"✓ Response generated in {generation_time:.2f} seconds")
        
        # Calculate total time
        total_time = time.time() - total_start_time
        
        # Results
        print("\n" + "="*60)
        print("TIMING SUMMARY:")
        print("="*60)
        print(f"Image loading:     {image_load_time:.2f}s")
        print(f"Model loading:     {model_load_time:.2f}s") 
        print(f"Input processing:  {processing_time:.2f}s")
        print(f"Text generation:   {generation_time:.2f}s")
        print(f"TOTAL TIME:        {total_time:.2f}s")
        print("="*60)
        
        # Performance feedback
        if generation_time <= 3.0:
            print("🚀 EXCELLENT! Generation time under 3 seconds!")
        elif generation_time <= 5.0:
            print("✅ GOOD! Generation time under 5 seconds")
        else:
            print("⏰ Consider GPU acceleration for faster inference")
        
        print("\n" + "="*50)
        print("SMOLVLM RESPONSE:")
        print("="*50)
        print(generated_texts[0])
        print("="*50)
        
        # Memory cleanup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
