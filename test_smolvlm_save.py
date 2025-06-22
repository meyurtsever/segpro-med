#!/usr/bin/env python3
"""
Test script for SmolVLM image saving functionality
"""

import os
import sys
import numpy as np
from PIL import Image

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Try to import the handler
try:
    from ui.smolvlm_handlers import SmolVLMHandlers
    from ui.state import AppState
    print("✅ Successfully imported SmolVLMHandlers")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_image_saving():
    """Test the image saving functionality"""
    print("🧪 Testing SmolVLM image saving functionality...")
    
    # Create a test image (grayscale medical-like image)
    test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # Create a mock image_annotator value
    mock_annotator_value = {
        "image": test_image,
        "annotations": []
    }
    
    # Initialize the handler
    state = AppState()
    handler = SmolVLMHandlers(state)
    
    print("🖼️ Created test image with shape:", test_image.shape)
    
    # Test the save functionality (this will try to run VLM inference but will save the image)
    result = handler.run_vlm_inference(
        mock_annotator_value, 
        identify_anomalies=True, 
        describe_slice=False
    )
    
    print("📝 VLM inference result:", result[:100] + "..." if len(result) > 100 else result)
    
    # Check if the image was saved
    expected_path = "latest_processed_smolvlm.jpg"
    if os.path.exists(expected_path):
        print(f"✅ Image successfully saved to: {expected_path}")
        # Get file size
        file_size = os.path.getsize(expected_path)
        print(f"📄 File size: {file_size} bytes")
        
        # Try to open and verify the image
        try:
            saved_image = Image.open(expected_path)
            print(f"🖼️ Saved image size: {saved_image.size}")
            print(f"🎨 Saved image mode: {saved_image.mode}")
            print("✅ Image verification successful")
        except Exception as e:
            print(f"❌ Error verifying saved image: {e}")
    else:
        print(f"❌ Image was not saved to expected path: {expected_path}")

if __name__ == "__main__":
    test_image_saving()
