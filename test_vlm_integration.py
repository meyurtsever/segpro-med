"""
Test script for SmolVLM integration in SegMed-Pro

This script tests the VLM functionality without running the full Gradio interface.
"""

import sys
import os
import numpy as np
from PIL import Image

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from ui.smolvlm_handlers import SmolVLMHandlers
from ui.state import AppState

def test_vlm_integration():
    """Test the VLM integration with a sample medical image"""
    
    print("Testing SmolVLM integration...")
    
    # Initialize handlers
    state = AppState()
    vlm_handlers = SmolVLMHandlers(state)
    
    # Create a sample medical image (simulated)
    # This creates a simple grayscale image that looks like a medical scan
    width, height = 512, 512
    image_array = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Add some structure to make it look more medical-like
    # Add circular structure (simulating brain cross-section)
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    mask = (x - center_x)**2 + (y - center_y)**2 < (width//3)**2
    image_array[mask] = image_array[mask] * 0.7 + 100  # Darker center
    
    # Convert to RGB for VLM
    image_rgb = np.stack([image_array] * 3, axis=-1)
    
    # Create the image_annotator value format
    annotator_value = {
        "image": image_rgb,
        "boxes": []
    }
    
    print(f"Created test image with shape: {image_rgb.shape}")
    print(f"Image data type: {image_rgb.dtype}")
    print(f"Image value range: {image_rgb.min()} - {image_rgb.max()}")
    
    # Test both prompt modes
    print("\nTesting 'Identify Anomalies' prompt...")
    result_anomalies = vlm_handlers.run_vlm_inference(annotator_value, identify_anomalies=True, describe_slice=False)
    
    print(f"\nVLM Result (Anomalies):")
    print("=" * 50)
    print(result_anomalies)
    print("=" * 50)
    
    print("\nTesting 'Describe MRI Slice' prompt...")
    result_describe = vlm_handlers.run_vlm_inference(annotator_value, identify_anomalies=False, describe_slice=True)
    
    print(f"\nVLM Result (Description):")
    print("=" * 50)
    print(result_describe)
    print("=" * 50)
    
    return result_anomalies, result_describe

def test_actual_image():
    """Test with an actual image file if available"""
    
    # Look for sample DICOM data
    sample_paths = [
        "cvm_48_t1",
        "cvm_t1", 
        "automatic_mask_test_results"
    ]
    
    for path in sample_paths:
        if os.path.exists(path):
            print(f"\nFound sample data directory: {path}")
            # For now, just note that it exists
            # In a real test, we could load a DICOM slice here
            break
    else:
        print("\nNo sample medical data found, using synthetic image only")

if __name__ == "__main__":
    print("SmolVLM Integration Test")
    print("=" * 40)
    
    # Check if SmolVLM CLI exists
    smolvlm_path = os.path.join("models", "smolvlm", "smolvlm_cli.py")
    if os.path.exists(smolvlm_path):
        print(f"✓ SmolVLM CLI found at: {smolvlm_path}")
    else:
        print(f"✗ SmolVLM CLI not found at: {smolvlm_path}")
        print("Please ensure SmolVLM is properly installed")
        sys.exit(1)
    
    try:
        result_anomalies, result_describe = test_vlm_integration()
        
        if result_anomalies.startswith("Error") or result_describe.startswith("Error"):
            print(f"\n❌ VLM integration test failed:")
            if result_anomalies.startswith("Error"):
                print(f"Anomalies prompt: {result_anomalies}")
            if result_describe.startswith("Error"):
                print(f"Describe prompt: {result_describe}")
            sys.exit(1)
        else:
            print(f"\n✅ VLM integration test passed!")
            print("The VLM successfully generated captions for both prompt types.")
    
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    test_actual_image()
    
    print(f"\n🎉 All tests completed!")
    print("You can now use the VLM feature in the Editor tab by:")
    print("1. Loading medical data")
    print("2. Navigating to a slice")  
    print("3. Selecting a prompt mode:")
    print("   - 'Identify Anomalies': Focus on abnormal regions and pathology")
    print("   - 'Describe MRI Slice': General anatomical structure description")
    print("4. Clicking the 'Run SmolVLM' button")
    print("5. Reading the generated caption based on your selected prompt")
