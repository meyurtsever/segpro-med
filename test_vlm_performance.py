"""
Test script to validate the fast SmolVLM integration using persistent service
"""

import sys
import os
import time
import numpy as np
from PIL import Image

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from ui.smolvlm_handlers import SmolVLMHandlers

def create_test_image():
    """Create a simple test brain MRI-like image"""
    # Create a 256x256 grayscale image that looks like a brain slice
    img_array = np.zeros((256, 256), dtype=np.uint8)
    
    # Create circular brain outline
    center = (128, 128)
    radius = 100
    
    for y in range(256):
        for x in range(256):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < radius:
                # Brain tissue
                img_array[y, x] = 120 + int(30 * np.sin(x/10) * np.cos(y/10))
            elif dist < radius + 5:
                # Brain outline
                img_array[y, x] = 200
    
    # Add some "abnormal" bright spots
    img_array[80:90, 80:90] = 255  # Bright spot
    img_array[150:160, 120:130] = 255  # Another bright spot
    
    return img_array

def test_vlm_performance():
    """Test VLM performance with both prompt modes"""
    print("Testing SmolVLM Performance with Persistent Service")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    
    # Create mock image_annotator_value
    image_annotator_value = {
        "image": test_image
    }
    
    # Initialize handler
    print("1. Initializing SmolVLM handler...")
    init_start = time.time()
    
    class MockState:
        pass
    
    handler = SmolVLMHandlers(MockState())
    init_time = time.time() - init_start
    print(f"   Handler initialized in {init_time:.2f}s")
    
    # Test 1: Anomaly identification prompt
    print("\n2. Testing 'Identify Anomalies' prompt...")
    test1_start = time.time()
    
    result1 = handler.run_vlm_inference(
        image_annotator_value,
        identify_anomalies=True,
        describe_slice=False
    )
    
    test1_time = time.time() - test1_start
    print(f"   ✓ First inference completed in {test1_time:.2f}s")
    print(f"   Result: {result1[:100]}...")
    
    # Test 2: Description prompt (should be much faster as model is already loaded)
    print("\n3. Testing 'Describe Slice' prompt...")
    test2_start = time.time()
    
    result2 = handler.run_vlm_inference(
        image_annotator_value,
        identify_anomalies=False,
        describe_slice=True
    )
    
    test2_time = time.time() - test2_start
    print(f"   ✓ Second inference completed in {test2_time:.2f}s")
    print(f"   Result: {result2[:100]}...")
    
    # Test 3: Another inference to confirm caching works
    print("\n4. Testing third inference (should be fastest)...")
    test3_start = time.time()
    
    result3 = handler.run_vlm_inference(
        image_annotator_value,
        identify_anomalies=True,
        describe_slice=False
    )
    
    test3_time = time.time() - test3_start
    print(f"   ✓ Third inference completed in {test3_time:.2f}s")
    print(f"   Result: {result3[:100]}...")
    
    # Summary
    total_time = time.time() - init_start
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Handler initialization: {init_time:.2f}s")
    print(f"First inference:        {test1_time:.2f}s (includes model loading)")
    print(f"Second inference:       {test2_time:.2f}s (model already loaded)")
    print(f"Third inference:        {test3_time:.2f}s (model already loaded)")
    print(f"Total time:             {total_time:.2f}s")
    
    # Performance feedback
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if test1_time > 15:
        print("⚠️  First inference took >15s - model loading is still slow")
        print("   Check if the persistent service is being used correctly")
    elif test1_time > 5:
        print("⚠️  First inference took >5s - acceptable but could be better")
        print("   Model loading is happening but relatively fast")
    else:
        print("✅ First inference <5s - Great! Service may already be running")
    
    if test2_time < 3:
        print("✅ Second inference <3s - Excellent! Model caching is working")
    elif test2_time < 5:
        print("✅ Second inference <5s - Good! Model is cached")
    else:
        print("⚠️  Second inference >5s - Model may not be properly cached")
    
    if test3_time < test2_time:
        print("✅ Third inference faster than second - Perfect caching!")
    else:
        print("ℹ️  Third inference similar to second - Normal behavior")
    
    # Cleanup
    print("\n5. Cleaning up...")
    handler.cleanup()
    print("   ✓ Cleanup completed")

if __name__ == "__main__":
    try:
        test_vlm_performance()
        print("\n🎉 VLM integration test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
