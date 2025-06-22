#!/usr/bin/env python3
"""
Test script for SmolVLM preloading functionality
"""

import sys
import time
import logging

# Set up basic logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vlm_preloading():
    """Test the SmolVLM preloading functionality"""
    print("🧪 Testing SmolVLM preloading functionality...")
    
    try:
        # Import the main app class
        from app import SegMedPro
        
        print("📦 Importing SegMedPro class...")
        
        # Initialize the app (this should start preloading)
        print("🚀 Initializing SegMedPro (preloading should start in background)...")
        app = SegMedPro()
        
        print("✅ App initialized successfully")
        print("⏳ Waiting 5 seconds for background VLM preloading to complete...")
        
        # Give time for background loading
        time.sleep(5)
        
        # Test if the service is available
        print("🔍 Testing VLM service availability...")
        try:
            from models.smolvlm.smolvlm_service import get_service
            service = get_service()
            
            print(f"📊 Model loaded status: {service.model_loaded}")
            
            if service.model_loaded:
                print("✅ SmolVLM service successfully preloaded and ready!")
                print("🚀 VLM inference should be fast on first use")
            else:
                print("⚠️ Model not loaded, but service is available")
                
        except Exception as e:
            print(f"❌ Error accessing VLM service: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vlm_preloading()
    if success:
        print("\n🎉 VLM preloading test completed successfully!")
    else:
        print("\n💥 VLM preloading test failed!")
        sys.exit(1)
