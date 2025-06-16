#!/usr/bin/env python3
"""
Test script to verify SAM2 Fast Masking integration
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        from brain_segmentation_configs import get_brain_config, BRAIN_CONFIGS
        print("✅ brain_segmentation_configs imported successfully")
        
        # Test configuration
        config = get_brain_config('fast')
        print(f"✅ Fast config loaded: {config['points_per_side']}x{config['points_per_side']} points")
        
    except ImportError as e:
        print(f"❌ Error importing brain_segmentation_configs: {e}")
        return False
    
    try:
        from ui.medsam2_handlers import MEDSAM2Handlers
        print("✅ MEDSAM2Handlers imported successfully")
        
    except ImportError as e:
        print(f"❌ Error importing MEDSAM2Handlers: {e}")
        return False
    
    try:        # Test if models exist
        models_dir = os.path.join(current_dir, "models", "medsam2")
        config_path = os.path.join(models_dir, "configs", "sam2.1_hiera_b+.yaml")
        checkpoint_path = os.path.join(models_dir, "checkpoints", "sam2.1_hiera_base_plus.pt")
        
        if os.path.exists(config_path):
            print("✅ SAM2 config file found")
        else:
            print(f"⚠️ SAM2 config file not found at: {config_path}")
        
        if os.path.exists(checkpoint_path):
            print("✅ SAM2 checkpoint file found")
        else:
            print(f"⚠️ SAM2 checkpoint file not found at: {checkpoint_path}")
            
    except Exception as e:
        print(f"❌ Error checking model files: {e}")
        return False
    
    return True

def test_dicom_folder():
    """Test if DICOM folder exists"""
    dicom_folder = os.path.join(current_dir, "cvm_48_t1")
    if os.path.exists(dicom_folder):
        dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
        print(f"✅ DICOM folder found with {len(dicom_files)} files")
        return True
    else:
        print(f"⚠️ DICOM folder not found at: {dicom_folder}")
        return False

if __name__ == "__main__":
    print("SAM2 Fast Masking Integration Test")
    print("=" * 50)
    
    imports_ok = test_imports()
    dicom_ok = test_dicom_folder()
    
    print("\n" + "=" * 50)
    if imports_ok and dicom_ok:
        print("✅ All tests passed! Integration should work correctly.")
        print("\nTo use SAM2 Fast Masking:")
        print("1. Run the main application: python app.py")
        print("2. Load DICOM data from a directory")
        print("3. Click 'Run SAM2 Fast Masking' button")
        print("4. Check 'Save Visualizations' if you want to save mask images")
        print("5. Select 'Single Slice' or 'All Records' processing mode")
    else:
        print("❌ Some tests failed. Please check the issues above.")
