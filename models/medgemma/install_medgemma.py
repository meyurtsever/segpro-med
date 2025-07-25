"""
MedGemma-4B Installation Script for SegMed-Pro

This script installs the required dependencies and downloads the MedGemma-4B model.
Run this script to set up MedGemma-4B for the first time.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required Python packages"""
    try:
        logger.info("Installing MedGemma dependencies...")
        
        # Required packages for MedGemma
        packages = [
            "torch>=2.0.0",
            "transformers>=4.36.0", 
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
            "protobuf>=3.20.0",
            "huggingface_hub>=0.17.0",
            "tqdm>=4.64.0"
        ]
        
        for package in packages:
            logger.info(f"Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to install {package}: {result.stderr}")
                return False
        
        logger.info("Dependencies installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def download_model():
    """Download the MedGemma-4B model"""
    try:
        logger.info("Downloading MedGemma-4B model...")
        
        # Import and run the download function
        from download_medgemma import download_medgemma_model
        success = download_medgemma_model(force_redownload=False)
        
        if success:
            logger.info("MedGemma-4B model downloaded successfully")
            return True
        else:
            logger.error("Failed to download MedGemma-4B model")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False


def verify_installation():
    """Verify that MedGemma is properly installed"""
    try:
        logger.info("Verifying MedGemma installation...")
        
        # Try to import and initialize the service
        from medgemma_service import MedGemmaService
        
        # Create a test service instance (this will test model loading)
        service = MedGemmaService(device="cpu")  # Use CPU for testing to avoid GPU issues
        
        # Clean up
        service.cleanup()
        
        logger.info("MedGemma installation verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Installation verification failed: {e}")
        return False


def setup_authentication():
    """Set up HuggingFace authentication"""
    try:
        print("Step 1.5: Setting up HuggingFace authentication...")
        
        from hf_auth import login_to_huggingface, setup_authentication_interactive, check_model_access
        
        # Try automatic authentication first
        if login_to_huggingface():
            # Check if we have access to MedGemma
            if check_model_access("google/medgemma-4b-it"):
                logger.info("HuggingFace authentication successful")
                return True
            else:
                print("⚠️ Authenticated but no access to MedGemma-4B model")
                print("Please request access at: https://huggingface.co/google/medgemma-4b-it")
                return False
        else:
            # Interactive setup
            print("Authentication required for MedGemma-4B (gated model)")
            return setup_authentication_interactive()
            
    except Exception as e:
        logger.error(f"Error setting up authentication: {e}")
        return False


def main():
    """Main installation process"""
    print("=== MedGemma-4B Installation for SegMed-Pro ===")
    print()
    
    # Step 1: Install dependencies
    print("Step 1: Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    print("✅ Dependencies installed")
    print()
    
    # Step 1.5: Set up authentication
    if not setup_authentication():
        print("❌ Failed to set up authentication")
        sys.exit(1)
    print("✅ Authentication configured")
    print()
    
    # Step 2: Download model
    print("Step 2: Downloading MedGemma-4B model...")
    print("Note: This may take several minutes depending on your internet connection")
    if not download_model():
        print("❌ Failed to download model")
        sys.exit(1)
    print("✅ Model downloaded")
    print()
    
    # Step 3: Verify installation
    print("Step 3: Verifying installation...")
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    print("✅ Installation verified")
    print()
    
    print("🎉 MedGemma-4B installation completed successfully!")
    print()
    print("You can now use MedGemma-4B in SegMed-Pro:")
    print("1. Start SegMed-Pro: python app.py")
    print("2. Go to the Editor tab")
    print("3. Select 'MedGemma-4B' from the VLM Model dropdown")
    print("4. Click 'Get Medical Analysis' to analyze MRI slices")


if __name__ == "__main__":
    main()
