"""
MedGemma-4B Model Download Script for SegMed-Pro

This script downloads the Google MedGemma-4B model checkpoints to the local models directory
if they are not already available. It handles the download process with progress tracking
and error handling.
"""

import os
import sys
import logging
from pathlib import Path
import shutil
from typing import Optional
import requests
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_models_directory() -> Path:
    """Get the models directory path"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # Navigate to the models directory
    models_dir = script_dir.parent.parent / "models" / "medgemma"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def check_model_exists(models_dir: Path) -> bool:
    """
    Check if MedGemma-4B model files already exist locally
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        True if model exists, False otherwise
    """
    try:
        # Check for essential model files
        required_files = [
            "config.json",
            "pytorch_model.bin",  # or model.safetensors
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        # Check if at least config.json exists (indicates model was downloaded)
        config_file = models_dir / "config.json"
        if config_file.exists():
            logger.info("MedGemma-4B model files found locally")
            return True
        
        logger.info("MedGemma-4B model files not found locally")
        return False
        
    except Exception as e:
        logger.error(f"Error checking for existing model: {e}")
        return False


def download_with_huggingface_hub(models_dir: Path, model_name: str = "google/medgemma-4b-it") -> bool:
    """
    Download model using huggingface_hub (preferred method)
    
    Args:
        models_dir: Directory to save the model
        model_name: HuggingFace model identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading {model_name} using huggingface_hub...")
        logger.info(f"Target directory: {models_dir}")
        
        # Download the entire model repository
        snapshot_download(
            repo_id=model_name,
            local_dir=models_dir,
            local_dir_use_symlinks=False,  # Use actual files instead of symlinks
            resume_download=True,  # Resume if partially downloaded
            ignore_patterns=["*.md", "*.txt", ".git*"]  # Skip unnecessary files
        )
        
        logger.info("Model downloaded successfully using huggingface_hub")
        return True
        
    except ImportError:
        logger.warning("huggingface_hub not available, trying alternative method")
        return False
    except Exception as e:
        logger.error(f"Error downloading with huggingface_hub: {e}")
        return False


def download_with_transformers(models_dir: Path, model_name: str = "google/medgemma-4b-it") -> bool:
    """
    Download model using transformers library (fallback method)
    
    Args:
        models_dir: Directory to save the model
        model_name: HuggingFace model identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        
        logger.info(f"Downloading {model_name} using transformers...")
        logger.info(f"Target directory: {models_dir}")
        
        # Download config first
        logger.info("Downloading model configuration...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.save_pretrained(models_dir)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(models_dir)
        
        # Download model (this will be the largest download)
        logger.info("Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",  # Let it choose appropriate dtype
            low_cpu_mem_usage=True  # Reduce memory usage during download
        )
        model.save_pretrained(models_dir)
        
        logger.info("Model downloaded successfully using transformers")
        return True
        
    except ImportError as e:
        logger.error(f"transformers library not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error downloading with transformers: {e}")
        return False


def verify_download(models_dir: Path) -> bool:
    """
    Verify that the downloaded model is complete and valid
    
    Args:
        models_dir: Directory containing the model
        
    Returns:
        True if model is valid, False otherwise
    """
    try:
        logger.info("Verifying downloaded model...")
        
        # Check for essential files
        essential_files = ["config.json"]
        for file_name in essential_files:
            file_path = models_dir / file_name
            if not file_path.exists():
                logger.error(f"Essential file missing: {file_name}")
                return False
        
        # Try to load the config to verify it's valid
        config_path = models_dir / "config.json"
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        if "model_type" not in config:
            logger.error("Invalid model configuration")
            return False
        
        logger.info("Model verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model: {e}")
        return False


def cleanup_incomplete_download(models_dir: Path):
    """
    Clean up incomplete downloads
    
    Args:
        models_dir: Directory to clean
    """
    try:
        if models_dir.exists():
            logger.info("Cleaning up incomplete download...")
            shutil.rmtree(models_dir)
            models_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning up: {e}")


def download_medgemma_model(force_redownload: bool = False) -> bool:
    """
    Main function to download MedGemma-4B model
    
    Args:
        force_redownload: If True, redownload even if model exists
        
    Returns:
        True if model is available (either already exists or successfully downloaded)
    """
    try:
        models_dir = get_models_directory()
        model_name = "google/medgemma-4b-it"
        
        logger.info(f"MedGemma-4B Download Script")
        logger.info(f"Models directory: {models_dir}")
        
        # Ensure HuggingFace authentication
        try:
            from hf_auth import login_to_huggingface, check_model_access, print_authentication_instructions
            
            # Try to authenticate
            auth_success = login_to_huggingface()
            if not auth_success:
                logger.error("HuggingFace authentication failed")
                print_authentication_instructions()
                return False
            
            # Check if we have access to the model
            if not check_model_access(model_name):
                logger.error(f"No access to model {model_name}")
                print_authentication_instructions()
                return False
                
        except ImportError:
            logger.error("Authentication module not available")
            return False
        
        # Check if model already exists
        if not force_redownload and check_model_exists(models_dir):
            logger.info("MedGemma-4B model already exists locally")
            return True
        
        if force_redownload:
            logger.info("Force redownload requested")
            cleanup_incomplete_download(models_dir)
        
        # Ensure we have the required dependencies
        missing_deps = []
        try:
            import torch
        except ImportError:
            missing_deps.append("torch")
        
        try:
            import transformers
        except ImportError:
            missing_deps.append("transformers")
        
        if missing_deps:
            logger.error(f"Missing required dependencies: {missing_deps}")
            logger.error("Please install them using: pip install torch transformers")
            return False
        
        # Try downloading with huggingface_hub first (preferred)
        success = download_with_huggingface_hub(models_dir, model_name)
        
        # Fallback to transformers if huggingface_hub failed
        if not success:
            logger.info("Trying fallback download method...")
            success = download_with_transformers(models_dir, model_name)
        
        if not success:
            logger.error("All download methods failed")
            cleanup_incomplete_download(models_dir)
            return False
        
        # Verify the download
        if not verify_download(models_dir):
            logger.error("Model verification failed")
            cleanup_incomplete_download(models_dir)
            return False
        
        logger.info("MedGemma-4B model download completed successfully!")
        logger.info(f"Model saved to: {models_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in download process: {e}")
        return False


def main():
    """Main entry point for the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MedGemma-4B model for SegMed-Pro")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force redownload even if model exists"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true", 
        help="Only check if model exists, don't download"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        models_dir = get_models_directory()
        exists = check_model_exists(models_dir)
        if exists:
            print(f"✓ MedGemma-4B model found at: {models_dir}")
            sys.exit(0)
        else:
            print(f"✗ MedGemma-4B model not found at: {models_dir}")
            sys.exit(1)
    
    success = download_medgemma_model(force_redownload=args.force)
    
    if success:
        print("✓ MedGemma-4B model is ready for use!")
        sys.exit(0)
    else:
        print("✗ Failed to download MedGemma-4B model")
        sys.exit(1)


if __name__ == "__main__":
    main()
