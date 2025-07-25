"""
HuggingFace Authentication Helper for SegMed-Pro

This module handles HuggingFace Hub authentication for accessing gated models like MedGemma-4B.
"""

import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_hf_token() -> Optional[str]:
    """
    Get HuggingFace token from various sources
    
    Returns:
        HuggingFace token if found, None otherwise
    """
    # Try environment variable first
    token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if token:
        logger.info("Found HuggingFace token in environment variable")
        return token
    
    # Try HuggingFace CLI token file
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            logger.info("Found HuggingFace token from CLI login")
            return token
    except ImportError:
        logger.debug("huggingface_hub not available for token retrieval")
    except Exception as e:
        logger.debug(f"Could not get token from HuggingFace CLI: {e}")
    
    # Try manual token file (if user created one)
    token_file = Path.home() / ".hf_token"
    if token_file.exists():
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
            if token:
                logger.info("Found HuggingFace token in manual token file")
                return token
        except Exception as e:
            logger.debug(f"Could not read manual token file: {e}")
    
    return None


def login_to_huggingface(token: Optional[str] = None) -> bool:
    """
    Authenticate with HuggingFace Hub
    
    Args:
        token: Optional HuggingFace token. If None, will try to find it automatically
        
    Returns:
        True if authentication successful, False otherwise
    """
    try:
        from huggingface_hub import login, HfFolder
        
        # Get token if not provided
        if token is None:
            token = get_hf_token()
        
        if not token:
            logger.error("No HuggingFace token found. Please provide a token.")
            return False
        
        # Attempt login
        login(token=token, add_to_git_credential=False)
        logger.info("Successfully authenticated with HuggingFace Hub")
        return True
        
    except ImportError:
        logger.error("huggingface_hub not available. Please install it: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace: {e}")
        return False


def check_model_access(model_name: str) -> bool:
    """
    Check if we have access to a specific model
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        True if we have access, False otherwise
    """
    try:
        from huggingface_hub import model_info
        
        # Try to get model info (this will fail if we don't have access)
        info = model_info(model_name)
        logger.info(f"Successfully accessed model info for {model_name}")
        return True
        
    except Exception as e:
        logger.warning(f"Cannot access model {model_name}: {e}")
        return False


def setup_authentication_interactive() -> bool:
    """
    Interactive setup for HuggingFace authentication
    
    Returns:
        True if setup successful, False otherwise
    """
    print("\n=== HuggingFace Authentication Setup ===")
    print()
    print("To access MedGemma-4B, you need to:")
    print("1. Create a HuggingFace account at https://huggingface.co")
    print("2. Request access to google/medgemma-4b-it")
    print("3. Create an access token at https://huggingface.co/settings/tokens")
    print("4. Provide the token below")
    print()
    
    token = input("Please enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    # Test the token
    success = login_to_huggingface(token)
    
    if success:
        # Save token to manual file for future use
        try:
            token_file = Path.home() / ".hf_token"
            with open(token_file, 'w') as f:
                f.write(token)
            print(f"✅ Token saved to {token_file}")
        except Exception as e:
            print(f"⚠️ Could not save token: {e}")
        
        print("✅ Authentication successful!")
        return True
    else:
        print("❌ Authentication failed")
        return False


def print_authentication_instructions():
    """Print instructions for HuggingFace authentication"""
    print("\n" + "="*60)
    print("HUGGINGFACE AUTHENTICATION REQUIRED")
    print("="*60)
    print()
    print("MedGemma-4B is a gated model. To access it, you need to:")
    print()
    print("1. 🌐 Visit: https://huggingface.co/google/medgemma-4b-it")
    print("2. 📝 Request access to the model (may take time for approval)")
    print("3. 🔑 Create a token at: https://huggingface.co/settings/tokens")
    print("4. 💻 Set up authentication using one of these methods:")
    print()
    print("   Method 1 - Environment Variable:")
    print("   export HUGGINGFACE_HUB_TOKEN='your_token_here'")
    print()
    print("   Method 2 - HuggingFace CLI:")
    print("   pip install huggingface_hub")
    print("   huggingface-cli login")
    print()
    print("   Method 3 - Manual token file:")
    print("   echo 'your_token_here' > ~/.hf_token")
    print()
    print("   Method 4 - Interactive setup:")
    print("   Run: python models/medgemma/hf_auth.py")
    print()
    print("="*60)


if __name__ == "__main__":
    # Interactive setup when run as script
    print("HuggingFace Authentication Setup for MedGemma-4B")
    success = setup_authentication_interactive()
    
    if success:
        # Test access to MedGemma
        print("\nTesting access to MedGemma-4B...")
        if check_model_access("google/medgemma-4b-it"):
            print("✅ Successfully authenticated and can access MedGemma-4B!")
        else:
            print("❌ Authentication successful but cannot access MedGemma-4B.")
            print("   Make sure you have requested access to the model.")
    else:
        print_authentication_instructions()
