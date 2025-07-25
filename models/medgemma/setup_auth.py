#!/usr/bin/env python3
"""
Quick Setup Script for MedGemma-4B Authentication

This script helps users set up HuggingFace authentication for MedGemma-4B.
"""

import os
import sys
from pathlib import Path

def main():
    print("🤖 MedGemma-4B Authentication Setup")
    print("=" * 50)
    print()
    
    # Add the current directory to Python path so we can import hf_auth
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    try:
        from hf_auth import setup_authentication_interactive, check_model_access
        
        print("Setting up authentication for google/medgemma-4b-it...")
        print()
        
        success = setup_authentication_interactive()
        
        if success:
            print()
            print("Testing access to MedGemma-4B...")
            if check_model_access("google/medgemma-4b-it"):
                print("✅ Success! You can now use MedGemma-4B in SegMed-Pro.")
                print()
                print("Next steps:")
                print("1. Run: python models/medgemma/install_medgemma.py")
                print("2. Or start SegMed-Pro: python app.py")
            else:
                print("⚠️ Authentication successful but model access denied.")
                print("Please request access at: https://huggingface.co/google/medgemma-4b-it")
                print("Access approval may take some time.")
        else:
            print("❌ Authentication setup failed.")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure you're running this from the correct directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
