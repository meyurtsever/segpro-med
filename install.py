#!/usr/bin/env python
import subprocess
import sys
import os

def check_dependency(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_dependencies():
    print("Installing SegMed-Pro dependencies...")
    
    # Install pip dependencies from requirements.txt
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Special handling for GDCM (can be tricky to install)
    if not check_dependency("gdcm"):
        print("Attempting alternative installation for GDCM...")
        try:
            # Try conda installation if available
            if check_dependency("conda"):
                subprocess.check_call(["conda", "install", "-c", "conda-forge", "gdcm"])
            else:
                # Fall back to pip with specific options
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdcm", "--find-links", 
                                      "https://github.com/SimpleITK/SimpleITK/releases/tag/v2.1.0"])
        except Exception as e:
            print(f"GDCM installation had issues: {str(e)}")
            print("Note: You may need to install GDCM manually for your specific platform.")
            print("See: https://pydicom.github.io/pydicom/stable/tutorials/installation.html#installing-gdcm")
    
    # Verify DICOM compression support
    try:
        import pydicom
        decoder_available = False
        
        # Check if any JPEG lossless decoder is available
        if hasattr(pydicom, 'pixel_data_handlers'):
            handlers = pydicom.config.pixel_data_handlers
            for handler in handlers:
                try:
                    if handler.is_available() and hasattr(handler, 'SUPPORTED_TRANSFER_SYNTAXES'):
                        syntaxes = handler.SUPPORTED_TRANSFER_SYNTAXES
                        if '1.2.840.10008.1.2.4.57' in syntaxes or '1.2.840.10008.1.2.4.70' in syntaxes:
                            decoder_available = True
                            print(f"Found JPEG Lossless decoder: {handler.__name__}")
                except:
                    pass
            
            if not decoder_available:
                print("Warning: No JPEG Lossless decoder is available.")
                print("Some DICOM files may not load correctly.")
        else:
            print("Warning: Could not verify DICOM compression support.")
    except:
        print("Could not verify DICOM compression support.")
    
    print("Installation complete. Run 'python app.py' to start SegMed-Pro.")

if __name__ == "__main__":
    install_dependencies()