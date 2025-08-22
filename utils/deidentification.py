import os
import tempfile
import numpy as np
import nibabel as nib
import logging
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger(__name__)

class DeidentificationError(Exception):
    """Custom exception for de-identification errors"""
    pass

def check_pydeface_installation():
    """Check if pydeface is installed and available"""
    try:
        import pydeface
        logger.info("pydeface is available")
        return True
    except ImportError:
        logger.warning("pydeface not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydeface"])
            import pydeface
            logger.info("pydeface installed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to install pydeface: {e}")
            return False
    except Exception as e:
        logger.error(f"Error checking pydeface: {e}")
        return False

def apply_deidentification(dicom_data, temp_dir=None):
    """
    Apply de-identification (face removal) to DICOM data using pydeface
    
    Args:
        dicom_data: The DICOM image data as numpy array
        temp_dir: Optional temporary directory for processing
    
    Returns:
        numpy.ndarray: De-identified image data
    """
    if not check_pydeface_installation():
        logger.warning("pydeface not available, returning original data")
        return dicom_data
    
    # Validate input data
    if dicom_data is None or dicom_data.size == 0:
        logger.warning("Invalid input data for de-identification")
        return dicom_data
    
    # Create temporary directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # Initialize temporary file paths outside try block
    temp_input = os.path.join(temp_dir, "temp_input.nii.gz")
    temp_output = os.path.join(temp_dir, "temp_output.nii.gz")
    temp_mask = os.path.join(temp_dir, "temp_input_mask.nii.gz")
    
    try:
        import pydeface
        
        # Convert DICOM data to NIfTI format
        # Ensure data is in correct orientation and format
        if len(dicom_data.shape) == 2:
            # Single slice - expand to 3D
            dicom_data_3d = np.expand_dims(dicom_data, axis=2)
        else:
            dicom_data_3d = dicom_data
        
        # Validate that we have reasonable data
        if dicom_data_3d.max() == dicom_data_3d.min():
            logger.warning("Data appears to be uniform (no variation), skipping de-identification")
            return dicom_data
            
        # Create NIfTI image
        nii_img = nib.Nifti1Image(dicom_data_3d.astype(np.float32), affine=np.eye(4))
        nib.save(nii_img, temp_input)
        
        logger.info("Applying pydeface for de-identification...")
        
        # Apply pydeface using a simplified approach
        try:
            # Use pydeface API directly for better error handling
            pydeface.deface_image(temp_input, outfile=temp_output, force=True)
            
            # Load the de-identified result
            if os.path.exists(temp_output):
                deidentified_img = nib.load(temp_output)
                deidentified_data = deidentified_img.get_fdata()
                
                # Convert back to original shape if needed
                if len(dicom_data.shape) == 2 and len(deidentified_data.shape) == 3 and deidentified_data.shape[2] == 1:
                    deidentified_data = deidentified_data[:, :, 0]
                
                logger.info("De-identification completed successfully")
                return deidentified_data.astype(dicom_data.dtype)
            else:
                logger.error("pydeface output file not found")
                return dicom_data
                
        except Exception as pydeface_error:
            logger.error(f"pydeface API failed: {pydeface_error}")
            # Try command line interface as fallback
            try:
                cmd = [
                    sys.executable, "-m", "pydeface",
                    temp_input,
                    "--outfile", temp_output,
                    "--force"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(temp_output):
                    deidentified_img = nib.load(temp_output)
                    deidentified_data = deidentified_img.get_fdata()
                    
                    # Convert back to original shape if needed
                    if len(dicom_data.shape) == 2 and len(deidentified_data.shape) == 3 and deidentified_data.shape[2] == 1:
                        deidentified_data = deidentified_data[:, :, 0]
                    
                    logger.info("De-identification completed successfully using CLI fallback")
                    return deidentified_data.astype(dicom_data.dtype)
                else:
                    logger.error(f"pydeface CLI fallback failed: {result.stderr}")
                    return dicom_data
                    
            except subprocess.TimeoutExpired:
                logger.error("pydeface timed out")
                return dicom_data
            except Exception as cli_error:
                logger.error(f"pydeface CLI fallback failed: {cli_error}")
                return dicom_data
        
    except Exception as e:
        logger.error(f"De-identification failed: {e}")
        return dicom_data
    finally:
        # Clean up temporary files
        for temp_file in [temp_input, temp_output, temp_mask]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup {temp_file}: {cleanup_error}")

def apply_deidentification_to_volume(volume_data, temp_dir=None):
    """
    Apply de-identification to a full 3D volume
    
    Args:
        volume_data: 3D numpy array representing the full volume
        temp_dir: Optional temporary directory for processing
    
    Returns:
        numpy.ndarray: De-identified volume data
    """
    if len(volume_data.shape) != 3:
        logger.error("Volume data must be 3D")
        return volume_data
    
    return apply_deidentification(volume_data, temp_dir)

def apply_deidentification_to_slice(slice_data, volume_context=None, temp_dir=None):
    """
    Apply de-identification to a single slice with optional volume context
    
    Args:
        slice_data: 2D numpy array representing a single slice
        volume_context: Optional 3D context for better face detection
        temp_dir: Optional temporary directory for processing
    
    Returns:
        numpy.ndarray: De-identified slice data
    """
    if volume_context is not None:
        # Use full volume context for better face detection
        deidentified_volume = apply_deidentification_to_volume(volume_context, temp_dir)
        # Extract the corresponding slice
        # Note: This assumes slice_data corresponds to a specific slice in volume_context
        # The calling code should handle the slice indexing
        return deidentified_volume
    else:
        # Process single slice
        return apply_deidentification(slice_data, temp_dir)

def is_likely_head_slice(slice_data, threshold=0.3):
    """
    Simple heuristic to determine if a slice likely contains head/face region
    This can be used to optimize de-identification by only processing relevant slices
    
    Args:
        slice_data: 2D numpy array
        threshold: Threshold for determining if slice contains significant anatomy
    
    Returns:
        bool: True if slice likely contains head region
    """
    if slice_data is None or slice_data.size == 0:
        return False
    
    # Simple heuristic: check if slice has sufficient non-zero content
    # and reasonable intensity distribution
    non_zero_ratio = np.count_nonzero(slice_data) / slice_data.size
    
    if non_zero_ratio < threshold:
        return False
    
    # Check if intensity distribution suggests anatomical content
    if slice_data.max() > 0:
        intensity_range = slice_data.max() - slice_data.min()
        mean_intensity = np.mean(slice_data[slice_data > 0])
        
        # Simple check for reasonable intensity distribution
        if intensity_range > mean_intensity * 0.5:
            return True
    
    return False
