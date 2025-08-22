import pydicom
import numpy as np
import os
from .metadata_extractor import extract_dicom_metadata
from .logger import logger
from .deidentification import apply_deidentification_to_slice, apply_deidentification_to_volume

def load_dicom_file(file_path, apply_deidentification=False):
    """
    Load a single DICOM file and convert it to a 3D numpy array
    Returns the array and a dictionary of metadata
    
    Args:
        file_path: Path to the DICOM file
        apply_deidentification: Whether to apply face removal using pydeface
    """
    try:
        ds = pydicom.dcmread(file_path)
        
        # Extract pixel data
        pixel_array = ds.pixel_array
        
        # Make sure we're returning a 3D array with shape (1, height, width)
        if len(pixel_array.shape) == 2:
            pixel_array = pixel_array[np.newaxis, :, :]
        
        # Normalize the data if needed based on modality
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Apply de-identification if requested
        if apply_deidentification:
            logger.info("Applying de-identification to DICOM data...")
            try:
                if len(pixel_array.shape) == 3 and pixel_array.shape[0] == 1:
                    # Single slice
                    deidentified_slice = apply_deidentification_to_slice(pixel_array[0])
                    pixel_array = deidentified_slice[np.newaxis, :, :]
                else:
                    # Volume data
                    pixel_array = apply_deidentification_to_volume(pixel_array)
                logger.info("De-identification completed successfully")
            except Exception as e:
                logger.error(f"De-identification failed: {e}")
                logger.warning("Continuing with original data...")
            
        # Extract and format metadata for display
        metadata = extract_dicom_metadata(ds)
        
        logger.info(f"Loaded single DICOM file: {file_path} with shape {pixel_array.shape}")
        return pixel_array, metadata
        
    except Exception as e:
        logger.error(f"Error loading DICOM file: {str(e)}")
        raise

def list_dicoms_in_directory(directory):
    """List all DICOM files in a directory"""
    dicom_files = []
    
    # If we're given a single file instead of a directory, check if it's a DICOM
    if os.path.isfile(directory):
        try:
            # Try to read as DICOM to validate
            pydicom.dcmread(directory, stop_before_pixels=True)
            return [directory]
        except:
            # Not a valid DICOM
            return []
    
    # Otherwise, scan the directory
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.dcm'):
                    file_path = os.path.join(root, file)
                    dicom_files.append(file_path)
                else:
                    # Try to read files without .dcm extension
                    try:
                        file_path = os.path.join(root, file)
                        pydicom.dcmread(file_path, stop_before_pixels=True)
                        dicom_files.append(file_path)
                    except:
                        # Not a DICOM file, skip
                        pass
    except Exception as e:
        logger.error(f"Error listing DICOM files: {str(e)}")

    return dicom_files