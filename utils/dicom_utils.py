import os
import pydicom
import numpy as np
from utils.debug_utils import logger, log_exception

def list_dicoms_in_directory(directory):
    """
    List all DICOM files in a directory or return single file if input is a file
    """
    # Handle single file path
    if os.path.isfile(directory) and directory.lower().endswith('.dcm'):
        return [directory]
    
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    return dicom_files

def get_dicom_metadata(dicom_path):
    """
    Extract and format metadata from a DICOM file
    """
    try:
        # Read DICOM without pixel data for faster metadata access
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
        
        # Extract key metadata fields
        metadata = {}
        
        # Extract patient metadata
        if hasattr(ds, 'PatientName'):
            metadata['PatientName'] = str(ds.PatientName)
        if hasattr(ds, 'PatientID'):
            metadata['PatientID'] = ds.PatientID
        if hasattr(ds, 'PatientBirthDate'):
            metadata['PatientBirthDate'] = ds.PatientBirthDate
        if hasattr(ds, 'PatientSex'):
            metadata['PatientSex'] = ds.PatientSex
        
        # Extract study metadata
        if hasattr(ds, 'StudyDate'):
            metadata['StudyDate'] = ds.StudyDate
        if hasattr(ds, 'StudyTime'):
            metadata['StudyTime'] = ds.StudyTime
        if hasattr(ds, 'StudyDescription'):
            metadata['StudyDescription'] = ds.StudyDescription
        if hasattr(ds, 'StudyInstanceUID'):
            metadata['StudyInstanceUID'] = ds.StudyInstanceUID
        
        # Extract series metadata
        if hasattr(ds, 'SeriesDescription'):
            metadata['SeriesDescription'] = ds.SeriesDescription
        if hasattr(ds, 'SeriesNumber'):
            metadata['SeriesNumber'] = ds.SeriesNumber
        if hasattr(ds, 'Modality'):
            metadata['Modality'] = ds.Modality
        
        # Extract image metadata
        if hasattr(ds, 'PixelSpacing'):
            metadata['PixelSpacing'] = [float(val) for val in ds.PixelSpacing]
        if hasattr(ds, 'SliceThickness'):
            metadata['SliceThickness'] = float(ds.SliceThickness) if ds.SliceThickness else None
        if hasattr(ds, 'ImageOrientationPatient'):
            metadata['ImageOrientationPatient'] = [float(val) for val in ds.ImageOrientationPatient]
        if hasattr(ds, 'ImagePositionPatient'):
            metadata['ImagePositionPatient'] = [float(val) for val in ds.ImagePositionPatient]
        
        # Check if compressed
        is_compressed = False
        compression_type = "None"
        if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
            transfer_syntax = ds.file_meta.TransferSyntaxUID
            metadata['TransferSyntax'] = str(transfer_syntax)
            if transfer_syntax != pydicom.uid.ExplicitVRLittleEndian and transfer_syntax != pydicom.uid.ImplicitVRLittleEndian:
                is_compressed = True
                compression_type = str(transfer_syntax)
        
        metadata['IsCompressed'] = is_compressed
        metadata['CompressionType'] = compression_type
        
        # Window settings for display
        if hasattr(ds, 'WindowCenter'):
            if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
                metadata['WindowCenter'] = float(ds.WindowCenter[0])
            else:
                metadata['WindowCenter'] = float(ds.WindowCenter)
        
        if hasattr(ds, 'WindowWidth'):
            if isinstance(ds.WindowWidth, pydicom.multival.MultiValue):
                metadata['WindowWidth'] = float(ds.WindowWidth[0])
            else:
                metadata['WindowWidth'] = float(ds.WindowWidth)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting DICOM metadata: {str(e)}")
        return {"error": str(e)}

def configure_dicom_handlers():
    """Configure DICOM handlers for compressed files"""
    # Reset handlers
    pydicom.config.pixel_data_handlers = []
    
    # Try to add each handler in preferred order
    handlers_added = []
    
    try:
        # Try GDCM handler first (good for JPEG2000)
        from pydicom.pixel_data_handlers import gdcm_handler
        if gdcm_handler.is_available():
            pydicom.config.pixel_data_handlers.append(gdcm_handler)
            handlers_added.append("gdcm")
    except ImportError:
        logger.warning("GDCM handler not available")
    except Exception as e:
        logger.error(f"Error setting up GDCM handler: {str(e)}")
    
    try:
        # Try pylibjpeg handler next (good for JPEG, JPEG-LS)
        from pydicom.pixel_data_handlers import pylibjpeg_handler
        if pylibjpeg_handler.is_available():
            pydicom.config.pixel_data_handlers.append(pylibjpeg_handler)
            handlers_added.append("pylibjpeg")
    except ImportError:
        logger.warning("pylibjpeg handler not available")
    except Exception as e:
        logger.error(f"Error setting up pylibjpeg handler: {str(e)}")
    
    try:
        # Try pillow handler next (for common JPEG)
        from pydicom.pixel_data_handlers import pillow_handler
        if pillow_handler.is_available():
            pydicom.config.pixel_data_handlers.append(pillow_handler)
            handlers_added.append("pillow")
    except ImportError:
        logger.warning("Pillow handler not available")
    except Exception as e:
        logger.error(f"Error setting up Pillow handler: {str(e)}")
    
    try:
        # Always add NumPy handler last (for uncompressed images)
        from pydicom.pixel_data_handlers import numpy_handler
        pydicom.config.pixel_data_handlers.append(numpy_handler)
        handlers_added.append("numpy")
    except Exception as e:
        logger.error(f"Error setting up NumPy handler: {str(e)}")
    
    # Enable pydicom debugging
    pydicom.config.debug(True)
    
    logger.info(f"Configured DICOM handlers: {', '.join(handlers_added)}")
    return handlers_added

def load_single_dicom(file_path):
    """
    Load a single DICOM file and return the pixel array and metadata
    """
    try:
        # Ensure DICOM handlers are configured
        configure_dicom_handlers()
        
        # Read the DICOM file
        logger.info(f"Loading DICOM file: {file_path}")
        ds = pydicom.dcmread(file_path)
        
        # Check if pixel data exists
        if not hasattr(ds, 'PixelData'):
            logger.error("DICOM file has no pixel data")
            return None, {"error": "DICOM file has no pixel data"}
        
        # Try to get pixel array
        try:
            pixel_array = ds.pixel_array
            logger.info(f"Pixel array shape: {pixel_array.shape}, dtype: {pixel_array.dtype}")
        except Exception as e:
            logger.error(f"Error accessing pixel array: {str(e)}")
            return None, {"error": f"Error accessing pixel array: {str(e)}"}
        
        # Apply rescale slope/intercept if available
        if hasattr(ds, 'RescaleSlope') or hasattr(ds, 'RescaleIntercept'):
            rescale_slope = getattr(ds, 'RescaleSlope', 1)
            rescale_intercept = getattr(ds, 'RescaleIntercept', 0)
            logger.info(f"Applying rescale: slope={rescale_slope}, intercept={rescale_intercept}")
            try:
                if rescale_slope != 1 or rescale_intercept != 0:
                    pixel_array = pixel_array * float(rescale_slope) + float(rescale_intercept)
            except Exception as e:
                logger.error(f"Error applying rescale: {str(e)}")
                # Continue without rescaling
        
        # Get metadata
        metadata = get_dicom_metadata(file_path)
        
        return pixel_array, metadata
    
    except Exception as e:
        logger.error(f"Error loading DICOM file: {str(e)}")
        return None, {"error": str(e)}

def load_dicom_series(directory):
    """
    Load a series of DICOM files from a directory
    
    Args:
        directory (str): Path to directory containing DICOM files
        
    Returns:
        tuple: (3D volume as numpy array, metadata dict, list of file paths)
    """
    logger.info(f"Loading DICOM series from directory: {directory}")
    
    # Find all DICOM files in the directory
    dicom_files = list_dicoms_in_directory(directory)
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in directory: {directory}")
    
    logger.info(f"Found {len(dicom_files)} DICOM files")
    
    # Read all DICOM files
    slices = []
    for file_path in dicom_files:
        try:
            logger.debug(f"Reading DICOM file: {file_path}")
            ds = pydicom.dcmread(file_path)
            slices.append(ds)
        except Exception as e:
            logger.error(f"Error reading DICOM file {file_path}: {str(e)}")
            # Continue with other files
    
    if not slices:
        raise ValueError("Could not read any DICOM files")
    
    logger.info(f"Successfully read {len(slices)} DICOM files")
    
    # Sort slices by Instance Number if available, or by Image Position Patient
    try:
        if hasattr(slices[0], 'InstanceNumber'):
            logger.info("Sorting by Instance Number")
            slices = sorted(slices, key=lambda x: int(x.InstanceNumber))
        elif hasattr(slices[0], 'ImagePositionPatient'):
            logger.info("Sorting by Image Position Patient")
            # Sort by slice position (z-coordinate)
            slices = sorted(slices, key=lambda x: float(x.ImagePositionPatient[2]))
        else:
            logger.warning("No slice ordering information found, using filename order")
    except Exception as e:
        logger.error(f"Error sorting slices: {str(e)}")
        logger.warning("Using original file order")
    
    # Get the ordered file paths
    ordered_files = [s.filename for s in slices]
    
    # Get pixel data from each slice
    pixel_arrays = []
    for s in slices:
        try:
            # Apply rescale slope and intercept if available
            pixel_array = s.pixel_array
            if hasattr(s, 'RescaleSlope') or hasattr(s, 'RescaleIntercept'):
                rescale_slope = getattr(s, 'RescaleSlope', 1)
                rescale_intercept = getattr(s, 'RescaleIntercept', 0)
                pixel_array = pixel_array * float(rescale_slope) + float(rescale_intercept)
            
            pixel_arrays.append(pixel_array)
        except Exception as e:
            logger.error(f"Error processing pixel data: {str(e)}")
            # Use a blank image of the same size as the first slice
            if pixel_arrays:
                pixel_arrays.append(np.zeros_like(pixel_arrays[0]))
            else:
                raise ValueError(f"Could not process pixel data from first slice: {str(e)}")
    
    # Stack all the pixel arrays into a 3D volume
    volume = np.stack(pixel_arrays)
    logger.info(f"Created 3D volume with shape {volume.shape}")
    
    # Get metadata from the first slice
    metadata = get_dicom_metadata(ordered_files[0])
    logger.info("Extracted metadata from first slice")
    
    # Add series information to metadata
    metadata['SeriesInfo'] = {
        'NumberOfSlices': len(slices),
        'SliceThickness': getattr(slices[0], 'SliceThickness', 0),
        'SeriesDescription': getattr(slices[0], 'SeriesDescription', ''),
        'PatientID': getattr(slices[0], 'PatientID', ''),
        'PatientName': str(getattr(slices[0], 'PatientName', '')),
        'StudyDate': getattr(slices[0], 'StudyDate', '')
    }
    
    return volume, metadata, ordered_files