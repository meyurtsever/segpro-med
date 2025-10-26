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
        
        # Mammography-specific metadata
        if hasattr(ds, 'ViewPosition'):
            metadata['ViewPosition'] = ds.ViewPosition
        if hasattr(ds, 'ImageLaterality'):
            metadata['ImageLaterality'] = ds.ImageLaterality
        if hasattr(ds, 'ViewCodeSequence'):
            try:
                # ViewCodeSequence is a sequence, extract relevant info
                view_code = ds.ViewCodeSequence[0] if ds.ViewCodeSequence else None
                if view_code:
                    metadata['ViewCodeSequence'] = {
                        'CodeValue': getattr(view_code, 'CodeValue', ''),
                        'CodeMeaning': getattr(view_code, 'CodeMeaning', '')
                    }
            except:
                pass
        
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
      # Disable pydicom debugging for normal operations to reduce log noise
    # Only enable debug mode when specifically needed for troubleshooting
    pydicom.config.debug(False)
    
    logger.info(f"Configured DICOM handlers: {', '.join(handlers_added)}")
    return handlers_added

def load_single_dicom(file_path, apply_deidentification=False):
    """
    Load a single DICOM file and return the pixel array and metadata
    
    Args:
        file_path: Path to the DICOM file
        apply_deidentification: Whether to apply face removal using pydeface
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
        
        # Apply de-identification if requested
        if apply_deidentification:
            logger.info("Applying de-identification to DICOM data...")
            try:
                from .deidentification import apply_deidentification_to_slice, apply_deidentification_to_volume
                
                if len(pixel_array.shape) == 2:
                    # Single slice
                    pixel_array = apply_deidentification_to_slice(pixel_array)
                else:
                    # Volume data
                    pixel_array = apply_deidentification_to_volume(pixel_array)
                logger.info("De-identification completed successfully")
            except Exception as e:
                logger.error(f"De-identification failed: {e}")
                logger.warning("Continuing with original data...")
        
        # Get metadata
        metadata = get_dicom_metadata(file_path)
        
        return pixel_array, metadata
    
    except Exception as e:
        logger.error(f"Error loading DICOM file: {str(e)}")
        return None, {"error": str(e)}

def load_dicom_series(directory, apply_deidentification=False, lazy_load_threshold=100):
    """
    Load a series of DICOM files from a directory
    
    Args:
        directory (str): Path to directory containing DICOM files
        apply_deidentification: Whether to apply face removal using pydeface
        lazy_load_threshold: If number of files > threshold, use lazy loading (default: 100)
        
    Returns:
        tuple: (3D volume as numpy array or LazyVolumeWrapper, metadata dict, list of file paths)
    """
    logger.info(f"Loading DICOM series from directory: {directory}")
    
    # Find all DICOM files in the directory
    dicom_files = list_dicoms_in_directory(directory)
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in directory: {directory}")
    
    logger.info(f"Found {len(dicom_files)} DICOM files")
    
    # Check if we should use lazy loading
    use_lazy_loading = len(dicom_files) > lazy_load_threshold
    
    if use_lazy_loading:
        logger.info(f"Using lazy loading for {len(dicom_files)} files (threshold: {lazy_load_threshold})")
        return _load_dicom_series_lazy(dicom_files, apply_deidentification)
    else:
        logger.info(f"Using standard loading for {len(dicom_files)} files")
        return _load_dicom_series_standard(dicom_files, apply_deidentification)


def _load_dicom_series_lazy(dicom_files, apply_deidentification=False):
    """
    Load DICOM series using lazy loading
    
    Args:
        dicom_files: List of DICOM file paths
        apply_deidentification: Whether to apply face removal
        
    Returns:
        tuple: (LazyVolumeWrapper, metadata dict, list of sorted file paths)
    """
    from utils.lazy_dicom_loader import LazyDICOMLoader, LazyVolumeWrapper
    
    # Sort files first (by reading minimal metadata)
    sorted_files = _sort_dicom_files(dicom_files)
    
    # Create lazy loader
    loader = LazyDICOMLoader(
        sorted_files, 
        apply_deidentification=apply_deidentification,
        cache_size=100,
        preload_window=20
    )
    
    # Wrap in numpy-like interface
    volume_wrapper = LazyVolumeWrapper(loader)
    
    # Get metadata
    metadata = loader.get_metadata()
    
    logger.info(f"Lazy loader initialized: shape={volume_wrapper.shape}")
    
    return volume_wrapper, metadata, sorted_files


def _sort_dicom_files(dicom_files):
    """
    Sort DICOM files by reading only metadata (no pixel data)
    
    Args:
        dicom_files: List of DICOM file paths
        
    Returns:
        List of sorted file paths
    """
    logger.info(f"Sorting {len(dicom_files)} DICOM files by metadata...")
    
    # Read minimal metadata from each file
    file_info = []
    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            
            # Extract sorting key
            if hasattr(ds, 'InstanceNumber'):
                sort_key = (0, int(ds.InstanceNumber))  # Priority 0: InstanceNumber
            elif hasattr(ds, 'ImagePositionPatient'):
                sort_key = (1, float(ds.ImagePositionPatient[2]))  # Priority 1: Z position
            else:
                sort_key = (2, file_path)  # Priority 2: filename
            
            file_info.append((sort_key, file_path))
            
        except Exception as e:
            logger.warning(f"Error reading metadata from {file_path}: {e}")
            # Add to end with lowest priority
            file_info.append(((3, file_path), file_path))
    
    # Sort by key
    file_info.sort(key=lambda x: x[0])
    sorted_files = [fp for _, fp in file_info]
    
    logger.info(f"Files sorted successfully")
    return sorted_files

def _load_dicom_series_standard(dicom_files, apply_deidentification=False):
    """
    Load DICOM series using standard (non-lazy) loading
    This is the original implementation for smaller datasets
    
    Args:
        dicom_files: List of DICOM file paths
        apply_deidentification: Whether to apply face removal
        
    Returns:
        tuple: (3D numpy array, metadata dict, list of sorted file paths)
    """
    
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
    try:
        volume = np.stack(pixel_arrays)
        logger.info(f"Created 3D volume with shape {volume.shape}")
    except ValueError as e:
        logger.error(f"Error stacking pixel arrays: {str(e)}")
        # Check if all arrays have the same shape
        shapes = [arr.shape for arr in pixel_arrays]
        unique_shapes = list(set(shapes))
        
        if len(unique_shapes) > 1:
            logger.error(f"Found {len(unique_shapes)} different slice shapes: {unique_shapes}")
            logger.info("Attempting to resize slices to match the most common shape...")
            
            # Find the most common shape
            from collections import Counter
            shape_counts = Counter(shapes)
            target_shape = shape_counts.most_common(1)[0][0]
            logger.info(f"Resizing all slices to target shape: {target_shape}")
            
            # Resize all slices to the target shape
            resized_arrays = []
            for i, arr in enumerate(pixel_arrays):
                if arr.shape != target_shape:
                    logger.warning(f"Resizing slice {i} from {arr.shape} to {target_shape}")
                    # Use skimage resize if available, otherwise use basic numpy interpolation
                    try:
                        from skimage.transform import resize
                        resized_arr = resize(arr, target_shape, preserve_range=True, anti_aliasing=False)
                        resized_arrays.append(resized_arr.astype(arr.dtype))
                    except ImportError:
                        # Fallback: use basic numpy interpolation
                        logger.warning("skimage not available, using basic resize")
                        # For now, pad or crop to match target shape
                        if arr.shape[0] < target_shape[0] or arr.shape[1] < target_shape[1]:
                            # Pad with zeros
                            padded = np.zeros(target_shape, dtype=arr.dtype)
                            padded[:min(arr.shape[0], target_shape[0]), :min(arr.shape[1], target_shape[1])] = arr[:target_shape[0], :target_shape[1]]
                            resized_arrays.append(padded)
                        else:
                            # Crop to target shape
                            cropped = arr[:target_shape[0], :target_shape[1]]
                            resized_arrays.append(cropped)
                else:
                    resized_arrays.append(arr)
            
            # Try stacking again with resized arrays
            volume = np.stack(resized_arrays)
            logger.info(f"Successfully created 3D volume with shape {volume.shape} after resizing")
        else:
            # Re-raise the original error if shapes are the same
            raise e
    
    # Get metadata from the first slice
    metadata = get_dicom_metadata(ordered_files[0])
    logger.info("Extracted metadata from first slice")
    
    # Apply de-identification if requested
    if apply_deidentification:
        logger.info("Applying de-identification to DICOM series...")
        try:
            from .deidentification import apply_deidentification_to_volume
            volume = apply_deidentification_to_volume(volume)
            logger.info("De-identification completed successfully for series")
        except Exception as e:
            logger.error(f"De-identification failed for series: {e}")
            logger.warning("Continuing with original data...")
    
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