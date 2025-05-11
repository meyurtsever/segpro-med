import os
import sys
import traceback
import logging
import pydicom
import numpy as np
from logging.handlers import RotatingFileHandler

# Configure logger
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
log_file = os.path.join(log_dir, 'segmed_pro_debug.log')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger = logging.getLogger('segmed_pro')
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# Create formatter and add to handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_exception(func):
    """
    Decorator to log exceptions from functions
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = ''.join(tb_lines)
            
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.debug(f"Traceback:\n{tb_text}")
            
            # Re-raise the exception with the same traceback
            raise
    
    return wrapper

def check_and_report_dicom_handlers():
    """Check all available DICOM decoders and report their status"""
    results = {}
    
    # Try GDCM handler
    try:
        from pydicom.pixel_data_handlers import gdcm_handler
        results["gdcm"] = gdcm_handler.is_available()
    except ImportError:
        results["gdcm"] = False
        logger.warning("GDCM handler not available (ImportError)")
    except Exception as e:
        results["gdcm"] = False
        logger.error(f"Error checking GDCM handler: {str(e)}")
    
    # Try pylibjpeg handler
    try:
        from pydicom.pixel_data_handlers import pylibjpeg_handler
        results["pylibjpeg"] = pylibjpeg_handler.is_available()
    except ImportError:
        results["pylibjpeg"] = False
        logger.warning("pylibjpeg handler not available (ImportError)")
    except Exception as e:
        results["pylibjpeg"] = False
        logger.error(f"Error checking pylibjpeg handler: {str(e)}")
    
    # Try Pillow handler
    try:
        from pydicom.pixel_data_handlers import pillow_handler
        results["pillow"] = pillow_handler.is_available()
    except ImportError:
        results["pillow"] = False
        logger.warning("Pillow handler not available (ImportError)")
    except Exception as e:
        results["pillow"] = False
        logger.error(f"Error checking Pillow handler: {str(e)}")
    
    # Try NumPy handler
    try:
        from pydicom.pixel_data_handlers import numpy_handler
        results["numpy"] = True  # NumPy handler is always available if pydicom is installed
    except ImportError:
        results["numpy"] = False
        logger.warning("NumPy handler not available (ImportError)")
    except Exception as e:
        results["numpy"] = False
        logger.error(f"Error checking NumPy handler: {str(e)}")
    
    # Try JPEG-LS handler if available
    try:
        from pydicom.pixel_data_handlers import jpeg_ls_handler
        results["jpeg_ls"] = jpeg_ls_handler.is_available()
    except ImportError:
        results["jpeg_ls"] = False
        logger.warning("JPEG-LS handler not available (ImportError)")
    except Exception as e:
        results["jpeg_ls"] = False
        logger.error(f"Error checking JPEG-LS handler: {str(e)}")
    
    return results

def configure_dicom_handlers_for_debug():
    """Configure DICOM handlers for debug operations"""
    # Reset handlers
    pydicom.config.pixel_data_handlers = []
    handlers_added = []
    
    # Try to add each handler in preferred order
    try:
        from pydicom.pixel_data_handlers import gdcm_handler
        if gdcm_handler.is_available():
            pydicom.config.pixel_data_handlers.append(gdcm_handler)
            handlers_added.append("gdcm")
            logger.info("Added GDCM handler for debugging")
    except Exception as e:
        logger.error(f"Error setting up GDCM handler: {str(e)}")
    
    try:
        from pydicom.pixel_data_handlers import pylibjpeg_handler
        if pylibjpeg_handler.is_available():
            pydicom.config.pixel_data_handlers.append(pylibjpeg_handler)
            handlers_added.append("pylibjpeg")
            logger.info("Added pylibjpeg handler for debugging")
    except Exception as e:
        logger.error(f"Error setting up pylibjpeg handler: {str(e)}")
    
    try:
        from pydicom.pixel_data_handlers import pillow_handler
        if pillow_handler.is_available():
            pydicom.config.pixel_data_handlers.append(pillow_handler)
            handlers_added.append("pillow")
            logger.info("Added Pillow handler for debugging")
    except Exception as e:
        logger.error(f"Error setting up Pillow handler: {str(e)}")
    
    try:
        from pydicom.pixel_data_handlers import numpy_handler
        pydicom.config.pixel_data_handlers.append(numpy_handler)
        handlers_added.append("numpy")
        logger.info("Added NumPy handler for debugging")
    except Exception as e:
        logger.error(f"Error setting up NumPy handler: {str(e)}")
    
    # Enable pydicom debugging
    pydicom.config.debug(True)
    
    return handlers_added

def inspect_dicom(file_path):
    """Inspect a DICOM file's metadata and transfer syntax"""
    try:
        # Read DICOM file metadata
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        
        # Get transfer syntax
        transfer_syntax = ds.file_meta.TransferSyntaxUID
        transfer_syntax_name = pydicom.uid.UID_dictionary.get(transfer_syntax, ('Unknown', 'Unknown'))[0]
        
        # Check if compressed
        is_compressed = False
        if transfer_syntax != pydicom.uid.ExplicitVRLittleEndian and transfer_syntax != pydicom.uid.ImplicitVRLittleEndian:
            is_compressed = True
        
        # Return info
        info = {
            'TransferSyntax': str(transfer_syntax),
            'TransferSyntaxName': transfer_syntax_name,
            'IsCompressed': is_compressed,
            'Modality': getattr(ds, 'Modality', 'Unknown'),
            'BitsAllocated': getattr(ds, 'BitsAllocated', 'Unknown'),
            'BitsStored': getattr(ds, 'BitsStored', 'Unknown'),
            'PixelRepresentation': getattr(ds, 'PixelRepresentation', 'Unknown'),
            'SamplesPerPixel': getattr(ds, 'SamplesPerPixel', 'Unknown'),
            'PhotometricInterpretation': getattr(ds, 'PhotometricInterpretation', 'Unknown'),
            'Rows': getattr(ds, 'Rows', 'Unknown'),
            'Columns': getattr(ds, 'Columns', 'Unknown')
        }
        
        # Check available handlers
        info['AvailableHandlers'] = check_and_report_dicom_handlers()
        
        return info
    
    except Exception as e:
        logger.error(f"Error inspecting DICOM file: {str(e)}")
        return {'error': str(e)}

@log_exception
def debug_dicom_loading(file_path):
    """
    Debug function for loading DICOM files
    """
    logger.info(f"Debug loading of DICOM file: {file_path}")
    
    try:
        # Step 1: Read metadata
        logger.info("Step 1: Reading DICOM file metadata")
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        
        # Step 2: Check transfer syntax
        logger.info("Step 2: Checking transfer syntax")
        transfer_syntax = ds.file_meta.TransferSyntaxUID
        logger.info(f"Transfer Syntax: {transfer_syntax}")
        
        # Check if compressed
        is_compressed = False
        if transfer_syntax != pydicom.uid.ExplicitVRLittleEndian and transfer_syntax != pydicom.uid.ImplicitVRLittleEndian:
            is_compressed = True
            logger.info("Compressed DICOM detected, checking available handlers")
            
            # Configure handlers for debug
            handlers = configure_dicom_handlers_for_debug()
            logger.info(f"Debug handlers configured: {handlers}")
        
        # Step 3: Load pixel data
        logger.info("Step 3: Loading pixel data")
        ds = pydicom.dcmread(file_path)
        
        if not hasattr(ds, 'PixelData'):
            logger.error("No pixel data found in DICOM file")
            return False
        
        logger.info("Pixel data is present")
        
        # Step 4: Convert to pixel array
        logger.info("Step 4: Converting to pixel array")
        pixel_array = ds.pixel_array
        
        shape = pixel_array.shape
        dtype = pixel_array.dtype
        
        logger.info(f"Pixel array loaded successfully: shape={shape}, dtype={dtype}")
        
        # Calculate basic statistics
        min_val = np.min(pixel_array)
        max_val = np.max(pixel_array)
        mean_val = np.mean(pixel_array)
        
        logger.info(f"Pixel array statistics: min={min_val}, max={max_val}, mean={mean_val}")
        
        # Step 5: Check rescale parameters
        logger.info("Step 5: Checking rescale parameters")
        rescale_slope = getattr(ds, 'RescaleSlope', 1)
        rescale_intercept = getattr(ds, 'RescaleIntercept', 0)
        
        logger.info(f"Rescale parameters: slope={rescale_slope}, intercept={rescale_intercept}")
        
        # Step 6: Apply rescale if needed
        if rescale_slope != 1 or rescale_intercept != 0:
            logger.info("Step 6: Applying rescale")
            rescaled_array = pixel_array * float(rescale_slope) + float(rescale_intercept)
            
            # Calculate rescaled statistics
            min_val = np.min(rescaled_array)
            max_val = np.max(rescaled_array)
            mean_val = np.mean(rescaled_array)
            
            logger.info(f"Rescaled array statistics: min={min_val}, max={max_val}, mean={mean_val}")
        else:
            logger.info("Step 6: No rescaling needed")
        
        # Step 7: Test normalization for display
        logger.info("Step 7: Testing normalization for display")
        
        # Test min-max normalization
        logger.info("Testing min-max normalization")
        norm_min = 0
        norm_max = 255
        
        try:
            # Avoid division by zero
            if max_val > min_val:
                normalized = ((pixel_array - min_val) / (max_val - min_val)) * norm_max
                logger.info(f"Min-max normalization successful: min={norm_min}, max={norm_max}")
            else:
                logger.warning("Cannot perform min-max normalization: max value equals min value")
        except Exception as e:
            logger.error(f"Error in min-max normalization: {str(e)}")
        
        # Test window/level adjustment if present
        logger.info("Testing window/level adjustment")
        if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
            try:
                window_center = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else float(ds.WindowCenter[0])
                window_width = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else float(ds.WindowWidth[0])
                
                logger.info(f"DICOM window parameters: center={window_center}, width={window_width}")
                
                # Calculate window min and max
                window_min = window_center - window_width/2
                window_max = window_center + window_width/2
                
                logger.info(f"Window min-max: {window_min}-{window_max}")
                
                # Apply windowing
                below = pixel_array <= window_min
                above = pixel_array >= window_max
                between = np.logical_and(~below, ~above)
                
                windowed = np.zeros_like(pixel_array)
                windowed[below] = norm_min
                windowed[above] = norm_max
                if window_max != window_min:
                    windowed[between] = ((pixel_array[between] - window_min) / (window_max - window_min)) * norm_max
                
                # Calculate windowed statistics
                min_val = np.min(windowed)
                max_val = np.max(windowed)
                
                logger.info(f"Window/level adjustment successful: min={min_val}, max={max_val}")
            except Exception as e:
                logger.error(f"Error in window/level adjustment: {str(e)}")
        
        logger.info("DICOM debugging complete")
        return True
    
    except Exception as e:
        logger.error(f"Error in DICOM debugging: {str(e)}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text = ''.join(tb_lines)
        logger.debug(f"Traceback:\n{tb_text}")
        return False