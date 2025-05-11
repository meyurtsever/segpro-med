import os
import subprocess
import tempfile
import shutil
import numpy as np
from scipy.io import savemat
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt

def dicom_to_nifti(dicom_dir, output_path=None):
    """
    Convert DICOM series to NIfTI format using dcm2niix.
    
    Args:
        dicom_dir (str): Directory containing DICOM files
        output_path (str, optional): Output NIfTI file path. If None, uses same name as directory.
        
    Returns:
        str: Path to output NIfTI file
    """
    if not os.path.isdir(dicom_dir):
        raise ValueError(f"Directory {dicom_dir} does not exist")
    
    if output_path is None:
        output_dir = os.path.dirname(dicom_dir)
        output_filename = os.path.basename(dicom_dir)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_dir = os.path.dirname(output_path)
        if not output_dir:
            output_dir = "."
    
    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Execute dcm2niix
        cmd = ["dcm2niix", "-z", "y", "-f", "%f", "-o", temp_dir, dicom_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"dcm2niix failed: {result.stderr}")
        
        # Find the generated NIfTI file
        nifti_files = [f for f in os.listdir(temp_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        
        if not nifti_files:
            raise RuntimeError("No NIfTI files were created")
        
        # Move the file to the desired location
        source_file = os.path.join(temp_dir, nifti_files[0])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Ensure output_path has the correct extension
        if not (output_path.endswith(".nii") or output_path.endswith(".nii.gz")):
            output_path += ".nii.gz"
        
        shutil.move(source_file, output_path)
        
        return output_path
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def dicom_to_mat(dicom_dir, output_path=None):
    """
    Convert DICOM series to MATLAB .mat format.
    
    Args:
        dicom_dir (str): Directory containing DICOM files
        output_path (str, optional): Output .mat file path
        
    Returns:
        str: Path to output .mat file
    """
    from utils.dicom_utils import load_dicom_series
    
    # Load DICOM series
    volume, metadata, _ = load_dicom_series(dicom_dir)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(dicom_dir), 
                                  f"{os.path.basename(dicom_dir)}.mat")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as .mat file
    save_dict = {
        'volume': volume,
        'metadata': metadata
    }
    
    savemat(output_path, save_dict)
    
    return output_path

def nifti_to_mat(nifti_path, output_path=None):
    """
    Convert NIfTI file to MATLAB .mat format.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        output_path (str, optional): Path to output .mat file
        
    Returns:
        str: Path to output .mat file
    """
    from utils.nifti_utils import nifti_to_mat as utils_nifti_to_mat
    return utils_nifti_to_mat(nifti_path, output_path)

def nifti_to_png(nifti_path, output_dir=None, axis=2):
    """
    Convert NIfTI file to a series of PNG images.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        output_dir (str, optional): Directory to save PNG files
        axis (int, optional): Axis along which to slice the volume (0, 1, or 2)
        
    Returns:
        str: Path to output directory
    """
    from utils.nifti_utils import nifti_to_png as utils_nifti_to_png
    return utils_nifti_to_png(nifti_path, output_dir, axis)

def perform_conversion(input_path, output_path, conversion_type):
    """
    Perform the specified conversion.
    
    Args:
        input_path (str): Path to input file or directory
        output_path (str): Path to output file or directory
        conversion_type (str): Type of conversion to perform
        
    Returns:
        str: Path to output file or directory
    """
    if conversion_type == "DICOM to NIFTI":
        return dicom_to_nifti(input_path, output_path)
    elif conversion_type == "NIFTI to MAT":
        return nifti_to_mat(input_path, output_path)
    elif conversion_type == "DICOM to MAT":
        return dicom_to_mat(input_path, output_path)
    elif conversion_type == "NIFTI to PNG":
        return nifti_to_png(input_path, output_path)
    else:
        raise ValueError(f"Unsupported conversion type: {conversion_type}")