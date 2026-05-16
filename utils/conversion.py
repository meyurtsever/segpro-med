import os
import subprocess
import tempfile
import shutil
import numpy as np
from scipy.io import savemat
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt


def _dicom_to_nifti_dcm2niix(dicom_dir, output_path, compress=True):
    """Convert DICOM to NIfTI using the dcm2niix CLI tool."""
    output_dir = os.path.dirname(output_path) or "."
    temp_dir = tempfile.mkdtemp()

    try:
        cmd = ["dcm2niix", "-z", "y" if compress else "n", "-f", "%f", "-o", temp_dir, dicom_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"dcm2niix failed: {result.stderr}")

        nifti_files = [f for f in os.listdir(temp_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

        if not nifti_files:
            raise RuntimeError("No NIfTI files were created by dcm2niix")

        source_file = os.path.join(temp_dir, nifti_files[0])
        os.makedirs(output_dir, exist_ok=True)
        shutil.move(source_file, output_path)
        return output_path
    finally:
        shutil.rmtree(temp_dir)


def _dicom_to_nifti_python(dicom_dir, output_path):
    """Pure Python DICOM to NIfTI fallback using pydicom + nibabel."""
    output_dir = os.path.dirname(output_path) or "."

    dicom_files = []
    for fname in sorted(os.listdir(dicom_dir)):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath)
            if hasattr(ds, 'pixel_array'):
                dicom_files.append(ds)
        except Exception:
            continue

    if not dicom_files:
        raise RuntimeError(f"No valid DICOM files with pixel data in: {dicom_dir}")

    try:
        dicom_files.sort(key=lambda d: int(d.InstanceNumber))
    except (AttributeError, ValueError):
        pass

    volume = np.stack([ds.pixel_array.astype(np.float32) for ds in dicom_files], axis=0)

    ds0 = dicom_files[0]
    affine = np.eye(4)
    try:
        ipp = [float(x) for x in ds0.ImagePositionPatient]
        ps = [float(x) for x in ds0.PixelSpacing]
        st = float(getattr(ds0, 'SliceThickness', 1.0))
        iop = [float(x) for x in ds0.ImageOrientationPatient]
        row_cos, col_cos = np.array(iop[:3]), np.array(iop[3:])
        slc_cos = np.cross(row_cos, col_cos)
        affine[:3, 0] = row_cos * ps[0]
        affine[:3, 1] = col_cos * ps[1]
        affine[:3, 2] = slc_cos * st
        affine[:3, 3] = ipp
    except (AttributeError, ValueError, TypeError):
        pass  # keep identity if headers are incomplete

    nifti_img = nib.Nifti1Image(volume, affine)
    os.makedirs(output_dir, exist_ok=True)
    nib.save(nifti_img, output_path)
    return output_path


def dicom_to_nifti(dicom_dir, output_path=None, compress=True):
    """
    Convert DICOM series to NIfTI format.
    Tries dcm2niix first; falls back to pure Python (pydicom + nibabel).

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

    # Ensure extension matches the requested compression mode.
    if output_path.endswith(".nii.gz") and not compress:
        output_path = output_path[:-3]
    elif output_path.endswith(".nii") and compress:
        output_path += ".gz"
    elif not (output_path.endswith(".nii") or output_path.endswith(".nii.gz")):
        output_path += ".nii.gz" if compress else ".nii"

    try:
        return _dicom_to_nifti_dcm2niix(dicom_dir, output_path, compress)
    except FileNotFoundError:
        print("[conversion] dcm2niix not found — using pydicom + nibabel fallback")
        return _dicom_to_nifti_python(dicom_dir, output_path)


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

def perform_conversion(input_path, output_path, conversion_type, axis=2, compress=True):
    """
    Perform the specified conversion.
    
    Args:
        input_path (str): Path to input file or directory
        output_path (str): Path to output file or directory
        conversion_type (str): Type of conversion to perform
        axis (int): Axis for NIfTI to PNG slicing
        compress (bool): Whether DICOM to NIfTI should produce .nii.gz
        
    Returns:
        str: Path to output file or directory
    """
    if conversion_type == "DICOM to NIFTI":
        return dicom_to_nifti(input_path, output_path, compress=compress)
    elif conversion_type == "NIFTI to MAT":
        return nifti_to_mat(input_path, output_path)
    elif conversion_type == "DICOM to MAT":
        return dicom_to_mat(input_path, output_path)
    elif conversion_type == "NIFTI to PNG":
        return nifti_to_png(input_path, output_path, axis=axis)
    else:
        raise ValueError(f"Unsupported conversion type: {conversion_type}")
