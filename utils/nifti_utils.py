import os
import numpy as np
import nibabel as nib
from scipy.io import savemat
import matplotlib.pyplot as plt
from pathlib import Path

def load_nifti_file(file_path):
    """
    Load a NIfTI file.
    
    Args:
        file_path (str): Path to NIfTI file
        
    Returns:
        tuple: (3D volume as numpy array, metadata dictionary)
    """
    # Load the NIfTI file
    try:
        nifti_img = nib.load(file_path)
    except Exception as e:
        raise ValueError(f"Error loading NIfTI file: {str(e)}")
    
    # Extract data as numpy array
    volume = nifti_img.get_fdata()
    
    # Handle 4D data (e.g., fMRI or DTI)
    if len(volume.shape) == 4:
        # For now, we'll just take the first 3D volume
        volume = volume[:, :, :, 0]
    
    # Extract metadata
    metadata = extract_nifti_metadata(nifti_img)
    
    # Add filename to metadata
    metadata["filename"] = os.path.basename(file_path)
    metadata["filepath"] = file_path
    
    return volume, metadata

def extract_nifti_metadata(nifti_img):
    """
    Extract metadata from a NIfTI image object.
    
    Args:
        nifti_img (nibabel.nifti1.Nifti1Image): NIfTI image object
        
    Returns:
        dict: Dictionary of metadata
    """
    header = nifti_img.header
    
    # Basic metadata
    metadata = {
        "shape": nifti_img.shape,
        "data_type": str(header.get_data_dtype()),
        "affine": nifti_img.affine.tolist(),
        "voxel_dimensions": header.get_zooms(),
        "voxel_size": {
            "x": float(header.get_zooms()[0]),
            "y": float(header.get_zooms()[1]),
            "z": float(header.get_zooms()[2]) if len(header.get_zooms()) > 2 else 1.0
        },
        "dimensions": {
            "x": int(nifti_img.shape[0]),
            "y": int(nifti_img.shape[1]),
            "z": int(nifti_img.shape[2]) if len(nifti_img.shape) > 2 else 1
        },
    }
    
    # Add header-specific fields
    metadata["header"] = {
        "sizeof_hdr": int(header["sizeof_hdr"]),
        "data_type": header.get_data_dtype().name,
        "pixdim": header["pixdim"].tolist(),
        "xyzt_units": int(header["xyzt_units"]),
        "cal_max": float(header["cal_max"]),
        "cal_min": float(header["cal_min"]),
        "slice_duration": float(header["slice_duration"]),
        "description": str(header.get("descrip", b"").decode("utf-8", errors="ignore"))
    }
    
    # Add orientation information
    qform = header.get_qform()
    sform = header.get_sform()
    
    metadata["orientation"] = {
        "qform_code": int(header["qform_code"]),
        "sform_code": int(header["sform_code"]),
        "qto_xyz": nib.affines.mat2str(qform) if qform is not None else "None",
        "sto_xyz": nib.affines.mat2str(sform) if sform is not None else "None"
    }
    
    # Extract quaternion parameters if available
    metadata["quaternion"] = {
        "qoffset_x": float(header["qoffset_x"]),
        "qoffset_y": float(header["qoffset_y"]),
        "qoffset_z": float(header["qoffset_z"]),
        "quatern_b": float(header["quatern_b"]),
        "quatern_c": float(header["quatern_c"]),
        "quatern_d": float(header["quatern_d"])
    }
    
    # Calculate physical dimensions
    voxel_size = metadata["voxel_size"]
    dimensions = metadata["dimensions"]
    metadata["physical_dimensions"] = {
        "x": voxel_size["x"] * dimensions["x"],
        "y": voxel_size["y"] * dimensions["y"],
        "z": voxel_size["z"] * dimensions["z"]
    }
    
    return metadata

def nifti_to_mat(nifti_path, output_path=None):
    """
    Convert NIfTI file to MATLAB .mat format.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        output_path (str, optional): Path to output .mat file. If None, 
                                     will use the same name as input with .mat extension.
                                     
    Returns:
        str: Path to output .mat file
    """
    if output_path is None:
        output_path = os.path.splitext(nifti_path)[0] + '.mat'
    
    # Load NIfTI data
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    metadata = extract_nifti_metadata(nifti_img)
    
    # Save to .mat format
    save_dict = {
        'volume': data,
        'affine': nifti_img.affine,
        'header': metadata,
        'source_file': nifti_path
    }
    
    savemat(output_path, save_dict)
    
    return output_path

def nifti_to_png(nifti_path, output_dir=None, axis=2, start_slice=None, end_slice=None, slice_step=1):
    """
    Convert NIfTI file to a series of PNG images.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        output_dir (str, optional): Directory to save PNG files. If None,
                                   will create a subdirectory with the name of the input file.
        axis (int, optional): Axis along which to slice the volume (0, 1, or 2)
        start_slice (int, optional): First slice to export (default: None = start from first)
        end_slice (int, optional): Last slice to export (default: None = end at last)
        slice_step (int, optional): Step between slices (default: 1 = export all)
        
    Returns:
        str: Path to output directory
    """
    if output_dir is None:
        base_name = os.path.basename(os.path.splitext(nifti_path)[0])
        output_dir = os.path.join(os.path.dirname(nifti_path), f"{base_name}_png")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load NIfTI data
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    
    # Handle 4D data
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    
    # Normalize the data for visualization
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        data_norm = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
    else:
        data_norm = np.zeros_like(data, dtype=np.uint8)
    
    # Determine slicing
    if axis == 0:
        total_slices = data.shape[0]
        if start_slice is None:
            start_slice = 0
        if end_slice is None:
            end_slice = total_slices
        slices = [data_norm[i, :, :] for i in range(start_slice, end_slice, slice_step)]
    elif axis == 1:
        total_slices = data.shape[1]
        if start_slice is None:
            start_slice = 0
        if end_slice is None:
            end_slice = total_slices
        slices = [data_norm[:, i, :] for i in range(start_slice, end_slice, slice_step)]
    else:  # axis == 2
        total_slices = data.shape[2]
        if start_slice is None:
            start_slice = 0
        if end_slice is None:
            end_slice = total_slices
        slices = [data_norm[:, :, i] for i in range(start_slice, end_slice, slice_step)]
    
    # Save slices as PNG
    output_files = []
    
    for i, slice_data in enumerate(slices):
        output_file = os.path.join(output_dir, f"slice_{start_slice + i * slice_step:04d}.png")
        plt.imsave(output_file, slice_data, cmap='gray')
        output_files.append(output_file)
    
    # Create an info file with metadata
    metadata = extract_nifti_metadata(nifti_img)
    info_file = os.path.join(output_dir, "info.txt")
    
    with open(info_file, "w") as f:
        f.write(f"Source file: {nifti_path}\n")
        f.write(f"Axis: {axis}\n")
        f.write(f"Slices: {start_slice} to {end_slice-1} (step {slice_step})\n")
        f.write(f"Total slices exported: {len(slices)}\n\n")
        f.write("Metadata:\n")
        for key, value in metadata.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    return output_dir

def create_nifti_from_array(data, affine=None, output_path=None):
    """
    Create a NIfTI file from a numpy array.
    
    Args:
        data (numpy.ndarray): 3D array of image data
        affine (numpy.ndarray, optional): 4x4 affine transformation matrix
        output_path (str, optional): Path to save NIfTI file
        
    Returns:
        tuple: (nibabel.nifti1.Nifti1Image, output_path if saved)
    """
    # Create default identity affine if not provided
    if affine is None:
        affine = np.eye(4)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(data, affine)
    
    # Save if output path provided
    if output_path:
        nib.save(nifti_img, output_path)
        return nifti_img, output_path
    
    return nifti_img

def reorient_nifti(nifti_path, orientation='RAS', output_path=None):
    """
    Reorient a NIfTI file to a specified orientation.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        orientation (str): Target orientation code (e.g., 'RAS', 'LPS')
        output_path (str, optional): Path to save reoriented NIfTI file
        
    Returns:
        str: Path to output NIfTI file
    """
    # Load NIfTI file
    nifti_img = nib.load(nifti_path)
    
    # Define target orientation
    target_orientation = nib.orientations.axcodes2ornt(orientation)
    
    # Get current orientation
    current_orientation = nib.io_orientation(nifti_img.affine)
    
    # Create transformation between orientations
    transform = nib.orientations.ornt_transform(current_orientation, target_orientation)
    
    # Apply transformation
    reoriented_img = nifti_img.as_reoriented(transform)
    
    # Save if output path provided
    if output_path is None:
        base_name = os.path.basename(os.path.splitext(nifti_path)[0])
        output_path = os.path.join(os.path.dirname(nifti_path), f"{base_name}_{orientation}.nii.gz")
    
    nib.save(reoriented_img, output_path)
    
    return output_path

def resample_nifti(nifti_path, new_voxel_size, output_path=None, interpolation='linear'):
    """
    Resample a NIfTI file to a new voxel size.
    
    Args:
        nifti_path (str): Path to input NIfTI file
        new_voxel_size (tuple): New voxel dimensions (x, y, z) in mm
        output_path (str, optional): Path to save resampled NIfTI file
        interpolation (str): Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        str: Path to output NIfTI file
    """
    # Import here to avoid dependency issues
    from scipy.ndimage import zoom
    
    # Load NIfTI file
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    
    # Get current voxel size
    current_voxel_size = nifti_img.header.get_zooms()[:3]
    
    # Calculate zoom factors
    zoom_factors = [curr / new for curr, new in zip(current_voxel_size, new_voxel_size)]
    
    # Determine interpolation order
    if interpolation == 'nearest':
        order = 0
    elif interpolation == 'linear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    else:
        order = 1  # Default to linear
    
    # Resample data
    resampled_data = zoom(data, zoom_factors, order=order)
    
    # Update affine matrix
    affine = nifti_img.affine.copy()
    
    # Scale the voxel size in the affine
    for i in range(3):
        affine[i, i] = affine[i, i] * (current_voxel_size[i] / new_voxel_size[i])
    
    # Create resampled NIfTI image
    resampled_img = nib.Nifti1Image(resampled_data, affine, nifti_img.header)
    
    # Update header with new voxel size
    resampled_img.header.set_zooms(new_voxel_size)
    
    # Save if output path provided
    if output_path is None:
        base_name = os.path.basename(os.path.splitext(nifti_path)[0])
        size_str = "x".join([str(s) for s in new_voxel_size])
        output_path = os.path.join(os.path.dirname(nifti_path), f"{base_name}_resampled_{size_str}.nii.gz")
    
    nib.save(resampled_img, output_path)
    
    return output_path