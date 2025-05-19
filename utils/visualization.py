import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import plotly.graph_objects as go
from skimage import measure

def load_itk_snap_labels(label_file_path):
    """
    Load ITK-SnAP Label Description File and create a colormap.
    
    Args:
        label_file_path (str): Path to the .label file
        
    Returns:
        dict: Colormap where keys are label indices and values are [R,G,B] lists
    """
    colormap = {}
    
    try:
        with open(label_file_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 7:  # Make sure we have at least [idx, R, G, B, A, vis, mesh, "label"]
                try:
                    idx = int(parts[0])
                    r = int(parts[1])
                    g = int(parts[2])
                    b = int(parts[3])
                    colormap[idx] = [r, g, b]
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error loading label file: {e}")
        # Return empty colormap on error
        pass
        
    return colormap

def normalize_array(array, percentile_low=0, percentile_high=100, window_level=None, window_width=None):
    """
    Normalize array values to 0-255 range with optional window/level adjustment.
    
    Args:
        array (numpy.ndarray): Input array
        percentile_low (int): Lower percentile for contrast adjustment
        percentile_high (int): Higher percentile for contrast adjustment
        window_level (float, optional): Window level (center)
        window_width (float, optional): Window width
        
    Returns:
        numpy.ndarray: Normalized array (uint8)
    """
    # Apply window/level if provided
    if window_level is not None and window_width is not None:
        low_val = window_level - window_width // 2
        high_val = window_level + window_width // 2
        array = np.clip(array, low_val, high_val)
    # Otherwise use percentiles
    elif percentile_low > 0 or percentile_high < 100:
        low_val = np.percentile(array, percentile_low)
        high_val = np.percentile(array, percentile_high)
        array = np.clip(array, low_val, high_val)
    
    # Normalize to 0-255
    min_val = array.min()
    max_val = array.max()
    
    if max_val > min_val:
        norm_array = ((array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        norm_array = np.zeros_like(array, dtype=np.uint8)
        
    return norm_array

def display_slice(volume, slice_idx=None, view='axial', window_level=None, window_width=None, crosshair=None, 
                  add_orientation_marker=True, add_scale=True):
    """
    Display a slice from a 3D volume.
    
    Args:
        volume (numpy.ndarray): 3D volume
        slice_idx (int, optional): Slice index
        view (str, optional): View orientation ('axial', 'sagittal', 'coronal')
        window_level (float, optional): Window level (center)
        window_width (float, optional): Window width
        crosshair (tuple, optional): Crosshair position (x, y, z)
        add_orientation_marker (bool): Whether to add orientation markers
        add_scale (bool): Whether to add a scale bar
        
    Returns:
        numpy.ndarray: Slice image as uint8 RGB array
    """
    # Select slice based on view
    if view.lower() == 'axial':
        if slice_idx is None or slice_idx >= volume.shape[0]:
            slice_idx = volume.shape[0] // 2
        slice_data = volume[slice_idx, :, :]
        orientation = 'ALPR'  # Anterior-Left-Posterior-Right
    elif view.lower() == 'sagittal':
        if slice_idx is None or slice_idx >= volume.shape[2]:
            slice_idx = volume.shape[2] // 2
        slice_data = volume[:, :, slice_idx]
        orientation = 'SAHF'  # Superior-Anterior-Inferior-Posterior
    elif view.lower() == 'coronal':
        if slice_idx is None or slice_idx >= volume.shape[1]:
            slice_idx = volume.shape[1] // 2
        slice_data = volume[:, slice_idx, :]
        orientation = 'SLIR'  # Superior-Left-Inferior-Right
    else:
        raise ValueError(f"Invalid view: {view}")
    
    # Normalize to 0-255
    norm_slice = normalize_array(slice_data, window_level=window_level, window_width=window_width)
    
    # Convert to RGB for display
    rgb_slice = np.stack((norm_slice,) * 3, axis=-1)
    
    # Create a PIL image for additional overlay elements
    pil_img = Image.fromarray(rgb_slice)
    draw = ImageDraw.Draw(pil_img)
    
    # Add crosshair if provided
    if crosshair is not None:
        x, y, z = crosshair
        
        if view.lower() == 'axial':
            # Draw horizontal and vertical lines through (x, y)
            draw.line((0, y, pil_img.width, y), fill=(255, 0, 0), width=1)
            draw.line((x, 0, x, pil_img.height), fill=(255, 0, 0), width=1)
        elif view.lower() == 'sagittal':
            # Draw lines through (y, z)
            draw.line((0, z, pil_img.width, z), fill=(255, 0, 0), width=1)
            draw.line((y, 0, y, pil_img.height), fill=(255, 0, 0), width=1)
        elif view.lower() == 'coronal':
            # Draw lines through (x, z)
            draw.line((0, z, pil_img.width, z), fill=(255, 0, 0), width=1)
            draw.line((x, 0, x, pil_img.height), fill=(255, 0, 0), width=1)
    
    # Add orientation markers
    if add_orientation_marker:
        font_size = 20
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            # Use default font if arial is not available
            font = ImageFont.load_default()
        
        margin = 10
        
        # Draw orientation labels based on view
        if view.lower() == 'axial':
            # A at top, P at bottom, L at right, R at left
            draw.text((pil_img.width // 2, margin), "A", fill=(0, 255, 0), font=font)
            draw.text((pil_img.width // 2, pil_img.height - margin - font_size), "P", fill=(0, 255, 0), font=font)
            draw.text((margin, pil_img.height // 2), "L", fill=(0, 255, 0), font=font)
            draw.text((pil_img.width - margin - font_size, pil_img.height // 2), "R", fill=(0, 255, 0), font=font)
        elif view.lower() == 'sagittal':
            # S at top, I at bottom, A at right, P at left
            draw.text((pil_img.width // 2, margin), "S", fill=(0, 255, 0), font=font)
            draw.text((pil_img.width // 2, pil_img.height - margin - font_size), "I", fill=(0, 255, 0), font=font)
            draw.text((margin, pil_img.height // 2), "A", fill=(0, 255, 0), font=font)
            draw.text((pil_img.width - margin - font_size, pil_img.height // 2), "P", fill=(0, 255, 0), font=font)
        elif view.lower() == 'coronal':
            # S at top, I at bottom, L at right, R at left
            draw.text((pil_img.width // 2, margin), "S", fill=(0, 255, 0), font=font)
            draw.text((pil_img.width // 2, pil_img.height - margin - font_size), "I", fill=(0, 255, 0), font=font)
            draw.text((margin, pil_img.height // 2), "L", fill=(0, 255, 0), font=font)
            draw.text((pil_img.width - margin - font_size, pil_img.height // 2), "R", fill=(0, 255, 0), font=font)
    
    # Add scale bar (10% of width)
    if add_scale:
        scale_length = pil_img.width // 10
        scale_height = 5
        scale_margin = 20
        
        # Draw scale bar at bottom right
        draw.rectangle(
            (pil_img.width - scale_margin - scale_length, 
             pil_img.height - scale_margin - scale_height,
             pil_img.width - scale_margin, 
             pil_img.height - scale_margin),
            fill=(255, 255, 255)
        )
    
    # Add current slice info
    slice_info_text = f"Slice: {slice_idx}"
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw slice info at top left
    draw.text((10, 10), slice_info_text, fill=(255, 255, 0), font=font)
    
    # Convert back to numpy array
    result_img = np.array(pil_img)
    
    return result_img

def create_crosshair_overlay(slice_img, position, view='axial'):
    """
    Create a crosshair overlay on the slice image.
    
    Args:
        slice_img (numpy.ndarray): 2D slice image
        position (tuple): 3D position (x, y, z)
        view (str): View orientation ('axial', 'sagittal', 'coronal')
        
    Returns:
        numpy.ndarray: Image with crosshair overlay
    """
    # Convert to PIL Image
    pil_img = Image.fromarray(slice_img)
    draw = ImageDraw.Draw(pil_img)
    
    # Extract crosshair coordinates based on view
    x, y, z = position
    if view.lower() == 'axial':
        h_line = (0, y, slice_img.shape[1], y)
        v_line = (x, 0, x, slice_img.shape[0])
    elif view.lower() == 'sagittal':
        h_line = (0, z, slice_img.shape[1], z)
        v_line = (y, 0, y, slice_img.shape[0])
    elif view.lower() == 'coronal':
        h_line = (0, z, slice_img.shape[1], z)
        v_line = (x, 0, x, slice_img.shape[0])
    
    # Draw crosshair
    draw.line(h_line, fill=(255, 0, 0), width=1)
    draw.line(v_line, fill=(255, 0, 0), width=1)
    
    # Convert back to numpy array
    result = np.array(pil_img)
    
    return result

def multi_planar_view(volume, position=None):
    """
    Create a multi-planar reconstruction view (axial, sagittal, coronal).
    
    Args:
        volume (numpy.ndarray): 3D volume
        position (tuple, optional): 3D position (x, y, z)
        
    Returns:
        numpy.ndarray: Combined MPR view as numpy array
    """
    if position is None:
        position = (volume.shape[2] // 2, volume.shape[1] // 2, volume.shape[0] // 2)
    
    x, y, z = position
    
    # Extract slices at the given position
    axial_slice = display_slice(volume, z, 'axial', crosshair=(x, y, z))
    sagittal_slice = display_slice(volume, x, 'sagittal', crosshair=(x, y, z))
    coronal_slice = display_slice(volume, y, 'coronal', crosshair=(x, y, z))
    
    # Get dimensions
    axial_h, axial_w, _ = axial_slice.shape
    sagittal_h, sagittal_w, _ = sagittal_slice.shape
    coronal_h, coronal_w, _ = coronal_slice.shape
    
    # Calculate output dimensions
    out_width = max(axial_w, coronal_w) + sagittal_w
    out_height = max(axial_h + coronal_h, sagittal_h)
    
    # Create output image
    output = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    
    # Place slices in the output image
    output[:axial_h, :axial_w] = axial_slice
    output[axial_h:axial_h+coronal_h, :coronal_w] = coronal_slice
    output[:sagittal_h, axial_w:axial_w+sagittal_w] = sagittal_slice
    
    return output

def overlay_segmentation(base_img, seg_slice, alpha=0.5, colormap=None):
    """
    Overlay a segmentation slice on top of a base image
    
    Parameters:
    -----------
    base_img : ndarray
        The base image to overlay segmentation on
    seg_slice : ndarray
        The segmentation slice (should be same dimensions as base_img)
    alpha : float
        Transparency factor (0-1)
    colormap : dict
        Maps label values to RGB colors
        
    Returns:
    --------
    ndarray
        The base image with segmentation overlay
    """
    # Make sure arrays are numpy arrays
    base_img = np.asarray(base_img)
    seg_slice = np.asarray(seg_slice)
    
    # Check that dimensions match
    if base_img.shape[:2] != seg_slice.shape[:2]:
        raise ValueError(f"Base image shape {base_img.shape} and segmentation shape {seg_slice.shape} don't match")
    
    # Convert grayscale to RGB if needed
    if len(base_img.shape) == 2:
        rgb_img = np.stack([base_img] * 3, axis=2)
    else:
        rgb_img = base_img.copy()
    
    # Default colormap if none provided
    if colormap is None:
        colormap = {
            0: [0, 0, 0],       # Background (transparent)
            1: [255, 0, 0],     # Label 1 (Red)
            2: [0, 255, 0],     # Label 2 (Green)
            3: [0, 0, 255],     # Label 3 (Blue)
            4: [255, 255, 0],   # Label 4 (Yellow)
            5: [0, 255, 255],   # Label 5 (Cyan)
            6: [255, 0, 255],   # Label 6 (Magenta)
        }
    
    # Create RGB overlay image
    overlay = np.zeros((*seg_slice.shape, 3), dtype=np.uint8)
    
    # Apply colormap to each unique label
    unique_labels = np.unique(seg_slice)
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        
        # Get color for this label (default to white if not in colormap)
        color = colormap.get(label, [255, 255, 255])
        
        # Create mask for this label
        mask = (seg_slice == label)
        
        # Apply color to overlay
        for i in range(3):  # RGB channels
            overlay[mask, i] = color[i]
    
    # Apply overlay with alpha blending
    mask = (seg_slice > 0)  # All non-zero labels
    if np.any(mask):  # Only blend if we have segmentation
        for i in range(3):  # RGB channels
            rgb_img[mask, i] = (1 - alpha) * rgb_img[mask, i] + alpha * overlay[mask, i]
    
    return rgb_img

def intensity_projection(volume, axis=0, mode='max', thickness=10):
    """
    Create an intensity projection along a specified axis.
    
    Args:
        volume (numpy.ndarray): 3D volume
        axis (int): Axis along which to project (0, 1, or 2)
        mode (str): Projection mode ('max', 'min', 'mean')
        thickness (int): Number of slices to include in projection
        
    Returns:
        numpy.ndarray: Projection image
    """
    if mode == 'max':
        proj_fn = np.max
    elif mode == 'min':
        proj_fn = np.min
    elif mode == 'mean':
        proj_fn = np.mean
    else:
        raise ValueError(f"Invalid projection mode: {mode}")
    
    # Select slices to include in projection
    mid_slice = volume.shape[axis] // 2
    start_slice = max(0, mid_slice - thickness // 2)
    end_slice = min(volume.shape[axis], mid_slice + thickness // 2)
    
    # Create projection
    if axis == 0:
        projection = proj_fn(volume[start_slice:end_slice, :, :], axis=0)
    elif axis == 1:
        projection = proj_fn(volume[:, start_slice:end_slice, :], axis=1)
    else:  # axis == 2
        projection = proj_fn(volume[:, :, start_slice:end_slice], axis=2)
    
    # Normalize the projection
    projection = normalize_array(projection)
    
    # Convert to RGB
    rgb_projection = np.stack((projection,) * 3, axis=-1)
    
    return rgb_projection

def segmentation_to_shapes(seg_slice, label=None):
    """
    Convert a segmentation slice to Plotly shapes for interactive editing.
    
    Args:
        seg_slice (numpy.ndarray): 2D segmentation mask
        label (int, optional): Specific label to convert, if None convert all labels
        
    Returns:
        list: List of dictionaries defining Plotly shapes
    """
    # Before using this function, make sure skimage is installed: pip install scikit-image
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image is required for contour detection. Please install with 'pip install scikit-image'")
    
    shapes = []
    
    # Get unique labels
    unique_labels = np.unique(seg_slice)
    if 0 in unique_labels:  # Remove background
        unique_labels = unique_labels[unique_labels != 0]
    
    # If specific label is requested, filter to just that label
    if label is not None and label in unique_labels:
        unique_labels = [label]
    
    # Process each label
    for label_val in unique_labels:
        # Create binary mask for this label
        binary_mask = (seg_slice == label_val)
        
        # Find contours using skimage
        contours = measure.find_contours(binary_mask, 0.5)
        
        for contour in contours:
            # Skip very small contours (likely noise)
            if len(contour) < 5:
                continue
            
            # Reduce the number of points for better performance and editing
            # but keep enough to maintain shape fidelity
            max_points = 15  # Target maximum number of points - reduced for better performance
            if len(contour) > max_points:
                # Simplify by taking every nth point
                step = len(contour) // max_points + 1
                contour = contour[::step]
            
            # Format points for Plotly - swap x and y coordinates as skimage returns (row, column)
            y_coords, x_coords = contour[:, 0], contour[:, 1]
            
            # Generate a color based on the label value
            color = f'rgba({label_val * 50 % 255}, {(label_val * 100) % 255}, {(label_val * 70) % 255}, 1)'
            fill_color = f'rgba({label_val * 50 % 255}, {(label_val * 100) % 255}, {(label_val * 70) % 255}, 0.3)'
            
            # Create a proper path that supports blending in Plotly's native format
            # This uses the SVG path format that Plotly understands
            path_str = f'M {x_coords[0]},{y_coords[0]} '
            for i in range(1, len(x_coords)):
                path_str += f'L {x_coords[i]},{y_coords[i]} '
            path_str += 'Z'  # Close the path
              # Create a single, editable path shape with the contour
            # This will work more like Plotly's native "Draw Area" tool
            shape = {
                'type': 'path',
                'path': path_str,
                'line': {
                    'color': color,
                    'width': 2,
                },
                'fillcolor': fill_color,
                'editable': True,  # This is key to allow editing the shape
                'name': f'Label {label_val}',
                # Setting fillrule and layer to make the shape selectable from inside
                'fillrule': 'evenodd',
                'layer': 'above',
                # Adding opacity to make sure it's visible and clickable
                'opacity': 1.0
            }
            shapes.append(shape)
    
    return shapes