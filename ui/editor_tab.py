"""
SegMed-Pro Editor Tab UI Components

This module contains the UI layout for the Editor tab,
including data loading, visualization with advanced annotation capabilities
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from gradio_image_annotation import image_annotator
from .viewer_tab import create_data_loading_section
import json
import os
import glob
import logging
from scipy import ndimage
from skimage import measure
import cv2
import nibabel as nib  # Import nibabel for NIfTI file handling
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

# Set up logger for this module
logger = logging.getLogger(__name__)

# Global dictionary to store slice-specific suggested labels
SLICE_SUGGESTED_LABELS = {}

# Global dictionary to track selected suggested labels (slice_index -> list of selected labels)
SELECTED_SUGGESTED_LABELS = {}

def load_annotation_data(output_dir="brain_target_results"):
    """Load annotation data from the results directory"""
    try:
        summary_path = os.path.join(output_dir, "summary.json")
        if not os.path.exists(summary_path):
            return None, "No annotation data found"
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        return summary, None
    except Exception as e:
        return None, f"Error loading annotation data: {str(e)}"

def load_slice_masks(output_dir="brain_target_results", slice_indices=None, score_threshold=None):
    """Load mask data for specified slices, applying the EXACT SAME score threshold as image_annotator"""
    logger.info(f"load_slice_masks called with score_threshold={score_threshold} (SAME filtering as image_annotator)")
    masks = {}
    masks_dir = os.path.join(output_dir, "masks")
    
    if not os.path.exists(masks_dir):
        return masks
    
    if slice_indices is None:
        return masks
    
    # Load scores if score_threshold is provided - USE SAME LOGIC AS image_annotator
    scores_data = {}
    if score_threshold is not None:
        try:
            # Load scores from summary.json EXACTLY like image_annotator does
            summary_path = os.path.join(output_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary_data = json.load(f)
                
                summary_slice_indices = summary_data.get('slice_indices', [])
                average_scores = summary_data.get('average_scores', [])
                
                if summary_slice_indices and average_scores:
                    logger.info(f"Using summary.json scores - {len(average_scores)} scores from {len(summary_slice_indices)} slices (SAME as image_annotator)")
                    
                    # Check if we have multiple scores per slice (SAME logic as image_annotator)
                    if len(average_scores) > len(summary_slice_indices):
                        scores_per_slice = len(average_scores) // len(summary_slice_indices)
                        logger.info(f"Detected {scores_per_slice} scores per slice (SAME as image_annotator)")
                        
                        # Process each score individually (SAME as image_annotator)
                        for slice_idx in summary_slice_indices:
                            slice_scores = []
                            for score_offset in range(scores_per_slice):
                                score_index = summary_slice_indices.index(slice_idx) * scores_per_slice + score_offset
                                if score_index < len(average_scores):
                                    slice_scores.append(average_scores[score_index])
                            
                            if slice_scores:
                                scores_data[slice_idx] = np.array(slice_scores)
                                logger.info(f"Loaded {len(slice_scores)} scores for slice {slice_idx}: {slice_scores} (from summary.json)")
                    else:
                        # Single score per slice (SAME as image_annotator)
                        for slice_idx, score in zip(summary_slice_indices, average_scores):
                            scores_data[slice_idx] = np.array([score])
                            logger.info(f"Loaded score for slice {slice_idx}: {score} (from summary.json)")
                else:
                    logger.warning("No slice data found in summary.json")
            else:
                logger.warning("No summary.json found, cannot apply score filtering")
        except Exception as e:
            logger.error(f"Error loading scores from summary.json: {e}")      # Process all requested slices for 3D visualization (including empty ones)
    # but apply score filtering only to slices that have annotation data
    slices_to_process = slice_indices
    
    # For score threshold filtering, we'll check each slice individually
    # This allows us to include empty slices while still filtering annotated ones
    if score_threshold is not None:
        logger.info(f"Score threshold {score_threshold} will be applied to slices with annotation data")
        logger.info(f"Processing complete slice range: {min(slice_indices)} to {max(slice_indices)} ({len(slice_indices)} slices)")
        logger.info(f"Slices with score data: {sorted(scores_data.keys()) if scores_data else 'None'}")
    else:
        logger.info(f"No score threshold - processing all {len(slice_indices)} slices")
    
    for slice_idx in slices_to_process:
        # Determine if this slice should have annotation data or be empty
        should_load_annotation = True
        
        # Apply score threshold filtering if available (SAME logic as image_annotator)
        if score_threshold is not None and slice_idx in scores_data:
            scores = scores_data[slice_idx]
            # Process individual scores like image_annotator does (SAME logic as filter_manual_results_by_score)
            qualifying_scores = 0            
            if isinstance(scores, np.ndarray):
                logger.info(f"Processing {len(scores)} scores for slice {slice_idx} (SAME as image_annotator)")
                for i, score in enumerate(scores):
                    if score >= score_threshold:
                        qualifying_scores += 1
                        logger.info(f"Slice {slice_idx} annotation {i} passed - score {score:.6f} >= threshold {score_threshold} (SAME as image_annotator)")
                    else:
                        logger.info(f"Slice {slice_idx} annotation {i} filtered - score {score:.6f} < threshold {score_threshold} (SAME as image_annotator)")
                
                # Only load annotation if it has at least one qualifying score
                if qualifying_scores == 0:
                    logger.info(f"Slice {slice_idx} - no qualifying annotations, will be empty in 3D (SAME filtering as image_annotator)")
                    should_load_annotation = False
                else:
                    logger.info(f"Slice {slice_idx} included with {qualifying_scores} qualifying annotations (SAME filtering as image_annotator)")
            else:
                # Single score
                if scores >= score_threshold:
                    qualifying_scores = 1
                    logger.info(f"Slice {slice_idx} included - score {scores:.6f} >= threshold {score_threshold} (SAME filtering as image_annotator)")
                else:
                    logger.info(f"Slice {slice_idx} filtered out - score {scores:.6f} < threshold {score_threshold}, will be empty in 3D (SAME filtering as image_annotator)")
                    should_load_annotation = False
        elif score_threshold is not None:
            # No score data for this slice - it should be empty
            logger.info(f"Slice {slice_idx} - no score data, will be empty in 3D")
            should_load_annotation = False
        
        # Try both 4-digit and 3-digit formatting, but only if we should load annotation
        mask_file = None
        if should_load_annotation:
            mask_file_4digit = os.path.join(masks_dir, f"slice_{slice_idx:04d}_mask.npy")
            mask_file_3digit = os.path.join(masks_dir, f"slice_{slice_idx:03d}_mask.npy")
            
            if os.path.exists(mask_file_4digit):
                mask_file = mask_file_4digit
            elif os.path.exists(mask_file_3digit):
                mask_file = mask_file_3digit
        
        if mask_file and should_load_annotation:
            try:
                mask = np.load(mask_file)
                # Apply the EXACT same binary threshold as image_annotator overlay_segmentation function (> 0)
                # This matches the logic in utils/visualization.py overlay_segmentation where mask = (seg_slice > 0)
                # The image_annotator uses overlay_segmentation which shows all pixels where seg_slice > 0
                binary_mask = (mask > 0).astype(np.uint8)
                masks[slice_idx] = binary_mask
                logger.info(f"3D Viewer: Loaded and thresholded mask for slice {slice_idx}: {mask.shape}, unique values: {np.unique(binary_mask)} (SAME > 0 threshold as image_annotator)")
            except Exception as e:
                logger.error(f"Error loading mask for slice {slice_idx}: {e}")
                # Create empty mask for failed loads to maintain spatial continuity
                if slice_indices:  # Ensure we have a reference for dimensions
                    # Try to get dimensions from other masks or use default
                    reference_mask = None
                    for ref_idx in slice_indices:
                        ref_file_4digit = os.path.join(masks_dir, f"slice_{ref_idx:04d}_mask.npy")
                        ref_file_3digit = os.path.join(masks_dir, f"slice_{ref_idx:03d}_mask.npy")
                        if os.path.exists(ref_file_4digit):
                            try:
                                reference_mask = np.load(ref_file_4digit)
                                break
                            except:
                                continue
                        elif os.path.exists(ref_file_3digit):
                            try:
                                reference_mask = np.load(ref_file_3digit)
                                break
                            except:
                                continue
                    
                    if reference_mask is not None:
                        masks[slice_idx] = np.zeros(reference_mask.shape, dtype=np.uint8)
                        logger.info(f"3D Viewer: Created empty mask for slice {slice_idx} (failed load)")
        else:
            # No mask file found OR we shouldn't load annotation - create empty mask for spatial continuity
            if should_load_annotation:
                logger.info(f"No mask file found for slice {slice_idx} - creating empty mask for spatial continuity")
            else:
                logger.info(f"Slice {slice_idx} filtered by score threshold - creating empty mask for 3D continuity")
            
            # Try to get dimensions from existing masks or other available mask files
            reference_mask = None
            
            # First, check if we already have a mask loaded
            if masks:
                reference_mask = next(iter(masks.values()))
            else:
                # Try to load any available mask to get dimensions
                for ref_idx in slice_indices:
                    if ref_idx == slice_idx:
                        continue
                    ref_file_4digit = os.path.join(masks_dir, f"slice_{ref_idx:04d}_mask.npy")
                    ref_file_3digit = os.path.join(masks_dir, f"slice_{ref_idx:03d}_mask.npy")
                    if os.path.exists(ref_file_4digit):
                        try:
                            reference_mask = np.load(ref_file_4digit)
                            break
                        except:
                            continue
                    elif os.path.exists(ref_file_3digit):
                        try:
                            reference_mask = np.load(ref_file_3digit)
                            break
                        except:
                            continue
            
            if reference_mask is not None:
                # Create empty mask with same dimensions
                masks[slice_idx] = np.zeros(reference_mask.shape, dtype=np.uint8)
                logger.info(f"3D Viewer: Created empty mask for slice {slice_idx}: {reference_mask.shape}")
            else:
                logger.warning(f"Could not determine mask dimensions for empty slice {slice_idx}")# Add summary logging to match image_annotator format EXACTLY
    if score_threshold is not None:
        included_slices = list(masks.keys())
        # Count actual qualifying scores per slice (SAME logic as image_annotator)
        annotations_per_slice = {}
        total_processed_scores = 0
        total_slices_processed = len(scores_data)
        
        # Process only slices that were in summary.json (SAME as image_annotator)
        for slice_idx in scores_data.keys():
            scores = scores_data[slice_idx]
            qualifying_count = 0
            
            if isinstance(scores, np.ndarray):
                total_processed_scores += len(scores)
                for score in scores:
                    if score >= score_threshold:
                        qualifying_count += 1
            else:
                total_processed_scores += 1
                if scores >= score_threshold:
                    qualifying_count += 1
                    
            if qualifying_count > 0:
                annotations_per_slice[slice_idx] = qualifying_count
        
        total_annotations = sum(annotations_per_slice.values())
        filtered_annotations = total_processed_scores - total_annotations
        
        # Log in EXACT same format as image_annotator
        logger.info(f"All records annotation completed successfully! Manual annotation processed {total_processed_scores} scores from {total_slices_processed} slices. {total_annotations} annotations passed threshold (>= {score_threshold}), {filtered_annotations} filtered out.")
        if annotations_per_slice:
            logger.info(f"Annotations per slice: {dict(sorted(annotations_per_slice.items()))}")
        
        # Check for slices with no qualifying annotations
        missing_slices = []
        for slice_idx in slice_indices:
            if slice_idx not in annotations_per_slice:
                missing_slices.append(slice_idx)
        
        if missing_slices:
            for slice_idx in missing_slices:
                logger.info(f"No annotation available for slice {slice_idx} (may not meet score threshold))")
    
    return masks

def create_3d_volume_from_masks(masks, slice_spacing=1.0):
    """Create a 3D volume from 2D slice masks, including empty slices for spatial continuity"""
    if not masks:
        return None
    
    slice_indices = sorted(masks.keys())
    if len(slice_indices) < 1:
        return None
    
    # Get mask dimensions from the first available mask
    first_mask = None
    for slice_idx in slice_indices:
        if np.any(masks[slice_idx] > 0):  # Find first non-empty mask
            first_mask = masks[slice_idx]
            break
    
    if first_mask is None:
        # If all masks are empty, use the first one for dimensions
        first_mask = masks[slice_indices[0]]
    
    height, width = first_mask.shape
    
    # Create volume spanning complete range from min to max slice
    min_slice = min(slice_indices)
    max_slice = max(slice_indices)
    volume_depth = max_slice - min_slice + 1
    
    logger.info(f"Creating 3D volume: dimensions {height}x{width}x{volume_depth}, slice range {min_slice}-{max_slice}")
    
    volume = np.zeros((height, width, volume_depth), dtype=np.uint8)
    
    # Fill volume with masks (empty slices will remain as zeros)
    annotated_count = 0
    empty_count = 0
    
    for slice_idx in slice_indices:
        z_pos = slice_idx - min_slice
        mask_data = masks[slice_idx]
        
        if np.any(mask_data > 0):
            volume[:, :, z_pos] = (mask_data > 0).astype(np.uint8)
            annotated_count += 1
        else:
            # Explicitly keep as zeros for empty slices
            empty_count += 1
    
    logger.info(f"3D Volume: {annotated_count} slices with annotations, {empty_count} empty slices, total depth: {volume_depth}")
    
    # Apply morphological closing to connect nearby regions across slices
    # Use a smaller kernel to preserve spatial accuracy while still connecting nearby structures
    if annotated_count > 1:  # Only apply morphological closing if we have multiple annotated slices
        volume = ndimage.binary_closing(volume, structure=np.ones((2, 2, 2))).astype(np.uint8)
        logger.info("Applied morphological closing to connect nearby structures across slices")
    else:
        logger.info("Single annotated slice - skipping morphological closing")
    
    return volume

def create_3d_mesh_from_volume(volume, threshold=0.5, slice_offset=0):
    """Create 3D mesh from volume using marching cubes with proper slice positioning"""
    if volume is None:
        return None, None, None
    
    try:
        # Use marching cubes to extract isosurface
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=threshold, spacing=(1, 1, 1), method='lewiner'
        )
        
        # Adjust the Z coordinates (slice direction) to use actual slice numbers
        # The volume Z-axis corresponds to slice indices, so add the offset
        verts[:, 2] = verts[:, 2] + slice_offset
        
        logger.info(f"3D Mesh: Applied slice offset {slice_offset}, Z range: {verts[:, 2].min():.1f} to {verts[:, 2].max():.1f}")
        
        return verts, faces, normals
    except Exception as e:
        logger.error(f"Error creating 3D mesh: {e}")
        return None, None, None

def create_3d_visualization(output_dir="brain_target_results", score_threshold=None):
    """Create 3D visualization of annotated regions using the same thresholding as image_annotator"""
    # Load annotation data
    summary, error = load_annotation_data(output_dir)
    if error:
        return create_empty_3d_plot(f"Error: {error}")
    
    # Get annotated slice indices - try from summary first, then auto-detect from available masks
    annotated_slice_indices = []
    if summary and 'slice_indices' in summary:
        annotated_slice_indices = summary['slice_indices']
    
    # If no slice indices in summary or less than 2, auto-detect from mask files
    if len(annotated_slice_indices) < 2:
        masks_dir = os.path.join(output_dir, "masks")
        if os.path.exists(masks_dir):
            import glob
            # Look for mask files with either 3 or 4 digit formatting
            mask_files = glob.glob(os.path.join(masks_dir, "*_mask.npy"))
            detected_indices = []
            for mask_file in mask_files:
                filename = os.path.basename(mask_file)
                # Extract slice number from filename                
                if filename.startswith("slice_") and filename.endswith("_mask.npy"):
                    slice_num_str = filename[6:-9]  # Remove "slice_" and "_mask.npy"
                    try:
                        slice_num = int(slice_num_str)
                        detected_indices.append(slice_num)
                    except ValueError:
                        continue
            annotated_slice_indices = sorted(detected_indices)
            logger.info(f"Auto-detected annotated slice indices: {annotated_slice_indices}")
    
    if len(annotated_slice_indices) < 1:  # Allow single slice for consistency
        return create_empty_3d_plot("No annotated slices found for 3D visualization")
    
    # Determine the full dataset range (0 to total_slices-1)
    # Try multiple methods to find the complete dataset size
    total_slices = None
    dataset_min_slice = 0  # Assume dataset starts from 0
    
    # Method 1: Check summary for total slice count
    if summary and 'total_slices' in summary:
        total_slices = summary['total_slices']
        logger.info(f"Found total_slices in summary: {total_slices}")
      # Method 2: Check windowing metadata which might contain volume info
    if total_slices is None:
        windowing_path = os.path.join(output_dir, "windowing_metadata.json")
        if os.path.exists(windowing_path):
            try:
                with open(windowing_path, 'r') as f:
                    windowing_data = json.load(f)
                if 'volume_shape' in windowing_data:
                    # Assuming volume_shape is [height, width, depth]
                    total_slices = windowing_data['volume_shape'][2] if len(windowing_data['volume_shape']) > 2 else None
                    logger.info(f"Found total_slices from windowing_metadata: {total_slices}")
            except Exception as e:
                logger.warning(f"Could not read windowing metadata: {e}")
    
    # Method 2.5: Try to examine original dataset directories for DICOM files
    if total_slices is None:
        # Look for common DICOM directories relative to the output directory
        possible_data_dirs = [
            os.path.join(os.path.dirname(output_dir), "cvm_48_t1"),
            os.path.join(os.path.dirname(output_dir), "cvm_t1"),
            # Add more possible directories as needed
        ]
        
        for data_dir in possible_data_dirs:
            if os.path.exists(data_dir):
                dicom_files = glob.glob(os.path.join(data_dir, "*.dcm"))
                if dicom_files:
                    total_slices = len(dicom_files)
                    logger.info(f"Found total_slices from DICOM directory {data_dir}: {total_slices}")
                    break
      # Method 3: For datasets starting from 0, use max annotated slice + some buffer
    # This assumes the dataset goes from 0 to at least the highest annotated slice
    if total_slices is None:
        max_annotated = max(annotated_slice_indices)
        min_annotated = min(annotated_slice_indices)
        
        # For your specific case: 20 slices indexed 0-19
        if min_annotated == 0 and max_annotated < 20:
            total_slices = 20  # Known dataset size
            logger.info(f"Detected standard 20-slice dataset (0-19)")
        elif min_annotated == 0:
            # General case: round up to nearest reasonable size
            total_slices = max(20, ((max_annotated // 10) + 1) * 10)
            logger.info(f"Estimated total_slices for 0-based dataset: {total_slices} (based on max annotated: {max_annotated})")
        else:
            # Fall back to just the annotated range if not starting from 0
            total_slices = max_annotated - min_annotated + 1
            dataset_min_slice = min_annotated
            logger.info(f"Using annotated range as fallback: {min_annotated} to {max_annotated}")
    
    # Create complete dataset range 
    dataset_max_slice = dataset_min_slice + total_slices - 1
    complete_slice_range = list(range(dataset_min_slice, dataset_max_slice + 1))
    
    logger.info(f"Complete dataset range for 3D visualization: {dataset_min_slice} to {dataset_max_slice} ({len(complete_slice_range)} slices)")
    logger.info(f"Annotated slices: {annotated_slice_indices} ({len(annotated_slice_indices)} slices)")
    logger.info(f"Empty slices to be included: {len([s for s in complete_slice_range if s not in annotated_slice_indices])} slices")
    
    # Load masks for the complete slice range (most will be empty/missing)
    logger.info(f"Loading masks for complete dataset range with score_threshold: {score_threshold}")
    masks = load_slice_masks(output_dir, complete_slice_range, score_threshold)
    
    # Filter masks to only include those that actually have annotation data
    # This ensures we don't create empty 3D visualizations
    valid_masks = {k: v for k, v in masks.items() if np.any(v > 0)}
    if not valid_masks:
        threshold_msg = f" (score threshold: {score_threshold})" if score_threshold else ""
        return create_empty_3d_plot(f"No valid mask data found{threshold_msg}")    
    
    logger.info(f"Loaded {len(masks)} masks from complete range, {len(valid_masks)} have annotation data")
    logger.info(f"Slices with valid annotation data: {sorted(valid_masks.keys())}")
    logger.info(f"Empty slices (will be zeros in 3D volume): {len([s for s in complete_slice_range if s not in valid_masks])} slices")
    
    if len(valid_masks) < 1:
        return create_empty_3d_plot("Need at least 1 annotated slice for 3D visualization")
    
    # Allow 3D visualization even with single annotated slice if there's a meaningful range
    if len(valid_masks) == 1 and len(complete_slice_range) < 3:
        return create_empty_3d_plot("Need at least 3 slices in range or 2+ annotated slices for meaningful 3D visualization")
      # Create 3D volume using masks (including empty slices for spatial continuity)
    volume = create_3d_volume_from_masks(masks)
    if volume is None:
        return create_empty_3d_plot("Could not create 3D volume")
    
    logger.info(f"3D Viewer: Created 3D volume with shape: {volume.shape}")
      # Create mesh with actual slice positioning
    min_slice = dataset_min_slice
    max_slice = dataset_max_slice
    verts, faces, normals = create_3d_mesh_from_volume(volume, threshold=0.5, slice_offset=min_slice)
    if verts is None:
        return create_empty_3d_plot("Could not create 3D mesh")
    
    logger.info(f"3D Viewer: Created 3D mesh with {len(verts)} vertices and {len(faces)} faces")
    logger.info(f"3D Viewer: Slice range in mesh: {verts[:, 2].min():.1f} to {verts[:, 2].max():.1f}")
    
    # Create plotly figure
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2],  # Z becomes X (slice direction) with actual slice numbers
            y=verts[:, 1],  # Y stays Y
            z=verts[:, 0],  # X becomes Z
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=verts[:, 2],
            colorscale='viridis',
            opacity=0.8,
            name='Annotated Region'
        )
    ])      # Update layout with black background and proper axis configuration
    fig.update_layout(
        title=f"3D Visualization - Slices {min_slice} to {max_slice} ({len(valid_masks)} annotated)",
        scene=dict(
            xaxis_title="Slice Number",
            yaxis_title="Width (pixels)",
            zaxis_title="Height (pixels)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor="black",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white"),
                tickmode='linear',
                dtick=1,  # Show integer tick marks
                range=[min_slice - 0.5, max_slice + 0.5]  # Proper range with padding
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            )        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        width=650,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_empty_3d_plot(message="No 3D data available"):
    """Create an empty 3D plot with a message and black background"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="white")
    )
    fig.update_layout(
        title="3D Viewer",
        scene=dict(
            xaxis_title="Slice Number",
            yaxis_title="Width (pixels)", 
            zaxis_title="Height (pixels)",
            bgcolor="black",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white"),
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            )        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        width=650,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def extract_current_labels_from_annotator(image_annotator_value):
    """Extract current label names from image_annotator shapes"""
    current_labels = []
    try:
        if image_annotator_value:
            logger.info(f"Extracting labels from image_annotator: {type(image_annotator_value)}")
            
            # Handle different possible structures of image_annotator_value
            annotations = None
            if isinstance(image_annotator_value, dict):
                # Try different possible keys
                annotations = (image_annotator_value.get('annotations') or 
                             image_annotator_value.get('shapes') or 
                             image_annotator_value.get('boxes') or [])
                logger.info(f"Found annotations in dict: {len(annotations) if annotations else 0}")
            elif isinstance(image_annotator_value, list):
                annotations = image_annotator_value
                logger.info(f"Image annotator value is list with {len(annotations)} items")
            
            if annotations:
                for i, shape in enumerate(annotations):
                    logger.info(f"Processing shape {i}: {type(shape)}")
                    if isinstance(shape, dict):
                        # Try different possible label keys
                        label = (shape.get('label') or 
                                shape.get('name') or 
                                shape.get('class_name') or 
                                shape.get('type'))
                        logger.info(f"Shape {i} label: {label}")
                        if label and str(label).strip() and label not in current_labels:
                            current_labels.append(str(label).strip())
                    
        logger.info(f"Extracted {len(current_labels)} unique labels: {current_labels}")
        return current_labels
    except Exception as e:
        logger.error(f"Error extracting labels from image_annotator: {e}")
        return []

def create_labels_dataset_samples(labels):
    """Create samples for gr.Dataset displaying labels as clickable items"""
    if not labels:
        logger.info("No labels provided, returning empty samples")
        return []
    
    # gr.Dataset with empty components still expects samples as list of lists
    # Each inner list represents one row, with one element per row for labels
    samples = []
    for label in labels:
        clean_label = str(label).strip()
        if clean_label:
            samples.append([clean_label])  # Wrapped in list for Dataset format
    
    logger.info(f"Created {len(samples)} dataset samples: {[s[0] for s in samples]}")
    return samples

def get_suggested_labels_for_slice(slice_index):
    """Get suggested labels for a specific slice"""
    global SLICE_SUGGESTED_LABELS
    slice_key = str(slice_index)
    if slice_key in SLICE_SUGGESTED_LABELS:
        logger.info(f"Retrieved {len(SLICE_SUGGESTED_LABELS[slice_key])} suggested labels for slice {slice_index}")
        return create_labels_dataset_samples(SLICE_SUGGESTED_LABELS[slice_key])
    else:
        logger.info(f"No suggested labels found for slice {slice_index}")
        return []

def store_suggested_labels_for_slice(slice_index, labels):
    """Store suggested labels for a specific slice - filter duplicates and existing labels"""
    global SLICE_SUGGESTED_LABELS
    slice_key = str(slice_index)
    
    # Remove duplicates while preserving order (case-insensitive)
    seen = set()
    unique_labels = []
    for label in labels:
        label_lower = str(label).lower().strip()
        if label_lower and label_lower not in seen:
            seen.add(label_lower)
            unique_labels.append(str(label).strip())
    
    # Additional filtering: check against existing suggested labels for this slice
    existing_labels = SLICE_SUGGESTED_LABELS.get(slice_key, [])
    existing_labels_lower = [label.lower() for label in existing_labels]
    
    # Filter out labels that already exist in suggestions
    filtered_labels = []
    for label in unique_labels:
        if label.lower() not in existing_labels_lower:
            filtered_labels.append(label)
    
    # Store the filtered, unique labels (append to existing, don't replace)
    if slice_key not in SLICE_SUGGESTED_LABELS:
        SLICE_SUGGESTED_LABELS[slice_key] = []
    
    # Add new labels to existing ones (avoiding duplicates)
    for label in filtered_labels:
        if label not in SLICE_SUGGESTED_LABELS[slice_key]:
            SLICE_SUGGESTED_LABELS[slice_key].append(label)
    
    total_stored = len(SLICE_SUGGESTED_LABELS[slice_key])
    logger.info(f"Stored {len(filtered_labels)} new suggested labels for slice {slice_index} (total: {total_stored}): {SLICE_SUGGESTED_LABELS[slice_key]}")

def filter_suggestions_against_current_labels(suggestions, current_labels):
    """Filter suggested labels against current labels to avoid duplicates"""
    try:
        if not current_labels:
            return suggestions
            
        # Create lowercase set for comparison
        current_labels_lower = set(label.lower() for label in current_labels)
        
        # Filter suggestions
        filtered_suggestions = []
        for suggestion in suggestions:
            if suggestion.lower() not in current_labels_lower:
                filtered_suggestions.append(suggestion)
        
        logger.info(f"Filtered {len(suggestions)} suggestions against {len(current_labels)} current labels: {len(filtered_suggestions)} remaining")
        return filtered_suggestions
        
    except Exception as e:
        logger.error(f"Error filtering suggestions against current labels: {e}")
        return suggestions  # Return original suggestions if filtering fails

def get_selected_suggested_labels_for_slice(slice_index):
    """Get selected suggested labels for a specific slice"""
    global SELECTED_SUGGESTED_LABELS
    slice_key = str(slice_index)
    return SELECTED_SUGGESTED_LABELS.get(slice_key, [])

def toggle_suggested_label_selection(slice_index, label_text):
    """Toggle selection of a suggested label for a specific slice"""
    global SELECTED_SUGGESTED_LABELS
    slice_key = str(slice_index)
    
    if slice_key not in SELECTED_SUGGESTED_LABELS:
        SELECTED_SUGGESTED_LABELS[slice_key] = []
    
    # Clean the label text (remove any selection indicators)
    clean_label = label_text.replace("✅ ", "").strip()
    
    if clean_label in SELECTED_SUGGESTED_LABELS[slice_key]:
        SELECTED_SUGGESTED_LABELS[slice_key].remove(clean_label)
        logger.info(f"Deselected suggested label '{clean_label}' for slice {slice_index}")
    else:
        SELECTED_SUGGESTED_LABELS[slice_key].append(clean_label)
        logger.info(f"Selected suggested label '{clean_label}' for slice {slice_index}")
    
    return SELECTED_SUGGESTED_LABELS[slice_key]

def clear_selected_suggested_labels_for_slice(slice_index):
    """Clear all selected suggested labels for a specific slice"""
    global SELECTED_SUGGESTED_LABELS
    slice_key = str(slice_index)
    SELECTED_SUGGESTED_LABELS[slice_key] = []
    logger.info(f"Cleared all selected suggested labels for slice {slice_index}")

def remove_labels_from_suggested_for_slice(slice_index, labels_to_remove):
    """Remove labels from suggested labels list for a specific slice (they've been accepted)"""
    global SLICE_SUGGESTED_LABELS
    slice_key = str(slice_index)
    
    if slice_key in SLICE_SUGGESTED_LABELS:
        original_count = len(SLICE_SUGGESTED_LABELS[slice_key])
        # Remove accepted labels from suggested labels
        SLICE_SUGGESTED_LABELS[slice_key] = [
            label for label in SLICE_SUGGESTED_LABELS[slice_key] 
            if label not in labels_to_remove
        ]
        removed_count = original_count - len(SLICE_SUGGESTED_LABELS[slice_key])
        logger.info(f"Removed {removed_count} accepted labels from suggested list for slice {slice_index}: {labels_to_remove}")
        return SLICE_SUGGESTED_LABELS[slice_key]
    else:
        return []

def save_labels_to_file(data_directory, slice_index, labels):
    """Save labels to a labels.txt file in the data directory - APPEND new labels to existing ones"""
    try:
        labels_file_path = os.path.join(data_directory, "labels.txt")
        
        # Read existing labels if file exists
        existing_labels = {}
        if os.path.exists(labels_file_path):
            with open(labels_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        slice_str, labels_str = line.split(':', 1)
                        try:
                            slice_num = int(slice_str.strip())
                            # Clean up labels and remove any extra whitespace/newlines
                            labels_list = []
                            for label in labels_str.split(','):
                                clean_label = ' '.join(label.strip().split())  # Remove extra whitespace/newlines
                                if clean_label:
                                    labels_list.append(clean_label)
                            existing_labels[slice_num] = labels_list
                        except ValueError:
                            continue
        
        # APPEND new labels to existing labels for the current slice (avoid duplicates)
        current_slice_labels = existing_labels.get(slice_index, [])
        for new_label in labels:
            # Clean the new label to remove any extra whitespace/newlines
            clean_new_label = ' '.join(str(new_label).strip().split())
            if clean_new_label and clean_new_label not in current_slice_labels:
                current_slice_labels.append(clean_new_label)
        
        existing_labels[slice_index] = current_slice_labels
        
        # Write updated labels back to file with proper formatting
        with open(labels_file_path, 'w') as f:
            for slice_num in sorted(existing_labels.keys()):
                if existing_labels[slice_num]:  # Only write slices that have labels
                    # Ensure all labels are clean and properly formatted
                    clean_labels = [' '.join(str(label).strip().split()) for label in existing_labels[slice_num] if str(label).strip()]
                    labels_str = ', '.join(clean_labels)
                    f.write(f"{slice_num}: {labels_str}\n")
        
        logger.info(f"APPENDED {len(labels)} new labels to slice {slice_index} (total: {len(current_slice_labels)} labels): {current_slice_labels}")
        return f"Added {len(labels)} labels to slice {slice_index} (total: {len(current_slice_labels)})"
        
    except Exception as e:
        logger.error(f"Error saving labels to file: {e}")
        return f"Error saving labels: {str(e)}"

def load_labels_from_file(data_directory, slice_index):
    """Load labels for a specific slice from labels.txt file"""
    try:
        labels_file_path = os.path.join(data_directory, "labels.txt")
        
        if not os.path.exists(labels_file_path):
            return []
        
        with open(labels_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    slice_str, labels_str = line.split(':', 1)
                    try:
                        slice_num = int(slice_str.strip())
                        if slice_num == slice_index:
                            # Clean up labels and remove any extra whitespace/newlines
                            labels_list = []
                            for label in labels_str.split(','):
                                clean_label = ' '.join(label.strip().split())  # Remove extra whitespace/newlines
                                if clean_label:
                                    labels_list.append(clean_label)
                            logger.info(f"Loaded {len(labels_list)} saved labels for slice {slice_index}: {labels_list}")
                            return labels_list
                    except ValueError:
                        continue
        
        return []  # No labels found for this slice
        
    except Exception as e:
        logger.error(f"Error loading labels from file: {e}")
        return []

def create_medgemma_label_suggestions(image_annotator_value, slice_index=None, current_labels=None):
    """Generate label suggestions using MedGemma and return as dataset samples"""
    try:
        if current_labels is None:
            current_labels = []
            
        # Import the MedGemma handlers
        from .medgemma_handlers import MedGemmaHandlers
        from .state import AppState
        
        # Create a temporary state and handler for the suggestion
        temp_state = AppState()
        medgemma_handler = MedGemmaHandlers(temp_state)
        
        # Get label suggestions using MedGemma
        suggestions_text = medgemma_handler.suggest_labels_for_annotations(image_annotator_value)
        
        # Parse the suggestions text to extract individual labels
        suggestions = []
        if suggestions_text and ("MedGemma" in suggestions_text or "label" in suggestions_text.lower()):
            # Extract the actual suggestions part
            content = suggestions_text.replace("MedGemma Label Suggestions:", "").strip()
            content = content.replace("MedGemma could not generate", "").strip()
            
            # Split by common delimiters and clean up - PRESERVE multi-word labels
            potential_labels = []
            for delimiter in [',', ';', '\n', '\t']:
                if delimiter in content:
                    potential_labels.extend([label.strip() for label in content.split(delimiter)])
                    break
            else:
                # If no delimiters found, try to extract phrases instead of individual words
                # Look for meaningful medical phrases first
                import re
                # Match common medical phrase patterns (2-3 words)
                phrase_patterns = [
                    r'\b(?:white|gray|grey)\s+matter\b',
                    r'\b(?:left|right)\s+\w+\b',
                    r'\b\w+\s+(?:tumor|lesion|mass|cyst)\b',
                    r'\b\w+\s+(?:ventricle|cortex|lobe)\b',
                ]
                
                for pattern in phrase_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    potential_labels.extend(matches)
                
                # Also extract individual words as fallback
                individual_words = content.split()
                potential_labels.extend(individual_words)
            
            # Clean and filter the labels - PRESERVE multi-word labels
            for label in potential_labels:
                cleaned_label = label.strip('.,;()[]{}"\' \n\t')
                if (cleaned_label and 
                    len(cleaned_label) > 1 and 
                    len(cleaned_label) < 25 and  # Increased length for multi-word
                    cleaned_label.lower() not in ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'a', 'an', 'is', 'are', 'was', 'were']):
                    suggestions.append(cleaned_label)
        
        # Remove duplicates while preserving order (case-insensitive)
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if suggestion_lower not in seen:
                seen.add(suggestion_lower)
                unique_suggestions.append(suggestion)
        
        # Limit to 10 suggestions and create dataset samples
        final_suggestions = unique_suggestions[:10]
        
        # Filter against current labels if provided
        if current_labels:
            final_suggestions = filter_suggestions_against_current_labels(final_suggestions, current_labels)
        
        # Store suggestions for this slice if slice_index is provided
        if slice_index is not None:
            store_suggested_labels_for_slice(slice_index, final_suggestions)
        
        return create_labels_dataset_samples(final_suggestions)
        
    except Exception as e:
        logger.error(f"Error generating MedGemma label suggestions: {e}")
        return []

def create_other_vlm_label_suggestions(vlm_model, image_annotator_value, slice_index=None, current_labels=None):
    """Generate label suggestions using other VLM models and return as dataset samples"""
    try:
        if current_labels is None:
            current_labels = []
            
        suggestions = []
        
        if vlm_model == "SmolVLM":
            from .smolvlm_handlers import SmolVLMHandlers
            from .state import AppState
            temp_state = AppState()
            handler = SmolVLMHandlers(temp_state)
            # SmolVLM doesn't have specific label suggestion method, use general inference
            response = handler.run_vlm_inference(image_annotator_value, True, False)
            
        elif vlm_model == "Med-R1":
            from .med_r1_handlers import MedR1Handlers
            from .state import AppState
            temp_state = AppState()
            handler = MedR1Handlers(temp_state)
            response = handler.suggest_labels_for_annotations(image_annotator_value)
            
        else:
            return []
        
        # Parse response for potential labels
        if response:
            # Look for common medical terms and anatomical structures (including multi-word)
            medical_terms = [
                'eye', 'tumor', 'lesion', 'ventricle', 'lvent', 'rvent', 'tvent', 
                'cortex', 'cerebellum', 'brainstem', 'hippocampus', 'thalamus',
                'skull', 'csf', 'white matter', 'gray matter', 'edema', 'hemorrhage',
                'frontal lobe', 'parietal lobe', 'temporal lobe', 'occipital lobe',
                'left ventricle', 'right ventricle', 'third ventricle', 'fourth ventricle'
            ]
            
            response_lower = response.lower()
            # Check for multi-word terms first (longer matches take priority)
            for term in sorted(medical_terms, key=len, reverse=True):
                if term in response_lower and term not in suggestions:
                    suggestions.append(term)
                    
            # Also try to extract labels from the response text more intelligently
            # Split by common delimiters and clean up - PRESERVE multi-word labels
            potential_labels = []
            for delimiter in [',', ';', '\n', '\t']:
                if delimiter in response:
                    potential_labels.extend([label.strip() for label in response.split(delimiter)])
                    break
            else:
                # If no delimiters found, try to extract phrases instead of individual words
                import re
                # Match common medical phrase patterns (2-3 words)
                phrase_patterns = [
                    r'\b(?:white|gray|grey)\s+matter\b',
                    r'\b(?:left|right)\s+\w+\b',
                    r'\b\w+\s+(?:tumor|lesion|mass|cyst)\b',
                    r'\b\w+\s+(?:ventricle|cortex|lobe)\b',
                ]
                
                for pattern in phrase_patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE)
                    potential_labels.extend(matches)
                
                # Also extract individual words as fallback
                individual_words = response.split()
                potential_labels.extend(individual_words)
            
            # Clean and filter additional labels - PRESERVE multi-word labels
            for label in potential_labels:
                cleaned_label = label.strip('.,;()[]{}"\' \n\t')
                if (cleaned_label and 
                    len(cleaned_label) > 2 and 
                    len(cleaned_label) < 25 and  # Increased length for multi-word
                    cleaned_label.lower() not in ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'a', 'an', 'is', 'are', 'was', 'were'] and
                    cleaned_label not in suggestions):
                    suggestions.append(cleaned_label)
        
        # Remove duplicates while preserving order (case-insensitive)
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            if suggestion_lower not in seen:
                seen.add(suggestion_lower)
                unique_suggestions.append(suggestion)
        
        # Limit to 8 suggestions and create dataset samples
        final_suggestions = unique_suggestions[:8]
        
        # Filter against current labels if provided
        if current_labels:
            final_suggestions = filter_suggestions_against_current_labels(final_suggestions, current_labels)
        
        # Store suggestions for this slice if slice_index is provided
        if slice_index is not None:
            store_suggested_labels_for_slice(slice_index, final_suggestions)
        
        return create_labels_dataset_samples(final_suggestions)
        
    except Exception as e:
        logger.error(f"Error generating {vlm_model} label suggestions: {e}")
        return []

def create_editor_tab() -> dict:
    """Create the complete editor tab layout"""
    with gr.TabItem("Editor", id=1):
        # Welcome Guide for Expert Users - Using Accordion as Modal Alternative
        with gr.Accordion("Crowdsourcing", open=False, visible=False) as welcome_guide:
            welcome_guide_content = gr.HTML("""
            <div style='padding: 15px; background: #1f2937; border-radius: 8px; color: white;'>
                <div style='background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 20px; border-radius: 8px; margin-bottom: 15px; text-align: center;'>
                    <h2 style='color: white; margin: 0 0 8px 0; font-size: 20px; font-weight: bold;'>
                        🎯 Ready to Start Annotating!
                    </h2>
                    <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;'>
                        Your dataset has been loaded and annotation tools are active
                    </p>
                </div>
                
                <div style='background: #374151; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #10b981;'>
                    <h3 style='color: #10b981; margin: 0 0 8px 0; font-size: 16px;'>📊 Current Assignment</h3>
                    <div style='color: #d1d5db; font-size: 14px; line-height: 1.4;'>
                        <p style='margin: 4px 0;'><strong>Campaign:</strong> Loading...</p>
                        <p style='margin: 4px 0;'><strong>Patient:</strong> Loading...</p>
                        <p style='margin: 4px 0;'><strong>Modality:</strong> Loading...</p>
                        <p style='margin: 4px 0;'><strong>Dataset:</strong> Loading...</p>
                    </div>
                </div>
                
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 15px;'>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #3b82f6;'>
                        <h4 style='color: #3b82f6; margin: 0 0 6px 0; font-size: 14px;'>🖱️ Annotate</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Click and drag to create annotations
                        </p>
                    </div>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #10b981;'>
                        <h4 style='color: #10b981; margin: 0 0 6px 0; font-size: 14px;'>🔍 Navigate</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Use slider or Prev/Next buttons
                        </p>
                    </div>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #f59e0b;'>
                        <h4 style='color: #f59e0b; margin: 0 0 6px 0; font-size: 14px;'>🤖 AI Help</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Use VLM Tools for suggestions
                        </p>
                    </div>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin: 0 0 6px 0; font-size: 14px;'>💾 Submit</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Scroll down to submit work
                        </p>
                    </div>
                </div>
                
                <div style='background: #065f46; padding: 12px; border-radius: 6px; text-align: center;'>
                    <p style='margin: 0; color: #d1fae5; font-size: 13px; font-weight: 500;'>
                        ✨ <strong>Pro Tip:</strong> Annotations are saved automatically as you work
                    </p>
                </div>
            </div>
            """)
            
            # Close button
            with gr.Row():
                welcome_guide_close = gr.Button("🚀 Got it! Start Annotating", variant="primary", size="sm")
        
        with gr.Row():
            # Column 1: All components (data loading, controls, etc.) except Metadata, Status/Errors, and Image Adjustments moved as specified
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Load File (DICOM, NIFTI, MAT)",
                    file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
                )
                dir_input = gr.Textbox(
                    label="Enter directory path containing DICOM files",
                    value=r"C:\Users\Yurtsever\Downloads\segpro-med\cvm_48_t1"
                )
                load_btn = gr.Button("Load Data")
                reset_dir_btn = gr.Button("Reset Directory")
                file_browser = gr.Dropdown(label="Available Files", choices=[], interactive=True)
                
                # Label File Loader
                label_file = gr.File(
                    label="Load Label File (.label)", 
                    file_types=[".label"]
                )
                
                # Image Adjustments moved here
                with gr.Accordion("Image Adjustments", open=False):
                    window_level = gr.Slider(
                        minimum=0, maximum=4000, value=500, step=10,
                        label="Window Level (Center)"
                    )
                    window_width = gr.Slider(
                        minimum=1, maximum=4000, value=1000, step=10,
                        label="Window Width"
                    )
                    apply_window_btn = gr.Button("Apply Window/Level")
                debug_btn = gr.Button("Debug Selected File")
                # For handler compatibility, include error_display and metadata_display in col1 (even if rendered in col2)
                # They will be created in col2 below
                col1 = (file_input, dir_input, load_btn, reset_dir_btn, file_browser, label_file, None, None, window_level, window_width, apply_window_btn, debug_btn)            # Column 2: View orientation and image annotator on top, then navigation controls, then Status/Errors and Metadata below
            with gr.Column(scale=4):
                # Main Visualization Area (top)
                with gr.Row():
                    view_selector = gr.Radio(
                        choices=["Axial", "Sagittal", "Coronal"],
                        value="Axial",
                        label="View Orientation"
                    )                # Image and 3D Viewer - dynamic layout based on processing mode
                with gr.Row(equal_height=True) as main_viewer_row:
                    # Image annotator column - dynamic scaling
                    image_column = gr.Column(scale=10)  # Full width initially
                    with image_column:
                        image_display = image_annotator(
                            value=None,
                            label="Medical Image", 
                            label_list=["Normal Tissue", "Tumor", "Organ", "Lesion", "ROI", "Other"],
                            label_colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
                            box_min_size=10,
                            handle_size=8,
                            box_thickness=2,
                            box_selected_thickness=3,
                            boxes_alpha=0.7,
                            height=600,                            width=1200,  # Maximum width initially - will be full screen
                            interactive=True,
                            show_label=True,
                            show_download_button=True,
                            show_clear_button=True,
                            show_remove_button=False,
                            use_default_label=False, # Do not use default label, opens modal for custom labels
                            handles_cursor=True,  # Enable cursor handling for drag mode
                            image_type="numpy",  # Important for medical images
                            single_box=False,  # Allow multiple boxes
                            disable_edit_boxes=False,  # Allow editing boxes
                            shape_creation_mode="drag",
                        )
                    
                    # 3D Viewer column - initially hidden
                    viewer_3d_column = gr.Column(scale=6, visible=False)
                    with viewer_3d_column:
                        # 3D Viewer - initially hidden, shown when "All Records" is selected and annotations exist
                        viewer_3d = gr.Plot(
                            value=create_empty_3d_plot("3D Viewer will appear here after 'All Records' annotation"),
                            label="3D Viewer",
                            visible=True  # Visible within its column
                        )
                        
                        # 3D Viewer Controls
                        with gr.Row() as viewer_3d_controls:
                            refresh_3d_btn = gr.Button("🔄 Refresh 3D View", size="sm")
                            export_3d_btn = gr.Button("💾 Export 3D", size="sm")# Navigation controls (middle)
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        label="Slice Navigation", visible=True
                    )
                    next_btn = gr.Button("Next")
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False, visible=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=True, visible=False)
                
                # Label Management Section
                with gr.Accordion("Label Management using VLMs", open=True):
                    # Current Labels Section
                    # gr.Markdown("**Current Labels**")
                    with gr.Row():
                        current_labels_dataset = gr.Dataset(
                            label="Current Labels",
                            components=[gr.Text(visible=False)],  # Simple text component for displaying labels
                            samples=[],
                            type="index",
                            samples_per_page=10
                        )
                    
                    # Suggested Labels Section
                    gr.Markdown("**Suggested Labels**")
                    
                    with gr.Row():
                        suggested_labels_dataset = gr.Dataset(
                            label="Suggested Labels",
                            components=[gr.Text(visible=False)],  # Simple text component for displaying labels
                            samples=[],
                            type="index",
                            samples_per_page=10
                        )  
                    
                    # Accept suggestions button - initially hidden
                    with gr.Row():
                        accept_suggestions_btn = gr.Button(
                            "✅ Accept Selected Suggestions",
                            variant="primary",
                            size="lg",
                            visible=False,  # Initially hidden until suggestions are selected
                            scale=1
                        )
                        
                    with gr.Row():
                        suggested_vlm_selector = gr.Dropdown(
                            choices=["SmolVLM", "Med-R1", "MedGemma-4B"],
                            value="MedGemma-4B",  # Default selected value
                            label="Select VLM to Suggest Labels",
                            scale=1,
                            container=False  # Remove extra container padding
                        )
                        suggest_labels_btn = gr.Button(
                            "🏷️ Suggest Labels", 
                            variant="primary", 
                            size="lg",  # Keep small size for better height matching
                            scale=1,  # Reduced scale to make button narrower
                            min_width=120  # Set minimum width to prevent button from being too narrow
                        )                       
                        
                # VLM (Visual Language Model) Tools Section
                with gr.Accordion("VLM Tools", open=False):
                    with gr.Row(elem_classes="vlm-row"):
                        vlm_model_selector = gr.Dropdown(
                            choices=["SmolVLM", "Med-R1", "MedGemma-4B"],
                            value="MedGemma-4B",  # Default selected value
                            label="VLM Model",
                            scale=1
                        )
                        with gr.Row(elem_classes="vlm-options-row"):
                            vlm_prompt_anomalies = gr.Checkbox(
                                label="Identify Anomalies",
                                value=True,  # Default selected
                                info="Focus on identifying abnormal regions",
                                scale=2
                            )
                            vlm_prompt_describe = gr.Checkbox(
                                label="Describe MRI Slice", 
                                value=False,
                                info="General description of anatomical structures",
                                scale=2
                            )
                        vlm_run_btn = gr.Button("🔍 Get Medical Analysis", variant="primary", size="lg", scale=1)
                        
                    
                    
                    # VLM Label Suggestion for Annotations
                    with gr.Row(elem_classes="vlm-row"):
                        vlm_suggest_labels_btn = gr.Button(                        "🏷️ Suggest Labels (VLM)", 
                            variant="secondary", 
                            size="lg", 
                            scale=2,
                            #info="Analyze annotations and suggest semantic labels"
                            visible=False
                        )
                        gr.HTML("<div style='flex: 1;'></div>")  # Spacer to maintain layout
                        
                    with gr.Row():
                        vlm_caption = gr.Textbox(
                            label="VLM Analysis",
                            interactive=False,
                            placeholder="Click 'Get Medical Analysis' or 'Suggest Labels (VLM)' to generate analysis...",
                            scale=1,
                            elem_classes="vlm-caption-text",
                            lines=4
                        )
                
                # VLM Prompt Selection
                #with gr.Row():
                    
                
                # Status/Errors and Metadata (bottom)
                error_display = gr.Textbox(label="Status/Errors", interactive=False)
                with gr.Accordion("Metadata", open=False):
                    metadata_display = gr.JSON(label=None, visible=True)
                
                col2 = (error_display, metadata_display, view_selector, image_display, image_column, viewer_3d_column, prev_btn, slice_slider, next_btn, slice_text, crosshair_info, current_labels_dataset, suggested_vlm_selector, suggest_labels_btn, suggested_labels_dataset, accept_suggestions_btn, vlm_model_selector, vlm_run_btn, vlm_suggest_labels_btn, vlm_caption, vlm_prompt_anomalies, vlm_prompt_describe, viewer_3d, viewer_3d_controls, refresh_3d_btn, export_3d_btn)
            
            # Column 3: Annotate with AI Models
            with gr.Column(scale=1):
                # Coordinate Selection for MEDSAM2
                gr.Markdown("### Point Selection")
                # Prompt type selection
                with gr.Row():
                    point_prompt_checkbox = gr.Checkbox(
                        label="Point-based Prompt",
                        value=False,
                        info="Use point coordinates for segmentation"
                    )
                    box_prompt_checkbox = gr.Checkbox(
                        label="Box-based Prompt", 
                        value=False,
                        info="Use bounding boxes for segmentation"
                    )
                coordinates_text = gr.Textbox(
                    label="Selected Coordinates (x,y)",
                    value="",
                    interactive=False,
                    info="Click on the image to select coordinates",
                    visible=False  # Initially hidden until point-based is selected
                )
                clear_coords_btn = gr.Button("Clear Coordinates & Prompts", variant="stop",)
                
                gr.Markdown("## Annotate with AI Models")
                ai_model_selector = gr.Dropdown(
                    label="Select AI Model",
                    choices=["MEDSAM2", "UNet (dummy)", "DeepLabV3 (dummy)", "SAM (dummy)", "Other (dummy)"],
                    value="MEDSAM2"
                )
                
                # Processing Mode Selection
                processing_mode = gr.Radio(
                    choices=["Single Slice", "All Records"],
                    value="Single Slice",
                    label="Processing Mode",
                    info="Single Slice: Process only current slice. All Records: Process entire volume with quality thresholding."
                )
                
                # Score Threshold for All Records mode
                score_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="Score Threshold (All Records mode)",
                    info="Minimum average score required to display annotations. Lower scores may indicate poor quality segmentations.",
                    visible=False  # Initially hidden, shown when "All Records" is selected
                )
                
                # MEDSAM2 specific controls
                with gr.Group(visible=True) as medsam2_controls:
                    with gr.Accordion("Annotation Settings", open=False):
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            value="brain_target_results",
                            info="Directory to save annotation results"
                        )
                        save_visualizations = gr.Checkbox(
                            label="Save Visualizations",
                            value=False,
                            info="Save visualization images along with segmentation"
                        )
                        device_selector = gr.Dropdown(
                            label="Device",
                            choices=["cpu", "cuda"],
                            value="cpu",
                            info="Processing device for MEDSAM2"
                        )                  # SAM2 Fast Masking Section
                gr.Markdown("### SAM2 Fast Masking")
                with gr.Row():
                    auto_brain_annotate_btn = gr.Button(
                        "🚀 Run SAM2 Fast Masking", 
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Accordion("SAM2 Fast Masking Info", open=False):
                    gr.Markdown("""
                    **SAM2 Fast Masking Pipeline:**
                    - Uses optimized 'fast' configuration for quick processing
                    - Automatically segments brain structures using SAM2AutomaticMaskGenerator
                    - Filters masks with area ≥ 500 pixels and IoU ≥ 0.8
                    - Respects "Save Visualizations" checkbox setting
                    - Works with both "Single Slice" and "All Records" modes
                    - Generates JSON mask files for each processed slice
                    - No comprehensive summary files for optimal speed
                    """)
                
                annotate_btn = gr.Button("Run MEDSAM2 Annotation (Manual Prompts)")
                annotation_status = gr.Textbox(
                    label="Annotation Status", 
                    interactive=False,
                    value="Ready to annotate"
                )
                
                # Crowdsourcing Controls (hidden by default, shown when in crowdsourcing mode)
                with gr.Accordion("Crowdsourcing Controls", open=True, visible=False) as crowdsourcing_accordion:
                    gr.Markdown("**Annotation Task Controls**")
                    gr.Markdown("Complete your annotation work and submit using the controls below.")
                    
                    with gr.Row():
                        submit_annotation_btn = gr.Button(
                            "✅ Submit Annotation",
                            variant="primary",
                            size="lg"
                        )
                        send_for_review_btn = gr.Button(
                            "📤 Send for Review",
                            variant="secondary",
                            size="lg"
                        )
                    
                    crowdsourcing_status = gr.Textbox(
                        label="Submission Status",
                        interactive=False,
                        value="Complete your annotation work above, then submit."
                    )
                    
                    gr.Markdown("""
                    **Instructions:**
                    1. Use the annotation tools above to complete your work
                    2. Review your annotations carefully
                    3. Click 'Submit Annotation' when finished
                    4. Use 'Send for Review' if you need supervisor feedback
                    """)
                
                col3 = (point_prompt_checkbox, box_prompt_checkbox, coordinates_text, clear_coords_btn, ai_model_selector, 
                        processing_mode, score_threshold,
                        output_dir, save_visualizations, device_selector, 
                        auto_brain_annotate_btn, annotate_btn, annotation_status,
                        crowdsourcing_accordion, submit_annotation_btn, send_for_review_btn, crowdsourcing_status)    # Helper functions for 3D viewer interactions and layout switching
    def toggle_layout_for_processing_mode(processing_mode):
        """Switch between full-width and split layout based on processing mode"""
        if processing_mode == "All Records":
            # Split layout: image scale 7, 3D viewer visible with scale 6
            return (
                gr.update(scale=7),  # image_column
                gr.update(scale=6, visible=True),  # viewer_3d_column
                gr.update(width=650),  # image_display - reduced width for split layout
            )
        else:
            # Full layout: image full width, 3D viewer hidden
            return (
                gr.update(scale=10),  # image_column - full width
                gr.update(scale=6, visible=False),  # viewer_3d_column - hidden
                gr.update(width=1200),  # image_display - maximum width for full layout
            )
    
    def reset_layout_and_clear():
        """Reset to full layout and clear everything"""
        return (            gr.update(scale=10),  # image_column - full width
            gr.update(scale=6, visible=False),  # viewer_3d_column - hidden  
            gr.update(width=1200),  # image_display - maximum width for initial state
            create_empty_3d_plot("3D Viewer will appear here after 'All Records' annotation"),  # Reset 3D viewer
            "",  # Clear coordinates
            "Single Slice"  # Reset processing mode to valid choice
        )
    
    def toggle_3d_viewer_visibility(processing_mode):
        """Show/hide 3D viewer based on processing mode - legacy function"""
        if processing_mode == "All Records":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    def refresh_3d_view(output_dir, score_threshold=None):
        """Refresh the 3D visualization"""
        try:
            fig = create_3d_visualization(output_dir, score_threshold)
            return fig
        except Exception as e:
            return create_empty_3d_plot(f"Error refreshing view: {str(e)}")
    
    def export_3d_view(output_dir, score_threshold=None):
        """Export 3D view as HTML"""
        try:
            fig = create_3d_visualization(output_dir, score_threshold)
            export_path = os.path.join(output_dir, "3d_visualization.html")
            fig.write_html(export_path)
            return f"3D view exported to: {export_path}"
        except Exception as e:
            return f"Error exporting 3D view: {str(e)}"
    
    # Return all components in a structured way
    return {
        'data_loading': col1,
        'visualization': col2,
        'ai_tools': col3,
        'crowdsourcing': {
            'accordion': crowdsourcing_accordion,
            'submit_btn': submit_annotation_btn,
            'review_btn': send_for_review_btn,
            'status': crowdsourcing_status
        },
        'welcome_modal': {
            'guide': welcome_guide,
            'content': welcome_guide_content,
            'close_btn': welcome_guide_close
        },
        '3d_viewer_functions': {
            'toggle_visibility': toggle_3d_viewer_visibility,
            'refresh_view': refresh_3d_view,
            'export_view': export_3d_view
        },
        'layout_functions': {
            'toggle_layout': toggle_layout_for_processing_mode,
            'reset_layout': reset_layout_and_clear
        }
    }
