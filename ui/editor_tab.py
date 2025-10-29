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
from .modal_components import create_editor_welcoming_modal_system, create_segmentation_modal_system
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

def detect_mg_orientation_from_metadata(metadata):
    """
    Detect mammography orientation from DICOM metadata
    Returns one of: 'LCC', 'LMLO', 'RCC', 'RMLO', or None
    """
    if not metadata:
        return None
    
    # Check if modality is MG (Mammography)
    modality = metadata.get('Modality', '')
    if modality != 'MG':
        return None
    
    # Try ViewPosition tag (0018,5101) - most direct way
    if 'ViewPosition' in metadata:
        view_position = str(metadata['ViewPosition']).upper()
        # ViewPosition can be: CC (Cranio-Caudal), MLO (Medio-Lateral Oblique)
        # Combined with ImageLaterality
        laterality = metadata.get('ImageLaterality', '').upper()
        if laterality in ['L', 'R'] and view_position in ['CC', 'MLO']:
            return f"{laterality}{view_position}"
    
    # Try ImageLaterality (0020,0062) + ViewCodeSequence
    laterality = metadata.get('ImageLaterality', '').upper()
    if 'ViewCodeSequence' in metadata:
        try:
            view_code = metadata['ViewCodeSequence']
            # This is more complex, might need specific code values
            if hasattr(view_code, 'CodeMeaning'):
                view_meaning = str(view_code.CodeMeaning).upper()
                if 'CC' in view_meaning and laterality:
                    return f"{laterality}CC"
                elif 'MLO' in view_meaning and laterality:
                    return f"{laterality}MLO"
        except:
            pass
    
    # Try SeriesDescription or StudyDescription as fallback
    for desc_field in ['SeriesDescription', 'StudyDescription']:
        if desc_field in metadata:
            desc = str(metadata[desc_field]).upper()
            # Look for patterns like "L CC", "R MLO", "LCC", "RMLO", etc.
            for orientation in ['LCC', 'LMLO', 'RCC', 'RMLO']:
                if orientation in desc.replace(' ', ''):
                    return orientation
    
    return None

def detect_mg_orientation_from_filename(filename):
    """
    Detect mammography orientation from filename
    Returns one of: 'LCC', 'LMLO', 'RCC', 'RMLO', or None
    """
    if not filename:
        return None
    
    # Convert to uppercase and remove extension
    name = os.path.splitext(os.path.basename(filename))[0].upper()
    
    # Check for exact matches or patterns
    for orientation in ['LCC', 'LMLO', 'RCC', 'RMLO']:
        if orientation in name:
            return orientation
    
    return None

def get_mg_orientations_from_directory(file_list):
    """
    Get available mammography orientations from a list of files
    Returns dict mapping orientation -> file_path
    """
    orientation_map = {}
    
    if not file_list:
        return orientation_map
    
    for file_path in file_list:
        # Try filename first (faster)
        orientation = detect_mg_orientation_from_filename(file_path)
        
        if orientation:
            orientation_map[orientation] = file_path
            logger.info(f"Detected {orientation} from filename: {os.path.basename(file_path)}")
    
    return orientation_map

def update_view_selector_for_modality(metadata, file_list=None):
    """
    Update view selector choices based on modality
    Returns: (choices, value, visible) for gr.Radio update
    """
    if not metadata:
        return ["Axial", "Sagittal", "Coronal"], "Axial", True
    
    modality = metadata.get('Modality', '')
    
    # Mammography (MG) - use specific orientations
    if modality == 'MG':
        # Check if we have files to detect orientations from
        if file_list:
            orientation_map = get_mg_orientations_from_directory(file_list)
            available_orientations = list(orientation_map.keys())
            
            if available_orientations:
                # Sort in standard order: LCC, LMLO, RCC, RMLO
                standard_order = ['LCC', 'LMLO', 'RCC', 'RMLO']
                sorted_orientations = [o for o in standard_order if o in available_orientations]
                
                logger.info(f"Mammography detected with orientations: {sorted_orientations}")
                return sorted_orientations, sorted_orientations[0], True
        
        # Fallback: provide all MG orientations even if not detected
        logger.info("Mammography detected - using default MG orientations")
        return ['LCC', 'LMLO', 'RCC', 'RMLO'], 'LCC', True
    
    # For other modalities (CT, MR, etc.) - use standard anatomical views
    else:
        return ["Axial", "Sagittal", "Coronal"], "Axial", True

def get_file_for_mg_orientation(orientation, file_list, current_data_directory=None):
    """
    Get the file path for a specific MG orientation
    Returns: file_path or None
    """
    if not file_list:
        return None
    
    # Build orientation map from file list
    orientation_map = get_mg_orientations_from_directory(file_list)
    
    # Return the file for the requested orientation
    file_path = orientation_map.get(orientation)
    
    if file_path:
        logger.info(f"Selected file for {orientation}: {os.path.basename(file_path)}")
    else:
        logger.warning(f"No file found for orientation: {orientation}")
    
    return file_path


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

def save_vlm_analysis_to_file(data_directory, slice_index, analysis_text):
    """Save VLM analysis to a vlm_analysis.txt file in the data directory - REPLACE analysis for the slice"""
    try:
        analysis_file_path = os.path.join(data_directory, "vlm_analysis.txt")
        
        # Read existing analyses if file exists
        existing_analyses = {}
        if os.path.exists(analysis_file_path):
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                current_slice = None
                current_analysis = []
                for line in f:
                    line = line.rstrip('\n')
                    if line.startswith('Slice ') and ':' in line:
                        # Save previous slice analysis if exists
                        if current_slice is not None and current_analysis:
                            existing_analyses[current_slice] = '\n'.join(current_analysis)
                        # Start new slice
                        current_slice = int(line.split(':')[0].replace('Slice ', ''))
                        current_analysis = []
                    elif current_slice is not None:
                        current_analysis.append(line)
                # Save last slice
                if current_slice is not None and current_analysis:
                    existing_analyses[current_slice] = '\n'.join(current_analysis)
        
        # REPLACE analysis for the current slice (don't append)
        clean_analysis = analysis_text.strip() if analysis_text else ""
        if clean_analysis:
            existing_analyses[slice_index] = clean_analysis
        elif slice_index in existing_analyses:
            # Remove empty analysis
            del existing_analyses[slice_index]
        
        # Write updated analyses back to file with proper formatting
        with open(analysis_file_path, 'w', encoding='utf-8') as f:
            for slice_num in sorted(existing_analyses.keys()):
                f.write(f"Slice {slice_num}:\n")
                f.write(f"{existing_analyses[slice_num]}\n\n")
        
        logger.info(f"Saved VLM analysis for slice {slice_index} (length: {len(clean_analysis)} chars)")
        return f"VLM analysis saved for slice {slice_index}"
        
    except Exception as e:
        logger.error(f"Error saving VLM analysis to file: {e}")
        return f"Error saving VLM analysis: {str(e)}"

def load_vlm_analysis_from_file(data_directory, slice_index):
    """Load VLM analysis for a specific slice from vlm_analysis.txt file"""
    try:
        analysis_file_path = os.path.join(data_directory, "vlm_analysis.txt")
        
        if not os.path.exists(analysis_file_path):
            return ""
        
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            current_slice = None
            current_analysis = []
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('Slice ') and ':' in line:
                    # Check if we found the target slice analysis
                    if current_slice == slice_index and current_analysis:
                        return '\n'.join(current_analysis).strip()
                    # Start new slice
                    current_slice = int(line.split(':')[0].replace('Slice ', ''))
                    current_analysis = []
                elif current_slice is not None:
                    current_analysis.append(line)
            
            # Check last slice
            if current_slice == slice_index and current_analysis:
                return '\n'.join(current_analysis).strip()
        
        return ""  # No analysis found for this slice
        
    except Exception as e:
        logger.error(f"Error loading VLM analysis from file: {e}")
        return ""

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

def create_editor_tab(current_user=None) -> dict:
    """
    Create the complete editor tab layout
    
    Args:
        current_user: Dictionary containing user info (with 'user_id' key), or None for guest
    """
    # Extract user_id from current_user, default to "guest"
    user_id = current_user.get('user_id', 'guest') if current_user else 'guest'
    
    with gr.TabItem("Editor", id=1):
        # Create modal system for welcome and help dialogs (auto-opening)
        modal_system = create_editor_welcoming_modal_system(user_id=user_id)
        
        # Create segmentation completion modal system
        segmentation_modal_system = create_segmentation_modal_system(user_id=user_id)
        
        # Create prompt tip modal systems
        from ui.modal_components import create_prompt_tip_modal_system
        point_tip_modal_system = create_prompt_tip_modal_system(user_id=user_id, modal_type="point")
        box_tip_modal_system = create_prompt_tip_modal_system(user_id=user_id, modal_type="box")
        
        # Note: Modal will auto-open when tab loads - no manual trigger needed
        
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
                
                # Import the new info tooltip system
                from ui.info_tooltips import create_load_medical_data_header
                
                # Create enhanced header with info tooltip (single component)
                create_load_medical_data_header()
                
                file_input = gr.File(
                    label="Load File (DICOM, NIFTI, MAT)",
                    file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
                )
                dir_input = gr.Textbox(
                    label="Enter directory path containing DICOM files",
                    value=r"/home/enesgazi/Downloads/segpro-med/cvm_48_t1"
                )
                load_btn = gr.Button("Load Data")
                reset_dir_btn = gr.Button("Reset Directory")
                file_browser = gr.Dropdown(label="Available Files", choices=[], interactive=True)
                
                # Load Label File section with tooltip
                from ui.info_tooltips import create_load_label_file_header
                create_load_label_file_header()
                
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
                        label="View Orientation",
                        info="Orientation changes based on modality"
                    )
                    deidentification_checkbox = gr.Checkbox(
                        label="De Identification",
                        value=False,
                        info="Remove faces from DICOM images using pydeface", visible=False
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
                            auto_scroll=True,
                            scroll_threshold=1.0,  # Enable scrolling when image exceeds container
                            preserve_resolution=True,  # Maintain original image quality
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
                            export_3d_btn = gr.Button("💾 Export 3D", size="sm")
                
                # AI Guided Annotation Accordion (initially hidden, shown with yellow highlight when activated)
                with gr.Accordion("AI-Guided Annotation", open=True, visible=False, elem_id="ai-tools-accordion") as ai_tools_accordion:
                    gr.HTML("""
                    <style>
                        #ai-tools-accordion {
                            background: linear-gradient(135deg, rgba(254, 240, 138, 0.15) 0%, rgba(253, 224, 71, 0.15) 100%) !important;
                            border: 1px solid rgba(234, 179, 8, 0.4) !important;
                            border-radius: 8px !important;
                            padding: 8px !important;
                            margin: 8px 0 !important;
                            animation: highlightPulse 2s ease-in-out !important;
                        }
                        
                        @keyframes highlightPulse {
                            0%, 100% { 
                                background: linear-gradient(135deg, rgba(254, 240, 138, 0.15) 0%, rgba(253, 224, 71, 0.15) 100%);
                            }
                            50% { 
                                background: linear-gradient(135deg, rgba(254, 240, 138, 0.25) 0%, rgba(253, 224, 71, 0.25) 100%);
                            }
                        }
                        
                        #ai-tools-accordion .label-wrap {
                            color: #ca8a04 !important;
                            font-weight: 600 !important;
                        }
                        
                        .compact-tip {
                            font-style: italic !important;
                            color: #9ca3af !important;
                            font-size: 0.9rem !important;
                            padding: 6px 0 !important;
                            margin: 4px 0 !important;
                        }
                    </style>
                    """)
                    
                    # Everything in a single compact row with columns
                    with gr.Row():
                        # Column 1: Quick Tips (italic text-only, initially hidden)
                        with gr.Column(scale=5):
                            point_info_accordion = gr.Markdown(
                                "*💡 Point-based: Click directly on the image at the location you want to segment. Best results when clicking the center of the structure.*",
                                visible=False,
                                elem_classes=["compact-tip"]
                            )
                            
                            box_info_accordion = gr.Markdown(
                                "*📦 Box-based: Draw a bounding box around the area using the rectangle tool from the toolbar. Make sure the box fully contains the structure with some margin.*",
                                visible=False,
                                elem_classes=["compact-tip"]
                            )
                        
                        # Column 2: Run Button
                        with gr.Column(scale=2):
                            annotate_btn = gr.Button("Run Guided Annotation", variant="primary")
                        
                        # Column 3: Clear Button
                        with gr.Column(scale=2):
                            clear_coords_btn = gr.Button("Clear Coordinates & Prompts", variant="stop")
                    
                    # Hidden coordinates textbox (kept for functionality but not displayed)
                    with gr.Row(visible=False):
                        coordinates_text = gr.Textbox(
                            label="Selected Coordinates (x,y)",
                            value="",
                            interactive=False
                        )
                
                # Navigation controls (middle)
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=1, value=0, step=1,
                        label="Slice Navigation", visible=False
                    )
                    next_btn = gr.Button("Next")
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False, visible=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=True, visible=False)
                
                # Crowdsourcing Controls (moved up and enhanced)
                with gr.Accordion("Crowdsourcing Controls", open=True, visible=False) as crowdsourcing_accordion:
                    #gr.Markdown("**Annotation Task Controls**")
                    
                    crowdsourcing_status = gr.HTML(
                        value="""
                        <div style='padding: 16px; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 12px; border: 1px solid #d1d5db; margin: 8px 0;'>
                            <div style='display: flex; align-items: center; gap: 12px;'>
                                <div style='width: 12px; height: 12px; background: #6b7280; border-radius: 50%; flex-shrink: 0;'></div>
                                <div style='color: #374151; font-weight: 500; font-size: 14px;'>
                                    Complete your annotation work and submit using the controls below.
                                </div>
                            </div>
                        </div>
                        """
                    )
                    
                    with gr.Row():
                        submit_annotation_btn = gr.Button(
                            "✅ Submit Annotation",
                            variant="primary",
                            size="lg"
                        )
                    
                    # Assignment progress and next task controls with Next.js style
                    with gr.Row():
                        assignments_remaining = gr.HTML(
                            value="",
                            visible=False
                        )
                    
                    with gr.Row():
                        next_assignment_btn = gr.Button(
                            "➡️ Load Next Assignment",
                            variant="primary",
                            size="lg",
                            visible=False
                        )
                    
                    # Instructions with Next.js styling
                    gr.HTML("""
                    <div style='padding: 16px; background: linear-gradient(135deg, #374151 0%, #4b5563 100%); border-radius: 12px; border: 1px solid #6b7280; margin: 8px 0;'>
                        <div style='display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px;'>
                            <div style='width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; margin-top: 4px;'></div>
                            <div style='color: #f9fafb; font-weight: 600; font-size: 14px;'>
                                Instructions:
                            </div>
                        </div>
                        <div style='color: #d1d5db; font-size: 13px; line-height: 1.6; margin-left: 24px;'>
                            <div style='margin-bottom: 6px;'><strong>1.</strong> Use the annotation tools above to complete your work</div>
                            <div style='margin-bottom: 6px;'><strong>2.</strong> Review your annotations carefully</div>
                            <div style='margin-bottom: 6px;'><strong>3.</strong> Click 'Submit Annotation' when finished</div>
                            <div style='margin-bottom: 0;'><strong>4.</strong> Click 'Load Next Assignment' to continue with remaining tasks</div>
                        </div>
                    </div>
                    """)
                
                # Label Management Section (moved down and set to closed)
                with gr.Accordion("Label Management using VLMs", open=False):
                    # Info message for VLM Label Management
                    from ui.info_components import create_vlm_label_management_info
                    vlm_info_accordion = create_vlm_label_management_info()
                    
                    # Current Labels Section
                    # gr.Markdown("**Current Labels**")
                    with gr.Row():
                        current_labels_placeholder = gr.HTML("""
                        <div id="current-labels-placeholder" style='padding: 6px; background: rgba(71, 85, 105, 0.2); border-radius: 6px; border: 1px dashed rgba(148, 163, 184, 0.3); text-align: left;'>
                            <p style='margin: 0; color: #94a3b8; font-size: 13px; font-style: italic;'>
                                Manual and AI-suggested labels will appear here
                            </p>
                        </div>
                        """, visible=True, elem_id="current-labels-placeholder-html")
                    with gr.Row():
                        current_labels_dataset = gr.Dataset(
                            label="Current Labels",
                            components=[gr.Text(visible=False)],  # Simple text component for displaying labels
                            samples=[],
                            type="index",
                            samples_per_page=10,
                            visible=False  # Initially hidden until labels are loaded
                        )
                    
                    # Suggested Labels Section - with placeholder
                    with gr.Row():
                        suggested_labels_placeholder = gr.HTML("""
                        <div id="suggested-labels-placeholder" style='padding: 6px; background: rgba(71, 85, 105, 0.2); border-radius: 6px; border: 1px dashed rgba(148, 163, 184, 0.3); text-align: left;'>
                            <p style='margin: 0; color: #94a3b8; font-size: 13px; font-style: italic;'>
                                AI-generated labels will be shown here
                            </p>
                        </div>
                        """, visible=True, elem_id="suggested-labels-placeholder-html")
                    
                    # Info message for label suggestion usage (initially hidden)
                    with gr.Row(elem_classes="label-suggestion-info-row", visible=False) as label_suggestion_info_row:
                        gr.HTML("""
                        <div style='padding: 8px; background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #10b981;'>
                            <div style='color: #d1fae5; font-size: 13px; font-weight: 600; margin-bottom: 2px;'>
                                🏷️ Label Suggestions Ready
                            </div>
                            <div style='color: #a7f3d0; font-size: 11px; line-height: 1.3;'>
                                Click on the suggested labels below to select them, then click "Accept Selected Suggestions" to add them to your current labels
                            </div>
                        </div>
                        """)
                    
                    with gr.Row():
                        suggested_labels_dataset = gr.Dataset(
                            label="Suggested Labels",
                            components=[gr.Text(visible=False)],  # Simple text component for displaying labels
                            samples=[],
                            type="index",
                            samples_per_page=10,
                            visible=False  # Initially hidden until suggestions are loaded
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
                    # Info message for VLM Tools
                    from ui.info_components import create_vlm_tools_info
                    vlm_tools_info_accordion = create_vlm_tools_info()
                    
                    with gr.Row(elem_classes="vlm-row"):
                        vlm_model_selector = gr.Dropdown(
                            choices=["SmolVLM", "Med-R1", "MedGemma-4B"],
                            value="MedGemma-4B",  # Default selected value
                            label="Select VLM Model",
                            scale=1
                        )

                    # Info message for Custom Prompts
                    from ui.info_components import create_vlm_custom_prompt_info
                    vlm_custom_prompt_info_accordion = create_vlm_custom_prompt_info()
                    
                    # Voice Input Section
                    with gr.Row(elem_classes="voice-input-row"):
                        voice_audio_input = gr.Audio(
                            label="Record Voice Input",
                            sources=["microphone"],
                            type="numpy",
                            scale=3,
                            interactive=True
                        )
                        voice_prompt_text = gr.Textbox(
                            label="Voice Prompt",
                            placeholder="Record your voice above or type here...",
                            scale=4,
                            lines=1,
                            interactive=True
                        )
                                    
                    with gr.Row(elem_classes="vlm-options-row"):
                            vlm_prompt_anomalies = gr.Checkbox(
                                label="Identify Anomalies",
                                value=True,  # Default selected
                                info="Focus on identifying abnormal regions",
                                scale=2
                            )
                            vlm_prompt_describe = gr.Checkbox(
                                label="Describe the Image", 
                                value=False,
                                info="General description of anatomical structures",
                                scale=2
                            )
                    
                    with gr.Row():
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
                    
                    # Voice Analysis Addition Section (appears after VLM analysis is complete)
                    with gr.Row(elem_classes="voice-analysis-row", visible=False) as voice_analysis_row:
                        gr.HTML("""
                        <div style='padding: 8px; background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #10b981;'>
                            <div style='color: #d1fae5; font-size: 13px; font-weight: 600; margin-bottom: 2px;'>
                                🎙️ Voice Analysis Notes
                            </div>
                            <div style='color: #a7f3d0; font-size: 11px; line-height: 1.3;'>
                                Record additional voice analysis or type notes to append to the VLM analysis above
                            </div>
                        </div>
                        """)
                    with gr.Row(elem_classes="voice-analysis-controls", visible=False) as voice_analysis_controls:
                        voice_analysis_audio = gr.Audio(
                            label="Record Voice Analysis",
                            sources=["microphone"],
                            type="numpy",
                            scale=3,
                            interactive=True
                        )
                        voice_analysis_text = gr.Textbox(
                            label="Voice Analysis Note",
                            placeholder="Record your voice above or type additional analysis here...",
                            scale=4,
                            lines=1,
                            interactive=True
                        )
                        save_to_analysis_btn = gr.Button(
                            "Save to Analysis",
                            variant="secondary",
                            size="sm",
                            scale=1
                        )
                
                # VLM Prompt Selection
                #with gr.Row():
                    
                
                # Status/Errors and Metadata (bottom)
                error_display = gr.Textbox(label="Status/Errors", interactive=False)
                with gr.Accordion("Metadata", open=False):
                    metadata_display = gr.JSON(label=None, visible=True)
                
                col2 = (error_display, metadata_display, view_selector, deidentification_checkbox, image_display, image_column, viewer_3d_column, prev_btn, slice_slider, next_btn, slice_text, crosshair_info, 
                        crowdsourcing_accordion, submit_annotation_btn, assignments_remaining, next_assignment_btn, crowdsourcing_status,
                        current_labels_dataset, current_labels_placeholder, suggested_vlm_selector, suggest_labels_btn, suggested_labels_dataset, suggested_labels_placeholder, accept_suggestions_btn, label_suggestion_info_row, vlm_model_selector, vlm_run_btn, vlm_suggest_labels_btn, vlm_caption, vlm_prompt_anomalies, vlm_prompt_describe, viewer_3d, viewer_3d_controls, refresh_3d_btn, export_3d_btn, voice_prompt_text, voice_audio_input, voice_analysis_row, voice_analysis_controls, voice_analysis_audio, voice_analysis_text, save_to_analysis_btn, vlm_info_accordion, vlm_tools_info_accordion, vlm_custom_prompt_info_accordion)
            
            # Column 3: Annotate with AI Models
            with gr.Column(scale=1):
                
                # Import the new info tooltip system
                from ui.info_tooltips import create_segmentation_settings_header
                
                # Create enhanced header with info tooltip (single component)
                create_segmentation_settings_header()
                
                ai_model_selector = gr.Dropdown(
                    label="Select AI Model",
                    choices=["MEDSAM2", "UNet (dummy)", "DeepLabV3 (dummy)", "SAM (dummy)", "Other (dummy)"],
                    value="MEDSAM2", visible=False # Initially hidden until more models are integrated
                )
                
                # MEDSAM2 specific controls
                with gr.Group(visible=True) as medsam2_controls:
                    with gr.Accordion("Output & Device Settings", open=False):
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
                            value="cuda",
                            info="Processing device for MEDSAM2"
                        )
                
                # Processing Mode Selection
                processing_mode = gr.Radio(
                    choices=["Single Slice", "All Records"],
                    value="Single Slice",
                    label="Processing Mode for AI Annotation",
                    info="Single Slice: Process only current slice. All Records: Process entire volume with quality thresholding."
                )
                
                # Score Threshold for All Records mode
                score_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="Score Threshold (All Records mode)",
                    info="Minimum average score required to display annotations. Lower scores may indicate poor quality segmentations.",
                    visible=False  # Initially hidden, shown when "All Records" is selected
                )
                
                # Coordinate Selection for MEDSAM2
                # Import the new info tooltip system
                from ui.info_tooltips import create_segmentation_with_ai_header
                
                # Create enhanced header with info tooltip (single component)
                create_segmentation_with_ai_header()
                
                # Guided Segmentation (Point/Box Selection) - Checkboxes to trigger modals and accordion
                with gr.Accordion("Guided Segmentation (Point/Box Selection)", open=False):
                    gr.Markdown("*Select a prompt type to activate AI-guided annotation in the highlighted section above.*")
                    
                    gr.HTML("""
                    <div style='padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; border-left: 3px solid #3b82f6; margin-bottom: 12px;'>
                        <p style='margin: 0; color: #93c5fd; font-size: 14px;'>
                            ℹ️ <strong>How to use:</strong> Check one of the options below. The <strong>AI-Guided Annotation</strong> 
                            section will appear above with all controls.
                        </p>
                    </div>
                    """)
                    
                    # These checkboxes are NOT duplicates - they are the ONLY checkboxes that trigger everything
                    # The ones in the AI-Guided Annotation accordion need to be removed
                    with gr.Row():
                        point_prompt_checkbox = gr.Checkbox(
                            label="Point-based Prompt",
                            value=False,
                            info="Click on image to select points"
                        )
                        box_prompt_checkbox = gr.Checkbox(
                            label="Box-based Prompt", 
                            value=False,
                            info="Draw bounding boxes"
                        )
                
                # Whole Area Segmentation Section
                with gr.Accordion("Whole Area Segmentation", open=False):
                    # Info box for whole area segmentation
                    from ui.info_components import create_info_message
                    whole_area_info_accordion = create_info_message(
                        message="""
                        <strong>Automatic Segmentation:</strong> Automatically detects and segments all major structures in the current view without manual guidance.
                        <br><br>
                        <strong>Tip:</strong> Works best on clear MRI images. Processing may take a few moments depending on image complexity.
                        """,
                        message_type="info",
                        visible=True,
                        open_state=False
                    )
                    
                    with gr.Row():
                        auto_brain_annotate_btn = gr.Button(
                            "Run Automatic Segmentation", 
                            variant="primary",
                            size="lg"
                        )
                
                col3 = (point_prompt_checkbox, box_prompt_checkbox, coordinates_text, clear_coords_btn, ai_model_selector, 
                        processing_mode, score_threshold,
                        output_dir, save_visualizations, device_selector, 
                        auto_brain_annotate_btn, annotate_btn, point_info_accordion, box_info_accordion, whole_area_info_accordion, ai_tools_accordion)    # Helper functions for 3D viewer interactions and layout switching
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
    
    # Sample data loading functionality (connected in main app)
    def load_sample_data_and_trigger(dataset_path):
        """Load sample dataset and update directory input"""
        try:
            logger.info(f"Loading sample data from: {dataset_path}")
            # Return the directory path and close modal
            return (
                dataset_path,  # Update directory input
                gr.update(visible=False),  # Hide backdrop
                gr.update(visible=False),  # Hide modal
            )
        except Exception as e:
            logger.error(f"Error loading sample data: {e}")
            return (
                dataset_path,  # Update directory input anyway
                gr.update(visible=False),  # Hide backdrop
                gr.update(visible=False),  # Hide modal
            )
    
    # Return all components in a structured way
    return {
        'data_loading': col1,
        'visualization': col2,
        'ai_tools': col3,
        'crowdsourcing': {
            'accordion': crowdsourcing_accordion,
            'submit_btn': submit_annotation_btn,
            'status': crowdsourcing_status,
            'assignments_remaining': assignments_remaining,
            'next_assignment_btn': next_assignment_btn
        },
        'welcome_modal': {
            'guide': welcome_guide,
            'content': welcome_guide_content,
            'close_btn': welcome_guide_close
        },
        'modal_system': modal_system,
        'voice_analysis': {
            'row': voice_analysis_row,
            'controls': voice_analysis_controls,
            'audio': voice_analysis_audio,
            'text': voice_analysis_text,
            'save_btn': save_to_analysis_btn
        },
        '3d_viewer_functions': {
            'toggle_visibility': toggle_3d_viewer_visibility,
            'refresh_view': refresh_3d_view,
            'export_view': export_3d_view
        },
        'layout_functions': {
            'toggle_layout': toggle_layout_for_processing_mode,
            'reset_layout': reset_layout_and_clear
        },
        'info_components': {
            'vlm_info_accordion': vlm_info_accordion,
            'vlm_tools_info_accordion': vlm_tools_info_accordion,
            'vlm_custom_prompt_info_accordion': vlm_custom_prompt_info_accordion,
            'point_info_accordion': point_info_accordion,
            'box_info_accordion': box_info_accordion,
            'whole_area_info_accordion': whole_area_info_accordion
        },
        'sample_loading': {
            'function': load_sample_data_and_trigger,
            'buttons': {
                'cvm': modal_system['load_cvm_btn'],
                'normal': modal_system['load_normal_btn'], 
                'hgg': modal_system['load_hgg_btn']
            }
        },
        'segmentation_modal': {
            'system': segmentation_modal_system,
            'show_function': segmentation_modal_system['show_segmentation_modal'],
            'hide_function': segmentation_modal_system['hide_segmentation_modal']
        },
        'point_tip_modal': {
            'system': point_tip_modal_system,
            'show_function': point_tip_modal_system['show_modal'],
            'hide_function': point_tip_modal_system['hide_modal']
        },
        'box_tip_modal': {
            'system': box_tip_modal_system,
            'show_function': box_tip_modal_system['show_modal'],
            'hide_function': box_tip_modal_system['hide_modal']
        },
        'mg_orientation_functions': {
            'detect_from_metadata': detect_mg_orientation_from_metadata,
            'detect_from_filename': detect_mg_orientation_from_filename,
            'get_orientations_from_directory': get_mg_orientations_from_directory,
            'update_view_selector': update_view_selector_for_modality,
            'get_file_for_orientation': get_file_for_mg_orientation
        }
    }
