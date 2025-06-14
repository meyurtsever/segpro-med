#!/usr/bin/env python3
"""
Brain MRI DICOM Inference Script using SAM2
This script processes brain MRI DICOM files and performs segmentation using SAM2 model.
"""

import os
import argparse
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
import torch
import cv2
from scipy import ndimage
from typing import Optional, List, Tuple, Union
import json
from pathlib import Path
import logging

# SAM2 imports
from build_sam import build_sam2
from sam2_image_predictor import SAM2ImagePredictor

# Set up logging - default to WARNING level for better performance
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def setup_args():
    """Setup command line arguments"""
    parser = argparse.ArgumentParser(description='Brain MRI DICOM SAM2 Inference')
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default="checkpoints/MedSAM2_latest.pt",
        help='Path to SAM2 checkpoint'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default="configs/sam2.1_hiera_t512.yaml",
        help='Path to SAM2 config file'
    )
    
    parser.add_argument(
        '--dicom_folder', 
        type=str, 
        required=True,
        help='Path to folder containing DICOM files'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="./brain_mri_results",
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default="cuda",
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--slice_range', 
        type=str, 
        default=None,
        help='Slice range to process (e.g., "10:50" or "all")'
    )
    
    parser.add_argument(
        '--window_center', 
        type=int, 
        default=40,
        help='Window center for DICOM display (HU)'
    )
    
    parser.add_argument(
        '--window_width', 
        type=int, 
        default=80,
        help='Window width for DICOM display (HU)'
    )
    
    parser.add_argument(
        '--prompt_points', 
        type=str, 
        default=None,
        help='JSON file with point prompts for each slice'
    )
    
    parser.add_argument(
        '--prompt_boxes', 
        type=str, 
        default=None,
        help='JSON file with box prompts for each slice'
    )
    
    parser.add_argument(
        '--single_slice', 
        action='store_true',
        help='Process only the slices specified in prompt files (faster for single slice annotation)'
    )
    parser.add_argument(
        '--auto_prompts', 
        action='store_true',
        help='Generate automatic prompts for brain structures'
    )
    
    parser.add_argument(
        '--save_visualizations', 
        action='store_true',
        help='Save visualization images'
    )
    
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Reduce logging output for better performance'
    )
    
    return parser.parse_args()

def extract_dicom_windowing(dicom):
    """
    Extract windowing parameters from DICOM metadata.
    
    Args:
        dicom: pydicom Dataset object
    
    Returns:
        tuple: (window_center, window_width, rescale_slope, rescale_intercept)
    """
    # Default values for MRI
    window_center = 40
    window_width = 80
    rescale_slope = 1.0
    rescale_intercept = 0.0
    
    try:
        # Try to get window center and width from DICOM tags
        if hasattr(dicom, 'WindowCenter'):
            if isinstance(dicom.WindowCenter, (list, tuple)):
                window_center = float(dicom.WindowCenter[0])
            else:
                window_center = float(dicom.WindowCenter)
        
        if hasattr(dicom, 'WindowWidth'):
            if isinstance(dicom.WindowWidth, (list, tuple)):
                window_width = float(dicom.WindowWidth[0])
            else:
                window_width = float(dicom.WindowWidth)
        
        # Get rescale parameters for proper value conversion
        if hasattr(dicom, 'RescaleSlope'):
            rescale_slope = float(dicom.RescaleSlope)
        
        if hasattr(dicom, 'RescaleIntercept'):
            rescale_intercept = float(dicom.RescaleIntercept)
            
        logger.info(f"Extracted windowing: Center={window_center}, Width={window_width}, Slope={rescale_slope}, Intercept={rescale_intercept}")
        
    except Exception as e:
        logger.warning(f"Could not extract windowing parameters, using defaults: {e}")
    
    return window_center, window_width, rescale_slope, rescale_intercept

def load_and_preprocess_dicom_with_metadata(dicom_path, override_window_center=None, override_window_width=None):
    """
    Load and preprocess a DICOM image using its own windowing metadata.
    
    Args:
        dicom_path: Path to DICOM file
        override_window_center: Override window center (optional)
        override_window_width: Override window width (optional)
    
    Returns:
        tuple: (preprocessed_image, original_pixel_array, windowing_params_dict)
    """
    try:
        # Load DICOM file
        dicom = pydicom.dcmread(dicom_path)
        
        # Get pixel array
        pixel_array = dicom.pixel_array.astype(np.float32)
        
        # Extract windowing parameters from DICOM metadata
        window_center, window_width, rescale_slope, rescale_intercept = extract_dicom_windowing(dicom)
        
        # Use override values if provided
        if override_window_center is not None:
            window_center = override_window_center
        if override_window_width is not None:
            window_width = override_window_width
        
        # Apply rescaling to get proper values
        rescaled_array = pixel_array * rescale_slope + rescale_intercept
        
        # Apply windowing
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Window the image
        windowed = np.clip(rescaled_array, window_min, window_max)
        
        # Normalize to 0-255
        windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
        
        # Convert to RGB
        rgb_image = np.stack([windowed, windowed, windowed], axis=-1)
        
        # Store windowing parameters for reference
        windowing_params = {
            'window_center': window_center,
            'window_width': window_width,
            'rescale_slope': rescale_slope,
            'rescale_intercept': rescale_intercept,
            'window_min': window_min,
            'window_max': window_max,
            'original_shape': pixel_array.shape,
            'dicom_path': dicom_path,
            'dicom_file': os.path.basename(dicom_path)
        }
        
        return rgb_image, pixel_array, windowing_params
        
    except Exception as e:
        logger.error(f"Error loading DICOM {dicom_path}: {e}")
        return None, None, None

class DICOMProcessor:
    """Process DICOM files for brain MRI with metadata-aware windowing"""
    
    def __init__(self, default_window_center=40, default_window_width=80, use_dicom_windowing=True):
        self.default_window_center = default_window_center
        self.default_window_width = default_window_width
        self.use_dicom_windowing = use_dicom_windowing
        self.windowing_metadata = {}  # Store windowing info for each file
    
    def load_dicom_series(self, dicom_folder: str, slice_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, List[pydicom.Dataset], dict]:
        """Load a series of DICOM files with windowing metadata
        
        Args:
            dicom_folder: Path to folder containing DICOM files
            slice_indices: Optional list of specific slice indices to load (0-based)
        """
        dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_folder}")
        
        logger.info(f"Found {len(dicom_files)} DICOM files")
        
        # Read all DICOM files first to allow proper sorting
        slices = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                slices.append(ds)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
                
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
                slices = sorted(slices, key=lambda x: x.filename)
        except Exception as e:
            logger.error(f"Error sorting slices: {str(e)}")
            logger.warning("Using original file order")
        
        # Get the ordered file paths
        dicom_files = [s.filename for s in slices]
        
        # If specific slice indices are provided, only process those slices
        if slice_indices is not None:
            logger.info(f"Single slice mode: Only loading slices {slice_indices}")
            selected_slices = []
            for idx in slice_indices:
                if 0 <= idx < len(slices):
                    selected_slices.append(slices[idx])
                else:
                    logger.warning(f"Slice index {idx} out of range (0-{len(slices)-1})")
            slices = selected_slices
            logger.info(f"Reduced from {len(dicom_files)} to {len(slices)} slices")
        
        dicom_datasets = []
        pixel_arrays = []
        windowing_metadata = {}
        
        for i, ds in enumerate(slices):
            try:
                file_path = ds.filename
                
                # Extract windowing parameters for this slice
                window_center, window_width, rescale_slope, rescale_intercept = extract_dicom_windowing(ds)
                
                # If we're loading specific slices, use the original slice index for metadata
                # Otherwise use sequential indexing
                if slice_indices is not None and i < len(slice_indices):
                    metadata_index = slice_indices[i] + 1  # 1-based for consistency with mask file naming
                else:
                    metadata_index = i + 1  # 1-based for consistency with mask file naming
                
                # Store windowing metadata
                windowing_metadata[metadata_index] = {
                    'window_center': window_center,
                    'window_width': window_width,
                    'rescale_slope': rescale_slope,
                    'rescale_intercept': rescale_intercept,
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path)
                }
                
                # Get pixel array and apply proper scaling
                pixel_array = ds.pixel_array.astype(np.float32)
                if self.use_dicom_windowing:
                    # Apply rescaling using DICOM metadata
                    scaled_array = pixel_array * rescale_slope + rescale_intercept
                else:
                    # Use original pixel values
                    scaled_array = pixel_array
                    
                pixel_arrays.append(scaled_array)
                dicom_datasets.append(ds)
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        if not pixel_arrays:
            raise ValueError("No valid DICOM files could be loaded")
        
        # Stack arrays into 3D volume
        volume = np.stack(pixel_arrays, axis=0)
        logger.info(f"Loaded DICOM volume with shape: {volume.shape}")
        
        # Store windowing metadata for later use
        self.windowing_metadata = windowing_metadata
        
        return volume, dicom_datasets, windowing_metadata
    
    def apply_window_level_with_metadata(self, pixel_array: np.ndarray, slice_idx: int) -> np.ndarray:
        """Apply window/level using slice-specific DICOM metadata"""
        if slice_idx in self.windowing_metadata and self.use_dicom_windowing:
            metadata = self.windowing_metadata[slice_idx]
            window_center = metadata['window_center']
            window_width = metadata['window_width']
        else:
            # Use default values
            window_center = self.default_window_center
            window_width = self.default_window_width
        
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        
        # Clip values to window range
        windowed = np.clip(pixel_array, min_val, max_val)
        
        # Normalize to 0-255
        windowed = ((windowed - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return windowed
    
    def normalize_for_display_with_metadata(self, pixel_array: np.ndarray, slice_idx: int) -> np.ndarray:
        """Normalize pixel array for display using slice-specific metadata"""
        # Apply window/level with metadata
        windowed = self.apply_window_level_with_metadata(pixel_array, slice_idx)
        
        # Convert to RGB
        rgb_image = np.stack([windowed] * 3, axis=-1)
        
        return rgb_image
    
    def get_windowing_info(self, slice_idx: int) -> dict:
        """Get windowing information for a specific slice"""
        if slice_idx in self.windowing_metadata:
            return self.windowing_metadata[slice_idx].copy()
        else:
            return {
                'window_center': self.default_window_center,
                'window_width': self.default_window_width,
                'rescale_slope': 1.0,
                'rescale_intercept': 0.0,
                'file_path': 'unknown',
                'file_name': 'unknown'
            }

class BrainMRISegmenter:
    """Brain MRI segmentation using SAM2"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Initializing SAM2 model on {device}")
        
        # Build SAM2 model
        self.sam2_model = build_sam2(config_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)
        
        logger.info("SAM2 model initialized successfully")
    
    def generate_auto_prompts(self, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate automatic prompts for brain structures"""
        h, w = image_shape
        
        # Center point for brain
        center_x, center_y = w // 2, h // 2
        
        # Points around brain center for different structures
        points = np.array([
            [center_x, center_y],  # Brain center
            [center_x - w//4, center_y],  # Left hemisphere
            [center_x + w//4, center_y],  # Right hemisphere
            [center_x, center_y - h//4],  # Upper brain
            [center_x, center_y + h//4],  # Lower brain
        ])        # All points are foreground
        labels = np.array([1, 1, 1, 1, 1])
        
        return points, labels
    
    def segment_slice(self, 
                     image: np.ndarray,
                     points: Optional[np.ndarray] = None,
                     labels: Optional[np.ndarray] = None,
                     boxes: Optional[np.ndarray] = None,
                     auto_prompts: bool = False,
                     use_overlap_detection: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment a single brain MRI slice"""
        
        # Set image for prediction
        self.predictor.set_image(image)
        
        # Generate auto prompts if requested and no manual prompts provided
        if auto_prompts and points is None and boxes is None:
            points, labels = self.generate_auto_prompts(image.shape[:2])        # Handle multiple boxes - predict for each box separately
        if boxes is not None and len(boxes.shape) == 2 and boxes.shape[0] > 1:
            # Multiple boxes: predict for each box and combine results
            all_masks = []
            all_scores = []
            all_logits = []
            
            for i, box in enumerate(boxes):
                # Predict masks for single box
                box_masks, box_scores, box_logits = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box.reshape(1, -1),  # Reshape to (1, 4) for single box
                    multimask_output=True
                )
                
                # Collect results
                all_masks.append(box_masks)
                all_scores.append(box_scores)
                all_logits.append(box_logits)
            
            # Combine all results
            combined_masks = np.concatenate(all_masks, axis=0)
            combined_scores = np.concatenate(all_scores, axis=0)
            combined_logits = np.concatenate(all_logits, axis=0)
              # Apply overlap-based processing ONLY for automatic brain detection
            if use_overlap_detection:
                # GOLDEN SOLUTION: Use precision brain mask creation for perfect annotations
                precision_masks, precision_scores = self.create_precision_brain_masks(
                    combined_masks, combined_scores, boxes, image,
                    anatomical_validation=True
                )
                  # Create corresponding logits for precision masks (average from contributing masks)
                precision_logits = []
                for precision_mask in precision_masks:
                    # Find which original masks contributed to this precision mask
                    contributing_logits = []
                    for i, orig_mask in enumerate(combined_masks):
                        if np.sum((orig_mask > 0.5) & (precision_mask > 0.5)) > 50:  # Significant overlap
                            contributing_logits.append(combined_logits[i])
                    
                    if contributing_logits:
                        avg_logits = np.mean(contributing_logits, axis=0)
                    else:
                        # Fallback: use the best original mask's logits
                        best_idx = np.argmax(combined_scores)
                        avg_logits = combined_logits[best_idx]
                    
                    precision_logits.append(avg_logits)
                
                precision_logits = np.array(precision_logits)
                
                return precision_masks, precision_scores, precision_logits
            else:
                # Manual prompts: return all combined masks without overlap processing
                return combined_masks, combined_scores, combined_logits
        
        else:
            # Single box or no boxes - use original logic
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=boxes,
                multimask_output=True
            )
            
            return masks, scores, logits
        
    def process_volume(self,
                      volume: np.ndarray,
                      dicom_processor: DICOMProcessor,
                      windowing_metadata: dict,
                      slice_range: Optional[str] = None,
                      prompts_data: Optional[dict] = None,                      auto_prompts: bool = False,
                      original_slice_mapping: Optional[dict] = None,
                      quiet: bool = False,
                      is_automatic_brain_detection: bool = False) -> List[dict]:
        """Process entire MRI volume with per-slice windowing metadata
        
        Args:
            volume: 3D numpy array of the MRI volume
            dicom_processor: DICOMProcessor instance
            windowing_metadata: Dict containing windowing metadata for slices
            slice_range: String indicating slice range to process
            prompts_data: Dict containing prompt data for slices
            auto_prompts: Boolean indicating whether to use auto prompts
            original_slice_mapping: Dict mapping volume indices to original slice indices
        """
        """Process entire MRI volume with per-slice windowing metadata
        
        Args:
            volume: 3D numpy array of the MRI volume
            dicom_processor: DICOMProcessor instance
            windowing_metadata: Dict containing windowing metadata for slices
            slice_range: String indicating slice range to process
            prompts_data: Dict containing prompt data for slices
            auto_prompts: Boolean indicating whether to use auto prompts
            original_slice_mapping: Dict mapping volume indices to original slice indices
        """
        
        results = []
        
        # Determine slice range
        if slice_range is None or slice_range == "all":
            start_slice, end_slice = 0, volume.shape[0]
        else:
            start_slice, end_slice = map(int, slice_range.split(':'))
            end_slice = min(end_slice, volume.shape[0])
        
        logger.info(f"Processing slices {start_slice} to {end_slice}")
        
        for volume_idx in range(start_slice, end_slice):
            # Map volume index to original slice index if mapping is provided
            if original_slice_mapping and volume_idx in original_slice_mapping:
                original_slice_idx = original_slice_mapping[volume_idx]
                metadata_key = original_slice_idx + 1  # 1-based for metadata
            else:
                original_slice_idx = volume_idx
                metadata_key = volume_idx + 1  # 1-based for metadata
                logger.info(f"Processing volume slice {volume_idx} (original slice {original_slice_idx})")
            
            # Get slice data
            slice_data = volume[volume_idx]
              # Get windowing metadata for this slice
            windowing_info = dicom_processor.get_windowing_info(metadata_key)
            # Only log windowing info if not in quiet mode
            if not quiet:
                logger.info(f"Slice {original_slice_idx}: Using window center={windowing_info['window_center']}, width={windowing_info['window_width']}")
            
            # Normalize for SAM2 using slice-specific windowing
            rgb_image = dicom_processor.normalize_for_display_with_metadata(slice_data, metadata_key)
            
            # Get prompts for this slice
            points = None
            labels = None
            boxes = None
            
            if prompts_data and str(original_slice_idx + 1) in prompts_data:  # Use 1-based original slice index
                slice_prompts = prompts_data[str(original_slice_idx + 1)]
                if 'points' in slice_prompts:
                    points = np.array(slice_prompts['points'])
                    labels = np.array(slice_prompts['labels'])
                if 'boxes' in slice_prompts:
                    boxes = np.array(slice_prompts['boxes'])
                    if not quiet:
                        logger.info(f"Slice {original_slice_idx}: Found {len(boxes)} boxes with shape {boxes.shape}")
              # Segment slice
            if not quiet:
                logger.info(f"Slice {original_slice_idx}: Starting segmentation with boxes shape: {boxes.shape if boxes is not None else 'None'}")
              # Determine if this is automatic brain detection
            # Automatic brain detection is identified by having exactly 4 boxes from brain ROI detector
            use_overlap_detection = (is_automatic_brain_detection and 
                                   boxes is not None and 
                                   len(boxes.shape) == 2 and 
                                   boxes.shape[0] == 4)
            
            masks, scores, logits = self.segment_slice(
                rgb_image, points, labels, boxes, auto_prompts, use_overlap_detection
            )
            
            if not quiet:
                if use_overlap_detection:
                    logger.info(f"Slice {original_slice_idx}: Automatic brain detection - overlap-based segmentation completed. Got {len(masks)} overlap-focused masks with shape {masks.shape}")
                    logger.info(f"Slice {original_slice_idx}: Overlap-based scores: {scores.shape} = {scores}")
                    logger.info(f"Slice {original_slice_idx}: Masks focus on regions with highest consensus/agreement")
                else:
                    logger.info(f"Slice {original_slice_idx}: Manual prompt segmentation completed. Got {len(masks)} masks with shape {masks.shape}")
                    logger.info(f"Slice {original_slice_idx}: Scores: {scores.shape} = {scores}")
              # Store results with windowing metadata
            slice_result = {
                'slice_idx': original_slice_idx,  # Use original slice index
                'masks': masks,
                'scores': scores,
                'logits': logits,
                'original_image': slice_data,
                'rgb_image': rgb_image,
                'points': points,
                'labels': labels,
                'boxes': boxes,
                'used_overlap_detection': use_overlap_detection,  # Store whether overlap detection was used                'windowing_metadata': windowing_info
            }
            
            results.append(slice_result)
        
        return results
    
    def create_precision_brain_masks(self, masks: np.ndarray, scores: np.ndarray,
                                    boxes: np.ndarray, image: np.ndarray,
                                    anatomical_validation: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        GOLDEN SOLUTION: Multi-stage precision mask creation for perfect brain annotations
        
        This method implements a sophisticated 4-stage approach:
        1. Quality-based filtering using confidence scores and anatomical validation
        2. Spatial relationship analysis for brain structure coherence  
        3. Multi-criteria consensus detection (overlap + shape + position)
        4. Adaptive thresholding based on brain anatomy
        
        Args:
            masks: Array of shape (N, H, W) containing N binary masks
            scores: Array of shape (N,) containing confidence scores
            boxes: Array of shape (4, 4) containing the 4 prompt boxes
            image: Original brain image for anatomical validation
            anatomical_validation: Whether to apply anatomical constraints
            
        Returns:
            Tuple of (precision_masks, precision_scores) - optimized for brain anatomy
        """
        if len(masks) <= 1:
            return masks, scores
              # STAGE 1: Adaptive thresholding based on brain characteristics
        adaptive_threshold, threshold_reasoning = self.adaptive_brain_thresholding(
            masks, scores, image
        )
        logger.info(f"Using adaptive threshold: {adaptive_threshold:.3f} - {threshold_reasoning}")
        
        # Apply adaptive threshold for quality filtering
        quality_threshold = max(0.6, adaptive_threshold)  # Ensure minimum quality
        size_threshold_min = 200  # Minimum reasonable brain structure size
        size_threshold_max = image.shape[0] * image.shape[1] * 0.4  # Max 40% of image
        
        valid_masks = []
        valid_scores = []
        valid_indices = []
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_binary = (mask > 0.5).astype(np.uint8)
            mask_area = np.sum(mask_binary)
            
            # Quality criteria
            score_ok = score >= quality_threshold
            size_ok = size_threshold_min <= mask_area <= size_threshold_max
            
            # Anatomical validation - check if mask is within brain boundaries
            anatomical_ok = True
            if anatomical_validation:
                # Check if mask center is in reasonable brain region (not too peripheral)
                mask_center_y, mask_center_x = np.mean(np.where(mask_binary), axis=1)
                h, w = image.shape[:2]
                # Brain should be roughly in central 80% of image
                center_margin = 0.1
                anatomical_ok = (center_margin * h <= mask_center_y <= (1-center_margin) * h and
                               center_margin * w <= mask_center_x <= (1-center_margin) * w)
            
            if score_ok and size_ok and anatomical_ok:
                valid_masks.append(mask)
                valid_scores.append(score)
                valid_indices.append(i)
                
        if len(valid_masks) == 0:
            # Fallback to top 3 by score if all filtered out
            logger.warning("All masks filtered out, falling back to top scores")
            top_indices = np.argsort(scores)[-3:][::-1]
            return masks[top_indices], scores[top_indices]
            
        valid_masks = np.array(valid_masks)
        valid_scores = np.array(valid_scores)
        
        logger.info(f"Stage 1: Filtered {len(masks)} -> {len(valid_masks)} masks using quality criteria")
        
        # STAGE 2: Spatial relationship analysis
        # Analyze which masks have good spatial relationships for brain structures
        binary_masks = (valid_masks > 0.5).astype(np.uint8)
        h, w = image.shape[:2]
        
        # Calculate spatial features for each mask
        spatial_scores = []
        for i, mask in enumerate(binary_masks):
            # Calculate spatial features
            mask_center_y, mask_center_x = np.mean(np.where(mask), axis=1)
            
            # Brain anatomy preferences (adjust based on your specific anatomy):
            # 1. Prefer masks closer to brain center
            image_center_y, image_center_x = h/2, w/2
            center_distance = np.sqrt((mask_center_y - image_center_y)**2 + (mask_center_x - image_center_x)**2)
            center_score = 1.0 - (center_distance / (h/2))  # Normalized distance score
            
            # 2. Prefer masks with good aspect ratio (not too elongated)
            mask_coords = np.where(mask)
            if len(mask_coords[0]) > 0:
                y_span = np.max(mask_coords[0]) - np.min(mask_coords[0])
                x_span = np.max(mask_coords[1]) - np.min(mask_coords[1])
                aspect_ratio = max(y_span, x_span) / max(min(y_span, x_span), 1)
                aspect_score = 1.0 / (1.0 + max(0, aspect_ratio - 3))  # Penalize very elongated shapes
            else:
                aspect_score = 0
                
            # 3. Prefer masks with reasonable compactness
            mask_area = np.sum(mask)
            if mask_area > 0:
                # Calculate perimeter using edge detection
                import cv2
                edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
                perimeter = np.sum(edges > 0)
                compactness = (4 * np.pi * mask_area) / max(perimeter**2, 1)  # Closer to 1 = more circular
            else:
                compactness = 0
                
            # Combine spatial scores
            spatial_score = (center_score * 0.4 + aspect_score * 0.3 + compactness * 0.3)
            spatial_scores.append(spatial_score)
            
        spatial_scores = np.array(spatial_scores)
        logger.info(f"Stage 2: Calculated spatial relationship scores")
        
        # STAGE 3: Multi-criteria consensus detection
        # Create consensus regions using overlap + spatial + confidence criteria
        overlap_map = np.sum(binary_masks, axis=0)
        
        # Dynamic overlap threshold based on number of valid masks
        dynamic_overlap_threshold = max(2, len(valid_masks) // 3)  # At least 1/3 of masks should agree
        
        # Find consensus regions
        consensus_regions = overlap_map >= dynamic_overlap_threshold
        
        if np.sum(consensus_regions) < 100:  # If no good consensus
            # STAGE 4A: Intelligent fallback - select best masks using combined criteria
            logger.info("No strong consensus found, using intelligent selection")
            
            # Combine confidence, spatial, and uniqueness scores
            combined_scores = []
            for i in range(len(valid_masks)):
                confidence_component = valid_scores[i]
                spatial_component = spatial_scores[i]
                
                # Uniqueness component - prefer masks that cover different regions
                uniqueness_component = 1.0
                for j in range(len(valid_masks)):
                    if i != j:
                        overlap_ratio = np.sum((binary_masks[i] > 0) & (binary_masks[j] > 0)) / max(np.sum(binary_masks[i] > 0), 1)
                        if overlap_ratio > 0.8:  # Very similar masks
                            uniqueness_component *= 0.7  # Reduce score for redundant masks
                
                final_score = (confidence_component * 0.5 + 
                             spatial_component * 0.3 + 
                             uniqueness_component * 0.2)
                combined_scores.append(final_score)
            
            # Select top 3 based on combined criteria
            best_indices = np.argsort(combined_scores)[-3:][::-1]
            selected_masks = valid_masks[best_indices]
            selected_scores = np.array([combined_scores[i] for i in best_indices])
            
            logger.info(f"Stage 4A: Selected {len(selected_masks)} masks using intelligent criteria")
            return selected_masks, selected_scores
            
        else:
            # STAGE 4B: Consensus-based mask creation
            logger.info("Strong consensus found, creating consensus-based masks")
            
            # Use connected components to separate consensus regions
            from scipy import ndimage
            labeled_consensus, num_features = ndimage.label(consensus_regions)
            
            consensus_masks = []
            consensus_scores = []
            
            for region_id in range(1, num_features + 1):
                region_mask = (labeled_consensus == region_id)
                region_area = np.sum(region_mask)
                
                if region_area < 100:  # Skip tiny regions
                    continue
                
                # Calculate consensus strength for this region
                region_overlap_values = overlap_map[region_mask]
                consensus_strength = np.mean(region_overlap_values) / len(valid_masks)
                
                # Weight by contributing mask qualities
                contributing_quality = 0
                total_contributors = 0
                
                for i, mask in enumerate(binary_masks):
                    contribution = np.sum(mask[region_mask])
                    if contribution > region_area * 0.1:  # At least 10% overlap
                        weight = contribution / region_area
                        contributing_quality += (valid_scores[i] * spatial_scores[i]) * weight
                        total_contributors += weight
                
                if total_contributors > 0:
                    final_consensus_score = (contributing_quality / total_contributors) * consensus_strength
                    consensus_masks.append(region_mask.astype(np.float32))
                    consensus_scores.append(final_consensus_score)
                    
                    logger.info(f"Consensus region: area={region_area}, "
                               f"strength={consensus_strength:.3f}, score={final_consensus_score:.3f}")
            
            if consensus_masks:
                # Sort by score and return top masks
                if len(consensus_masks) > 3:
                    top_consensus_indices = np.argsort(consensus_scores)[-3:][::-1]
                    consensus_masks = [consensus_masks[i] for i in top_consensus_indices]
                    consensus_scores = [consensus_scores[i] for i in top_consensus_indices]
                
                consensus_masks_array = np.array(consensus_masks)
                consensus_scores_array = np.array(consensus_scores)
                
                logger.info(f"Stage 4B: Created {len(consensus_masks)} high-quality consensus masks")
                return consensus_masks_array, consensus_scores_array
            else:
                # Fallback to stage 4A if consensus creation failed
                logger.info("Consensus creation failed, falling back to intelligent selection")
                best_indices = np.argsort(valid_scores * spatial_scores)[-3:][::-1]
                return valid_masks[best_indices], valid_scores[best_indices]
            top_indices = np.argsort(scores)[-3:][::-1]
            return masks[top_indices], scores[top_indices]
        
        # Sort by score and return top 3 overlap-based masks
        if len(overlap_masks) > 3:
            sorted_indices = np.argsort(overlap_scores)[-3:][::-1]
            overlap_masks = [overlap_masks[i] for i in sorted_indices]
            overlap_scores = [overlap_scores[i] for i in sorted_indices]
        
        # Convert to numpy arrays
        overlap_masks_array = np.array(overlap_masks)
        overlap_scores_array = np.array(overlap_scores)
        
        logger.info(f"Created {len(overlap_masks)} overlap-based masks from {len(masks)} original masks")
        
        return overlap_masks_array, overlap_scores_array

    def adaptive_brain_thresholding(self, masks: np.ndarray, scores: np.ndarray, 
                                  image: np.ndarray, brain_characteristics: Optional[dict] = None) -> Tuple[float, str]:
        """
        GOLDEN SOLUTION: Adaptive thresholding based on brain anatomy and image characteristics
        
        This method analyzes the specific brain image and masks to determine optimal thresholds:
        1. Analyzes brain tissue contrast and intensity distribution
        2. Evaluates mask quality metrics specific to brain anatomy
        3. Adapts thresholds based on brain size, position, and clarity
        4. Provides reasoning for threshold selection
        
        Args:
            masks: Generated masks to analyze
            scores: MEDSAM2 confidence scores
            image: Original brain image
            brain_characteristics: Optional brain analysis results
            
        Returns:
            Tuple of (optimal_threshold, reasoning)
        """
        if len(masks) == 0 or len(scores) == 0:
            return 0.5, "Fallback threshold - no masks to analyze"
        
        # STEP 1: Analyze brain image characteristics
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Calculate image quality metrics
        image_std = np.std(gray_image)
        image_mean = np.mean(gray_image)
        
        # Analyze contrast (difference between brain and background)
        # Assume background is in the corners
        h, w = gray_image.shape
        corner_regions = [
            gray_image[0:h//10, 0:w//10],           # Top-left
            gray_image[0:h//10, -w//10:],           # Top-right  
            gray_image[-h//10:, 0:w//10],           # Bottom-left
            gray_image[-h//10:, -w//10:]            # Bottom-right
        ]
        background_intensity = np.mean([np.mean(region) for region in corner_regions])
        brain_background_contrast = abs(image_mean - background_intensity) / max(image_mean, 1)
        
        # STEP 2: Analyze mask quality metrics
        mask_quality_scores = []
        anatomical_scores = []
        
        for i, mask in enumerate(masks):
            binary_mask = (mask > 0.5).astype(np.uint8)
            
            # Quality metrics
            mask_area = np.sum(binary_mask)
            mask_compactness = self._calculate_mask_compactness(binary_mask)
            mask_position_score = self._calculate_anatomical_position_score(binary_mask, gray_image.shape)
            edge_clarity_score = self._calculate_edge_clarity_score(binary_mask, gray_image)
            
            # Combine quality metrics
            quality_score = (
                min(1.0, mask_area / (h * w * 0.1)) * 0.3 +  # Size reasonableness
                mask_compactness * 0.3 +                      # Shape quality
                mask_position_score * 0.2 +                   # Anatomical position
                edge_clarity_score * 0.2                      # Edge quality
            )
            
            mask_quality_scores.append(quality_score)
            anatomical_scores.append(mask_position_score)
        
        mask_quality_scores = np.array(mask_quality_scores)
        anatomical_scores = np.array(anatomical_scores)
        
        # STEP 3: Determine adaptive threshold based on image and mask characteristics
        base_threshold = 0.5
        
        # Adjustments based on image quality
        if brain_background_contrast > 0.3:  # High contrast image
            contrast_adjustment = 0.1  # Can be more selective
            contrast_reason = "high contrast allows selective thresholding"
        elif brain_background_contrast < 0.15:  # Low contrast image
            contrast_adjustment = -0.15  # Need to be more inclusive
            contrast_reason = "low contrast requires inclusive thresholding"
        else:
            contrast_adjustment = 0.0
            contrast_reason = "moderate contrast uses standard thresholding"
        
        # Adjustments based on mask quality distribution
        avg_quality = np.mean(mask_quality_scores)
        quality_std = np.std(mask_quality_scores)
        
        if avg_quality > 0.8 and quality_std < 0.1:  # High quality, consistent masks
            quality_adjustment = 0.1  # Can afford to be selective
            quality_reason = "high-quality consistent masks allow selective threshold"
        elif avg_quality < 0.5:  # Poor quality masks overall
            quality_adjustment = -0.1  # Be more inclusive
            quality_reason = "poor mask quality requires inclusive threshold"
        else:
            quality_adjustment = 0.0
            quality_reason = "moderate mask quality uses standard threshold"
        
        # Adjustments based on score distribution
        score_range = np.max(scores) - np.min(scores)
        if score_range > 0.4:  # Good score separation
            score_adjustment = 0.05  # Can discriminate better
            score_reason = "good score separation allows discrimination"
        elif score_range < 0.2:  # Poor score separation
            score_adjustment = -0.05  # Be more conservative
            score_reason = "poor score separation requires conservative threshold"
        else:
            score_adjustment = 0.0
            score_reason = "moderate score separation uses standard threshold"
        
        # Adjustments based on anatomical positioning
        avg_anatomical_score = np.mean(anatomical_scores)
        if avg_anatomical_score > 0.8:  # Masks in good anatomical positions
            anatomical_adjustment = 0.05
            anatomical_reason = "excellent anatomical positioning"
        elif avg_anatomical_score < 0.5:  # Poor anatomical positioning
            anatomical_adjustment = -0.1
            anatomical_reason = "poor anatomical positioning requires lower threshold"
        else:
            anatomical_adjustment = 0.0
            anatomical_reason = "moderate anatomical positioning"
        
        # Calculate final adaptive threshold
        adaptive_threshold = base_threshold + contrast_adjustment + quality_adjustment + score_adjustment + anatomical_adjustment
        
        # Clamp to reasonable range
        adaptive_threshold = max(0.1, min(0.9, adaptive_threshold))
        
        # Create reasoning explanation
        reasoning = (f"Adaptive threshold {adaptive_threshold:.3f} determined by: "
                    f"{contrast_reason}; {quality_reason}; {score_reason}; {anatomical_reason}. "
                    f"Image stats: contrast={brain_background_contrast:.3f}, "
                    f"avg_mask_quality={avg_quality:.3f}, score_range={score_range:.3f}")
        
        logger.info(f"Adaptive thresholding: {reasoning}")
        
        return adaptive_threshold, reasoning
    
    def _calculate_mask_compactness(self, binary_mask: np.ndarray) -> float:
        """Calculate compactness score (how circular/compact the mask is)"""
        area = np.sum(binary_mask)
        if area == 0:
            return 0.0
        
        # Find contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Compactness = 4π × area / perimeter²  (closer to 1 = more circular)
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return min(1.0, compactness)
    
    def _calculate_anatomical_position_score(self, binary_mask: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate how well the mask is positioned for brain anatomy"""
        h, w = image_shape
        
        if np.sum(binary_mask) == 0:
            return 0.0
        
        # Find mask center
        mask_coords = np.where(binary_mask)
        center_y = np.mean(mask_coords[0])
        center_x = np.mean(mask_coords[1])
        
        # Brain anatomy preferences:
        # 1. Should be in central region (not too peripheral)
        center_score = 1.0 - min(1.0, 2 * max(
            abs(center_x - w/2) / (w/2),
            abs(center_y - h/2) / (h/2)
        ))
        
        # 2. Should not be too close to edges
        edge_distance_x = min(center_x / w, (w - center_x) / w)
        edge_distance_y = min(center_y / h, (h - center_y) / h)
        edge_score = min(1.0, 4 * min(edge_distance_x, edge_distance_y))
        
        # 3. Should have reasonable vertical position (brain is usually in upper 2/3)
        vertical_preference = 1.0 if center_y < 0.75 * h else 0.5
        
        anatomical_score = center_score * 0.4 + edge_score * 0.4 + vertical_preference * 0.2
        return anatomical_score
    
    def _calculate_edge_clarity_score(self, binary_mask: np.ndarray, gray_image: np.ndarray) -> float:
        """Calculate how clear/sharp the mask edges are in the original image"""
        # Create mask boundary
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        boundary = dilated - binary_mask
        
        if np.sum(boundary) == 0:
            return 0.0
        
        # Calculate gradient magnitude at boundary
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Average gradient at boundary
        boundary_gradients = gradient_magnitude[boundary > 0]
        if len(boundary_gradients) == 0:
            return 0.0
        
        avg_gradient = np.mean(boundary_gradients)
        # Normalize gradient score (typical brain edge gradients are 20-100)
        edge_clarity = min(1.0, avg_gradient / 50.0)
        
        return edge_clarity

    # ...existing code...
def visualize_results(result: dict, save_path: str):
    """Visualize segmentation results with windowing metadata"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get windowing info for title
    windowing_info = result.get('windowing_metadata', {})
    window_center = windowing_info.get('window_center', 'N/A')
    window_width = windowing_info.get('window_width', 'N/A')
    file_name = windowing_info.get('file_name', 'N/A')
    
    # Original image
    axes[0, 0].imshow(result['original_image'], cmap='gray')
    axes[0, 0].set_title(f'Original DICOM\n{file_name}')
    axes[0, 0].axis('off')
    
    # RGB image with windowing info
    axes[0, 1].imshow(result['rgb_image'])
    axes[0, 1].set_title(f'Windowed RGB\nC:{window_center} W:{window_width}')
    axes[0, 1].axis('off')
    
    # Show prompts on RGB image
    axes[0, 2].imshow(result['rgb_image'])
    if result['points'] is not None:
        points = result['points']
        labels = result['labels']
        for i, (point, label) in enumerate(zip(points, labels)):
            color = 'red' if label == 1 else 'blue'
            axes[0, 2].scatter(point[0], point[1], c=color, s=50, marker='*')
    if result['boxes'] is not None:
        for box in result['boxes']:
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                               fill=False, color='green', linewidth=2)
            axes[0, 2].add_patch(rect)
    axes[0, 2].set_title('Prompts')
    axes[0, 2].axis('off')
    
    # Best mask (highest score)
    best_mask_idx = np.argmax(result['scores'])
    axes[1, 0].imshow(result['masks'][best_mask_idx], cmap='viridis')
    axes[1, 0].set_title(f'Best Mask (Score: {result["scores"][best_mask_idx]:.3f})')
    axes[1, 0].axis('off')
    
    # Overlay
    overlay = result['rgb_image'].copy()
    mask_overlay = result['masks'][best_mask_idx]
    overlay[mask_overlay > 0] = [255, 0, 0]  # Red overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')    # All masks - show different visualization based on processing mode
    if len(result['masks']) > 1:
        # Create overlap intensity map
        combined_masks = np.sum(result['masks'], axis=0)
        axes[1, 2].imshow(combined_masks, cmap='hot', alpha=0.8)
        axes[1, 2].imshow(result['rgb_image'], alpha=0.3)  # Show underlying image
        
        # Set title based on processing mode
        if result.get('used_overlap_detection', False):
            axes[1, 2].set_title(f'Overlap-Based Masks (n={len(result["masks"])})')
        else:
            axes[1, 2].set_title(f'All Masks Combined (n={len(result["masks"])})')
    else:
        axes[1, 2].imshow(result['masks'][0], cmap='viridis')
        if result.get('used_overlap_detection', False):
            axes[1, 2].set_title('Single Overlap Mask')
        else:
            axes[1, 2].set_title('Single Mask')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Slice {result["slice_idx"]} Segmentation Results\nWindow: {window_center}/{window_width}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_results(results: List[dict], output_dir: str, windowing_metadata: dict, save_visualizations: bool = False):
    """Save segmentation results with windowing metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have any results
    if not results:
        logger.warning("No results to save")
        return
    
    # Save masks
    masks_dir = os.path.join(output_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Save visualizations
    if save_visualizations:
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
    
    # Save summary (without windowing metadata to improve performance)
    summary = {
        'total_slices': len(results),
        'slice_indices': [r['slice_idx'] for r in results],
        'average_scores': [],
        'processing_info': {
            'timestamp': str(Path().cwd()),
            'description': 'Brain MRI DICOM segmentation using slice-specific windowing parameters'
        }
    }
    
    # Save detailed windowing info per slice
    windowing_summary = {}
    
    for result in results:
        try:
            slice_idx = result['slice_idx']
            
            # Check if we have masks and scores
            if 'masks' not in result or 'scores' not in result:
                logger.warning(f"Skipping slice {slice_idx} - missing masks or scores")
                continue
                
            masks = result['masks']
            scores = result['scores']
            
            if len(masks) == 0 or len(scores) == 0:
                logger.warning(f"Skipping slice {slice_idx} - empty masks or scores")
                continue
            
            # Save best mask
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Save as PNG
            mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))
            mask_image.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_mask.png'))
            
            # Save as NPY for exact values
            np.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_mask.npy'), best_mask)
            
            # Save all masks and scores
            np.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_all_masks.npy'), masks)
            np.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_scores.npy'), scores)
            
            # Save slice-specific windowing metadata
            if 'windowing_metadata' in result:
                windowing_summary[slice_idx] = result['windowing_metadata']
            
            # Add to summary
            summary['average_scores'].append(float(np.mean(scores)))
            
            # Save visualization
            if save_visualizations:
                viz_path = os.path.join(viz_dir, f'slice_{slice_idx:04d}_visualization.png')
                visualize_results(result, viz_path)
            
        except Exception as e:
            logger.error(f"Error saving results for slice {result.get('slice_idx', 'unknown')}: {e}")
            continue
            windowing_summary[slice_idx] = result['windowing_metadata']
        
        # Add to summary
        summary['average_scores'].append(float(np.mean(result['scores'])))
        
        # Save visualization
        if save_visualizations:
            viz_path = os.path.join(viz_dir, f'slice_{slice_idx:04d}_visualization.png')
            visualize_results(result, viz_path)
        
        logger.info(f"Saved results for slice {slice_idx} (Window: {result.get('windowing_metadata', {}).get('window_center', 'N/A')}/{result.get('windowing_metadata', {}).get('window_width', 'N/A')})")
    
    # Save summary
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed windowing information
    with open(os.path.join(output_dir, 'windowing_metadata.json'), 'w') as f:
        json.dump(windowing_summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Windowing metadata saved to {os.path.join(output_dir, 'windowing_metadata.json')}")

def load_prompts(prompts_file: str) -> dict:
    """Load prompts from JSON file"""
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r') as f:
            return json.load(f)
    return {}

def main():
    args = setup_args()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)  # Only show errors
        logger.setLevel(logging.ERROR)
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = "cpu"
    
    # Initialize processors with metadata-aware windowing
    # If user provided specific window values, they will override DICOM metadata
    use_dicom_windowing = args.window_center == 40 and args.window_width == 80  # Using defaults means use DICOM metadata
    dicom_processor = DICOMProcessor(
        default_window_center=args.window_center, 
        default_window_width=args.window_width,
        use_dicom_windowing=use_dicom_windowing
    )
    
    segmenter = BrainMRISegmenter(args.checkpoint, args.config, args.device)
    
    # Load prompts first to determine which slices to load
    prompts_data = {}
    if args.prompt_points:
        prompts_data.update(load_prompts(args.prompt_points))
    if args.prompt_boxes:
        box_prompts = load_prompts(args.prompt_boxes)
        for slice_idx, box_data in box_prompts.items():
            if slice_idx not in prompts_data:
                prompts_data[slice_idx] = {}
            prompts_data[slice_idx]['boxes'] = box_data['boxes']
    
    # Determine which slice indices to load for single slice mode
    slice_indices_to_load = None
    if args.single_slice and prompts_data:
        # Get all slice indices that have prompts (convert to 0-based indexing)
        prompt_slice_keys = list(prompts_data.keys())
        if prompt_slice_keys:
            slice_indices_to_load = [int(s) - 1 for s in prompt_slice_keys]  # Convert to 0-based
            logger.info(f"Single slice mode: Will load only slices {[i+1 for i in slice_indices_to_load]} (1-based)")
    
    # Load DICOM data with optional selective loading
    logger.info(f"Loading DICOM data from {args.dicom_folder}")
    logger.info(f"Using DICOM metadata windowing: {use_dicom_windowing}")
    volume, dicom_datasets, windowing_metadata = dicom_processor.load_dicom_series(
        args.dicom_folder, 
        slice_indices_to_load
    )
    
    # Create slice mapping for selective loading
    original_slice_mapping = None
    if slice_indices_to_load is not None:
        # Map volume indices to original slice indices
        original_slice_mapping = {i: slice_indices_to_load[i] for i in range(len(slice_indices_to_load))}
        logger.info(f"Created slice mapping: {original_slice_mapping}")
    
    # Log windowing information
    logger.info("Windowing information summary:")
    for slice_idx, metadata in windowing_metadata.items():
        logger.info(f"  Slice {slice_idx}: Center={metadata['window_center']}, Width={metadata['window_width']}, File={metadata['file_name']}")
    
    # Determine slice range for processing
    slice_range = args.slice_range
    if args.single_slice and slice_indices_to_load:
        # For single slice mode, process all loaded slices (they're already filtered)
        slice_range = f"0:{volume.shape[0]}"  # Process all loaded slices
        logger.info(f"Single slice mode: Processing all {volume.shape[0]} loaded slices")
    elif args.single_slice and prompts_data:
        # Fallback for when slice_indices_to_load wasn't set
        prompt_slices = list(prompts_data.keys())
        if prompt_slices:
            slice_indices = [int(s) for s in prompt_slices]
            min_slice = min(slice_indices)
            max_slice = max(slice_indices) + 1
            slice_range = f"{min_slice}:{max_slice}"
            logger.info(f"Single slice mode fallback: Processing slices {prompt_slices}")
    
    # Load prompts again (redundant but keeping for compatibility)
    # This section can be removed in future optimization
    if not prompts_data:  # Only load if not already loaded
        if args.prompt_points:
            prompts_data.update(load_prompts(args.prompt_points))
        if args.prompt_boxes:
            box_prompts = load_prompts(args.prompt_boxes)
            for slice_idx, box_data in box_prompts.items():
                if slice_idx not in prompts_data:
                    prompts_data[slice_idx] = {}
                prompts_data[slice_idx]['boxes'] = box_data['boxes']    # Detect if this is automatic brain detection
    # Automatic brain detection is identified by:
    # 1. Using box prompts (--prompt_boxes)
    # 2. Each slice having exactly 4 boxes
    # 3. Box prompts coming from brain_target_prompts.json (generated by brain ROI detector)
    is_automatic_brain_detection = False
    if args.prompt_boxes and prompts_data:
        # Check if this looks like automatic brain detection
        sample_slice_data = next(iter(prompts_data.values()), {})
        if ('boxes' in sample_slice_data and 
            len(sample_slice_data['boxes']) == 4 and
            'brain_target_prompts.json' in args.prompt_boxes):
            is_automatic_brain_detection = True
            logger.info("Detected automatic brain detection mode - will use overlap-based processing")
        else:
            logger.info("Detected manual prompt mode - will use standard processing")
    
    # Process volume with windowing metadata
    if not args.quiet:
        logger.info("Starting segmentation with per-slice windowing...")
    results = segmenter.process_volume(
        volume, 
        dicom_processor,
        windowing_metadata,
        slice_range,  # Use the determined slice range
        prompts_data, 
        args.auto_prompts,
        original_slice_mapping,  # Pass the slice mapping
        args.quiet,  # Pass quiet flag
        is_automatic_brain_detection  # Pass automatic detection flag
    )
      # Save results with windowing metadata
    if not args.quiet:
        logger.info("Saving results...")
    save_results(results, args.output_dir, windowing_metadata, args.save_visualizations)
    
    if not args.quiet:
        logger.info("Brain MRI DICOM inference completed successfully!")
        logger.info(f"Windowing metadata saved for {len(windowing_metadata)} slices")

if __name__ == "__main__":
    main()
