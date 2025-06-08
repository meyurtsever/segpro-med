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
        ])
        
        # All points are foreground
        labels = np.array([1, 1, 1, 1, 1])
        
        return points, labels
    
    def segment_slice(self, 
                     image: np.ndarray,
                     points: Optional[np.ndarray] = None,
                     labels: Optional[np.ndarray] = None,
                     boxes: Optional[np.ndarray] = None,
                     auto_prompts: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Segment a single brain MRI slice"""
        
        # Set image for prediction
        self.predictor.set_image(image)
        
        # Generate auto prompts if requested and no manual prompts provided
        if auto_prompts and points is None and boxes is None:
            points, labels = self.generate_auto_prompts(image.shape[:2])
        
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=boxes,
            multimask_output=True        )
        
        return masks, scores, logits
        
    def process_volume(self,
                      volume: np.ndarray,
                      dicom_processor: DICOMProcessor,
                      windowing_metadata: dict,
                      slice_range: Optional[str] = None,
                      prompts_data: Optional[dict] = None,
                      auto_prompts: bool = False,
                      original_slice_mapping: Optional[dict] = None,
                      quiet: bool = False) -> List[dict]:
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
            
            # Segment slice
            masks, scores, logits = self.segment_slice(
                rgb_image, points, labels, boxes, auto_prompts
            )
            
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
                'windowing_metadata': windowing_info
            }
            
            results.append(slice_result)
        
        return results

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
    axes[1, 1].axis('off')
    
    # All masks
    if len(result['masks']) > 1:
        combined_masks = np.sum(result['masks'], axis=0)
        axes[1, 2].imshow(combined_masks, cmap='viridis')
        axes[1, 2].set_title('All Masks Combined')
    else:
        axes[1, 2].imshow(result['masks'][0], cmap='viridis')
        axes[1, 2].set_title('Single Mask')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Slice {result["slice_idx"]} Segmentation Results\nWindow: {window_center}/{window_width}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_results(results: List[dict], output_dir: str, windowing_metadata: dict, save_visualizations: bool = False):
    """Save segmentation results with windowing metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
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
        slice_idx = result['slice_idx']
        
        # Save best mask
        best_mask_idx = np.argmax(result['scores'])
        best_mask = result['masks'][best_mask_idx]
        
        # Save as PNG
        mask_image = Image.fromarray((best_mask * 255).astype(np.uint8))
        mask_image.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_mask.png'))
        
        # Save as NPY for exact values
        np.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_mask.npy'), best_mask)
        
        # Save all masks and scores
        np.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_all_masks.npy'), result['masks'])
        np.save(os.path.join(masks_dir, f'slice_{slice_idx:04d}_scores.npy'), result['scores'])
        
        # Save slice-specific windowing metadata
        if 'windowing_metadata' in result:
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
                prompts_data[slice_idx]['boxes'] = box_data['boxes']
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
        args.quiet  # Pass quiet flag
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
