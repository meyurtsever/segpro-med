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

# Set up logging
logging.basicConfig(level=logging.INFO)
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
    
    def load_dicom_series(self, dicom_folder: str) -> Tuple[np.ndarray, List[pydicom.Dataset], dict]:
        """Load a series of DICOM files with windowing metadata"""
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
        
        dicom_datasets = []
        pixel_arrays = []
        windowing_metadata = {}
        
        for i, ds in enumerate(slices):
            try:
                file_path = ds.filename
                
                # Extract windowing parameters for this slice
                window_center, window_width, rescale_slope, rescale_intercept = extract_dicom_windowing(ds)
                
                # Store windowing metadata with 1-based index for consistency with mask file naming
                windowing_metadata[i+1] = {
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
            multimask_output=True
        )
        
        return masks, scores, logits
    
    def process_volume(self,
                      volume: np.ndarray,
                      dicom_processor: DICOMProcessor,
                      windowing_metadata: dict,
                      slice_range: Optional[str] = None,
                      prompts_data: Optional[dict] = None,
                      auto_prompts: bool = False) -> List[dict]:
        """Process entire MRI volume with per-slice windowing metadata"""
        
        results = []
        
        # Determine slice range
        if slice_range is None or slice_range == "all":
            start_slice, end_slice = 0, volume.shape[0]
        else:
            start_slice, end_slice = map(int, slice_range.split(':'))
            end_slice = min(end_slice, volume.shape[0])
        
        logger.info(f"Processing slices {start_slice} to {end_slice}")
        
        for slice_idx in range(start_slice, end_slice):
            logger.info(f"Processing slice {slice_idx}/{volume.shape[0]}")
            
            # Get slice data
            slice_data = volume[slice_idx]
            
            # Get windowing metadata for this slice
            windowing_info = dicom_processor.get_windowing_info(slice_idx)
            logger.info(f"Slice {slice_idx}: Using window center={windowing_info['window_center']}, width={windowing_info['window_width']}")
            
            # Normalize for SAM2 using slice-specific windowing
            rgb_image = dicom_processor.normalize_for_display_with_metadata(slice_data, slice_idx)
            
            # Get prompts for this slice
            points = None
            labels = None
            boxes = None
            
            if prompts_data and str(slice_idx) in prompts_data:
                slice_prompts = prompts_data[str(slice_idx)]
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
                'slice_idx': slice_idx,
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
    
    # Load DICOM data with windowing metadata
    logger.info(f"Loading DICOM data from {args.dicom_folder}")
    logger.info(f"Using DICOM metadata windowing: {use_dicom_windowing}")
    volume, dicom_datasets, windowing_metadata = dicom_processor.load_dicom_series(args.dicom_folder)
    
    # Log windowing information
    logger.info("Windowing information summary:")
    for slice_idx, metadata in windowing_metadata.items():
        logger.info(f"  Slice {slice_idx}: Center={metadata['window_center']}, Width={metadata['window_width']}, File={metadata['file_name']}")
    
    # Load prompts
    prompts_data = {}
    if args.prompt_points:
        prompts_data.update(load_prompts(args.prompt_points))
    if args.prompt_boxes:
        box_prompts = load_prompts(args.prompt_boxes)
        for slice_idx, box_data in box_prompts.items():
            if slice_idx not in prompts_data:
                prompts_data[slice_idx] = {}
            prompts_data[slice_idx]['boxes'] = box_data['boxes']
    
    # Determine slice range - if single_slice option is used, only process slices with prompts
    slice_range = args.slice_range
    if args.single_slice and prompts_data:
        # Get all slice indices that have prompts
        prompt_slices = list(prompts_data.keys())
        if prompt_slices:
            # Convert to integers and get min/max range
            slice_indices = [int(s) for s in prompt_slices]
            min_slice = min(slice_indices)
            max_slice = max(slice_indices) + 1  # +1 because range is exclusive
            slice_range = f"{min_slice}:{max_slice}"
            logger.info(f"Single slice mode: Processing only slices with prompts: {prompt_slices}")
    
    # Process volume with windowing metadata
    logger.info("Starting segmentation with per-slice windowing...")
    results = segmenter.process_volume(
        volume, 
        dicom_processor,
        windowing_metadata,
        slice_range,  # Use the determined slice range
        prompts_data, 
        args.auto_prompts
    )
    
    # Save results with windowing metadata
    logger.info("Saving results...")
    save_results(results, args.output_dir, windowing_metadata, args.save_visualizations)
    
    logger.info("Brain MRI DICOM inference completed successfully!")
    logger.info(f"Windowing metadata saved for {len(windowing_metadata)} slices")

if __name__ == "__main__":
    main()
