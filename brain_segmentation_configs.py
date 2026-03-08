#!/usr/bin/env python3
"""
SAM2 Configuration Profiles for Brain Structure Segmentation

This module provides optimized parameter configurations for different
brain imaging and segmentation tasks using SAM2AutomaticMaskGenerator.

Usage:
    from brain_segmentation_configs import BRAIN_CONFIGS
    config = BRAIN_CONFIGS['high_detail']
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model, **config)
"""

# Base configuration - balanced performance and quality
BASE_BRAIN_CONFIG = {
    'points_per_side': 32,
    'points_per_batch': 64,
    'pred_iou_thresh': 0.65,
    'stability_score_thresh': 0.75,
    'stability_score_offset': 1.0,
    'mask_threshold': 0.0,
    'box_nms_thresh': 0.7,
    'crop_n_layers': 1,
    'crop_nms_thresh': 0.7,
    'crop_overlap_ratio': 0.3413,
    'crop_n_points_downscale_factor': 1,
    'min_mask_region_area': 25,
    'output_mode': 'binary_mask',
    'use_m2m': True,
    'multimask_output': True
}

# Configuration profiles for different use cases
BRAIN_CONFIGS = {
    
    # High detail configuration for research and fine analysis
    'high_detail': {
        **BASE_BRAIN_CONFIG,
        'points_per_side': 40,              # Even higher resolution sampling
        'points_per_batch': 80,             # More points per batch
        'pred_iou_thresh': 0.60,            # Lower threshold for more candidates
        'stability_score_thresh': 0.70,     # Lower for subtle structures
        'crop_n_layers': 2,                 # Two crop layers for multi-scale
        'min_mask_region_area': 15,         # Capture very small structures
        'use_m2m': True                     # Essential for consistency
    },
    
    # Optimized for small structure detection (ventricles, eyes)
    'small_structures': {
        **BASE_BRAIN_CONFIG,
        'points_per_side': 36,              # Dense sampling
        'pred_iou_thresh': 0.55,            # Very permissive threshold
        'stability_score_thresh': 0.65,     # Lower stability requirement
        'crop_n_layers': 1,                 # Multi-scale processing
        'min_mask_region_area': 10,         # Minimal area filtering
        'box_nms_thresh': 0.6,              # Less aggressive NMS
        'use_m2m': True                     # Refinement important
    },
    
    # Balanced configuration (default recommendation)
    'balanced': BASE_BRAIN_CONFIG,
    
    # Fast processing with reasonable quality
    'fast': {
        **BASE_BRAIN_CONFIG,
        'points_per_side': 24,              # Fewer sample points
        'points_per_batch': 48,             # Smaller batches
        'pred_iou_thresh': 0.80,            # Higher threshold = fewer candidates - was 0.7
        'stability_score_thresh': 0.80,     # Higher stability requirement
        'crop_n_layers': 0,                 # No crop layers
        'min_mask_region_area': 300,         # Filter more small regions - was 50
        'use_m2m': False                    # Skip refinement for speed
    },
    
    # Tumor segmentation focused
    'tumor_detection': {
        **BASE_BRAIN_CONFIG,
        'points_per_side': 32,
        'pred_iou_thresh': 0.65,            # Good balance for pathology
        'stability_score_thresh': 0.75,     # Stable boundaries important
        'crop_n_layers': 1,                 # Multi-scale for complex shapes
        'min_mask_region_area': 30,         # Capture small tumors
        'box_nms_thresh': 0.65,             # Less aggressive NMS for overlapping
        'use_m2m': True                     # Refinement for better boundaries
    },
    
    # Skull stripping focused
    'skull_stripping': {
        **BASE_BRAIN_CONFIG,
        'points_per_side': 28,              # Moderate resolution sufficient
        'pred_iou_thresh': 0.75,            # Higher confidence for large structures
        'stability_score_thresh': 0.85,     # Stable boundaries crucial
        'crop_n_layers': 0,                 # Single scale sufficient
        'min_mask_region_area': 100,        # Filter small artifacts
        'use_m2m': True                     # Smooth boundaries
    },
    
    # Mammography (MG) focused - optimized for small masses and white spots
    'mammography': {
        **BASE_BRAIN_CONFIG,
        'points_per_side': 40,              # Safer for 16GB VRAM, still high resolution
        'points_per_batch': 96,             # Larger batch for efficiency
        'pred_iou_thresh': 0.50,            # Very permissive for tiny masses
        'stability_score_thresh': 0.60,     # Lower for subtle features
        'crop_n_layers': 1,                 # Single scale for memory safety
        'min_mask_region_area': 5,          # Capture very small masses (critical!)
        'box_nms_thresh': 0.5,              # Less aggressive NMS for clustered masses
        'use_m2m': True,                    # Refinement for precise boundaries
        'multimask_output': True            # Multiple candidates for ambiguous masses
    }
}

# Utility function to get configuration
def get_brain_config(config_name='balanced'):
    """
    Get a brain segmentation configuration by name.
    
    Args:
        config_name (str): Configuration name from BRAIN_CONFIGS keys
        
    Returns:
        dict: Configuration parameters for SAM2AutomaticMaskGenerator
        
    Available configurations:
        - 'high_detail': Maximum quality, slower processing
        - 'small_structures': Optimized for ventricles, eyes, etc.
        - 'balanced': Good balance of speed and quality (default)
        - 'fast': Faster processing, reasonable quality
        - 'tumor_detection': Optimized for pathology detection
        - 'skull_stripping': Optimized for brain extraction
    """
    if config_name not in BRAIN_CONFIGS:
        available = list(BRAIN_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return BRAIN_CONFIGS[config_name].copy()

# Configuration descriptions for documentation
CONFIG_DESCRIPTIONS = {
    'high_detail': "Maximum quality configuration for research and detailed analysis",
    'small_structures': "Optimized for detecting small anatomical structures like ventricles and eyes",
    'balanced': "Recommended default configuration with good speed/quality balance",
    'fast': "Faster processing with reasonable quality for quick analysis",
    'tumor_detection': "Specialized for pathology and abnormal tissue detection",
    'skull_stripping': "Optimized for brain extraction and skull removal",
    'mammography': "Optimized for mammography - detects small masses and microcalcifications with minimal area filtering"
}

def print_config_info():
    """Print information about available configurations."""
    print("Available SAM2 Brain Segmentation Configurations:")
    print("=" * 60)
    for name, description in CONFIG_DESCRIPTIONS.items():
        config = BRAIN_CONFIGS[name]
        print(f"\n{name.upper()}:")
        print(f"  Description: {description}")
        print(f"  Points per side: {config['points_per_side']}")
        print(f"  IoU threshold: {config['pred_iou_thresh']}")
        print(f"  Min area: {config['min_mask_region_area']}")
        print(f"  Use M2M: {config['use_m2m']}")
        print(f"  Crop layers: {config['crop_n_layers']}")

if __name__ == "__main__":
    print_config_info()
