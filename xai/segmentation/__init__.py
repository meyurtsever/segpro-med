"""
XAI Segmentation Module

Explainability features for segmentation models (MedSAM2).

PRIMARY: LayerCAM - Fast, feature-based XAI for Hiera backbone
"""

from .attention_extractor import SAM2AttentionExtractor, AttentionMaps
from .mask_confidence import (
    MaskConfidenceExtractor,
    get_mask_confidence_extractor,
    compute_mask_confidence,
    compute_boundary_uncertainty,
    create_focus_map,
    create_prompt_influence_map,
)

# LayerCAM - PRIMARY XAI METHOD
from .layercam_extractor import (
    LayerCAMExtractor,
    get_layercam_extractor,
    compute_layercam_for_sam2,
)

# GradCAM - COMMENTED OUT
# from .gradcam_extractor import (
#     SAM2GradCAMExtractor,
#     get_gradcam_extractor,
#     compute_gradcam_for_sam2,
#     GRADCAM_AVAILABLE,
# )

__all__ = [
    'SAM2AttentionExtractor', 
    'AttentionMaps',
    'MaskConfidenceExtractor',
    'get_mask_confidence_extractor',
    'compute_mask_confidence',
    'compute_boundary_uncertainty',
    'create_focus_map',
    'create_prompt_influence_map',
    # LayerCAM exports
    'LayerCAMExtractor',
    'get_layercam_extractor',
    'compute_layercam_for_sam2',
    # GradCAM exports - COMMENTED OUT
    # 'SAM2GradCAMExtractor',
    # 'get_gradcam_extractor',
    # 'compute_gradcam_for_sam2',
    # 'GRADCAM_AVAILABLE',
]
