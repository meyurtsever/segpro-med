"""
XAI for Vision-Language Models (VLMs)

Provides explainability tools for multimodal models like MedGemma.

Supports multiple XAI methods:

1. Attention-Based (FAST - Recommended):
   - AttentionVisualGrounding: Uses model's attention weights directly
   - No backward pass needed, uses cached generation outputs
   - Much faster than gradient-based methods

2. Gradient-Based (HistoLens-inspired):
   - GradCAM: Standard gradient-weighted class activation mapping
   - GradCAM++: Improved with alpha-weighted positive gradients
   - HiResCAM: Pixel-precise localization
   - Guided GradCAM: Combines guided backprop with CAM for fine-grained detail
"""

from .visual_grounding import (
    MedGemmaVisualGrounding,
    create_visual_grounding_overlay,
    GradCAM,
    GradCAMPlusPlus,
    GradCAMPP,
    HiResCAM,
    GuidedBackprop,
    BaseCAM,
    XAIMethod
)

from .attention_visual_grounding import (
    AttentionVisualGrounding,
    HybridVisualGrounding
)

__all__ = [
    # Attention-based (FAST)
    'AttentionVisualGrounding',
    'HybridVisualGrounding',
    # Gradient-based
    'MedGemmaVisualGrounding',
    'create_visual_grounding_overlay',
    'GradCAM',
    'GradCAMPlusPlus',
    'GradCAMPP',
    'HiResCAM',
    'GuidedBackprop',
    'BaseCAM',
    'XAIMethod'
]
