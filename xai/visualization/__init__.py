"""
XAI Visualization Module

Visualization components for XAI features including
heatmap overlays, attention maps, and uncertainty displays.
"""

from .heatmap_overlay import (
    create_attention_heatmap_overlay,
    create_uncertainty_overlay,
    blend_heatmap_with_image,
    create_colorbar_legend
)

__all__ = [
    'create_attention_heatmap_overlay',
    'create_uncertainty_overlay', 
    'blend_heatmap_with_image',
    'create_colorbar_legend'
]
