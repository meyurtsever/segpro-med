"""
XAI Configuration Module

Centralized configuration for all XAI features in SegMed-Pro.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class XAIConfig:
    """Configuration for XAI features"""
    
    # Global toggle
    enabled: bool = True
    
    # Segmentation XAI - MedSAM2
    show_attention_maps: bool = False  # Off by default, user toggles on
    show_uncertainty_maps: bool = False
    show_prompt_influence: bool = False
    attention_layer_names: List[str] = field(default_factory=lambda: [
        'blocks', 'attention', 'attn', 'self_attn', 'cross_attn'
    ])
    uncertainty_threshold: float = 0.3
    
    # VLM XAI - MedGemma, SmolVLM, Med-R1
    show_confidence_scores: bool = False
    show_visual_grounding: bool = False
    show_reasoning_chain: bool = False
    confidence_threshold: float = 0.5
    
    # Performance settings
    cache_explanations: bool = True
    max_attention_layers: int = 4  # Limit number of layers to extract for performance
    
    # Visualization settings
    attention_colormap: str = "jet"  # jet, viridis, hot, plasma
    uncertainty_colormap: str = "RdYlGn_r"  # Red=uncertain, Green=confident
    overlay_alpha: float = 0.4
    heatmap_alpha: float = 0.7  # Increased opacity for better visibility (especially for gradient XAI)
    
    # Attention visualization specific
    attention_normalize: bool = True
    attention_threshold: float = 0.1  # Threshold to mask low attention values
    blur_sigma: float = 2.0  # Gaussian blur sigma for smoothing attention maps
    
    def enable_attention_visualization(self):
        """Enable attention visualization feature"""
        self.show_attention_maps = True
        logger.info("XAI: Attention visualization enabled")
    
    def disable_attention_visualization(self):
        """Disable attention visualization feature"""
        self.show_attention_maps = False
        logger.info("XAI: Attention visualization disabled")
    
    def toggle_attention_visualization(self, value: bool):
        """Toggle attention visualization on/off"""
        self.show_attention_maps = value
        logger.info(f"XAI: Attention visualization {'enabled' if value else 'disabled'}")
        return value


# Global instance - singleton pattern
xai_config = XAIConfig()


def get_xai_config() -> XAIConfig:
    """Get the global XAI configuration instance"""
    return xai_config


def reset_xai_config():
    """Reset XAI config to defaults"""
    global xai_config
    xai_config = XAIConfig()
    return xai_config
