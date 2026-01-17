"""
XAI (Explainable AI) Module for SegMed-Pro

This module provides explainability features for:
- MedSAM2 segmentation (attention visualization, uncertainty maps)
- VLM models (confidence scoring, visual grounding)

Author: SegMed-Pro Team
Version: 1.0
"""

from .config import xai_config, XAIConfig

__all__ = ['xai_config', 'XAIConfig']
__version__ = '1.0.0'
