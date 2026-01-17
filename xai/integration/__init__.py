"""
XAI Integration Module

Integration hooks and bridges for connecting XAI features
with existing model handlers.
"""

from .medsam2_hooks import (
    XAIMedSAM2Integration,
    get_xai_integration,
    XAI_AVAILABLE,
)

__all__ = [
    'XAIMedSAM2Integration',
    'get_xai_integration',
    'XAI_AVAILABLE',
]
