"""
SegMed-Pro UI Components Package

This package contains all UI-related modules for the SegMed-Pro application,
organized following Gradio best practices for modular application structure.

Modules:
- state: Application state management
- viewer_tab: Main viewer tab UI components
- conversion_tab: Format conversion tab UI components
- handlers: Core event handlers for data loading and viewer operations
- segmentation_handlers: Segmentation-specific event handlers
"""

from .state import AppState
from .viewer_tab import create_viewer_tab
from .conversion_tab import create_conversion_tab
from .handlers import DataLoadingHandlers, ViewerHandlers, PlotToolHandlers, ConversionHandlers
from .segmentation_handlers import SegmentationHandlers

__all__ = [
    'AppState',
    'create_viewer_tab',
    'create_conversion_tab', 
    'DataLoadingHandlers',
    'ViewerHandlers',
    'PlotToolHandlers',
    'ConversionHandlers',
    'SegmentationHandlers'
]
