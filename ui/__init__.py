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
from .handlers import DataLoadingHandlers, ConversionHandlers
from .plot_handlers import PlotViewerHandlers, PlotToolHandlers
from .image_handlers import ImageViewerHandlers, ImagePlotToolHandlers
from .segmentation_handlers import SegmentationHandlers
from .label_manager_tab_new import create_label_manager_tab
from .label_manager_handlers import LabelManagerHandlers

__all__ = [
    'AppState',
    'create_viewer_tab',
    'create_conversion_tab', 
    'DataLoadingHandlers',
    'PlotViewerHandlers',
    'ImageViewerHandlers',
    'PlotToolHandlers',
    'ImagePlotToolHandlers',
    'ConversionHandlers',
    'SegmentationHandlers',
    'create_label_manager_tab',
    'LabelManagerHandlers'
]
