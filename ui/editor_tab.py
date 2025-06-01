"""
SegMed-Pro Editor Tab UI Components

This module contains the UI layout for the Editor tab,
including data loading, visualization (without segmentation tools),
and AI annotation tools.
"""

import gradio as gr
import sys
import os
from .viewer_tab import create_data_loading_section

# Add custom_components path to sys.path if not already there
custom_components_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_components")
if custom_components_path not in sys.path:
    sys.path.append(custom_components_path)

# Import our custom PlotlyBridge component
try:
    from custom_components.plotly_bridge import PlotlyBridge
    HAS_PLOTLY_BRIDGE = True
except ImportError:
    HAS_PLOTLY_BRIDGE = False

def create_editor_tab() -> dict:
    """Create the complete editor tab layout"""
    with gr.TabItem("Editor"):
        with gr.Row():
            # Column 1: All components (data loading, controls, etc.) except Metadata, Status/Errors, and Image Adjustments moved as specified
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Load File (DICOM, NIFTI, MAT)",
                    file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
                )
                dir_input = gr.Textbox(label="Enter directory path containing DICOM files")
                load_btn = gr.Button("Load Data")
                reset_dir_btn = gr.Button("Reset Directory")
                file_browser = gr.Dropdown(label="Available Files", choices=[], interactive=True)
                # Image Adjustments moved here
                with gr.Accordion("Image Adjustments", open=False):
                    window_level = gr.Slider(
                        minimum=0, maximum=4000, value=500, step=10,
                        label="Window Level (Center)"
                    )
                    window_width = gr.Slider(
                        minimum=1, maximum=4000, value=1000, step=10,
                        label="Window Width"
                    )
                    apply_window_btn = gr.Button("Apply Window/Level")               
                    debug_btn = gr.Button("Debug Selected File")
                    
                # For handler compatibility, include error_display and metadata_display in col1 (even if rendered in col2)
                # They will be created in col2 below
                col1 = (file_input, dir_input, load_btn, reset_dir_btn, file_browser, None, None, window_level, window_width, apply_window_btn, debug_btn)
            
            # Column 2: View orientation and plotly on top, then navigation controls, then Status/Errors and Metadata below
            with gr.Column(scale=4):
                # Main Visualization Area (top)
                with gr.Row():
                    view_selector = gr.Radio(
                        choices=["Axial", "Sagittal", "Coronal"],
                        value="Axial",
                        label="View Orientation"
                    )
                
                with gr.Row():
                    image_plot = gr.Plot(label="Image Plot", show_label=True, elem_id="plotly_image_plot")
                    
                    # Add the PlotlyBridge component if available
                    if HAS_PLOTLY_BRIDGE:
                        plotly_bridge = PlotlyBridge(plot_id="plotly_image_plot", sync_crosshair=True)
                    else:
                        plotly_bridge = None
                # Navigation controls (middle)
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        label="Slice Navigation", visible=True                    )
                    next_btn = gr.Button("Next")
                
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=True)
                    # Add a dedicated coordinates display component
                    coords_display = gr.Textbox(
                        label="Coordinates", 
                        interactive=False,
                        value="X: -- Y: --",
                        elem_id="coordinates-display"
                    )
                
                # Status/Errors and Metadata (bottom)
                error_display = gr.Textbox(label="Status/Errors", interactive=False)
                with gr.Accordion("Metadata", open=False):
                    metadata_display = gr.JSON(label=None, visible=True)
                col2 = (error_display, metadata_display, view_selector, image_plot, prev_btn, slice_slider, next_btn, slice_text, crosshair_info, plotly_bridge, coords_display)

            # Column 3: Annotate with AI Models
            with gr.Column(scale=1):
                gr.Markdown("## Annotate with AI Models")
                ai_model_selector = gr.Dropdown(
                    label="Select AI Model",
                    choices=["UNet (dummy)", "DeepLabV3 (dummy)", "SAM (dummy)", "Other (dummy)"]
                )
                annotate_btn = gr.Button("Run Annotation (dummy)")
                col3 = (ai_model_selector, annotate_btn)

    # Return all components in a structured way
    return {
        'data_loading': col1,
        'visualization': col2,
        'ai_tools': col3,
        'plotly_bridge': plotly_bridge  # Ensure plotly_bridge is directly accessible
    }
