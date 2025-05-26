"""
SegMed-Pro Viewer Tab UI Components

This module contains the UI layout for the main viewer tab,
including data loading, visualization, and segmentation tools.
"""

import gradio as gr
import os


def create_data_loading_section() -> tuple:
    """Create the data loading section of the UI"""
    with gr.Column(scale=1):
        # Data Loading Section
        file_input = gr.File(
            label="Load File (DICOM, NIFTI, MAT)",
            file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
        )
        dir_input = gr.Textbox(label="Enter directory path containing DICOM files")
        load_btn = gr.Button("Load Data")
        reset_dir_btn = gr.Button("Reset Directory")
        
        # File Browser Section (when directory is loaded)
        file_browser = gr.Dropdown(label="Available Files", choices=[], interactive=True)
        
        # Metadata Display
        metadata_display = gr.JSON(label="Metadata")
        
        # Error display
        error_display = gr.Textbox(label="Status/Errors", interactive=False)

        # Window/Level controls
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
        
        # Debug button
        debug_btn = gr.Button("Debug Selected File")
    
    return (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
            metadata_display, error_display, window_level, window_width, 
            apply_window_btn, debug_btn)


def create_segmentation_tools_section() -> tuple:
    """Create the segmentation tools section"""
    # Segmentation Tools in the middle column
    gr.Markdown("### Segmentation Tools")
    with gr.Row():
        with gr.Column(scale=1):
            segmentation_file = gr.File(
                label="Load Segmentation (NIfTI .nii.gz)", 
                file_types=[".nii", ".nii.gz"]
            )
            load_seg_btn = gr.Button("Load Segmentation")
            seg_opacity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.1, 
                label="Segmentation Opacity"
            )
        with gr.Column(scale=1):
            label_file = gr.File(
                label="Load Label File (.label)", 
                file_types=[".label"]
            )
            clear_seg_btn = gr.Button("Clear Segmentation")
    
    with gr.Row():
        convert_to_shapes_btn = gr.Button("Convert to Editable Shapes")
    
    with gr.Row():
        seg_status = gr.Textbox(
            label="Segmentation Status", 
            value="No segmentation loaded"
        )
    
    return (segmentation_file, load_seg_btn, seg_opacity, label_file, 
            clear_seg_btn, convert_to_shapes_btn, seg_status)


def create_plot_tools_section() -> tuple:
    """Create the plot tools section"""
    with gr.Column(scale=1):
        # Third column: Plot tool buttons (vertical stack)
        gr.Markdown("## Plot Tools")
        draw_circle_btn = gr.Button("Draw Circle")
        draw_rect_btn = gr.Button("Draw Rectangle")
        draw_line_btn = gr.Button("Draw Line")
        draw_openpath_btn = gr.Button("Draw Open Path")
        draw_closedpath_btn = gr.Button("Draw Closed Path")
        erase_shape_btn = gr.Button("Erase Shape")
        pan_btn = gr.Button("Pan")
        zoom_btn = gr.Button("Zoom")
        reset_btn = gr.Button("Reset")
    
    return (draw_circle_btn, draw_rect_btn, draw_line_btn, draw_openpath_btn,
            draw_closedpath_btn, erase_shape_btn, pan_btn, zoom_btn, reset_btn)


def create_viewer_tab() -> tuple:
    """Create the complete viewer tab layout"""
    with gr.TabItem("Viewer"):
        with gr.Row():
            # Left column: Data loading and controls
            data_loading_components = create_data_loading_section()
            
            # Middle column: Visualization and Segmentation
            with gr.Column(scale=4):
                # Main Visualization Area
                with gr.Row():
                    view_selector = gr.Radio(
                        choices=["Axial", "Sagittal", "Coronal"], 
                        value="Axial", 
                        label="View Orientation"
                    )
                
                with gr.Row():
                    image_plot = gr.Plot(label="Image Plot", show_label=True)
                
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1, 
                        label="Slice Navigation", visible=True
                    )
                    next_btn = gr.Button("Next")
                
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=False)
                
                # Segmentation tools integrated in the middle column
                seg_components = create_segmentation_tools_section()
            
            # Right column: Plot tools
            plot_tools_components = create_plot_tools_section()
    
    # Return all components in a structured way
    viz_components = (view_selector, image_plot, prev_btn, slice_slider, next_btn, 
                     slice_text, crosshair_info)
    
    return {
        'data_loading': data_loading_components,
        'visualization': viz_components,
        'plot_tools': plot_tools_components,
        'segmentation': seg_components
    }
