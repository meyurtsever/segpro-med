"""
SegMed-Pro Editor Tab UI Components

This module contains the UI layout for the Editor tab,
including data loading, visualization with advanced annotation capabilities,
and AI annotation tools using gradio_image_annotation.
"""

import gradio as gr
from gradio_image_annotation import image_annotator
from .viewer_tab import create_data_loading_section

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
                dir_input = gr.Textbox(
                    label="Enter directory path containing DICOM files",
                    value=r"C:\Users\Yurtsever\Downloads\segpro-med\cvm_48_t1"
                )
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

            # Column 2: View orientation and image annotator on top, then navigation controls, then Status/Errors and Metadata below
            with gr.Column(scale=4):
                # Main Visualization Area (top)
                with gr.Row():
                    view_selector = gr.Radio(
                        choices=["Axial", "Sagittal", "Coronal"],
                        value="Axial",
                        label="View Orientation"
                    )
                with gr.Row():                    
                    image_display = image_annotator(
                        value=None,
                        label="Medical Image", 
                        label_list=["Normal Tissue", "Tumor", "Organ", "Lesion", "ROI", "Other"],
                        label_colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
                        box_min_size=10,
                        handle_size=8,
                        box_thickness=2,
                        box_selected_thickness=3,
                        boxes_alpha=0.7,
                        height=600,
                        width=800,
                        interactive=True,
                        show_label=True,
                        show_download_button=True,
                        show_clear_button=True,
                        show_remove_button=True,
                        use_default_label=False, # Do not use default label, opens modal for custom labels
                        handles_cursor=True,  # Enable cursor handling for drag mode
                        image_type="numpy",  # Important for medical images
                        single_box=False,  # Allow multiple boxes
                        disable_edit_boxes=False,  # Allow editing boxes
                        shape_creation_mode="drag",
                    )
                # Navigation controls (middle)
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        label="Slice Navigation", visible=True
                    )
                    next_btn = gr.Button("Next")
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=True)
                # Status/Errors and Metadata (bottom)
                error_display = gr.Textbox(label="Status/Errors", interactive=False)
                with gr.Accordion("Metadata", open=False):
                    metadata_display = gr.JSON(label=None, visible=True)
                col2 = (error_display, metadata_display, view_selector, image_display, prev_btn, slice_slider, next_btn, slice_text, crosshair_info)
              # Column 3: Annotate with AI Models
            with gr.Column(scale=1):
                # Coordinate Selection for MEDSAM2
                gr.Markdown("### Point Selection")
                # Prompt type selection
                with gr.Row():
                    point_prompt_checkbox = gr.Checkbox(
                        label="Point-based Prompt",
                        value=False,
                        info="Use point coordinates for segmentation"
                    )
                    box_prompt_checkbox = gr.Checkbox(
                        label="Box-based Prompt", 
                        value=False,
                        info="Use bounding boxes for segmentation"
                    )
                coordinates_text = gr.Textbox(
                    label="Selected Coordinates (x,y)",
                    value="",
                    interactive=False,
                    info="Click on the image to select coordinates",
                    visible=False  # Initially hidden until point-based is selected
                )
                clear_coords_btn = gr.Button("Clear Coordinates & Prompts")
                clear_overlays_btn = gr.Button(
                    "Clear Annotation Overlays", 
                    variant="stop",  # Makes the button red
                    visible=False    # Hidden by default, shown after successful annotation
                )
                gr.Markdown("## Annotate with AI Models")
                ai_model_selector = gr.Dropdown(
                    label="Select AI Model",
                    choices=["MEDSAM2", "UNet (dummy)", "DeepLabV3 (dummy)", "SAM (dummy)", "Other (dummy)"],
                    value="MEDSAM2"
                )
                
                # Processing Mode Selection
                processing_mode = gr.Radio(
                    choices=["Single Slice", "All Records"],
                    value="Single Slice",
                    label="Processing Mode",
                    info="Single Slice: Process only current slice. All Records: Process entire volume with quality thresholding."
                )
                
                # Score Threshold for All Records mode
                score_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="Score Threshold (All Records mode)",
                    info="Minimum average score required to display annotations. Lower scores may indicate poor quality segmentations.",
                    visible=False  # Initially hidden, shown when "All Records" is selected
                )
                
                # MEDSAM2 specific controls
                with gr.Group(visible=True) as medsam2_controls:
                    with gr.Accordion("Annotation Settings", open=False):
                        output_dir = gr.Textbox(
                            label="Output Directory",
                            value="brain_target_results",
                            info="Directory to save annotation results"
                        )
                        save_visualizations = gr.Checkbox(
                            label="Save Visualizations",
                            value=True,
                            info="Save visualization images along with segmentation"
                        )
                        device_selector = gr.Dropdown(
                            label="Device",
                            choices=["cpu", "cuda"],
                            value="cpu",
                            info="Processing device for MEDSAM2"
                        )
                
                # Automatic Brain Annotation Section
                gr.Markdown("### Automatic Brain Detection")
                with gr.Row():
                    auto_brain_annotate_btn = gr.Button(
                        "🧠 Auto-Annotate Brain Structures", 
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Accordion("Auto-Annotation Info", open=False):
                    gr.Markdown("""
                    **Automatic Brain Structure Detection:**
                    - Automatically detects brain regions and eye structures
                    - Uses anatomical priors and image processing
                    - Generates bounding box prompts for MEDSAM2
                    - Works with "All Records" mode to process entire volumes
                    - No manual coordinate selection required
                    """)
                
                annotate_btn = gr.Button("Run MEDSAM2 Annotation (Manual Prompts)")
                annotation_status = gr.Textbox(
                    label="Annotation Status", 
                    interactive=False,
                    value="Ready to annotate"
                )
                
                col3 = (point_prompt_checkbox, box_prompt_checkbox, coordinates_text, clear_coords_btn, clear_overlays_btn, ai_model_selector, 
                        processing_mode, score_threshold,
                        output_dir, save_visualizations, device_selector, 
                        auto_brain_annotate_btn, annotate_btn, annotation_status)

    # Return all components in a structured way
    return {
        'data_loading': col1,
        'visualization': col2,
        'ai_tools': col3
    }
