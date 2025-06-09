"""
SegMed-Pro Custom Annotator Tab UI Components

This module contains the UI layout for the custom image annotation tab,
using the gradio-image-annotation plugin for advanced annotation features
while integrating medical image loading capabilities from the Editor tab.
"""

import gradio as gr
from gradio_image_annotation import image_annotator


def create_custom_annotator_tab():
    """Create the Custom Annotator tab with gradio-image-annotation plugin and medical image support"""
    
    with gr.Tab("Custom Annotator"):
        gr.Markdown("## Custom Image Annotator")
        gr.Markdown("Advanced image annotation tool with bounding boxes, labels, and medical image support.")
        
        with gr.Row():
            # Column 1: Data Loading and Controls (adapted from Editor tab)
            with gr.Column(scale=1):
                # Medical Data Loading Section (from Editor tab)
                gr.Markdown("### Medical Data Loading")
                file_input = gr.File(
                    label="Load File (DICOM, NIFTI, MAT)",
                    file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
                )
                dir_input = gr.Textbox(label="Enter directory path containing DICOM files")
                load_btn = gr.Button("Load Data", variant="primary")
                reset_dir_btn = gr.Button("Reset Directory")
                file_browser = gr.Dropdown(label="Available Files", choices=[], interactive=True)
                
                # Image Adjustments
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
                
                # Status and Debug
                error_display = gr.Textbox(label="Status/Errors", interactive=False)
                debug_btn = gr.Button("Debug Selected File")
            
            # Column 2: Main Annotator (using image_annotator plugin)
            with gr.Column(scale=2):
                # View Orientation Selector (moved above annotator)
                view_selector = gr.Radio(
                    choices=["Axial", "Sagittal", "Coronal"],
                    value="Axial",
                    label="View Orientation"
                )
                
                # Main annotator component
                annotator = image_annotator(
                    value=None,
                    label="Medical Image Annotator",
                    label_list=["Tumor", "Normal Tissue", "Organ", "Lesion", "ROI", "Other"],
                    label_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
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
                    use_default_label=True,
                    image_type="numpy"  # Important for medical images
                )
                
                # Navigation Controls (moved below annotator)
                gr.Markdown("### Navigation Controls")
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        label="Slice Navigation", visible=True
                    )
                    next_btn = gr.Button("Next")
                
                slice_text = gr.Textbox(label="Slice", interactive=False)
            
            # Column 3: Annotation Controls and Management
            with gr.Column(scale=1):
                # Control panel
                with gr.Group():
                    gr.Markdown("### Annotation Controls")
                    
                    # Annotation settings
                    with gr.Accordion("Annotation Settings", open=True):
                        opacity_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                            label="Box Opacity"
                        )
                        thickness_slider = gr.Slider(
                            minimum=1, maximum=5, value=2, step=1,
                            label="Box Thickness"
                        )
                        min_size_slider = gr.Slider(
                            minimum=5, maximum=50, value=10, step=5,
                            label="Minimum Box Size"
                        )
                    
                    # Label management
                    with gr.Accordion("Label Management", open=False):
                        new_label_input = gr.Textbox(
                            label="Add New Label",
                            placeholder="Enter label name..."
                        )
                        add_label_btn = gr.Button("Add Label")                        # Label list display
                        current_labels = gr.JSON(
                            label="Current Labels",
                            value=["Tumor", "Normal Tissue", "Organ", "Lesion", "ROI", "Other"]
                        )
                    
                    # Export options
                    with gr.Accordion("Export Options", open=False):
                        export_format = gr.Dropdown(
                            choices=["JSON", "COCO", "YOLO", "CSV"],
                            value="JSON",
                            label="Export Format"
                        )
                        export_btn = gr.Button("Export Annotations")
                        
                        # Export output
                        export_output = gr.File(label="Download Annotations")
                  # Statistics panel
                with gr.Group():
                    gr.Markdown("### Annotation Statistics")
                    stats_display = gr.JSON(
                        label="Current Statistics",
                        value={"total_boxes": 0, "labels_used": {}}
                    )
                    refresh_stats_btn = gr.Button("Refresh Statistics")
                
                # Metadata display
                with gr.Accordion("Medical Metadata", open=False):
                    metadata_display = gr.JSON(label="DICOM/Medical Metadata", value={})
        
        # Bottom row for batch operations
        with gr.Row():
            with gr.Accordion("Batch Operations", open=False):
                batch_process_btn = gr.Button("Process Multiple Images")
                batch_status = gr.Textbox(label="Batch Status", interactive=False)      # Component dictionary for easy access - structured like Editor tab
    components = {
        'data_loading': (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
                        metadata_display, error_display, window_level, window_width, 
                        apply_window_btn, debug_btn),
        'visualization': (view_selector, prev_btn, slice_slider, next_btn, slice_text),
        'annotator': {
            'main': annotator,
            'opacity_slider': opacity_slider,
            'thickness_slider': thickness_slider,
            'min_size_slider': min_size_slider,
            'new_label_input': new_label_input,
            'add_label_btn': add_label_btn,
            'current_labels': current_labels,
            'export_format': export_format,
            'export_btn': export_btn,
            'export_output': export_output,
            'stats_display': stats_display,
            'refresh_stats_btn': refresh_stats_btn,
            'batch_process_btn': batch_process_btn,
            'batch_status': batch_status
        }
    }
    
    return components
