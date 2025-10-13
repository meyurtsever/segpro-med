"""
SegMed-Pro Viewer Tab UI Components

This module contains the UI layout for the main viewer tab,
including data loading, visualization, and segmentation tools.
"""

import gradio as gr
import os
from gradio_image_annotation import image_annotator


def update_image_annotator_labels(image_annotator_component, label_list, label_colors):
    """
    Update the image_annotator component with new labels and colors from a loaded .label file
    
    Args:
        image_annotator_component: The image_annotator gradio component
        label_list: List of label names from the .label file
        label_colors: List of RGB tuples corresponding to the labels
    
    Returns:
        Updated image_annotator component with new labels and colors
    """
    try:
        # Update the component's configuration
        updated_component = gr.update(
            label_list=label_list,
            label_colors=label_colors
        )
        return updated_component
    except Exception as e:
        print(f"Error updating image_annotator labels: {e}")
        return gr.update()


def prepare_labels_for_annotator(labelmap, colormap):
    """
    Convert labelmap and colormap from .label file format to image_annotator format
    
    Args:
        labelmap: Dict with label indices as keys and label names as values
        colormap: Dict with label indices as keys and [R,G,B] lists as values
    
    Returns:
        tuple: (label_list, label_colors) formatted for image_annotator
    """
    if not labelmap or not colormap:
        # Return default labels if no custom labels loaded
        return (
            ["Normal Tissue", "Tumor", "Organ", "Lesion", "ROI", "Other"],
            [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        )
    
    # Sort by label index to maintain consistent order
    sorted_indices = sorted(labelmap.keys())
    
    label_list = []
    label_colors = []
    
    for idx in sorted_indices:
        if idx == 0:  # Skip background label typically
            continue
            
        label_name = labelmap.get(idx, f"Label_{idx}")
        label_list.append(label_name)
        
        # Convert [R,G,B] list to (R,G,B) tuple
        color_rgb = colormap.get(idx, [128, 128, 128])  # Default gray if color not found
        if isinstance(color_rgb, list) and len(color_rgb) >= 3:
            label_colors.append((color_rgb[0], color_rgb[1], color_rgb[2]))
        else:
            label_colors.append((128, 128, 128))  # Default gray
    
    # Ensure we have at least one label
    if not label_list:
        label_list = ["Label_1"]
        label_colors = [(255, 0, 0)]
    
    return label_list, label_colors


def create_data_loading_section() -> tuple:
    """Create the data loading section of the UI"""
    with gr.Column(scale=1):
        # Data Loading Section
        file_input = gr.File(
            label="Load File (DICOM, NIFTI, MAT)",
            file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
        )
        dir_input = gr.Textbox(label="Enter directory path containing DICOM files",
                               value=r"E:\Gazi\TR_TBP_Anonymised_enc\Anonymised\500 MR\NORMAL\normal (50)\flair")
        load_btn = gr.Button("Load Data")
        reset_dir_btn = gr.Button("Reset Directory")
        
        # Patient Retrieval System Section
        with gr.Accordion("Retrieval System", open=True):
            with gr.Row():
                patient_search_input = gr.Textbox(
                    label="Search Patients",
                    placeholder="Type to search for patients (e.g., 'hgg', 'lgg', etc.)",
                    interactive=True,
                    scale=4
                )
                clear_search_btn = gr.Button("Clear", scale=1, size="sm")
            patient_dropdown = gr.Dropdown(
                label="Matching Patients",
                choices=[],
                value=None,
                interactive=True,
                allow_custom_value=True
            )
            # Manual Segmentation Controls for Retrieval System
            with gr.Row():
                load_retrieval_seg_btn = gr.Button("🔘 Load Segmentation", variant="secondary", scale=1)
                clear_retrieval_seg_btn = gr.Button("🔘 Clear Overlays", variant="secondary", scale=1)
        
        # File Browser Section (when directory is loaded)
        file_browser = gr.Dropdown(label="Available Files", choices=[], interactive=True)
        
        # Metadata Display
        metadata_display = gr.JSON(label="Metadata", open=False)
        
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
    
    return (file_input, dir_input, load_btn, reset_dir_btn, patient_search_input, 
            clear_search_btn, patient_dropdown, load_retrieval_seg_btn, clear_retrieval_seg_btn,
            file_browser, metadata_display, error_display, 
            window_level, window_width, apply_window_btn, debug_btn)


def create_segmentation_tools_section() -> tuple:
    """Create the segmentation tools section"""
    # Segmentation Tools in the middle column    gr.Markdown("### Segmentation Tools")
    with gr.Row():
        with gr.Column(scale=1):
            segmentation_file = gr.File(
                label="Load Segmentation (NIfTI .nii.gz)", 
                file_types=[".nii", ".nii.gz"]
            )
            load_seg_btn = gr.Button("Load Segmentation", visible=False)
            
            seg_opacity = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.1, 
                label="Segmentation Opacity", visible=False)
            
        with gr.Column(scale=1):
            label_file = gr.File(
                label="Load Label File (.label)", 
                file_types=[".label"]
            )
            clear_seg_btn = gr.Button("Clear Segmentation", visible=False)    
    with gr.Row():
        seg_status = gr.Textbox(
            label="Segmentation Status", 
            value="No segmentation loaded"
        )
    
    return (segmentation_file, load_seg_btn, seg_opacity, label_file, 
            clear_seg_btn, seg_status)


def create_viewer_controls_section() -> tuple:
    """Create viewer-specific controls for the image_annotator"""
    with gr.Column(scale=1):
        # Viewer Controls Section
        gr.Markdown("## Viewer Controls")
        
        # Mammography Zoom Controls (for Direct Pixel Extractor)
        with gr.Accordion("Mammography Zoom", open=True):
            gr.Markdown("**Lossless Zoom Controls for Mammography Images**")
            with gr.Row():
                zoom_in_btn = gr.Button("🔍 Zoom In", size="sm")
                zoom_out_btn = gr.Button("🔍 Zoom Out", size="sm")
            zoom_reset_btn = gr.Button("🏠 Reset View", variant="secondary")
            zoom_status = gr.Textbox(
                label="Zoom Status",
                interactive=False,
                value="No zoom active"
            )
        
        # Enhanced Export Options
        with gr.Accordion("Export Options", open=False):
            # Export format selection
            export_format = gr.Dropdown(
                label="Export Format",
                choices=["NIfTI (*.nii)", "JSON", "YOLO", "CSV"],
                value="JSON",
                info="Select the export format for annotations"
            )
            
            # Include Overlays option with explanation
            # gr.Markdown("*If selected, PNG images will be saved showing the visible overlays on each exported slice.*")
            include_overlays = gr.Checkbox(
                label="Include Overlays (PNG)",
                value=False,
                info="Export PNG images with visible overlays"
            )
            
            # Export buttons side by side
            with gr.Row():
                export_single_btn = gr.Button("Export Single Slice", variant="primary")
                export_all_btn = gr.Button("Export All", variant="secondary")
            
            # Output directory selection
            output_dir = gr.Textbox(
                label="Output Directory",
                value="exports",
                info="Directory where exported files will be saved"
            )
            
            # Export status
            export_status = gr.Textbox(
                label="Export Status",
                interactive=False,
                value="Ready to export"
            )
    
    return (export_format, include_overlays, export_single_btn, export_all_btn, 
            output_dir, export_status)


def create_viewer_tab() -> tuple:
    """Create the complete viewer tab layout"""
    with gr.TabItem("Viewer", id=0):
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
                    # Create image_annotator with dynamic labels
                    image_display = create_dynamic_image_annotator()
                
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1, 
                        label="Slice Navigation", visible=True
                    )
                    next_btn = gr.Button("Next")
                
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False, visible=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=False, visible=False)
                
                # Segmentation tools integrated in the middle column
                seg_components = create_segmentation_tools_section()
            
            # Right column: Viewer controls (replacing plot tools)
            viewer_controls_components = create_viewer_controls_section()
    
    # Return all components in a structured way
    viz_components = (view_selector, image_display, prev_btn, slice_slider, next_btn, 
                     slice_text, crosshair_info)
    
    return {
        'data_loading': data_loading_components,
        'visualization': viz_components,
        'viewer_controls': viewer_controls_components,
        'segmentation': seg_components
    }


def create_dynamic_image_annotator():
    """Create an image_annotator component that can be updated with new labels"""
    return image_annotator(
        value=None,
        label="Medical Image Viewer", 
        label_list=["Normal Tissue", "Tumor", "Organ", "Lesion", "ROI", "Other"],
        label_colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
        box_min_size=10,
        handle_size=8,
        box_thickness=2,
        box_selected_thickness=3,
        boxes_alpha=0.7,
        height=600,
        width=1200,
        interactive=True,
        show_label=True,
        show_download_button=True,
        show_clear_button=True,
        show_remove_button=True,
        use_default_label=False,
        handles_cursor=True,
        image_type="numpy",
        single_box=False,
        disable_edit_boxes=False,
        shape_creation_mode="drag",
    )
