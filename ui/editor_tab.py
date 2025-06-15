"""
SegMed-Pro Editor Tab UI Components

This module contains the UI layout for the Editor tab,
including data loading, visualization with advanced annotation capabilities,
AI annotation tools using gradio_image_annotation, and 3D visualization.
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from gradio_image_annotation import image_annotator
from .viewer_tab import create_data_loading_section
import json
import os
import glob
from scipy import ndimage
from skimage import measure
import cv2

def load_annotation_data(output_dir="brain_target_results"):
    """Load annotation data from the results directory"""
    try:
        summary_path = os.path.join(output_dir, "summary.json")
        if not os.path.exists(summary_path):
            return None, "No annotation data found"
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        return summary, None
    except Exception as e:
        return None, f"Error loading annotation data: {str(e)}"

def load_slice_masks(output_dir="brain_target_results", slice_indices=None):
    """Load mask data for specified slices"""
    masks = {}
    masks_dir = os.path.join(output_dir, "masks")
    
    if not os.path.exists(masks_dir):
        return masks
    
    if slice_indices is None:
        return masks
    
    for slice_idx in slice_indices:
        # Try both 4-digit and 3-digit formatting
        mask_file_4digit = os.path.join(masks_dir, f"slice_{slice_idx:04d}_mask.npy")
        mask_file_3digit = os.path.join(masks_dir, f"slice_{slice_idx:03d}_mask.npy")
        
        mask_file = None
        if os.path.exists(mask_file_4digit):
            mask_file = mask_file_4digit
        elif os.path.exists(mask_file_3digit):
            mask_file = mask_file_3digit
        
        if mask_file:
            try:
                mask = np.load(mask_file)
                masks[slice_idx] = mask
                print(f"Loaded mask for slice {slice_idx}: {mask.shape}")
            except Exception as e:
                print(f"Error loading mask for slice {slice_idx}: {e}")
        else:
            print(f"No mask file found for slice {slice_idx}")
    
    return masks

def create_3d_volume_from_masks(masks, slice_spacing=1.0):
    """Create a 3D volume from 2D slice masks"""
    if not masks:
        return None
    
    slice_indices = sorted(masks.keys())
    if len(slice_indices) < 2:
        return None
    
    # Get mask dimensions
    first_mask = masks[slice_indices[0]]
    height, width = first_mask.shape
    
    # Create volume with proper spacing
    min_slice = min(slice_indices)
    max_slice = max(slice_indices)
    volume_depth = max_slice - min_slice + 1
    
    volume = np.zeros((height, width, volume_depth), dtype=np.uint8)
    
    # Fill volume with masks
    for slice_idx in slice_indices:
        z_pos = slice_idx - min_slice
        volume[:, :, z_pos] = (masks[slice_idx] > 0).astype(np.uint8)
    
    # Apply morphological closing to connect nearby regions
    volume = ndimage.binary_closing(volume, structure=np.ones((3, 3, 3))).astype(np.uint8)
    
    return volume

def create_3d_mesh_from_volume(volume, threshold=0.5):
    """Create 3D mesh from volume using marching cubes"""
    if volume is None:
        return None, None, None
    
    try:
        # Use marching cubes to extract isosurface
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=threshold, spacing=(1, 1, 1), method='lewiner'
        )
        
        return verts, faces, normals
    except Exception as e:
        print(f"Error creating 3D mesh: {e}")
        return None, None, None

def create_3d_visualization(output_dir="brain_target_results"):
    """Create 3D visualization of annotated regions"""
    # Load annotation data
    summary, error = load_annotation_data(output_dir)
    if error:
        return create_empty_3d_plot(f"Error: {error}")
    
    # Get slice indices - try from summary first, then auto-detect from available masks
    slice_indices = []
    if summary and 'slice_indices' in summary:
        slice_indices = summary['slice_indices']
    
    # If no slice indices in summary or less than 2, auto-detect from mask files
    if len(slice_indices) < 2:
        masks_dir = os.path.join(output_dir, "masks")
        if os.path.exists(masks_dir):
            import glob
            # Look for mask files with either 3 or 4 digit formatting
            mask_files = glob.glob(os.path.join(masks_dir, "*_mask.npy"))
            detected_indices = []
            for mask_file in mask_files:
                filename = os.path.basename(mask_file)
                # Extract slice number from filename
                if filename.startswith("slice_") and filename.endswith("_mask.npy"):
                    slice_num_str = filename[6:-9]  # Remove "slice_" and "_mask.npy"
                    try:
                        slice_num = int(slice_num_str)
                        detected_indices.append(slice_num)
                    except ValueError:
                        continue
            slice_indices = sorted(detected_indices)
            print(f"Auto-detected slice indices: {slice_indices}")
    
    if len(slice_indices) < 2:
        return create_empty_3d_plot("Need at least 2 annotated slices for 3D visualization")
    
    # Load masks
    masks = load_slice_masks(output_dir, slice_indices)
    if not masks:
        return create_empty_3d_plot("No mask data found")
    
    print(f"Loaded {len(masks)} masks for 3D visualization")
    
    # Create 3D volume
    volume = create_3d_volume_from_masks(masks)
    if volume is None:
        return create_empty_3d_plot("Could not create 3D volume")
    
    print(f"Created 3D volume with shape: {volume.shape}")
    
    # Create mesh
    verts, faces, normals = create_3d_mesh_from_volume(volume)
    if verts is None:
        return create_empty_3d_plot("Could not create 3D mesh")
    
    print(f"Created 3D mesh with {len(verts)} vertices and {len(faces)} faces")
    
    # Create plotly figure
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2],  # Z becomes X for better orientation
            y=verts[:, 1],  # Y stays Y
            z=verts[:, 0],  # X becomes Z
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=verts[:, 2],
            colorscale='viridis',
            opacity=0.8,
            name='Annotated Region'
        )
    ])
      # Update layout with black background
    fig.update_layout(
        title=f"3D Visualization - {len(slice_indices)} Slices",
        scene=dict(
            xaxis_title="Slice Direction",
            yaxis_title="Width",
            zaxis_title="Height",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor="black",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            )        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        width=650,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def create_empty_3d_plot(message="No 3D data available"):
    """Create an empty 3D plot with a message and black background"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="white")
    )
    fig.update_layout(
        title="3D Viewer",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="Z",
            bgcolor="black",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(128,128,128,0.3)",
                title_font=dict(color="white")
            )        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        width=650,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

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
                col1 = (file_input, dir_input, load_btn, reset_dir_btn, file_browser, None, None, window_level, window_width, apply_window_btn, debug_btn)            # Column 2: View orientation and image annotator on top, then navigation controls, then Status/Errors and Metadata below
            with gr.Column(scale=4):
                # Main Visualization Area (top)
                with gr.Row():
                    view_selector = gr.Radio(
                        choices=["Axial", "Sagittal", "Coronal"],
                        value="Axial",
                        label="View Orientation"
                    )                # Image and 3D Viewer side by side - no margins, medical image wider
                with gr.Row(equal_height=True):
                    with gr.Column(scale=7):
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
                            width=650,  # Increased width for better viewing 800
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
                    
                    with gr.Column(scale=6):
                        # 3D Viewer - initially hidden, shown when "All Records" is selected and annotations exist
                        viewer_3d = gr.Plot(
                            value=create_empty_3d_plot("3D Viewer will appear here after 'All Records' annotation"),
                            label="3D Viewer",
                            visible=False  # Initially hidden
                        )
                        
                        # 3D Viewer Controls
                        with gr.Row(visible=False) as viewer_3d_controls:
                            refresh_3d_btn = gr.Button("🔄 Refresh 3D View", size="sm")
                            export_3d_btn = gr.Button("💾 Export 3D", size="sm")# Navigation controls (middle)
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    slice_slider = gr.Slider(
                        minimum=0, maximum=0, value=0, step=1,
                        label="Slice Navigation", visible=True
                    )
                    next_btn = gr.Button("Next")
                with gr.Row():
                    slice_text = gr.Textbox(label="Slice", interactive=False)
                    crosshair_info = gr.Textbox(label="Crosshair", interactive=True)                # Status/Errors and Metadata (bottom)
                error_display = gr.Textbox(label="Status/Errors", interactive=False)
                with gr.Accordion("Metadata", open=False):
                    metadata_display = gr.JSON(label=None, visible=True)
                
                col2 = (error_display, metadata_display, view_selector, image_display, prev_btn, slice_slider, next_btn, slice_text, crosshair_info, viewer_3d, viewer_3d_controls, refresh_3d_btn, export_3d_btn)
            
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
                clear_coords_btn = gr.Button("Clear Coordinates & Prompts", variant="stop",)
                
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
                            value=False,
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

    # Helper functions for 3D viewer interactions
    def toggle_3d_viewer_visibility(processing_mode):
        """Show/hide 3D viewer based on processing mode"""
        if processing_mode == "All Records":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)
    
    def refresh_3d_view(output_dir):
        """Refresh the 3D visualization"""
        try:
            fig = create_3d_visualization(output_dir)
            return fig
        except Exception as e:
            return create_empty_3d_plot(f"Error refreshing view: {str(e)}")
    
    def export_3d_view(output_dir):
        """Export 3D view as HTML"""
        try:
            fig = create_3d_visualization(output_dir)
            export_path = os.path.join(output_dir, "3d_visualization.html")
            fig.write_html(export_path)
            return f"3D view exported to: {export_path}"
        except Exception as e:
            return f"Error exporting 3D view: {str(e)}"

    # Return all components in a structured way
    return {
        'data_loading': col1,
        'visualization': col2,
        'ai_tools': col3,
        '3d_viewer_functions': {
            'toggle_visibility': toggle_3d_viewer_visibility,
            'refresh_view': refresh_3d_view,
            'export_view': export_3d_view
        }
    }
