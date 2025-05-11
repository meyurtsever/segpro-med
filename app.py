import os
import sys
import gradio as gr
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.dicom_utils import load_dicom_series, load_single_dicom, get_dicom_metadata, list_dicoms_in_directory
from utils.nifti_utils import load_nifti_file, nifti_to_mat, nifti_to_png
from utils.visualization import display_slice, create_crosshair_overlay, multi_planar_view, normalize_array
from utils.conversion import perform_conversion, dicom_to_nifti, dicom_to_mat
from utils.debug_utils import logger, log_exception, debug_dicom_loading, inspect_dicom

# Configure pydicom handlers for compressed files
def setup_dicom_handlers():
    """Setup DICOM handlers for compressed DICOM files"""
    logger.info("Setting up DICOM pixel data handlers")
    
    # Reset handlers
    pydicom.config.pixel_data_handlers = []
    handlers_added = []
    
    # Try to add each handler in preferred order
    try:
        from pydicom.pixel_data_handlers import gdcm_handler
        if gdcm_handler.is_available():
            pydicom.config.pixel_data_handlers.append(gdcm_handler)
            handlers_added.append("gdcm")
            logger.info("Added GDCM handler")
    except Exception as e:
        logger.error(f"Could not add GDCM handler: {str(e)}")
        
    try:
        from pydicom.pixel_data_handlers import pylibjpeg_handler
        if pylibjpeg_handler.is_available():
            pydicom.config.pixel_data_handlers.append(pylibjpeg_handler)
            handlers_added.append("pylibjpeg")
            logger.info("Added pylibjpeg handler")
    except Exception as e:
        logger.error(f"Could not add pylibjpeg handler: {str(e)}")
        
    try:
        from pydicom.pixel_data_handlers import pillow_handler
        if pillow_handler.is_available():
            pydicom.config.pixel_data_handlers.append(pillow_handler)
            handlers_added.append("pillow")
            logger.info("Added Pillow handler")
    except Exception as e:
        logger.error(f"Could not add Pillow handler: {str(e)}")
        
    try:
        from pydicom.pixel_data_handlers import numpy_handler
        pydicom.config.pixel_data_handlers.append(numpy_handler)
        handlers_added.append("numpy")
        logger.info("Added NumPy handler")
    except Exception as e:
        logger.error(f"Could not add NumPy handler: {str(e)}")
    
    # Enable pydicom debugging
    pydicom.config.debug(True)
    
    logger.info(f"Configured DICOM handlers: {', '.join(handlers_added)}")
    return pydicom.config.pixel_data_handlers

# helper to wrap numpy slice array into Plotly figure
def make_slice_figure(slice_array):
    """Create a Plotly figure from a slice array, preserving original resolution"""
    # Ensure we're using high-quality image rendering
    fig = go.Figure(go.Image(
        z=slice_array,
        # Use 'none' interpolation to avoid blurring pixels
        hoverinfo='none',
        colormodel='rgb'
    ))
    
    # Configure layout for high-quality display
    fig.update_layout(
        dragmode='drawclosedpath',
        newshape_line_color='cyan',
        modebar=dict(
            add=[
                'drawline','drawopenpath','drawclosedpath','drawcircle',
                'drawrect','eraseshape',
                'select2d','lasso2d',
                'pan2d','zoom2d','zoomIn2d','zoomOut2d',
                'autoScale2d','resetScale2d','toImage'
            ],
            remove=[
                'hoverClosestCartesian','hoverCompareCartesian','toggleSpikelines'
            ]
        ),
        # Preserve aspect ratio with exact 1:1 scaling
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        xaxis=dict(
            constrain="domain",
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        # Maximize image size by eliminating all margins
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        # Set plot background to black for medical imaging
        plot_bgcolor='black',
        paper_bgcolor='black',
        # Disable autoscaling which can reduce quality
        autosize=False
    )
    
    # Set the config to use high-quality image rendering
    fig.update_traces(
        hovertemplate=None,
        hoverinfo='none'
    )
    
    return fig

# Main application class for SegMed-Pro
class SegMedPro:
    def __init__(self):
        # Set up DICOM handlers
        self.dicom_handlers = setup_dicom_handlers()
        logger.info(f"Initialized DICOM handlers: {len(self.dicom_handlers)} handlers available")
        
        self.current_data = None
        self.current_data_type = None  # "dicom" or "nifti"
        self.current_metadata = None
        self.current_directory = None
        self.file_list = []
        self.current_slice_idx = 0
        self.current_view = "axial"  # axial, sagittal, coronal
        self.crosshair_position = None
        self.current_shape = (0, 0, 0)
        # Store the current plot tool
        self.current_plot_tool = "drawclosedpath"
        
        # Add segmentation overlay support
        self.segmentation_data = None
        self.segmentation_loaded = False
        self.segmentation_colormap = {
            0: [0, 0, 0],      # Background (transparent)
            1: [255, 0, 0],    # Label 1 (Red)
            2: [0, 255, 0],    # Label 2 (Green)
            3: [0, 0, 255],    # Label 3 (Blue)
            4: [255, 255, 0],  # Label 4 (Yellow)
            5: [0, 255, 255],  # Label 5 (Cyan)
            6: [255, 0, 255],  # Label 6 (Magenta)
            7: [255, 165, 0],  # Label 7 (Orange)
            8: [128, 0, 128]   # Label 8 (Purple)
        }
        self.segmentation_alpha = 0.5  # Default transparency

        # Add path for custom components
        self.custom_components_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_components")
        if os.path.exists(self.custom_components_path):
            sys.path.append(self.custom_components_path)
            try:
                # Try to import the custom component
                from plot_tools.component import PlotTools
                self.has_plot_tools = True
            except ImportError:
                self.has_plot_tools = False
        else:
            self.has_plot_tools = False

    def build_interface(self):
        """Build the Gradio interface for SegMed-Pro"""
        with gr.Blocks(title="SegMed-Pro") as app:
            gr.Markdown("# SegMed-Pro: Medical Imaging Annotation Tool")
            
            with gr.Tabs() as tabs:
                # Viewer Tab
                with gr.TabItem("Viewer"):
                    with gr.Row():
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
                            
                            # Added error display
                            error_display = gr.Textbox(label="Status/Errors", interactive=False)

                            # Window/Level controls
                            with gr.Accordion("Image Adjustments", open=False):
                                window_level = gr.Slider(minimum=0, maximum=4000, value=500, step=10, label="Window Level (Center)")
                                window_width = gr.Slider(minimum=1, maximum=4000, value=1000, step=10, label="Window Width")
                                apply_window_btn = gr.Button("Apply Window/Level")
                            
                            # Debug button
                            debug_btn = gr.Button("Debug Selected File")
                            
                            # Segmentation tools
                            with gr.Accordion("Segmentation Tools", open=True):
                                segmentation_file = gr.File(label="Load Segmentation (NIfTI .nii.gz)", file_types=[".nii", ".nii.gz"])
                                load_seg_btn = gr.Button("Load Segmentation")
                                seg_opacity = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Segmentation Opacity")
                                clear_seg_btn = gr.Button("Clear Segmentation")
                                seg_status = gr.Textbox(label="Segmentation Status", value="No segmentation loaded")
                        
                        with gr.Column(scale=4):
                            # Main Visualization Area (middle column)
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
                                slice_slider = gr.Slider(minimum=0, maximum=0, value=0, step=1, label="Slice Navigation", visible=True)
                                next_btn = gr.Button("Next")
                            with gr.Row():
                                slice_text = gr.Textbox(label="Slice", interactive=False)
                                crosshair_info = gr.Textbox(label="Crosshair", interactive=False)
                        
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
                            # Add more as needed

                # Conversion Tab
                with gr.TabItem("Conversion"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Format Conversion")
                            conversion_input = gr.File(label="Input File(s)")
                            conversion_dir = gr.Textbox(label="Or Input Directory")
                            
                            conversion_type = gr.Dropdown(
                                choices=[
                                    "DICOM to NIFTI", 
                                    "NIFTI to MAT", 
                                    "DICOM to MAT",
                                    "NIFTI to PNG"
                                ],
                                label="Conversion Type",
                                value="DICOM to NIFTI"
                            )
                            
                            output_dir = gr.Textbox(label="Output Directory")
                            convert_btn = gr.Button("Convert")
                            conversion_status = gr.Textbox(label="Conversion Status")
                            conversion_output = gr.File(label="Output File (if available)")
            
            # Event handlers for the viewer tab
            @log_exception
            def load_data(file_obj, directory):
                """Load DICOM data from file or directory"""
                logger.info(f"Loading data: file={file_obj}, directory={directory}")
                
                # Reset current data
                self.current_data = None
                self.current_data_type = None
                self.current_metadata = None
                self.current_directory = None
                self.file_list = []
                self.current_slice_idx = 0
                self.crosshair_position = None
                
                # if user entered a directory path, load entire series
                if directory:
                    path = directory
                    self.current_data, self.current_metadata, self.file_list = load_dicom_series(path)
                else:
                    # load single file by extension
                    path = file_obj.name if file_obj else None
                    if not path or not os.path.exists(path):
                        return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", "No data loaded"
                    ext = os.path.splitext(path)[1].lower()
                    if ext in ['.dcm']:
                        self.current_data, self.current_metadata, self.file_list = load_dicom_series(path)
                    elif ext in ['.nii', '.gz']:
                        img3d, meta = load_nifti_file(path)
                        self.current_data = img3d
                        self.current_metadata = meta
                        self.file_list = [path]
                    elif ext == '.mat':
                        arr = nifti_to_mat(path) if False else None  # placeholder for mat loading
                        # Note: implement MAT loading util or adapt existing
                        self.current_data = arr
                        self.current_metadata = {}
                        self.file_list = [path]
                    else:
                        return None, gr.Dropdown(choices=[]), {}, gr.Slider(visible=False), "0/0", "x: 0, y: 0, z: 0", f"Unsupported file type: {ext}"
                self.current_data_type = "dicom"
                shape = self.current_data.shape
                self.current_shape = shape
                # center crosshair
                self.crosshair_position = (shape[2]//2, shape[1]//2, shape[0]//2)
                # prepare file names
                file_names = [os.path.basename(f) for f in self.file_list]
                # slider range and visibility
                slider_min, slider_max = 0, shape[0] - 1
                self.current_slice_idx = 0
                visible_flag = slider_max > 0
                
                # Extract WindowCenter and WindowWidth from metadata if available
                window_center = None
                window_width = None
                
                if self.current_metadata and 'WindowCenter' in self.current_metadata:
                    window_center = self.current_metadata['WindowCenter']
                    # Handle list format if needed
                    if isinstance(window_center, list):
                        window_center = window_center[0]
                    logger.info(f"Using WindowCenter from metadata: {window_center}")
                
                if self.current_metadata and 'WindowWidth' in self.current_metadata:
                    window_width = self.current_metadata['WindowWidth']
                    # Handle list format if needed
                    if isinstance(window_width, list):
                        window_width = window_width[0]
                    logger.info(f"Using WindowWidth from metadata: {window_width}")
                
                # initial image
                img = display_slice(
                    self.current_data, 
                    0, 
                    self.current_view.lower(), 
                    window_level=window_center, 
                    window_width=window_width,
                    crosshair=self.crosshair_position
                )
                
                fig = make_slice_figure(img)
                
                crosshair_text = f"x: {self.crosshair_position[0]}, y: {self.crosshair_position[1]}, z: {self.crosshair_position[2]}"
                
                # Also update the window/level sliders with the extracted values
                window_level_value = window_center if window_center is not None else 500
                window_width_value = window_width if window_width is not None else 1000
                
                return (
                    fig,
                    gr.Dropdown(choices=file_names, value=file_names[0] if file_names else None),
                    self.current_metadata,
                    gr.Slider(minimum=slider_min, maximum=slider_max, value=0, step=1, label="Slice Navigation", visible=visible_flag),
                    f"0/{slider_max}",
                    crosshair_text,
                    f"DICOM data loaded: {len(self.file_list)} slice(s)",
                    window_level_value,  # Return window level value for slider
                    window_width_value   # Return window width value for slider
                )
            
            load_btn.click(
                fn=load_data,
                inputs=[file_input, dir_input],
                outputs=[
                    image_plot, file_browser, metadata_display,
                    slice_slider, slice_text, crosshair_info, error_display
                ]
            )
            
            @log_exception
            def reset_directory():
                """Clear the directory textbox"""
                return ""
            reset_dir_btn.click(
                fn=reset_directory,
                inputs=[],
                outputs=[dir_input]
            )

            # Auto-trigger load_data on file selection
            file_input.change(
                fn=load_data,
                inputs=[file_input, dir_input],
                outputs=[
                    image_plot, file_browser, metadata_display,
                    slice_slider, slice_text, crosshair_info, error_display
                ]
            )
            
            @log_exception
            def debug_selected_file(file_obj, directory):
                """Debug the currently selected file"""
                file_path = None
                
                if file_obj is not None:
                    file_path = file_obj.name
                elif directory and os.path.exists(directory) and not os.path.isdir(directory):
                    file_path = directory
                
                if file_path and file_path.lower().endswith('.dcm'):
                    logger.info(f"Debugging DICOM file: {file_path}")
                    result = debug_dicom_loading(file_path)
                    return f"Debug complete. Check log file for details: {os.path.abspath('utils/segmed_pro_debug.log')}"
                else:
                    return "Please select a valid DICOM (.dcm) file to debug"
            
            debug_btn.click(
                fn=debug_selected_file,
                inputs=[file_input, dir_input],
                outputs=[error_display]
            )
            
            @log_exception
            def update_slice(slider_value, view_type):
                """Update the displayed slice based on slider value and view"""
                if self.current_data is None:
                    return None, "0/0", "x: 0, y: 0, z: 0", {}
                
                logger.info(f"Updating slice: value={slider_value}, view={view_type}")
                
                # Update the current slice index and view
                self.current_slice_idx = int(slider_value)
                self.current_view = view_type.lower()
                
                # Determine slice count based on view
                total_slices = 0
                if self.current_view == "axial":
                    total_slices = self.current_shape[0]
                elif self.current_view == "sagittal":
                    total_slices = self.current_shape[2]
                elif self.current_view == "coronal":
                    total_slices = self.current_shape[1]
                
                # Update the crosshair position based on the current slice
                if self.current_view == "axial":
                    x, y, _ = self.crosshair_position
                    self.crosshair_position = (x, y, self.current_slice_idx)
                elif self.current_view == "sagittal":
                    _, y, z = self.crosshair_position
                    self.crosshair_position = (self.current_slice_idx, y, z)
                elif self.current_view == "coronal":
                    x, _, z = self.crosshair_position
                    self.crosshair_position = (x, self.current_slice_idx, z)
                
                # Generate the slice image
                try:
                    img = display_slice(
                        self.current_data, 
                        self.current_slice_idx, 
                        self.current_view,
                        crosshair=self.crosshair_position
                    )
                    
                    # Apply segmentation overlay if segmentation is loaded
                    if self.segmentation_loaded and self.segmentation_data is not None:
                        try:
                            # Extract the appropriate segmentation slice based on view
                            if self.current_view == 'axial':
                                seg_slice = self.segmentation_data[self.current_slice_idx, :, :]
                            elif self.current_view == 'sagittal':
                                seg_slice = self.segmentation_data[:, :, self.current_slice_idx]
                            elif self.current_view == 'coronal':
                                seg_slice = self.segmentation_data[:, self.current_slice_idx, :]
                            
                            # Apply the segmentation overlay
                            from utils.visualization import overlay_segmentation
                            img = overlay_segmentation(
                                img, 
                                seg_slice, 
                                alpha=self.segmentation_alpha,
                                colormap=self.segmentation_colormap
                            )
                            logger.info(f"Applied segmentation overlay to slice {self.current_slice_idx}")
                        except Exception as e:
                            logger.error(f"Error applying segmentation overlay: {str(e)}")
                    
                    fig = make_slice_figure(img)
                    logger.info(f"Generated slice image with shape {img.shape}")
                    
                    # Update crosshair info
                    x, y, z = self.crosshair_position
                    crosshair_text = f"x: {x}, y: {y}, z: {z}"
                    
                    # Initialize metadata with current metadata
                    metadata = self.current_metadata
                    
                    # If we have a file list, update the file browser to show the current file
                    if self.current_view == "axial" and self.current_data_type == "dicom" and self.file_list:
                        # Only update metadata for axial view in DICOM series
                        # This ensures we get per-slice metadata
                        if 0 <= self.current_slice_idx < len(self.file_list):
                            try:
                                # Update metadata to reflect the current slice
                                current_file = self.file_list[self.current_slice_idx]
                                logger.info(f"Loading metadata for file: {current_file}")
                                
                                # Get fresh metadata for the current slice
                                metadata = get_dicom_metadata(current_file)
                                
                                # Add series information to metadata
                                metadata['SeriesInfo'] = {
                                    'CurrentSlice': self.current_slice_idx + 1,
                                    'TotalSlices': len(self.file_list),
                                    'CurrentFile': os.path.basename(current_file)
                                }
                                # Update the current metadata
                                self.current_metadata = metadata
                                logger.info(f"Updated metadata for slice {self.current_slice_idx}")
                            except Exception as e:
                                logger.error(f"Error updating metadata: {str(e)}")
                    
                    return fig, f"{self.current_slice_idx}/{total_slices-1}", crosshair_text, metadata
                except Exception as e:
                    logger.error(f"Error generating slice image: {str(e)}")
                    return None, f"Error: {str(e)}", "x: 0, y: 0, z: 0", {}
            
            slice_slider.change(
                fn=update_slice,
                inputs=[slice_slider, view_selector],
                outputs=[image_plot, slice_text, crosshair_info, metadata_display]
            )
            
            def change_view(view):
                """Change the viewing orientation"""
                if self.current_data is None:
                    return gr.Slider(), "0/0", None
                
                # Update current view
                self.current_view = view.lower()
                
                # Determine slice count and range based on view
                slider_min = 0
                if self.current_view == "axial":
                    slider_max = self.current_shape[0] - 1
                    self.current_slice_idx = self.crosshair_position[2]
                elif self.current_view == "sagittal":
                    slider_max = self.current_shape[2] - 1
                    self.current_slice_idx = self.crosshair_position[0]
                elif self.current_view == "coronal":
                    slider_max = self.current_shape[1] - 1
                    self.current_slice_idx = self.crosshair_position[1]
                
                # Generate the slice image
                img = display_slice(
                    self.current_data, 
                    self.current_slice_idx, 
                    self.current_view,
                    crosshair=self.crosshair_position
                )
                fig = make_slice_figure(img)
                
                return (gr.Slider(minimum=slider_min, maximum=slider_max, value=self.current_slice_idx), f"{self.current_slice_idx}/{slider_max}", fig)
            
            view_selector.change(
                fn=change_view,
                inputs=[view_selector],
                outputs=[slice_slider, slice_text, image_plot]
            )
            
            def update_window_level(level, width):
                """Apply window/level adjustments to the current image"""
                if self.current_data is None:
                    return None
                
                # Generate the slice image with window/level adjustments
                img = display_slice(
                    self.current_data, 
                    self.current_slice_idx, 
                    self.current_view,
                    window_level=level,
                    window_width=width,
                    crosshair=self.crosshair_position
                )
                fig = make_slice_figure(img)
                
                return fig
            
            apply_window_btn.click(
                fn=update_window_level,
                inputs=[window_level, window_width],
                outputs=[image_plot]
            )
            
            def select_file_from_browser(selected_file):
                """When a file is selected from the browser dropdown"""
                if not selected_file or not self.file_list:
                    return None, "0/0", "x: 0, y: 0, z: 0", {}, 0
                
                try:
                    # Find the selected file in the file list
                    selected_path = None
                    for file_path in self.file_list:
                        if os.path.basename(file_path) == selected_file:
                            selected_path = file_path
                            break
                    
                    if not selected_path:
                        return None, "0/0", "x: 0, y: 0, z: 0", {"error": "File not found"}, 0
                    
                    # Load the selected DICOM file (no actual re-read of series data)
                    slice_idx = self.file_list.index(selected_path)
                    self.current_slice_idx = slice_idx
                    # Update crosshair z position
                    x, y, _ = self.crosshair_position
                    self.crosshair_position = (x, y, slice_idx)
                    
                    # Generate image at the new index
                    img = display_slice(
                        self.current_data, 
                        slice_idx, 
                        "axial",
                        crosshair=self.crosshair_position
                    )
                    fig = make_slice_figure(img)
                    
                    # Update crosshair info
                    x, y, z = self.crosshair_position
                    crosshair_text = f"x: {x}, y: {y}, z: {z}"
                    
                    # Get metadata for this slice
                    metadata = get_dicom_metadata(selected_path)
                    
                    # Return image, slice text, crosshair, metadata, and updated slider value
                    return fig, f"{slice_idx}/{len(self.file_list)-1}", crosshair_text, metadata, slice_idx
                except Exception as e:
                    return None, "0/0", "x: 0, y: 0, z: 0", {"error": str(e)}, 0
            
            file_browser.change(
                fn=select_file_from_browser,
                inputs=[file_browser],
                outputs=[image_plot, slice_text, crosshair_info, metadata_display, slice_slider]
            )
            
            # --- Plot tool selection logic ---
            # Store the current plot tool
            self.current_plot_tool = "drawclosedpath"

            def update_plot_tool(tool_name):
                """Update the plotly figure to select the given drawing tool without reloading the image"""
                if self.current_data is None:
                    return None
                
                # Map button to plotly tool
                tool_map = {
                    "draw_circle": "drawcircle",
                    "draw_rect": "drawrect",
                    "draw_line": "drawline",
                    "draw_openpath": "drawopenpath",
                    "draw_closedpath": "drawclosedpath",
                    "erase_shape": "eraseshape",
                    "pan": "pan2d",
                    "zoom": "zoom2d",
                    "reset": "resetScale2d"
                }
                plotly_tool = tool_map.get(tool_name, "drawclosedpath")
                self.current_plot_tool = plotly_tool
                
                # Create JavaScript for updating dragmode without reloading
                script = f"""
                <script>
                (function() {{
                    const plotDiv = document.querySelector('div[data-testid="plotly-graph-div"]');
                    if (plotDiv && plotDiv._fullLayout) {{
                        Plotly.relayout(plotDiv, {{dragmode: '{plotly_tool}'}});
                        console.log('Set dragmode to: {plotly_tool}');
                    }} else {{
                        console.log('Plotly element not found');
                    }}
                }})();
                </script>
                """
                
                # Return HTML component with the script that changes the dragmode
                return gr.HTML(script)

            draw_circle_btn.click(lambda: update_plot_tool("draw_circle"), None, [gr.HTML()])
            draw_rect_btn.click(lambda: update_plot_tool("draw_rect"), None, [gr.HTML()])
            draw_line_btn.click(lambda: update_plot_tool("draw_line"), None, [gr.HTML()])
            draw_openpath_btn.click(lambda: update_plot_tool("draw_openpath"), None, [gr.HTML()])
            draw_closedpath_btn.click(lambda: update_plot_tool("draw_closedpath"), None, [gr.HTML()])
            erase_shape_btn.click(lambda: update_plot_tool("erase_shape"), None, [gr.HTML()])
            pan_btn.click(lambda: update_plot_tool("pan"), None, [gr.HTML()])
            zoom_btn.click(lambda: update_plot_tool("zoom"), None, [gr.HTML()])
            reset_btn.click(lambda: update_plot_tool("reset"), None, [gr.HTML()])

            # Event handlers for the conversion tab
            def convert_files(file_obj, directory, conv_type, out_dir):
                """Convert files between formats"""
                try:
                    input_path = None
                    
                    # Determine input source
                    if file_obj is not None:
                        input_path = file_obj.name
                    elif directory and os.path.exists(directory):
                        input_path = directory
                    else:
                        return "No valid input file or directory provided.", None
                    
                    # Ensure output directory exists
                    if not out_dir:
                        out_dir = os.path.dirname(input_path)
                    
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    
                    # Determine output path
                    base_name = os.path.basename(input_path)
                    output_path = None
                    
                    if conv_type == "DICOM to NIFTI":
                        output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}.nii.gz")
                        result_path = dicom_to_nifti(input_path, output_path)
                    elif conv_type == "NIFTI to MAT":
                        output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}.mat")
                        result_path = nifti_to_mat(input_path, output_path)
                    elif conv_type == "DICOM to MAT":
                        output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}.mat")
                        result_path = dicom_to_mat(input_path, output_path)
                    elif conv_type == "NIFTI to PNG":
                        output_path = os.path.join(out_dir, f"{os.path.splitext(base_name)[0]}_png")
                        result_path = nifti_to_png(input_path, output_path)
                    
                    # Create success message
                    success_msg = f"Conversion successful: {conv_type}\nOutput: {result_path}"
                    
                    # Return output file if it's a single file
                    output_file = None
                    if os.path.isfile(result_path):
                        output_file = result_path
                    
                    return success_msg, output_file
                
                except Exception as e:
                    return f"Error during conversion: {str(e)}", None
            
            convert_btn.click(
                fn=convert_files,
                inputs=[conversion_input, conversion_dir, conversion_type, output_dir],
                outputs=[conversion_status, conversion_output]
            )
            
            # Previous/Next button functionality
            def prev_slice(slider_value):
                """Go to previous slice by decrementing slider value"""
                current_value = int(slider_value)
                return max(0, current_value - 1)
            
            def next_slice(slider_value):
                """Go to next slice by incrementing slider value"""
                if self.current_data is None:
                    return slider_value
                
                current_value = int(slider_value)
                # Determine max value based on view orientation
                max_value = 0
                if self.current_view == "axial":
                    max_value = self.current_shape[0] - 1
                elif self.current_view == "sagittal":
                    max_value = self.current_shape[2] - 1
                elif self.current_view == "coronal":
                    max_value = self.current_shape[1] - 1
                
                return min(current_value + 1, max_value)
            
            # Connect the Previous/Next buttons to update the slider
            prev_btn.click(
                fn=prev_slice,
                inputs=[slice_slider],
                outputs=[slice_slider]
            )
            
            next_btn.click(
                fn=next_slice,
                inputs=[slice_slider],
                outputs=[slice_slider]
            )
            
            # Segmentation loading and overlay functions
            @log_exception
            def load_segmentation(seg_file):
                """Load segmentation file (NIfTI .nii.gz) and overlay on image"""
                if self.current_data is None:
                    return "Please load a DICOM dataset first", None
                
                if seg_file is None:
                    return "No segmentation file provided", None
                
                try:
                    # Load the segmentation NIfTI file
                    seg_path = seg_file.name
                    logger.info(f"Loading segmentation file: {seg_path}")
                    
                    seg_data, seg_meta = load_nifti_file(seg_path)
                    
                    # Check dimensions match the main data
                    if seg_data.shape != self.current_data.shape:
                        logger.error(f"Segmentation dimensions {seg_data.shape} don't match data dimensions {self.current_data.shape}")
                        return f"Error: Segmentation dimensions {seg_data.shape} don't match data dimensions {self.current_data.shape}", None
                    
                    # Store the segmentation data
                    self.segmentation_data = seg_data
                    self.segmentation_loaded = True
                    
                    # Update display with segmentation overlay
                    img = display_slice(
                        self.current_data, 
                        self.current_slice_idx, 
                        self.current_view,
                        crosshair=self.crosshair_position
                    )
                    
                    # Get the corresponding segmentation slice
                    if self.current_view == 'axial':
                        seg_slice = self.segmentation_data[self.current_slice_idx, :, :]
                    elif self.current_view == 'sagittal':
                        seg_slice = self.segmentation_data[:, :, self.current_slice_idx]
                    elif self.current_view == 'coronal':
                        seg_slice = self.segmentation_data[:, self.current_slice_idx, :]
                    
                    # Overlay segmentation on the image
                    from utils.visualization import overlay_segmentation
                    overlaid_img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.segmentation_alpha,
                        colormap=self.segmentation_colormap
                    )
                    
                    fig = make_slice_figure(overlaid_img)
                    
                    # Check unique labels in segmentation
                    unique_labels = np.unique(seg_data)
                    label_str = ", ".join(map(str, unique_labels))
                    
                    return f"Segmentation loaded, found labels: {label_str}", fig
                
                except Exception as e:
                    logger.error(f"Error loading segmentation: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.segmentation_loaded = False
                    self.segmentation_data = None
                    return f"Error loading segmentation: {str(e)}", None
            
            @log_exception
            def direct_load_segmentation(seg_file):
                """Direct method to load segmentation file with axis swapping for dimension mismatches"""
                if self.current_data is None:
                    return "Please load a DICOM dataset first", None
                
                if seg_file is None:
                    return "No segmentation file provided", None
                
                try:
                    import nibabel as nib
                    import numpy as np
                    
                    # Get file path from the uploaded file object
                    seg_path = seg_file.name
                    logger.info(f"Direct loading segmentation file: {seg_path}")
                    
                    # Load the NIfTI file directly with nibabel
                    nii_img = nib.load(seg_path)
                    
                    # Get data as array with explicit cast to int for segmentation labels
                    seg_data_orig = np.asarray(nii_img.get_fdata()).astype(np.int32)
                    logger.info(f"Original segmentation dimensions: {seg_data_orig.shape}")
                    logger.info(f"DICOM data dimensions: {self.current_data.shape}")
                    
                    # Handle dimension mismatch
                    if seg_data_orig.shape != self.current_data.shape:
                        # Check if it's just an axis ordering issue
                        dicom_dims = set(self.current_data.shape)
                        seg_dims = set(seg_data_orig.shape)
                        
                        # For your specific case: (400, 512, 20) vs (20, 512, 400)
                        # This is clearly just axes being in different order
                        if dicom_dims == seg_dims:
                            logger.info("Dimensions match but in different order, performing axis swapping")
                            
                            # For this specific case, we need to swap axes 0 and 2
                            if (seg_data_orig.shape[0] == self.current_data.shape[2] and 
                                seg_data_orig.shape[2] == self.current_data.shape[0]):
                                # Swap axes 0 and 2
                                seg_data = np.transpose(seg_data_orig, (2, 1, 0))
                                logger.info(f"After swapping axes: segmentation shape {seg_data.shape}")
                            else:
                                # More general case - find the right permutation
                                perm = []
                                for i in range(3):
                                    for j in range(3):
                                        if self.current_data.shape[i] == seg_data_orig.shape[j]:
                                            perm.append(j)
                                            break
                                        
                                if len(perm) == 3:
                                    seg_data = np.transpose(seg_data_orig, perm)
                                    logger.info(f"General axes permutation {perm}: segmentation shape {seg_data.shape}")
                                else:
                                    # Fallback if we can't find a clean permutation
                                    seg_data = seg_data_orig
                                    logger.warning("Could not find a valid axis permutation. Using original data.")
                        else:
                            logger.warning("Dimensions don't match exactly. Using original segmentation data.")
                            seg_data = seg_data_orig
                    else:
                        # Dimensions already match
                        seg_data = seg_data_orig
                    
                    # Store the segmentation data
                    self.segmentation_data = seg_data
                    self.segmentation_loaded = True
                    
                    # Update display with segmentation overlay
                    img = display_slice(
                        self.current_data, 
                        self.current_slice_idx, 
                        self.current_view,
                        crosshair=self.crosshair_position
                    )
                    
                    # Get the corresponding segmentation slice
                    if self.current_view == 'axial':
                        seg_slice = self.segmentation_data[self.current_slice_idx, :, :]
                    elif self.current_view == 'sagittal':
                        seg_slice = self.segmentation_data[:, :, self.current_slice_idx]
                    elif self.current_view == 'coronal':
                        seg_slice = self.segmentation_data[:, self.current_slice_idx, :]
                    
                    # Overlay segmentation on the image
                    from utils.visualization import overlay_segmentation
                    overlaid_img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.segmentation_alpha,
                        colormap=self.segmentation_colormap
                    )
                    
                    fig = make_slice_figure(overlaid_img)
                    
                    # Check unique labels in segmentation
                    unique_labels = np.unique(seg_data)
                    label_str = ", ".join(map(str, unique_labels))
                    
                    return f"Segmentation loaded with axis reorientation. Found labels: {label_str}", fig
                
                except Exception as e:
                    logger.error(f"Error in direct segmentation loading: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.segmentation_loaded = False
                    self.segmentation_data = None
                    return f"Error loading segmentation: {str(e)}", None
            
            def load_segmentation_file(self, seg_file):
                """Load segmentation file (NIfTI .nii.gz) and prepare for overlay on image"""
                if self.current_data is None:
                    return "Please load a DICOM dataset first", None
                
                if seg_file is None:
                    return "No segmentation file provided", None
                
                try:
                    import nibabel as nib
                    import numpy as np
                    
                    # Get file path from the uploaded file object
                    seg_path = seg_file.name
                    logger.info(f"Loading segmentation file: {seg_path}")
                    
                    # Load the NIfTI file directly with nibabel to avoid possible issues
                    nii_img = nib.load(seg_path)
                    
                    # Get data as array with explicit cast to int for segmentation labels
                    seg_data = np.asarray(nii_img.get_fdata()).astype(np.int32)
                    
                    # Check dimensions match the main data
                    if seg_data.shape != self.current_data.shape:
                        logger.error(f"Segmentation dimensions {seg_data.shape} don't match data dimensions {self.current_data.shape}")
                        return f"Error: Segmentation dimensions {seg_data.shape} don't match data dimensions {self.current_data.shape}", None
                    
                    # Store the segmentation data
                    self.segmentation_data = seg_data
                    self.segmentation_loaded = True
                    
                    # Update display with segmentation overlay
                    img = display_slice(
                        self.current_data, 
                        self.current_slice_idx, 
                        self.current_view,
                        crosshair=self.crosshair_position
                    )
                    
                    # Get the corresponding segmentation slice
                    if self.current_view == 'axial':
                        seg_slice = self.segmentation_data[self.current_slice_idx, :, :]
                    elif self.current_view == 'sagittal':
                        seg_slice = self.segmentation_data[:, :, self.current_slice_idx]
                    elif self.current_view == 'coronal':
                        seg_slice = self.segmentation_data[:, self.current_slice_idx, :]
                    
                    # Overlay segmentation on the image
                    from utils.visualization import overlay_segmentation
                    overlaid_img = overlay_segmentation(
                        img, 
                        seg_slice, 
                        alpha=self.segmentation_alpha,
                        colormap=self.segmentation_colormap
                    )
                    
                    fig = make_slice_figure(overlaid_img)
                    
                    # Check unique labels in segmentation
                    unique_labels = np.unique(seg_data)
                    label_str = ", ".join(map(str, unique_labels))
                    
                    return f"Segmentation loaded, found labels: {label_str}", fig
                
                except Exception as e:
                    logger.error(f"Error loading segmentation: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    self.segmentation_loaded = False
                    self.segmentation_data = None
                    return f"Error loading segmentation: {str(e)}", None
            
            @log_exception
            def update_segmentation_opacity(opacity):
                """Update the opacity/transparency of the segmentation overlay"""
                if not self.segmentation_loaded or self.segmentation_data is None:
                    return "No segmentation loaded", None
                
                self.segmentation_alpha = opacity
                
                # Re-generate the image with updated opacity
                img = display_slice(
                    self.current_data, 
                    self.current_slice_idx, 
                    self.current_view,
                    crosshair=self.crosshair_position
                )
                
                # Get the corresponding segmentation slice
                if self.current_view == 'axial':
                    seg_slice = self.segmentation_data[self.current_slice_idx, :, :]
                elif self.current_view == 'sagittal':
                    seg_slice = self.segmentation_data[:, :, self.current_slice_idx]
                elif self.current_view == 'coronal':
                    seg_slice = self.segmentation_data[:, self.current_slice_idx, :]
                
                # Overlay segmentation on the image
                from utils.visualization import overlay_segmentation
                overlaid_img = overlay_segmentation(
                    img, 
                    seg_slice, 
                    alpha=self.segmentation_alpha,
                    colormap=self.segmentation_colormap
                )
                
                fig = make_slice_figure(overlaid_img)
                
                return f"Segmentation opacity updated to {opacity:.1f}", fig
            
            @log_exception
            def clear_segmentation():
                """Clear the current segmentation overlay"""
                self.segmentation_data = None
                self.segmentation_loaded = False
                
                if self.current_data is None:
                    return "No data loaded", None
                
                # Re-display image without segmentation
                img = display_slice(
                    self.current_data, 
                    self.current_slice_idx, 
                    self.current_view,
                    crosshair=self.crosshair_position
                )
                
                fig = make_slice_figure(img)
                
                return "Segmentation cleared", fig
                
            # Connect segmentation buttons to functions
            load_seg_btn.click(
                fn=direct_load_segmentation,
                inputs=[segmentation_file],
                outputs=[seg_status, image_plot]
            )
            
            seg_opacity.change(
                fn=update_segmentation_opacity,
                inputs=[seg_opacity],
                outputs=[seg_status, image_plot]
            )
            
            clear_seg_btn.click(
                fn=clear_segmentation,
                inputs=[],
                outputs=[seg_status, image_plot]
            )
            
        return app

# Initialize and launch the application
if __name__ == "__main__":
    logger.info("Starting SegMed-Pro application")
    segmed_pro = SegMedPro()
    app = segmed_pro.build_interface()
    app.launch(share=False)