"""
SegMed-Pro Main Application

Refactored main application class that orchestrates the UI components and event handlers
following Gradio best practices for modular application structure.
"""

import gradio as gr
import logging
import numpy as np
import os
import sys
import threading

# ===== CROWDSOURCING CONFIGURATION =====
ENABLE_AUTH = True  # Set to False to disable authentication and crowdsourcing features
# =======================================

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom logging filter to only show specific INFO messages for debugging
class DebugFilter(logging.Filter):
    """Filter to only show specific INFO messages for debugging annotation and 3D viewer logic"""
    def filter(self, record):
        if record.levelno == logging.INFO:
            # Only allow INFO messages that contain these specific keywords
            allowed_keywords = [
                "Annotation Status",
                "3D Viewer status",
                "3D Viewer:",  # For 3D viewer debug messages
                "Annotation Status:",  # For annotation status messages
                "DEBUG",  # For debug messages
                "Current labels update",  # For label update messages
                "Loading next assignment",  # For next assignment debugging
                "annotated_value",  # For image format debugging
                "load_dataset_from_contribute",  # For dataset loading debugging
                "image_annotator",  # For image annotator debugging
                "Patient ID from task",  # For patient ID debugging
                # Annotation submission debugging
                "Processing image_annotator_data",  # For annotation data processing
                "Saving current slice annotations",  # For annotation saving
                "Found user_annotations storage",  # For user annotation storage
                "Available slice annotations",  # For slice annotation debugging
                "Processing slice",  # For slice processing
                "No stored user_annotations found",  # For annotation storage debugging
                "Processing current image_annotator_data",  # For fallback processing
                "Found",  # For "Found X raw annotations to process"
                "Final annotation collection",  # For final annotation count
                "Submitting annotation with",  # For submission summary
            ]
            message = record.getMessage()
            return any(keyword in message for keyword in allowed_keywords)
        # Allow all non-INFO messages (ERROR, WARNING, DEBUG, etc.)
        return record.levelno != logging.INFO

# Apply the filter to the root logger
root_logger = logging.getLogger()
debug_filter = DebugFilter()
for handler in root_logger.handlers:
    handler.addFilter(debug_filter)

# Also apply to specific loggers that might bypass the root logger
specific_loggers = [
    'segmed_pro',
    '__main__',
    'ui.image_handlers',
    'ui.editor_tab',
    'utils.debug_utils',
    'utils.dicom_utils'
]
for logger_name in specific_loggers:
    specific_logger = logging.getLogger(logger_name)
    for handler in specific_logger.handlers:
        handler.addFilter(debug_filter)
    # If no handlers, the filter on root logger will catch it

# Suppress excessive pydicom debug logs
pydicom_logger = logging.getLogger('pydicom')
pydicom_logger.setLevel(logging.WARNING)

# Also suppress pydicom.pixel_data_handlers debug logs
pixel_handlers_logger = logging.getLogger('pydicom.pixel_data_handlers')
pixel_handlers_logger.setLevel(logging.WARNING)

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our modular components
from ui.state import AppState
from ui.viewer_tab import create_viewer_tab
from ui.conversion_tab import create_conversion_tab
from ui.label_manager_tab_new import create_label_manager_tab
from ui.editor_tab import create_editor_tab
from ui.custom_annotator_tab import create_custom_annotator_tab
from ui.handlers import DataLoadingHandlers, ConversionHandlers
from ui.image_handlers import ImageViewerHandlers, ImagePlotToolHandlers
from ui.segmentation_handlers import SegmentationHandlers
from ui.label_manager_handlers import LabelManagerHandlers
from ui.medsam2_handlers import MEDSAM2Handlers
from ui.custom_annotator_handlers import CustomAnnotatorHandlers
from ui.smolvlm_handlers import SmolVLMHandlers
from ui.med_r1_handlers import MedR1Handlers
from ui.medgemma_handlers import MedGemmaHandlers
from ui.medgemma_handlers import MedGemmaHandlers
from ui.patient_retrieval_handlers import PatientRetrievalHandlers
from utils.voice_input import transcribe_voice_input

# Crowdsourcing imports (only imported if authentication is enabled)
if ENABLE_AUTH:
    from ui.login_tab import create_login_interface
    from ui.management_tab import create_management_tab
    from ui.contribute_tab import create_contribute_tab


class SegMedPro:
    """Main application class for SegMed-Pro"""
    
    def __init__(self):
        """Initialize the application"""
        # Initialize application state
        self.state = AppState()
        
        # Crowdsourcing state
        self.current_user = None
        self.crowdsourcing_mode = False
        
        # Initialize checkbox state tracking
        self._point_checkbox_state = False
        
        # Initialize SHARED handlers that can be used across tabs without interference
        self.data_handlers = DataLoadingHandlers(self.state)
        self.conversion_handlers = ConversionHandlers(self.state)
        self.label_manager_handlers = LabelManagerHandlers(self.state)
        self.patient_retrieval_handlers = PatientRetrievalHandlers(self.state)
        self.custom_annotator_handlers = CustomAnnotatorHandlers(self.state)
        
        # Initialize SEPARATE handlers for VIEWER TAB (read-only, no AI annotations)
        self.viewer_segmentation_handlers = SegmentationHandlers(self.state)
        self.viewer_image_handlers = ImageViewerHandlers(self.state, segmentation_handlers=self.viewer_segmentation_handlers)
        
        # Initialize SEPARATE handlers for EDITOR TAB (full AI annotation capabilities)
        self.editor_segmentation_handlers = SegmentationHandlers(self.state)
        self.editor_image_handlers = ImagePlotToolHandlers(self.state)
        self.editor_medsam2_handlers = MEDSAM2Handlers(self.state)
        self.editor_smolvlm_handlers = SmolVLMHandlers(self.state)
        self.editor_med_r1_handlers = MedR1Handlers(self.state)
        self.editor_medgemma_handlers = MedGemmaHandlers(self.state)
        self.editor_medgemma_handlers = MedGemmaHandlers(self.state)
        
        # Connect editor-specific handlers (only editor tab gets AI functionality)
        self.editor_image_handlers.medsam2_handlers = self.editor_medsam2_handlers
        
        # Keep legacy aliases for backward compatibility
        self.segmentation_handlers = self.viewer_segmentation_handlers  # Viewer gets priority for legacy code
        self.image_viewer_handlers = self.viewer_image_handlers
        self.image_plot_tool_handlers = self.editor_image_handlers
        self.medsam2_handlers = self.editor_medsam2_handlers
        self.smolvlm_handlers = self.editor_smolvlm_handlers
        self.med_r1_handlers = self.editor_med_r1_handlers
        self.medgemma_handlers = self.editor_medgemma_handlers
        
        # Preload SmolVLM service in background for instant availability
        self._preload_vlm_service()
        
        # Skip Med-R1 preloading due to meta tensor issues - use lazy loading instead
        self._preload_med_r1_service()
        #logger.info("📋 Med-R1 service will use lazy loading (load on first use) to avoid meta tensor issues")
        
        logger.info("🔄 Tab separation complete - Viewer and Editor tabs now use independent handlers")
        
    def build_interface(self):
        """Build the complete Gradio interface"""
        if ENABLE_AUTH:
            return self._build_authenticated_interface()
        else:
            return self._build_standard_interface()
    
    def _build_authenticated_interface(self):
        """Build interface with authentication and crowdsourcing features"""
        custom_css = """
        .vlm-caption-text textarea {
            font-size: 16px !important;
            line-height: 1.4 !important;
        }
        /* Ensure consistent heights for VLM row components */
        .vlm-row {
            align-items: stretch !important;
        }
        .vlm-row > * {
            height: 100% !important;
        }
        /* Make button match other component heights */
        .vlm-row button {
            height: auto !important;
            min-height: 42px !important;
        }
        /* Ensure checkboxes align properly */
        .vlm-row .gr-checkbox {
            display: flex !important;
            align-items: center !important;
            height: 100% !important;
        }
        /* Style for logout button to match tab height and be red */
        .logout-btn {
            background-color: #dc3545 !important;
            border-color: #dc3545 !important;
            color: white !important;
            height: 42px !important;
            padding: 8px 16px !important;
            font-size: 14px !important;
            border-radius: 6px !important;
        }
        .logout-btn:hover {
            background-color: #c82333 !important;
            border-color: #bd2130 !important;
        }
        /* Align user info and logout button to the right */
        .user-controls {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-end !important;
            gap: 12px !important;
            padding: 8px 0 !important;
        }
        /* Voice input styling */
        .voice-input-row {
            align-items: center !important;
            gap: 8px !important;
        }
        .voice-input-row button {
            min-width: 50px !important;
            height: 42px !important;
            font-size: 18px !important;
            border-radius: 50% !important;
            padding: 8px !important;
        }
        .voice-input-row button:hover {
            background-color: #f0f0f0 !important;
            transform: scale(1.05) !important;
            transition: all 0.2s ease !important;
        }
        .voice-input-row .gr-textbox {
            flex: 4 !important;
        }
        """
        
        with gr.Blocks(title="SegMed-Pro", css=custom_css) as app:
            # Authentication state
            is_logged_in = gr.State(False)
            current_user_state = gr.State(None)
            
            # Login interface (shown initially)
            with gr.Column(visible=True) as login_interface:
                with gr.Row():
                    with gr.Column(scale=1):
                        pass  # Empty column for centering
                    
                    with gr.Column(scale=2):
                        gr.Markdown("# SegMed-Pro: Medical Imaging Annotation Tool")
                        gr.Markdown("## Please login to continue")
                        
                        username_input = gr.Textbox(
                            label="Username",
                            placeholder="Enter your username",
                            value="admin1",
                            #value="jane_smith",
                            interactive=True
                        )
                        
                        password_input = gr.Textbox(
                            label="Password",
                            placeholder="Enter your password",
                            type="password",
                            value="adminpass",
                            #value="expert456",
                            interactive=True
                        )
                        
                        login_btn = gr.Button("Login", variant="primary", size="lg")
                        
                        login_status = gr.Markdown("", visible=False)
                        
                        # Demo credentials info
                        gr.Markdown("""
                        ### Demo Credentials:
                        **Admin:** username: `admin1`, password: `adminpass`  
                        **Expert:** username: `john_doe`, password: `pass123`  
                        **Expert:** username: `jane_smith`, password: `expert456`
                        """)
                    
                    with gr.Column(scale=1):
                        pass  # Empty column for centering
            
            # Main application interface (hidden initially)
            with gr.Column(visible=False) as main_interface:
                # Header with user info and logout aligned to the right
                with gr.Row():
                    with gr.Column(scale=8):
                        gr.Markdown("# SegMed-Pro: Medical Imaging Annotation Tool")
                    
                    with gr.Column(scale=4, min_width=300, elem_classes=["user-controls"]):
                        with gr.Row():
                            user_info = gr.HTML(
                                value="<span style='color: orange; font-weight: bold; font-size: 14px;'></span>",
                                container=False
                            )
                            logout_btn = gr.Button(
                                "Logout", 
                                variant="stop",  # This gives a red button
                                size="sm",
                                scale=0,
                                min_width=80,
                                elem_classes=["logout-btn"]
                            )
                
                with gr.Tabs() as tabs:
                    # Standard tabs - revert back to normal tab creation functions
                    viewer_components = create_viewer_tab()
                    editor_components = create_editor_tab()
                    conversion_components = create_conversion_tab()
                    label_manager_components = create_label_manager_tab()
                    
                    # Admin-only Management tab
                    with gr.Tab("Management", visible=False, id=4) as management_tab:
                        management_components = create_management_tab()
                    
                    # Expert-only Contribute tab
                    with gr.Tab("Contribute", visible=False, id=5) as contribute_tab:
                        contribute_components = create_contribute_tab()
                
                # Hidden state variable to track active tab (for programmatic tab switching)
                active_tab_state = gr.State(value=0)  # Default to Viewer tab (id=0)
            
            # Tab switching function (similar to the demo)
            def change_tab(tab_id):
                """Change to the specified tab"""
                return gr.Tabs(selected=tab_id)
            
            # Login handler
            def handle_login(username, password):
                """Handle login and switch to main app"""
                from auth.auth_manager import AuthManager
                auth_manager = AuthManager()
                
                user_data = auth_manager.authenticate(username, password)
                if user_data:
                    self.current_user = user_data
                    
                    # Show appropriate tabs based on role
                    management_visible = user_data['role'] == 'admin'
                    contribute_visible = user_data['role'] == 'expert'
                    
                    # Create styled user display HTML
                    user_display = f"<div style='text-align: right; padding: 8px 0;'><span style='color: orange; font-weight: bold; font-size: 14px;'>Logged in as: {username} ({user_data['role']})</span></div>"
                    
                    return (
                        True,  # is_logged_in
                        user_data,  # current_user_state
                        gr.update(visible=False),  # Hide login interface
                        gr.update(visible=True),   # Show main interface
                        user_display,  # Update user info
                        gr.update(visible=management_visible),  # Management tab
                        gr.update(visible=contribute_visible),  # Contribute tab
                        user_data['user_id'],  # Update contribute tab user state
                        "Login successful!"  # Login status message
                    )
                else:
                    return (
                        False,  # is_logged_in
                        None,   # current_user_state
                        gr.update(visible=True),   # Keep login interface visible
                        gr.update(visible=False),  # Keep main interface hidden
                        "",     # user_info
                        gr.update(visible=False),  # Management tab
                        gr.update(visible=False),  # Contribute tab
                        "",     # contribute user state
                        "❌ Invalid username or password"  # Login status message
                    )
            
            # Logout handler
            def handle_logout():
                """Handle logout and return to login screen"""
                self.current_user = None
                return (
                    False,  # is_logged_in
                    None,   # current_user_state
                    gr.update(visible=True),   # Show login interface
                    gr.update(visible=False),  # Hide main interface
                    "",     # Clear user info
                    gr.update(visible=False),  # Hide management tab
                    gr.update(visible=False),  # Hide contribute tab
                    "",     # Clear contribute user state
                    "",     # Clear username
                    "",     # Clear password
                    ""      # Clear login status
                )
            
            # Connect login
            login_btn.click(
                fn=handle_login,
                inputs=[username_input, password_input],
                outputs=[
                    is_logged_in,
                    current_user_state,
                    login_interface,
                    main_interface,
                    user_info,
                    management_tab,
                    contribute_tab,
                    contribute_components['current_user_state'],
                    login_status
                ]
            ).then(
                fn=lambda msg: gr.update(value=msg, visible=True),
                inputs=[login_status],
                outputs=[login_status]
            )
            
            # Allow Enter key to trigger login
            password_input.submit(
                fn=handle_login,
                inputs=[username_input, password_input],
                outputs=[
                    is_logged_in,
                    current_user_state,
                    login_interface,
                    main_interface,
                    user_info,
                    management_tab,
                    contribute_tab,
                    contribute_components['current_user_state'],
                    login_status
                ]
            ).then(
                fn=lambda msg: gr.update(value=msg, visible=True),
                inputs=[login_status],
                outputs=[login_status]
            )
            
            # Connect logout
            logout_btn.click(
                fn=handle_logout,
                outputs=[
                    is_logged_in,
                    current_user_state,
                    login_interface,
                    main_interface,
                    user_info,
                    management_tab,
                    contribute_tab,
                    contribute_components['current_user_state'],
                    username_input,
                    password_input,
                    login_status
                ]
            )
            
            # Connect standard handlers
            self._connect_viewer_handlers(viewer_components)
            self._connect_editor_handlers(editor_components)
            self._connect_conversion_handlers(conversion_components)
            self._connect_label_manager_handlers(label_manager_components)
            
            # Connect crowdsourcing handlers if contribute tab exists
            if contribute_components:
                self._connect_contribute_handlers(contribute_components, editor_components, tabs, active_tab_state, change_tab)
            
            return app
    
    def _build_standard_interface(self):
        """Build standard interface without authentication"""
        custom_css = """
        .vlm-caption-text textarea {
            font-size: 16px !important;
            line-height: 1.4 !important;
        }
        /* Ensure consistent heights for VLM row components */
        .vlm-row {
            align-items: stretch !important;
        }
        .vlm-row > * {
            height: 100% !important;
        }
        /* Make button match other component heights */
        .vlm-row button {
            height: auto !important;
            min-height: 42px !important;
        }
        /* Ensure checkboxes align properly */
        .vlm-row .gr-checkbox {
            display: flex !important;
            align-items: center !important;
            height: 100% !important;
        }
        /* Voice input styling */
        .voice-input-row {
            align-items: center !important;
            gap: 8px !important;
        }
        .voice-input-row button {
            min-width: 50px !important;
            height: 42px !important;
            font-size: 18px !important;
            border-radius: 50% !important;
            padding: 8px !important;
        }
        .voice-input-row button:hover {
            background-color: #f0f0f0 !important;
            transform: scale(1.05) !important;
            transition: all 0.2s ease !important;
        }
        .voice-input-row .gr-textbox {
            flex: 4 !important;
        }
        """
        
        with gr.Blocks(title="SegMed-Pro", css=custom_css) as app:
            gr.Markdown("# SegMed-Pro: Medical Imaging Annotation Tool")
            with gr.Tabs() as tabs:
                # Create viewer tab
                viewer_components = create_viewer_tab()
                # Create editor tab
                editor_components = create_editor_tab()
                # Create conversion tab
                conversion_components = create_conversion_tab()
                # Create label manager tab
                label_manager_components = create_label_manager_tab()
                # Create custom annotator tab - HIDDEN
                # custom_annotator_components = create_custom_annotator_tab()
            
            # Connect event handlers for viewer tab
            self._connect_viewer_handlers(viewer_components)
            # Connect event handlers for editor tab
            self._connect_editor_handlers(editor_components)
            # Connect event handlers for conversion tab
            self._connect_conversion_handlers(conversion_components)
            # Connect event handlers for label manager tab
            self._connect_label_manager_handlers(label_manager_components)
            # Connect event handlers for custom annotator tab - HIDDEN
            # self._connect_custom_annotator_handlers(custom_annotator_components)
            
            return app
    
    def _preload_vlm_service(self):
        """Preload SmolVLM service in background thread for instant availability"""
        def load_service():
            try:
                logger.info("🔄 Preloading SmolVLM service in background...")
                # Import here to avoid import issues during module loading
                from models.smolvlm.smolvlm_service import get_service
                
                # Initialize the service (this will load the model)
                service = get_service()
                
                if service.model_loaded:
                    logger.info("✅ SmolVLM service preloaded successfully - ready for instant inference")
                else:
                    logger.warning("⚠️ SmolVLM service preloading failed - model not loaded")
                    
            except Exception as e:
                logger.error(f"❌ Failed to preload SmolVLM service: {e}")
                logger.info("VLM will still work but with slower first inference time")
        
        # Start loading in background thread to not block UI initialization
        preload_thread = threading.Thread(target=load_service, daemon=True)
        preload_thread.start()
        logger.info("🚀 SmolVLM preloading started in background thread")

    def _preload_med_r1_service(self):
        """Preload Med-R1 service in background thread for instant availability"""
        def load_service():
            try:
                logger.info("🔄 Med-R1 preloading disabled to avoid checkpoint conflicts")
                logger.info("Med-R1 will load on first use with proper local checkpoint detection")
                    
            except Exception as e:
                logger.error(f"❌ Failed to preload Med-R1 service: {e}")
                logger.info("Med-R1 will still work but with slower first inference time")
        
        # Start loading in background thread to not block UI initialization
        preload_thread = threading.Thread(target=load_service, daemon=True)
        preload_thread.start()
        logger.info("🚀 Med-R1 preloading started in background thread")

    def _connect_viewer_handlers(self, components):
        """Connect all event handlers for the viewer tab"""
        # Unpack components
        data_loading = components['data_loading']
        visualization = components['visualization']
        viewer_controls = components['viewer_controls']
        segmentation = components['segmentation']
          # Unpack individual components
        (file_input, dir_input, load_btn, reset_dir_btn, patient_search_input,
         clear_search_btn, patient_dropdown, load_retrieval_seg_btn, clear_retrieval_overlays_btn,
         file_browser, metadata_display, error_display, 
         window_level, window_width, apply_window_btn, debug_btn) = data_loading
        (view_selector, image_display, prev_btn, slice_slider, next_btn, 
         slice_text, crosshair_info) = visualization
        (export_format, include_overlays, export_single_btn, export_all_btn, 
         output_dir, export_status) = viewer_controls
        (segmentation_file, load_seg_btn, seg_opacity, label_file, 
         clear_seg_btn, seg_status) = segmentation
        
        # Connect data loading handlers
        load_btn.click(
            fn=self.data_handlers.load_data_for_annotator,
            inputs=[file_input, dir_input],
            outputs=[
                image_display, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width
            ]
        )
        
        reset_dir_btn.click(
            fn=self.data_handlers.reset_directory,
            inputs=[],
            outputs=[dir_input]
        )
        
        # Connect patient retrieval handlers
        patient_search_input.change(
            fn=self.patient_retrieval_handlers.search_patients,
            inputs=[patient_search_input],
            outputs=[patient_dropdown]
        )
        
        clear_search_btn.click(
            fn=self.patient_retrieval_handlers.clear_search,
            inputs=[],
            outputs=[patient_search_input, patient_dropdown]
        )
        
        patient_dropdown.change(
            fn=self.patient_retrieval_handlers.load_selected_patient,
            inputs=[patient_dropdown],
            outputs=[
                image_display, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width, seg_status
            ]
        )
        
        # Connect manual segmentation controls for patient retrieval
        load_retrieval_seg_btn.click(
            fn=self.patient_retrieval_handlers.load_manual_segmentation,
            inputs=[patient_dropdown],
            outputs=[image_display, seg_status]
        )
        
        clear_retrieval_overlays_btn.click(
            fn=self.patient_retrieval_handlers.clear_manual_segmentation,
            inputs=[],
            outputs=[image_display, seg_status]
        )        # Auto-trigger load_data on file selection
        file_input.change(
            fn=self.data_handlers.load_data_for_annotator,
            inputs=[file_input, dir_input],
            outputs=[
                image_display, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width
            ]
        )
        debug_btn.click(
            fn=self.data_handlers.debug_selected_file,
            inputs=[file_input, dir_input],
            outputs=[error_display]
        )
        
        # Connect viewer handlers (using image_annotator-specific methods) (VIEWER-SPECIFIC)
        slice_slider.change(
            fn=self.viewer_image_handlers.update_slice,
            inputs=[slice_slider, view_selector],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width]
        )
        
        view_selector.change(
            fn=self.viewer_image_handlers.change_view,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, image_display]
        )
        
        apply_window_btn.click(
            fn=self.viewer_image_handlers.update_window_level,
            inputs=[window_level, window_width],
            outputs=[image_display]
        )
        file_browser.change(
            fn=self.viewer_image_handlers.select_file_from_browser,
            inputs=[file_browser],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, slice_slider, window_level, window_width]
        )
        
        # Previous/Next button functionality (VIEWER-SPECIFIC)
        prev_btn.click(
            fn=self.viewer_image_handlers.prev_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        next_btn.click(
            fn=self.viewer_image_handlers.next_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        # Connect viewer control handlers (VIEWER-SPECIFIC)        # Connect new export handlers (VIEWER-SPECIFIC)
        export_single_btn.click(
            fn=self.viewer_image_handlers.export_single_slice,
            inputs=[export_format, include_overlays, output_dir, image_display],
            outputs=[export_status]
        )
        
        export_all_btn.click(
            fn=self.viewer_image_handlers.export_all_slices,
            inputs=[export_format, include_overlays, output_dir, image_display],
            outputs=[export_status]
        )
        
        # Connect segmentation handlers (VIEWER-SPECIFIC)
        load_seg_btn.click(
            fn=self.viewer_segmentation_handlers.direct_load_segmentation,
            inputs=[segmentation_file, label_file],
            outputs=[seg_status, image_display]
        )
        
        # Auto-trigger load segmentation when file is selected (VIEWER-SPECIFIC)
        segmentation_file.change(
            fn=self.viewer_segmentation_handlers.direct_load_segmentation,
            inputs=[segmentation_file, label_file],
            outputs=[seg_status, image_display]
        )
          # Bind segmentation file clear (X button) to clear segmentation (VIEWER-SPECIFIC)
        segmentation_file.clear(
            fn=self.viewer_segmentation_handlers.clear_segmentation,
            inputs=[],
            outputs=[seg_status, image_display]
        )
        
        seg_opacity.change(
            fn=self.viewer_segmentation_handlers.update_segmentation_opacity,
            inputs=[seg_opacity],
            outputs=[seg_status, image_display]
        )
        
        clear_seg_btn.click(
            fn=self.viewer_segmentation_handlers.clear_segmentation,
            inputs=[],
            outputs=[seg_status, image_display]
        )
          # Update image_annotator labels/colors when a .label file is loaded (VIEWER-SPECIFIC)
        label_file.change(
            fn=self.viewer_segmentation_handlers.load_label_file_and_update_annotator,
            inputs=[label_file],
            outputs=[seg_status, image_display]
        )
        
        # Handle user annotation changes in the viewer tab (VIEWER-SPECIFIC)
        image_display.change(
            fn=self.viewer_image_handlers.on_annotation_change_viewer,
            inputs=[image_display],
            outputs=[]  # No outputs to avoid circular dependency
        )
        
        # Handle image removal (X button) - Clear image_annotator on viewer tab (VIEWER-SPECIFIC)
        image_display.clear(
            fn=self.viewer_image_handlers.handle_image_remove_viewer,
            inputs=[],
            outputs=[image_display]
        )

    # Plot tool handlers removed - image_annotator has built-in annotation tools
    
    def _connect_conversion_handlers(self, conversion_components):
        """Connect event handlers for the conversion tab"""
        (conversion_input, conversion_dir, conversion_type, output_dir,
         convert_btn, conversion_status, conversion_output) = conversion_components
        
        convert_btn.click(
            fn=self.conversion_handlers.convert_files,
            inputs=[conversion_input, conversion_dir, conversion_type, output_dir],
            outputs=[conversion_status, conversion_output]
        )

    def _connect_label_manager_handlers(self, components):
        """Connect event handlers for the label manager tab"""
        (label_file_input, load_btn, save_btn, save_quick_btn, save_filename, label_table, 
         new_label_name, new_label_color, add_label_btn,
         selected_label_name, delete_label_btn, status_box,
         update_label_dropdown) = components  # updated to receive update_label_dropdown
        
        handlers = self.label_manager_handlers        # Load label set
        load_btn.click(
            fn=handlers.load_label_set,
            inputs=[label_file_input],
            outputs=[label_table, status_box]
        ).then(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name]
        )
        
        # Sync table edits back to internal storage when user edits table directly
        label_table.change(
            fn=handlers.sync_table_to_internal,
            inputs=[label_table],
            outputs=[]
        )
        
        # Save label set with custom filename
        save_btn.click(
            fn=lambda table_data, filename: handlers.save_label_set_with_filename(table_data, filename)[1],
            inputs=[label_table, save_filename],
            outputs=[status_box]
        )
          # Quick save with default filename
        save_quick_btn.click(
            fn=lambda table_data: handlers.save_label_set(table_data)[1],  # Return only status message
            inputs=[label_table],
            outputs=[status_box]
        )
        
        # Add new label
        add_label_btn.click(
            fn=handlers.add_new_label,
            inputs=[label_table, new_label_name, new_label_color],
            outputs=[label_table, status_box]
        ).then(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name]
        )
          # Sync table changes to internal storage when user edits the table
        label_table.change(
            fn=handlers.sync_table_to_internal,
            inputs=[label_table],
            outputs=[]
        )
        
        # Delete selected label (by name)
        delete_label_btn.click(
            fn=handlers.delete_label_by_name,
            inputs=[label_table, selected_label_name],
            outputs=[label_table, status_box]
        ).then(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name]
        )
    
    def _connect_custom_annotator_handlers(self, components):
        """Connect event handlers for the custom annotator tab"""
        # Unpack components following the new structure
        data_loading = components['data_loading']
        visualization = components['visualization']
        point_selection = components['point_selection']
        annotator_components = components['annotator']
        
        # Unpack data loading components
        (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
         metadata_display, error_display, window_level, window_width, 
         apply_window_btn, debug_btn) = data_loading
        
        # Unpack visualization components
        (view_selector, prev_btn, slice_slider, next_btn, slice_text) = visualization
        
        # Unpack point selection components
        (coordinates_text, clear_coords_btn, clear_overlays_btn) = point_selection
        
        # Unpack annotator components
        annotator = annotator_components['main']
        opacity_slider = annotator_components['opacity_slider']
        thickness_slider = annotator_components['thickness_slider']
        min_size_slider = annotator_components['min_size_slider']
        new_label_input = annotator_components['new_label_input']
        add_label_btn = annotator_components['add_label_btn']
        current_labels = annotator_components['current_labels']
        export_format = annotator_components['export_format']
        export_btn = annotator_components['export_btn']
        export_output = annotator_components['export_output']
        stats_display = annotator_components['stats_display']
        refresh_stats_btn = annotator_components['refresh_stats_btn']
        batch_process_btn = annotator_components['batch_process_btn']
        batch_status = annotator_components['batch_status']
        
        # Connect data loading handlers
        load_btn.click(
            fn=self.custom_annotator_handlers.load_data_for_annotator,
            inputs=[file_input, dir_input],
            outputs=[
                annotator, error_display, metadata_display, 
                window_level, window_width, slice_slider, slice_text, file_browser
            ]
        )
        
        reset_dir_btn.click(
            fn=self.custom_annotator_handlers.reset_directory,
            inputs=[],
            outputs=[dir_input]
        )
        
        # Auto-trigger load_data on file selection
        file_input.change(
            fn=self.custom_annotator_handlers.load_data_for_annotator,
            inputs=[file_input, dir_input],
            outputs=[
                annotator, error_display, metadata_display, 
                window_level, window_width, slice_slider, slice_text, file_browser
            ]
        )
        
        debug_btn.click(
            fn=self.custom_annotator_handlers.debug_selected_file,
            inputs=[file_input, dir_input],
            outputs=[error_display]
        )
        
        # Connect browser selection handler
        file_browser.change(
            fn=self.custom_annotator_handlers.select_file_from_browser,
            inputs=[file_browser],
            outputs=[
                annotator, error_display, metadata_display, 
                window_level, window_width, slice_slider, slice_text
            ]
        )
        
        # Connect navigation handlers
        prev_btn.click(
            fn=self.custom_annotator_handlers.prev_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        next_btn.click(
            fn=self.custom_annotator_handlers.next_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        slice_slider.change(
            fn=self.custom_annotator_handlers.update_slice,
            inputs=[slice_slider, view_selector],
            outputs=[annotator, slice_text]
        )
        
        view_selector.change(
            fn=self.custom_annotator_handlers.change_view,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, annotator]
        )
        
        apply_window_btn.click(
            fn=self.custom_annotator_handlers.update_window_level,
            inputs=[window_level, window_width],
            outputs=[annotator]
        )
        
        # Settings update handlers
        opacity_slider.change(
            fn=self.custom_annotator_handlers.update_annotation_settings,
            inputs=[opacity_slider, thickness_slider, min_size_slider],
            outputs=[error_display]
        )
        
        thickness_slider.change(
            fn=self.custom_annotator_handlers.update_annotation_settings,
            inputs=[opacity_slider, thickness_slider, min_size_slider],
            outputs=[error_display]
        )
        
        min_size_slider.change(
            fn=self.custom_annotator_handlers.update_annotation_settings,
            inputs=[opacity_slider, thickness_slider, min_size_slider],
            outputs=[error_display]
        )
        
        # Label management
        add_label_btn.click(
            fn=self.custom_annotator_handlers.add_new_label,
            inputs=[new_label_input],
            outputs=[current_labels]
        )
        
        # Statistics refresh
        refresh_stats_btn.click(
            fn=self.custom_annotator_handlers.calculate_annotation_stats,
            inputs=[annotator],
            outputs=[stats_display]
        )
        
        # Export functionality
        export_btn.click(
            fn=self.custom_annotator_handlers.export_annotations,
            inputs=[annotator, export_format],
            outputs=[export_output, error_display]
        )
          # Batch processing
        batch_process_btn.click(
            fn=self.custom_annotator_handlers.process_batch_images,
            inputs=[],
            outputs=[batch_status]
        )          # Connect Point Selection (Custom Annotator) handlers
        # The image_annotator now supports Events.select after we added it to EVENTS list
        annotator.select(
            fn=self.custom_annotator_handlers.handle_image_select,
            inputs=[],
            outputs=[coordinates_text]
        )
          # Also keep change event for annotation updates
        annotator.change(
            fn=self.custom_annotator_handlers.on_annotation_change_editor,
            inputs=[annotator],
            outputs=[]
        )
        
        clear_coords_btn.click(
            fn=self.custom_annotator_handlers.clear_selected_coordinates,
            inputs=[],
            outputs=[coordinates_text]
        )
        
        clear_overlays_btn.click(
            fn=self.editor_medsam2_handlers.clear_annotation_overlays,
            inputs=[],
            outputs=[error_display, annotator]
        )

    def _connect_editor_handlers(self, components):
        """Connect all event handlers for the editor tab (mirroring viewer tab, plus MEDSAM2 functionality)"""
        data_loading = components['data_loading']
        visualization = components['visualization']
        ai_tools = components['ai_tools']  # Now we have AI tools including MEDSAM2        
        # Unpack components
        (file_input, dir_input, load_btn, reset_dir_btn, file_browser, label_file,
         metadata_display_dl, error_display_dl, window_level, window_width, apply_window_btn, debug_btn) = data_loading
        (error_display, metadata_display, view_selector, image_display, image_column, viewer_3d_column, prev_btn, slice_slider, next_btn, slice_text, crosshair_info, 
         crowdsourcing_accordion, submit_annotation_btn, assignments_remaining, next_assignment_btn, crowdsourcing_status,
         current_labels_dataset, suggested_vlm_selector, suggest_labels_btn, suggested_labels_dataset, accept_suggestions_btn, vlm_model_selector, vlm_run_btn, vlm_suggest_labels_btn, vlm_caption, vlm_prompt_anomalies, vlm_prompt_describe, viewer_3d, viewer_3d_controls, refresh_3d_btn, export_3d_btn, voice_prompt_text, voice_audio_input) = visualization        
        (point_prompt_checkbox, box_prompt_checkbox, coordinates_text, clear_coords_btn, ai_model_selector, 
         processing_mode, score_threshold,
         output_dir, save_visualizations, device_selector, 
         auto_brain_annotate_btn, annotate_btn, annotation_status) = ai_tools
        
        # Get layout functions
        layout_functions = components.get('layout_functions', {})
        toggle_layout = layout_functions.get('toggle_layout')
        reset_layout = layout_functions.get('reset_layout')
          # Data loading handlers (updated for annotator compatibility)
        load_btn.click(
            fn=self.data_handlers.load_data_for_annotator,
            inputs=[file_input, dir_input],
            outputs=[
                image_display, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width
            ]
        )
        reset_dir_btn.click(
            fn=self.data_handlers.reset_directory,
            inputs=[],
            outputs=[dir_input]
        )
        file_input.change(
            fn=self.data_handlers.load_data_for_annotator,
            inputs=[file_input, dir_input],
            outputs=[
                image_display, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width
            ]
        )
        
        debug_btn.click(
            fn=self.data_handlers.debug_selected_file,
            inputs=[file_input, dir_input],
            outputs=[error_display]
        )
        
        # Label file handler for updating image_annotator labels (EDITOR-SPECIFIC)
        label_file.change(
            fn=self.editor_segmentation_handlers.load_label_file_and_update_annotator,
            inputs=[label_file],
            outputs=[annotation_status, image_display]
        )
        
        # Viewer handlers (using annotator-specific methods) (EDITOR-SPECIFIC)
        def handle_slice_change_with_labels(slice_value, image_annotator_value):
            """Handle slice slider change and update current labels and suggested labels"""
            # Get the main slider change result
            result = self.editor_image_handlers.handle_annotator_slider_change(slice_value, image_annotator_value)
            # result contains: (image_display, slice_text, crosshair_info, metadata_display, window_level, window_width)
            
            # Extract current labels from the updated image and load saved labels
            from ui.editor_tab import (extract_current_labels_from_annotator, create_labels_dataset_samples, 
                                     get_suggested_labels_for_slice, clear_selected_suggested_labels_for_slice,
                                     load_labels_from_file)
            try:
                # Use the first element of result which should be the updated image_display
                updated_image = result[0] if result else image_annotator_value
                
                # Extract labels from the image annotator
                annotator_labels = extract_current_labels_from_annotator(updated_image)
                
                # Load saved labels from file if we have a data directory
                saved_labels = []
                data_directory = getattr(self.state, 'current_directory', None)
                if data_directory:
                    saved_labels = load_labels_from_file(data_directory, slice_value)
                
                # Combine annotator labels and saved labels (avoid duplicates)
                all_labels = list(annotator_labels)
                for label in saved_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                
                labels_samples = create_labels_dataset_samples(all_labels)
                current_labels_dataset = gr.Dataset(samples=labels_samples)
                
                # Get suggested labels for this slice (without selection indicators)
                suggested_samples = get_suggested_labels_for_slice(slice_value)
                suggested_labels_dataset = gr.Dataset(samples=suggested_samples)
                
                # Clear any previous selections when changing slices
                clear_selected_suggested_labels_for_slice(slice_value)
                accept_btn_hidden = gr.update(visible=False)
                
                logger.info(f"Slice {slice_value} labels: {len(annotator_labels)} from annotator + {len(saved_labels)} saved = {len(all_labels)} total")
                
            except Exception as e:
                logger.error(f"Error updating labels on slice change: {e}")
                current_labels_dataset = gr.Dataset(samples=[])
                suggested_labels_dataset = gr.Dataset(samples=[])
                accept_btn_hidden = gr.update(visible=False)
            
            # Return original result plus both label datasets and hidden accept button
            return result + (current_labels_dataset, suggested_labels_dataset, accept_btn_hidden)
        
        slice_slider.change(
            fn=handle_slice_change_with_labels,
            inputs=[slice_slider, image_display],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width, current_labels_dataset, suggested_labels_dataset, accept_suggestions_btn]
        )
        def handle_view_change_with_labels(view_value):
            """Handle view selector change and update current labels and suggested labels"""
            # Get the main view change result
            result = self.editor_image_handlers.change_view_for_annotator(view_value)
            # result contains: (slice_slider, slice_text, image_display)
            
            # Extract current labels from the updated image and load saved labels
            from ui.editor_tab import (extract_current_labels_from_annotator, create_labels_dataset_samples, 
                                     get_suggested_labels_for_slice, clear_selected_suggested_labels_for_slice,
                                     load_labels_from_file)
            try:
                # Use the third element of result which should be the updated image_display
                updated_image = result[2] if len(result) > 2 else None
                
                # Extract labels from the image annotator
                annotator_labels = extract_current_labels_from_annotator(updated_image)
                
                # Load saved labels from file if we have a data directory
                saved_labels = []
                data_directory = getattr(self.state, 'current_directory', None)
                current_slice_idx = self.state.current_slice_idx
                if data_directory:
                    saved_labels = load_labels_from_file(data_directory, current_slice_idx)
                
                # Combine annotator labels and saved labels (avoid duplicates)
                all_labels = list(annotator_labels)
                for label in saved_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                
                labels_samples = create_labels_dataset_samples(all_labels)
                current_labels_dataset = gr.Dataset(samples=labels_samples)
                
                # Get suggested labels for current slice
                suggested_samples = get_suggested_labels_for_slice(current_slice_idx)
                suggested_labels_dataset = gr.Dataset(samples=suggested_samples)
                
                # Clear selections when changing views
                clear_selected_suggested_labels_for_slice(current_slice_idx)
                accept_btn_hidden = gr.update(visible=False)
                
                logger.info(f"View change - Slice {current_slice_idx} labels: {len(annotator_labels)} from annotator + {len(saved_labels)} saved = {len(all_labels)} total")
                
            except Exception as e:
                logger.error(f"Error updating labels on view change: {e}")
                current_labels_dataset = gr.Dataset(samples=[])
                suggested_labels_dataset = gr.Dataset(samples=[])
                accept_btn_hidden = gr.update(visible=False)
            
            # Return original result plus both label datasets and hidden accept button
            return result + (current_labels_dataset, suggested_labels_dataset, accept_btn_hidden)
        
        view_selector.change(
            fn=handle_view_change_with_labels,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, image_display, current_labels_dataset, suggested_labels_dataset, accept_suggestions_btn]
        )
        apply_window_btn.click(
            fn=self.editor_image_handlers.update_window_level_for_annotator,
            inputs=[window_level, window_width],
            outputs=[image_display]
        )
        file_browser.change(
            fn=self.editor_image_handlers.select_file_from_browser_for_annotator,
            inputs=[file_browser],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, slice_slider, window_level, window_width]        )
          # Navigation buttons with annotation saving
        def handle_prev_navigation(current_slider_value, current_annotated_value):
            """Handle previous button click with annotation saving (EDITOR-SPECIFIC)"""
            result_tuple, new_slider_value = self.editor_image_handlers.handle_annotator_navigation(
                "prev", current_slider_value, current_annotated_value
            )
            # result_tuple contains: (image_display, slice_text, crosshair_info, metadata, window_level, window_width)
            
            # Extract current labels from the updated image and get suggested labels
            from ui.editor_tab import (extract_current_labels_from_annotator, create_labels_dataset_samples, 
                                     get_suggested_labels_for_slice, clear_selected_suggested_labels_for_slice,
                                     load_labels_from_file)
            try:
                updated_image = result_tuple[0] if result_tuple else current_annotated_value
                
                # Extract labels from the image annotator
                annotator_labels = extract_current_labels_from_annotator(updated_image)
                
                # Load saved labels from file if we have a data directory
                saved_labels = []
                data_directory = getattr(self.state, 'current_directory', None)
                if data_directory:
                    saved_labels = load_labels_from_file(data_directory, new_slider_value)
                
                # Combine annotator labels and saved labels (avoid duplicates)
                all_labels = list(annotator_labels)
                for label in saved_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                
                labels_samples = create_labels_dataset_samples(all_labels)
                current_labels_dataset = gr.Dataset(samples=labels_samples)
                
                # Get suggested labels for the new slice
                suggested_samples = get_suggested_labels_for_slice(new_slider_value)
                suggested_labels_dataset = gr.Dataset(samples=suggested_samples)
                
                # Clear selections when navigating
                clear_selected_suggested_labels_for_slice(new_slider_value)
                accept_btn_hidden = gr.update(visible=False)
                
                logger.info(f"Prev navigation - Slice {new_slider_value} labels: {len(annotator_labels)} from annotator + {len(saved_labels)} saved = {len(all_labels)} total")
                
            except Exception as e:
                logger.error(f"Error updating labels on prev navigation: {e}")
                current_labels_dataset = gr.Dataset(samples=[])
                suggested_labels_dataset = gr.Dataset(samples=[])
                accept_btn_hidden = gr.update(visible=False)
            
            return result_tuple + (new_slider_value, current_labels_dataset, suggested_labels_dataset, accept_btn_hidden)
        
        def handle_next_navigation(current_slider_value, current_annotated_value):
            """Handle next button click with annotation saving (EDITOR-SPECIFIC)"""  
            result_tuple, new_slider_value = self.editor_image_handlers.handle_annotator_navigation(
                "next", current_slider_value, current_annotated_value
            )
            # result_tuple contains: (image_display, slice_text, crosshair_info, metadata, window_level, window_width)
            
            # Extract current labels from the updated image and get suggested labels
            from ui.editor_tab import (extract_current_labels_from_annotator, create_labels_dataset_samples, 
                                     get_suggested_labels_for_slice, clear_selected_suggested_labels_for_slice,
                                     load_labels_from_file)
            try:
                updated_image = result_tuple[0] if result_tuple else current_annotated_value
                
                # Extract labels from the image annotator
                annotator_labels = extract_current_labels_from_annotator(updated_image)
                
                # Load saved labels from file if we have a data directory
                saved_labels = []
                data_directory = getattr(self.state, 'current_directory', None)
                if data_directory:
                    saved_labels = load_labels_from_file(data_directory, new_slider_value)
                
                # Combine annotator labels and saved labels (avoid duplicates)
                all_labels = list(annotator_labels)
                for label in saved_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                
                labels_samples = create_labels_dataset_samples(all_labels)
                current_labels_dataset = gr.Dataset(samples=labels_samples)
                
                # Get suggested labels for the new slice
                suggested_samples = get_suggested_labels_for_slice(new_slider_value)
                suggested_labels_dataset = gr.Dataset(samples=suggested_samples)
                
                # Clear selections when navigating
                clear_selected_suggested_labels_for_slice(new_slider_value)
                accept_btn_hidden = gr.update(visible=False)
                
                logger.info(f"Next navigation - Slice {new_slider_value} labels: {len(annotator_labels)} from annotator + {len(saved_labels)} saved = {len(all_labels)} total")
                
            except Exception as e:
                logger.error(f"Error updating labels on next navigation: {e}")
                current_labels_dataset = gr.Dataset(samples=[])
                suggested_labels_dataset = gr.Dataset(samples=[])
                accept_btn_hidden = gr.update(visible=False)
            
            return result_tuple + (new_slider_value, current_labels_dataset, suggested_labels_dataset, accept_btn_hidden)
        
        prev_btn.click(
            fn=handle_prev_navigation,
            inputs=[slice_slider, image_display],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width, slice_slider, current_labels_dataset, suggested_labels_dataset, accept_suggestions_btn]
        )
        
        next_btn.click(            fn=handle_next_navigation,            inputs=[slice_slider, image_display],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width, slice_slider, current_labels_dataset, suggested_labels_dataset, accept_suggestions_btn]
        )
          # Connect processing mode change event with layout switching
        def handle_processing_mode_change(mode):
            """Handle processing mode change - switch layout and show/hide score threshold and 3D viewer"""
            score_threshold_update = gr.Slider(visible=(mode == "All Records"))
            # Use the layout toggle function
            if toggle_layout:
                layout_updates = toggle_layout(mode)
                return (score_threshold_update,) + layout_updates
            else:
                # Fallback if toggle_layout is not available
                return score_threshold_update, gr.update(), gr.update(), gr.update()
        
        processing_mode.change(
            fn=handle_processing_mode_change,
            inputs=[processing_mode],
            outputs=[score_threshold, image_column, viewer_3d_column, image_display]
        )
          # 3D Viewer event handlers
        def handle_refresh_3d_view(output_dir_value):
            """Handle refresh 3D view button click (EDITOR-SPECIFIC)"""
            try:
                # Import the 3D visualization function from editor_tab
                from ui.editor_tab import create_3d_visualization
                # Get the current score threshold from editor medsam2_handlers
                current_score_threshold = getattr(self.editor_medsam2_handlers, 'score_threshold', 0.3)
                fig = create_3d_visualization(output_dir_value, current_score_threshold)
                return fig
            except Exception as e:
                from ui.editor_tab import create_empty_3d_plot
                return create_empty_3d_plot(f"Error refreshing view: {str(e)}")
        
        def handle_export_3d_view(output_dir_value):
            """Handle export 3D view button click (EDITOR-SPECIFIC)"""
            try:
                from ui.editor_tab import create_3d_visualization
                import os
                # Get the current score threshold from editor medsam2_handlers
                current_score_threshold = getattr(self.editor_medsam2_handlers, 'score_threshold', 0.3)
                fig = create_3d_visualization(output_dir_value, current_score_threshold)
                export_path = os.path.join(output_dir_value, "3d_visualization.html")
                fig.write_html(export_path)
                return f"✅ 3D view exported to: {export_path}"
            except Exception as e:
                return f"❌ Error exporting 3D view: {str(e)}"
        
        # Connect 3D viewer button handlers
        refresh_3d_btn.click(
            fn=handle_refresh_3d_view,
            inputs=[output_dir],
            outputs=[viewer_3d]
        )
        
        export_3d_btn.click(
            fn=handle_export_3d_view,
            inputs=[output_dir],
            outputs=[annotation_status]  # Show export status in annotation_status
        )
        
        # Connect checkbox handlers for prompt type selection
        def handle_point_prompt_change(point_checked, box_checked):
            """Handle point prompt checkbox change - ensure mutual exclusivity and clear coords (EDITOR-SPECIFIC)"""
            if point_checked:
                # If point is checked, uncheck box and show coordinates
                self.editor_medsam2_handlers.enable_point_mode()
                return True, False, gr.update(visible=True)
            else:                # If point is unchecked, hide coordinates and clear them
                self.editor_medsam2_handlers.disable_point_mode()
                self.editor_medsam2_handlers.selected_coordinates = []  # Clear coordinates from handler state
                return False, box_checked, gr.update(visible=False)
        
        def handle_box_prompt_change(box_checked, point_checked):
            """Handle box prompt checkbox change - ensure mutual exclusivity and enable box mode (EDITOR-SPECIFIC)"""
            if box_checked:
                # If box is checked, uncheck point, hide coordinates, and enable box mode
                self.editor_medsam2_handlers.disable_point_mode()
                self.editor_medsam2_handlers.enable_box_mode()
                self.editor_medsam2_handlers.selected_coordinates = []  # Clear coordinates from handler state
                self.editor_medsam2_handlers.prompt_boxes = []  # Clear any existing box prompts                # Show instructions for box mode
                return (True, False, 
                       gr.update(visible=True, 
                               label="Box Prompt Status", 
                               value="📦 Box mode enabled. Ready to capture box coordinates.\n\nSteps:\n1. Use the box tool in the image editor above\n2. Draw rectangles around regions to segment\n3. Box coordinates are automatically captured\n4. Click 'Run MEDSAM2 Annotation' when ready",
                               info="Draw boxes on the image. Coordinates are automatically captured for MEDSAM2."))
            else:
                # If box is unchecked, disable box mode and keep point state
                self.editor_medsam2_handlers.disable_box_mode()
                if point_checked:
                    self.editor_medsam2_handlers.enable_point_mode()
                    return (False, True, 
                           gr.update(visible=True, 
                                   label="Selected Coordinates (x,y)", 
                                   value="",
                                   info="Click on the image to select coordinates"))
                else:
                    self.editor_medsam2_handlers.disable_point_mode()
                    return (False, False, 
                           gr.update(visible=True, 
                                   label="Mode Status", 
                                   value="📝 Box mode disabled. Box prompts cleared. Use the clear button to remove any remaining shapes if needed.",
                                   info="No prompt mode selected. Check Point-based or Box-based prompt to continue."))
        
        point_prompt_checkbox.change(
            fn=handle_point_prompt_change,
            inputs=[point_prompt_checkbox, box_prompt_checkbox],
            outputs=[point_prompt_checkbox, box_prompt_checkbox, coordinates_text]        )
        
        box_prompt_checkbox.change(
            fn=handle_box_prompt_change,
            inputs=[box_prompt_checkbox, point_prompt_checkbox],
            outputs=[box_prompt_checkbox, point_prompt_checkbox, coordinates_text]
        )        # VLM checkbox mutual exclusivity handlers
        def handle_anomalies_checkbox_change(anomalies_checked, describe_checked):
            """Handle anomalies checkbox change - ensure mutual exclusivity"""
            if anomalies_checked:
                # When anomalies is clicked and becomes checked, uncheck describe
                return True, False  # (anomalies=True, describe=False)
            else:
                # When anomalies is clicked and becomes unchecked, check describe
                return False, True  # (anomalies=False, describe=True)
        
        def handle_describe_checkbox_change(describe_checked, anomalies_checked):
            """Handle describe checkbox change - ensure mutual exclusivity"""
            if describe_checked:
                # When describe is clicked and becomes checked, uncheck anomalies
                return True, False  # (describe=True, anomalies=False)
            else:
                # When describe is clicked and becomes unchecked, check anomalies
                return False, True  # (describe=False, anomalies=True)
        
        vlm_prompt_anomalies.change(
            fn=handle_anomalies_checkbox_change,
            inputs=[vlm_prompt_anomalies, vlm_prompt_describe],
            outputs=[vlm_prompt_anomalies, vlm_prompt_describe]
        )
        
        vlm_prompt_describe.change(
            fn=handle_describe_checkbox_change,
            inputs=[vlm_prompt_describe, vlm_prompt_anomalies],
            outputs=[vlm_prompt_describe, vlm_prompt_anomalies]
        )
        
        def handle_image_select_conditionally(evt: gr.SelectData):
            """Handle image select events only when point mode is enabled, ignore in box mode (EDITOR-SPECIFIC)"""
            # Check if point mode is enabled
            if (hasattr(self.editor_medsam2_handlers, 'point_mode_enabled') and 
                self.editor_medsam2_handlers.point_mode_enabled and 
                not getattr(self.editor_medsam2_handlers, 'box_mode_enabled', False)):
                return self.editor_medsam2_handlers.handle_image_click(evt)
            else:
                # In box mode or when point mode is disabled, don't process select events at all
                return ""
        
        # Connect MEDSAM2 handlers - conditional based on mode (EDITOR-SPECIFIC)
        image_display.select(
            fn=handle_image_select_conditionally,
            inputs=[],
            outputs=[coordinates_text]
        )
        
        def handle_image_annotation_change(annotated_image_value):
            """Handle annotation changes, box prompt extraction, and label updates (EDITOR-SPECIFIC)"""
            try:                
                # First, let the normal annotation handler process the change
                self.editor_image_handlers.on_annotation_change_editor_save(annotated_image_value)
                
                # If box mode is enabled, also extract box prompts for MEDSAM2
                if hasattr(self.editor_medsam2_handlers, 'box_mode_enabled') and self.editor_medsam2_handlers.box_mode_enabled:
                    status_msg = self.editor_medsam2_handlers.handle_box_annotation(annotated_image_value)
                    logger.info(f"Annotation Status: Box annotation handled - {status_msg}")
                
                # Update current labels dataset - combine annotator labels with saved labels
                from ui.editor_tab import (extract_current_labels_from_annotator, create_labels_dataset_samples,
                                         load_labels_from_file)
                
                # Extract labels from the image annotator
                annotator_labels = extract_current_labels_from_annotator(annotated_image_value)
                
                # Load saved labels from file if we have a data directory
                saved_labels = []
                data_directory = getattr(self.state, 'current_directory', None)
                current_slice_idx = getattr(self.state, 'current_slice_idx', 0)
                if data_directory:
                    saved_labels = load_labels_from_file(data_directory, current_slice_idx)
                
                # Combine annotator labels and saved labels (avoid duplicates)
                all_labels = list(annotator_labels)
                for label in saved_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                
                # Create dataset samples with combined labels
                samples = create_labels_dataset_samples(all_labels)
                
                logger.info(f"Annotation change - Slice {current_slice_idx} labels: {len(annotator_labels)} from annotator + {len(saved_labels)} saved = {len(all_labels)} total")
                
                # Return gr.Dataset update with combined labels
                return gr.Dataset(samples=samples)
            except Exception as e:
                logger.error(f"Error handling image annotation change: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return []
          # Save annotations immediately when they change (edits, deletions, additions) AND update current labels
        image_display.change(
            fn=handle_image_annotation_change,
            inputs=[image_display],
            outputs=[current_labels_dataset]
        )
        
        def clear_all_prompts_and_reset_layout():
            """Clear all prompts, coordinates, annotation overlays, and reset layout (EDITOR-SPECIFIC)"""
            self.editor_medsam2_handlers.clear_coordinates()
            self.editor_medsam2_handlers.prompt_boxes = []
            status, image = self.editor_medsam2_handlers.clear_annotation_overlays()
            
            # Use the reset layout function
            if reset_layout:
                layout_updates = reset_layout()                # layout_updates contains: (image_column, viewer_3d_column, image_display, viewer_3d, coordinates, processing_mode)
                # We need: (coordinates_text, annotation_status, image_display, image_column, viewer_3d_column, viewer_3d, processing_mode)
                return (
                    layout_updates[4],  # coordinates ("")
                    status,             # annotation_status
                    image,              # image_display (cleared image)
                    layout_updates[0],  # image_column (scale=10)
                    layout_updates[1],  # viewer_3d_column (scale=6, visible=False)
                    layout_updates[3],  # viewer_3d (reset plot)
                    layout_updates[5]   # processing_mode ("Single Slice")
                )
            else:
                # Fallback if reset_layout is not available
                return "", status, image, gr.update(), gr.update(), gr.update(), "Single Slice"
        
        clear_coords_btn.click(
            fn=clear_all_prompts_and_reset_layout,
            inputs=[],
            outputs=[coordinates_text, annotation_status, image_display, image_column, viewer_3d_column, viewer_3d, processing_mode]
        )
        
        # Custom wrapper function to handle the 3-tuple return and button visibility (EDITOR-SPECIFIC)
        def handle_annotation_workflow(output_dir, save_visualizations, device_selector, processing_mode, score_threshold, image_display_data):
            logger.info(f"Annotation Status: workflow started with processing_mode={processing_mode}")
            status, image, success = self.editor_medsam2_handlers.run_full_annotation_workflow(
                output_dir, save_visualizations, device_selector, processing_mode, score_threshold, image_display_data
            )
            logger.info(f"Annotation Status: workflow completed with success={success}")
            # If processing mode is "All Records" and annotation was successful, refresh 3D viewer
            # Always return a valid Plotly figure to avoid the __module__ attribute error
            from ui.editor_tab import create_empty_3d_plot, create_3d_visualization
            if processing_mode == "All Records" and success:
                logger.info(f"3D Viewer status: Updating 3D viewer for All Records mode with output_dir={output_dir}")
                try:
                    viewer_3d_update = create_3d_visualization(output_dir, score_threshold)
                    logger.info("3D Viewer status: 3D viewer updated successfully")
                except Exception as e:
                    logger.info(f"3D Viewer status: Error updating 3D view: {str(e)}")
                    viewer_3d_update = create_empty_3d_plot(f"Error updating 3D view: {str(e)}")
            else:
                # For non-"All Records" mode, show message
                logger.info(f"3D Viewer status: Not updating 3D viewer: mode={processing_mode}, success={success}")
                viewer_3d_update = create_empty_3d_plot("3D Viewer available in 'All Records' mode")
            
            return status, image, viewer_3d_update
        
        annotate_btn.click(
            fn=handle_annotation_workflow,
            inputs=[output_dir, save_visualizations, device_selector, processing_mode, score_threshold, image_display],
            outputs=[annotation_status, image_display, viewer_3d]
        )          # Unified VLM handler for all models (EDITOR-SPECIFIC)
        def handle_vlm_inference(vlm_model, image_display, identify_anomalies, describe_slice):
            """Handle VLM inference based on selected model"""
            try:
                if vlm_model == "SmolVLM":
                    return self.editor_smolvlm_handlers.run_vlm_inference(
                        image_display, identify_anomalies, describe_slice
                    )
                elif vlm_model == "Med-R1":
                    return self.editor_med_r1_handlers.run_med_r1_inference(
                        image_display, identify_anomalies, describe_slice
                    )
                elif vlm_model == "MedGemma-4B":
                    return self.editor_medgemma_handlers.run_vlm_inference(
                        image_display, identify_anomalies, describe_slice
                    )
                else:
                    return f"Unknown VLM model: {vlm_model}"
            except Exception as e:
                logger.error(f"Error in VLM inference: {e}")
                return f"Error during VLM analysis: {str(e)}"

        # Med-R1 label suggestion button handler for annotated shapes (EDITOR-SPECIFIC)
        vlm_suggest_labels_btn.click(
            fn=self.editor_med_r1_handlers.suggest_labels_for_annotations,
            inputs=[image_display],
            outputs=[vlm_caption]
        )
        
        # Voice Input Handlers
        def handle_voice_transcription(audio_data):
            """Handle voice transcription from recorded audio - optimized for English speech"""
            try:
                if audio_data is None:
                    return "No audio recorded"
                
                # Debug audio data format
                if isinstance(audio_data, tuple) and len(audio_data) == 2:
                    sample_rate, audio_array = audio_data
                    logger.info(f"Voice input debug: sample_rate={sample_rate}, audio_shape={audio_array.shape if audio_array is not None else 'None'}, audio_dtype={audio_array.dtype if audio_array is not None else 'None'}")
                    if audio_array is not None and len(audio_array) > 0:
                        logger.info(f"Audio stats: min={np.min(audio_array):.6f}, max={np.max(audio_array):.6f}, duration={len(audio_array)/sample_rate:.2f}s")
                elif isinstance(audio_data, str):
                    logger.info(f"Voice input debug: file_path={audio_data}")
                else:
                    logger.info(f"Voice input debug: unexpected format={type(audio_data)}")
                
                transcribed_text, success = transcribe_voice_input(audio_data)
                if success:
                    logger.info(f"Voice transcription successful: '{transcribed_text}'")
                    return transcribed_text
                else:
                    logger.error(f"Voice transcription failed: {transcribed_text}")
                    return f"Transcription failed: {transcribed_text}"
            except Exception as e:
                logger.error(f"Error in voice transcription: {str(e)}")
                return f"Error: {str(e)}"
        
        def handle_voice_analysis(voice_text, vlm_model, image_display, identify_anomalies, describe_slice):
            """Handle VLM analysis with voice input instead of checkboxes"""
            try:
                if not voice_text or voice_text.strip() == "":
                    return "No voice prompt provided"
                
                # Override checkbox values based on voice input content
                voice_lower = voice_text.lower()
                
                # Analyze voice input to determine intent
                if any(keyword in voice_lower for keyword in ["anomaly", "abnormal", "lesion", "tumor", "pathology", "disease"]):
                    identify_anomalies = True
                if any(keyword in voice_lower for keyword in ["describe", "anatomy", "structure", "region", "what is", "what do you see"]):
                    describe_slice = True
                
                # If no specific intent detected, default to both
                if not identify_anomalies and not describe_slice:
                    identify_anomalies = True
                    describe_slice = True
                
                logger.info(f"Voice analysis with intent - anomalies: {identify_anomalies}, describe: {describe_slice}")
                
                # Call the existing VLM inference handler
                return handle_vlm_inference(vlm_model, image_display, identify_anomalies, describe_slice)
                
            except Exception as e:
                logger.error(f"Error in voice analysis: {str(e)}")
                return f"Error: {str(e)}"
        
        # Voice input event handlers - Direct audio recording
        voice_audio_input.change(
            fn=handle_voice_transcription,
            inputs=[voice_audio_input],
            outputs=[voice_prompt_text]
        )
        
        # Updated VLM run button to use voice input if available
        def enhanced_vlm_inference(vlm_model, image_display, voice_text, identify_anomalies, describe_slice):
            """Enhanced VLM inference that prioritizes voice input"""
            if voice_text and voice_text.strip():
                # Use voice analysis if voice text is available
                return handle_voice_analysis(voice_text, vlm_model, image_display, identify_anomalies, describe_slice)
            else:
                # Fall back to original checkbox-based analysis
                return handle_vlm_inference(vlm_model, image_display, identify_anomalies, describe_slice)
        
        # Update the VLM run button to include voice input
        vlm_run_btn.click(
            fn=enhanced_vlm_inference,
            inputs=[vlm_model_selector, image_display, voice_prompt_text, vlm_prompt_anomalies, vlm_prompt_describe],
            outputs=[vlm_caption]
        )
        
        # Label Management handlers (NEW)
        # Note: Current labels are now updated in the main image_display.change handler above
        
        # Handle VLM label suggestions
        def handle_vlm_label_suggestions(vlm_model, image_annotator_value):
            """Handle VLM label suggestions using the selected model with loading feedback and slice-specific storage"""
            try:
                logger.info(f"VLM label suggestions called with model: {vlm_model}")
                from ui.editor_tab import create_medgemma_label_suggestions, create_other_vlm_label_suggestions, load_labels_from_file
                
                # Get current slice index for slice-specific storage
                current_slice_idx = self.state.current_slice_idx
                
                # Get current labels for this slice to filter duplicates
                current_labels = []
                if self.state.current_directory:
                    saved_labels = load_labels_from_file(self.state.current_directory, current_slice_idx)
                    current_labels = saved_labels if saved_labels else []
                
                if vlm_model == "MedGemma-4B":
                    result_samples = create_medgemma_label_suggestions(image_annotator_value, current_slice_idx, current_labels)
                else:
                    result_samples = create_other_vlm_label_suggestions(vlm_model, image_annotator_value, current_slice_idx, current_labels)
                
                logger.info(f"VLM suggestions result: {len(result_samples)} samples for slice {current_slice_idx}")
                return gr.Dataset(samples=result_samples)
                    
            except Exception as e:
                logger.error(f"Error generating VLM label suggestions: {e}")
                return gr.Dataset(samples=[])
        
        # Show loading state during VLM processing
        def show_loading_state():
            """Show loading state for suggested labels"""
            loading_samples = [["⏳ Generating suggestions..."]]
            return gr.Dataset(samples=loading_samples)
        
        suggest_labels_btn.click(
            fn=show_loading_state,
            inputs=[],
            outputs=[suggested_labels_dataset]
        ).then(
            fn=handle_vlm_label_suggestions,
            inputs=[suggested_vlm_selector, image_display],
            outputs=[suggested_labels_dataset]
        )
        
        # Handle label selection from datasets
        def handle_current_label_selection(evt: gr.SelectData):
            """Handle selection from current labels dataset"""
            try:
                # evt.index gives us the index of the selected item
                # We can use this to get the label name from the dataset
                return f"Selected current label: {evt.value}" if evt.value else "No label selected"
            except Exception as e:
                logger.error(f"Error handling current label selection: {e}")
                return "Error selecting label"
        
        def handle_suggested_label_selection(evt: gr.SelectData):
            """Handle toggle selection from suggested labels dataset"""
            try:
                from ui.editor_tab import toggle_suggested_label_selection, get_selected_suggested_labels_for_slice, get_suggested_labels_for_slice
                
                # Get current slice index and the label that was clicked
                current_slice_idx = self.state.current_slice_idx
                
                # Handle both string and list values from dataset selection
                if isinstance(evt.value, list) and len(evt.value) > 0:
                    label_text = str(evt.value[0]).strip()
                elif isinstance(evt.value, str):
                    label_text = str(evt.value).strip()
                else:
                    label_text = ""
                
                if not label_text or label_text.startswith("⏳"):
                    return gr.update(), gr.update(visible=False)  # Ignore loading or empty labels
                
                # Clean the label text (remove selection indicators if present)
                clean_label_text = label_text.replace("✅ ", "").strip()
                
                # Toggle the selection
                selected_labels = toggle_suggested_label_selection(current_slice_idx, clean_label_text)
                
                # Show/hide accept button based on whether anything is selected
                accept_btn_visible = len(selected_labels) > 0
                
                # Update the suggested labels dataset to show selection state
                all_suggestions = get_suggested_labels_for_slice(current_slice_idx)
                if all_suggestions:
                    # Convert raw samples to label strings
                    all_suggestion_labels = [sample[0] for sample in all_suggestions]
                    
                    # Create new samples with selection indicators
                    updated_samples = []
                    for suggestion_label in all_suggestion_labels:
                        if suggestion_label in selected_labels:
                            updated_samples.append([f"✅ {suggestion_label}"])  # Selected
                        else:
                            updated_samples.append([suggestion_label])  # Not selected
                    
                    return gr.Dataset(samples=updated_samples), gr.update(visible=accept_btn_visible)
                else:
                    return gr.update(), gr.update(visible=accept_btn_visible)
                    
            except Exception as e:
                logger.error(f"Error handling suggested label selection: {e}")
                return gr.update(), gr.update(visible=False)
        
        # Connect dataset selection handlers
        current_labels_dataset.select(
            fn=handle_current_label_selection,
            inputs=[],
            outputs=[annotation_status]
        )
        
        suggested_labels_dataset.select(
            fn=handle_suggested_label_selection,
            inputs=[],
            outputs=[suggested_labels_dataset, accept_suggestions_btn]
        )
        
        # Handle accepting suggested labels
        def handle_accept_suggestions():
            """Accept selected suggested labels and APPEND them to current labels"""
            try:
                from ui.editor_tab import (get_selected_suggested_labels_for_slice, 
                                         clear_selected_suggested_labels_for_slice,
                                         save_labels_to_file, extract_current_labels_from_annotator,
                                         create_labels_dataset_samples, load_labels_from_file,
                                         remove_labels_from_suggested_for_slice, get_suggested_labels_for_slice)
                
                current_slice_idx = self.state.current_slice_idx
                selected_labels = get_selected_suggested_labels_for_slice(current_slice_idx)
                
                if not selected_labels:
                    return gr.update(), gr.update(visible=False), gr.update(), "No labels selected"
                
                # Get current data directory for saving labels
                data_directory = getattr(self.state, 'current_directory', None)
                if not data_directory:
                    return gr.update(), gr.update(visible=False), gr.update(), "Error: No data directory available"
                
                # Clean labels (remove selection indicators)
                clean_labels = []
                for label in selected_labels:
                    clean_label = label.replace("✅ ", "").strip()
                    if clean_label:
                        clean_labels.append(clean_label)
                
                # APPEND to labels.txt file (new save function handles appending)
                save_result = save_labels_to_file(data_directory, current_slice_idx, clean_labels)
                
                # Get current labels from image annotator
                current_image = getattr(self.state, 'image_annotator_state', None)
                annotator_labels = extract_current_labels_from_annotator(current_image)
                
                # Load ALL saved labels from file (includes the ones we just appended)
                saved_labels = load_labels_from_file(data_directory, current_slice_idx)
                
                # Combine annotator labels and saved labels (avoid duplicates)
                all_labels = list(annotator_labels)
                for label in saved_labels:
                    if label not in all_labels:
                        all_labels.append(label)
                
                # Create updated current labels dataset
                current_labels_samples = create_labels_dataset_samples(all_labels)
                updated_current_labels = gr.Dataset(samples=current_labels_samples)
                
                # Remove accepted labels from suggested labels list (so they don't appear again)
                remaining_suggestions = remove_labels_from_suggested_for_slice(current_slice_idx, clean_labels)
                updated_suggested_labels = gr.Dataset(samples=create_labels_dataset_samples(remaining_suggestions))
                
                # Clear selected suggestions and hide accept button
                clear_selected_suggested_labels_for_slice(current_slice_idx)
                
                logger.info(f"APPENDED {len(clean_labels)} suggestions to slice {current_slice_idx}: {clean_labels}")
                logger.info(f"Total labels for slice {current_slice_idx}: {len(all_labels)} - {all_labels}")
                logger.info(f"Remaining suggestions for slice {current_slice_idx}: {len(remaining_suggestions)} - {remaining_suggestions}")
                
                return updated_current_labels, gr.update(visible=False), updated_suggested_labels, save_result
                
            except Exception as e:
                logger.error(f"Error accepting suggestions: {e}")
                return gr.update(), gr.update(visible=False), gr.update(), f"Error: {str(e)}"
        
        accept_suggestions_btn.click(
            fn=handle_accept_suggestions,
            inputs=[],
            outputs=[current_labels_dataset, accept_suggestions_btn, suggested_labels_dataset, annotation_status]
        )
          # Auto-brain annotation handler - Updated to use SAM2 Fast Masking Pipeline (EDITOR-SPECIFIC)
        def handle_auto_brain_annotation(output_dir, save_visualizations, device_selector, processing_mode, score_threshold, dir_input_value):
            """Handle automatic brain structure annotation using SAM2 Fast Masking Pipeline (EDITOR-SPECIFIC)"""
            # Get the current directory from state or use directory input as fallback
            dicom_folder = getattr(self.state, 'current_directory', None)
            
            if not dicom_folder:
                # Fallback to directory input field
                dicom_folder = dir_input_value.strip() if dir_input_value else None
            
            if not dicom_folder:
                return "Error: No DICOM folder specified. Please load data first or enter a directory path.", None, gr.update(visible=False)
            
            if not os.path.exists(dicom_folder):
                return f"Error: DICOM folder not found: {dicom_folder}", None, gr.update(visible=False)
            
            # Check if we have loaded data for this directory
            if not hasattr(self.state, 'current_data') or self.state.current_data is None:
                return "Error: Please load DICOM data first using the 'Load Data' button before running automatic annotation.", None, gr.update(visible=False)            
            # Run the SAM2 Fast Masking Pipeline (replaces old automatic annotation)
            status, annotated_result, success = self.editor_medsam2_handlers.run_sam2_fast_masking(
                dicom_folder, output_dir, save_visualizations, processing_mode
            )
              # If processing mode is "All Records" and annotation was successful, refresh 3D viewer
            # Always return a valid Plotly figure to avoid the __module__ attribute error
            from ui.editor_tab import create_empty_3d_plot, create_3d_visualization
            if processing_mode == "All Records" and success:
                try:
                    viewer_3d_update = create_3d_visualization(output_dir, score_threshold)
                except Exception as e:
                    viewer_3d_update = create_empty_3d_plot(f"Error updating 3D view: {str(e)}")
            else:
                # For non-"All Records" mode, show message
                viewer_3d_update = create_empty_3d_plot("3D Viewer available in 'All Records' mode")
              # Return the annotated result directly to the image_annotator
            return status, annotated_result, viewer_3d_update
        auto_brain_annotate_btn.click(
            fn=handle_auto_brain_annotation,
            inputs=[output_dir, save_visualizations, device_selector, processing_mode, score_threshold, dir_input],
            outputs=[annotation_status, image_display, viewer_3d]
        )
          # Handle image removal from image_annotator (X button / clear button) (EDITOR-SPECIFIC)
    
        image_display.clear(
            fn=self.editor_image_handlers.handle_image_remove,
            inputs=[],
            outputs=[image_display]
        )
        
        def update_box_instructions(prompt_type):
            """Update instructions based on prompt type selection"""
            if prompt_type == "Box-based Prompt":
                return gr.update(
                    value="📦 **BOX PROMPT MODE**: \n"
                    "1. Draw a rectangle around the region you want to segment\n"
                    "2. You may see a label dialog - you can ignore it or use any label\n" 
                    "3. Click 'Run MEDSAM2 Annotation (Manual Prompts)' to segment the boxed region\n"
                    "4. MEDSAM2 will create precise masks based on your box prompt",
                    visible=True
                )
            elif prompt_type == "Point-based Prompt":                return gr.update(
                    value="📍 **POINT PROMPT MODE**: \n"
                    "1. Click points on the image to guide segmentation\n"
                    "2. Use multiple points for better accuracy\n"
                    "3. Click 'Run MEDSAM2 Annotation (Manual Prompts)' to segment",
                    visible=True
                )
            else:  # SAM2 Fast Masking Mode
                return gr.update(
                    value="🚀 **SAM2 FAST MASKING MODE**: Click 'Run SAM2 Fast Masking' for automatic brain structure segmentation",
                    visible=True
                )        # Add instruction text component
        '''instruction_text = gr.Markdown(
            value="🚀 **SAM2 FAST MASKING MODE**: Click 'Run SAM2 Fast Masking' for automatic brain structure segmentation",
            visible=True        )
'''
    
    def _connect_contribute_handlers(self, contribute_components, editor_components, tabs, active_tab_state, change_tab):
        """Connect event handlers for the contribute tab"""
        if not contribute_components or not ENABLE_AUTH:
            return
        
        # Update the user state in contribute tab when user changes
        def update_contribute_user_state():
            if self.current_user:
                return self.current_user['user_id']
            return ""
        
        # Connect dataset loading from contribute tab to editor tab
        def load_dataset_from_contribute(dataset_path, crowdsourcing_mode):
            """Load dataset from contribute tab into editor tab"""
            logger.info(f"DEBUG: load_dataset_from_contribute called with dataset_path: {dataset_path}, crowdsourcing_mode: {crowdsourcing_mode}")
            
            if not dataset_path or not crowdsourcing_mode:
                logger.info(f"DEBUG: load_dataset_from_contribute early return - no dataset selected")
                return "No dataset selected", gr.update(), None, None, None, None, None, None, None
            
            try:
                # Use the existing data loading handler that returns annotated format
                # Use load_data_for_annotator instead of load_data to get the proper format for image_annotator
                logger.info(f"DEBUG: Calling self.data_handlers.load_data_for_annotator with dataset_path: {dataset_path}")
                result = self.data_handlers.load_data_for_annotator(None, dataset_path)
                logger.info(f"DEBUG: load_data_for_annotator returned type: {type(result)}, length: {len(result) if result else 'None'}")
                
                # result is a tuple: (annotated_value, dropdown, metadata, slider, slice_info, crosshair_text, status_message, window_level, window_width)
                if result and len(result) >= 9:
                    annotated_value, dropdown, metadata, slider, slice_info, crosshair_text, status_message, window_level, window_width = result
                    logger.info(f"DEBUG: Extracted annotated_value type: {type(annotated_value)}")
                    if isinstance(annotated_value, dict):
                        logger.info(f"DEBUG: annotated_value keys: {list(annotated_value.keys())}")
                    else:
                        logger.info(f"DEBUG: annotated_value is not a dict! Value: {annotated_value}")
                    
                    if isinstance(status_message, str) and "loaded" in status_message.lower():
                        self.crowdsourcing_mode = True
                        
                        # annotated_value is already in the correct format from load_data_for_annotator
                        # It's a dict with {"image": img_rgb, "boxes": [], "orientation": 0}
                        logger.info(f"DEBUG: load_dataset_from_contribute returning success with annotated_value type: {type(annotated_value)}")
                        
                        # Return complete initialization data for editor tab
                        return (
                            "Dataset loaded successfully!",  # Simple success message for contribute tab
                            gr.update(visible=True),          # Show crowdsourcing controls
                            annotated_value,                   # Initialize image display (already properly formatted)
                            metadata,                          # Initialize metadata
                            slider,                            # Initialize slider
                            slice_info,                        # Initialize slice info
                            crosshair_text,                    # Initialize crosshair text
                            window_level,                      # Initialize window level
                            window_width                       # Initialize window width
                        )
                    else:
                        logger.info(f"DEBUG: load_dataset_from_contribute status check failed: {status_message}")
                        return (
                            status_message if isinstance(status_message, str) else "Failed to load dataset",
                            gr.update(visible=False),
                            None, None, None, None, None, None, None
                        )
                else:
                    logger.info(f"DEBUG: load_dataset_from_contribute invalid result length or None")
                    return (
                        "Failed to load dataset",
                        gr.update(visible=False),
                        None, None, None, None, None, None, None
                    )
                    
            except Exception as e:
                logger.error(f"Error loading dataset from contribute tab: {e}")
                return (
                    f"Error loading dataset: {str(e)}",
                    gr.update(visible=False),
                    None, None, None, None, None, None, None
                )
        
        # Wrapper function to handle submission and clearing overlays
        def handle_submit_and_clear(task_selection, user_id, image_annotator_data):
            """Handle annotation submission and clear overlays after successful submission"""
            # First handle the submission
            status_message = handle_submit_annotation(task_selection, user_id, image_annotator_data)
            
            # Check if submission was successful
            if status_message.startswith("✅"):
                # Create Next.js styled success message
                styled_message = f"""
                <div style='padding: 16px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border: 1px solid #10b981; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <div style='display: flex; align-items: center; gap: 12px;'>
                        <div style='width: 12px; height: 12px; background: #10b981; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3);'></div>
                        <div style='color: #047857; font-weight: 500; font-size: 14px; line-height: 1.5;'>
                            {status_message.replace('✅', '')}
                        </div>
                    </div>
                </div>
                """
                
                # Successful submission - clear overlays
                if isinstance(image_annotator_data, dict) and 'image' in image_annotator_data:
                    # Create cleared image_annotator data (keep image, clear overlays)
                    cleared_data = {
                        'image': image_annotator_data['image'],
                        'boxes': [],  # Clear all overlays
                        'orientation': image_annotator_data.get('orientation', 0)
                    }
                    return styled_message, cleared_data
                else:
                    # If no image data, just return update
                    return styled_message, gr.update()
            else:
                # Create Next.js styled error message
                styled_message = f"""
                <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 12px; border: 1px solid #ef4444; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <div style='display: flex; align-items: center; gap: 12px;'>
                        <div style='width: 12px; height: 12px; background: #ef4444; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.3);'></div>
                        <div style='color: #dc2626; font-weight: 500; font-size: 14px; line-height: 1.5;'>
                            {status_message.replace('❌', '')}
                        </div>
                    </div>
                </div>
                """
                # Failed submission - don't clear overlays
                return styled_message, gr.update()
        
        # Handle annotation submission
        def handle_submit_annotation(task_selection, user_id, image_annotator_data):
            """Handle annotation submission from editor tab with annotation data"""
            if not task_selection or not user_id:
                return "Please select a task and ensure you're logged in"
            
            try:
                campaign_id, patient_id, _ = [x.strip() for x in task_selection.split('|')]
                from crowdsourcing.campaign_manager import CrowdsourcingManager
                crowdsourcing_manager = CrowdsourcingManager()
                
                # Prepare annotation data for saving
                annotation_data = None
                if image_annotator_data:
                    try:
                        # Extract annotation information from image annotator
                        annotations = []
                        labels = []
                        
                        logger.info(f"Processing image_annotator_data type: {type(image_annotator_data)}")
                        
                        # Get current slice information from the state first
                        current_slice = 0  # Default slice
                        if hasattr(self.state, 'current_slice_idx'):
                            current_slice = self.state.current_slice_idx
                        
                        # First, ensure current slice annotations are saved before processing
                        if hasattr(self, 'editor_image_handlers') and image_annotator_data:
                            logger.info(f"Saving current slice annotations before submission for slice {current_slice}")
                            self.editor_image_handlers.save_user_annotations(image_annotator_data, current_slice)
                        
                        # Collect annotations from all slices, not just current
                        all_slice_annotations = []
                        all_slice_labels = []
                        
                        # Check if we have saved annotations for multiple slices
                        # Look in the image handlers' user_annotations storage where slice navigation saves them
                        if hasattr(self, 'editor_image_handlers') and hasattr(self.editor_image_handlers, 'user_annotations') and self.editor_image_handlers.user_annotations:
                            logger.info(f"Found user_annotations storage with {len(self.editor_image_handlers.user_annotations)} slices")
                            logger.info(f"Available slice annotations: {list(self.editor_image_handlers.user_annotations.keys())}")
                            # Process all slices with user annotations
                            for slice_idx, user_annotations in self.editor_image_handlers.user_annotations.items():
                                if isinstance(user_annotations, list) and user_annotations:
                                    logger.info(f"Processing slice {slice_idx} with {len(user_annotations)} user annotations")
                                    # Get current view type (default to axial)
                                    view_type = getattr(self.state, 'current_view', 'axial')
                                    
                                    # Process annotations for this slice
                                    for i, user_annotation in enumerate(user_annotations):
                                        # Extract the actual annotation data
                                        if isinstance(user_annotation, dict) and 'data' in user_annotation:
                                            annotation = user_annotation['data']
                                        else:
                                            annotation = user_annotation
                                        
                                        if isinstance(annotation, dict):
                                            # Add slice information to annotation
                                            ann_info = {
                                                'annotation_id': f"slice_{slice_idx}_ann_{i}",
                                                'slice_index': slice_idx,
                                                'view_type': view_type,
                                                'type': annotation.get('type', 'box'),
                                                'label': annotation.get('label', ''),
                                                'coordinates': [],
                                                'bbox': [],
                                                'area': 0,
                                                'points': [],
                                                'color': annotation.get('color', None)  # Extract color from annotation
                                            }
                                            
                                            # Extract coordinates based on shape type
                                            shape_type = ann_info['type']
                                            if shape_type == 'polygon' and 'points' in annotation:
                                                # Polygon shapes store coordinates as points
                                                points = annotation['points']
                                                if isinstance(points, list):
                                                    coordinates = []
                                                    for point in points:
                                                        if isinstance(point, dict) and 'x' in point and 'y' in point:
                                                            coordinates.append([point['x'], point['y']])
                                                    ann_info['coordinates'] = coordinates
                                                    ann_info['points'] = coordinates
                                                    
                                                    # Calculate bbox for polygon
                                                    if coordinates:
                                                        x_coords = [p[0] for p in coordinates]
                                                        y_coords = [p[1] for p in coordinates]
                                                        ann_info['bbox'] = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                                        
                                                        # Calculate approximate area using shoelace formula
                                                        area = 0
                                                        n = len(coordinates)
                                                        for j in range(n):
                                                            k = (j + 1) % n
                                                            area += coordinates[j][0] * coordinates[k][1]
                                                            area -= coordinates[k][0] * coordinates[j][1]
                                                        ann_info['area'] = abs(area) / 2
                                            
                                            elif shape_type == 'freehand' and 'points' in annotation:
                                                # Freehand shapes also store coordinates as points
                                                points = annotation['points']
                                                if isinstance(points, list):
                                                    coordinates = []
                                                    for point in points:
                                                        if isinstance(point, dict) and 'x' in point and 'y' in point:
                                                            coordinates.append([point['x'], point['y']])
                                                    ann_info['coordinates'] = coordinates
                                                    ann_info['points'] = coordinates
                                                    
                                                    # Calculate bbox for freehand
                                                    if coordinates:
                                                        x_coords = [p[0] for p in coordinates]
                                                        y_coords = [p[1] for p in coordinates]
                                                        ann_info['bbox'] = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                                        ann_info['area'] = len(coordinates)  # Approximation for freehand
                                            
                                            elif 'xmin' in annotation and 'ymin' in annotation and 'xmax' in annotation and 'ymax' in annotation:
                                                # Box/rectangle shapes
                                                xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
                                                ann_info['bbox'] = [xmin, ymin, xmax, ymax]
                                                ann_info['coordinates'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]  # Rectangle corners
                                                ann_info['area'] = (xmax - xmin) * (ymax - ymin)
                                            
                                            elif 'x' in annotation and 'y' in annotation and 'width' in annotation and 'height' in annotation:
                                                # Alternative box format
                                                x, y, width, height = annotation['x'], annotation['y'], annotation['width'], annotation['height']
                                                ann_info['bbox'] = [x, y, x + width, y + height]
                                                ann_info['coordinates'] = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
                                                ann_info['area'] = width * height
                                            
                                            all_slice_annotations.append(ann_info)
                                            
                                            # Collect labels with slice information and RGB colors
                                            if ann_info['label']:
                                                label_with_slice = {
                                                    'label': ann_info['label'],
                                                    'slice_index': slice_idx,
                                                    'view_type': view_type,
                                                    'color': ann_info.get('color', None)  # Include color from annotation
                                                }
                                                if label_with_slice not in all_slice_labels:
                                                    all_slice_labels.append(label_with_slice)
                        else:
                            logger.info("No stored user_annotations found - checking for current image data only")
                        
                        # Fallback: if no multi-slice data, process current image_annotator_data
                        if not all_slice_annotations and isinstance(image_annotator_data, dict):
                            logger.info(f"Processing current image_annotator_data as fallback")
                            logger.info(f"image_annotator_data keys: {list(image_annotator_data.keys())}")
                            # Get annotations from image annotator - try multiple possible keys
                            raw_annotations = (image_annotator_data.get('boxes', []) or 
                                             image_annotator_data.get('annotations', []) or 
                                             image_annotator_data.get('shapes', []))
                            
                            logger.info(f"Found {len(raw_annotations)} raw annotations to process")
                            
                            for i, annotation in enumerate(raw_annotations):
                                logger.info(f"Processing annotation {i}: {type(annotation)} - {annotation}")
                                
                                if isinstance(annotation, dict):
                                    # Extract shape type
                                    shape_type = annotation.get('type', 'box')
                                    label = annotation.get('label', '')
                                    
                                    # Initialize annotation info with slice information
                                    ann_info = {
                                        'annotation_id': f"slice_{current_slice}_ann_{i}",
                                        'slice_index': current_slice,
                                        'view_type': 'axial',  # Default view
                                        'type': shape_type,
                                        'label': label,
                                        'coordinates': [],
                                        'bbox': [],
                                        'area': 0,
                                        'points': [],
                                        'color': annotation.get('color', None)  # Extract color from annotation
                                    }
                                    
                                    # Extract coordinates based on shape type (same logic as before)
                                    if shape_type == 'polygon' and 'points' in annotation:
                                        # Polygon shapes store coordinates as points
                                        points = annotation['points']
                                        if isinstance(points, list):
                                            coordinates = []
                                            for point in points:
                                                if isinstance(point, dict) and 'x' in point and 'y' in point:
                                                    coordinates.append([point['x'], point['y']])
                                            ann_info['coordinates'] = coordinates
                                            ann_info['points'] = coordinates
                                            
                                            # Calculate bbox for polygon
                                            if coordinates:
                                                x_coords = [p[0] for p in coordinates]
                                                y_coords = [p[1] for p in coordinates]
                                                ann_info['bbox'] = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                                
                                                # Calculate approximate area using shoelace formula
                                                area = 0
                                                n = len(coordinates)
                                                for j in range(n):
                                                    k = (j + 1) % n
                                                    area += coordinates[j][0] * coordinates[k][1]
                                                    area -= coordinates[k][0] * coordinates[j][1]
                                                ann_info['area'] = abs(area) / 2
                                    
                                    elif shape_type == 'freehand' and 'points' in annotation:
                                        # Freehand shapes also store coordinates as points
                                        points = annotation['points']
                                        if isinstance(points, list):
                                            coordinates = []
                                            for point in points:
                                                if isinstance(point, dict) and 'x' in point and 'y' in point:
                                                    coordinates.append([point['x'], point['y']])
                                            ann_info['coordinates'] = coordinates
                                            ann_info['points'] = coordinates
                                            
                                            # Calculate bbox for freehand
                                            if coordinates:
                                                x_coords = [p[0] for p in coordinates]
                                                y_coords = [p[1] for p in coordinates]
                                                ann_info['bbox'] = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                                ann_info['area'] = len(coordinates)  # Approximation for freehand
                                    
                                    elif 'xmin' in annotation and 'ymin' in annotation and 'xmax' in annotation and 'ymax' in annotation:
                                        # Box/rectangle shapes
                                        xmin, ymin, xmax, ymax = annotation['xmin'], annotation['ymin'], annotation['xmax'], annotation['ymax']
                                        ann_info['bbox'] = [xmin, ymin, xmax, ymax]
                                        ann_info['coordinates'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]  # Rectangle corners
                                        ann_info['area'] = (xmax - xmin) * (ymax - ymin)
                                    
                                    elif 'x' in annotation and 'y' in annotation and 'width' in annotation and 'height' in annotation:
                                        # Alternative box format
                                        x, y, width, height = annotation['x'], annotation['y'], annotation['width'], annotation['height']
                                        ann_info['bbox'] = [x, y, x + width, y + height]
                                        ann_info['coordinates'] = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
                                        ann_info['area'] = width * height
                                    
                                    else:
                                        # Fallback for unknown formats
                                        logger.warning(f"Unknown annotation format for annotation {i}: {annotation}")
                                        # Try to extract any coordinate data
                                        if 'coordinates' in annotation:
                                            ann_info['coordinates'] = annotation['coordinates']
                                        if 'bbox' in annotation:
                                            ann_info['bbox'] = annotation['bbox']
                                        if 'area' in annotation:
                                            ann_info['area'] = annotation['area']
                                    
                                    all_slice_annotations.append(ann_info)
                                    
                                    # Collect labels with slice info and RGB colors
                                    if ann_info['label']:
                                        label_with_slice = {
                                            'label': ann_info['label'],
                                            'slice_index': current_slice,
                                            'view_type': 'axial',
                                            'color': ann_info.get('color', None)  # Include color from annotation
                                        }
                                        if label_with_slice not in all_slice_labels:
                                            all_slice_labels.append(label_with_slice)
                        
                        # Use all collected annotations
                        annotations = all_slice_annotations
                        labels = all_slice_labels
                        
                        logger.info(f"Final annotation collection: {len(annotations)} annotations from {len(set(ann.get('slice_index', 0) for ann in annotations))} unique slices")
                        
                        annotation_data = {
                            'annotations': annotations,
                            'labels': labels,
                            'image': image_annotator_data.get('image') if isinstance(image_annotator_data, dict) else None,
                            'total_annotations': len(annotations),
                            'submission_time': None  # Will be set by campaign manager
                        }
                        
                        logger.info(f"Submitting annotation with {len(annotations)} annotations across multiple slices and {len(labels)} unique labels")
                        
                    except Exception as e:
                        logger.warning(f"Error processing annotation data: {e}")
                        annotation_data = {
                            'annotations': [],
                            'labels': [],
                            'total_annotations': 0,
                            'error': f"Error processing annotations: {str(e)}"
                        }
                
                # Submit with annotation data
                success = crowdsourcing_manager.mark_completed(campaign_id, user_id, patient_id, annotation_data)
                
                if success:
                    annotation_count = len(annotation_data.get('annotations', [])) if annotation_data else 0
                    
                    # Get remaining assignments for this user
                    remaining_tasks = crowdsourcing_manager.get_remaining_assignments_for_user(user_id)
                    remaining_count = len(remaining_tasks)
                    
                    success_msg = f"✅ Annotation for patient {patient_id} submitted successfully! ({annotation_count} annotations saved)"
                    
                    if remaining_count > 0:
                        success_msg += f"\n\n🎯 You have {remaining_count} assignment{'s' if remaining_count != 1 else ''} remaining. \n\nClick 'Load Next Assignment' to continue."
                    else:
                        success_msg += f"\n\n🎉 Congratulations! You have completed all your assignments."
                    
                    return success_msg
                else:
                    return "❌ Failed to submit annotation"
                    
            except Exception as e:
                logger.error(f"Error submitting annotation: {e}")
                return f"❌ Error submitting annotation: {str(e)}"
        
        # Handle review request
        def handle_send_for_review():
            """Handle send for review request"""
            return "📤 Review functionality will be implemented in future versions"
        
        # Get remaining assignments count for current user
        def get_remaining_assignments_info(user_id, hide_submit_btn=False):
            """Get information about remaining assignments for the current user with Next.js styling"""
            if not user_id:
                return (
                    gr.update(value="", visible=False),  # assignments_remaining
                    gr.update(visible=False),            # next_assignment_btn
                    gr.update(visible=False)             # submit_btn
                )
            
            try:
                from crowdsourcing.campaign_manager import CrowdsourcingManager
                crowdsourcing_manager = CrowdsourcingManager()
                
                # Get remaining assignments for this user
                remaining_tasks = crowdsourcing_manager.get_remaining_assignments_for_user(user_id)
                remaining_count = len(remaining_tasks)
                
                if remaining_count > 0:
                    # Next.js style progress display
                    progress_html = f"""
                    <div style='padding: 16px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 12px; border: 1px solid #3b82f6; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                        <div style='display: flex; align-items: center; gap: 12px;'>
                            <div style='width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);'></div>
                            <div style='color: #1e40af; font-weight: 600; font-size: 14px;'>
                                📊 {remaining_count} assignment{'s' if remaining_count != 1 else ''} remaining
                            </div>
                        </div>
                    </div>
                    """
                    return (
                        gr.update(value=progress_html, visible=True),  # assignments_remaining
                        gr.update(visible=True),                       # next_assignment_btn
                        gr.update(visible=not hide_submit_btn)         # submit_btn - Hide if requested
                    )
                else:
                    # Next.js style completion display
                    completion_html = f"""
                    <div style='padding: 16px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border: 1px solid #10b981; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                        <div style='display: flex; align-items: center; gap: 12px;'>
                            <div style='width: 12px; height: 12px; background: #10b981; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3);'></div>
                            <div style='color: #047857; font-weight: 600; font-size: 14px;'>
                                🎉 All assignments completed!
                            </div>
                        </div>
                    </div>
                    """
                    return (
                        gr.update(value=completion_html, visible=True),  # assignments_remaining
                        gr.update(visible=False),                        # next_assignment_btn
                        gr.update(visible=False)                         # submit_btn - Hide when no more tasks
                    )
                    
            except Exception as e:
                logger.error(f"Error getting remaining assignments: {e}")
                return (
                    gr.update(value="", visible=False),  # assignments_remaining
                    gr.update(visible=False),            # next_assignment_btn
                    gr.update(visible=True)              # submit_btn
                )
        
        # Load next assignment for current user
        def load_next_assignment(user_id):
            """Load the next available assignment for the current user - simplified direct loading"""
            import os  # Import os for path operations
            
            if not user_id:
                return [
                    gr.update(),  # tabs (no change)
                    gr.update(),  # dicom_viewer
                    gr.update(),  # image_annotator
                    gr.update(),  # slice_slider
                    gr.update(),  # prev_slice_btn
                    gr.update(),  # next_slice_btn
                    gr.update(),  # metadata_display
                    gr.update(),  # submit_btn
                    gr.update(value="❌ Please log in first", visible=True),  # submission_status
                    gr.update(),  # welcome_guide
                    gr.update(),  # welcome_guide_content
                    gr.update(),  # assignments_remaining
                    gr.update(),  # next_assignment_btn
                    gr.update(),  # selected_task_info
                ]
            
            try:
                from crowdsourcing.campaign_manager import CrowdsourcingManager
                crowdsourcing_manager = CrowdsourcingManager()
                
                # Get remaining assignments for this user
                remaining_tasks = crowdsourcing_manager.get_remaining_assignments_for_user(user_id)
                
                if not remaining_tasks:
                    return [
                        gr.update(),  # tabs (no change)
                        gr.update(),  # dicom_viewer
                        gr.update(),  # image_annotator
                        gr.update(),  # slice_slider
                        gr.update(),  # prev_slice_btn
                        gr.update(),  # next_slice_btn
                        gr.update(),  # metadata_display
                        gr.update(),  # submit_btn
                        gr.update(value="🎉 No more assignments available. You've completed all your tasks!", visible=True),  # submission_status
                        gr.update(),  # welcome_guide
                        gr.update(),  # welcome_guide_content
                        gr.update(value="🎉 All assignments completed!", visible=True),  # assignments_remaining
                        gr.update(visible=False),  # next_assignment_btn
                        gr.update(),  # selected_task_info
                    ]
                
                # Get the next specific task assigned to this user
                next_task = remaining_tasks[0]
                
                # Use the task data directly
                campaign_id = next_task['campaign_id']
                patient_id = next_task['patient_id'].strip()  # Strip any whitespace
                dataset_path = next_task['dataset_path']
                
                logger.info(f"Loading next assignment for user {user_id}: campaign='{campaign_id}', patient='{patient_id}'")
                
                # Build patient path and find modality (same as Load Task in Editor)
                patient_path = os.path.join(dataset_path, patient_id)
                
                # Verify that the patient path exists
                if not os.path.exists(patient_path):
                    logger.error(f"Patient path does not exist: {patient_path}")
                    return [
                        gr.update(),  # tabs (no change)
                        gr.update(),  # dicom_viewer
                        gr.update(),  # image_annotator
                        gr.update(),  # slice_slider
                        gr.update(),  # prev_slice_btn
                        gr.update(),  # next_slice_btn
                        gr.update(),  # metadata_display
                        gr.update(),  # submit_btn
                        gr.update(value=f"❌ Error: Patient directory not found: {patient_path}", visible=True),  # submission_status
                        gr.update(),  # welcome_guide
                        gr.update(),  # welcome_guide_content
                        gr.update(),  # assignments_remaining
                        gr.update(),  # next_assignment_btn
                        gr.update(),  # selected_task_info
                    ]
                
                # Find the first available modality directory
                valid_modalities = ['flair', 't1', 't1c', 't2']
                modality_path = None
                
                for modality in valid_modalities:
                    potential_path = os.path.join(patient_path, modality)
                    if os.path.exists(potential_path) and os.path.isdir(potential_path):
                        modality_path = potential_path
                        break
                
                if not modality_path:
                    logger.error(f"No valid modality found for patient {patient_id}")
                    return [
                        gr.update(),  # tabs (no change)
                        gr.update(),  # dicom_viewer
                        gr.update(),  # image_annotator
                        gr.update(),  # slice_slider
                        gr.update(),  # prev_slice_btn
                        gr.update(),  # next_slice_btn
                        gr.update(),  # metadata_display
                        gr.update(),  # submit_btn
                        gr.update(value=f"❌ Error: No valid modality found for patient {patient_id}", visible=True),  # submission_status
                        gr.update(),  # welcome_guide
                        gr.update(),  # welcome_guide_content
                        gr.update(),  # assignments_remaining
                        gr.update(),  # next_assignment_btn
                        gr.update(),  # selected_task_info
                    ]
                
                # Update current directory in state
                self.state.current_directory = modality_path
                
                # Create task_selection for the welcome guide - ensure format matches dropdown choices
                # The dropdown choices use the exact format from the database, so we need to match that
                task_selection = f"{campaign_id}|{patient_id}|{dataset_path}"
                
                # Direct data loading - replicate the exact logic from "Load Task in Editor"
                try:
                    logger.info(f"DEBUG: Direct loading DICOM data from: {modality_path}")
                    
                    # Call load_data_for_annotator directly and get the properly formatted result
                    result = self.data_handlers.load_data_for_annotator(None, modality_path)
                    
                    if not result or len(result) < 9:
                        logger.error(f"Failed to load assignment: Invalid result")
                        return [
                            gr.update(),  # tabs (no change)
                            gr.update(),  # dicom_viewer
                            gr.update(),  # image_annotator
                            gr.update(),  # slice_slider
                            gr.update(),  # prev_slice_btn
                            gr.update(),  # next_slice_btn
                            gr.update(),  # metadata_display
                            gr.update(),  # submit_btn
                            gr.update(value=f"❌ Error loading assignment data", visible=True),  # submission_status
                            gr.update(),  # welcome_guide
                            gr.update(),  # welcome_guide_content
                            gr.update(),  # assignments_remaining
                            gr.update(),  # next_assignment_btn
                            gr.update(),  # selected_task_info
                        ]
                    
                    # Extract the data (annotated_value is already in correct format)
                    annotated_value, dropdown, metadata, slider, slice_info, crosshair_text, status_message, window_level, window_width = result
                    
                    logger.info(f"DEBUG: Loaded annotated_value type: {type(annotated_value)}")
                    if isinstance(annotated_value, dict):
                        logger.info(f"DEBUG: annotated_value keys: {list(annotated_value.keys())}")
                    
                    # Enable crowdsourcing mode
                    self.crowdsourcing_mode = True
                    
                    # Clear any previous annotation state to prevent old data persistence
                    if hasattr(self.state, 'image_annotator_state'):
                        self.state.image_annotator_state = None
                    if hasattr(self, 'current_annotations'):
                        self.current_annotations = []
                    
                    # Get updated assignment info
                    remaining_after = len(remaining_tasks) - 1
                    
                    # Create Next.js styled assignment progress
                    if remaining_after > 0:
                        progress_html = f"""
                        <div style='padding: 16px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 12px; border: 1px solid #3b82f6; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                            <div style='display: flex; align-items: center; gap: 12px;'>
                                <div style='width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);'></div>
                                <div style='color: #1e40af; font-weight: 600; font-size: 14px;'>
                                    📊 {remaining_after} assignment{'s' if remaining_after != 1 else ''} remaining
                                </div>
                            </div>
                        </div>
                        """
                    else:
                        progress_html = f"""
                        <div style='padding: 16px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 12px; border: 1px solid #f59e0b; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                            <div style='display: flex; align-items: center; gap: 12px;'>
                                <div style='width: 12px; height: 12px; background: #f59e0b; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.3);'></div>
                                <div style='color: #92400e; font-weight: 600; font-size: 14px;'>
                                    🎉 This is your last assignment!
                                </div>
                            </div>
                        </div>
                        """
                    
                    # Create Next.js styled submission status (ready to work)
                    ready_status_html = f"""
                    <div style='padding: 16px; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 12px; border: 1px solid #6b7280; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                        <div style='display: flex; align-items: center; gap: 12px;'>
                            <div style='width: 12px; height: 12px; background: #6b7280; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(107, 114, 128, 0.3);'></div>
                            <div style='color: #374151; font-weight: 500; font-size: 14px;'>
                                ✅ Assignment loaded successfully! Complete your annotation work above, then submit.
                            </div>
                        </div>
                    </div>
                    """
                    
                    # Return updates - minimal changes, no tab switching, SHOW PROGRESS AND STATUS
                    return [
                        gr.update(),  # tabs (stay where we are)
                        gr.update(visible=False),  # hide dicom_viewer 
                        annotated_value,  # image_annotator - direct value, no gr.update()
                        slider,  # slice_slider - use loaded slider
                        gr.update(visible=True),  # prev_slice_btn
                        gr.update(visible=True),  # next_slice_btn
                        metadata,  # metadata_display - use loaded metadata
                        gr.update(visible=True),  # submit_btn
                        gr.update(value=ready_status_html, visible=True),  # submission_status - SHOW with ready message
                        gr.update(visible=True, open=True),  # welcome_guide
                        self._populate_welcome_guide_info(task_selection),  # welcome_guide_content
                        gr.update(value=progress_html, visible=True),  # assignments_remaining - SHOW with progress
                        gr.update(visible=remaining_after > 0),  # next_assignment_btn
                        gr.update(value=task_selection),  # selected_task_info - UPDATE with new task selection
                    ]
                        
                except Exception as load_error:
                    logger.error(f"Error loading DICOM data: {load_error}")
                    return [
                        gr.update(),  # tabs (no change)
                        gr.update(),  # dicom_viewer
                        gr.update(),  # image_annotator
                        gr.update(),  # slice_slider
                        gr.update(),  # prev_slice_btn
                        gr.update(),  # next_slice_btn
                        gr.update(),  # metadata_display
                        gr.update(),  # submit_btn
                        gr.update(value=f"❌ Error loading DICOM data: {str(load_error)}", visible=True),  # submission_status
                        gr.update(),  # welcome_guide
                        gr.update(),  # welcome_guide_content
                        gr.update(),  # assignments_remaining
                        gr.update(),  # next_assignment_btn
                        gr.update(),  # selected_task_info
                    ]
                    
            except Exception as e:
                logger.error(f"Error loading next assignment: {e}")
                return [
                    gr.update(),  # tabs (no change)
                    gr.update(),  # dicom_viewer
                    gr.update(),  # image_annotator
                    gr.update(),  # slice_slider
                    gr.update(),  # prev_slice_btn
                    gr.update(),  # next_slice_btn
                    gr.update(),  # metadata_display
                    gr.update(),  # submit_btn
                    gr.update(value=f"❌ Error loading next assignment: {str(e)}", visible=True),  # submission_status
                    gr.update(),  # welcome_guide
                    gr.update(),  # welcome_guide_content
                    gr.update(),  # assignments_remaining
                    gr.update(),  # next_assignment_btn
                    gr.update(),  # selected_task_info
                ]
        
        # Connect the dataset loading with auto-switch to Editor tab
        if ('selected_dataset_path' in contribute_components and 
            'crowdsourcing_mode' in contribute_components and
            'crowdsourcing' in editor_components):
            
            contribute_components['selected_dataset_path'].change(
                fn=load_dataset_from_contribute,
                inputs=[contribute_components['selected_dataset_path'], contribute_components['crowdsourcing_mode']],
                outputs=[
                    contribute_components['load_status'],           # Status message
                    editor_components['crowdsourcing']['accordion'], # Show crowdsourcing controls
                    editor_components['visualization'][3],          # image_display
                    editor_components['visualization'][1],          # metadata_display  
                    editor_components['visualization'][7],          # slice_slider
                    editor_components['visualization'][9],          # slice_text
                    editor_components['visualization'][10],         # crosshair_info
                    # Window level and width are in data_loading section
                    editor_components['data_loading'][8],           # window_level
                    editor_components['data_loading'][9]            # window_width
                ]
            ).then(
                # Auto-switch to Editor tab after successful loading
                fn=lambda: change_tab(1),  # Editor tab has id=1
                outputs=[tabs]
            ).then(
                # Show welcome guide when switching to Editor tab after dataset loading
                fn=self._populate_welcome_guide_info,
                inputs=[contribute_components['selected_task_info']],
                outputs=[editor_components['welcome_modal']['content']]
            ).then(
                # Show the accordion after content is populated
                fn=lambda: gr.update(visible=True, open=True),
                outputs=[editor_components['welcome_modal']['guide']]
            )
            
            # Connect crowdsourcing controls in editor tab
            editor_components['crowdsourcing']['submit_btn'].click(
                fn=handle_submit_and_clear,
                inputs=[contribute_components['selected_task_info'], contribute_components['current_user_state'], editor_components['visualization'][3]],  # image_display is at index 3 in visualization
                outputs=[editor_components['crowdsourcing']['status'], editor_components['visualization'][3]]  # Also update image_annotator to clear overlays
            ).then(
                # Update assignment progress after submission and control button visibility - hide submit button
                fn=lambda user_id: get_remaining_assignments_info(user_id, hide_submit_btn=True),
                inputs=[contribute_components['current_user_state']],
                outputs=[
                    editor_components['crowdsourcing']['assignments_remaining'], 
                    editor_components['crowdsourcing']['next_assignment_btn'],
                    editor_components['crowdsourcing']['submit_btn']
                ]
            ).then(
                # Refresh Contribute tab table after submission
                fn=contribute_components['refresh_assignments_after_submission'],
                inputs=[contribute_components['current_user_state']],
                outputs=[
                    contribute_components['assignments_table'],
                    contribute_components['task_selection_radio'],
                    contribute_components['load_status']
                ]
            )
            
            # Connect next assignment button - use the new load_next_assignment from contribute tab
            def handle_next_assignment_wrapper(user_id):
                """Wrapper to handle next assignment loading and update both editor and contribute tabs"""
                # Load the next assignment using contribute tab's function
                load_status, dataset_path, success, task_info = contribute_components['load_next_assignment'](user_id)
                
                if success and dataset_path:
                    # Load the dataset into editor
                    try:
                        result = self.data_handlers.load_data_for_annotator(None, dataset_path)
                        if result and len(result) >= 9:
                            annotated_value, dropdown, metadata, slider, slice_info, crosshair_text, status_message, window_level, window_width = result
                            
                            # Update contribute tab dataset to show current status
                            contribute_dataset, contribute_status = contribute_components['get_assigned_tasks_dataset'](user_id)
                            
                            # Prepare remaining assignments info
                            from crowdsourcing.campaign_manager import CrowdsourcingManager
                            cm = CrowdsourcingManager()
                            remaining = cm.get_remaining_assignments_for_user(user_id)
                            remaining_count = len(remaining) - 1  # Subtract 1 because we just loaded one
                            
                            # Create assignments remaining display
                            if remaining_count > 0:
                                assignments_display = f"""
                                <div style='padding: 12px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 8px; border: 1px solid #3b82f6; margin: 8px 0;'>
                                    <div style='color: #1e40af; font-weight: 600; font-size: 13px; text-align: center;'>
                                        📋 {remaining_count} assignments remaining after this one
                                    </div>
                                </div>
                                """
                            else:
                                assignments_display = f"""
                                <div style='padding: 12px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 8px; border: 1px solid #10b981; margin: 8px 0;'>
                                    <div style='color: #047857; font-weight: 600; font-size: 13px; text-align: center;'>
                                        🎉 This is your final assignment!
                                    </div>
                                </div>
                                """
                            
                            return [
                                gr.update(selected=1),  # Switch to Editor tab
                                gr.update(),  # dicom_viewer
                                annotated_value,  # image_display
                                slider,  # slice_slider  
                                gr.update(visible=True),  # prev_slice_btn
                                gr.update(visible=True),  # next_slice_btn
                                metadata,  # metadata_display
                                gr.update(visible=True),  # submit_btn
                                gr.update(value=load_status, visible=True),  # submission_status
                                gr.update(visible=True, open=True),  # welcome_guide
                                self._populate_welcome_guide_info(task_info),  # welcome_guide_content
                                gr.update(value=assignments_display, visible=True),  # assignments_remaining
                                gr.update(visible=remaining_count > 0),  # next_assignment_btn
                                task_info,  # selected_task_info
                                contribute_dataset,  # Update contribute tab dataset
                                contribute_status,  # Update contribute tab status
                            ]
                    except Exception as e:
                        logger.error(f"Error loading next assignment data: {e}")
                
                # Failed to load - return error states
                return [
                    gr.update(),  # tabs (no change)
                    gr.update(),  # dicom_viewer
                    gr.update(),  # image_display
                    gr.update(),  # slice_slider
                    gr.update(),  # prev_slice_btn
                    gr.update(),  # next_slice_btn
                    gr.update(),  # metadata_display
                    gr.update(),  # submit_btn
                    gr.update(value=load_status, visible=True),  # submission_status (show error)
                    gr.update(),  # welcome_guide
                    gr.update(),  # welcome_guide_content
                    gr.update(),  # assignments_remaining
                    gr.update(),  # next_assignment_btn
                    "",  # selected_task_info
                    gr.update(),  # contribute tab dataset
                    gr.update(),  # contribute tab status
                ]
            
            editor_components['crowdsourcing']['next_assignment_btn'].click(
                fn=handle_next_assignment_wrapper,
                inputs=[contribute_components['current_user_state']],
                outputs=[
                    tabs,  # tabs
                    editor_components['visualization'][0],                       # dicom_viewer
                    editor_components['visualization'][3],                       # image_display (this is the image_annotator)
                    editor_components['visualization'][7],                       # slice_slider
                    editor_components['visualization'][6],                       # prev_slice_btn
                    editor_components['visualization'][8],                       # next_slice_btn
                    editor_components['visualization'][1],                       # metadata_display
                    editor_components['crowdsourcing']['submit_btn'],            # submit_btn
                    editor_components['crowdsourcing']['status'],                # submission_status
                    editor_components['welcome_modal']['guide'],                 # welcome_guide
                    editor_components['welcome_modal']['content'],               # welcome_guide_content
                    editor_components['crowdsourcing']['assignments_remaining'], # assignments_remaining
                    editor_components['crowdsourcing']['next_assignment_btn'],   # next_assignment_btn
                    contribute_components['selected_task_info'],                 # selected_task_info
                    contribute_components['tasks_dataset'],                      # contribute tab dataset
                    contribute_components['task_status'],                        # contribute tab status
                ]
            )
            
            # Connect welcome guide close button
            editor_components['welcome_modal']['close_btn'].click(
                fn=lambda: gr.update(visible=False, open=False),
                outputs=[editor_components['welcome_modal']['guide']]
            )
        
        # Auto-refresh contribute tab when tab becomes visible
        def handle_tab_change(tab_id):
            """Handle tab change and auto-refresh contribute tab when selected"""
            if tab_id == 5:  # Contribute tab has id=5
                # Auto-refresh the tasks when contribute tab is selected
                user_id = update_contribute_user_state()
                if user_id and 'get_assigned_tasks_dataset' in contribute_components:
                    try:
                        dataset, status = contribute_components['get_assigned_tasks_dataset'](user_id)
                        return dataset, status
                    except Exception as e:
                        logger.error(f"Error auto-refreshing contribute tab: {e}")
                        return gr.update(), "Error loading assignments"
            return gr.update(), gr.update()
        
        # Connect tab change to auto-refresh
        if hasattr(tabs, 'change'):
            tabs.change(
                fn=handle_tab_change,
                inputs=[active_tab_state],
                outputs=[contribute_components['tasks_dataset'], contribute_components['task_status']]
            )
        
        # Update user state periodically
        if 'current_user_state' in contribute_components:
            contribute_components['current_user_state'].value = update_contribute_user_state()
    
    def _populate_welcome_guide_info(self, task_selection):
        """Populate the welcome guide with campaign and dataset information"""
        try:
            # Extract campaign and patient info from task selection
            campaign_id = "Loading..."
            patient_id = "Loading..."
            modality_type = "Loading..."
            dataset_path = "Loading..."
            
            if task_selection and '|' in task_selection:
                parts = task_selection.split('|')
                if len(parts) >= 3:
                    campaign_id = parts[0].strip()
                    patient_id = parts[1].strip()
                    dataset_path = parts[2].strip()  # Get dataset path directly from task selection
                    
                    # Try to extract modality from dataset path
                    if dataset_path:
                        path_lower = dataset_path.lower()
                        logger.info(f"Analyzing dataset path for modality: {dataset_path}")
                        
                        if 'flair' in path_lower:
                            modality_type = "FLAIR MRI"
                        elif 't1' in path_lower and 'flair' not in path_lower:
                            modality_type = "T1-weighted MRI"
                        elif 't2' in path_lower:
                            modality_type = "T2-weighted MRI"
                        elif 't1c' in path_lower:
                            modality_type = "T1 Contrast-enhanced MRI"
                        elif 'dwi' in path_lower:
                            modality_type = "Diffusion-weighted MRI"
                        else:
                            # Try to extract from the last part of the path
                            path_parts = dataset_path.replace('\\', '/').split('/')
                            if path_parts:
                                last_part = path_parts[-1].lower()
                                logger.info(f"Checking last part of path: {last_part}")
                                if 'flair' in last_part:
                                    modality_type = "FLAIR MRI"
                                elif any(mod in last_part for mod in ['t1', 't2', 'dwi']):
                                    modality_type = f"MRI ({last_part.upper()})"
                                else:
                                    modality_type = "Medical Image"
                        
                        logger.info(f"Detected modality: {modality_type} from path: {dataset_path}")
                    else:
                        dataset_path = "Path not available"
                        modality_type = "Unknown"
            
            # Generate updated HTML with the actual values
            updated_html = f"""
            <div style='padding: 15px; background: #1f2937; border-radius: 8px; color: white;'>
                <div style='background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 20px; border-radius: 8px; margin-bottom: 15px; text-align: center;'>
                    <h2 style='color: white; margin: 0 0 8px 0; font-size: 20px; font-weight: bold;'>
                        🎯 All Set!
                    </h2>
                    <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;'>
                        Your dataset has been loaded and annotation tools are active
                    </p>
                </div>
                
                <div style='background: #374151; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #10b981;'>
                    <h3 style='color: #10b981; margin: 0 0 8px 0; font-size: 16px;'>📊 Current Assignment</h3>
                    <div style='color: #d1d5db; font-size: 14px; line-height: 1.4;'>
                        <p style='margin: 4px 0;'><strong>Campaign:</strong> {campaign_id}</p>
                        <p style='margin: 4px 0;'><strong>Patient:</strong> {patient_id}</p>
                        <p style='margin: 4px 0;'><strong>Modality:</strong> {modality_type}</p>                      
                    </div>
                </div>
                
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 15px;'>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #3b82f6;'>
                        <h4 style='color: #3b82f6; margin: 0 0 6px 0; font-size: 14px;'>🖱️ Annotate</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Click and drag to create annotations
                        </p>
                    </div>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #10b981;'>
                        <h4 style='color: #10b981; margin: 0 0 6px 0; font-size: 14px;'>🔍 Navigate</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Use slider or Prev/Next buttons
                        </p>
                    </div>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #f59e0b;'>
                        <h4 style='color: #f59e0b; margin: 0 0 6px 0; font-size: 14px;'>🤖 AI Help</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Use VLM Tools for suggestions, SAM models for segmtentation
                        </p>
                    </div>
                    <div style='background: #374151; padding: 12px; border-radius: 6px; border-left: 3px solid #ef4444;'>
                        <h4 style='color: #ef4444; margin: 0 0 6px 0; font-size: 14px;'>💾 Submit</h4>
                        <p style='color: #d1d5db; margin: 0; font-size: 12px; line-height: 1.3;'>
                            Scroll down to submit work
                        </p>
                    </div>
                </div>
                
                <div style='background: #065f46; padding: 12px; border-radius: 6px; text-align: center;'>
                    <p style='margin: 0; color: #d1fae5; font-size: 13px; font-weight: 500;'>
                        ⚠️ <strong>Caution:</strong> Annotations are NOT saved automatically as you work. 
                    </p>
                </div>
            </div>
            """
            
            return updated_html
            
        except Exception as e:
            logger.error(f"Error populating welcome guide info: {e}")
            return """
            <div style='padding: 15px; background: #1f2937; border-radius: 8px; color: white;'>
                <div style='text-align: center;'>
                    <h2 style='color: white; margin: 0 0 8px 0; font-size: 20px;'>🎯 Welcome to the Editor</h2>
                    <p style='color: rgba(255,255,255,0.9); margin: 0;'>Ready to start annotating!</p>
                </div>
            </div>
            """

# Initialize and launch the application
if __name__ == "__main__":
    logger.info("Starting SegMed-Pro application")
    segmed_pro = SegMedPro()
    app = segmed_pro.build_interface()
    app.launch(share=False)
