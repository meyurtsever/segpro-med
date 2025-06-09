"""
SegMed-Pro Main Application

Refactored main application class that orchestrates the UI components and event handlers
following Gradio best practices for modular application structure.
"""

import gradio as gr
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from ui.plot_handlers import PlotViewerHandlers, PlotToolHandlers
from ui.image_handlers import ImageViewerHandlers, ImagePlotToolHandlers
from ui.segmentation_handlers import SegmentationHandlers
from ui.label_manager_handlers import LabelManagerHandlers
from ui.medsam2_handlers import MEDSAM2Handlers
from ui.custom_annotator_handlers import CustomAnnotatorHandlers


class SegMedPro:
    """Main application class for SegMed-Pro"""
    
    def __init__(self):
        """Initialize the application"""
        # Initialize application state
        self.state = AppState()
          # Initialize event handlers
        self.data_handlers = DataLoadingHandlers(self.state)
        self.plot_viewer_handlers = PlotViewerHandlers(self.state)  # For viewer tab (gr.Plot)
        self.image_viewer_handlers = ImageViewerHandlers(self.state)  # For editor tab (gr.Image)
        self.plot_tool_handlers = PlotToolHandlers(self.state)  # For viewer tab plot tools
        self.image_plot_tool_handlers = ImagePlotToolHandlers(self.state)  # For editor tab (dummy)
        self.conversion_handlers = ConversionHandlers(self.state)
        self.segmentation_handlers = SegmentationHandlers(self.state)
        self.label_manager_handlers = LabelManagerHandlers(self.state)
        self.medsam2_handlers = MEDSAM2Handlers(self.state)
        self.custom_annotator_handlers = CustomAnnotatorHandlers(self.state)
          # Update image viewer handlers with medsam2 reference for overlay support
        self.image_viewer_handlers.medsam2_handlers = self.medsam2_handlers
        
        logger.info("SegMed-Pro application initialized")
    
    def build_interface(self):
        """Build the complete Gradio interface"""
        with gr.Blocks(title="SegMed-Pro") as app:
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
                # Create custom annotator tab
                custom_annotator_components = create_custom_annotator_tab()            # Connect event handlers for viewer tab
            self._connect_viewer_handlers(viewer_components)
            # Connect event handlers for editor tab
            self._connect_editor_handlers(editor_components)
            # Connect event handlers for conversion tab
            self._connect_conversion_handlers(conversion_components)
            # Connect event handlers for label manager tab
            self._connect_label_manager_handlers(label_manager_components)
            # Connect event handlers for custom annotator tab
            self._connect_custom_annotator_handlers(custom_annotator_components)
            # Connect event handlers for custom annotator tab
            self._connect_custom_annotator_handlers(custom_annotator_components)
            return app

    def _connect_viewer_handlers(self, components):
        """Connect all event handlers for the viewer tab"""
        # Unpack components
        data_loading = components['data_loading']
        visualization = components['visualization']
        plot_tools = components['plot_tools']
        segmentation = components['segmentation']
        
        # Unpack individual components
        (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
         metadata_display, error_display, window_level, window_width, 
         apply_window_btn, debug_btn) = data_loading
        
        (view_selector, image_plot, prev_btn, slice_slider, next_btn, 
         slice_text, crosshair_info) = visualization
        
        (draw_circle_btn, draw_rect_btn, draw_line_btn, draw_openpath_btn,
         draw_closedpath_btn, erase_shape_btn, pan_btn, zoom_btn, reset_btn) = plot_tools
        
        (segmentation_file, load_seg_btn, seg_opacity, label_file, 
         clear_seg_btn, convert_to_shapes_btn, seg_status) = segmentation
          # Connect data loading handlers
        load_btn.click(
            fn=self.data_handlers.load_data_for_plot,
            inputs=[file_input, dir_input],
            outputs=[
                image_plot, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width
            ]
        )
        
        reset_dir_btn.click(
            fn=self.data_handlers.reset_directory,
            inputs=[],
            outputs=[dir_input]
        )
        
        # Auto-trigger load_data on file selection
        file_input.change(
            fn=self.data_handlers.load_data_for_plot,
            inputs=[file_input, dir_input],
            outputs=[
                image_plot, file_browser, metadata_display,
                slice_slider, slice_text, crosshair_info, error_display,
                window_level, window_width
            ]
        )
        debug_btn.click(
            fn=self.data_handlers.debug_selected_file,
            inputs=[file_input, dir_input],
            outputs=[error_display]
        )
          # Connect viewer handlers (using plot-specific methods)
        slice_slider.change(
            fn=self.plot_viewer_handlers.update_slice_for_plot,
            inputs=[slice_slider, view_selector],
            outputs=[image_plot, slice_text, crosshair_info, metadata_display, window_level, window_width]
        )
        
        view_selector.change(
            fn=self.plot_viewer_handlers.change_view_for_plot,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, image_plot]
        )
        
        apply_window_btn.click(
            fn=self.plot_viewer_handlers.update_window_level_for_plot,
            inputs=[window_level, window_width],
            outputs=[image_plot]
        )
        file_browser.change(
            fn=self.plot_viewer_handlers.select_file_from_browser_for_plot,
            inputs=[file_browser],
            outputs=[image_plot, slice_text, crosshair_info, metadata_display, slice_slider, window_level, window_width]
        )
        
        # Previous/Next button functionality
        prev_btn.click(
            fn=self.plot_viewer_handlers.prev_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        next_btn.click(
            fn=self.plot_viewer_handlers.next_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
          # Connect plot tool handlers (using plot-specific methods for gr.Plot)
        self._connect_plot_tool_handlers_for_plot(plot_tools, image_plot)
        
        # Connect segmentation handlers
        load_seg_btn.click(
            fn=self.segmentation_handlers.direct_load_segmentation,
            inputs=[segmentation_file, label_file],
            outputs=[seg_status, image_plot]
        )
        
        seg_opacity.change(
            fn=self.segmentation_handlers.update_segmentation_opacity,
            inputs=[seg_opacity],
            outputs=[seg_status, image_plot]
        )
        
        clear_seg_btn.click(
            fn=self.segmentation_handlers.clear_segmentation,
            inputs=[],
            outputs=[seg_status, image_plot]
        )
        
        convert_to_shapes_btn.click(
            fn=self.segmentation_handlers.convert_segmentation_to_shapes,
            inputs=[],
            outputs=[seg_status, image_plot]
        )
    
    def _connect_plot_tool_handlers(self, plot_tools, image_plot):
        """Connect plot tool button handlers"""
        (draw_circle_btn, draw_rect_btn, draw_line_btn, draw_openpath_btn,
         draw_closedpath_btn, erase_shape_btn, pan_btn, zoom_btn, reset_btn) = plot_tools
          # Connect plot tool buttons
        draw_circle_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("draw_circle"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_rect_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("draw_rect"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_line_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("draw_line"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_openpath_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("draw_openpath"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_closedpath_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("draw_closedpath"),
            inputs=None,
            outputs=image_plot
        )
        
        erase_shape_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("erase_shape"),
            inputs=None,
            outputs=image_plot
        )
        pan_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("pan"),
            inputs=None,
            outputs=image_plot
        )
        
        zoom_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("zoom"),
            inputs=None,
            outputs=image_plot
        )
        reset_btn.click(
            fn=lambda: self.image_plot_tool_handlers.update_plot_tool("reset"),
            inputs=None,
            outputs=image_plot
        )

    def _connect_plot_tool_handlers_for_plot(self, plot_tools, image_plot):
        """Connect plot tool button handlers for gr.Plot component"""
        (draw_circle_btn, draw_rect_btn, draw_line_btn, draw_openpath_btn,
         draw_closedpath_btn, erase_shape_btn, pan_btn, zoom_btn, reset_btn) = plot_tools
          # Connect plot tool buttons for plotly figures
        draw_circle_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("draw_circle"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_rect_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("draw_rect"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_line_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("draw_line"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_openpath_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("draw_openpath"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_closedpath_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("draw_closedpath"),
            inputs=None,
            outputs=image_plot
        )
        
        erase_shape_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("erase_shape"),
            inputs=None,
            outputs=image_plot
        )
        pan_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("pan"),
            inputs=None,
            outputs=image_plot
        )
        
        zoom_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("zoom"),
            inputs=None,
            outputs=image_plot
        )
        
        reset_btn.click(
            fn=lambda: self.plot_tool_handlers.update_plot_tool_for_plot("reset"),
            inputs=None,
            outputs=image_plot
        )
    
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
        annotator_components = components['annotator']
        
        # Unpack data loading components
        (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
         metadata_display, error_display, window_level, window_width, 
         apply_window_btn, debug_btn) = data_loading
        
        # Unpack visualization components
        (view_selector, prev_btn, slice_slider, next_btn, slice_text) = visualization
        
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
        )

    def _connect_editor_handlers(self, components):
        """Connect all event handlers for the editor tab (mirroring viewer tab, plus MEDSAM2 functionality)"""
        data_loading = components['data_loading']
        visualization = components['visualization']
        ai_tools = components['ai_tools']  # Now we have AI tools including MEDSAM2
        
        (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
         metadata_display_dl, error_display_dl, window_level, window_width, apply_window_btn, debug_btn) = data_loading
        
        (error_display, metadata_display, view_selector, image_display, prev_btn, slice_slider, next_btn, slice_text, crosshair_info) = visualization
        
        (coordinates_text, clear_coords_btn, clear_overlays_btn, ai_model_selector, 
         processing_mode, score_threshold,
         output_dir, save_visualizations, device_selector, 
         annotate_btn, annotation_status) = ai_tools
        
        # Data loading handlers
        load_btn.click(
            fn=self.data_handlers.load_data,
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
            fn=self.data_handlers.load_data,
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
        
        # Viewer handlers
        slice_slider.change(
            fn=self.image_viewer_handlers.update_slice,
            inputs=[slice_slider, view_selector],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width]
        )
        view_selector.change(
            fn=self.image_viewer_handlers.change_view,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, image_display]
        )
        apply_window_btn.click(
            fn=self.image_viewer_handlers.update_window_level,
            inputs=[window_level, window_width],
            outputs=[image_display]
        )
        file_browser.change(
            fn=self.image_viewer_handlers.select_file_from_browser,
            inputs=[file_browser],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, slice_slider, window_level, window_width]
        )
        prev_btn.click(
            fn=self.image_viewer_handlers.prev_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        next_btn.click(
            fn=self.image_viewer_handlers.next_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        # Connect processing mode change event
        processing_mode.change(
            fn=lambda mode: gr.Slider(visible=(mode == "All Records")),
            inputs=[processing_mode],
            outputs=[score_threshold]       
        )
        
        # Connect MEDSAM2 handlers
        image_display.select(
            fn=self.medsam2_handlers.handle_image_click,
            inputs=[],
            outputs=[coordinates_text]
        )
        
        clear_coords_btn.click(
            fn=self.medsam2_handlers.clear_coordinates,
            inputs=[],
            outputs=[coordinates_text]
        )
        
        clear_overlays_btn.click(
            fn=self.medsam2_handlers.clear_annotation_overlays,
            inputs=[],
            outputs=[annotation_status, image_display]
        )
        
        # Custom wrapper function to handle the 3-tuple return and button visibility
        def handle_annotation_workflow(output_dir, save_visualizations, device_selector, processing_mode, score_threshold):
            status, image, success = self.medsam2_handlers.run_full_annotation_workflow(
                output_dir, save_visualizations, device_selector, processing_mode, score_threshold
            )
            return status, image, gr.update(visible=success)
        
        annotate_btn.click(
            fn=handle_annotation_workflow,
            inputs=[output_dir, save_visualizations, device_selector, processing_mode, score_threshold],
            outputs=[annotation_status, image_display, clear_overlays_btn]
        )


# Initialize and launch the application
if __name__ == "__main__":
    logger.info("Starting SegMed-Pro application")
    segmed_pro = SegMedPro()
    app = segmed_pro.build_interface()
    app.launch(share=False)
