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
        
        # Initialize checkbox state tracking
        self._point_checkbox_state = False
        
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
        
        # Connect MEDSAM2 handlers to other components that need them
        self.image_viewer_handlers.medsam2_handlers = self.medsam2_handlers
        self.image_plot_tool_handlers.medsam2_handlers = self.medsam2_handlers
    
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
            fn=self.custom_annotator_handlers.handle_annotation_change,
            inputs=[annotator],
            outputs=[]
        )
        
        clear_coords_btn.click(
            fn=self.custom_annotator_handlers.clear_selected_coordinates,
            inputs=[],
            outputs=[coordinates_text]
        )
        
        clear_overlays_btn.click(
            fn=self.medsam2_handlers.clear_annotation_overlays,
            inputs=[],
            outputs=[error_display, annotator]
        )

    def _connect_editor_handlers(self, components):
        """Connect all event handlers for the editor tab (mirroring viewer tab, plus MEDSAM2 functionality)"""
        data_loading = components['data_loading']
        visualization = components['visualization']
        ai_tools = components['ai_tools']  # Now we have AI tools including MEDSAM2
        
        (file_input, dir_input, load_btn, reset_dir_btn, file_browser, 
         metadata_display_dl, error_display_dl, window_level, window_width, apply_window_btn, debug_btn) = data_loading
        (error_display, metadata_display, view_selector, image_display, prev_btn, slice_slider, next_btn, slice_text, crosshair_info, viewer_3d, viewer_3d_controls, refresh_3d_btn, export_3d_btn) = visualization        
        (point_prompt_checkbox, box_prompt_checkbox, coordinates_text, clear_coords_btn, clear_overlays_btn, ai_model_selector, 
         processing_mode, score_threshold,
         output_dir, save_visualizations, device_selector, 
         auto_brain_annotate_btn, annotate_btn, annotation_status) = ai_tools
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
        )        # Viewer handlers (using annotator-specific methods)
        slice_slider.change(
            fn=self.image_plot_tool_handlers.handle_annotator_slider_change,
            inputs=[slice_slider, image_display],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width]
        )
        view_selector.change(
            fn=self.image_plot_tool_handlers.change_view_for_annotator,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, image_display]
        )
        apply_window_btn.click(
            fn=self.image_plot_tool_handlers.update_window_level_for_annotator,
            inputs=[window_level, window_width],
            outputs=[image_display]
        )
        file_browser.change(
            fn=self.image_plot_tool_handlers.select_file_from_browser_for_annotator,
            inputs=[file_browser],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, slice_slider, window_level, window_width]        )
          # Navigation buttons with annotation saving
        def handle_prev_navigation(current_slider_value, current_annotated_value):
            """Handle previous button click with annotation saving"""
            result_tuple, new_slider_value = self.image_plot_tool_handlers.handle_annotator_navigation(
                "prev", current_slider_value, current_annotated_value
            )
            # result_tuple contains (image_display, slice_text, crosshair_info, metadata, window_level, window_width)
            return result_tuple + (new_slider_value,)
        
        def handle_next_navigation(current_slider_value, current_annotated_value):
            """Handle next button click with annotation saving"""  
            result_tuple, new_slider_value = self.image_plot_tool_handlers.handle_annotator_navigation(
                "next", current_slider_value, current_annotated_value
            )
            # result_tuple contains (image_display, slice_text, crosshair_info, metadata, window_level, window_width)
            return result_tuple + (new_slider_value,)
        
        prev_btn.click(
            fn=handle_prev_navigation,
            inputs=[slice_slider, image_display],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width, slice_slider]
        )
        
        next_btn.click(            fn=handle_next_navigation,            inputs=[slice_slider, image_display],
            outputs=[image_display, slice_text, crosshair_info, metadata_display, window_level, window_width, slice_slider]
        )
        
        # Connect processing mode change event
        def handle_processing_mode_change(mode):
            """Handle processing mode change - show/hide score threshold and 3D viewer"""
            score_threshold_update = gr.Slider(visible=(mode == "All Records"))
            viewer_3d_update = gr.Plot(visible=(mode == "All Records"))
            viewer_controls_update = gr.Row(visible=(mode == "All Records"))
            return score_threshold_update, viewer_3d_update, viewer_controls_update
        
        processing_mode.change(
            fn=handle_processing_mode_change,
            inputs=[processing_mode],
            outputs=[score_threshold, viewer_3d, viewer_3d_controls]
        )
        
        # 3D Viewer event handlers
        def handle_refresh_3d_view(output_dir_value):
            """Handle refresh 3D view button click"""
            try:
                # Import the 3D visualization function from editor_tab
                from ui.editor_tab import create_3d_visualization
                fig = create_3d_visualization(output_dir_value)
                return fig
            except Exception as e:
                from ui.editor_tab import create_empty_3d_plot
                return create_empty_3d_plot(f"Error refreshing view: {str(e)}")
        
        def handle_export_3d_view(output_dir_value):
            """Handle export 3D view button click"""
            try:
                from ui.editor_tab import create_3d_visualization
                import os
                fig = create_3d_visualization(output_dir_value)
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
            """Handle point prompt checkbox change - ensure mutual exclusivity and clear coords"""
            if point_checked:
                # If point is checked, uncheck box and show coordinates
                self.medsam2_handlers.enable_point_mode()
                return True, False, gr.update(visible=True)
            else:                # If point is unchecked, hide coordinates and clear them
                self.medsam2_handlers.disable_point_mode()
                self.medsam2_handlers.selected_coordinates = []  # Clear coordinates from handler state
                return False, box_checked, gr.update(visible=False)
        
        def handle_box_prompt_change(box_checked, point_checked):
            """Handle box prompt checkbox change - ensure mutual exclusivity and enable box mode"""
            if box_checked:
                # If box is checked, uncheck point, hide coordinates, and enable box mode
                self.medsam2_handlers.disable_point_mode()
                self.medsam2_handlers.enable_box_mode()
                self.medsam2_handlers.selected_coordinates = []  # Clear coordinates from handler state
                self.medsam2_handlers.prompt_boxes = []  # Clear any existing box prompts                # Show instructions for box mode
                return (True, False, 
                       gr.update(visible=True, 
                               label="Box Prompt Status", 
                               value="📦 Box mode enabled. Ready to capture box coordinates.\n\nSteps:\n1. Use the box tool in the image editor above\n2. Draw rectangles around regions to segment\n3. Box coordinates are automatically captured\n4. Click 'Run MEDSAM2 Annotation' when ready",
                               info="Draw boxes on the image. Coordinates are automatically captured for MEDSAM2."))
            else:
                # If box is unchecked, disable box mode and keep point state
                self.medsam2_handlers.disable_box_mode()
                if point_checked:
                    self.medsam2_handlers.enable_point_mode()
                    return (False, True, 
                           gr.update(visible=True, 
                                   label="Selected Coordinates (x,y)", 
                                   value="",
                                   info="Click on the image to select coordinates"))
                else:
                    self.medsam2_handlers.disable_point_mode()
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
        )
        
        def handle_image_select_conditionally(evt: gr.SelectData):
            """Handle image select events only when point mode is enabled, ignore in box mode"""
            # Check if point mode is enabled
            if (hasattr(self.medsam2_handlers, 'point_mode_enabled') and 
                self.medsam2_handlers.point_mode_enabled and 
                not getattr(self.medsam2_handlers, 'box_mode_enabled', False)):
                return self.medsam2_handlers.handle_image_click(evt)
            else:
                # In box mode or when point mode is disabled, don't process select events at all
                return ""
        
        # Connect MEDSAM2 handlers - conditional based on mode
        image_display.select(
            fn=handle_image_select_conditionally,
            inputs=[],
            outputs=[coordinates_text]
        )
        
        def handle_image_annotation_change(annotated_image_value):
            """Handle both normal annotation changes and box prompt extraction"""
            try:
                # First, let the normal annotation handler process the change
                self.image_plot_tool_handlers.on_annotation_change(annotated_image_value)
                
                # If box mode is enabled, also extract box prompts for MEDSAM2
                if hasattr(self.medsam2_handlers, 'box_mode_enabled') and self.medsam2_handlers.box_mode_enabled:
                    status_msg = self.medsam2_handlers.handle_box_annotation(annotated_image_value)
                    logger.info(f"Box annotation handled: {status_msg}")
                
                return None  # No outputs to avoid circular dependency
            except Exception as e:
                logger.error(f"Error handling image annotation change: {str(e)}")
                return None
          # Save annotations immediately when they change (edits, deletions, additions)
        image_display.change(
            fn=handle_image_annotation_change,
            inputs=[image_display],
            outputs=[]  # No outputs to avoid circular dependency
        )
        
        def clear_all_prompts_and_overlays():
            """Clear all prompts, coordinates, and annotation overlays"""
            self.medsam2_handlers.clear_coordinates()
            self.medsam2_handlers.prompt_boxes = []
            status, image = self.medsam2_handlers.clear_annotation_overlays()
            return "", status, image, gr.update(visible=False)  # Clear coordinates, update status, clear overlays, hide clear button
        
        clear_coords_btn.click(
            fn=clear_all_prompts_and_overlays,
            inputs=[],
            outputs=[coordinates_text, annotation_status, image_display, clear_overlays_btn]        )
        
        clear_overlays_btn.click(
            fn=self.medsam2_handlers.clear_annotation_overlays,
            inputs=[],
            outputs=[annotation_status, image_display]
        )
        
        # Custom wrapper function to handle the 3-tuple return and button visibility
        def handle_annotation_workflow(output_dir, save_visualizations, device_selector, processing_mode, score_threshold, image_display_data):
            status, image, success = self.medsam2_handlers.run_full_annotation_workflow(
                output_dir, save_visualizations, device_selector, processing_mode, score_threshold, image_display_data
            )
            
            # If processing mode is "All Records" and annotation was successful, refresh 3D viewer
            # Always return a valid Plotly figure to avoid the __module__ attribute error
            from ui.editor_tab import create_empty_3d_plot, create_3d_visualization
            if processing_mode == "All Records" and success:
                try:
                    viewer_3d_update = create_3d_visualization(output_dir)
                except Exception as e:
                    viewer_3d_update = create_empty_3d_plot(f"Error updating 3D view: {str(e)}")
            else:
                # For non-"All Records" mode, show message
                viewer_3d_update = create_empty_3d_plot("3D Viewer available in 'All Records' mode")
            
            return status, image, gr.update(visible=success), viewer_3d_update
        
        annotate_btn.click(
            fn=handle_annotation_workflow,
            inputs=[output_dir, save_visualizations, device_selector, processing_mode, score_threshold, image_display],
            outputs=[annotation_status, image_display, clear_overlays_btn, viewer_3d]
        )
        
        # Auto-brain annotation handler
        def handle_auto_brain_annotation(output_dir, save_visualizations, device_selector, processing_mode, score_threshold, dir_input_value):
            """Handle automatic brain structure annotation"""
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
            # Run the automatic annotation with the user's score threshold
            status, annotated_result, success = self.medsam2_handlers.run_medsam2_annotation_automatic(
                dicom_folder, output_dir, save_visualizations, device_selector, processing_mode, score_threshold
            )
            
            # If processing mode is "All Records" and annotation was successful, refresh 3D viewer
            # Always return a valid Plotly figure to avoid the __module__ attribute error
            from ui.editor_tab import create_empty_3d_plot, create_3d_visualization
            if processing_mode == "All Records" and success:
                try:
                    viewer_3d_update = create_3d_visualization(output_dir)
                except Exception as e:
                    viewer_3d_update = create_empty_3d_plot(f"Error updating 3D view: {str(e)}")
            else:
                # For non-"All Records" mode, show message
                viewer_3d_update = create_empty_3d_plot("3D Viewer available in 'All Records' mode")
            
            # Return the annotated result directly to the image_annotator
            return status, annotated_result, gr.update(visible=success), viewer_3d_update
        
        auto_brain_annotate_btn.click(
            fn=handle_auto_brain_annotation,
            inputs=[output_dir, save_visualizations, device_selector, processing_mode, score_threshold, dir_input],
            outputs=[annotation_status, image_display, clear_overlays_btn, viewer_3d]
        )
          # Handle image removal from image_annotator (X button / clear button)
        image_display.clear(
            fn=self.image_plot_tool_handlers.handle_image_remove,
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
            elif prompt_type == "Point-based Prompt":
                return gr.update(
                    value="📍 **POINT PROMPT MODE**: \n"
                    "1. Click points on the image to guide segmentation\n"
                    "2. Use multiple points for better accuracy\n"
                    "3. Click 'Run MEDSAM2 Annotation (Manual Prompts)' to segment",
                    visible=True
                )
            else:  # Automatic Brain Detection
                return gr.update(
                    value="🧠 **AUTOMATIC MODE**: Click 'Auto-Annotate Brain Structures' for automatic brain region detection",
                    visible=True
                )

        # Add instruction text component
        instruction_text = gr.Markdown(
            value="🧠 **AUTOMATIC MODE**: Click 'Auto-Annotate Brain Structures' for automatic brain region detection",
            visible=True        )


# Initialize and launch the application
if __name__ == "__main__":
    logger.info("Starting SegMed-Pro application")
    segmed_pro = SegMedPro()
    app = segmed_pro.build_interface()
    app.launch(share=False)
