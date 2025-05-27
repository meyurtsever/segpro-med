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

# Add current directory to path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our modular components
from ui.state import AppState
from ui.viewer_tab import create_viewer_tab
from ui.conversion_tab import create_conversion_tab
from ui.label_manager_tab_new import create_label_manager_tab
from ui.handlers import DataLoadingHandlers, ViewerHandlers, PlotToolHandlers, ConversionHandlers
from ui.segmentation_handlers import SegmentationHandlers
from ui.label_manager_handlers import LabelManagerHandlers


class SegMedPro:
    """Main application class for SegMed-Pro"""
    
    def __init__(self):
        """Initialize the application"""
        # Initialize application state
        self.state = AppState()
          # Initialize event handlers
        self.data_handlers = DataLoadingHandlers(self.state)
        self.viewer_handlers = ViewerHandlers(self.state)
        self.plot_handlers = PlotToolHandlers(self.state)
        self.conversion_handlers = ConversionHandlers(self.state)
        self.segmentation_handlers = SegmentationHandlers(self.state)
        self.label_manager_handlers = LabelManagerHandlers(self.state)
        
        logger.info("SegMed-Pro application initialized")
    
    def build_interface(self):
        """Build the complete Gradio interface"""
        with gr.Blocks(title="SegMed-Pro") as app:
            gr.Markdown("# SegMed-Pro: Medical Imaging Annotation Tool")
            with gr.Tabs() as tabs:
                # Create viewer tab
                viewer_components = create_viewer_tab()
                # Create conversion tab
                conversion_components = create_conversion_tab()
                # Create label manager tab
                label_manager_components = create_label_manager_tab()

            # Connect event handlers for viewer tab
            self._connect_viewer_handlers(viewer_components)
            # Connect event handlers for conversion tab
            self._connect_conversion_handlers(conversion_components)
            # Connect event handlers for label manager tab
            self._connect_label_manager_handlers(label_manager_components)
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
            fn=self.data_handlers.load_data,
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
            fn=self.data_handlers.load_data,
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
        
        # Connect viewer handlers
        slice_slider.change(
            fn=self.viewer_handlers.update_slice,
            inputs=[slice_slider, view_selector],
            outputs=[image_plot, slice_text, crosshair_info, metadata_display, window_level, window_width]
        )
        
        view_selector.change(
            fn=self.viewer_handlers.change_view,
            inputs=[view_selector],
            outputs=[slice_slider, slice_text, image_plot]
        )
        
        apply_window_btn.click(
            fn=self.viewer_handlers.update_window_level,
            inputs=[window_level, window_width],
            outputs=[image_plot]
        )
        
        file_browser.change(
            fn=self.viewer_handlers.select_file_from_browser,
            inputs=[file_browser],
            outputs=[image_plot, slice_text, crosshair_info, metadata_display, slice_slider, window_level, window_width]
        )
        
        # Previous/Next button functionality
        prev_btn.click(
            fn=self.viewer_handlers.prev_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        next_btn.click(
            fn=self.viewer_handlers.next_slice,
            inputs=[slice_slider],
            outputs=[slice_slider]
        )
        
        # Connect plot tool handlers
        self._connect_plot_tool_handlers(plot_tools, image_plot)
        
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
            fn=lambda: self.plot_handlers.update_plot_tool("draw_circle"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_rect_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("draw_rect"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_line_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("draw_line"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_openpath_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("draw_openpath"),
            inputs=None,
            outputs=image_plot
        )
        
        draw_closedpath_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("draw_closedpath"),
            inputs=None,
            outputs=image_plot
        )
        
        erase_shape_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("erase_shape"),
            inputs=None,
            outputs=image_plot
        )
        
        pan_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("pan"),
            inputs=None,
            outputs=image_plot
        )
        
        zoom_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("zoom"),
            inputs=None,
            outputs=image_plot
        )
        
        reset_btn.click(
            fn=lambda: self.plot_handlers.update_plot_tool("reset"),
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
        (label_file_input, load_btn, save_btn, label_table, 
         new_label_name, new_label_color, add_label_btn,
         selected_label_name, delete_label_btn, status_box,
         update_label_dropdown) = components  # updated to receive update_label_dropdown
        
        handlers = self.label_manager_handlers

        # Load label set
        load_btn.click(
            fn=handlers.load_label_set,
            inputs=[label_file_input],
            outputs=[label_table, status_box]
        ).then(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name]
        )
        # Save label set
        save_btn.click(
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


# Initialize and launch the application
if __name__ == "__main__":
    logger.info("Starting SegMed-Pro application")
    segmed_pro = SegMedPro()
    app = segmed_pro.build_interface()
    app.launch(share=False)
