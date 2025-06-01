# filepath: c:\Users\Yurtsever\Downloads\segpro-med\debug_plotly_bridge.py
"""
Debug script for testing the PlotlyBridge component in isolation.
Run this script to check if the component is working properly.
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the PlotlyBridge component
try:
    from custom_components.plotly_bridge import PlotlyBridge
    HAS_PLOTLY_BRIDGE = True
    logger.info("PlotlyBridge component imported successfully")
except ImportError:
    HAS_PLOTLY_BRIDGE = False
    logger.warning("PlotlyBridge component not found, will use fallback")

# Create a mock state for testing
class MockState:
    def __init__(self):
        self.crosshair_position = (0, 0, 0)
        self.current_view = 'axial'
        self.current_slice_idx = 0
        
        # Create a test volume (10x10x10)
        self.current_data = np.zeros((10, 10, 10))
        for i in range(10):
            self.current_data[i, :, :] = i  # Fill each slice with its index
            
        logger.info("Mock state initialized")

# Create a mock handler
class MockPlotlyBridgeHandler:
    def __init__(self, state):
        self.state = state
        logger.info("Mock handler initialized")
    
    def handle_hover_event(self, hover_data_str):
        """Handle hover events from PlotlyBridge"""
        print(f"Received hover data: {hover_data_str}")
        
        if not hover_data_str:
            return "No data"
        
        try:
            data = json.loads(hover_data_str)
            x, y = data.get('x', 0), data.get('y', 0)
            self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            return f"Hover at x={x}, y={y}"
        except Exception as e:
            print(f"Error processing hover: {e}")
            return f"Error: {str(e)}"
    
    def handle_click_event(self, click_data_str):
        """Handle click events from PlotlyBridge"""
        print(f"Received click data: {click_data_str}")
        
        if not click_data_str:
            return "No data"
        
        try:
            data = json.loads(click_data_str)
            x, y = data.get('x', 0), data.get('y', 0)
            self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            return f"Click at x={x}, y={y}"
        except Exception as e:
            print(f"Error processing click: {e}")
            return f"Error: {str(e)}"
    
    def update_crosshair_display(self):
        """Update the display with the current crosshair position"""
        print(f"Updating display with crosshair at {self.state.crosshair_position}")
        
        x, y, z = self.state.crosshair_position
        
        # Create a heatmap figure for testing
        fig = px.imshow(
            self.state.current_data[z],
            labels=dict(x="X", y="Y", color="Value"),
            title=f"Slice {z} with Crosshair at ({x}, {y})"
        )
        
        # Add crosshair lines
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=9, line=dict(color="red", width=2))
        fig.add_shape(type="line", x0=0, y0=y, x1=9, y1=y, line=dict(color="red", width=2))
        
        return fig

# Build the debug interface
def build_debug_interface():
    state = MockState()
    handler = MockPlotlyBridgeHandler(state)
    
    with gr.Blocks(title="PlotlyBridge Debug Tool") as demo:
        gr.Markdown("# PlotlyBridge Debug Tool")
        gr.Markdown("This tool tests the PlotlyBridge component in isolation to verify hover and click events.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Create a Plotly plot with a specific elem_id
                plot_id = "debug_plotly_plot"
                with gr.Box():
                    gr.Markdown("### Test Plot")
                    plot = gr.Plot(
                        handler.update_crosshair_display(),
                        elem_id=plot_id
                    )
                    
                # Status and coordinate display
                with gr.Row():
                    event_info = gr.Textbox(
                        label="Event Info",
                        value="Hover or click on the plot",
                        interactive=False
                    )
                    coords_display = gr.Textbox(
                        label="Coordinates",
                        value="X: -- Y: --",
                        interactive=False,
                        elem_id="debug-coords-display"
                    )
            
            with gr.Column(scale=1):
                with gr.Box():
                    gr.Markdown("### Component Status")
                    component_status = gr.Textbox(
                        label="Component Status",
                        value="Initializing...",
                        interactive=False
                    )
                    
                    # Test Controls
                    refresh_btn = gr.Button("Refresh Plot")
                    test_hover_btn = gr.Button("Simulate Hover Event")
                    test_click_btn = gr.Button("Simulate Click Event")
                    
                    # Log display
                    event_log = gr.Textbox(
                        label="Event Log",
                        value="",
                        lines=10,
                        max_lines=10,
                        interactive=False
                    )
        
        # Add PlotlyBridge component if available
        if HAS_PLOTLY_BRIDGE:
            plotly_bridge = PlotlyBridge(plot_id=plot_id, sync_crosshair=True)
            component_status.update(value="PlotlyBridge component loaded successfully")
            
            # Connect hover events
            plotly_bridge.on_hover(
                fn=handler.handle_hover_event
            ).then(
                fn=lambda text: text,
                outputs=[event_info]
            ).then(
                fn=handler.update_crosshair_display,
                outputs=[plot]
            ).then(
                fn=lambda data: f"X: {state.crosshair_position[0]} Y: {state.crosshair_position[1]}",
                outputs=[coords_display]
            ).then(
                fn=lambda data: f"{time.strftime('%H:%M:%S')} - Hover event processed\n" + event_log.value,
                outputs=[event_log]
            )
            
            # Connect click events
            plotly_bridge.on_click(
                fn=handler.handle_click_event
            ).then(
                fn=lambda text: text,
                outputs=[event_info]
            ).then(
                fn=handler.update_crosshair_display,
                outputs=[plot]
            ).then(
                fn=lambda data: f"X: {state.crosshair_position[0]} Y: {state.crosshair_position[1]}",
                outputs=[coords_display]
            ).then(
                fn=lambda data: f"{time.strftime('%H:%M:%S')} - Click event processed\n" + event_log.value,
                outputs=[event_log]
            )
        else:
            component_status.update(value="PlotlyBridge component NOT available")
        
        # Button handlers
        refresh_btn.click(
            fn=handler.update_crosshair_display,
            outputs=[plot]
        ).then(
            fn=lambda: f"{time.strftime('%H:%M:%S')} - Plot refreshed\n" + event_log.value,
            outputs=[event_log]
        )
        
        # Simulate a hover event
        test_hover_btn.click(
            fn=lambda: handler.handle_hover_event(json.dumps({"x": 5, "y": 5, "timestamp": time.time()})),
            outputs=[event_info]
        ).then(
            fn=handler.update_crosshair_display,
            outputs=[plot]
        ).then(
            fn=lambda: f"X: {state.crosshair_position[0]} Y: {state.crosshair_position[1]}",
            outputs=[coords_display]
        ).then(
            fn=lambda: f"{time.strftime('%H:%M:%S')} - Simulated hover event\n" + event_log.value,
            outputs=[event_log]
        )
        
        # Simulate a click event
        test_click_btn.click(
            fn=lambda: handler.handle_click_event(json.dumps({"x": 7, "y": 3, "timestamp": time.time()})),
            outputs=[event_info]
        ).then(
            fn=handler.update_crosshair_display,
            outputs=[plot]
        ).then(
            fn=lambda: f"X: {state.crosshair_position[0]} Y: {state.crosshair_position[1]}",
            outputs=[coords_display]
        ).then(
            fn=lambda: f"{time.strftime('%H:%M:%S')} - Simulated click event\n" + event_log.value,
            outputs=[event_log]
        )
        
    return demo

# Launch the debug interface
if __name__ == "__main__":
    logger.info("Starting PlotlyBridge debug tool")
    demo = build_debug_interface()
    demo.launch()