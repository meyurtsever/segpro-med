# filepath: c:\Users\Yurtsever\Downloads\segpro-med\test_plotly_bridge.py
"""
Test script for PlotlyBridge component.
This provides a minimal test environment to verify the PlotlyBridge is working.
"""

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom component
from custom_components.plotly_bridge import PlotlyBridge

# Create a simple state class for testing
class TestState:
    def __init__(self):
        self.crosshair_position = (0, 0, 0)
        self.current_view = 'axial'
        self.current_slice_idx = 0
        self.current_data = np.random.rand(10, 10, 10)  # Dummy data
        
# Create a handler for PlotlyBridge events
class TestHandler:
    def __init__(self, state):
        self.state = state
        
    def handle_hover_event(self, hover_data_str):
        """Handle hover events from PlotlyBridge"""
        print(f"Received hover data: {hover_data_str}")
        
        if not hover_data_str:
            return "No data"
            
        try:
            data = json.loads(hover_data_str)
            x, y = data.get('x', 0), data.get('y', 0)
            self.state.crosshair_position = (x, y, self.state.current_slice_idx)
            return f"Hover: x={x}, y={y}"
        except Exception as e:
            print(f"Error handling hover: {e}")
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
            return f"Click: x={x}, y={y}"
        except Exception as e:
            print(f"Error handling click: {e}")
            return f"Error: {str(e)}"
            
    def update_plot(self):
        """Update the plot with current crosshair"""
        x, y, z = self.state.crosshair_position
        
        # Create a simple plot
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=self.state.current_data[:, :, z],
            colorscale='Viridis'
        ))
        
        # Add crosshair
        fig.add_shape(
            type="line", 
            x0=x, y0=0, x1=x, y1=10,
            line=dict(color="red", width=2)
        )
        fig.add_shape(
            type="line",
            x0=0, y0=y, x1=10, y1=y,
            line=dict(color="red", width=2)
        )
        
        fig.update_layout(
            title=f"Crosshair at {self.state.crosshair_position}",
            width=400, height=400
        )
        
        return fig

# Build the test interface
def build_test_interface():
    # Create test state and handler
    state = TestState()
    handler = TestHandler(state)
    
    with gr.Blocks() as demo:
        gr.Markdown("# PlotlyBridge Test")
        
        with gr.Row():
            with gr.Column():
                # Create a plot with a specific ID
                plot_id = "test_plotly_plot"
                plot = gr.Plot(handler.update_plot(), elem_id=plot_id)
                
                # Create our PlotlyBridge component
                bridge = PlotlyBridge(plot_id=plot_id, sync_crosshair=True)
                
                # Display hover/click info
                info = gr.Textbox(label="Hover/Click Info")
            
            with gr.Column():
                gr.Markdown("### Debug Info")
                debug_btn = gr.Button("Force Update")
                status = gr.Textbox(label="Component Status")
        
        # Connect events
        bridge.on_hover(fn=handler.handle_hover_event, inputs=[], outputs=[info])
        bridge.on_click(fn=handler.handle_click_event, inputs=[], outputs=[info])
        
        # Debug button to force update
        debug_btn.click(
            fn=lambda: f"PlotlyBridge status: ID={plot_id}, Found={bridge is not None}",
            inputs=[],
            outputs=[status]
        )
    
    return demo

# Launch the test interface
if __name__ == "__main__":
    demo = build_test_interface()
    demo.launch()