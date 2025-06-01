"""
Simple test script for the PlotlyBridge component
This script creates a minimal test app to verify that hover events are being processed correctly.
"""

import gradio as gr
import plotly.graph_objects as go
import numpy as np
import json
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add custom_components path to Python path
custom_components_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_components")
if custom_components_path not in sys.path:
    sys.path.append(custom_components_path)

# Import our custom PlotlyBridge component
try:
    from custom_components.plotly_bridge import PlotlyBridge
    HAS_PLOTLY_BRIDGE = True
except ImportError:
    HAS_PLOTLY_BRIDGE = False
    logger.error("PlotlyBridge component not found, hover events will not be captured.")

def create_test_figure():
    """Create a simple test figure"""
    # Create a sample 2D array (100x100)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # Create a Plotly figure
    fig = go.Figure(data=go.Heatmap(z=Z, colorscale='Viridis'))
    
    # Configure layout for hover events
    fig.update_layout(
        title="Hover Test Figure",
        dragmode='pan',
        hovermode='closest',
        hoverdistance=10
    )
    
    # Enable hover template
    fig.update_traces(
        hoverinfo='x+y+z',
        hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}'
    )
    
    return fig

def handle_hover(hover_data_str):
    """Handle hover event data"""
    logger.info(f"Received hover data: {hover_data_str}")
    
    if not hover_data_str:
        return "No hover data received"
    
    try:
        # Parse the hover data
        hover_data = json.loads(hover_data_str)
        
        # Extract coordinates
        x = hover_data.get('x', 'N/A')
        y = hover_data.get('y', 'N/A')
        
        # Create hover text
        hover_text = f"Hover at x: {x}, y: {y}"
        logger.info(hover_text)
        
        return hover_text
    except Exception as e:
        logger.error(f"Error handling hover event: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def update_display(hover_data_str):
    """Update the display with crosshair at hover position"""
    if not hover_data_str:
        return create_test_figure()
    
    try:
        # Parse the hover data
        hover_data = json.loads(hover_data_str)
        
        # Extract coordinates
        hover_x = int(hover_data.get('x', 50))
        hover_y = int(hover_data.get('y', 50))
        
        # Create a new figure
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        # Create a Plotly figure
        fig = go.Figure()
        
        # Add the heatmap
        fig.add_trace(go.Heatmap(z=Z, colorscale='Viridis'))
        
        # Add vertical crosshair line
        fig.add_trace(go.Scatter(
            x=[hover_x/10, hover_x/10],
            y=[0, 10],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
        
        # Add horizontal crosshair line
        fig.add_trace(go.Scatter(
            x=[0, 10],
            y=[hover_y/10, hover_y/10],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
        
        # Configure layout
        fig.update_layout(
            title=f"Crosshair at x={hover_x}, y={hover_y}",
            dragmode='pan',
            hovermode='closest',
            hoverdistance=10
        )
        
        # Enable hover template
        fig.update_traces(
            hoverinfo='x+y+z',
            hovertemplate='x: %{x}<br>y: %{y}<br>z: %{z}'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error updating display: {str(e)}", exc_info=True)
        return create_test_figure()

def main():
    """Create and launch the test interface"""
    with gr.Blocks(title="PlotlyBridge Test") as demo:
        gr.Markdown("# PlotlyBridge Hover Test")
        
        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(value=create_test_figure())
                
                # Add the PlotlyBridge component if available
                if HAS_PLOTLY_BRIDGE:
                    bridge = PlotlyBridge(plot_id="plot")
                    gr.Markdown("✅ PlotlyBridge component is active")
                else:
                    bridge = None
                    gr.Markdown("❌ PlotlyBridge component is not available")
            
            with gr.Column(scale=1):
                hover_info = gr.Textbox(label="Hover Info", value="Move mouse over plot to see hover data")
                
                # Debug info
                with gr.Accordion("Debug Info", open=False):
                    raw_hover_data = gr.Textbox(label="Raw Hover Data")
                    
                # Buttons
                reset_btn = gr.Button("Reset Plot")
                test_btn = gr.Button("Test Connection")
          # Connect the PlotlyBridge to handle hover events
        if bridge is not None:
            # Set up event chain
            bridge.on_hover(
                fn=handle_hover
            ).then(
                fn=lambda x: x,
                inputs=[],
                outputs=[hover_info]
            )
            
            # Also show raw hover data for debugging
            bridge.hover_data.change(
                fn=lambda x: x,
                inputs=[bridge.hover_data],
                outputs=[raw_hover_data]
            )
            
            # Update display with crosshair
            bridge.hover_data.change(
                fn=update_display,
                inputs=[bridge.hover_data],
                outputs=[plot]
            )
        
        # Reset button should recreate the figure
        reset_btn.click(
            fn=lambda: create_test_figure(),
            inputs=[],
            outputs=[plot]
        )
        
        # Test button to verify component is working
        test_btn.click(
            fn=lambda: "Connection test successful!",
            inputs=[],
            outputs=[hover_info]
        )
    
    # Launch the demo
    demo.launch()

if __name__ == "__main__":
    main()
