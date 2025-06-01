"""
Demo file to test the PlotlyBridge component
"""

import os
import sys
import gradio as gr
import plotly.graph_objects as go
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    logger.warning("PlotlyBridge component not found, hover events will not be captured.")

def create_plotly_figure():
    """Create a sample Plotly figure"""
    # Create a sample image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a gradient
    for i in range(100):
        for j in range(100):
            img[i, j, 0] = i * 255 // 100  # Red gradient
            img[i, j, 1] = j * 255 // 100  # Green gradient
            img[i, j, 2] = 100             # Constant blue
    
    # Create a Plotly figure
    fig = go.Figure(go.Image(
        z=img,
        hoverinfo='x+y',
        colormodel='rgb'
    ))
    
    # Configure layout
    fig.update_layout(
        dragmode='pan',
        hovermode='closest',
        hoverdistance=10,
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
        margin=dict(l=0, r=0, t=10, b=0, pad=0),
        plot_bgcolor='black',
        paper_bgcolor='black',
        autosize=False
    )
    
    # Enable hover template
    fig.update_traces(
        hovertemplate='x: %{x}<br>y: %{y}'
    )
    
    return fig

def handle_hover(hover_data_str):
    """Handle hover event data from the PlotlyBridge component"""
    if not hover_data_str:
        return "No hover data received"
    
    try:
        # Parse the hover data
        hover_data = json.loads(hover_data_str)
        
        # Extract coordinates
        x = int(hover_data.get('x', 0))
        y = int(hover_data.get('y', 0))
        
        # Create crosshair text
        crosshair_text = f"x: {x}, y: {y}"
        logger.info(f"Received hover event: {crosshair_text}")
        
        return crosshair_text
    except Exception as e:
        logger.error(f"Error handling hover event: {str(e)}")
        return f"Error processing hover data: {str(e)}"

def update_crosshair_display(hover_data_str):
    """Update display with crosshair at hover position"""
    if not hover_data_str:
        return create_plotly_figure()
    
    try:
        # Parse the hover data
        hover_data = json.loads(hover_data_str)
        
        # Extract coordinates
        x = int(hover_data.get('x', 0))
        y = int(hover_data.get('y', 0))
        
        # Create a new figure with crosshair
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create a gradient
        for i in range(100):
            for j in range(100):
                img[i, j, 0] = i * 255 // 100  # Red gradient
                img[i, j, 1] = j * 255 // 100  # Green gradient
                img[i, j, 2] = 100             # Constant blue
        
        # Add crosshair
        if 0 <= x < 100 and 0 <= y < 100:
            # Horizontal line
            img[y, :, :] = [255, 0, 0]  # Red line
            # Vertical line
            img[:, x, :] = [255, 0, 0]  # Red line
        
        # Create a Plotly figure
        fig = go.Figure(go.Image(
            z=img,
            hoverinfo='x+y',
            colormodel='rgb'
        ))
        
        # Configure layout
        fig.update_layout(
            dragmode='pan',
            hovermode='closest',
            hoverdistance=10,
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
            margin=dict(l=0, r=0, t=10, b=0, pad=0),
            plot_bgcolor='black',
            paper_bgcolor='black',
            autosize=False
        )
        
        # Enable hover template
        fig.update_traces(
            hovertemplate='x: %{x}<br>y: %{y}'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error updating crosshair display: {str(e)}")
        return create_plotly_figure()

def main():
    """Create and launch the demo interface"""
    with gr.Blocks(title="PlotlyBridge Demo") as demo:
        gr.Markdown("# PlotlyBridge Demo")
        
        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(value=create_plotly_figure())
                
                # Add the PlotlyBridge component if available
                if HAS_PLOTLY_BRIDGE:
                    bridge = PlotlyBridge(plot_id="plot")
                else:
                    bridge = None
            
            with gr.Column(scale=1):
                crosshair_info = gr.Textbox(label="Crosshair Position", value="x: 0, y: 0")
                
                # Button to reset the plot
                reset_btn = gr.Button("Reset Plot")
        
        # Connect the PlotlyBridge to handle hover events
        if bridge is not None:
            bridge.on_hover(
                fn=handle_hover,
                inputs=[],
                outputs=[crosshair_info]
            ).then(
                fn=update_crosshair_display,
                inputs=[bridge.hover_data],
                outputs=[plot]
            )
        
        # Reset button should recreate the figure
        reset_btn.click(
            fn=lambda: create_plotly_figure(),
            inputs=[],
            outputs=[plot]
        )
    
    # Launch the demo
    demo.launch()

if __name__ == "__main__":
    main()
