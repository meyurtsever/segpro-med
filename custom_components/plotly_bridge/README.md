# PlotlyBridge Component for SegMed-Pro

## Overview

The PlotlyBridge component is a custom Svelte/JS component that wraps Plotly and enables bidirectional communication between Plotly hover events and Python backend. This allows for features such as updating crosshair position in the medical imaging application as the user hovers over the Plotly visualization.

## Key Features

1. **Hover Event Tracking**: Captures Plotly hover events and sends the coordinates to Python
2. **Crosshair Synchronization**: Updates the crosshair position based on hover location
3. **Non-intrusive**: Preserves all Plotly functionality without compromise
4. **Real-time Updates**: Provides immediate visual feedback as users interact with the visualization

## File Structure

```
custom_components/
  plotly_bridge/
    __init__.py            - Package initialization file
    component.py           - Python Gradio component implementation
    PlotlyBridge.svelte    - Svelte component implementation
ui/
  plotly_bridge_handler.py - Handler for processing hover events and updating display
```

## Usage

### In Editor Tab

The PlotlyBridge component is automatically integrated in the Editor tab. When you hover over the Plotly visualization, the crosshair will automatically update to reflect your cursor position.

### Programmatic Usage

To use the PlotlyBridge component in your own code:

```python
# Import the component
from custom_components.plotly_bridge import PlotlyBridge

# Create a Plotly visualization
plot = gr.Plot(value=your_plotly_figure)

# Add the PlotlyBridge component
bridge = PlotlyBridge(plot_id="your_plot_id")

# Connect hover events to your handler
bridge.on_hover(
    fn=your_hover_handler_function,
    inputs=[],
    outputs=[your_output_component]
).then(
    fn=your_update_display_function,
    inputs=[bridge.hover_data],
    outputs=[plot]
)
```

## Demo

A standalone demo is provided in `plotly_bridge_demo.py`. Run it to see how the component works:

```bash
python plotly_bridge_demo.py
```

## Implementation Details

1. The component uses JavaScript to detect and monitor Plotly hover events
2. Hover coordinates are captured and sent to Python via a hidden Gradio textbox
3. The Python handler processes the coordinates and updates the crosshair
4. The display is refreshed with the new crosshair position while maintaining all Plotly functionality

## Technical Notes

- The component automatically finds the Plotly element in the DOM, even if it's loaded dynamically
- Hover events are debounced to prevent excessive updates
- The component works with all Plotly dragmodes and interaction features
