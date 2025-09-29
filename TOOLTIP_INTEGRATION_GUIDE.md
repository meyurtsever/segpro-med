"""
Integration Guide for Material-UI Style Tooltips
===============================================

This guide shows how to integrate the clean tooltip system into your app.py.

## Quick Integration Steps:

1. Import the tooltip CSS at the top of your Gradio app:

```python
from ui.info_tooltips import TOOLTIP_CSS

# In your Gradio Blocks definition:
with gr.Blocks(css=TOOLTIP_CSS.replace('<style>', '').replace('</style>', '')) as app:
    # Your app content
```

2. Replace section headers:

BEFORE:
```python
gr.Markdown("## Load Medical Data")
```

AFTER:
```python
from ui.info_tooltips import create_load_medical_data_header
create_load_medical_data_header()
```

3. Add tooltips to individual components:

```python
from ui.info_tooltips import create_info_tooltip, get_medical_tooltip

# Method 1: Component with separate tooltip
file_input = gr.File(label="Load Medical File")
create_info_tooltip(tooltip_text=get_medical_tooltip("file_input"))

# Method 2: Row with component and tooltip
with gr.Row():
    with gr.Column(scale=10):
        slider = gr.Slider(label="Quality Threshold")
    with gr.Column(scale=1, min_width=30):
        create_info_tooltip(tooltip_text="Your tooltip text here")
```

## Available Header Functions:

- `create_load_medical_data_header()` - For data loading sections
- `create_segmentation_settings_header()` - For AI settings sections  
- `create_segmentation_with_ai_header()` - For AI segmentation sections

## Available Tooltip Functions:

- `create_info_tooltip(tooltip_text)` - Basic tooltip with small dot
- `get_medical_tooltip(key)` - Predefined medical tooltips
- `create_section_header_with_info(title, info_text, level)` - Custom headers

## Predefined Tooltip Keys:

- "file_input" - File upload tooltip
- "dir_input" - Directory path tooltip  
- "view_selector" - View orientation tooltip
- "processing_mode" - Processing mode tooltip
- "score_threshold" - Quality threshold tooltip
- "window_level" - Window level tooltip
- "window_width" - Window width tooltip

## Example Usage:

```python
import gradio as gr
from ui.info_tooltips import (
    TOOLTIP_CSS,
    create_load_medical_data_header,
    create_info_tooltip,
    get_medical_tooltip
)

with gr.Blocks(css=TOOLTIP_CSS.replace('<style>', '').replace('</style>', '')) as app:
    
    # Section header with tooltip
    create_load_medical_data_header()
    
    # Component with tooltip
    with gr.Row():
        with gr.Column(scale=10):
            file_input = gr.File(label="Load Medical File")
        with gr.Column(scale=1, min_width=30):
            create_info_tooltip(tooltip_text=get_medical_tooltip("file_input"))
    
    # More components...
```

## Troubleshooting:

1. **Tooltips not showing**: Make sure TOOLTIP_CSS is included in your Gradio Blocks
2. **Layout issues**: Ensure you're using the Row/Column structure as shown above
3. **Overflow to next line**: Use `min_width=30` on tooltip columns
4. **Styling conflicts**: The CSS uses `!important` to override conflicts

## Features:

- ✅ Clean Material-UI inspired design
- ✅ No emoji icons - just simple dots
- ✅ Responsive and accessible
- ✅ Works in all major browsers
- ✅ Smooth hover animations
- ✅ High z-index to appear above other elements
- ✅ Professional appearance
"""
