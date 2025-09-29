"""
Demo script showcasing the Material-UI style info tooltip system
Run this to see the clean, simple tooltips in action before integrating into the main app
"""

import gradio as gr
from ui.info_tooltips import (
    create_info_tooltip,
    create_section_header_with_info,
    create_segmentation_settings_header,
    create_load_medical_data_header,
    create_segmentation_with_ai_header,
    create_info_enhanced_component,
    get_medical_tooltip,
    TOOLTIP_CSS
)

def create_tooltip_demo():
    """Create a demo interface showing various Material-UI style tooltip implementations"""
    
    with gr.Blocks(
        title="Material-UI Style Tooltips Demo",
        css="""
        .header-with-info {
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 12px;
        }
        .component-with-info-row {
            align-items: flex-start;
            gap: 8px;
        }
        """ + TOOLTIP_CSS.replace('<style>', '').replace('</style>', '')
    ) as demo:
        
        gr.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='margin: 0; font-size: 24px; font-weight: bold;'>✨ Material-UI Style Tooltips</h1>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>Clean, simple tooltips without icons - hover over the small dots to see them in action</p>
        </div>
        """)
        
        # Demo 1: Section Headers with Info Tooltips (Single Components)
        gr.Markdown("### 📋 Section Headers with Clean Tooltips")
        
        create_load_medical_data_header()
        create_segmentation_settings_header()
        create_segmentation_with_ai_header()
        
        # Demo 2: Individual Components with Tooltips
        gr.Markdown("### 🎛️ Components with Individual Tooltips")
        
        with gr.Row(elem_classes="component-with-info-row"):
            with gr.Column(scale=10):
                file_input = gr.File(
                    label="Load Medical File",
                    file_types=[".dcm", ".nii", ".nii.gz", ".mat"]
                )
            with gr.Column(scale=1, min_width=30):
                create_info_tooltip(tooltip_text=get_medical_tooltip("file_input"))
        
        with gr.Row(elem_classes="component-with-info-row"):
            with gr.Column(scale=10):
                view_selector = gr.Radio(
                    choices=["Axial", "Sagittal", "Coronal"],
                    value="Axial",
                    label="View Orientation"
                )
            with gr.Column(scale=1, min_width=30):
                create_info_tooltip(tooltip_text=get_medical_tooltip("view_selector"))
        
        with gr.Row(elem_classes="component-with-info-row"):
            with gr.Column(scale=10):
                processing_mode = gr.Radio(
                    choices=["Single Slice", "All Records"],
                    value="Single Slice",
                    label="Processing Mode"
                )
            with gr.Column(scale=1, min_width=30):
                create_info_tooltip(tooltip_text=get_medical_tooltip("processing_mode"))
        
        with gr.Row(elem_classes="component-with-info-row"):
            with gr.Column(scale=10):
                score_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                    label="Quality Threshold"
                )
            with gr.Column(scale=1, min_width=30):
                create_info_tooltip(tooltip_text=get_medical_tooltip("score_threshold"))
        
        # Demo 3: Custom Tooltips with different content
        gr.Markdown("### 🎨 Custom Tooltip Styles")
        
        with gr.Row():
            create_info_tooltip(
                tooltip_text="<strong>Important Note!</strong><br>This is a custom styled tooltip with clean Material-UI design principles."
            )
            create_info_tooltip(
                tooltip_text="<strong>Safety Information:</strong><br>This tooltip shows important safety information for medical data processing."
            )
            create_info_tooltip(
                tooltip_text="<strong>Pro Tip:</strong><br>Clean design without icons improves accessibility and maintains professional appearance."
            )
            create_info_tooltip(
                tooltip_text="<strong>User Experience:</strong><br>Simple dots are less distracting while still providing contextual help when needed."
            )
        
        gr.HTML("""
        <div style='text-align: center; padding: 15px; background: #f8fafc; border-radius: 8px; margin-top: 30px; border: 1px solid #e2e8f0;'>
            <p style='margin: 0; color: #64748b; font-size: 14px;'>
                ✨ <strong>Clean & Professional:</strong> Material-UI inspired tooltips without visual clutter
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_tooltip_demo()
    demo.launch(
        share=False,
        server_name="localhost", 
        server_port=7861,
        show_error=True
    )
