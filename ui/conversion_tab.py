"""
SegMed-Pro Conversion Tab UI Components

This module contains the UI layout for the conversion tab,
which handles format conversion between different medical image formats.
"""

import gradio as gr


def create_conversion_tab() -> tuple:
    """Create the conversion tab layout"""
    with gr.TabItem("Conversion", id=2):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Format Conversion")
                conversion_input = gr.File(label="Input File(s)")
                conversion_dir = gr.Textbox(label="Or Input Directory")
                
                conversion_type = gr.Dropdown(
                    choices=[
                        "DICOM to NIFTI", 
                        "NIFTI to MAT", 
                        "DICOM to MAT",
                        "NIFTI to PNG"
                    ],
                    label="Conversion Type",
                    value="DICOM to NIFTI"
                )
                
                output_dir = gr.Textbox(label="Output Directory")
                convert_btn = gr.Button("Convert")
                conversion_status = gr.Textbox(label="Conversion Status")
                conversion_output = gr.File(label="Output File (if available)")
    
    return (conversion_input, conversion_dir, conversion_type, output_dir,
            convert_btn, conversion_status, conversion_output)
