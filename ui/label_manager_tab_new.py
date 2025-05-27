"""
SegMed-Pro Label Manager Tab UI Components

This module contains the UI layout for the label manager tab,
including loading, editing, and saving ITK-SNAP label sets.
"""

import gradio as gr


def create_label_manager_tab():
    with gr.TabItem("Label Manager"):
        gr.Markdown("## Label Manager: Edit and Manage Label Sets")
        
        # File operations section
        with gr.Row():
            with gr.Column(scale=2):
                label_file_input = gr.File(
                    label="📁 Load ITK-SNAP Label Set (.label)", 
                    file_types=[".label"]
                )
            with gr.Column(scale=1):
                load_btn = gr.Button("🔄 Load Labels", variant="primary")
                save_btn = gr.Button("💾 Save Labels", variant="secondary")
          # Current labels display - editable table (only name column can be edited)
        gr.Markdown("### Current Labels")
        label_table = gr.Dataframe(
            headers=["ID", "Name", "Color Preview", "R", "G", "B"],
            datatype=["number", "str", "str", "number", "number", "number"],
            interactive=True,  # Allow editing
            label="Label Set",
            wrap=True
        )
        
        # Add new label section
        gr.Markdown("### ➕ Add New Label")
        with gr.Row():
            with gr.Column(scale=2):
                new_label_name = gr.Textbox(
                    label="Label Name",
                    placeholder="Enter new label name...",
                    value="New Label"
                )
            with gr.Column(scale=1):
                new_label_color = gr.ColorPicker(
                    label="Label Color",
                    value="#FF0000"
                )
            with gr.Column(scale=1):
                add_label_btn = gr.Button("➕ Add Label", variant="primary")
        
        # Label operations section
        gr.Markdown("### 🔧 Label Operations")
        with gr.Row():
            with gr.Column():
                selected_label_idx = gr.Number(
                    label="Label ID to Delete",
                    precision=0,
                    value=1,
                    minimum=0
                )
            with gr.Column():
                delete_label_btn = gr.Button("🗑️ Delete Label", variant="stop")
        
        # Status display
        status_box = gr.Textbox(label="📋 Status", interactive=False, lines=2)
        
        return (
            label_file_input, load_btn, save_btn, label_table, 
            new_label_name, new_label_color, add_label_btn,
            selected_label_idx, delete_label_btn, status_box
        )
