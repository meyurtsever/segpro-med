"""
SegMed-Pro Label Manager Tab UI Components

This module contains the UI layout for the label manager tab,
including loading, editing, and saving ITK-SNAP label sets.
"""

import gradio as gr


def create_label_manager_tab():
    with gr.TabItem("Label Manager"):
        gr.Markdown("## Label Manager: Edit and Manage Label Sets")
        
        # Main 3-column layout
        with gr.Row():
            # First column: File operations
            with gr.Column(scale=1):
                # First row: Data load                
                label_file_input = gr.File(
                    label="📁 Load ITK-SNAP Label Set (.label)", 
                    file_types=[".label"]
                )
                
                # Second and third rows: Load and Save buttons
                load_btn = gr.Button("🔄 Load Labels", variant="primary")
                save_btn = gr.Button("💾 Save Labels", variant="secondary")
                
            # Second column: Current labels table
            with gr.Column(scale=2):
                gr.Markdown("### Current Labels")
                label_table = gr.Dataframe(
                    #headers=["ID", "Name", "Color Preview", "R", "G", "B"],
                    headers=["ID", "Name", "Color Preview"],
                    #datatype=["number", "str", "html", "number", "number", "number"]
                    datatype=["number", "str", "html"],  # Enable HTML rendering for color preview
                    column_widths=[60, 120, 80],  # Set width for columns
                    interactive=True,  # Allow editing
                    wrap=True
                )
            
            # Third column: Add new label and operations
            with gr.Column(scale=1):
                # First row: Add new label section
                gr.Markdown("### ➕ Add New Label")
                new_label_name = gr.Textbox(
                    label="Label Name",
                    placeholder="Enter new label name...",
                    value="New Label"
                )
                new_label_color = gr.ColorPicker(
                    label="Label Color",
                    value="#FF0000"
                )
                add_label_btn = gr.Button("➕ Add Label", variant="primary")
                
                # Second row: Label operations
                gr.Markdown("### 🔧 Label Operations")
                selected_label_idx = gr.Number(
                    label="Label ID to Delete",
                    precision=0,
                    value=1,
                    minimum=0
                )
                delete_label_btn = gr.Button("🗑️ Delete Label", variant="stop")
        
        # Status display spanning all columns at the bottom
        status_box = gr.Textbox(label="📋 Status", interactive=False, lines=2)
        
        return (
            label_file_input, load_btn, save_btn, label_table, 
            new_label_name, new_label_color, add_label_btn,
            selected_label_idx, delete_label_btn, status_box
        )
