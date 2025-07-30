"""
SegMed-Pro Label Manager Tab UI Components

This module contains the UI layout for the label manager tab,
including loading, editing, and saving ITK-SNAP label sets.
"""

import gradio as gr


def create_label_manager_tab():
    with gr.TabItem("Label Manager", id=3):
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
                
                # Save section with filename input
                gr.Markdown("### 💾 Save Options")
                save_filename = gr.Textbox(
                    label="Save As (filename)",
                    placeholder="my_labels.label",
                    value="label_set_edited.label",                    info="Enter filename with .label extension"
                )
                with gr.Row():
                    save_btn = gr.Button("💾 Save Labels", variant="secondary", scale=2)
                    save_quick_btn = gr.Button("⚡ Quick Save", variant="primary", scale=1)
                
                # Save button tips
                with gr.Row():
                    gr.Markdown(
                        "💡 **Save Labels**: Creates a new file with your custom filename",
                        elem_classes=["tip-text"]
                    )
                    gr.Markdown(
                        "💡 **Quick Save**: Overwrites the default 'label_set_edited.label'",
                        elem_classes=["tip-text"]
                    )
            # Second column: Current labels table
            with gr.Column(scale=2):
                gr.Markdown("### Current Labels")
                label_table = gr.Dataframe(
                    headers=["ID", "Name", "Color Preview"],
                    datatype=["number", "str", "html"],
                    column_widths=[60, 120, 80],
                    interactive=True,  # Allow editing and row delete
                    wrap=True
                )
                gr.Markdown(
                    """
                    **Tips**
                    - **Edit names:** Click on any label name in the table to edit it directly  
                    - **Delete labels:** Select the label from **Label Operations** and use the button to delete the label
                    """
                )
              # Third column: Add new label and operations
            with gr.Column(scale=1):
                # First row: Add new label section
                gr.Markdown("### ➕ Add New Label")
                with gr.Row():
                    new_label_name = gr.Textbox(
                        label="Label Name",
                        placeholder="Enter new label name...",
                        value="New Label",
                        scale=1
                    )
                    new_label_color = gr.ColorPicker(
                        label="Label Color",
                        value="#FF0000",
                        scale=2
                    )
                add_label_btn = gr.Button("➕ Add Label", variant="primary")
                  # Second row: Label operations
                gr.Markdown("### 🔧 Label Operations")
                selected_label_name = gr.Dropdown(
                    label="Label Name to Delete",
                    choices=[],  # Will be populated dynamically
                    value=None,  # Ensure initial value is None
                    interactive=True,
                    allow_custom_value=False  # Prevent custom values
                )
                delete_label_btn = gr.Button("🗑️ Delete Label", variant="stop")

        # Status display spanning all columns at the bottom
        status_box = gr.Textbox(label="📋 Status", interactive=False, lines=2)

        # --- DYNAMIC DROPDOWN POPULATION ---
        def update_label_dropdown(table_data):
            import pandas as pd
            if isinstance(table_data, pd.DataFrame):
                names = table_data.iloc[:, 1].tolist() if table_data.shape[1] > 1 else []
            elif table_data and isinstance(table_data, list):
                names = [row[1] for row in table_data if len(row) > 1]
            else:
                names = []

            # Return Gradio update with choices and value=None to avoid the warning
            return gr.update(choices=names, value=None)

        def delete_label_from_table(table_data, label_name):
            import pandas as pd
            if isinstance(table_data, pd.DataFrame):
                table_data = table_data.values.tolist()
            # Filter out the row with the selected label name
            updated_table = [row for row in table_data if row[1] != label_name]
            return updated_table, f"[OK] Deleted label '{label_name}'" if len(updated_table) < len(table_data) else (table_data, f"[ERROR] Label '{label_name}' not found")

        label_table.change(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name]
        )

        delete_label_btn.click(
            fn=delete_label_from_table,
            inputs=[label_table, selected_label_name],
            outputs=[label_table, status_box]
        ).then(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name]
        )        # Expose update_label_dropdown for use in app.py
        return (
            label_file_input, load_btn, save_btn, save_quick_btn, save_filename, label_table, 
            new_label_name, new_label_color, add_label_btn,
            selected_label_name, delete_label_btn, status_box,
            update_label_dropdown  # <-- add this to the return tuple
        )
