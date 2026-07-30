"""UI layout for the existing SegMed-Pro Label Manager tab."""

import gradio as gr

from utils.label_config import get_default_labels


def _initial_table_rows():
    rows = [
        [
            label['id'],
            label['name'],
            label['color'],
            label['active'],
        ]
        for label in get_default_labels()
    ]
    rows.append(["", "+", "#808080", True])
    return rows


def create_label_manager_tab():
    with gr.TabItem("Label Manager", id=3):
        gr.Markdown("## Label Manager: Edit and Manage Label Sets")

        # Preserve the existing three-column Label Manager structure.
        with gr.Row():
            with gr.Column(scale=1):
                label_file_input = gr.File(
                    label="📁 Load ITK-SNAP Label Set (.label)",
                    file_types=[".label"],
                )
                load_btn = gr.Button("🔄 Load Labels", variant="primary")

                gr.Markdown("### 💾 Save Options")
                save_filename = gr.Textbox(
                    label="Save As (filename)",
                    placeholder="my_labels.label",
                    value="label_set_edited.label",
                    info="Enter filename with .label extension",
                )
                with gr.Row():
                    save_btn = gr.Button(
                        "💾 Save Labels", variant="secondary", scale=2
                    )
                    save_quick_btn = gr.Button(
                        "⚡ Quick Save", variant="primary", scale=1
                    )

                with gr.Row():
                    gr.Markdown(
                        "💡 **Save Labels**: Creates a new file with your "
                        "custom filename",
                        elem_classes=["tip-text"],
                    )
                    gr.Markdown(
                        "💡 **Quick Save**: Overwrites the default "
                        "`label_set_edited.label`",
                        elem_classes=["tip-text"],
                    )

            with gr.Column(scale=2):
                gr.Markdown("### Current Labels")
                gr.HTML(
                    """
                    <div style="padding: 10px 12px; margin-bottom: 8px;
                    border-left: 4px solid #22c55e; border-radius: 6px;
                    background: rgba(34, 197, 94, 0.12); color: #bbf7d0;">
                    Add your own labels with the existing controls, then keep
                    <strong>Active</strong> checked for fast selection after
                    drawing a shape.
                    </div>
                    """
                )
                gr.HTML(
                    """
                    <div style="padding: 10px 12px; margin-bottom: 8px;
                    border-left: 4px solid #eab308; border-radius: 6px;
                    background: rgba(234, 179, 8, 0.12); color: #fef08a;">
                    Clear <strong>Active</strong> for labels you rarely use.
                    Your labels and choices are saved privately to your account.
                    </div>
                    """
                )
                label_table = gr.Dataframe(
                    value=_initial_table_rows(),
                    headers=["ID", "Name", "Color", "Active"],
                    datatype=["number", "str", "str", "bool"],
                    row_count=(1, "dynamic"),
                    column_widths=[60, 150, 90, 75],
                    static_columns=[0],
                    interactive=True,
                    wrap=True,
                )
                gr.Markdown(
                    """
                    **Tips**
                    - **Set active/passive:** Check or clear **Active** directly in the table.
                    - **Edit labels:** Change a name or hexadecimal color directly in the table.
                    - **Quick add:** Use the **+** row, then enter its name and color.
                    - **Delete custom labels:** Select the label in **Label Operations** and delete it.
                    """
                )

            with gr.Column(scale=1):
                gr.Markdown("### ➕ Add New Label")
                with gr.Row():
                    new_label_name = gr.Textbox(
                        label="Label Name",
                        placeholder="Enter new label name...",
                        value="New Label",
                        scale=1,
                    )
                    new_label_color = gr.ColorPicker(
                        label="Label Color",
                        value="#FF0000",
                        scale=2,
                    )
                add_label_btn = gr.Button(
                    "➕ Add Label", variant="primary"
                )

                gr.Markdown("### 🔧 Label Operations")
                selected_label_name = gr.Dropdown(
                    label="Custom Label to Delete",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=False,
                )
                delete_label_btn = gr.Button(
                    "🗑️ Delete Label", variant="stop"
                )

        status_box = gr.Textbox(
            label="📋 Status", interactive=False, lines=2
        )

        def update_label_dropdown(table_data):
            if hasattr(table_data, 'values'):
                rows = table_data.values.tolist()
            elif isinstance(table_data, list):
                rows = table_data
            else:
                rows = []
            names = [
                row[1] for row in rows
                if (
                    len(row) > 1
                    and str(row[1]).strip()
                    and str(row[1]).strip() != "+"
                )
            ]
            return gr.update(choices=names, value=None)

        label_table.change(
            fn=update_label_dropdown,
            inputs=[label_table],
            outputs=[selected_label_name],
        )

        return (
            label_file_input,
            load_btn,
            save_btn,
            save_quick_btn,
            save_filename,
            label_table,
            new_label_name,
            new_label_color,
            add_label_btn,
            selected_label_name,
            delete_label_btn,
            status_box,
            update_label_dropdown,
        )
