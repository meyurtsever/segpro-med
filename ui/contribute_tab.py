"""
Contribute tab for expert users to work on assigned annotation tasks
"""

import gradio as gr
import logging
import os
from crowdsourcing.campaign_manager import CrowdsourcingManager

logger = logging.getLogger(__name__)

def create_info_message(message):
    """Create a Next.js styled info message with light green background"""
    return f"""
    <div style='padding: 16px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border: 1px solid #10b981; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
        <div style='display: flex; align-items: center; gap: 12px;'>
            <div style='width: 12px; height: 12px; background: #10b981; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3);'></div>
            <div style='color: #047857; font-weight: 500; font-size: 14px; line-height: 1.5;'>
                {message}
            </div>
        </div>
    </div>
    """

def create_error_message(message):
    """Create a Next.js styled error message with light red background"""
    return f"""
    <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 12px; border: 1px solid #ef4444; margin: 8px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
        <div style='display: flex; align-items: center; gap: 12px;'>
            <div style='width: 12px; height: 12px; background: #ef4444; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.3);'></div>
            <div style='color: #dc2626; font-weight: 500; font-size: 14px; line-height: 1.5;'>
                {message}
            </div>
        </div>
    </div>
    """

def create_contribute_tab():
    """Create the contribute tab for experts"""
    
    # Create a fresh instance and ensure we load the latest data
    crowdsourcing_manager = CrowdsourcingManager()
    
    def get_assigned_tasks_table(user_id):
        """Get assigned tasks for the current expert as table data and radio choices"""
        if not user_id:
            empty_table = [["No user logged in", "", "", "Please log in first"]]
            return empty_table, [], create_info_message("🔑 Please log in to view your assignments.")
        
        # Force reload assignments from file to ensure we have the latest data
        crowdsourcing_manager.load_assignments()
        
        # Get both remaining and completed assignments
        all_assigned = crowdsourcing_manager.get_assigned_tasks(user_id)
        completed = crowdsourcing_manager.assignments
        
        if not all_assigned:
            empty_table = [["No assignments", "", "", "No tasks assigned"]]
            return empty_table, [], create_info_message("📋 No tasks assigned to you yet. Contact your administrator for task assignments.")
        
        # Build table data and radio choices
        table_data = []
        radio_choices = []
        task_counter = 1
        
        for task in all_assigned:
            # Handle both 'campaign' and 'campaign_id' keys for compatibility
            campaign_id = task.get('campaign_id') or task.get('campaign')
            patient_id = task['patient_id']
            dataset_path = task['dataset_path']
            
            # Check if this task is completed
            status = "Pending"
            if campaign_id in completed:
                if user_id in completed[campaign_id].get('completed', {}):
                    if patient_id in completed[campaign_id]['completed'][user_id]:
                        status = "Annotations Sent"
            
            # Create table row for gr.Dataset
            status_icon = "🟢" if status == "Annotations Sent" else "🟡"
            
            # Style completed rows differently to show they're disabled
            if status == "Annotations Sent":
                # Add visual indication for completed/disabled rows
                task_cell = f"<span style='color: #6b7280; opacity: 0.6;'>{status_icon} Task {task_counter}</span>"
                campaign_cell = f"<span style='color: #6b7280; opacity: 0.6;'>{campaign_id}</span>"
                patient_cell = f"<span style='color: #6b7280; opacity: 0.6;'>{patient_id}</span>"
                status_cell = f"<span style='color: #10b981; font-weight: 500;'>{status}</span>"
            else:
                # Normal active row styling
                task_cell = f"{status_icon} Task {task_counter}"
                campaign_cell = campaign_id
                patient_cell = patient_id
                status_cell = f"<span style='color: #f59e0b; font-weight: 500;'>{status}</span>"
            
            table_data.append([
                task_cell,
                campaign_cell,
                patient_cell,
                status_cell
            ])
            
            # Create radio choice for the separate radio component (backup)
            radio_value = f"{campaign_id}|{patient_id}|{dataset_path}"
            radio_label = f"{status_icon} Task {task_counter}: {campaign_id} - {patient_id} ({status})"
            radio_choices.append((radio_label, radio_value))
            
            task_counter += 1
        
        pending_count = len([task for task in all_assigned 
                           if not (task.get('campaign_id', task.get('campaign')) in completed and 
                                 user_id in completed[task.get('campaign_id', task.get('campaign'))].get('completed', {}) and
                                 task['patient_id'] in completed[task.get('campaign_id', task.get('campaign'))]['completed'][user_id])])
        
        if pending_count == 0:
            status_msg = create_info_message(f"🎉 All {len(all_assigned)} assignments completed! You have finished all your assigned tasks.")
        else:
            status_msg = create_info_message(f"📊 You have {pending_count} pending tasks out of {len(all_assigned)} total assignments.")
        
        return table_data, radio_choices, status_msg
    
    def get_assigned_tasks(user_id):
        """Get assigned tasks for the current expert (excluding completed ones) - Legacy dropdown support"""
        if not user_id:
            return gr.update(choices=[], value=None), "Please log in first"
        
        # Get remaining assignments (excludes completed tasks)
        remaining = crowdsourcing_manager.get_remaining_assignments_for_user(user_id)
        
        if not remaining:
            return gr.update(choices=[], value=None), "No remaining tasks assigned to you."
        
        # Create choices for dropdown (still needed for backward compatibility)
        choices = []
        for task in remaining:
            # Handle both 'campaign' and 'campaign_id' keys for compatibility
            campaign_id = task.get('campaign_id') or task.get('campaign')
            choice_label = f"{campaign_id} - {task['patient_id']}"
            choice_value = f"{campaign_id}|{task['patient_id']}|{task['dataset_path']}"
            choices.append((choice_label, choice_value))
        
        status = f"You have {len(remaining)} remaining assigned tasks."
        return gr.update(choices=choices, value=None), status
    
    def load_selected_task_from_dataset(dataset_selection, user_id):
        """Load the selected task from dataset selection into the editor"""
        if not dataset_selection or not user_id:
            return "Please select a task from the table", "", False, ""
        
        try:
            # dataset_selection contains the selected row data
            # Format: ["🟡 Task 1", "Campaign Name", "Patient ID", "Status"]
            selected_row = dataset_selection
            
            # Extract campaign and patient info from the selected row
            campaign_id = selected_row[1]  # Campaign Name column
            patient_id = selected_row[2]   # Patient ID column
            
            # Get the dataset path from campaign manager
            if campaign_id not in crowdsourcing_manager.assignments:
                return f"❌ Campaign {campaign_id} not found", "", False, ""
            
            dataset_path = crowdsourcing_manager.assignments[campaign_id]['dataset_path']
            patient_path = os.path.join(dataset_path, patient_id)
            
            if not os.path.exists(patient_path):
                return f"❌ Patient data not found: {patient_path}", "", False, ""
            
            # Find the first available modality directory
            valid_modalities = ['flair', 't1', 't1c', 't2']
            modality_path = None
            
            for modality in valid_modalities:
                potential_path = os.path.join(patient_path, modality)
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    modality_path = potential_path
                    break
            
            if not modality_path:
                return f"❌ No valid modality found for patient {patient_id}", "", False, ""
            
            # Create task selection string for backward compatibility
            task_selection = f"{campaign_id}|{patient_id}|{dataset_path}"
            
            status = f"✅ Loaded patient {patient_id} from campaign {campaign_id} (modality: {os.path.basename(modality_path)})"
            return status, modality_path, True, task_selection
            
        except Exception as e:
            logger.error(f"Error loading task from dataset selection: {e}")
            return f"❌ Error loading task: {str(e)}", "", False, ""
    
    def load_selected_task_from_radio(selected_task_value, user_id):
        """Load the selected task from radio selection into the editor"""
        if not selected_task_value or not user_id:
            return create_error_message("Please select a task from the list"), "", False, ""
        
        try:
            campaign_id, patient_id, dataset_path = selected_task_value.split('|')
            patient_path = os.path.join(dataset_path, patient_id)
            
            if not os.path.exists(patient_path):
                return create_error_message(f"Patient data not found: {patient_path}"), "", False, ""
            
            # Find the first available modality directory
            valid_modalities = ['flair', 't1', 't1c', 't2']
            modality_path = None
            
            for modality in valid_modalities:
                potential_path = os.path.join(patient_path, modality)
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    modality_path = potential_path
                    break
            
            if not modality_path:
                return create_error_message(f"No valid modality found for patient {patient_id}"), "", False, ""
            
            status = create_info_message(f"✅ Loaded patient {patient_id} from campaign {campaign_id} (modality: {os.path.basename(modality_path)})")
            return status, modality_path, True, selected_task_value
            
        except Exception as e:
            logger.error(f"Error loading task from radio selection: {e}")
            return create_error_message(f"Error loading task: {str(e)}"), "", False, ""
    
    def load_selected_task(task_selection, user_id):
        """Load the selected task into the editor - Legacy dropdown support"""
        if not task_selection or not user_id:
            return "Please select a task", "", False
        
        try:
            campaign_id, patient_id, dataset_path = task_selection.split('|')
            patient_path = os.path.join(dataset_path, patient_id)
            
            if not os.path.exists(patient_path):
                return f"❌ Patient data not found: {patient_path}", "", False
            
            # Find the first available modality directory
            valid_modalities = ['flair', 't1', 't1c', 't2']
            modality_path = None
            
            for modality in valid_modalities:
                potential_path = os.path.join(patient_path, modality)
                if os.path.exists(potential_path) and os.path.isdir(potential_path):
                    modality_path = potential_path
                    break
            
            if not modality_path:
                return f"❌ No valid modality found for patient {patient_id}", "", False
            
            status = f"✅ Loaded patient {patient_id} from campaign {campaign_id} (modality: {os.path.basename(modality_path)})"
            return status, modality_path, True
            
        except Exception as e:
            logger.error(f"Error loading task: {e}")
            return f"❌ Error loading task: {str(e)}", "", False
    
    def submit_annotation_from_table(table_data, selected_rows, user_id):
        """Submit completed annotation from table selection"""
        if not selected_rows or not user_id or not table_data:
            return "Please select a task and ensure you're logged in", []
        
        try:
            # Get the selected row
            selected_row_idx = selected_rows[0]
            selected_task_data = table_data[selected_row_idx]
            campaign_id = selected_task_data[1]
            patient_id = selected_task_data[2]
            
            success = crowdsourcing_manager.mark_completed(campaign_id, user_id, patient_id)
            
            if success:
                # Update the table data to reflect the status change
                updated_table_data = []
                for i, row in enumerate(table_data):
                    if i == selected_row_idx:
                        updated_row = row.copy()
                        updated_row[3] = "Annotations Sent"  # Update status
                        updated_table_data.append(updated_row)
                    else:
                        updated_table_data.append(row)
                
                return f"✅ Annotation for patient {patient_id} submitted successfully!", updated_table_data
            else:
                return "❌ Failed to submit annotation", table_data
                
        except Exception as e:
            logger.error(f"Error submitting annotation from table: {e}")
            return f"❌ Error submitting annotation: {str(e)}", table_data
    
    def submit_annotation(task_selection, user_id):
        """Submit completed annotation - Legacy dropdown support"""
        if not task_selection or not user_id:
            return "Please select a task and ensure you're logged in"
        
        try:
            campaign_id, patient_id, _ = task_selection.split('|')
            success = crowdsourcing_manager.mark_completed(campaign_id, user_id, patient_id)
            
            if success:
                return f"✅ Annotation for patient {patient_id} submitted successfully!"
            else:
                return "❌ Failed to submit annotation"
                
        except Exception as e:
            logger.error(f"Error submitting annotation: {e}")
            return f"❌ Error submitting annotation: {str(e)}"
    
    with gr.Tab("Contribute") as tab:
        # Professional header with Next.js styling
        gr.HTML("""
        <div style='padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; border: 1px solid #6366f1; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
            <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 12px;'>
                <div style='width: 16px; height: 16px; background: #3b82f6; border-radius: 50%; flex-shrink: 0;'></div>
                <h1 style='color: white; font-weight: 700; font-size: 28px; margin: 0;'>
                    Expert Contribution Hub
                </h1>
            </div>
            <p style='color: #e0e7ff; font-size: 16px; margin: 0; line-height: 1.5;'>
                Select and work on your assigned annotation tasks with professional tools and workflows
            </p>
        </div>
        """)
        
        # User state (will be set by parent app)
        current_user_state = gr.State("")
        
        # Task selection section with modern table interface
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Your Assigned Tasks")
                
                refresh_tasks_button = gr.Button("🔄 Refresh Tasks", variant="secondary")
                
                task_status = gr.HTML(create_info_message("🔄 Loading your assignments..."))
                
                # Task selection radio for backend compatibility (hidden)
                task_selection_radio = gr.Radio(
                    label="Selected Task (Internal)",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False  # Hidden but used for backend processing
                )
                
                # Display table with dataframe for proper interaction
                assignments_table = gr.Dataframe(
                    label="Your Assignment Tasks",
                    headers=["Task #", "Campaign Name", "Patient ID", "Status"],
                    datatype=["html", "html", "html", "html"],  # Allow HTML for styling
                    value=[["Please log in or click 'Refresh Tasks'", "to load your assignments", "", ""]],
                    interactive=False,
                    wrap=True
                )
                
                with gr.Row():
                    load_status = gr.HTML("")  # Use HTML for better formatting
                
                # Hidden legacy dropdown for backward compatibility with Editor tab
                task_dropdown = gr.Dropdown(
                    label="Legacy Task Selection",
                    choices=[],
                    interactive=True,
                    visible=False  # Hidden but maintained for compatibility
                )
                
                # Hidden state for task management
                selected_dataset_path = gr.State("")
                crowdsourcing_mode = gr.State(False)
                selected_task_info = gr.State("")  # For app.py compatibility
        
        # Instructions with Next.js styling matching Editor tab
        with gr.Row():
            gr.HTML("""
            <div style='padding: 16px; background: linear-gradient(135deg, #374151 0%, #4b5563 100%); border-radius: 12px; border: 1px solid #6b7280; margin: 8px 0;'>
                <div style='display: flex; align-items: flex-start; gap: 12px; margin-bottom: 12px;'>
                    <div style='width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; margin-top: 4px;'></div>
                    <div style='color: #f9fafb; font-weight: 600; font-size: 14px;'>
                        Instructions:
                    </div>
                </div>
                <div style='color: #d1d5db; font-size: 13px; line-height: 1.6; margin-left: 24px;'>
                    <div style='margin-bottom: 6px;'><strong>1.</strong> Click on any pending task (🟡) in the table below to load it</div>
                    <div style='margin-bottom: 6px;'><strong>2.</strong> The Editor tab will open automatically with your selected patient data</div>
                    <div style='margin-bottom: 6px;'><strong>3.</strong> Complete your annotations using the tools in the Editor tab</div>
                    <div style='margin-bottom: 6px;'><strong>4.</strong> Submit your work using the 'Submit Annotation' button</div>
                    <div style='margin-bottom: 0;'><strong>5.</strong> Return here to see updated status and select your next task</div>
                </div>
                <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid #4b5563;'>
                    <div style='color: #9ca3af; font-size: 12px; line-height: 1.4;'>
                        <strong>Status Legend:</strong> 🟡 Pending (available for work) • 🟢 Annotations Sent (completed, not clickable)
                    </div>
                </div>
            </div>
            """)
        
        # Additional functions for app.py compatibility
        def get_assigned_tasks_dataset(user_id):
            """Get assigned tasks as dataset for app.py compatibility"""
            html_table, radio_choices, status = get_assigned_tasks_table(user_id)
            
            # Convert to simple table data for dataframe compatibility
            if not user_id:
                return [["No user logged in", "", "", "Please log in first"]], status
            
            # Get assignments and build simple table data
            all_assigned = crowdsourcing_manager.get_assigned_tasks(user_id)
            completed = crowdsourcing_manager.assignments
            
            if not all_assigned:
                return [["No assignments", "", "", "No tasks assigned"]], status
            
            table_data = []
            task_counter = 1
            
            for task in all_assigned:
                campaign_id = task.get('campaign_id') or task.get('campaign')
                patient_id = task['patient_id']
                
                # Check if this task is completed
                status_val = "Pending"
                if campaign_id in completed:
                    if user_id in completed[campaign_id].get('completed', {}):
                        if patient_id in completed[campaign_id]['completed'][user_id]:
                            status_val = "Annotations Sent"
                
                status_icon = "🟢" if status_val == "Annotations Sent" else "🟡"
                table_data.append([
                    f"{status_icon} Task {task_counter}",
                    campaign_id,
                    patient_id,
                    status_val
                ])
                task_counter += 1
            
            return table_data, status
        
        def load_next_assignment(user_id):
            """Load next assignment for app.py compatibility"""
            remaining = crowdsourcing_manager.get_remaining_assignments_for_user(user_id)
            if not remaining:
                return "No remaining assignments", "", False, ""
            
            # Load the first remaining task
            next_task = remaining[0]
            # Handle both 'campaign' and 'campaign_id' keys for compatibility
            campaign_id = next_task.get('campaign_id') or next_task.get('campaign')
            task_selection = f"{campaign_id}|{next_task['patient_id']}|{next_task['dataset_path']}"
            status, dataset_path, success = load_selected_task(task_selection, user_id)
            return status, dataset_path, success, task_selection
        
        # Event handlers
        def update_assignments_display(user_id):
            """Update both dataframe and radio components"""
            if not user_id:
                # Return empty data if no user
                return (
                    [["Please log in first", "", "", ""]],
                    gr.update(choices=[]),
                    create_info_message("🔑 Please log in to view your assignments")
                )
            
            try:
                table_data, radio_choices, status = get_assigned_tasks_table(user_id)
                
                # Return updates for all components
                return (
                    table_data,  # Dataframe expects raw data
                    gr.update(choices=radio_choices),  # Update radio choices
                    status  # Update status
                )
            except Exception as e:
                logger.error(f"Error updating assignments display: {e}")
                return (
                    [["Error loading tasks", str(e), "", ""]],
                    gr.update(choices=[]),
                    create_error_message(f"Error: {str(e)}")
                )
        
        # Auto-refresh when tab is selected (using tab select event)
        tab.select(
            fn=update_assignments_display,
            inputs=[current_user_state],
            outputs=[assignments_table, task_selection_radio, task_status]
        )
        
        refresh_tasks_button.click(
            fn=update_assignments_display,
            inputs=[current_user_state],
            outputs=[assignments_table, task_selection_radio, task_status]
        )
        
        # Also update legacy dropdown for backward compatibility
        refresh_tasks_button.click(
            fn=get_assigned_tasks,
            inputs=[current_user_state],
            outputs=[task_dropdown, gr.Textbox(visible=False)]  # Hidden status output
        )
        
        # Auto-refresh tasks when user state changes (when user logs in)
        current_user_state.change(
            fn=update_assignments_display,
            inputs=[current_user_state],
            outputs=[assignments_table, task_selection_radio, task_status]
        )
        
        # Also update legacy dropdown automatically
        current_user_state.change(
            fn=get_assigned_tasks,
            inputs=[current_user_state],
            outputs=[task_dropdown, gr.Textbox(visible=False)]
        )
        
        # Load task from dataframe selection (triggered by table row clicks)
        def handle_dataframe_selection(evt: gr.SelectData, user_id):
            """Handle dataframe row selection event"""
            if not evt or not hasattr(evt, 'index') or not user_id:
                return create_error_message("Please select a task from the table"), "", False, ""
            
            # Get the selected row data
            try:
                # Get tasks data to find the selected row
                table_data, radio_choices, _ = get_assigned_tasks_table(user_id)
                if not table_data or evt.index[0] >= len(table_data):
                    return create_error_message("Invalid selection"), "", False, ""
                
                selected_row = table_data[evt.index[0]]  # evt.index is a tuple (row, col) for dataframe
                
                # Check if the task is completed by looking at the radio choices which contain the raw status
                # The radio choices have the format: "🟢 Task 1: campaign - patient (Annotations Sent)"
                if evt.index[0] < len(radio_choices):
                    radio_label = radio_choices[evt.index[0]][0]  # Get the label part
                    
                    # Check if the task is completed (status is "Annotations Sent")
                    if "Annotations Sent" in radio_label:
                        return create_error_message("This task has already been completed and submitted. Please select a different task."), "", False, ""
                    
                    radio_value = radio_choices[evt.index[0]][1]  # Get the value part of (label, value) tuple
                    # Use the radio-based loading function which works correctly
                    return load_selected_task_from_radio(radio_value, user_id)
                else:
                    return create_error_message("Invalid task selection"), "", False, ""
                    
            except Exception as e:
                logger.error(f"Error handling dataframe selection: {e}")
                return create_error_message(f"Error: {str(e)}"), "", False, ""
        
        assignments_table.select(
            fn=handle_dataframe_selection,
            inputs=[current_user_state],
            outputs=[load_status, selected_dataset_path, crowdsourcing_mode, selected_task_info]
        )
        
        # Also update the hidden radio component when dataframe row is selected
        def update_radio_from_dataframe_selection(evt: gr.SelectData, user_id):
            """Update hidden radio component when dataframe row is selected"""
            if not evt or not hasattr(evt, 'index') or not user_id:
                return gr.update()
            
            try:
                # Get tasks data to find the corresponding radio value
                table_data, radio_choices, _ = get_assigned_tasks_table(user_id)
                if not table_data or evt.index[0] >= len(radio_choices):
                    return gr.update()
                
                # Get the radio value for the selected row
                radio_value = radio_choices[evt.index[0]][1]  # Get the value part of (label, value) tuple
                return gr.update(value=radio_value)
            except Exception as e:
                logger.error(f"Error updating radio from dataframe selection: {e}")
                return gr.update()
        
        assignments_table.select(
            fn=update_radio_from_dataframe_selection,
            inputs=[current_user_state],
            outputs=[task_selection_radio]
        )
        
        # Update legacy dropdown when dataframe selection changes
        def update_dropdown_from_dataframe_selection(evt: gr.SelectData, user_id):
            """Update legacy dropdown when dataframe row is selected"""
            if not evt or not hasattr(evt, 'index') or not user_id:
                return gr.update()
            
            try:
                # Get tasks data to find the corresponding radio value
                table_data, radio_choices, _ = get_assigned_tasks_table(user_id)
                if not table_data or evt.index[0] >= len(radio_choices):
                    return gr.update()
                
                # Get the radio value for the selected row
                radio_value = radio_choices[evt.index[0]][1]  # Get the value part of (label, value) tuple
                return gr.update(value=radio_value)
            except Exception as e:
                logger.error(f"Error updating dropdown from dataframe selection: {e}")
                return gr.update()
        
        assignments_table.select(
            fn=update_dropdown_from_dataframe_selection,
            inputs=[current_user_state],
            outputs=[task_dropdown]
        )
        
        # Function to clear info messages when leaving the tab
        def clear_info_messages():
            """Clear info messages when user leaves the Contribute tab"""
            return "", ""  # Clear both task_status and load_status
        
        # Function to refresh assignments table (for use by app.py after submission)
        def refresh_assignments_after_submission(user_id):
            """Refresh assignments table after submission - for app.py integration"""
            try:
                # Force reload from the assignments.json file to get latest completed status
                crowdsourcing_manager.load_assignments()  # Reload from file
                table_data, radio_choices, status = get_assigned_tasks_table(user_id)
                
                return (
                    table_data,  # Updated table data
                    gr.update(choices=radio_choices),  # Updated radio choices
                    status  # Updated status message
                )
            except Exception as e:
                logger.error(f"Error refreshing assignments after submission: {e}")
                return (
                    [["Error refreshing", str(e), "", ""]],
                    gr.update(choices=[]),
                    create_error_message(f"Error: {str(e)}")
                )
    
    return {
        'tab': tab,
        'current_user_state': current_user_state,
        'selected_dataset_path': selected_dataset_path,
        'crowdsourcing_mode': crowdsourcing_mode,
        'task_dropdown': task_dropdown,  # Legacy support
        'load_status': load_status,
        'selected_task_info': selected_task_info,  # For app.py compatibility
        'tasks_dataset': assignments_table,  # Updated to use the dataframe component
        'task_status': task_status,  # For app.py compatibility
        'assignments_table': assignments_table,  # New dataframe table component
        'task_selection_radio': task_selection_radio,  # New radio selection
        'get_assigned_tasks_dataset': get_assigned_tasks_dataset,  # Function for app.py
        'load_next_assignment': load_next_assignment,  # Function for app.py
        'refresh_assignments_after_submission': refresh_assignments_after_submission,  # New refresh function
        'clear_info_messages': clear_info_messages,  # Function to clear messages when leaving tab
    }
