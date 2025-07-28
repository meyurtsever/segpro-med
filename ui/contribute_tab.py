"""
Contribute tab for expert users to work on assigned annotation tasks
"""

import gradio as gr
import logging
import os
from crowdsourcing.campaign_manager import CrowdsourcingManager

logger = logging.getLogger(__name__)

def create_contribute_tab():
    """Create the contribute tab for experts"""
    
    crowdsourcing_manager = CrowdsourcingManager()
    
    def get_assigned_tasks(user_id):
        """Get assigned tasks for the current expert"""
        if not user_id:
            return [], "Please log in first"
        
        assigned = crowdsourcing_manager.get_assigned_patients(user_id)
        
        if not assigned:
            return [], "No tasks assigned to you yet."
        
        # Create choices for dropdown
        choices = []
        for task in assigned:
            choice_label = f"{task['campaign_name']} - {task['patient_id']}"
            choice_value = f"{task['campaign_id']}|{task['patient_id']}|{task['dataset_path']}"
            choices.append((choice_label, choice_value))
        
        status = f"You have {len(assigned)} assigned tasks."
        return choices, status
    
    def load_selected_task(task_selection, user_id):
        """Load the selected task into the editor"""
        if not task_selection or not user_id:
            return "Please select a task", "", False
        
        try:
            campaign_id, patient_id, dataset_path = task_selection.split('|')
            patient_path = os.path.join(dataset_path, patient_id)
            
            if not os.path.exists(patient_path):
                return f"❌ Patient data not found: {patient_path}", "", False
            
            status = f"✅ Loaded patient {patient_id} from campaign {campaign_id}"
            return status, patient_path, True
            
        except Exception as e:
            logger.error(f"Error loading task: {e}")
            return f"❌ Error loading task: {str(e)}", "", False
    
    def submit_annotation(task_selection, user_id):
        """Submit completed annotation"""
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
        gr.Markdown("# Contribute to Annotation Tasks")
        gr.Markdown("Select and work on your assigned annotation tasks.")
        
        # User state (will be set by parent app)
        current_user_state = gr.State("")
        
        # Task selection section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Your Assigned Tasks")
                
                refresh_tasks_button = gr.Button("Refresh Tasks", variant="secondary")
                
                task_status = gr.Markdown("Click 'Refresh Tasks' to load your assignments.")
                
                task_dropdown = gr.Dropdown(
                    label="Select Task",
                    choices=[],
                    interactive=True
                )
                
                load_task_button = gr.Button("Load Task in Editor", variant="primary")
                load_status = gr.Markdown("")
                
                # Hidden state for task management
                selected_dataset_path = gr.State("")
                crowdsourcing_mode = gr.State(False)
        
        # Crowdsourcing-specific controls
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Annotation Controls")
                gr.Markdown("Use the Editor tab to annotate the loaded dataset, then return here to submit.")
                
                submit_button = gr.Button("Submit Annotation", variant="primary")
                submit_status = gr.Markdown("")
                
                review_button = gr.Button("Send for Review", variant="secondary")
                review_status = gr.Markdown("")
        
        # Instructions
        with gr.Row():
            gr.Markdown("""
            ## Instructions
            1. **Refresh Tasks**: Click to load your assigned annotation tasks
            2. **Select Task**: Choose a patient from your assigned tasks
            3. **Load Task**: Load the patient data into the Editor tab
            4. **Annotate**: Switch to the Editor tab to perform annotation
            5. **Submit**: Return here and click 'Submit Annotation' when complete
            """)
        
        # Event handlers
        refresh_tasks_button.click(
            fn=get_assigned_tasks,
            inputs=[current_user_state],
            outputs=[task_dropdown, task_status]
        )
        
        load_task_button.click(
            fn=load_selected_task,
            inputs=[task_dropdown, current_user_state],
            outputs=[load_status, selected_dataset_path, crowdsourcing_mode]
        )
        
        submit_button.click(
            fn=submit_annotation,
            inputs=[task_dropdown, current_user_state],
            outputs=[submit_status]
        )
        
        review_button.click(
            fn=lambda: "Review functionality will be implemented in future versions",
            outputs=[review_status]
        )
    
    return {
        'tab': tab,
        'current_user_state': current_user_state,
        'selected_dataset_path': selected_dataset_path,
        'crowdsourcing_mode': crowdsourcing_mode,
        'task_dropdown': task_dropdown,
        'load_status': load_status
    }
