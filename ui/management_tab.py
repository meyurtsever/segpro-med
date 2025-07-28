"""
Management tab for admin users to create and manage crowdsourcing campaigns
"""

import gradio as gr
import logging
import os
from crowdsourcing.campaign_manager import CrowdsourcingManager
from auth.auth_manager import AuthManager

logger = logging.getLogger(__name__)

def create_management_tab():
    """Create the management tab for admins"""
    
    crowdsourcing_manager = CrowdsourcingManager()
    auth_manager = AuthManager()
    
    def scan_dataset_folder(dataset_path):
        """Scan the selected dataset folder"""
        if not dataset_path:
            return "Please select a dataset path", "", ""
        
        total_patients, patient_list = crowdsourcing_manager.scan_dataset(dataset_path)
        
        if total_patients == 0:
            return "No valid patients found in the selected directory", "", ""
        
        summary = f"**Dataset Analysis:**\n- Total Patients: {total_patients}\n- Patient IDs: {', '.join(patient_list[:10])}"
        if len(patient_list) > 10:
            summary += f" and {len(patient_list) - 10} more..."
        
        return summary, str(total_patients), ", ".join(patient_list)
    
    def create_campaign(campaign_name, dataset_path):
        """Create a new crowdsourcing campaign"""
        if not campaign_name or not dataset_path:
            return "Please provide both campaign name and dataset path", ""
        
        campaign_id = crowdsourcing_manager.create_campaign(campaign_name, dataset_path)
        return f"✅ Campaign '{campaign_name}' created successfully (ID: {campaign_id})", refresh_campaigns()
    
    def refresh_campaigns():
        """Refresh the campaigns list"""
        campaigns_html = ""
        
        for campaign_id, campaign_data in crowdsourcing_manager.assignments.items():
            progress = crowdsourcing_manager.get_campaign_progress(campaign_id)
            
            if progress:
                progress_percentage = (progress['completed'] / max(progress['total'], 1)) * 100
                
                campaigns_html += f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin: 8px 0; background: #f9f9f9;">
                    <h3>{campaign_data['name']}</h3>
                    <p><strong>Campaign ID:</strong> {campaign_id}</p>
                    <p><strong>Dataset Path:</strong> {campaign_data['dataset_path']}</p>
                    <p><strong>Created:</strong> {campaign_data.get('created_at', 'Unknown')}</p>
                    <div style="display: flex; gap: 20px; margin: 10px 0;">
                        <span><strong>Total:</strong> {progress['total']}</span>
                        <span><strong>Assigned:</strong> {progress['assigned']}</span>
                        <span><strong>Completed:</strong> {progress['completed']}</span>
                        <span><strong>Reviewed:</strong> {progress['reviewed']}</span>
                    </div>
                    <div style="background: #e0e0e0; border-radius: 4px; height: 20px; margin: 10px 0;">
                        <div style="background: #4CAF50; height: 100%; border-radius: 4px; width: {progress_percentage}%;"></div>
                    </div>
                    <p><strong>Progress:</strong> {progress_percentage:.1f}% complete</p>
                </div>
                """
        
        if not campaigns_html:
            campaigns_html = "<p>No campaigns created yet.</p>"
        
        return campaigns_html
    
    def get_campaign_details(campaign_id):
        """Get detailed information about a campaign for assignment"""
        if not campaign_id or campaign_id not in crowdsourcing_manager.assignments:
            return "", [], []
        
        campaign = crowdsourcing_manager.assignments[campaign_id]
        experts = auth_manager.get_experts()
        unassigned_patients = crowdsourcing_manager.get_unassigned_patients(campaign_id)
        
        expert_choices = list(experts.keys())
        
        details = f"""
        **Campaign:** {campaign['name']}  
        **Total Patients:** {campaign['total_patients']}  
        **Unassigned Patients:** {len(unassigned_patients)}
        """
        
        return details, expert_choices, unassigned_patients
    
    def assign_patients(campaign_id, selected_expert, selected_patients):
        """Assign selected patients to an expert"""
        if not campaign_id or not selected_expert or not selected_patients:
            return "Please select campaign, expert, and patients", ""
        
        success = crowdsourcing_manager.assign_patients(campaign_id, selected_expert, selected_patients)
        
        if success:
            return f"✅ Assigned {len(selected_patients)} patients to {selected_expert}", refresh_campaigns()
        else:
            return "❌ Failed to assign patients", ""
    
    with gr.Tab("Management") as tab:
        gr.Markdown("# Campaign Management")
        gr.Markdown("Create and manage crowdsourcing campaigns for medical image annotation.")
        
        with gr.Row():
            # Left column: Create new campaign
            with gr.Column(scale=1):
                gr.Markdown("## Create New Campaign")
                
                campaign_name_input = gr.Textbox(
                    label="Campaign Name",
                    placeholder="Enter campaign name",
                    interactive=True
                )
                
                dataset_path_input = gr.Textbox(
                    label="Dataset Path",
                    placeholder="Enter path to dataset directory",
                    interactive=True
                )
                
                scan_button = gr.Button("Scan Dataset", variant="secondary")
                scan_results = gr.Markdown("", visible=False)
                
                # Hidden components to store scan results
                total_patients_state = gr.State("")
                patient_list_state = gr.State("")
                
                create_button = gr.Button("Create Campaign", variant="primary")
                create_status = gr.Markdown("")
            
            # Right column: Campaign management
            with gr.Column(scale=1):
                gr.Markdown("## Assign Tasks")
                
                campaign_id_input = gr.Textbox(
                    label="Campaign ID",
                    placeholder="Enter campaign ID for assignment",
                    interactive=True
                )
                
                get_details_button = gr.Button("Get Campaign Details", variant="secondary")
                campaign_details = gr.Markdown("")
                
                expert_dropdown = gr.Dropdown(
                    label="Select Expert",
                    choices=[],
                    interactive=True
                )
                
                patients_checklist = gr.CheckboxGroup(
                    label="Select Patients to Assign",
                    choices=[],
                    interactive=True
                )
                
                assign_button = gr.Button("Assign Patients", variant="primary")
                assign_status = gr.Markdown("")
        
        # Campaigns overview
        gr.Markdown("## Current Campaigns")
        campaigns_display = gr.HTML(value=refresh_campaigns())
        
        refresh_button = gr.Button("Refresh Campaigns", variant="secondary")
        
        # Event handlers
        scan_button.click(
            fn=scan_dataset_folder,
            inputs=[dataset_path_input],
            outputs=[scan_results, total_patients_state, patient_list_state]
        ).then(
            fn=lambda results: gr.update(value=results, visible=True),
            inputs=[scan_results],
            outputs=[scan_results]
        )
        
        create_button.click(
            fn=create_campaign,
            inputs=[campaign_name_input, dataset_path_input],
            outputs=[create_status, campaigns_display]
        )
        
        get_details_button.click(
            fn=get_campaign_details,
            inputs=[campaign_id_input],
            outputs=[campaign_details, expert_dropdown, patients_checklist]
        )
        
        assign_button.click(
            fn=assign_patients,
            inputs=[campaign_id_input, expert_dropdown, patients_checklist],
            outputs=[assign_status, campaigns_display]
        )
        
        refresh_button.click(
            fn=lambda: refresh_campaigns(),
            outputs=[campaigns_display]
        )
    
    return {
        'tab': tab,
        'campaign_name_input': campaign_name_input,
        'dataset_path_input': dataset_path_input,
        'campaigns_display': campaigns_display
    }
