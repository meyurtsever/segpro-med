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
        
        success = crowdsourcing_manager.create_campaign(campaign_name, dataset_path)
        if success:
            return f"✅ Campaign '{campaign_name}' created successfully", refresh_campaigns()
        else:
            return f"❌ Failed to create campaign '{campaign_name}'. Check if dataset path is valid and contains patients.", ""
    
    def refresh_campaigns():
        """Refresh the campaigns list"""
        campaigns_html = ""
        
        for campaign_name, campaign_data in crowdsourcing_manager.assignments.items():
            progress = crowdsourcing_manager.get_campaign_progress(campaign_name)
            
            if progress:
                progress_percentage = (progress['completed'] / max(progress['total_patients'], 1)) * 100
                
                campaigns_html += f"""
                <div style="border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 20px; margin: 12px 0; background: var(--background-fill-secondary); box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: box-shadow 0.2s ease;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border-color-secondary);">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, var(--color-accent), var(--color-accent-soft)); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px;">📊</div>
                            <div>
                                <h3 style="color: var(--body-text-color); margin: 0; font-size: 20px; font-weight: 600; line-height: 1.2;">{campaign_data['name']}</h3>
                                <div style="color: var(--body-text-color); opacity: 0.6; font-size: 13px; margin-top: 2px;">Medical Image Annotation Campaign</div>
                            </div>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <span style="background: var(--color-accent-soft); color: var(--color-accent); padding: 6px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Active</span>
                            <span style="background: var(--background-fill-primary); color: var(--body-text-color); padding: 6px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; border: 1px solid var(--border-color-secondary);">Campaign</span>
                        </div>
                    </div>
                    <div style="color: var(--body-text-color); margin-bottom: 16px;">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
                            <div style="background: var(--background-fill-primary); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--color-accent);">
                                <div style="font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">Campaign ID</div>
                                <div style="font-weight: 500; font-family: monospace;">{campaign_name}</div>
                            </div>
                            <div style="background: var(--background-fill-primary); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #3B82F6;">
                                <div style="font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">Dataset Path</div>
                                <div style="font-weight: 500; font-size: 13px; word-break: break-all;">{campaign_data['dataset_path']}</div>
                            </div>
                            <div style="background: var(--background-fill-primary); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #10B981;">
                                <div style="font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px;">Created</div>
                                <div style="font-weight: 500;">{campaign_data.get('created_at', 'Unknown')}</div>
                            </div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 16px 0;">
                        <div style="color: var(--body-text-color); background: var(--background-fill-primary); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color-secondary); text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="font-size: 18px; font-weight: 600; color: var(--color-accent);">{progress['total_patients']}</div>
                            <div style="font-size: 12px; opacity: 0.8; margin-top: 2px;">Total</div>
                        </div>
                        <div style="color: var(--body-text-color); background: var(--background-fill-primary); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color-secondary); text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="font-size: 18px; font-weight: 600; color: #3B82F6;">{progress['assigned_patients']}</div>
                            <div style="font-size: 12px; opacity: 0.8; margin-top: 2px;">Assigned</div>
                        </div>
                        <div style="color: var(--body-text-color); background: var(--background-fill-primary); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color-secondary); text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="font-size: 18px; font-weight: 600; color: #10B981;">{progress['completed']}</div>
                            <div style="font-size: 12px; opacity: 0.8; margin-top: 2px;">Completed</div>
                        </div>
                        <div style="color: var(--body-text-color); background: var(--background-fill-primary); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color-secondary); text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="font-size: 18px; font-weight: 600; color: #8B5CF6;">{progress['reviewed']}</div>
                            <div style="font-size: 12px; opacity: 0.8; margin-top: 2px;">Reviewed</div>
                        </div>
                    </div>
                    <div style="margin: 16px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="color: var(--body-text-color); font-weight: 500;">Progress</span>
                            <span style="color: var(--body-text-color); font-weight: 600; font-size: 14px;">{progress_percentage:.1f}%</span>
                        </div>
                        <div style="background: var(--background-fill-primary); border-radius: 8px; height: 8px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="background: linear-gradient(90deg, var(--color-accent), var(--color-accent-soft)); height: 100%; border-radius: 8px; width: {progress_percentage}%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                </div>
                """
        
        if not campaigns_html:
            campaigns_html = '<p style="color: var(--body-text-color); text-align: center; padding: 20px;">No campaigns created yet. Create your first campaign using the "Create Campaign" tab.</p>'
        
        return campaigns_html
    
    def get_campaign_choices():
        """Get list of available campaigns for dropdown"""
        campaigns = list(crowdsourcing_manager.assignments.keys())
        return campaigns if campaigns else []
    
    def get_all_experts():
        """Get all experts for initial dropdown population"""
        experts = auth_manager.get_experts()
        logger.info(f"Loading all experts: {experts}")
        return experts
    
    def get_campaign_assignment_details(campaign_name):
        """Get assignment details for selected campaign"""
        if not campaign_name or campaign_name not in crowdsourcing_manager.assignments:
            return gr.update(choices=[], value=None), gr.update(choices=[], value=[])
        
        experts = auth_manager.get_experts()
        unassigned_patients = crowdsourcing_manager.get_unassigned_patients(campaign_name)
        
        logger.info(f"Found {len(experts)} experts: {experts}")
        logger.info(f"Found {len(unassigned_patients)} unassigned patients for campaign {campaign_name}")
        
        return gr.update(choices=experts, value=None), gr.update(choices=unassigned_patients, value=[])
    
    def assign_patients(campaign_name, selected_expert, selected_patients):
        """Assign selected patients to an expert"""
        if not campaign_name or not selected_expert or not selected_patients:
            return "Please select campaign, expert, and patients", ""
        
        success = crowdsourcing_manager.assign_patients(campaign_name, selected_expert, selected_patients)
        
        if success:
            return f"✅ Assigned {len(selected_patients)} patients to {selected_expert}", refresh_campaigns()
        else:
            return "❌ Failed to assign patients", ""
    
    gr.Markdown("Create and manage crowdsourcing campaigns for medical image annotation.")
    
    with gr.Tabs() as sub_tabs:
            # Default tab: Current Campaigns
            with gr.Tab("Current Campaigns") as campaigns_tab:
                gr.Markdown("## Current Campaigns")
                campaigns_display = gr.HTML(value=refresh_campaigns())
                refresh_button = gr.Button("Refresh Campaigns", variant="secondary")
                
                # Task Assignment Section
                gr.Markdown("## 👥 Assign Tasks")
                
                with gr.Group():
                    gr.Markdown("### Select Campaign and Assign Patients to Experts")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            campaign_dropdown = gr.Dropdown(
                                label="📋 Select Campaign",
                                choices=get_campaign_choices(),
                                value=None,
                                interactive=True
                            )
                            
                            expert_dropdown = gr.Dropdown(
                                label="👨‍⚕️ Select Expert",
                                choices=get_all_experts(),
                                value=None,
                                interactive=True
                            )
                        
                        with gr.Column(scale=1):
                            patients_checklist = gr.CheckboxGroup(
                                label="🏥 Select Patients to Assign",
                                choices=[],
                                interactive=True
                            )
                            
                            with gr.Row():
                                assign_button = gr.Button("✅ Assign Selected Patients", variant="primary", scale=2)
                                refresh_assignment_button = gr.Button("🔄 Refresh", variant="secondary", scale=1)
                    
                    assign_status = gr.Markdown("", visible=False)
            
            # Second tab: Create New Campaign
            with gr.Tab("Create Campaign") as create_tab:
                gr.Markdown("## Create New Campaign")
                
                with gr.Row():
                    with gr.Column(scale=1):
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
                    
                    with gr.Column(scale=1):
                        scan_button = gr.Button("Scan Dataset", variant="secondary")
                        scan_results = gr.Markdown("", visible=False)
                        
                        # Hidden components to store scan results
                        total_patients_state = gr.State("")
                        patient_list_state = gr.State("")
                        
                        create_button = gr.Button("Create Campaign", variant="primary")
                        create_status = gr.Markdown("")
    
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
    ).then(
        fn=lambda: gr.update(choices=get_campaign_choices()),
        outputs=[campaign_dropdown]
    )
    
    campaign_dropdown.change(
        fn=get_campaign_assignment_details,
        inputs=[campaign_dropdown],
        outputs=[expert_dropdown, patients_checklist]
    )
    
    assign_button.click(
        fn=assign_patients,
        inputs=[campaign_dropdown, expert_dropdown, patients_checklist],
        outputs=[assign_status, campaigns_display]
    ).then(
        fn=lambda status: gr.update(value=status, visible=True),
        inputs=[assign_status],
        outputs=[assign_status]
    ).then(
        fn=get_campaign_assignment_details,
        inputs=[campaign_dropdown],
        outputs=[expert_dropdown, patients_checklist]
    )
    
    refresh_button.click(
        fn=lambda: refresh_campaigns(),
        outputs=[campaigns_display]
    )
    
    refresh_assignment_button.click(
        fn=lambda: [gr.update(choices=get_campaign_choices()), gr.update(choices=get_all_experts())],
        outputs=[campaign_dropdown, expert_dropdown]
    )
    
    return {
        'sub_tabs': sub_tabs,
        'campaigns_tab': campaigns_tab,
        'create_tab': create_tab,
        'campaign_name_input': campaign_name_input,
        'dataset_path_input': dataset_path_input,
        'campaigns_display': campaigns_display,
        'scan_button': scan_button,
        'create_campaign_button': create_button,
        'refresh_button': refresh_button,
        'campaign_dropdown': campaign_dropdown,
        'expert_dropdown': expert_dropdown,
        'patients_checklist': patients_checklist,
        'assign_button': assign_button,
        'refresh_assignment_button': refresh_assignment_button
    }
