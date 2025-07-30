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
    
    def scan_dataset_folder(dataset_path):
        """Scan the selected dataset folder and return styled HTML results"""
        if not dataset_path:
            return """
            <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 8px; border: 1px solid #ef4444; text-align: center;'>
                <div style='color: #dc2626; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>
                    ⚠️ Missing Dataset Path
                </div>
                <div style='color: #b91c1c; font-size: 13px;'>
                    Please provide a valid dataset directory path
                </div>
            </div>
            """, "", ""
        
        total_patients, patient_list = crowdsourcing_manager.scan_dataset(dataset_path)
        
        if total_patients == 0:
            return """
            <div style='padding: 16px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 8px; border: 1px solid #f59e0b; text-align: center;'>
                <div style='color: #d97706; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>
                    📂 No Valid Patients Found
                </div>
                <div style='color: #b45309; font-size: 13px;'>
                    The directory doesn't contain valid medical imaging data
                </div>
            </div>
            """, "", ""
        
        # Create styled success result
        patient_preview = ', '.join(patient_list[:5])
        if len(patient_list) > 5:
            patient_preview += f' and {len(patient_list) - 5} more...'
        
        result_html = f"""
        <div style='padding: 20px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border: 1px solid #10b981; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                <div style='width: 16px; height: 16px; background: #10b981; border-radius: 50%; flex-shrink: 0;'></div>
                <h3 style='color: #047857; font-weight: 600; font-size: 18px; margin: 0;'>
                    ✅ Dataset Analysis Complete
                </h3>
            </div>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;'>
                <div style='background: rgba(16, 185, 129, 0.1); padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;'>
                    <div style='color: #065f46; font-size: 24px; font-weight: 700; margin-bottom: 4px;'>{total_patients}</div>
                    <div style='color: #047857; font-size: 14px; font-weight: 500;'>Total Patients Found</div>
                </div>
                <div style='background: rgba(59, 130, 246, 0.1); padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                    <div style='color: #1e40af; font-size: 16px; font-weight: 600; margin-bottom: 4px;'>Ready for Assignment</div>
                    <div style='color: #2563eb; font-size: 14px;'>All patients validated</div>
                </div>
            </div>
            <div style='background: rgba(255, 255, 255, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);'>
                <div style='color: #047857; font-size: 13px; font-weight: 600; margin-bottom: 4px;'>Sample Patient IDs:</div>
                <div style='color: #065f46; font-size: 12px; font-family: monospace; line-height: 1.4;'>{patient_preview}</div>
            </div>
        </div>
        """
        
        return result_html, str(total_patients), ", ".join(patient_list)
    
    def create_campaign(campaign_name, dataset_path):
        """Create a new crowdsourcing campaign and return styled HTML status"""
        if not campaign_name or not dataset_path:
            error_html = """
            <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 8px; border: 1px solid #ef4444; text-align: center;'>
                <div style='color: #dc2626; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>
                    ❌ Missing Information
                </div>
                <div style='color: #b91c1c; font-size: 13px;'>
                    Please provide both campaign name and dataset path
                </div>
            </div>
            """
            return error_html, ""
        
        success = crowdsourcing_manager.create_campaign(campaign_name, dataset_path)
        if success:
            success_html = f"""
            <div style='padding: 20px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border: 1px solid #10b981; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
                    <div style='width: 16px; height: 16px; background: #10b981; border-radius: 50%; flex-shrink: 0;'></div>
                    <h3 style='color: #047857; font-weight: 600; font-size: 18px; margin: 0;'>
                        🎉 Campaign Created Successfully!
                    </h3>
                </div>
                <div style='background: rgba(255, 255, 255, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(16, 185, 129, 0.3);'>
                    <div style='color: #047857; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>Campaign Name:</div>
                    <div style='color: #065f46; font-size: 13px; font-family: monospace;'>{campaign_name}</div>
                </div>
                <div style='color: #047857; font-size: 13px; margin-top: 12px; text-align: center;'>
                    You can now assign patients to experts in the "Current Campaigns" tab
                </div>
            </div>
            """
            return success_html, refresh_campaigns()
        else:
            error_html = f"""
            <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 8px; border: 1px solid #ef4444; text-align: center;'>
                <div style='color: #dc2626; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>
                    ❌ Campaign Creation Failed
                </div>
                <div style='color: #b91c1c; font-size: 13px;'>
                    Failed to create campaign '{campaign_name}'. Please check if the dataset path is valid and contains patients.
                </div>
            </div>
            """
            return error_html, ""
    
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
        """Assign selected patients to an expert and return styled HTML status"""
        if not campaign_name or not selected_expert or not selected_patients:
            return """
            <div style='padding: 16px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 8px; border: 1px solid #f59e0b; text-align: center;'>
                <div style='color: #d97706; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>
                    ⚠️ Incomplete Selection
                </div>
                <div style='color: #b45309; font-size: 13px;'>
                    Please select campaign, expert, and at least one patient
                </div>
            </div>
            """, ""
        
        success = crowdsourcing_manager.assign_patients(campaign_name, selected_expert, selected_patients)
        
        if success:
            success_html = f"""
            <div style='padding: 20px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 12px; border: 1px solid #10b981; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 12px;'>
                    <div style='width: 16px; height: 16px; background: #10b981; border-radius: 50%; flex-shrink: 0;'></div>
                    <h3 style='color: #047857; font-weight: 600; font-size: 18px; margin: 0;'>
                        ✅ Assignment Successful!
                    </h3>
                </div>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px;'>
                    <div style='background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 8px; text-align: center;'>
                        <div style='color: #1e40af; font-size: 18px; font-weight: 600;'>{len(selected_patients)}</div>
                        <div style='color: #2563eb; font-size: 12px;'>Patients</div>
                    </div>
                    <div style='background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 8px; text-align: center;'>
                        <div style='color: #047857; font-size: 16px; font-weight: 600;'>Assigned</div>
                        <div style='color: #065f46; font-size: 12px;'>Successfully</div>
                    </div>
                    <div style='background: rgba(139, 92, 246, 0.1); padding: 12px; border-radius: 8px; text-align: center;'>
                        <div style='color: #7c3aed; font-size: 14px; font-weight: 600;'>Expert</div>
                        <div style='color: #6d28d9; font-size: 12px;'>{selected_expert}</div>
                    </div>
                </div>
            </div>
            """
            return success_html, refresh_campaigns()
        else:
            return """
            <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 8px; border: 1px solid #ef4444; text-align: center;'>
                <div style='color: #dc2626; font-size: 14px; font-weight: 500; margin-bottom: 4px;'>
                    ❌ Assignment Failed
                </div>
                <div style='color: #b91c1c; font-size: 13px;'>
                    Unable to assign patients. Please try again.
                </div>
            </div>
            """, ""
    
    gr.Markdown("Create and manage crowdsourcing campaigns for medical image annotation.")
    
    with gr.Tabs() as sub_tabs:
            # Default tab: Current Campaigns
            with gr.Tab("Current Campaigns") as campaigns_tab:
                gr.Markdown("## Current Campaigns")
                campaigns_display = gr.HTML(value=refresh_campaigns())
                refresh_button = gr.Button("Refresh Campaigns", variant="secondary")
                
                # Task Assignment Section with modern Next.js styling
                gr.HTML("""
                <div style='padding: 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; border: 1px solid #6366f1; margin: 24px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 12px;'>
                        <div style='width: 16px; height: 16px; background: #3b82f6; border-radius: 50%; flex-shrink: 0;'></div>
                        <h2 style='color: white; font-weight: 700; font-size: 24px; margin: 0;'>
                            👥 Task Assignment Hub
                        </h2>
                    </div>
                    <p style='color: #e0e7ff; font-size: 16px; margin: 0; line-height: 1.5;'>
                        Efficiently assign patients to expert annotators with intelligent task distribution
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Campaign Selection Card
                        gr.HTML("""
                        <div style='padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                                <div style='width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #1e293b; font-weight: 600; font-size: 18px; margin: 0;'>
                                    Campaign & Expert Selection
                                </h3>
                            </div>
                            <p style='color: #64748b; font-size: 14px; margin: 0; line-height: 1.5;'>
                                Choose the campaign and expert annotator for task assignment
                            </p>
                        </div>
                        """)
                        
                        campaign_dropdown = gr.Dropdown(
                            label="📋 Select Campaign",
                            choices=get_campaign_choices(),
                            value=None,
                            interactive=True,
                            elem_classes=["modern-dropdown"]
                        )
                        
                        expert_dropdown = gr.Dropdown(
                            label="👨‍⚕️ Select Expert Annotator",
                            choices=get_all_experts(),
                            value=None,
                            interactive=True,
                            elem_classes=["modern-dropdown"]
                        )
                                              
                        with gr.Row():
                            assign_button = gr.Button(
                                "✅ Assign Selected Patients", 
                                variant="primary", 
                                size="lg",
                                elem_classes=["assign-btn"]
                            )
                            refresh_assignment_button = gr.Button(
                                "🔄 Refresh Data", 
                                variant="secondary",
                                elem_classes=["refresh-btn"]
                            )
                        
                        # Assignment Status Display (shown after assignment)
                        assign_status = gr.HTML("", visible=False)
                    
                    with gr.Column(scale=1):
                        # Patient Selection Card
                        gr.HTML("""
                        <div style='padding: 20px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 12px; border: 1px solid #10b981; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                                <div style='width: 12px; height: 12px; background: #10b981; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #065f46; font-weight: 600; font-size: 18px; margin: 0;'>
                                    🏥 Patient Assignment Pool
                                </h3>
                            </div>
                            <p style='color: #047857; font-size: 14px; margin: 0; line-height: 1.5;'>
                                Select patients to assign from the available unassigned pool
                            </p>
                        </div>
                        """)
                        
                        patients_checklist = gr.CheckboxGroup(
                            label="Available Patients for Assignment",
                            choices=[],
                            interactive=True,
                            elem_classes=["patient-checklist"]
                        )
            
            # Second tab: Create New Campaign with modern Next.js styling
            with gr.Tab("Create Campaign") as create_tab:
                # Modern Header
                gr.HTML("""
                <div style='padding: 24px; background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); border-radius: 16px; border: 1px solid #0891b2; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 12px;'>
                        <div style='width: 16px; height: 16px; background: #0284c7; border-radius: 50%; flex-shrink: 0;'></div>
                        <h2 style='color: white; font-weight: 700; font-size: 24px; margin: 0;'>
                            🚀 Create New Campaign
                        </h2>
                    </div>
                    <p style='color: #cffafe; font-size: 16px; margin: 0; line-height: 1.5;'>
                        Set up a new medical image annotation campaign
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # Campaign Details Card
                        gr.HTML("""
                        <div style='padding: 20px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border-radius: 12px; border: 1px solid #0ea5e9; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                                <div style='width: 12px; height: 12px; background: #0ea5e9; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #0c4a6e; font-weight: 600; font-size: 18px; margin: 0;'>
                                    📝 Campaign Configuration
                                </h3>
                            </div>
                            <p style='color: #0369a1; font-size: 14px; margin: 0; line-height: 1.5;'>
                                Define your campaign name and specify the dataset location
                            </p>
                        </div>
                        """)
                        
                        campaign_name_input = gr.Textbox(
                            label="🏷️ Campaign Name",
                            placeholder="Enter a descriptive campaign name (e.g., 'Brain MRI Segmentation Q1 2025')",
                            interactive=True,
                            elem_classes=["modern-textbox"]
                        )
                        
                        dataset_path_input = gr.Textbox(
                            label="📁 Dataset Path",
                            placeholder="Enter the full path to your dataset directory",
                            interactive=True,
                            elem_classes=["modern-textbox"]
                        )
                                              
                        with gr.Row():
                            scan_button = gr.Button(
                                "🔍 Analyze Dataset", 
                                variant="secondary",
                                size="lg",
                                elem_classes=["scan-btn"]
                            )
                            create_button = gr.Button(
                                "✨ Create Campaign", 
                                variant="primary",
                                size="lg",
                                elem_classes=["create-btn"]
                            )
                    
                    with gr.Column(scale=1):
                        # Dataset Analysis Card
                        gr.HTML("""
                        <div style='padding: 20px; background: linear-gradient(135deg, #f7fee7 0%, #ecfccb 100%); border-radius: 12px; border: 1px solid #65a30d; margin-bottom: 16px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                                <div style='width: 12px; height: 12px; background: #65a30d; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #365314; font-weight: 600; font-size: 18px; margin: 0;'>
                                    📊 Dataset Analysis
                                </h3>
                            </div>
                        </div>
                        """)
                        
                        scan_results = gr.HTML(
                            value="""
                            <div style='padding: 16px; background: #f8fafc; border-radius: 8px; border: 2px dashed #cbd5e1; text-align: center;'>
                                <div style='color: #64748b; font-size: 14px; margin-bottom: 8px;'>
                                    📈 Dataset Analysis Results
                                </div>
                                <div style='color: #94a3b8; font-size: 13px;'>
                                    Click "Analyze Dataset" to scan your data directory
                                </div>
                            </div>
                            """,
                            visible=True
                        )
                        
                        # Status Display Card
                        gr.HTML("""
                        <div style='padding: 20px; background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); border-radius: 12px; border: 1px solid #eab308; margin-top: 16px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                                <div style='width: 12px; height: 12px; background: #eab308; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #713f12; font-weight: 600; font-size: 18px; margin: 0;'>
                                    📋 Creation Status
                                </h3>
                            </div>
                        </div>
                        """)
                        
                        create_status = gr.HTML(
                            value="""
                            <div style='padding: 16px; background: #f8fafc; border-radius: 8px; border: 2px dashed #cbd5e1; text-align: center;'>
                                <div style='color: #64748b; font-size: 14px; margin-bottom: 8px;'>
                                    🎯 Campaign Status
                                </div>
                                <div style='color: #94a3b8; font-size: 13px;'>
                                    Ready to create your new campaign
                                </div>
                            </div>
                            """
                        )
                        
                        # Hidden components to store scan results
                        total_patients_state = gr.State("")
                        patient_list_state = gr.State("")
    
    # Event handlers
    scan_button.click(
        fn=scan_dataset_folder,
        inputs=[dataset_path_input],
        outputs=[scan_results, total_patients_state, patient_list_state]
    )
    
    create_button.click(
        fn=create_campaign,
        inputs=[campaign_name_input, dataset_path_input],
        outputs=[create_status, campaigns_display]
    ).then(
        fn=lambda: gr.update(choices=get_campaign_choices()),
        outputs=[campaign_dropdown]
    ).then(
        fn=lambda name, path: ("", ""),
        inputs=[campaign_name_input, dataset_path_input],
        outputs=[campaign_name_input, dataset_path_input]
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
        fn=lambda: gr.update(visible=True),
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
