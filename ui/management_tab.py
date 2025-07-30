"""
Management tab for admin users to create and manage crowdsourcing campaigns
"""

import gradio as gr
from gradio import SelectData
import logging
import os
import json
import pandas as pd
from datetime import datetime
from crowdsourcing.campaign_manager import CrowdsourcingManager
from auth.auth_manager import AuthManager
from gradio_image_annotation import image_annotator

logger = logging.getLogger(__name__)

# Global state for review interface navigation
current_review_state = None

def create_management_tab():
    """Create the management tab for admins"""
    
    def shorten_campaign_name(campaign_name, max_words=2):
        """Shorten campaign name to first N words + '...' if longer"""
        if not campaign_name:
            return campaign_name
        
        words = campaign_name.split()
        if len(words) <= max_words:
            return campaign_name
        
        return ' '.join(words[:max_words]) + '...'
    
    crowdsourcing_manager = CrowdsourcingManager()
    auth_manager = AuthManager()
    
    def get_campaign_choices():
        """Get list of available campaigns for dropdown"""
        # Force reload assignments from file to ensure we have the latest data
        crowdsourcing_manager.load_assignments()
        
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
    
    def get_pending_approvals():
        """Get all completed but not reviewed submissions for approval"""
        import pandas as pd
        
        # Force reload assignments from file to ensure we have the latest data
        crowdsourcing_manager.load_assignments()
        
        pending_records = []
        
        for campaign_name, campaign_data in crowdsourcing_manager.assignments.items():
            completed = campaign_data.get('completed', {})
            reviewed = campaign_data.get('reviewed', {})
            
            for expert_id, patient_list in completed.items():
                # patient_list is a list of patient_id strings
                reviewed_patients = []
                if expert_id in reviewed:
                    # Extract patient_ids from reviewed items (they might be objects with patient_id keys)
                    for item in reviewed[expert_id]:
                        if isinstance(item, dict) and 'patient_id' in item:
                            reviewed_patients.append(item['patient_id'])
                        elif isinstance(item, str):
                            reviewed_patients.append(item)
                
                for patient_id in patient_list:
                    # Check if this submission has not been reviewed yet
                    if patient_id not in reviewed_patients:
                        pending_records.append({
                            'Campaign': shorten_campaign_name(campaign_name),
                            'Expert': expert_id,
                            'Patient ID': patient_id,
                            'Status': 'Pending Review',
                            'Full_Campaign': campaign_name  # Store full name for processing
                        })
        
        if pending_records:
            df = pd.DataFrame(pending_records)
            # Only show the display columns, keep Full_Campaign for internal use
            display_df = df[['Campaign', 'Expert', 'Patient ID', 'Status']].copy()
            return display_df, len(pending_records)
        else:
            return pd.DataFrame(columns=['Campaign', 'Expert', 'Patient ID', 'Status']), 0
    
    def create_success_message(title, subtitle="", icon="✅"):
        """Create a styled success message with Next.js design"""
        return f"""
        <div style='padding: 16px; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 10px; border: 1px solid #10b981; margin: 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <div style='width: 14px; height: 14px; background: #10b981; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;'>{icon}</div>
                <div>
                    <div style='color: #047857; font-weight: 600; font-size: 15px; margin-bottom: 2px;'>{title}</div>
                    {f"<div style='color: #065f46; font-size: 13px; opacity: 0.8;'>{subtitle}</div>" if subtitle else ""}
                </div>
            </div>
        </div>
        """
    
    def create_info_message(title, subtitle="", icon="ℹ️"):
        """Create a styled info message with Next.js design"""
        return f"""
        <div style='padding: 16px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 10px; border: 1px solid #3b82f6; margin: 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);'>
            <div style='display: flex; align-items: center; gap: 10px;'>
                <div style='width: 14px; height: 14px; background: #3b82f6; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;'>{icon}</div>
                <div>
                    <div style='color: #1e40af; font-weight: 600; font-size: 15px; margin-bottom: 2px;'>{title}</div>
                    {f"<div style='color: #1e3a8a; font-size: 13px; opacity: 0.8;'>{subtitle}</div>" if subtitle else ""}
                </div>
            </div>
        </div>
        """
    
    def get_full_campaign_name(shortened_name, expert_id, patient_id):
        """Get the full campaign name from shortened display name"""
        # Force reload assignments from file to ensure we have the latest data
        crowdsourcing_manager.load_assignments()
        
        for campaign_name, campaign_data in crowdsourcing_manager.assignments.items():
            if shorten_campaign_name(campaign_name) == shortened_name:
                # Verify this combination exists
                completed = campaign_data.get('completed', {})
                if expert_id in completed and patient_id in completed[expert_id]:
                    return campaign_name
        return None
    
    def load_annotation_for_review(record_key):
        """Load annotation data and DICOM images for admin review"""
        if not record_key:
            return None, "No record selected", "", gr.update(visible=False)
        
        try:
            # Parse record key: campaign|expert|patient
            campaign_name, expert_id, patient_id = record_key.split('|')
            
            # Get campaign data to find dataset path
            if campaign_name not in crowdsourcing_manager.assignments:
                return None, "Campaign not found", "", gr.update(visible=False)
                
            campaign_data = crowdsourcing_manager.assignments[campaign_name]
            dataset_path = campaign_data.get('dataset_path', '')
            
            if not dataset_path:
                return None, "Dataset path not found", "", gr.update(visible=False)
            
            # Construct patient directory path
            patient_dir = os.path.join(dataset_path, patient_id)
            if not os.path.exists(patient_dir):
                return None, f"Patient directory not found: {patient_dir}", "", gr.update(visible=False)
            
            # Load DICOM data using the same logic as viewer tab
            try:
                from utils.dicom_utils import load_dicom_series
                from utils.visualization import display_slice
                import numpy as np
                
                # Look for FLAIR directory specifically within patient directory
                flair_dir = None
                potential_flair_dirs = []
                
                # Check for FLAIR subdirectories
                if os.path.exists(patient_dir):
                    for item in os.listdir(patient_dir):
                        item_path = os.path.join(patient_dir, item)
                        if os.path.isdir(item_path):
                            # Check if directory name contains 'flair' (case-insensitive)
                            if 'flair' in item.lower():
                                potential_flair_dirs.append(item_path)
                
                # Use the first FLAIR directory found, or fallback to patient directory
                if potential_flair_dirs:
                    flair_dir = potential_flair_dirs[0]
                    logger.info(f"Found FLAIR directory: {flair_dir}")
                else:
                    # Fallback: check if patient directory itself contains FLAIR scans
                    flair_dir = patient_dir
                    logger.info(f"No FLAIR subdirectory found, using patient directory: {flair_dir}")
                
                # Load DICOM series from FLAIR directory
                logger.info(f"Loading FLAIR DICOM series from: {flair_dir}")
                dicom_data, dicom_metadata, file_list = load_dicom_series(flair_dir)
                
                if dicom_data is None:
                    return None, f"Failed to load FLAIR DICOM data from {flair_dir}", "", gr.update(visible=False)
                    
                logger.info(f"Loaded FLAIR DICOM data shape: {dicom_data.shape}")
                
                # Get window center and width from metadata
                window_center = 500  # Default values
                window_width = 1000
                
                if dicom_metadata and 'WindowCenter' in dicom_metadata:
                    window_center = dicom_metadata['WindowCenter']
                    if isinstance(window_center, list):
                        window_center = window_center[0]
                
                if dicom_metadata and 'WindowWidth' in dicom_metadata:
                    window_width = dicom_metadata['WindowWidth']
                    if isinstance(window_width, list):
                        window_width = window_width[0]
                
                # Display the first slice (axial view)
                img = display_slice(
                    dicom_data, 
                    0,  # First slice
                    'axial',  # Default to axial view
                    window_level=window_center,
                    window_width=window_width,
                    crosshair=None
                )
                
                # Convert to RGB format for image_annotator
                if len(img.shape) == 2:
                    img_rgb = np.stack([img] * 3, axis=-1)
                else:
                    img_rgb = img
                
                if img_rgb.dtype != np.uint8:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                
                # Load saved annotations if they exist
                annotation_dir = f"db/submitted_annotations/{campaign_name}_{expert_id}_{patient_id}"
                boxes = []
                annotation_info = {}
                
                if os.path.exists(annotation_dir):
                    # Look for annotation files
                    for file in os.listdir(annotation_dir):
                        if file.endswith('.json'):
                            annotation_file = os.path.join(annotation_dir, file)
                            try:
                                with open(annotation_file, 'r') as f:
                                    annotation_data = json.load(f)
                                    if 'boxes' in annotation_data:
                                        boxes = annotation_data['boxes']
                                    annotation_info = annotation_data
                                    break
                            except Exception as e:
                                logger.error(f"Error loading annotation file {annotation_file}: {e}")
                
                # Create AnnotatedImageValue format with saved annotations
                annotated_value = {
                    "image": img_rgb,
                    "boxes": boxes,
                    "orientation": 0
                }
                
                # Store data for navigation (global state for now - should be improved)
                review_state = {
                    'dicom_data': dicom_data,
                    'dicom_metadata': dicom_metadata,
                    'file_list': file_list,
                    'current_slice': 0,
                    'patient_dir': patient_dir,
                    'flair_dir': flair_dir,  # Store FLAIR directory path
                    'annotation_dir': annotation_dir,
                    'window_center': window_center,
                    'window_width': window_width,
                    'record_key': record_key,
                    'boxes': boxes,
                    'annotation_info': annotation_info
                }
                
                # Store in a global variable for navigation
                global current_review_state
                current_review_state = review_state
                
                info_text = f"📋 Record: {record_key}\n🏥 Patient: {patient_id}\n👨‍⚕️ Expert: {expert_id}\n📊 Campaign: {campaign_name}\n🧠 Scan Type: FLAIR\n🔢 Slices: {dicom_data.shape[0]}\n📝 Annotations: {len(boxes)}"
                
                success_message = create_success_message(
                    title=f"FLAIR DICOM Data Loaded",
                    subtitle=f"Patient: {patient_id} • {dicom_data.shape[0]} slices • {len(boxes)} annotations",
                    icon="🧠"
                )
                
                return annotated_value, success_message, info_text, gr.update(visible=True)
                
            except ImportError as e:
                logger.error(f"Missing required modules for DICOM loading: {e}")
                error_message = f"""
                <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 10px; border: 1px solid #ef4444; margin: 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='width: 14px; height: 14px; background: #ef4444; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;'>❌</div>
                        <div>
                            <div style='color: #dc2626; font-weight: 600; font-size: 15px; margin-bottom: 2px;'>DICOM Loading Modules Missing</div>
                            <div style='color: #b91c1c; font-size: 13px; opacity: 0.8;'>{str(e)}</div>
                        </div>
                    </div>
                </div>
                """
                return None, error_message, "", gr.update(visible=False)
            except Exception as e:
                logger.error(f"Error loading DICOM data: {e}")
                error_message = f"""
                <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 10px; border: 1px solid #ef4444; margin: 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='width: 14px; height: 14px; background: #ef4444; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;'>❌</div>
                        <div>
                            <div style='color: #dc2626; font-weight: 600; font-size: 15px; margin-bottom: 2px;'>Error Loading DICOM</div>
                            <div style='color: #b91c1c; font-size: 13px; opacity: 0.8;'>{str(e)}</div>
                        </div>
                    </div>
                </div>
                """
                return None, error_message, "", gr.update(visible=False)
                
        except Exception as e:
            logger.error(f"Error in load_annotation_for_review: {e}")
            error_message = f"""
            <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 10px; border: 1px solid #ef4444; margin: 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <div style='width: 14px; height: 14px; background: #ef4444; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;'>❌</div>
                    <div>
                        <div style='color: #dc2626; font-weight: 600; font-size: 15px; margin-bottom: 2px;'>Error Loading Annotation</div>
                        <div style='color: #b91c1c; font-size: 13px; opacity: 0.8;'>{str(e)}</div>
                    </div>
                </div>
            </div>
            """
            return None, error_message, "", gr.update(visible=False)
    
    def update_review_slice(slider_value, view_type):
        """Update the displayed slice in review mode (adapted from viewer handlers)"""
        global current_review_state
        
        if 'current_review_state' not in globals() or not current_review_state:
            return None, "No data loaded", "x: 0, y: 0, z: 0", {}
        
        try:
            from utils.visualization import display_slice
            import numpy as np
            
            dicom_data = current_review_state['dicom_data']
            window_center = current_review_state['window_center']
            window_width = current_review_state['window_width']
            boxes = current_review_state['boxes']
            
            slice_idx = int(slider_value)
            current_review_state['current_slice'] = slice_idx
            
            # Display the slice
            img = display_slice(
                dicom_data,
                slice_idx,
                view_type.lower() if view_type else 'axial',
                window_level=window_center,
                window_width=window_width,
                crosshair=None
            )
            
            # Convert to RGB format
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            
            # Create annotated value with existing boxes
            annotated_value = {
                "image": img_rgb,
                "boxes": boxes,  # Keep the same boxes for all slices
                "orientation": 0
            }
            
            slice_text = f"{slice_idx}/{dicom_data.shape[0]-1}"
            crosshair_text = f"Slice: {slice_idx}, View: {view_type}"
            
            return annotated_value, slice_text, crosshair_text, current_review_state['dicom_metadata']
            
        except Exception as e:
            logger.error(f"Error updating review slice: {e}")
            return None, "Error updating slice", "Error", {}
    
    def review_prev_slice(slider_value):
        """Go to previous slice in review mode"""
        current_value = int(slider_value)
        return max(0, current_value - 1)
    
    def review_next_slice(slider_value):
        """Go to next slice in review mode"""
        global current_review_state
        
        if 'current_review_state' not in globals() or not current_review_state:
            return slider_value
        
        current_value = int(slider_value)
        max_slices = current_review_state['dicom_data'].shape[0] - 1
        return min(current_value + 1, max_slices)
    
    def apply_annotations_to_image(image_data, annotations):
        """Apply saved annotations to the image data for review"""
        if not image_data or not annotations:
            return image_data
        
        try:
            # Convert annotations back to image_annotator format
            if isinstance(image_data, dict) and 'image' in image_data:
                # Add annotations to the image data
                image_data['boxes'] = []
                
                for ann in annotations:
                    if ann.get('coordinates') and ann.get('label'):
                        # Convert coordinates back to image_annotator format
                        if ann.get('type') == 'polygon':
                            points = [{'x': coord[0], 'y': coord[1]} for coord in ann['coordinates']]
                            image_data['boxes'].append({
                                'type': 'polygon',
                                'points': points,
                                'label': ann['label']
                            })
                        elif ann.get('bbox'):
                            xmin, ymin, xmax, ymax = ann['bbox']
                            image_data['boxes'].append({
                                'type': 'box',
                                'xmin': xmin,
                                'ymin': ymin,
                                'xmax': xmax,
                                'ymax': ymax,
                                'label': ann['label']
                            })
            
            return image_data
            
        except Exception as e:
            logger.error(f"Error applying annotations to image: {e}")
            return image_data
    
    def update_expert_score(expert_id, new_score):
        """Update expert score in users.txt file"""
        try:
            users_file = "db/users.txt"
            if not os.path.exists(users_file):
                return False
            
            # Read current users
            lines = []
            with open(users_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update the expert's score
            updated = False
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('id,'):
                    parts = line.strip().split(',')
                    if len(parts) >= 4 and parts[0] == expert_id:
                        parts[3] = str(new_score)
                        lines[i] = ','.join(parts) + '\n'
                        updated = True
                        break
            
            if updated:
                with open(users_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                logger.info(f"Updated expert {expert_id} score to {new_score}")
                return True
            
        except Exception as e:
            logger.error(f"Error updating expert score: {e}")
        
        return False
    
    def process_admin_decision(record_key, decision, score, selected_rows):
        """Process admin approval/rejection decision"""
        if not record_key or not decision:
            return "Please select a record and make a decision", get_pending_approvals()[0]
        
        try:
            campaign_name, expert_id, patient_id = record_key.split('|')
            
            # Update the reviewed status in assignments.json
            if 'reviewed' not in crowdsourcing_manager.assignments[campaign_name]:
                crowdsourcing_manager.assignments[campaign_name]['reviewed'] = {}
            
            if expert_id not in crowdsourcing_manager.assignments[campaign_name]['reviewed']:
                crowdsourcing_manager.assignments[campaign_name]['reviewed'][expert_id] = []
            
            # Store the decision and timestamp
            review_data = {
                'patient_id': patient_id,
                'decision': decision,
                'timestamp': datetime.now().isoformat(),
                'admin_score': score
            }
            
            crowdsourcing_manager.assignments[campaign_name]['reviewed'][expert_id].append(review_data)
            crowdsourcing_manager._save_assignments()
            
            # Update expert score if provided
            if score and 1 <= score <= 5:
                update_expert_score(expert_id, score)
            
            # Create decision icon
            decision_icon = "✅" if decision == "accepted" else "❌"
            score_text = f" • Score: {score}/5" if score else ""
            
            decision_message = create_success_message(
                title=f"{decision.title()} Decision Recorded",
                subtitle=f"Expert: {expert_id} → Patient: {patient_id}{score_text}",
                icon=decision_icon
            )
            
            # Refresh the pending approvals list
            updated_df, _ = get_pending_approvals()
            
            return decision_message, updated_df
            
        except Exception as e:
            logger.error(f"Error processing admin decision: {e}")
            error_message = f"""
            <div style='padding: 16px; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 10px; border: 1px solid #ef4444; margin: 8px 0; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <div style='width: 14px; height: 14px; background: #ef4444; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: white; font-weight: bold;'>❌</div>
                    <div>
                        <div style='color: #dc2626; font-weight: 600; font-size: 15px; margin-bottom: 2px;'>Error Processing Decision</div>
                        <div style='color: #b91c1c; font-size: 13px; opacity: 0.8;'>{str(e)}</div>
                    </div>
                </div>
            </div>
            """
            return error_message, get_pending_approvals()[0]
    
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
        # Force reload assignments from file to ensure we have the latest data
        crowdsourcing_manager.load_assignments()
        
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
        # Force reload assignments from file to ensure we have the latest data
        crowdsourcing_manager.load_assignments()
        
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
                            Task Assignment Hub
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
                            label="Select Campaign",
                            choices=get_campaign_choices(),
                            value=None,
                            interactive=True,
                            elem_classes=["modern-dropdown"]
                        )
                        
                        expert_dropdown = gr.Dropdown(
                            label="Select Expert Annotator",
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
                                "Refresh Data", 
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
                                    Patient Assignment Pool
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
            
            # Pending Approvals tab
            with gr.Tab("Pending Approvals") as approvals_tab:
                gr.HTML("""
                <div style='padding: 12px; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 12px; border: 1px solid #7c3aed; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);'>
                    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 6px;'>
                        <div style='width: 14px; height: 14px; background: #6d28d9; border-radius: 50%; flex-shrink: 0;'></div>
                        <h2 style='color: white; font-weight: 700; font-size: 20px; margin: 0;'>
                            Pending Approvals
                        </h2>
                    </div>
                    <p style='color: #e9d5ff; font-size: 14px; margin: 0; line-height: 1.4;'>
                        Review and approve submitted annotations from expert annotators
                    </p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Pending Records Section
                        gr.HTML("""
                        <div style='padding: 12px; background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%); border-radius: 8px; border: 1px solid #8b5cf6; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                                <div style='width: 10px; height: 10px; background: #8b5cf6; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #581c87; font-weight: 600; font-size: 16px; margin: 0;'>
                                    Submitted Annotations
                                </h3>
                            </div>
                            <p style='color: #7c2d92; font-size: 13px; margin: 0; line-height: 1.4;'>
                                Select a record to review the submitted annotation
                            </p>
                        </div>
                        """)
                        
                        # Get initial pending approvals data
                        initial_df, count = get_pending_approvals()
                        current_time = datetime.now().strftime('%H:%M:%S')
                        
                        pending_dataframe = gr.Dataframe(
                            value=initial_df,
                            label=f"{count} Records Pending Approval • Last Updated: {current_time}",
                            interactive=False,
                            wrap=True,
                            column_widths=["25%", "17%", "15%", "15%"],
                            headers=["Campaign", "Expert", "Patient ID", "Status"]
                        )
                        
                        with gr.Row():
                            refresh_approvals_btn = gr.Button("🔄 Refresh Pending Approvals", variant="primary", size="sm")
                        
                        approval_status = gr.HTML("", visible=False)
                        
                        # Metadata Display in Accordion (below FLAIR DICOM Data Loaded message)
                        with gr.Accordion("Metadata", open=False, visible=False) as metadata_accordion:
                            review_metadata_display = gr.JSON(label="Metadata", open=False)
                    
                    with gr.Column(scale=3):
                        # Review Interface Section
                        gr.HTML("""
                        <div style='padding: 12px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 8px; border: 1px solid #10b981; margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);'>
                            <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>
                                <div style='width: 10px; height: 10px; background: #10b981; border-radius: 50%; flex-shrink: 0;'></div>
                                <h3 style='color: #065f46; font-weight: 600; font-size: 16px; margin: 0;'>
                                    Annotation Review Interface
                                </h3>
                            </div>
                            <p style='color: #047857; font-size: 13px; margin: 0; line-height: 1.4;'>
                                Review the submitted annotations and make approval decisions
                            </p>
                        </div>
                        """)
                        
                        # Review info display
                        review_info = gr.Textbox(
                            label="📊 Record Information",
                            value="",
                            interactive=False,
                            lines=3,
                            visible=False
                        )
                        
                        # View Orientation Control
                        review_view_selector = gr.Radio(
                            choices=["Axial", "Sagittal", "Coronal"], 
                            value="Axial", 
                            label="View Orientation",
                            visible=False
                        )
                        
                        # Medical Image Viewer (adapted from Viewer tab)
                        review_image_annotator = image_annotator(
                            value=None,
                            label="Review Mode", 
                            label_list=["Normal Tissue", "Tumor", "Organ", "Lesion", "ROI", "Other"],
                            label_colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)],
                            box_min_size=10,
                            handle_size=8,
                            box_thickness=2,
                            box_selected_thickness=3,
                            boxes_alpha=0.7,
                            height=450,
                            width=1200,
                            interactive=True,  # Read-only for review
                            show_label=True,
                            show_download_button=True,
                            show_clear_button=True,
                            show_remove_button=True,
                            use_default_label=False,
                            handles_cursor=True,
                            image_type="numpy",
                            single_box=False,
                            disable_edit_boxes=False,  # Disable editing - only review ?
                            shape_creation_mode="disabled",
                            visible=False
                        )
                        
                        # Navigation Controls (adapted from Viewer tab)
                        with gr.Row(visible=False) as nav_controls:
                            review_prev_btn = gr.Button("Previous")
                            review_slice_slider = gr.Slider(
                                minimum=0, maximum=0, value=0, step=1, 
                                label="Slice Navigation", visible=True
                            )
                            review_next_btn = gr.Button("Next")
                            
                        # Additional viewer controls
                        with gr.Row(visible=False) as viewer_metadata_controls:
                            review_slice_text = gr.Textbox(label="Slice", interactive=False, visible=False)
                            review_crosshair_info = gr.Textbox(label="Crosshair", interactive=False, visible=False)
                        
                        # Expert Rating
                        expert_rating = gr.Radio(
                            choices=[1, 2, 3, 4, 5],
                            value=None,
                            label="Rate Expert Performance (1-5)",
                            info="Rate the quality of this annotation for expert scoring",
                            visible=False
                        )
                        
                        # Admin Decision Controls
                        with gr.Row(visible=False) as decision_controls:
                            accept_btn = gr.Button("Accept Annotation", variant="primary", size="lg")
                            reject_btn = gr.Button("Reject Annotation", variant="stop", size="lg")
                        
                        # Decision status
                        decision_status = gr.HTML("", visible=False)
                        
                        # Hidden state to store currently selected record key
                        selected_record_state = gr.State("")
            
            # Second tab: Create New Campaign with modern Next.js styling
            with gr.Tab("Create Campaign") as create_tab:
                # Modern Header
                gr.HTML("""
                <div style='padding: 24px; background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%); border-radius: 16px; border: 1px solid #0891b2; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                    <div style='display: flex; align-items: center; gap: 16px; margin-bottom: 12px;'>
                        <div style='width: 16px; height: 16px; background: #0284c7; border-radius: 50%; flex-shrink: 0;'></div>
                        <h2 style='color: white; font-weight: 700; font-size: 24px; margin: 0;'>
                            Create New Campaign
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
                                    Campaign Configuration
                                </h3>
                            </div>
                            <p style='color: #0369a1; font-size: 14px; margin: 0; line-height: 1.5;'>
                                Define your campaign name and specify the dataset location
                            </p>
                        </div>
                        """)
                        
                        campaign_name_input = gr.Textbox(
                            label="Campaign Name",
                            placeholder="Enter a descriptive campaign name (e.g., 'Brain MRI Segmentation Q1 2025')",
                            interactive=True,
                            elem_classes=["modern-textbox"]
                        )
                        
                        dataset_path_input = gr.Textbox(
                            label="Dataset Path",
                            placeholder="Enter the full path to your dataset directory",
                            interactive=True,
                            elem_classes=["modern-textbox"]
                        )
                                              
                        with gr.Row():
                            scan_button = gr.Button(
                                "Analyze Dataset", 
                                variant="secondary",
                                size="lg",
                                elem_classes=["scan-btn"]
                            )
                            create_button = gr.Button(
                                "Create Campaign", 
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
                                    Dataset Analysis
                                </h3>
                            </div>
                        </div>
                        """)
                        
                        scan_results = gr.HTML(
                            value="""
                            <div style='padding: 16px; background: #f8fafc; border-radius: 8px; border: 2px dashed #cbd5e1; text-align: center;'>
                                <div style='color: #64748b; font-size: 14px; margin-bottom: 8px;'>
                                    Dataset Analysis Results
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
                                    Creation Status
                                </h3>
                            </div>
                        </div>
                        """)
                        
                        create_status = gr.HTML(
                            value="""
                            <div style='padding: 16px; background: #f8fafc; border-radius: 8px; border: 2px dashed #cbd5e1; text-align: center;'>
                                <div style='color: #64748b; font-size: 14px; margin-bottom: 8px;'>
                                    Campaign Status
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
    
    # Pending Approvals event handlers
    def handle_load_selected_record(evt: SelectData):
        """Handle loading selected record for review"""
        logger.info(f"Selected row index: {evt.index}, Selected cell value: {evt.value}")
        
        # Get the current dataframe data and extract the selected row
        current_df, _ = get_pending_approvals()
        
        if current_df.empty or evt.index[0] >= len(current_df):
            logger.error(f"Invalid row index: {evt.index[0]}, dataframe length: {len(current_df)}")
            return None, "Invalid row selection", "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(minimum=0, maximum=0, value=0), "", ""
        
        # Extract campaign, expert, patient from selected row using dataframe iloc
        row_data = current_df.iloc[evt.index[0]]
        shortened_campaign = row_data['Campaign']
        expert_name = row_data['Expert']
        patient_id = row_data['Patient ID']
        
        # Get full campaign name for processing
        campaign_name = get_full_campaign_name(shortened_campaign, expert_name, patient_id)
        
        logger.info(f"Extracted from row {evt.index[0]}: Shortened Campaign={shortened_campaign}, Full Campaign={campaign_name}, Expert={expert_name}, Patient={patient_id}")
        
        if not campaign_name or not expert_name or not patient_id:
            no_record_message = create_info_message(
                title="No Records Available",
                subtitle="Please select a valid record with complete information",
                icon="ℹ️"
            )
            return None, no_record_message, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(minimum=0, maximum=0, value=0), "", ""
        
        # Create record key from extracted data (using pipe separator as expected by load_annotation_for_review)
        record_key = f"{campaign_name}|{expert_name}|{patient_id}"
        logger.info(f"Loading record: {record_key} (Campaign: {campaign_name}, Expert: {expert_name}, Patient: {patient_id})")
        
        # Load the annotation for review
        result, status, info, controls_visible = load_annotation_for_review(record_key)
        
        # Set up slider based on loaded data
        global current_review_state
        if 'current_review_state' in globals() and current_review_state:
            max_slices = current_review_state['dicom_data'].shape[0] - 1
            slider_update = gr.update(minimum=0, maximum=max_slices, value=0, visible=True)
        else:
            slider_update = gr.update(minimum=0, maximum=0, value=0, visible=False)
        
        return (
            result,  # review_image_annotator
            status,  # approval_status 
            info,    # review_info
            controls_visible,  # review_view_selector visibility
            controls_visible,  # review_image_annotator visibility  
            controls_visible,  # nav_controls visibility
            controls_visible,  # decision_controls visibility
            controls_visible,  # expert_rating visibility
            controls_visible,  # viewer_metadata_controls visibility
            controls_visible,  # metadata_accordion visibility
            controls_visible,  # metadata display visibility
            slider_update,     # review_slice_slider update
            record_key,      # selected_record_state
            ""               # decision_status (clear previous decision messages)
        )
    
    def handle_admin_decision(decision_type, rating, record_key):
        """Handle admin approval/rejection decision"""
        logger.info(f"Admin decision: {decision_type}, rating: {rating}, record_key: {record_key}")
        
        if not record_key:
            return "No record selected", gr.update(), gr.update(value=None)
        
        try:
            decision_message, updated_df = process_admin_decision(record_key, decision_type, rating, None)
            
            # Get updated count and create proper dataframe update with timestamp
            current_time = datetime.now().strftime('%H:%M:%S')
            count = len(updated_df) if updated_df is not None and not updated_df.empty else 0
            
            df_update = gr.update(
                value=updated_df,
                label=f"{count} Records Pending Approval • Updated: {current_time}"
            )
            
            # Clear the rating selection after decision
            return decision_message, df_update, gr.update(value=None)
            
        except Exception as e:
            logger.error(f"Error in handle_admin_decision: {e}")
            updated_df, count = get_pending_approvals()
            current_time = datetime.now().strftime('%H:%M:%S')
            
            df_update = gr.update(
                value=updated_df,
                label=f"{count} Records Pending Approval • Updated: {current_time}"
            )
            
            return f"Error processing decision: {str(e)}", df_update, gr.update(value=None)
    
    def refresh_pending_with_feedback():
        """Refresh pending approvals and return updated dataframe with count"""
        updated_df, count = get_pending_approvals()
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Update the dataframe with new label
        df_update = gr.update(
            value=updated_df,
            label=f"{count} Records Pending Approval • Last Updated: {current_time}"
        )
        
        # Create a feedback message about the refresh
        refresh_message = create_info_message(
            title=f"Refreshed - {count} Records Found",
            subtitle=f"Updated at {current_time}",
            icon="🔄"
        ) if count > 0 else create_info_message(
            title="No Pending Records",
            subtitle=f"All annotations have been reviewed • Updated at {current_time}",
            icon="✅"
        )
        
        return df_update, refresh_message
    
    def auto_refresh_on_tab_select():
        """Auto-refresh when tab is selected"""
        updated_df, count = get_pending_approvals()
        current_time = datetime.now().strftime('%H:%M:%S')
        
        return gr.update(
            value=updated_df,
            label=f"{count} Records Pending Approval • Auto-Updated: {current_time}"
        )
    
    refresh_approvals_btn.click(
        fn=refresh_pending_with_feedback,
        outputs=[pending_dataframe, approval_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[approval_status]
    )
    
    # Auto-refresh pending approvals when tab becomes active
    approvals_tab.select(
        fn=auto_refresh_on_tab_select,
        outputs=[pending_dataframe]
    )
    
    # Auto-load when dataframe row is selected
    pending_dataframe.select(
        fn=handle_load_selected_record,
        outputs=[
            review_image_annotator, 
            approval_status, 
            review_info,
            review_view_selector,
            review_image_annotator,
            nav_controls,
            decision_controls,
            expert_rating,
            viewer_metadata_controls,
            metadata_accordion,
            review_metadata_display,
            review_slice_slider,
            selected_record_state,
            decision_status
        ]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[approval_status]
    )
    
    # Navigation event handlers (adapted from viewer tab)
    review_slice_slider.change(
        fn=update_review_slice,
        inputs=[review_slice_slider, review_view_selector],
        outputs=[review_image_annotator, review_slice_text, review_crosshair_info, review_metadata_display]
    )
    
    review_view_selector.change(
        fn=update_review_slice,
        inputs=[review_slice_slider, review_view_selector],
        outputs=[review_image_annotator, review_slice_text, review_crosshair_info, review_metadata_display]
    )
    
    review_prev_btn.click(
        fn=review_prev_slice,
        inputs=[review_slice_slider],
        outputs=[review_slice_slider]
    )
    
    review_next_btn.click(
        fn=review_next_slice,
        inputs=[review_slice_slider],
        outputs=[review_slice_slider]
    )
    
    accept_btn.click(
        fn=lambda rating, record_key: handle_admin_decision("accepted", rating, record_key),
        inputs=[expert_rating, selected_record_state],
        outputs=[decision_status, pending_dataframe, expert_rating]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[decision_status]
    )
    
    reject_btn.click(
        fn=lambda rating, record_key: handle_admin_decision("rejected", rating, record_key),
        inputs=[expert_rating, selected_record_state],
        outputs=[decision_status, pending_dataframe, expert_rating]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[decision_status]
    )
    
    return {
        'sub_tabs': sub_tabs,
        'campaigns_tab': campaigns_tab,
        'approvals_tab': approvals_tab,
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
        'refresh_assignment_button': refresh_assignment_button,
        'pending_dataframe': pending_dataframe,
        'refresh_approvals_btn': refresh_approvals_btn,
        'review_image_annotator': review_image_annotator,
        'accept_btn': accept_btn,
        'reject_btn': reject_btn,
        'expert_rating': expert_rating,
        'selected_record_state': selected_record_state
    }
