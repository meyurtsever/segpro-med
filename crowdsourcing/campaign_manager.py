"""
Crowdsourcing campaign management utilities
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class CrowdsourcingManager:
    """Manages crowdsourcing campaigns and assignments"""
    
    def __init__(self, assignments_file_path="db/assignments.json"):
        self.assignments_file_path = assignments_file_path
        self.assignments = self._load_assignments()
    
    def _load_assignments(self):
        """Load assignments from JSON file"""
        if not os.path.exists(self.assignments_file_path):
            return {}
        
        try:
            with open(self.assignments_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading assignments: {e}")
            return {}
    
    def _save_assignments(self):
        """Save assignments to JSON file"""
        try:
            with open(self.assignments_file_path, 'w', encoding='utf-8') as file:
                json.dump(self.assignments, file, indent=2)
        except Exception as e:
            logger.error(f"Error saving assignments: {e}")
    
    def scan_dataset(self, dataset_path):
        """Scan dataset directory for patients with valid modalities"""
        if not os.path.exists(dataset_path):
            return 0, []
        
        valid_modalities = ['flair', 't1', 't1c', 't2']
        patients = []
        
        try:
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains valid modalities
                    subdirs = [d.lower() for d in os.listdir(item_path) 
                              if os.path.isdir(os.path.join(item_path, d))]
                    
                    if any(modality in subdirs for modality in valid_modalities):
                        patients.append(item)
            
            logger.info(f"Found {len(patients)} patients in dataset: {dataset_path}")
            return len(patients), sorted(patients)
            
        except Exception as e:
            logger.error(f"Error scanning dataset {dataset_path}: {e}")
            return 0, []
    
    def create_campaign(self, campaign_name, dataset_path):
        """Create a new crowdsourcing campaign"""
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        total_patients, patient_list = self.scan_dataset(dataset_path)
        
        self.assignments[campaign_id] = {
            'name': campaign_name,
            'dataset_path': dataset_path,
            'created_at': datetime.now().isoformat(),
            'total_patients': total_patients,
            'patient_list': patient_list,
            'assigned': {},
            'completed': {},
            'reviewed': {}
        }
        
        self._save_assignments()
        logger.info(f"Created campaign {campaign_id} with {total_patients} patients")
        return campaign_id
    
    def assign_patients(self, campaign_id, expert_id, patient_ids):
        """Assign patients to an expert"""
        if campaign_id not in self.assignments:
            return False
        
        if expert_id not in self.assignments[campaign_id]['assigned']:
            self.assignments[campaign_id]['assigned'][expert_id] = []
        
        # Add new assignments (avoid duplicates)
        for patient_id in patient_ids:
            if patient_id not in self.assignments[campaign_id]['assigned'][expert_id]:
                self.assignments[campaign_id]['assigned'][expert_id].append(patient_id)
        
        self._save_assignments()
        logger.info(f"Assigned {len(patient_ids)} patients to {expert_id} in {campaign_id}")
        return True
    
    def get_assigned_patients(self, expert_id):
        """Get all patients assigned to a specific expert across all campaigns"""
        assigned = []
        for campaign_id, campaign_data in self.assignments.items():
            if expert_id in campaign_data.get('assigned', {}):
                for patient_id in campaign_data['assigned'][expert_id]:
                    assigned.append({
                        'campaign_id': campaign_id,
                        'campaign_name': campaign_data['name'],
                        'patient_id': patient_id,
                        'dataset_path': campaign_data['dataset_path']
                    })
        return assigned
    
    def get_campaign_progress(self, campaign_id):
        """Get progress statistics for a campaign"""
        if campaign_id not in self.assignments:
            return None
        
        campaign = self.assignments[campaign_id]
        total = campaign['total_patients']
        
        # Count assigned patients
        assigned_patients = set()
        for expert_assignments in campaign.get('assigned', {}).values():
            assigned_patients.update(expert_assignments)
        assigned = len(assigned_patients)
        
        # Count completed patients
        completed_patients = set()
        for expert_completions in campaign.get('completed', {}).values():
            completed_patients.update(expert_completions)
        completed = len(completed_patients)
        
        # Count reviewed patients
        reviewed_patients = set()
        for expert_reviews in campaign.get('reviewed', {}).values():
            reviewed_patients.update(expert_reviews)
        reviewed = len(reviewed_patients)
        
        return {
            'total': total,
            'assigned': assigned,
            'completed': completed,
            'reviewed': reviewed
        }
    
    def mark_completed(self, campaign_id, expert_id, patient_id):
        """Mark a patient as completed by an expert"""
        if campaign_id not in self.assignments:
            return False
        
        if expert_id not in self.assignments[campaign_id]['completed']:
            self.assignments[campaign_id]['completed'][expert_id] = []
        
        if patient_id not in self.assignments[campaign_id]['completed'][expert_id]:
            self.assignments[campaign_id]['completed'][expert_id].append(patient_id)
        
        self._save_assignments()
        return True
    
    def get_unassigned_patients(self, campaign_id):
        """Get list of unassigned patients for a campaign"""
        if campaign_id not in self.assignments:
            return []
        
        campaign = self.assignments[campaign_id]
        all_patients = set(campaign['patient_list'])
        
        # Get all assigned patients
        assigned_patients = set()
        for expert_assignments in campaign.get('assigned', {}).values():
            assigned_patients.update(expert_assignments)
        
        unassigned = list(all_patients - assigned_patients)
        return sorted(unassigned)
