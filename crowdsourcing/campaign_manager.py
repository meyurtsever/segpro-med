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
    
    def create_campaign(self, campaign_name, dataset_path, description=""):
        """Create a new crowdsourcing campaign"""
        if not campaign_name or not dataset_path:
            return False
            
        total_patients, patient_list = self.scan_dataset(dataset_path)
        if total_patients == 0:
            return False
        
        self.assignments[campaign_name] = {
            'name': campaign_name,
            'dataset_path': dataset_path,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'total_patients': total_patients,
            'patients': patient_list,
            'assigned': {},
            'completed': {},
            'reviewed': {}
        }
        
        self._save_assignments()
        logger.info(f"Created campaign {campaign_name} with {total_patients} patients")
        return True
    
    def assign_patients(self, campaign_name, expert_id, patient_ids):
        """Assign patients to an expert"""
        if campaign_name not in self.assignments:
            return False
        
        # Validate that all patient_ids exist in the campaign
        campaign_patients = self.assignments[campaign_name]['patients']
        for patient_id in patient_ids:
            if patient_id not in campaign_patients:
                return False
        
        if expert_id not in self.assignments[campaign_name]['assigned']:
            self.assignments[campaign_name]['assigned'][expert_id] = []
        
        # Add new assignments (avoid duplicates)
        for patient_id in patient_ids:
            if patient_id not in self.assignments[campaign_name]['assigned'][expert_id]:
                self.assignments[campaign_name]['assigned'][expert_id].append(patient_id)
        
        self._save_assignments()
        logger.info(f"Assigned {len(patient_ids)} patients to {expert_id} in {campaign_name}")
        return True
    
    def get_assigned_tasks(self, expert_id):
        """Get all tasks assigned to a specific expert across all campaigns"""
        assigned = []
        for campaign_name, campaign_data in self.assignments.items():
            if expert_id in campaign_data.get('assigned', {}):
                for patient_id in campaign_data['assigned'][expert_id]:
                    assigned.append({
                        'campaign': campaign_name,
                        'patient_id': patient_id,
                        'dataset_path': campaign_data['dataset_path']
                    })
        return assigned
    
    def get_remaining_assignments_for_user(self, expert_id):
        """Get remaining (uncompleted) assignments for a specific expert"""
        remaining = []
        for campaign_name, campaign_data in self.assignments.items():
            if expert_id in campaign_data.get('assigned', {}):
                # Get assigned patients for this expert
                assigned_patients = campaign_data['assigned'][expert_id]
                
                # Get completed patients for this expert
                completed_patients = campaign_data.get('completed', {}).get(expert_id, [])
                
                # Find remaining patients (assigned but not completed)
                for patient_id in assigned_patients:
                    if patient_id not in completed_patients:
                        remaining.append({
                            'campaign_id': campaign_name,
                            'patient_id': patient_id,
                            'dataset_path': campaign_data['dataset_path']
                        })
        
        logger.info(f"Found {len(remaining)} remaining assignments for expert {expert_id}")
        return remaining
    
    def get_campaign_progress(self, campaign_name):
        """Get progress statistics for a campaign"""
        if campaign_name not in self.assignments:
            return None
        
        campaign = self.assignments[campaign_name]
        total_patients = campaign['total_patients']
        
        # Count assigned patients
        assigned_patients = set()
        for expert_assignments in campaign.get('assigned', {}).values():
            assigned_patients.update(expert_assignments)
        assigned_count = len(assigned_patients)
        
        # Count completed patients
        completed_patients = set()
        for expert_completions in campaign.get('completed', {}).values():
            completed_patients.update(expert_completions)
        completed_count = len(completed_patients)
        
        # Count reviewed patients
        reviewed_patients = set()
        for expert_reviews in campaign.get('reviewed', {}).values():
            reviewed_patients.update(expert_reviews)
        reviewed_count = len(reviewed_patients)
        
        # Calculate unassigned
        unassigned_count = total_patients - assigned_count
        
        return {
            'total_patients': total_patients,
            'assigned_patients': assigned_count,
            'completed': completed_count,
            'reviewed': reviewed_count,
            'unassigned_patients': unassigned_count
        }
    
    def mark_completed(self, campaign_name, expert_id, patient_id, annotation_data=None):
        """Mark a patient as completed by an expert and save annotation data"""
        if campaign_name not in self.assignments:
            return False
        
        # Initialize completed section if not exists
        if 'completed' not in self.assignments[campaign_name]:
            self.assignments[campaign_name]['completed'] = {}
        
        if expert_id not in self.assignments[campaign_name]['completed']:
            self.assignments[campaign_name]['completed'][expert_id] = []
        
        # Only add if not already completed (avoid duplicates)
        if patient_id not in self.assignments[campaign_name]['completed'][expert_id]:
            self.assignments[campaign_name]['completed'][expert_id].append(patient_id)
            
            # Save annotation data for admin inspection
            if annotation_data:
                self._save_annotation_submission(campaign_name, expert_id, patient_id, annotation_data)
            
            self._save_assignments()
            logger.info(f"Marked patient {patient_id} as completed by {expert_id} in campaign {campaign_name}")
            return True
        
        return True  # Already completed
    
    def _save_annotation_submission(self, campaign_name, expert_id, patient_id, annotation_data):
        """Save annotation submission for admin inspection"""
        import json
        import os
        from datetime import datetime
        import base64
        from PIL import Image
        import io
        import numpy as np
        
        # Create submission directory structure
        submissions_dir = os.path.join("db", "submitted_annotations")
        campaign_dir = os.path.join(submissions_dir, campaign_name)
        expert_dir = os.path.join(campaign_dir, expert_id)
        patient_dir = os.path.join(expert_dir, patient_id)
        
        os.makedirs(patient_dir, exist_ok=True)
        
        # Prepare submission metadata
        submission_metadata = {
            "campaign_name": campaign_name,
            "expert_id": expert_id,
            "patient_id": patient_id,
            "submission_timestamp": datetime.now().isoformat(),
            "submission_id": f"{campaign_name}_{expert_id}_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Save annotation data (overlays, labels, positions)
        annotation_file = os.path.join(patient_dir, "annotation_data.json")
        try:
            # Extract relevant annotation information
            annotation_info = {
                "metadata": submission_metadata,
                "annotations": [],
                "labels": [],
                "total_annotations": 0
            }
            
            # Process annotation data based on its structure
            if annotation_data and hasattr(annotation_data, 'get'):
                # Handle image annotator data structure
                annotations = annotation_data.get('annotations', [])
                for idx, annotation in enumerate(annotations):
                    ann_data = {
                        "annotation_id": idx,
                        "type": annotation.get('type', 'unknown'),
                        "label": annotation.get('label', ''),
                        "coordinates": annotation.get('coordinates', []),
                        "bbox": annotation.get('bbox', []),
                        "area": annotation.get('area', 0)
                    }
                    annotation_info["annotations"].append(ann_data)
                    
                    # Collect unique labels
                    if ann_data["label"] and ann_data["label"] not in annotation_info["labels"]:
                        annotation_info["labels"].append(ann_data["label"])
                
                annotation_info["total_annotations"] = len(annotations)
                
                # Save annotated image if available
                if 'image' in annotation_data:
                    self._save_annotated_image(patient_dir, annotation_data['image'], submission_metadata["submission_id"], annotation_data)
            
            # Save annotation data to JSON
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved annotation submission: {annotation_file}")
            
        except Exception as e:
            logger.error(f"Error saving annotation submission: {e}")
    
    def _save_annotated_image(self, patient_dir, image_data, submission_id, annotation_data=None):
        """Save annotated image as PNG with overlays drawn from annotation data"""
        try:
            import base64
            from PIL import Image, ImageDraw, ImageFont
            import io
            import numpy as np
            
            # Handle different image data formats
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # Handle base64 encoded images
                header, data = image_data.split(',', 1)
                image_bytes = base64.b64decode(data)
                image = Image.open(io.BytesIO(image_bytes))
            elif hasattr(image_data, 'save'):
                # Handle PIL Image objects
                image = image_data
            elif isinstance(image_data, np.ndarray):
                # Handle numpy arrays
                if image_data.dtype != np.uint8:
                    if image_data.max() <= 1.0:
                        image_data = (image_data * 255).astype(np.uint8)
                    else:
                        image_data = image_data.astype(np.uint8)
                
                if len(image_data.shape) == 2:
                    # Grayscale to RGB
                    image_data = np.stack([image_data] * 3, axis=-1)
                elif len(image_data.shape) == 3 and image_data.shape[2] == 1:
                    image_data = np.stack([image_data.squeeze()] * 3, axis=-1)
                
                image = Image.fromarray(image_data)
            else:
                logger.warning(f"Unsupported image data format for saving: {type(image_data)}")
                return
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create annotated version if annotation data is available
            if annotation_data and annotation_data.get('annotations'):
                image = self._draw_annotations_on_image(image, annotation_data['annotations'])
            
            # Save as PNG
            image_file = os.path.join(patient_dir, f"annotated_slice_{submission_id}.png")
            image.save(image_file, 'PNG')
            logger.info(f"Saved annotated image with {len(annotation_data.get('annotations', [])) if annotation_data else 0} overlays: {image_file}")
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
    
    def _draw_annotations_on_image(self, image, annotations):
        """Draw annotations on the image using coordinates, bbox, and labels"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a copy to avoid modifying the original
            annotated_image = image.copy()
            
            # Create overlay for semi-transparent fills
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Draw on the base image for outlines
            base_draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a font for labels
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
            
            # Color palette for different annotations
            colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
            ]
            
            for idx, annotation in enumerate(annotations):
                try:
                    # Get annotation properties
                    ann_type = annotation.get('type', 'unknown')
                    label = annotation.get('label', f'Annotation {idx}')
                    coordinates = annotation.get('coordinates', [])
                    bbox = annotation.get('bbox', [])
                    
                    # Select color based on annotation index
                    color = colors[idx % len(colors)]
                    fill_color = color + (128,)  # Semi-transparent
                    
                    logger.info(f"Drawing annotation {idx}: type={ann_type}, label={label}, coords={len(coordinates)}, bbox={bbox}")
                    
                    # Draw based on annotation type and available data
                    if ann_type in ['polygon', 'freehand'] and coordinates:
                        # Draw polygon/freehand from coordinates
                        if len(coordinates) >= 3:  # Need at least 3 points for a polygon
                            # Convert coordinates to tuple format
                            points = []
                            for coord in coordinates:
                                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                    points.append((int(coord[0]), int(coord[1])))
                            
                            if len(points) >= 3:
                                # Fill with semi-transparent color
                                overlay_draw.polygon(points, fill=fill_color)
                                # Draw outline
                                base_draw.polygon(points, outline=color, width=2)
                                
                                # Draw label near the first point
                                if points:
                                    label_x, label_y = points[0]
                                    base_draw.text((label_x + 5, label_y - 20), label, fill=color, font=font)
                    
                    elif bbox and len(bbox) >= 4:
                        # Draw rectangle from bbox
                        x1, y1, x2, y2 = bbox[:4]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Fill with semi-transparent color
                        overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
                        # Draw outline
                        base_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        
                        # Draw label at top-left corner
                        base_draw.text((x1 + 5, y1 - 20), label, fill=color, font=font)
                    
                    elif coordinates and len(coordinates) >= 2:
                        # Fallback: draw as polygon if we have coordinates
                        points = []
                        for coord in coordinates:
                            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                points.append((int(coord[0]), int(coord[1])))
                        
                        if len(points) >= 2:
                            if len(points) == 2:
                                # Draw line for 2 points
                                base_draw.line(points, fill=color, width=3)
                            else:
                                # Draw polygon for multiple points
                                overlay_draw.polygon(points, fill=fill_color)
                                base_draw.polygon(points, outline=color, width=2)
                            
                            # Draw label
                            label_x, label_y = points[0]
                            base_draw.text((label_x + 5, label_y - 20), label, fill=color, font=font)
                    
                    else:
                        logger.warning(f"Annotation {idx} has insufficient coordinate data: type={ann_type}, coords={coordinates}, bbox={bbox}")
                
                except Exception as e:
                    logger.error(f"Error drawing annotation {idx}: {e}")
                    continue
            
            # Composite the overlay onto the base image
            annotated_image = annotated_image.convert('RGBA')
            final_image = Image.alpha_composite(annotated_image, overlay)
            return final_image.convert('RGB')
            
        except Exception as e:
            logger.error(f"Error drawing annotations on image: {e}")
            return image  # Return original image if drawing fails
    
    def get_unassigned_patients(self, campaign_name):
        """Get list of unassigned patients for a campaign"""
        if campaign_name not in self.assignments:
            return []
        
        campaign = self.assignments[campaign_name]
        all_patients = set(campaign['patients'])
        
        # Get all assigned patients
        assigned_patients = set()
        for expert_assignments in campaign.get('assigned', {}).values():
            assigned_patients.update(expert_assignments)
        
        unassigned = list(all_patients - assigned_patients)
        return sorted(unassigned)
