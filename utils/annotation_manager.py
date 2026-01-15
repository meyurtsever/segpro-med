"""
SegMed-Pro Annotation Manager

This module provides comprehensive annotation storage and retrieval functionality
for all types of annotations (manual, guided, whole-area, VLM-suggested).

Storage Structure:
    db/annotation_records/{user_id}/{study_hash}/annotations.json
    
Each annotation record contains:
    - user_id: ID from db/users.txt
    - study_info: Complete path/metadata of the source data
    - slice_annotations: Array of all annotations for all slices
    - annotation_type: 'manual', 'guided_segmentation', 'whole_area_segmentation'
    - vlm_labels: Suggested and accepted labels from VLMs
    - timestamps: Creation and modification times
"""

import os
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class AnnotationManager:
    """Manages persistent storage and retrieval of annotations"""
    
    def __init__(self, base_dir: str = "db/annotation_records"):
        """
        Initialize annotation manager
        
        Args:
            base_dir: Base directory for storing annotations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Annotation manager initialized with base dir: {self.base_dir}")
    
    def _get_study_hash(self, study_path: str) -> str:
        """
        Generate a unique hash for a study/dataset
        
        Args:
            study_path: Absolute path to the study directory or file
            
        Returns:
            str: MD5 hash of the study path
        """
        # Normalize path for consistent hashing
        normalized_path = os.path.normpath(study_path).lower()
        return hashlib.md5(normalized_path.encode()).hexdigest()[:16]
    
    def _get_annotation_dir(self, user_id: str, study_path: str) -> Path:
        """
        Get the annotation directory for a specific user and study
        
        Args:
            user_id: User ID from db/users.txt
            study_path: Path to the study/dataset
            
        Returns:
            Path: Directory path for annotations
        """
        study_hash = self._get_study_hash(study_path)
        annotation_dir = self.base_dir / user_id / study_hash
        annotation_dir.mkdir(parents=True, exist_ok=True)
        return annotation_dir
    
    def _get_annotation_file(self, user_id: str, study_path: str) -> Path:
        """
        Get the annotation file path for a specific user and study
        
        Args:
            user_id: User ID
            study_path: Path to the study/dataset
            
        Returns:
            Path: Full path to annotations.json file
        """
        annotation_dir = self._get_annotation_dir(user_id, study_path)
        return annotation_dir / "annotations.json"
    
    def save_annotations(
        self,
        user_id: str,
        study_path: str,
        slice_annotations: List[Dict[str, Any]],
        annotation_type: str,
        study_metadata: Optional[Dict[str, Any]] = None,
        vlm_labels: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save annotations for a user and study
        
        Args:
            user_id: User ID from db/users.txt
            study_path: Absolute path to the study directory or file
            slice_annotations: List of annotation dictionaries for all slices
            annotation_type: Type of annotation ('manual', 'guided_segmentation', 'whole_area_segmentation')
            study_metadata: Optional metadata about the study (modality, dimensions, etc.)
            vlm_labels: Optional VLM-suggested and accepted labels
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            annotation_file = self._get_annotation_file(user_id, study_path)

            # Load existing annotations if they exist
            existing_data = self.load_annotations(user_id, study_path)

            if existing_data:
                # Update existing record metadata
                annotation_data = existing_data
                annotation_data['modified_at'] = datetime.now().isoformat()
                annotation_data['modification_count'] = annotation_data.get('modification_count', 0) + 1
            else:
                # Create new record
                annotation_data = {
                    'user_id': user_id,
                    'study_path': os.path.normpath(study_path),
                    'study_hash': self._get_study_hash(study_path),
                    'created_at': datetime.now().isoformat(),
                    'modified_at': datetime.now().isoformat(),
                    'modification_count': 0
                }

            # Ensure annotation_type and study metadata are set/merged
            annotation_data['annotation_type'] = annotation_type
            if study_metadata:
                # Merge study metadata, preferring new values
                merged_meta = annotation_data.get('study_metadata', {}) or {}
                merged_meta.update(study_metadata)
                annotation_data['study_metadata'] = merged_meta

            # Merge slice annotations rather than overwrite
            existing_slice_annotations = annotation_data.get('slice_annotations', []) or []

            # Build a map of existing annotations keyed by annotation_id when available
            existing_map: Dict[str, Dict[str, Any]] = {}
            for ann in existing_slice_annotations:
                ann_id = ann.get('annotation_id')
                if ann_id:
                    existing_map[ann_id] = ann

            # Add or update incoming annotations
            new_count = 0
            for ann in slice_annotations:
                ann_id = ann.get('annotation_id')
                if not ann_id:
                    # Ensure every annotation has an id
                    ann_id = f"slice_{ann.get('slice_idx', 'unknown')}_{int(time.time() * 1000)}"
                    ann['annotation_id'] = ann_id

                if ann_id in existing_map:
                    # Update existing annotation (replace fields)
                    existing_map[ann_id].update(ann)
                else:
                    # New annotation -> add
                    existing_map[ann_id] = ann
                    new_count += 1

            # Convert back to list
            merged_annotations = list(existing_map.values())
            annotation_data['slice_annotations'] = merged_annotations

            # Merge/replace VLM labels conservatively
            if vlm_labels:
                vlm_existing = annotation_data.get('vlm_labels', {}) or {}
                # Update or add keys
                for k, v in vlm_labels.items():
                    vlm_existing[k] = v
                annotation_data['vlm_labels'] = vlm_existing

            # Save to file
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Saved annotations for user '{user_id}' and study '{os.path.basename(study_path)}' "
                f"(added {new_count} new, total {len(annotation_data.get('slice_annotations', []))} annotations, type: {annotation_type})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotations: {e}", exc_info=True)
            return False
    
    def load_annotations(self, user_id: str, study_path: str) -> Optional[Dict[str, Any]]:
        """
        Load annotations for a user and study
        
        Args:
            user_id: User ID
            study_path: Path to the study/dataset
            
        Returns:
            dict: Annotation data or None if not found
        """
        try:
            annotation_file = self._get_annotation_file(user_id, study_path)
            
            if not annotation_file.exists():
                logger.debug(f"No annotations found for user '{user_id}' and study '{os.path.basename(study_path)}'")
                return None
            
            # Check if file is empty
            if annotation_file.stat().st_size == 0:
                logger.warning(f"Annotation file exists but is empty: {annotation_file}")
                # Delete empty file and return None
                annotation_file.unlink()
                return None
            
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)
            
            logger.info(f"Loaded annotations for user '{user_id}' and study '{os.path.basename(study_path)}' "
                       f"({len(annotation_data.get('slice_annotations', []))} slices)")
            return annotation_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in annotation file: {e}")
            # Optionally delete corrupted file
            annotation_file = self._get_annotation_file(user_id, study_path)
            if annotation_file.exists():
                logger.warning(f"Deleting corrupted annotation file: {annotation_file}")
                annotation_file.unlink()
            return None
        except Exception as e:
            logger.error(f"Error loading annotations: {e}", exc_info=True)
            return None
    
    def get_slice_annotations(
        self, 
        user_id: str, 
        study_path: str, 
        slice_idx: int,
        view_type: str = "axial"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get annotations for a specific slice
        
        Args:
            user_id: User ID
            study_path: Path to the study/dataset
            slice_idx: Slice index
            view_type: View type (axial, sagittal, coronal)
            
        Returns:
            list: List of annotations for the slice or None if not found
        """
        try:
            annotation_data = self.load_annotations(user_id, study_path)
            
            if not annotation_data or 'slice_annotations' not in annotation_data:
                return None
            
            # Filter annotations for the specific slice and view
            slice_annotations = [
                ann for ann in annotation_data['slice_annotations']
                if ann.get('slice_idx') == slice_idx and ann.get('view_type', 'axial') == view_type
            ]
            
            return slice_annotations if slice_annotations else None
            
        except Exception as e:
            logger.error(f"Error getting slice annotations: {e}", exc_info=True)
            return None
    
    def update_vlm_labels(
        self,
        user_id: str,
        study_path: str,
        slice_idx: int,
        annotation_id: str,
        suggested_labels: Optional[List[str]] = None,
        accepted_label: Optional[str] = None
    ) -> bool:
        """
        Update VLM labels for a specific annotation
        
        Args:
            user_id: User ID
            study_path: Path to the study/dataset
            slice_idx: Slice index
            annotation_id: Unique annotation ID
            suggested_labels: List of VLM-suggested labels
            accepted_label: User-accepted label
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            annotation_data = self.load_annotations(user_id, study_path)
            
            if not annotation_data:
                logger.warning("No annotation data found to update VLM labels")
                return False
            
            # Initialize vlm_labels structure if it doesn't exist
            if 'vlm_labels' not in annotation_data:
                annotation_data['vlm_labels'] = {}
            
            # Create unique key for this annotation
            label_key = f"slice_{slice_idx}_{annotation_id}"
            
            if label_key not in annotation_data['vlm_labels']:
                annotation_data['vlm_labels'][label_key] = {
                    'slice_idx': slice_idx,
                    'annotation_id': annotation_id,
                    'suggested_labels': [],
                    'accepted_label': None,
                    'label_history': []
                }
            
            # Update suggested labels
            if suggested_labels:
                annotation_data['vlm_labels'][label_key]['suggested_labels'] = suggested_labels
                annotation_data['vlm_labels'][label_key]['last_suggestion_time'] = datetime.now().isoformat()
            
            # Update accepted label
            if accepted_label:
                # Store history of label changes
                if annotation_data['vlm_labels'][label_key]['accepted_label']:
                    annotation_data['vlm_labels'][label_key]['label_history'].append({
                        'previous_label': annotation_data['vlm_labels'][label_key]['accepted_label'],
                        'new_label': accepted_label,
                        'changed_at': datetime.now().isoformat()
                    })
                
                annotation_data['vlm_labels'][label_key]['accepted_label'] = accepted_label
                annotation_data['vlm_labels'][label_key]['accepted_at'] = datetime.now().isoformat()
            
            # Save updated data
            annotation_file = self._get_annotation_file(user_id, study_path)
            annotation_data['modified_at'] = datetime.now().isoformat()
            
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated VLM labels for annotation '{annotation_id}' on slice {slice_idx}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating VLM labels: {e}", exc_info=True)
            return False
    
    def delete_slice_annotations(
        self,
        user_id: str,
        study_path: str,
        slice_idx: int,
        view_type: str = "axial"
    ) -> bool:
        """
        Delete all annotations for a specific slice
        
        Args:
            user_id: User ID
            study_path: Path to the study/dataset
            slice_idx: Slice index
            view_type: View type
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            annotation_data = self.load_annotations(user_id, study_path)
            
            if not annotation_data:
                return True  # Nothing to delete
            
            # Filter out annotations for the specific slice
            if 'slice_annotations' in annotation_data:
                annotation_data['slice_annotations'] = [
                    ann for ann in annotation_data['slice_annotations']
                    if not (ann.get('slice_idx') == slice_idx and ann.get('view_type', 'axial') == view_type)
                ]
            
            # Save updated data
            annotation_file = self._get_annotation_file(user_id, study_path)
            annotation_data['modified_at'] = datetime.now().isoformat()
            
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Deleted annotations for slice {slice_idx} ({view_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting slice annotations: {e}", exc_info=True)
            return False
    
    def get_all_user_annotations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all annotations for a specific user across all studies
        
        Args:
            user_id: User ID
            
        Returns:
            list: List of all annotation records for the user
        """
        try:
            user_dir = self.base_dir / user_id
            
            if not user_dir.exists():
                logger.debug(f"No annotations found for user '{user_id}'")
                return []
            
            all_annotations = []
            
            # Iterate through all study directories
            for study_dir in user_dir.iterdir():
                if study_dir.is_dir():
                    annotation_file = study_dir / "annotations.json"
                    if annotation_file.exists():
                        try:
                            with open(annotation_file, 'r', encoding='utf-8') as f:
                                annotation_data = json.load(f)
                                all_annotations.append(annotation_data)
                        except Exception as e:
                            logger.error(f"Error loading annotation file {annotation_file}: {e}")
            
            logger.info(f"Found {len(all_annotations)} annotation records for user '{user_id}'")
            return all_annotations
            
        except Exception as e:
            logger.error(f"Error getting all user annotations: {e}", exc_info=True)
            return []
    
    def export_annotations(
        self,
        user_id: str,
        study_path: str,
        output_format: str = "json"
    ) -> Optional[str]:
        """
        Export annotations in various formats
        
        Args:
            user_id: User ID
            study_path: Path to the study/dataset
            output_format: Export format ('json', 'csv', 'coco')
            
        Returns:
            str: Path to exported file or None if failed
        """
        try:
            annotation_data = self.load_annotations(user_id, study_path)
            
            if not annotation_data:
                logger.warning("No annotations to export")
                return None
            
            # For now, only JSON export is implemented
            if output_format == "json":
                annotation_file = self._get_annotation_file(user_id, study_path)
                return str(annotation_file)
            else:
                logger.warning(f"Export format '{output_format}' not yet implemented")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting annotations: {e}", exc_info=True)
            return None


# Global instance
_annotation_manager = None

def get_annotation_manager() -> AnnotationManager:
    """Get the global annotation manager instance"""
    global _annotation_manager
    if _annotation_manager is None:
        _annotation_manager = AnnotationManager()
    return _annotation_manager
