"""
SegMed-Pro Export Handlers

This module contains all the export-related functionality for the viewer tab,
including NIfTI, JSON, YOLO, and CSV export formats with optional overlay capture.
"""

import os
import json
import csv
import numpy as np
import nibabel as nib
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw

# Add utils to path for imports
import sys
utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils")
if utils_path not in sys.path:
    sys.path.append(utils_path)

from utils.visualization import display_slice, overlay_segmentation, make_image_for_gradio

logger = logging.getLogger(__name__)


class ExportHandlers:
    """Handlers for various export operations from the viewer tab"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
    
    def capture_overlay_image(self, slice_idx: int, view: str, annotator_value: dict = None) -> Optional[np.ndarray]:
        """
        Capture the current view with overlays as displayed in image_annotator, using annotator_value if provided.
        
        Args:
            slice_idx: The slice index to capture
            view: The view orientation (axial, sagittal, coronal)
            annotator_value: Current state of image_annotator component, if available
            
        Returns:
            np.ndarray: RGB image array with overlays, or None if failed
        """
        try:
            if self.state.current_data is None:
                logger.warning("No data loaded for overlay capture")
                return None
            
            # Get the base slice image
            img = display_slice(
                self.state.current_data,
                slice_idx,
                view,
                window_level=getattr(self.state, 'current_window_center', 500),
                window_width=getattr(self.state, 'current_window_width', 1000),
                crosshair=None,
                add_orientation_marker=False
            )            # FOR OVERLAY PNG EXPORT: Show current state combining user edits + original data
            # Priority 1: Apply user annotations from image_annotator (current slice) or stored user annotations
            user_annotations_applied = False
            
            # Check current annotator value for current slice
            if (annotator_value and 'boxes' in annotator_value and 
                slice_idx == self.state.current_slice_idx and len(annotator_value['boxes']) > 0):
                img = self._draw_annotations_on_image(img, annotator_value['boxes'])
                user_annotations_applied = True
                logger.info(f"Applied current image_annotator annotations to overlay for slice {slice_idx}")
            
            # Check for stored user annotations for any slice
            elif (hasattr(self.state, 'image_handlers') and self.state.image_handlers and
                  hasattr(self.state.image_handlers, 'user_annotations') and 
                  slice_idx in self.state.image_handlers.user_annotations):
                # This slice has stored user annotations, extract and draw them
                user_anns = self.state.image_handlers.user_annotations[slice_idx]
                if user_anns:  # If there are user annotations for this slice
                    boxes_to_draw = []
                    for ann in user_anns:
                        if isinstance(ann, dict) and 'data' in ann:
                            ann_data = ann['data']
                            # Convert stored annotation to box format for drawing
                            box = self._convert_stored_annotation_to_box(ann_data)
                            if box:
                                boxes_to_draw.append(box)
                    
                    if boxes_to_draw:
                        img = self._draw_annotations_on_image(img, boxes_to_draw)
                        user_annotations_applied = True
                        logger.info(f"Applied stored user annotations to overlay for slice {slice_idx}")
            
            # Priority 2: If no user annotations for this slice, apply original segmentation overlay
            if not user_annotations_applied:
                if (hasattr(self.state, 'segmentation_loaded') and 
                    self.state.segmentation_loaded and 
                    hasattr(self.state, 'segmentation_data') and 
                    self.state.segmentation_data is not None):
                    try:
                        seg_slice = self.state.get_segmentation_slice(view, slice_idx)
                        if seg_slice is not None:
                            img = overlay_segmentation(
                                img, 
                                seg_slice, 
                                alpha=getattr(self.state, 'segmentation_alpha', 0.5),
                                colormap=getattr(self.state, 'segmentation_colormap', None)
                            )
                            logger.info(f"Applied original segmentation overlay to slice {slice_idx} (no user edits)")
                    except Exception as e:
                        logger.error(f"Error applying segmentation overlay: {str(e)}")
                
                # Priority 3: If no user annotations and no segmentation, apply MEDSAM2 overlays
                if (hasattr(self.state, 'medsam2_handlers') and 
                    self.state.medsam2_handlers and
                    hasattr(self.state.medsam2_handlers, 'annotation_overlays')):
                    try:
                        overlays = self.state.medsam2_handlers.annotation_overlays
                        if slice_idx in overlays:
                            overlay_data = overlays[slice_idx]
                            if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                                # Single mask format
                                mask_array = overlay_data['mask']
                                img = overlay_segmentation(
                                    img,
                                    mask_array,
                                    alpha=0.4,
                                    colormap={1: [255, 0, 0]}  # Red overlay for MEDSAM2
                                )
                            else:
                                # Multiple annotations format
                                for annotation_id, annotation_data in overlay_data.items():
                                    if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                                        mask_array = annotation_data['mask']
                                        img = overlay_segmentation(
                                            img,
                                            mask_array,
                                            alpha=0.4,
                                            colormap={1: [255, 0, 0]}
                                        )
                            logger.info(f"Applied MEDSAM2 annotation overlays to slice {slice_idx} (no user edits)")
                    except Exception as e:
                        logger.error(f"Error applying MEDSAM2 overlays: {str(e)}")
            else:
                logger.info(f"User annotations present for slice {slice_idx}, skipping original overlays")
            
            # Convert to RGB format
            if len(img.shape) == 2:
                img_rgb = np.stack([img] * 3, axis=-1)
            else:
                img_rgb = img
            
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            
            return img_rgb
            
        except Exception as e:
            logger.error(f"Error capturing overlay image: {str(e)}")
            return None
    
    def collect_annotation_data(self, slice_idx: int = None, current_annotator_value: dict = None) -> Dict[str, Any]:
        """
        Collect all annotation data for export
        
        Args:
            slice_idx: Specific slice index, or None for all slices
            current_annotator_value: Current state of image_annotator component
            
        Returns:
            Dict containing annotation data with slice info, labels, colors, and coordinates
        """
        annotations = {}
        
        # Determine slice range
        if slice_idx is not None:
            slice_range = [slice_idx]
        else:
            max_slice = self.state.get_max_slice_for_view(self.state.current_view)
            slice_range = range(max_slice + 1)
        
        for s_idx in slice_range:
            slice_annotations = []
            
            # Priority 1: If we have current annotator value for the current slice, use it ONLY
            if (current_annotator_value is not None and 
                slice_idx is not None and s_idx == self.state.current_slice_idx):
                # Extract annotations from current image_annotator state (this is the most up-to-date)
                boxes = current_annotator_value.get("boxes", [])
                for box in boxes:
                    annotation_info = self._extract_annotation_from_box(box, s_idx)
                    if annotation_info:
                        slice_annotations.append(annotation_info)
                # Skip other sources for this slice to avoid duplicates
            else:
                # Priority 2: Get user annotations from stored data (for other slices)
                if hasattr(self.state, 'image_handlers') and self.state.image_handlers:
                    handlers = self.state.image_handlers
                    if (hasattr(handlers, 'user_annotations') and 
                        s_idx in handlers.user_annotations):
                        user_anns = handlers.user_annotations[s_idx]
                        for ann in user_anns:
                            if isinstance(ann, dict) and 'data' in ann:
                                ann_data = ann['data']
                                # Extract coordinates and metadata
                                annotation_info = self._extract_annotation_info(ann_data, s_idx)
                                if annotation_info:
                                    slice_annotations.append(annotation_info)
                
                # Priority 3: Add segmentation data only if no user annotations exist
                if not slice_annotations:
                    if (hasattr(self.state, 'segmentation_loaded') and 
                        self.state.segmentation_loaded and 
                        hasattr(self.state, 'segmentation_data') and 
                        self.state.segmentation_data is not None):
                        try:
                            seg_slice = self.state.get_segmentation_slice(self.state.current_view, s_idx)
                            if seg_slice is not None:
                                # Extract segmentation annotations
                                seg_annotations = self._extract_segmentation_annotations(seg_slice, s_idx)
                                slice_annotations.extend(seg_annotations)
                        except Exception as e:
                            logger.error(f"Error extracting segmentation data for slice {s_idx}: {str(e)}")
                
                # Priority 4: Add MEDSAM2 annotations only if no user annotations exist
                if not slice_annotations:
                    if (hasattr(self.state, 'medsam2_handlers') and 
                        self.state.medsam2_handlers and
                        hasattr(self.state.medsam2_handlers, 'annotation_overlays')):
                        try:
                            overlays = self.state.medsam2_handlers.annotation_overlays
                            if s_idx in overlays:
                                medsam2_annotations = self._extract_medsam2_annotations(overlays[s_idx], s_idx)
                                slice_annotations.extend(medsam2_annotations)
                        except Exception as e:
                            logger.error(f"Error extracting MEDSAM2 data for slice {s_idx}: {str(e)}")
            
            if slice_annotations:
                annotations[s_idx] = slice_annotations
        
        return annotations
    
    def _extract_annotation_info(self, ann_data: Dict, slice_idx: int) -> Optional[Dict]:
        """Extract annotation information from image_annotator data"""
        try:
            annotation = {
                'slice_number': slice_idx,
                'label_name': ann_data.get('label', 'Unknown'),
                'label_color': self._get_label_color(ann_data.get('label', 'Unknown')),
                'type': ann_data.get('type', 'unknown')
            }
            
            # Extract coordinates based on annotation type
            if ann_data.get('type') == 'polygon' and 'points' in ann_data:
                coordinates = []
                for point in ann_data['points']:
                    coordinates.append([point['x'], point['y']])
                annotation['coordinates'] = coordinates
                annotation['format'] = 'polygon'
            elif 'xmin' in ann_data and 'ymin' in ann_data:
                # Bounding box format
                annotation['coordinates'] = [
                    [ann_data['xmin'], ann_data['ymin']],
                    [ann_data['xmax'], ann_data['ymax']]
                ]
                annotation['format'] = 'bbox'
            else:
                return None
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error extracting annotation info: {str(e)}")
            return None
    
    def _extract_segmentation_annotations(self, seg_slice: np.ndarray, slice_idx: int) -> List[Dict]:
        """Extract annotations from segmentation data"""
        annotations = []
        
        try:
            unique_labels = np.unique(seg_slice)
            for label in unique_labels:
                if label == 0:  # Skip background
                    continue
                
                mask = (seg_slice == label)
                if not np.any(mask):
                    continue
                
                # Find contours for this label
                import cv2
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if len(contour) >= 3:  # Need at least 3 points for a polygon
                        coordinates = []
                        for point in contour:
                            x, y = point[0]
                            coordinates.append([int(x), int(y)])
                        
                        annotation = {
                            'slice_number': slice_idx,
                            'label_name': self._get_label_name(label),
                            'label_color': self._get_segmentation_color(label),
                            'coordinates': coordinates,
                            'format': 'polygon',
                            'type': 'segmentation'
                        }
                        annotations.append(annotation)
                        
        except Exception as e:
            logger.error(f"Error extracting segmentation annotations: {str(e)}")
        
        return annotations
    
    def _extract_medsam2_annotations(self, overlay_data: Any, slice_idx: int) -> List[Dict]:
        """Extract annotations from MEDSAM2 overlay data"""
        annotations = []
        
        try:
            if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                # Single mask format
                mask = overlay_data['mask']
                annotation = self._mask_to_annotation(mask, slice_idx, 'MEDSAM2')
                if annotation:
                    annotations.append(annotation)
            else:
                # Multiple annotations format
                for annotation_id, annotation_data in overlay_data.items():
                    if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                        mask = annotation_data['mask']
                        annotation = self._mask_to_annotation(mask, slice_idx, f'MEDSAM2_{annotation_id}')
                        if annotation:
                            annotations.append(annotation)
                            
        except Exception as e:
            logger.error(f"Error extracting MEDSAM2 annotations: {str(e)}")
        
        return annotations
    
    def _mask_to_annotation(self, mask: np.ndarray, slice_idx: int, label_name: str) -> Optional[Dict]:
        """Convert a mask to annotation format"""
        try:
            import cv2
            
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                if len(largest_contour) >= 3:
                    coordinates = []
                    for point in largest_contour:
                        x, y = point[0]
                        coordinates.append([int(x), int(y)])
                    
                    return {
                        'slice_number': slice_idx,
                        'label_name': label_name,
                        'label_color': [255, 0, 0],  # Red for MEDSAM2
                        'coordinates': coordinates,
                        'format': 'polygon',
                        'type': 'medsam2'
                    }
            
        except Exception as e:
            logger.error(f"Error converting mask to annotation: {str(e)}")
        
        return None
    
    def _get_label_color(self, label_name: str) -> List[int]:
        """Get RGB color for a label name"""
        # Default color mapping
        color_map = {
            "Normal Tissue": [0, 255, 0],
            "Tumor": [255, 0, 0],
            "Organ": [0, 0, 255],
            "Lesion": [255, 255, 0],
            "ROI": [255, 0, 255],
            "Other": [0, 255, 255]
        }
        return color_map.get(label_name, [128, 128, 128])  # Default gray
    
    def _get_label_name(self, label_value: int) -> str:
        """Get label name from segmentation value"""
        if hasattr(self.state, 'segmentation_labelmap') and self.state.segmentation_labelmap:
            return self.state.segmentation_labelmap.get(label_value, f'Label_{label_value}')
        return f'Label_{label_value}'
    
    def _get_segmentation_color(self, label_value: int) -> List[int]:
        """Get RGB color for segmentation label"""
        if hasattr(self.state, 'segmentation_colormap') and self.state.segmentation_colormap:
            color = self.state.segmentation_colormap.get(label_value, [128, 128, 128])
            if isinstance(color, (list, tuple)) and len(color) >= 3:
                return [int(color[0]), int(color[1]), int(color[2])]
        
        # Default color scheme
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
            [255, 0, 255], [0, 255, 255], [255, 128, 0], [128, 255, 0]
        ]
        return colors[label_value % len(colors)]
    
    def export_nifti(self, output_dir: str, slice_idx: int = None, include_overlays: bool = False, annotator_value=None):
        """
        Export annotations in NIfTI format
        
        Args:
            output_dir: Output directory path
            slice_idx: Specific slice index, or None for all slices
            include_overlays: Whether to include overlay PNG images
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.state.current_data is None:
                return False, "No data loaded for export"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Collect annotation data
            annotations = self.collect_annotation_data(slice_idx, annotator_value)
            if not annotations:
                return False, "No annotations found to export"
            
            # Create a mask volume based on annotations
            data_shape = self.state.current_data.shape
            mask_volume = np.zeros(data_shape, dtype=np.uint8)
            
            for s_idx, slice_annotations in annotations.items():
                for ann in slice_annotations:
                    if ann['format'] == 'polygon':
                        # Convert polygon to mask
                        mask = self._polygon_to_mask(ann['coordinates'], data_shape[1:])
                        if mask is not None:
                            # Use label index based on label name
                            label_value = self._get_label_value_for_export(ann['label_name'])
                            mask_volume[s_idx][mask] = label_value
              # Create NIfTI file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if slice_idx is not None:
                nifti_filename = f"annotations_slice_{slice_idx:03d}_{timestamp}.nii"
                # For single slice, create a single-slice volume
                single_slice_volume = mask_volume[slice_idx:slice_idx+1]
                nifti_img = nib.Nifti1Image(single_slice_volume, np.eye(4))
            else:
                nifti_filename = f"annotations_volume_{timestamp}.nii"
                nifti_img = nib.Nifti1Image(mask_volume, np.eye(4))
            
            nifti_path = os.path.join(output_dir, nifti_filename)
            nib.save(nifti_img, nifti_path)
              # Export overlay images if requested
            overlay_message = ""
            if include_overlays:
                overlay_count = self._export_overlay_images(output_dir, slice_idx, timestamp, annotator_value)
                overlay_message = f" and {overlay_count} overlay PNG images"
            
            if slice_idx is not None:
                message = f"Successfully exported slice {slice_idx} annotations to {nifti_filename}{overlay_message}"
            else:
                total_slices = len(annotations)
                message = f"Successfully exported {total_slices} slices to {nifti_filename}{overlay_message}"
            
            return True, message
            
        except Exception as e:
            logger.error(f"Error exporting NIfTI: {str(e)}")
            return False, f"Export failed: {str(e)}"
    
    def export_json(self, output_dir: str, slice_idx: int = None, include_overlays: bool = False, annotator_value=None) -> Tuple[bool, str]:
        """Export annotations in JSON format"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            annotations = self.collect_annotation_data(slice_idx, annotator_value)
            if not annotations:
                return False, "No annotations found to export"
            
            # Create export data structure
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "view_type": self.state.current_view,
                    "data_source": getattr(self.state, 'current_file_path', "unknown"),
                    "format": "SegMed-Pro JSON v1.0"
                },
                "annotations": {}
            }
            
            # Convert annotations to export format
            for s_idx, slice_annotations in annotations.items():
                export_data["annotations"][str(s_idx)] = slice_annotations
            
            # Save JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if slice_idx is not None:
                json_filename = f"annotations_slice_{slice_idx:03d}_{timestamp}.json"
            else:
                json_filename = f"annotations_volume_{timestamp}.json"
            
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
              # Export overlay images if requested
            overlay_message = ""
            if include_overlays:
                overlay_count = self._export_overlay_images(output_dir, slice_idx, timestamp, annotator_value)
                overlay_message = f" and {overlay_count} overlay PNG images"
            
            if slice_idx is not None:
                message = f"Successfully exported slice {slice_idx} annotations to {json_filename}{overlay_message}"
            else:
                total_annotations = sum(len(anns) for anns in annotations.values())
                message = f"Successfully exported {total_annotations} annotations from {len(annotations)} slices to {json_filename}{overlay_message}"
            
            return True, message
            
        except Exception as e:
            logger.error(f"Error exporting JSON: {str(e)}")
            return False, f"Export failed: {str(e)}"
    
    def export_yolo(self, output_dir: str, slice_idx: int = None, include_overlays: bool = False, annotator_value=None) -> Tuple[bool, str]:
        """Export annotations in YOLO format"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            annotations = self.collect_annotation_data(slice_idx, annotator_value)
            if not annotations:
                return False, "No annotations found to export"
            
            # Create class mapping
            all_labels = set()
            for slice_annotations in annotations.values():
                for ann in slice_annotations:
                    all_labels.add(ann['label_name'])
            
            class_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
            
            # Save class mapping
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            classes_file = os.path.join(output_dir, f"classes_{timestamp}.txt")
            with open(classes_file, 'w') as f:
                for label in sorted(all_labels):
                    f.write(f"{label}\n")
            
            exported_files = []
            
            # Export YOLO format files
            for s_idx, slice_annotations in annotations.items():
                if slice_idx is not None and s_idx != slice_idx:
                    continue
                
                yolo_filename = f"slice_{s_idx:03d}_{timestamp}.txt"
                yolo_path = os.path.join(output_dir, yolo_filename)
                
                with open(yolo_path, 'w') as f:
                    for ann in slice_annotations:
                        if ann['format'] == 'bbox' and len(ann['coordinates']) == 2:
                            # Convert bbox to YOLO format
                            x1, y1 = ann['coordinates'][0]
                            x2, y2 = ann['coordinates'][1]
                            
                            # Normalize coordinates (assuming image size from data shape)
                            img_width = self.state.current_data.shape[2]
                            img_height = self.state.current_data.shape[1]
                            
                            center_x = (x1 + x2) / 2 / img_width
                            center_y = (y1 + y2) / 2 / img_height
                            width = abs(x2 - x1) / img_width
                            height = abs(y2 - y1) / img_height
                            
                            class_id = class_mapping[ann['label_name']]
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        
                        elif ann['format'] == 'polygon':
                            # Convert polygon to bounding box for YOLO
                            coords = np.array(ann['coordinates'])
                            x1, y1 = coords.min(axis=0)
                            x2, y2 = coords.max(axis=0)
                            
                            img_width = self.state.current_data.shape[2]
                            img_height = self.state.current_data.shape[1]
                            
                            center_x = (x1 + x2) / 2 / img_width
                            center_y = (y1 + y2) / 2 / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height
                            
                            class_id = class_mapping[ann['label_name']]
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                
                exported_files.append(yolo_filename)
              # Export overlay images if requested
            overlay_message = ""
            if include_overlays:
                overlay_count = self._export_overlay_images(output_dir, slice_idx, timestamp, annotator_value)
                overlay_message = f" and {overlay_count} overlay PNG images"
            
            message = f"Successfully exported {len(exported_files)} YOLO files and classes file{overlay_message}"
            return True, message
            
        except Exception as e:
            logger.error(f"Error exporting YOLO: {str(e)}")
            return False, f"Export failed: {str(e)}"
    
    def export_csv(self, output_dir: str, slice_idx: int = None, include_overlays: bool = False, annotator_value=None) -> Tuple[bool, str]:
        """Export annotations in CSV format"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            annotations = self.collect_annotation_data(slice_idx, annotator_value)
            if not annotations:
                return False, "No annotations found to export"
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if slice_idx is not None:
                csv_filename = f"annotations_slice_{slice_idx:03d}_{timestamp}.csv"
            else:
                csv_filename = f"annotations_volume_{timestamp}.csv"
            
            csv_path = os.path.join(output_dir, csv_filename)
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'slice_number', 'label_name', 'label_color_r', 'label_color_g', 'label_color_b',
                    'coordinates', 'format', 'type', 'num_points'
                ])
                
                total_annotations = 0
                for s_idx, slice_annotations in annotations.items():
                    for ann in slice_annotations:
                        color = ann['label_color']
                        coords_str = ';'.join([f"{x},{y}" for x, y in ann['coordinates']])
                        
                        writer.writerow([
                            ann['slice_number'],
                            ann['label_name'],
                            color[0], color[1], color[2],
                            coords_str,
                            ann['format'],
                            ann['type'],
                            len(ann['coordinates'])
                        ])
                        total_annotations += 1
              # Export overlay images if requested
            overlay_message = ""
            if include_overlays:
                overlay_count = self._export_overlay_images(output_dir, slice_idx, timestamp, annotator_value)
                overlay_message = f" and {overlay_count} overlay PNG images"
            
            message = f"Successfully exported {total_annotations} annotations to {csv_filename}{overlay_message}"
            return True, message
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
            return False, f"Export failed: {str(e)}"
    
    def _export_overlay_images(self, output_dir: str, slice_idx: int = None, timestamp: str = None, current_annotator_value: dict = None) -> int:
        """Export overlay PNG images showing exactly what the user sees in image_annotator"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        overlay_dir = os.path.join(output_dir, f"overlays_{timestamp}")
        os.makedirs(overlay_dir, exist_ok=True)
        
        exported_count = 0        
        # Determine slice range
        if slice_idx is not None:
            slice_range = [slice_idx]
        else:
            max_slice = self.state.get_max_slice_for_view(self.state.current_view)
            slice_range = range(max_slice + 1)
        
        for s_idx in slice_range:
            # For single slice export, ALWAYS use the current annotator value for that slice
            # For all slices export, only use annotator value for the current slice
            annotator_value_for_slice = None
            if current_annotator_value is not None:
                if slice_idx is not None:
                    # Single slice export - always use the current annotator value
                    annotator_value_for_slice = current_annotator_value
                elif s_idx == self.state.current_slice_idx:
                    # All slices export - only use for current slice
                    annotator_value_for_slice = current_annotator_value
            
            overlay_image = self.capture_overlay_image(s_idx, self.state.current_view, annotator_value_for_slice)
            if overlay_image is not None:
                png_filename = f"slice_{s_idx:03d}_overlay.png"
                png_path = os.path.join(overlay_dir, png_filename)
                
                # Convert to PIL and save
                pil_image = Image.fromarray(overlay_image)
                pil_image.save(png_path)
                exported_count += 1
        
        return exported_count
    
    def _polygon_to_mask(self, coordinates: List[List[int]], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Convert polygon coordinates to binary mask"""
        try:
            import cv2
            
            mask = np.zeros(image_shape, dtype=np.uint8)
            pts = np.array(coordinates, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
            return mask.astype(bool)
            
        except Exception as e:
            logger.error(f"Error converting polygon to mask: {str(e)}")
            return None
    
    def _get_label_value_for_export(self, label_name: str) -> int:
        """Get numerical label value for export"""
        # Create a mapping of label names to values
        label_mapping = {
            "Normal Tissue": 1,
            "Tumor": 2,
            "Organ": 3,
            "Lesion": 4,
            "ROI": 5,
            "Other": 6,
            "MEDSAM2": 7
        }
        
        # Check for MEDSAM2 variants
        if label_name.startswith("MEDSAM2"):
            return 7
        
        return label_mapping.get(label_name, 8)  # Default value for unknown labels
    
    def export_single_slice(self, export_format: str, output_dir: str, include_overlays: bool, current_annotator_value: dict = None) -> Tuple[bool, str]:
        """Export annotations for the current slice, using the current image_annotator value if provided."""
        current_slice = self.state.current_slice_idx
        if export_format == "NIfTI (*.nii)":
            return self.export_nifti(output_dir, current_slice, include_overlays, current_annotator_value)
        elif export_format == "JSON":
            return self.export_json(output_dir, current_slice, include_overlays, current_annotator_value)
        elif export_format == "YOLO":
            return self.export_yolo(output_dir, current_slice, include_overlays, current_annotator_value)
        elif export_format == "CSV":
            return self.export_csv(output_dir, current_slice, include_overlays, current_annotator_value)
        else:
            return False, f"Unsupported export format: {export_format}"

    def export_all_slices(self, export_format: str, output_dir: str, include_overlays: bool, all_annotator_values: dict = None) -> Tuple[bool, str]:
        """Export annotations for all slices, using the current image_annotator values if provided."""
        if export_format == "NIfTI (*.nii)":
            return self.export_nifti(output_dir, None, include_overlays, all_annotator_values)
        elif export_format == "JSON":
            return self.export_json(output_dir, None, include_overlays, all_annotator_values)
        elif export_format == "YOLO":
            return self.export_yolo(output_dir, None, include_overlays, all_annotator_values)
        elif export_format == "CSV":
            return self.export_csv(output_dir, None, include_overlays, all_annotator_values)
        else:
            return False, f"Unsupported export format: {export_format}"

    def export_nifti(self, output_dir, slice_idx=None, include_overlays=False, annotator_value=None):
        """Export segmentation as NIfTI (.nii) file, using current annotation state."""
        try:
            if self.state.current_data is None:
                return False, "No data loaded for export"
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Collect annotation data
            annotations = self.collect_annotation_data(slice_idx, annotator_value)
            if not annotations:
                return False, "No annotations found to export"
            
            # Create a mask volume based on annotations
            data_shape = self.state.current_data.shape
            mask_volume = np.zeros(data_shape, dtype=np.uint8)
            
            for s_idx, slice_annotations in annotations.items():
                for ann in slice_annotations:
                    if ann['format'] == 'polygon':
                        # Convert polygon to mask
                        mask = self._polygon_to_mask(ann['coordinates'], data_shape[1:])
                        if mask is not None:
                            # Use label index based on label name
                            label_value = self._get_label_value_for_export(ann['label_name'])
                            mask_volume[s_idx][mask] = label_value            
            # Create NIfTI file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if slice_idx is not None:
                nifti_filename = f"annotations_slice_{slice_idx:03d}_{timestamp}.nii"
                # For single slice, create a single-slice volume
                single_slice_volume = mask_volume[slice_idx:slice_idx+1]
                nifti_img = nib.Nifti1Image(single_slice_volume, np.eye(4))
            else:
                nifti_filename = f"annotations_volume_{timestamp}.nii"
                nifti_img = nib.Nifti1Image(mask_volume, np.eye(4))
            
            nifti_path = os.path.join(output_dir, nifti_filename)
            nib.save(nifti_img, nifti_path)
            
            return True, f"NIfTI exported to {nifti_path}"
        except Exception as e:
            logger.error(f"Error exporting NIfTI: {str(e)}")
            return False, str(e)
    
    def _extract_annotation_from_box(self, box: dict, slice_idx: int) -> Optional[Dict]:
        """Extract annotation information from image_annotator box data"""
        try:
            annotation = {
                'slice_number': slice_idx,
                'label_name': box.get('label', 'Unknown'),
                'label_color': box.get('color', [255, 0, 0]),
                'type': box.get('type', 'unknown')
            }
            
            # Extract coordinates based on annotation type
            if box.get('type') == 'polygon' and 'points' in box:
                coordinates = []
                for point in box['points']:
                    coordinates.append([point['x'], point['y']])
                annotation['coordinates'] = coordinates
                annotation['format'] = 'polygon'
            elif all(k in box for k in ('xmin', 'ymin', 'xmax', 'ymax')):
                # Bounding box format
                annotation['coordinates'] = [
                    [box['xmin'], box['ymin']],
                    [box['xmax'], box['ymax']]
                ]
                annotation['format'] = 'bbox'
            else:
                return None
            
            return annotation
            
        except Exception as e:
            logger.error(f"Error extracting annotation from box: {str(e)}")
            return None

    def _draw_annotations_on_image(self, img, boxes):
        """Draw annotation boxes/polygons on the image (RGB np.ndarray) FILLED with label color for overlay export"""
        pil_img = Image.fromarray(img)
        
        # Create a transparent overlay layer for filled shapes
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Also draw on the base image for outlines
        base_draw = ImageDraw.Draw(pil_img)
        
        for box in boxes:
            color = tuple(box.get('color', (255, 0, 0)))
            # Create semi-transparent fill color (50% opacity)
            fill_color = color + (128,)  # Add alpha channel
            
            if box.get('type') == 'polygon' and 'points' in box:
                # Draw FILLED polygon for overlay export
                points = [(p['x'], p['y']) for p in box['points']]
                # Fill with semi-transparent color on overlay
                overlay_draw.polygon(points, fill=fill_color)
                # Draw outline on base image for visibility
                base_draw.polygon(points, outline=color, width=2)
                
            elif all(k in box for k in ('xmin','ymin','xmax','ymax')):
                # Draw FILLED rectangle for overlay export
                coords = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                # Fill with semi-transparent color on overlay
                overlay_draw.rectangle(coords, fill=fill_color)
                # Draw outline on base image for visibility  
                base_draw.rectangle(coords, outline=color, width=2)
          # Composite the overlay onto the base image
        pil_img = pil_img.convert('RGBA')
        composite = Image.alpha_composite(pil_img, overlay)
        return np.array(composite.convert('RGB'))
    
    def _convert_stored_annotation_to_box(self, ann_data: dict) -> Optional[dict]:
        """Convert stored annotation data to box format for drawing, prioritizing polygons"""
        try:
            box = {
                'label': ann_data.get('label', 'Unknown'),
                'color': self._get_label_color(ann_data.get('label', 'Unknown')),
                'type': ann_data.get('type', 'unknown')
            }
            
            # Convert coordinates based on annotation type - prioritize polygons
            if ann_data.get('type') == 'polygon' and 'points' in ann_data:
                # Polygon format (preferred for medical imaging)
                box['type'] = 'polygon'
                box['points'] = ann_data['points']  # Already in correct format
                
            elif 'xmin' in ann_data and 'ymin' in ann_data and 'xmax' in ann_data and 'ymax' in ann_data:
                # Convert bounding box to polygon for better visualization
                box['type'] = 'polygon'
                box['points'] = [
                    {'x': ann_data['xmin'], 'y': ann_data['ymin']},  # Top-left
                    {'x': ann_data['xmax'], 'y': ann_data['ymin']},  # Top-right
                    {'x': ann_data['xmax'], 'y': ann_data['ymax']},  # Bottom-right
                    {'x': ann_data['xmin'], 'y': ann_data['ymax']}   # Bottom-left
                ]
                
            elif ann_data.get('type') == 'rectangle' and all(k in ann_data for k in ('xmin', 'ymin', 'xmax', 'ymax')):
                # Convert rectangle to polygon for consistency
                box['type'] = 'polygon'
                box['points'] = [
                    {'x': ann_data['xmin'], 'y': ann_data['ymin']},
                    {'x': ann_data['xmax'], 'y': ann_data['ymin']},
                    {'x': ann_data['xmax'], 'y': ann_data['ymax']},
                    {'x': ann_data['xmin'], 'y': ann_data['ymax']}
                ]
                
            else:
                logger.warning(f"Unsupported annotation format: {ann_data.get('type', 'unknown')}")
                return None
            
            return box
            
        except Exception as e:
            logger.error(f"Error converting stored annotation to box: {str(e)}")
            return None
