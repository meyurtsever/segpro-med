"""
SegMed-Pro Custom Annotator Handlers

This module contains the event handlers for the custom image annotation tab,
managing interactions with the gradio-image-annotation plugin and medical data integration.
"""

import gradio as gr
import numpy as np
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)


class CustomAnnotatorHandlers:
    """Handles all events for the custom annotator tab"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self.current_labels = ["Tumor", "Normal Tissue", "Organ", "Lesion", "ROI", "Other"]
        self.label_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
    def load_medical_slice_to_annotator(self) -> Optional[Dict[str, Any]]:
        """Load current medical slice into the annotator plugin"""
        try:
            if not hasattr(self.state, 'current_data') or self.state.current_data is None:
                logger.warning("No medical data loaded")
                return None
                
            # Get current slice and view
            current_slice = getattr(self.state, 'current_slice', 0)
            current_view = getattr(self.state, 'current_view', 'axial')
            
            # Get image data using the state's method if available
            if hasattr(self.state, 'get_image_for_display'):
                image_array = self.state.get_image_for_display(current_slice, current_view)
            else:
                # Fallback method
                data = self.state.current_data
                if len(data.shape) == 3:
                    if current_view.lower() == 'axial':
                        image_array = data[:, :, current_slice]
                    elif current_view.lower() == 'sagittal':
                        image_array = data[current_slice, :, :]
                    elif current_view.lower() == 'coronal':
                        image_array = data[:, current_slice, :]
                    else:
                        image_array = data[:, :, current_slice]
                else:
                    image_array = data
            
            # Convert to proper format for image_annotator
            # The plugin expects RGB numpy array with values 0-255
            if image_array.dtype != np.uint8:
                # Apply windowing if available
                if hasattr(self.state, 'window_level') and hasattr(self.state, 'window_width'):
                    window_level = getattr(self.state, 'window_level', 500)
                    window_width = getattr(self.state, 'window_width', 1000)
                    
                    # Apply windowing
                    window_min = window_level - window_width // 2
                    window_max = window_level + window_width // 2
                    
                    # Clip and normalize
                    windowed = np.clip(image_array, window_min, window_max)
                    image_normalized = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
                else:
                    # Simple normalization
                    image_min = np.min(image_array)
                    image_max = np.max(image_array)
                    if image_max > image_min:
                        image_normalized = ((image_array - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                    else:
                        image_normalized = np.zeros_like(image_array, dtype=np.uint8)
            else:
                image_normalized = image_array
            
            # Convert grayscale to RGB if needed
            if len(image_normalized.shape) == 2:
                image_rgb = np.stack([image_normalized] * 3, axis=-1)
            else:
                image_rgb = image_normalized
                
            # Create the annotation value for the plugin
            annotation_value = {
                "image": image_rgb,
                "boxes": []  # Start with no annotations
            }
            
            logger.info(f"Loaded medical slice {current_slice} ({current_view}) with shape: {image_rgb.shape}")
            return annotation_value
            
        except Exception as e:
            logger.error(f"Error loading medical slice to annotator: {e}")
            return None
    
    def update_annotation_settings(self, opacity: float, thickness: int, min_size: int) -> str:
        """Update annotation display settings"""
        try:
            # Note: The image_annotator plugin doesn't support dynamic setting updates
            # These would need to be applied when creating a new annotator instance
            logger.info(f"Settings updated - Opacity: {opacity}, Thickness: {thickness}, Min Size: {min_size}")
            return f"Settings noted - Opacity: {opacity}, Thickness: {thickness}, Min Size: {min_size}\\nNote: Apply these settings by reloading the image."
        except Exception as e:
            logger.error(f"Error updating annotation settings: {e}")
            return f"Error updating settings: {str(e)}"    
        
    def add_new_label(self, new_label: str) -> list:
        """Add a new label to the label list"""
        try:
            if new_label and new_label.strip():
                label = new_label.strip()
                if label not in self.current_labels:
                    self.current_labels.append(label)
                    # Add a random color for the new label
                    import random
                    new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    self.label_colors.append(new_color)
                    logger.info(f"Added new label: {label}")
                else:
                    logger.warning(f"Label '{label}' already exists")
            return self.current_labels
        except Exception as e:
            logger.error(f"Error adding new label: {e}")
            return self.current_labels

    def calculate_annotation_stats(self, annotation_data: Dict[str, Any]) -> dict:
        """Calculate statistics from annotation data"""
        try:
            if not annotation_data or 'boxes' not in annotation_data:
                return {"total_boxes": 0, "labels_used": {}}
            boxes = annotation_data['boxes']
            total_boxes = len(boxes)
            # Count labels
            label_counts = {}
            for box in boxes:
                label = box.get('label', 'Unknown')
                label_counts[label] = label_counts.get(label, 0) + 1
            # Calculate areas
            total_area = 0
            for box in boxes:
                width = box['xmax'] - box['xmin']
                height = box['ymax'] - box['ymin']
                area = width * height
                total_area += area
            stats = {
                "total_boxes": total_boxes,
                "labels_used": label_counts,
                "total_annotated_area": total_area,
                "average_box_area": total_area / total_boxes if total_boxes > 0 else 0
            }
            logger.info(f"Calculated annotation statistics: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error calculating annotation stats: {e}")
            return {"total_boxes": 0, "labels_used": {}, "error": str(e)}
    
    def export_annotations(self, annotation_data: Dict[str, Any], format_type: str) -> Tuple[str, str]:
        """Export annotations in the specified format"""
        try:
            if not annotation_data or 'boxes' not in annotation_data:
                return None, "No annotation data to export"
            
            boxes = annotation_data['boxes']
            
            if format_type == "JSON":
                # Export as JSON
                export_data = {
                    "image_info": {
                        "format": "medical_annotation",
                        "timestamp": str(np.datetime64('now')),
                        "total_boxes": len(boxes)
                    },
                    "annotations": boxes
                }
                
                # Create temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(export_data, f, indent=2)
                    temp_path = f.name
                
                return temp_path, f"Exported {len(boxes)} annotations as JSON"
                
            elif format_type == "CSV":
                # Export as CSV
                import csv
                import tempfile
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['label', 'xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'area'])
                    
                    for box in boxes:
                        width = box['xmax'] - box['xmin']
                        height = box['ymax'] - box['ymin']
                        area = width * height
                        writer.writerow([
                            box.get('label', 'Unknown'),
                            box['xmin'], box['ymin'], box['xmax'], box['ymax'],
                            width, height, area
                        ])
                    
                    temp_path = f.name
                
                return temp_path, f"Exported {len(boxes)} annotations as CSV"
                
            elif format_type == "COCO":
                # Export in COCO format
                coco_data = {
                    "info": {
                        "description": "Medical Image Annotations",
                        "version": "1.0",
                        "contributor": "SegMed-Pro"
                    },
                    "images": [{
                        "id": 1,
                        "width": 512,  # Default, should be actual image width
                        "height": 512, # Default, should be actual image height
                        "file_name": "medical_image.jpg"
                    }],
                    "categories": [
                        {"id": i+1, "name": label} 
                        for i, label in enumerate(self.current_labels)
                    ],
                    "annotations": []
                }
                
                for i, box in enumerate(boxes):
                    label = box.get('label', 'Unknown')
                    category_id = self.current_labels.index(label) + 1 if label in self.current_labels else 1
                    
                    width = box['xmax'] - box['xmin']
                    height = box['ymax'] - box['ymin']
                    
                    annotation = {
                        "id": i + 1,
                        "image_id": 1,
                        "category_id": category_id,
                        "bbox": [box['xmin'], box['ymin'], width, height],
                        "area": width * height,
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation)
                
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(coco_data, f, indent=2)
                    temp_path = f.name
                
                return temp_path, f"Exported {len(boxes)} annotations in COCO format"
            
            else:
                return None, f"Unsupported export format: {format_type}"
                
        except Exception as e:
            logger.error(f"Error exporting annotations: {e}")
            return None, f"Export failed: {str(e)}"
    def process_batch_images(self) -> str:
        """Process multiple images for batch annotation"""
        try:
            # This is a placeholder for batch processing functionality
            # In a real implementation, this would handle multiple images
            logger.info("Batch processing initiated")
            return "Batch processing feature coming soon. This will allow processing multiple medical images at once."
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return f"Batch processing failed: {str(e)}"
    def load_data_for_annotator(self, file_obj, dir_path):
        """Load data for the custom annotator tab, matching Editor tab output order"""
        try:
            # Check if we have a file object or directory path
            if not file_obj and not dir_path:
                return None, "Please select a file or enter a directory path", {}, 500, 1000, 0, "0/0", []
                
            # Process file or directory
            if file_obj:
                # Process single file (like in handlers.py)
                path = file_obj.name if file_obj else None
                if not path or not os.path.exists(path):
                    return None, "File not found or invalid", {}, 500, 1000, 0, "0/0", []
                
                # Here you would load the data - in a real implementation,
                # this would call functions from utils directory
                # For now, just set it to the state for the annotator to use
                # This is a placeholder - in reality you'd need to load the actual data
                self.state.current_data = np.zeros((10, 10, 10))  # Placeholder data
                self.state.current_slice = 0
                self.state.current_view = 'axial'
                self.state.window_level = 500
                self.state.window_width = 1000
                
                file_list = [os.path.basename(path)]
            elif dir_path:
                # Process directory
                if not os.path.isdir(dir_path):
                    return None, f"Directory not found: {dir_path}", {}, 500, 1000, 0, "0/0", []
                
                # Get list of applicable files (this is a placeholder)
                import os
                file_list = [f for f in os.listdir(dir_path) 
                             if f.endswith(('.dcm', '.nii', '.nii.gz', '.mat'))]
                
                if not file_list:
                    return None, f"No applicable files found in {dir_path}", {}, 500, 1000, 0, "0/0", []
                
                # Load first file as a placeholder
                # In a real implementation, you'd load the DICOM series
                # This is a placeholder
                self.state.current_data = np.zeros((10, 10, 10))
                self.state.current_slice = 0
                self.state.current_view = 'axial'
                self.state.window_level = 500
                self.state.window_width = 1000
            
            # 1. Load the image and annotation value
            annotation_value = self.load_medical_slice_to_annotator()
            if annotation_value is None:
                # Create a placeholder value if we couldn't load real data
                # This prevents JSON decoding errors
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
                error = "Could not load medical data. Using placeholder."
            else:
                error = ""
                
            # 3. Metadata dict
            metadata = getattr(self.state, 'current_metadata', {})
            
            # 4. Window level/width
            window_level = getattr(self.state, 'window_level', 500)
            window_width = getattr(self.state, 'window_width', 1000)
            
            # 5. Slice slider value and text
            slice_idx = getattr(self.state, 'current_slice', 0)
            max_slice = getattr(self.state, 'current_data', np.zeros((1,1,1))).shape[2] - 1
            slice_text = f"{slice_idx}/{max_slice if max_slice > 0 else 0}"
            
            logger.info(f"Loaded data for annotator with slice: {slice_idx}")
            return annotation_value, error, metadata, window_level, window_width, slice_idx, slice_text, file_list
        except Exception as e:
            logger.error(f"Error loading data for annotator: {e}")
            # Return default values with error message
            # Create a placeholder image to prevent JSON decoding errors
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder, f"Error: {str(e)}", {}, 500, 1000, 0, "0/0", []
            
    def reset_directory(self):
        """Reset the directory input field"""
        logger.info("Resetting directory")
        return ""    def select_file_from_browser(self, selected_file):
        """Handle file selection from browser"""
        try:
            logger.info(f"Selected file from browser: {selected_file}")
            # Here you would load the selected file
            # For now, just a placeholder implementation
            
            # Create a placeholder image if needed
            annotation_value = self.load_medical_slice_to_annotator()
            if annotation_value is None:
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
                error = "Could not load selected file. Using placeholder."
            else:
                error = ""
                
            metadata = getattr(self.state, 'current_metadata', {})
            window_level = getattr(self.state, 'window_level', 500)
            window_width = getattr(self.state, 'window_width', 1000)
            slice_idx = getattr(self.state, 'current_slice', 0)
            slice_text = f"{slice_idx}/0"
            
            return annotation_value, error, metadata, window_level, window_width, slice_idx, slice_text
        except Exception as e:
            logger.error(f"Error selecting file from browser: {e}")
            # Create a placeholder for error case
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder, f"Error: {str(e)}", {}, 500, 1000, 0, "0/0"

    def prev_slice(self, slider_value):
        return max(0, int(slider_value) - 1)

    def next_slice(self, slider_value):
        return int(slider_value) + 1    def update_slice(self, slider_value, view_type):
        """Update display when slice slider changes"""
        try:
            # Update state
            self.state.current_slice = int(slider_value)
            self.state.current_view = view_type.lower()
            
            # Get new annotation value
            annotation_value = self.load_medical_slice_to_annotator()
            if annotation_value is None:
                # Create a placeholder value to prevent JSON decoding errors
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
            
            # Calculate slice text
            max_slice = 0
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    max_slice = data_shape[2] - 1
            
            slice_text = f"{slider_value}/{max_slice}"
            
            logger.info(f"Updated slice to {slider_value} with view {view_type}")
            return annotation_value, slice_text
        except Exception as e:
            logger.error(f"Error updating slice: {e}")
            # Create a placeholder value to prevent JSON decoding errors
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder, "0/0"

    def change_view(self, view):
        # For now, just reset slider and text, and reload image
        slice_slider = 0
        slice_text = "0/0"
        annotation_value = self.load_medical_slice_to_annotator()
        return slice_slider, slice_text, annotation_value

    def update_window_level(self, level, width):        
        annotation_value = self.load_medical_slice_to_annotator()
        return annotation_value

    def debug_selected_file(self, file_obj, dir_path):
        return "Debug info not implemented for custom annotator."
