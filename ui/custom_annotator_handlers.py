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
import random
import tempfile
import csv
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import io
import base64

# Import DICOM utilities for proper data loading
from utils.dicom_utils import load_dicom_series, load_single_dicom, list_dicoms_in_directory

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
                # Fallback method - use correct slice indexing pattern to match visualization.py
                data = self.state.current_data
                if len(data.shape) == 3:
                    if current_view.lower() == 'axial':
                        # For axial view, slices are along axis 0
                        image_array = data[current_slice, :, :]
                    elif current_view.lower() == 'sagittal':
                        # For sagittal view, slices are along axis 2
                        image_array = data[:, :, current_slice]
                    elif current_view.lower() == 'coronal':
                        # For coronal view, slices are along axis 1
                        image_array = data[:, current_slice, :]
                    else:
                        # Default to axial
                        image_array = data[current_slice, :, :]
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
                  # Check if we have saved annotations for this slice/view combination
            saved_annotations = self.state.get_slice_annotations(current_view, current_slice)
            
            # Create the annotation value for the plugin
            if saved_annotations and 'boxes' in saved_annotations:
                # Use saved annotations
                annotation_value = {
                    "image": image_rgb,
                    "boxes": saved_annotations['boxes']
                }
                logger.info(f"Loaded medical slice {current_slice} ({current_view}) with {len(saved_annotations['boxes'])} saved annotations")
            else:
                # Start with no annotations
                annotation_value = {
                    "image": image_rgb,
                    "boxes": []
                }
                logger.info(f"Loaded medical slice {current_slice} ({current_view}) with no saved annotations")
            
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
                    self.current_labels.append(label)                    # Add a random color for the new label
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
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(export_data, f, indent=2)
                    temp_path = f.name
                
                return temp_path, f"Exported {len(boxes)} annotations as JSON"
                
            elif format_type == "CSV":
                # Export as CSV
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
                        "area": width * height,                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(annotation)
                
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
    
    def on_annotation_change_editor(self, annotation_data: Dict[str, Any]) -> None:
        """Handle changes to annotations from the image_annotator component and track coordinates (editor tab specific)"""
        try:
            if annotation_data and 'boxes' in annotation_data:
                boxes = annotation_data['boxes']
                logger.info(f"Annotation changed: {len(boxes)} boxes present")
                
                # Check if a new box was added (simple heuristic)
                if hasattr(self, '_last_box_count'):
                    if len(boxes) > self._last_box_count:
                        # New box added, extract center coordinates
                        new_box = boxes[-1]  # Get the last (newest) box
                        center_x = int((new_box['xmin'] + new_box['xmax']) / 2)
                        center_y = int((new_box['ymin'] + new_box['ymax']) / 2)
                        
                        # Store coordinates using the same structure as MEDSAM2
                        if not hasattr(self, 'selected_coordinates'):
                            self.selected_coordinates = []
                        
                        self.selected_coordinates.append((center_x, center_y))
                        
                        logger.info(f"New annotation box created, center coordinates: ({center_x}, {center_y})")
                
                self._last_box_count = len(boxes)
            else:
                logger.info("Annotation data cleared or empty")
                if hasattr(self, '_last_box_count'):
                    self._last_box_count = 0
                    
        except Exception as e:
            logger.error(f"Error handling annotation change: {e}")
    
    def load_data_for_annotator(self, file_obj, dir_path, apply_deidentification=False):        
        """Load data for the custom annotator tab, matching Editor tab output order"""
        try:
            # Initialize file_list to ensure it's always defined
            file_list = []
            
            # Check if file was explicitly cleared (file_obj is None but we had data before)
            # In this case, clear everything even if dir_path has a value
            if file_obj is None and hasattr(self.state, 'current_data') and self.state.current_data is not None:
                # Clear the state
                self.state.current_data = None
                self.state.current_metadata = {}
                self.state.current_directory = None
                self.state.file_list = []
                # Create a placeholder image
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                placeholder = {"image": image, "boxes": []}
                return placeholder, "File cleared", {}, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "0/0", gr.Dropdown(choices=[], value=None)
            
            # Check if we have a file object or directory path
            if not file_obj and not dir_path:
                # Create a placeholder image to prevent JSON decoding errors
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                placeholder = {"image": image, "boxes": []}
                return placeholder, "Please select a file or enter a directory path", {}, 500, 1000, 0, "0/0", gr.Dropdown(choices=[], value=None)
                
            # Process file or directory
            if file_obj:
                # Process single file (like in handlers.py)
                path = file_obj.name if file_obj else None
                if not path or not os.path.exists(path):
                    # Create a placeholder image to prevent JSON decoding errors
                    image = np.zeros((100, 100, 3), dtype=np.uint8)
                    placeholder = {"image": image, "boxes": []}
                    return placeholder, "File not found or invalid", {}, 500, 1000, 0, "0/0", gr.Dropdown(choices=[], value=None)                # Load the actual DICOM data using existing utilities
                try:
                    if path.lower().endswith('.dcm'):
                        # Load single DICOM file
                        pixel_array, metadata = load_single_dicom(path, apply_deidentification=apply_deidentification)
                        if pixel_array is not None:
                            # Convert 2D image to 3D for consistency
                            if len(pixel_array.shape) == 2:
                                pixel_array = np.expand_dims(pixel_array, axis=2)
                            
                            self.state.current_data = pixel_array
                            self.state.current_metadata = metadata
                            logger.info(f"Loaded single DICOM with shape: {pixel_array.shape}")
                        else:
                            # Fallback to placeholder if loading fails
                            self.state.current_data = np.zeros((512, 512, 1))
                            self.state.current_metadata = {}
                            logger.warning(f"Failed to load DICOM, using placeholder")
                    else:
                        # For other file types, use placeholder (could add more loaders here)
                        self.state.current_data = np.zeros((512, 512, 1))
                        self.state.current_metadata = {}
                        
                except Exception as e:
                    logger.error(f"Error loading file {path}: {str(e)}")
                    # Fallback to placeholder
                    self.state.current_data = np.zeros((512, 512, 1))
                    self.state.current_metadata = {}
                    
                self.state.current_slice = 0
                self.state.current_view = 'axial'
                # Set window values from metadata if available
                if hasattr(self.state, 'current_metadata') and self.state.current_metadata:
                    self.state.window_level = self.state.current_metadata.get('WindowCenter', 500)
                    self.state.window_width = self.state.current_metadata.get('WindowWidth', 1000)
                else:
                    self.state.window_level = 500
                    self.state.window_width = 1000
                
                file_list = [os.path.basename(path)]
            elif dir_path:
                # Process directory
                if not os.path.isdir(dir_path):
                    # Create a placeholder image to prevent JSON decoding errors
                    image = np.zeros((100, 100, 3), dtype=np.uint8)
                    placeholder = {"image": image, "boxes": []}
                    return placeholder, f"Directory not found: {dir_path}", {}, 500, 1000, 0, "0/0", gr.Dropdown(choices=[], value=None)
                  # Get list of applicable files (this is a placeholder)
                file_list = [f for f in os.listdir(dir_path) 
                             if f.endswith(('.dcm', '.nii', '.nii.gz', '.mat'))]
                
                if not file_list:
                    # Create a placeholder image to prevent JSON decoding errors
                    image = np.zeros((100, 100, 3), dtype=np.uint8)
                    placeholder = {"image": image, "boxes": []}
                    return placeholder, f"No applicable files found in {dir_path}", {}, 500, 1000, 0, "0/0", gr.Dropdown(choices=[], value=None)
                
                # Sort file list for consistent ordering
                file_list.sort()
                
                # Store the directory path and file list in state for later use
                self.state.current_directory = dir_path
                self.state.current_file_list = file_list
                  # Load the actual DICOM series using existing utilities
                try:
                    dicom_files = list_dicoms_in_directory(dir_path)
                    if dicom_files:
                        # Load the full DICOM series
                        volume, metadata, ordered_files = load_dicom_series(dir_path, apply_deidentification=apply_deidentification)
                        
                        self.state.current_data = volume
                        self.state.current_metadata = metadata
                        self.state.current_file_list = [os.path.basename(f) for f in ordered_files]
                        
                        logger.info(f"Loaded DICOM series with shape: {volume.shape}")
                        logger.info(f"Found {len(ordered_files)} DICOM files")
                        
                        # Update file_list to match the loaded files
                        file_list = self.state.current_file_list
                    else:
                        # No DICOM files found, use placeholder
                        self.state.current_data = np.zeros((512, 512, 10))
                        self.state.current_metadata = {}
                        logger.warning("No DICOM files found, using placeholder")
                        
                except Exception as e:
                    logger.error(f"Error loading DICOM series from {dir_path}: {str(e)}")
                    # Fallback to placeholder
                    self.state.current_data = np.zeros((512, 512, 10))
                    self.state.current_metadata = {}
                    
                self.state.current_slice = 0
                self.state.current_view = 'axial'
                # Set window values from metadata if available
                if hasattr(self.state, 'current_metadata') and self.state.current_metadata:
                    self.state.window_level = self.state.current_metadata.get('WindowCenter', 500)
                    self.state.window_width = self.state.current_metadata.get('WindowWidth', 1000)
                else:
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
                error = f"Successfully loaded {len(file_list)} file(s)"
                
            # 3. Metadata dict
            metadata = getattr(self.state, 'current_metadata', {})
            
            # 4. Window level/width
            window_level = getattr(self.state, 'window_level', 500)
            window_width = getattr(self.state, 'window_width', 1000)
              # 5. Slice slider value and text
            slice_idx = getattr(self.state, 'current_slice', 0)
            # Get max slice number safely - use correct axis based on current view
            max_slice = 0
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    current_view = getattr(self.state, 'current_view', 'axial').lower()
                    if current_view == 'axial':
                        max_slice = data_shape[0] - 1  # Slices along axis 0
                    elif current_view == 'sagittal':
                        max_slice = data_shape[2] - 1  # Slices along axis 2
                    elif current_view == 'coronal':
                        max_slice = data_shape[1] - 1  # Slices along axis 1
                    else:
                        max_slice = data_shape[0] - 1  # Default to axial
                        
            slice_text = f"{slice_idx}/{max_slice}"
              # 6. Create dropdown update with choices and default value
            if file_list:
                # Set the first file as the default selected value
                dropdown_update = gr.Dropdown(choices=file_list, value=file_list[0])
            else:
                dropdown_update = gr.Dropdown(choices=[], value=None)
            
            # 7. Also return slider update to set proper maximum
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    current_view = getattr(self.state, 'current_view', 'axial').lower()
                    if current_view == 'axial':
                        slider_max = data_shape[0] - 1
                    elif current_view == 'sagittal':
                        slider_max = data_shape[2] - 1  
                    elif current_view == 'coronal':
                        slider_max = data_shape[1] - 1
                    else:
                        slider_max = data_shape[0] - 1
                else:
                    slider_max = 0
                slice_slider_update = gr.Slider(minimum=0, maximum=slider_max, value=slice_idx, step=1)
            else:
                slice_slider_update = gr.Slider(minimum=0, maximum=1, value=0, step=1, visible=False)
            
            logger.info(f"Loaded data for annotator with slice: {slice_idx}, files: {len(file_list)}, slider_max: {slider_max}")
            return annotation_value, error, metadata, window_level, window_width, slice_slider_update, slice_text, dropdown_update
        except Exception as e:
            logger.error(f"Error loading data for annotator: {e}")            # Return default values with error message
            # Create a placeholder image to prevent JSON decoding errors
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder, f"Error: {str(e)}", {}, 500, 1000, gr.Slider(minimum=0, maximum=1, value=0, step=1, visible=False), "0/0", gr.Dropdown(choices=[], value=None)

    def reset_directory(self):
        """Reset the directory input field"""
        logger.info("Resetting directory")
        return ""
    
    def select_file_from_browser(self, selected_file):
        """Handle file selection from browser"""
        try:
            logger.info(f"Selected file from browser: {selected_file}")
            
            if not selected_file or not hasattr(self.state, 'current_directory'):
                # No file selected or no directory loaded
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
                return annotation_value, "No file selected or directory not loaded", {}, 500, 1000, gr.Slider(minimum=0, maximum=1, value=0, step=1, visible=False), "0/0"
            
            # Construct full path
            selected_path = os.path.join(self.state.current_directory, selected_file)
            
            # Load the selected DICOM file
            if not os.path.exists(selected_path):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
                return annotation_value, f"File not found: {selected_path}", {}, 500, 1000, gr.Slider(minimum=0, maximum=1, value=0, step=1, visible=False), "0/0"
            
            # Find index of selected file in file list
            if hasattr(self.state, 'current_file_list') and self.state.current_file_list:
                try:
                    slice_idx = self.state.current_file_list.index(selected_file)
                    self.state.current_slice = slice_idx
                    logger.info(f"Set current slice to: {slice_idx}")
                except ValueError:
                    slice_idx = 0
                    self.state.current_slice = 0
            else:
                slice_idx = 0
                self.state.current_slice = 0
            
            # Load the new slice using existing method
            annotation_value = self.load_medical_slice_to_annotator()
            if annotation_value is None:
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
                error = "Could not load selected file. Using placeholder."
            else:
                error = f"Loaded file: {selected_file}"
                
            # Get metadata and window settings
            metadata = getattr(self.state, 'current_metadata', {})
            window_level = getattr(self.state, 'window_level', 500)
            window_width = getattr(self.state, 'window_width', 1000)
            
            # Calculate max slice for current view and create slice text
            max_slice = 0
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    current_view = getattr(self.state, 'current_view', 'axial').lower()
                    if current_view == 'axial':
                        max_slice = data_shape[0] - 1
                    elif current_view == 'sagittal':
                        max_slice = data_shape[2] - 1
                    elif current_view == 'coronal':
                        max_slice = data_shape[1] - 1
                    else:
                        max_slice = data_shape[0] - 1
            
            slice_text = f"{slice_idx}/{max_slice}"
            
            # Create slider update with correct maximum
            slice_slider_update = gr.Slider(minimum=0, maximum=max_slice, value=slice_idx, step=1)
            
            logger.info(f"File browser selection complete: {selected_file}, slice: {slice_idx}/{max_slice}")
            return annotation_value, error, metadata, window_level, window_width, slice_slider_update, slice_text
            
        except Exception as e:
            logger.error(f"Error selecting file from browser: {e}")
            # Return default values with error message
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder, f"Error: {str(e)}", {}, 500, 1000, gr.Slider(minimum=0, maximum=1, value=0, step=1, visible=False), "0/0"

    def prev_slice(self, slider_value):
        """Navigate to previous slice"""
        try:
            # Get current slider value and ensure it's an integer
            current = int(slider_value) if slider_value else 0
            # Return new value, ensuring it doesn't go below 0
            new_value = max(0, current - 1)
            logger.info(f"Moving to previous slice: {new_value}")
            
            # Update state slice
            self.state.current_slice = new_value
            
            return new_value        
        except Exception as e:
            logger.error(f"Error navigating to previous slice: {e}")
            return 0

    def next_slice(self, slider_value):
        """Navigate to next slice"""
        try:
            # Get current slider value and ensure it's an integer
            current = int(slider_value) if slider_value else 0
            # Calculate max slice based on current view (if available)
            max_slice = 0
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    current_view = getattr(self.state, 'current_view', 'axial').lower()
                    if current_view == 'axial':
                        max_slice = data_shape[0] - 1  # Slices along axis 0
                    elif current_view == 'sagittal':
                        max_slice = data_shape[2] - 1  # Slices along axis 2
                    elif current_view == 'coronal':
                        max_slice = data_shape[1] - 1  # Slices along axis 1
                    else:
                        max_slice = data_shape[0] - 1  # Default to axial
            
            # Return new value, ensuring it doesn't exceed max
            new_value = min(max_slice, current + 1)
            logger.info(f"Moving to next slice: {new_value}")
            
            # Update state slice
            self.state.current_slice = new_value
            
            return new_value
        except Exception as e:
            logger.error(f"Error navigating to next slice: {e}")
            return 0

    def update_slice(self, slider_value, view_type):
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
              # Calculate slice text - use correct max slice calculation based on view
            max_slice = 0
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    current_view = view_type.lower()
                    if current_view == 'axial':
                        max_slice = data_shape[0] - 1  # Slices along axis 0
                    elif current_view == 'sagittal':
                        max_slice = data_shape[2] - 1  # Slices along axis 2
                    elif current_view == 'coronal':
                        max_slice = data_shape[1] - 1  # Slices along axis 1
                    else:
                        max_slice = data_shape[0] - 1  # Default to axial
            
            slice_text = f"{slider_value}/{max_slice}"
            
            logger.info(f"Updated slice to {slider_value} with view {view_type}")
            return annotation_value, slice_text
        except Exception as e:
            logger.error(f"Error updating slice: {e}")            # Create a placeholder value to prevent JSON decoding errors
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder, "0/0"

    def change_view(self, view):
        """Change view orientation (axial, sagittal, coronal)"""
        try:
            # Update state
            self.state.current_view = view.lower()
            # Reset slice to 0
            self.state.current_slice = 0
            
            # Calculate slice text - use correct max slice calculation based on view
            max_slice = 0
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                data_shape = self.state.current_data.shape
                if len(data_shape) >= 3:
                    if view.lower() == 'axial':
                        max_slice = data_shape[0] - 1  # Slices along axis 0
                    elif view.lower() == 'sagittal':
                        max_slice = data_shape[2] - 1  # Slices along axis 2
                    elif view.lower() == 'coronal':
                        max_slice = data_shape[1] - 1  # Slices along axis 1
            
            slice_slider_update = gr.Slider(minimum=0, maximum=max_slice, value=0, step=1)
            slice_text = f"0/{max_slice}"
            
            # Get annotation value for new view
            annotation_value = self.load_medical_slice_to_annotator()
            if annotation_value is None:
                # Create a placeholder value to prevent JSON decoding errors
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
            
            logger.info(f"Changed view to {view}")
            return slice_slider_update, slice_text, annotation_value
        except Exception as e:
            logger.error(f"Error changing view: {e}")
            # Create a placeholder value to prevent JSON decoding errors
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return gr.Slider(minimum=0, maximum=1, value=0, step=1, visible=False), "0/0", placeholder

    def update_window_level(self, level, width):
        """Update window level/width for image display"""
        try:
            # Update state
            self.state.window_level = level
            self.state.window_width = width
            
            # Get updated annotation value
            annotation_value = self.load_medical_slice_to_annotator()
            if annotation_value is None:
                # Create a placeholder value to prevent JSON decoding errors
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                annotation_value = {"image": image, "boxes": []}
            
            logger.info(f"Updated window level: {level}, width: {width}")
            return annotation_value
        except Exception as e:
            logger.error(f"Error updating window level/width: {e}")
            # Create a placeholder value to prevent JSON decoding errors
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            placeholder = {"image": image, "boxes": []}
            return placeholder

    def debug_selected_file(self, file_obj, dir_path):
        """Debug function for troubleshooting file loading"""
        try:
            # Get file info
            file_info = "No file selected"
            if file_obj:
                file_info = f"File: {file_obj.name}\n"
                file_info += f"Size: {os.path.getsize(file_obj.name)} bytes\n"
                file_info += f"Type: {os.path.splitext(file_obj.name)[1]}\n"
                
            # Get directory info
            dir_info = "No directory specified"
            if dir_path:
                if os.path.isdir(dir_path):
                    files = os.listdir(dir_path)
                    dir_info = f"Directory: {dir_path}\n"
                    dir_info += f"Contains {len(files)} files\n"
                    dir_info += f"First 5 files: {', '.join(files[:5])}"
                else:
                    dir_info = f"Directory not found: {dir_path}"
            
            # Get state info
            state_info = "State information:\n"
            if hasattr(self.state, 'current_data') and self.state.current_data is not None:
                state_info += f"Data shape: {self.state.current_data.shape}\n"
                state_info += f"Data type: {self.state.current_data.dtype}\n"
            else:
                state_info += "No data loaded in state\n"
                
            state_info += f"Current slice: {getattr(self.state, 'current_slice', 'None')}\n"
            state_info += f"Current view: {getattr(self.state, 'current_view', 'None')}\n"
            
            debug_info = f"{file_info}\n\n{dir_info}\n\n{state_info}"
            logger.info(f"Debug info generated: {debug_info}")
            return debug_info
        except Exception as e:
            logger.error(f"Error in debug_selected_file: {e}")
            return f"Error generating debug info: {str(e)}"
        
    def handle_image_select(self, evt: gr.SelectData) -> str:
        """Handle select events on the image_annotator to capture coordinates"""
        try:
            logger.info(f"Image select event received: {type(evt)}, data: {evt}")
            
            if evt is None:
                logger.warning("Select event is None")
                return "No click data received (None event)"
                
            # Check if event has index attribute (coordinates)
            if not hasattr(evt, 'index'):
                logger.warning(f"Select event missing index attribute. Available attributes: {dir(evt)}")
                return "No click data received (missing index)"
                
            x, y = evt.index
            
            # Use the same coordinate tracking as MEDSAM2
            if not hasattr(self, 'selected_coordinates'):
                self.selected_coordinates = []
                
            self.selected_coordinates.append((x, y))
            
            # Format coordinates for display
            coords_str = "; ".join([f"({x},{y})" for x, y in self.selected_coordinates])
            
            logger.info(f"Added coordinate from image_annotator: ({x}, {y}). Total points: {len(self.selected_coordinates)}")
            return coords_str
            
        except Exception as e:
            logger.error(f"Error handling image select: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

    def clear_selected_coordinates(self) -> str:
        """Clear selected coordinates for custom annotator"""
        try:
            if hasattr(self, 'selected_coordinates'):
                self.selected_coordinates = []
            logger.info("Cleared selected coordinates in custom annotator")
            return ""
        except Exception as e:
            logger.error(f"Error clearing coordinates: {str(e)}")
            return f"Error: {str(e)}"
