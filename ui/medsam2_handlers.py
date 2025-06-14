"""
MEDSAM2 Integration Handlers

This module contains the handlers for integrating MEDSAM2 medical image annotation
into the SegMed-Pro application.

IMPORTANT NOTE ON INDEXING:
- UI slices are 1-based (starting from 1)
- MEDSAM2 expects prompt keys as strings but converts them to 0-based indices by subtracting 1
- We send prompt keys as (UI slice + 1) to compensate for MEDSAM2's internal subtraction
- MEDSAM2 outputs mask files with the same indexing as UI slices (slice_0010_mask.npy for UI slice 10)
- When loading masks, we use the same index as the UI slice
"""

import os
import json
import logging
import subprocess
import gradio as gr
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image

from utils.debug_utils import log_exception
from utils.visualization import display_slice, overlay_segmentation
from utils.brain_roi_detector import BrainROIDetector

logger = logging.getLogger(__name__)


class MEDSAM2Handlers:
    """Handlers for MEDSAM2 annotation operations"""
    def __init__(self, state):
        self.state = state
        self.selected_coordinates = []  # Store [(x, y), ...] coordinate pairs
        self.annotated_slice = None  # Store the slice number that was annotated
        self.annotation_overlays = {}  # Store overlays for each slice {slice_index: overlay_data}
        self.score_threshold = 0.3  # Default score threshold for filtering annotations
        self.point_mode_enabled = False  # Flag to control whether clicks should be processed
        
        # Initialize brain ROI detector for automatic prompts
        self.brain_roi_detector = BrainROIDetector()
    
    def handle_image_click(self, evt: gr.SelectData) -> str:
        """Handle click events on the image to capture coordinates"""
        try:
            # Only process clicks if point mode is enabled
            # Use getattr to provide a default value if attribute doesn't exist
            point_mode_enabled = getattr(self, 'point_mode_enabled', True)  # Default to True for backward compatibility
            if not point_mode_enabled:
                logger.info("Point mode not enabled, ignoring click")
                return ""
            
            logger.info(f"Image click event received: {type(evt)}, data: {evt}")
            logger.info(f"Event attributes: {dir(evt) if evt else 'None'}")
            
            if evt is None:
                logger.warning("Event is None")
                return "No click data received (None event)"
            
            # Check for different possible coordinate attributes
            coordinates = None
            if hasattr(evt, 'index') and evt.index is not None:
                coordinates = evt.index
                logger.info(f"Using evt.index: {coordinates}")
            elif hasattr(evt, 'value') and evt.value is not None:
                # For image_annotator, coordinates might be in value
                logger.info(f"Event value: {evt.value}")
                if isinstance(evt.value, (list, tuple)) and len(evt.value) >= 2:
                    coordinates = evt.value[:2]  # Take first two elements as x, y
                    logger.info(f"Using evt.value as coordinates: {coordinates}")
            elif hasattr(evt, 'target') and evt.target is not None:
                logger.info(f"Event target: {evt.target}")
                # Try to extract coordinates from target
                if hasattr(evt.target, 'value') and isinstance(evt.target.value, (list, tuple)):
                    coordinates = evt.target.value[:2]
                    logger.info(f"Using evt.target.value: {coordinates}")
            
            if coordinates is None:
                logger.warning(f"No coordinates found. Event details: index={getattr(evt, 'index', None)}, value={getattr(evt, 'value', None)}")
                return "No click coordinates received. Please try clicking directly on the image."
            
            # Ensure we have exactly 2 coordinates
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
                logger.warning(f"Invalid coordinates format: {coordinates}")
                return f"Invalid coordinates format: {coordinates}"
            
            x, y = int(coordinates[0]), int(coordinates[1])
            self.selected_coordinates.append((x, y))
            
            # Format coordinates for display
            coords_str = "; ".join([f"({x},{y})" for x, y in self.selected_coordinates])
            
            logger.info(f"Added coordinate: ({x}, {y}). Total points: {len(self.selected_coordinates)}")
            return coords_str
            
        except Exception as e:
            logger.error(f"Error handling image click: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"
    @log_exception
    def clear_coordinates(self) -> str:
        """Clear all selected coordinates and reset annotated slice"""
        self.selected_coordinates = []
        self.annotated_slice = None  # Reset annotated slice when clearing coordinates
        logger.info("Cleared all coordinates and reset annotated slice")
        return ""    
    @log_exception
    def clear_annotation_overlays(self) -> Tuple[str, Optional[Dict]]:
        """Clear all annotation overlays from memory and refresh current image"""
        self.annotation_overlays = {}
        logger.info("Cleared all annotation overlays")
        
        # Refresh the current image display to show the cleared overlays
        if self.state.current_data is not None:
            current_slice = self.state.current_slice_idx
            original_slice_img = display_slice(
                self.state.current_data,
                current_slice,
                self.state.current_view,
                crosshair=self.state.crosshair_position
            )
            
            # Convert to format expected by image_annotator
            if len(original_slice_img.shape) == 2:
                img_rgb = np.stack([original_slice_img] * 3, axis=-1)
            else:
                img_rgb = original_slice_img
            
            if img_rgb.dtype != np.uint8:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            
            # Create AnnotatedImageValue format
            annotated_value = {
                "image": img_rgb,
                "boxes": [],  # Clear all boxes/annotations
                "orientation": 0
            }
            
            return "Annotation overlays cleared", annotated_value
        else:
            return "Annotation overlays cleared", None
    @log_exception
    def generate_prompt_json(self, dicom_folder: str) -> Tuple[bool, str]:
        """Generate the brain_target_prompts.json file from selected coordinates"""
        if not self.selected_coordinates:
            return False, "No coordinates selected"
            
        if not dicom_folder or not os.path.exists(dicom_folder):
            return False, f"DICOM folder not found: {dicom_folder}"
        try:
            # Get current slice information from state
            current_slice = self.state.current_slice_idx
            logger.info(f"Current slice from state: {current_slice} (type: {type(current_slice)})")
              # Store the slice that will be annotated (keep as 1-based for UI consistency)
            self.annotated_slice = current_slice
            logger.info(f"Stored annotated slice: {self.annotated_slice} (UI-based)")
            
            # Prepare points and labels for the current slice
            points = []
            labels = []
            
            for x, y in self.selected_coordinates:
                points.append([int(x), int(y)])  # Ensure integers
                labels.append(1)  # 1 for foreground point            # Create the prompt structure expected by MEDSAM2
            # prompt_key = UI_slice + 1 (MEDSAM2 will convert this to 0-based by subtracting 1 internally)
            # We add 1 to compensate for MEDSAM2's internal subtraction
            prompt_key = str(current_slice + 1)
            prompt_data = {
                prompt_key: {
                    "points": points,
                    "labels": labels
                }
            }
            
            # Save to brain_target_prompts.json in the project root (not in DICOM folder)
            prompt_file = "brain_target_prompts.json"
            with open(prompt_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
            logger.info(f"Generated prompt file: {prompt_file} for slice {current_slice} with {len(self.selected_coordinates)} points")
            logger.info(f"Using prompt key: {prompt_key} (UI slice {current_slice} + 1) for MEDSAM2")
            logger.info(f"Points: {points}")
            logger.info(f"Stored annotated slice number: {self.annotated_slice}")
            return True, prompt_file
        except Exception as e:
            logger.error(f"Error generating prompt JSON: {str(e)}")
            return False, f"Error: {str(e)}"
    
    @log_exception
    def generate_prompt_json_all_slices(self, dicom_folder: str) -> Tuple[bool, str]:
        """Generate prompt JSON for all slices using selected coordinates as template"""
        if not self.selected_coordinates:
            return False, "No coordinates selected"
            
        if not dicom_folder or not os.path.exists(dicom_folder):
            return False, f"DICOM folder not found: {dicom_folder}"
        
        try:
            # Get total number of slices from current data
            if self.state.current_data is None:
                return False, "No DICOM data loaded"
            
            total_slices = self.state.current_data.shape[2]  # Assuming axial view
            logger.info(f"Generating prompts for {total_slices} slices using user coordinates")
            
            # Prepare points and labels from user selection
            points = []
            labels = []
            
            for x, y in self.selected_coordinates:
                points.append([int(x), int(y)])  # Ensure integers
                labels.append(1)  # 1 for foreground point
            
            # Create prompt structure for all slices using the same coordinates
            prompt_data = {}
            for slice_idx in range(total_slices):
                # Use 0-based indexing for slice_idx, but MEDSAM2 will convert the string key to 0-based internally
                # So we use the UI slice number (1-based) + 1 as the key to compensate for MEDSAM2's subtraction
                ui_slice = slice_idx + 1  # Convert to 1-based UI slice
                prompt_key = str(ui_slice + 1)  # Add 1 more to compensate for MEDSAM2's internal subtraction
                prompt_data[prompt_key] = {
                    "points": points,
                    "labels": labels
                }
            
            logger.info(f"Prompt data for all slices: {json.dumps(prompt_data, indent=2)}")
            
            # Save to brain_target_prompts.json in the project root
            prompt_file = "brain_target_prompts.json"
            with open(prompt_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
            
            logger.info(f"Generated all-records prompt file: {prompt_file} for {total_slices} slices")
            logger.info(f"Using prompt keys: UI slice + 1 to compensate for MEDSAM2's internal subtraction")
            logger.info(f"Using coordinates: {points}")
            return True, prompt_file
            
        except Exception as e:
            logger.error(f"Error generating all-records prompt JSON: {str(e)}")
            return False, f"Error: {str(e)}"

    @log_exception
    def generate_automatic_brain_prompts(self, dicom_folder: str, processing_mode: str = "All Records") -> Tuple[bool, str]:
        """
        Generate automatic brain structure prompts using anatomical detection
        
        Args:
            dicom_folder: Path to DICOM folder
            processing_mode: "Single Slice" or "All Records"
        
        Returns:
            Tuple of (success, message_or_filename)
        """
        if not dicom_folder or not os.path.exists(dicom_folder):
            return False, f"DICOM folder not found: {dicom_folder}"
        
        if self.state.current_data is None:
            return False, "No DICOM data loaded"
        
        try:
            import pydicom
            import glob
            
            # Get DICOM files
            dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
            if not dicom_files:
                return False, f"No DICOM files found in {dicom_folder}"
            
            # Sort files by slice position if possible
            dicom_datasets = []
            for file_path in sorted(dicom_files):
                try:
                    ds = pydicom.dcmread(file_path)
                    dicom_datasets.append((file_path, ds))
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue
            
            # Sort by instance number or position
            try:
                if dicom_datasets and hasattr(dicom_datasets[0][1], 'InstanceNumber'):
                    dicom_datasets.sort(key=lambda x: int(x[1].InstanceNumber))
                elif dicom_datasets and hasattr(dicom_datasets[0][1], 'ImagePositionPatient'):
                    dicom_datasets.sort(key=lambda x: float(x[1].ImagePositionPatient[2]))
            except Exception as e:
                logger.warning(f"Could not sort DICOM files: {e}")
            
            prompt_data = {}
            
            if processing_mode == "Single Slice":
                # Generate prompts for current slice only
                current_slice = self.state.current_slice_idx
                
                if current_slice - 1 >= len(dicom_datasets):
                    return False, f"Current slice {current_slice} out of range"
                
                file_path, dicom_ds = dicom_datasets[current_slice - 1]  # Convert to 0-based
                
                # Get the current slice image
                slice_image = display_slice(
                    self.state.current_data,
                    current_slice,
                    self.state.current_view,
                    crosshair=None  # Don't include crosshair in analysis
                )
                
                # Generate bounding boxes for this slice
                boxes = self.brain_roi_detector.generate_prompt_boxes(
                    slice_image, 
                    dicom_ds, 
                    include_eyes=True
                )
                
                if boxes:
                    # Convert to MEDSAM2 format (UI slice + 1 to compensate for internal subtraction)
                    prompt_key = str(current_slice + 1)
                    prompt_data[prompt_key] = {"boxes": boxes}
                    
                    logger.info(f"Generated {len(boxes)} automatic prompts for slice {current_slice}")
                    self.annotated_slice = current_slice  # Store the annotated slice
                else:
                    return False, f"No brain regions detected in slice {current_slice}"
                    
            else:  # "All Records"
                # Generate prompts for all slices
                total_slices = len(dicom_datasets)
                successful_slices = 0
                
                for i, (file_path, dicom_ds) in enumerate(dicom_datasets):
                    try:
                        # Get the slice image (convert from 0-based dataset index to 1-based UI slice)
                        ui_slice = i + 1
                        slice_image = display_slice(
                            self.state.current_data,
                            ui_slice,
                            self.state.current_view,
                            crosshair=None
                        )
                        
                        # Generate bounding boxes for this slice
                        boxes = self.brain_roi_detector.generate_prompt_boxes(
                            slice_image, 
                            dicom_ds, 
                            include_eyes=True
                        )
                        
                        if boxes:
                            # Convert to MEDSAM2 format (UI slice + 1)
                            prompt_key = str(ui_slice + 1)
                            prompt_data[prompt_key] = {"boxes": boxes}
                            successful_slices += 1
                            
                            logger.info(f"Generated {len(boxes)} automatic prompts for slice {ui_slice}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to generate prompts for slice {i+1}: {e}")
                        continue
                
                if successful_slices == 0:
                    return False, "No brain regions detected in any slice"
                
                logger.info(f"Generated automatic prompts for {successful_slices}/{total_slices} slices")
            
            # Save prompt file
            prompt_file = "brain_target_prompts.json"
            with open(prompt_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
            
            logger.info(f"Saved automatic brain prompts to {prompt_file}")
            logger.info(f"Prompt data sample: {json.dumps(dict(list(prompt_data.items())[:2]), indent=2)}")
            
            return True, prompt_file
            
        except Exception as e:
            logger.error(f"Error generating automatic brain prompts: {str(e)}")
            return False, f"Error: {str(e)}"    @log_exception
    def run_medsam2_annotation_automatic(self, dicom_folder: str, output_dir: str, 
                                        save_visualizations: bool, device: str, 
                                        processing_mode: str = "All Records", 
                                        score_threshold: float = 0.3) -> Tuple[str, Optional[Dict], bool]:
        """
        Run MEDSAM2 annotation with automatic brain structure detection
        Returns tuple for compatibility with UI handler: (status_message, annotated_image_data, success_flag)
        
        Args:
            dicom_folder: Path to DICOM folder
            output_dir: Output directory for results
            save_visualizations: Whether to save visualizations
            device: Processing device (cpu/cuda)
            processing_mode: "Single Slice" or "All Records"
            score_threshold: Score threshold for filtering results (All Records mode)
        
        Returns:
            Tuple of (status_message, annotated_image_data, success_flag)
        """
        if not dicom_folder or not os.path.exists(dicom_folder):
            return f"Error: DICOM folder not found: {dicom_folder}", None, False
        
        # Store the score threshold
        self.score_threshold = score_threshold
        
        try:
            # Generate automatic prompts
            success, prompt_file_or_error = self.generate_automatic_brain_prompts(dicom_folder, processing_mode)
            if not success:
                return f"Error generating automatic prompts: {prompt_file_or_error}", None, False
            
            prompt_file = prompt_file_or_error
            
            # Prepare the MEDSAM2 command
            script_path = os.path.join("models", "medsam2", "brain_mri_dicom_inference.py")
            if not os.path.exists(script_path):
                return f"Error: MEDSAM2 script not found at {script_path}", None, False
            
            # Get absolute paths for checkpoint and config
            medsam2_dir = os.path.join("models", "medsam2")
            checkpoint_path = os.path.join(medsam2_dir, "checkpoints", "MedSAM2_latest.pt")
            config_path = os.path.join(medsam2_dir, "configs", "sam2.1_hiera_t512.yaml")
            
            # Verify checkpoint and config exist
            if not os.path.exists(checkpoint_path):
                return f"Error: Checkpoint not found at {checkpoint_path}", None, False
            if not os.path.exists(config_path):
                return f"Error: Config not found at {config_path}", None, False
            
            # Build command
            cmd = [
                "python", script_path,
                "--dicom_folder", dicom_folder,
                "--output_dir", output_dir,
                "--prompt_boxes", prompt_file,  # Use box prompts instead of point prompts
                "--checkpoint", checkpoint_path,
                "--config", config_path,
                "--device", device,
                "--quiet"  # Reduce logging for better performance
            ]
            
            # Add processing mode specific flags
            if processing_mode == "Single Slice":
                cmd.append("--single_slice")
            
            if save_visualizations:
                cmd.append("--save_visualizations")
                
            logger.info(f"Running MEDSAM2 automatic annotation command: {' '.join(cmd)}")
            
            # Run the annotation
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )            
            # Process the result
            status_msg = ""
            annotated_result = None
            success_flag = False
            
            if result.returncode == 0:
                status_msg = f"✅ Automatic brain annotation completed successfully!\n"
                status_msg += f"Output directory: {output_dir}\n"
                if result.stdout:
                    # Extract useful information from stdout
                    stdout_lines = result.stdout.strip().split('\n')
                    for line in stdout_lines[-10:]:  # Show last 10 lines
                        if any(keyword in line.lower() for keyword in ['processed', 'saved', 'completed', 'slice']):
                            status_msg += f"• {line}\n"
                
                success_flag = True                # Filter results by score threshold and load masks into annotation_overlays
                if processing_mode == "All Records":
                    try:
                        filter_msg = self.filter_automatic_results_by_score(output_dir, score_threshold)
                        status_msg += f"\n{filter_msg}\n"
                        logger.info(f"Filtered automatic annotation results: {filter_msg}")
                    except Exception as e:
                        logger.warning(f"Could not filter results by score: {e}")
                        status_msg += f"\nNote: Could not filter results by score: {str(e)}\n"
                
                # Try to load results and create annotation display
                try:
                    load_msg, result_image = self.load_annotation_results(output_dir, processing_mode)
                    if result_image is not None:
                        status_msg += f"\n{load_msg}"
                        
                        # Convert result to annotated format like manual workflow
                        from utils.visualization import display_slice, create_annotation_boxes_from_mask
                        
                        # Get clean image
                        clean_img = display_slice(
                            self.state.current_data,
                            self.state.current_slice_idx,
                            self.state.current_view,
                            crosshair=self.state.crosshair_position
                        )
                        
                        # Ensure it's RGB and uint8
                        if len(clean_img.shape) == 2:
                            img_rgb = np.stack([clean_img] * 3, axis=-1)
                        else:
                            img_rgb = clean_img
                        if img_rgb.dtype != np.uint8:
                            img_rgb = (img_rgb * 255).astype(np.uint8)
                          # Convert MEDSAM2 masks to polygon shapes
                        annotation_shapes = []
                        if (hasattr(self, 'annotation_overlays') and 
                            self.state.current_slice_idx in self.annotation_overlays):
                            
                            overlay_data = self.annotation_overlays[self.state.current_slice_idx]
                            
                            # Handle both old format (single mask) and new format (multiple annotations)
                            if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                                # Old format - single annotation
                                mask_array = overlay_data['mask']
                                annotation_shapes = create_annotation_boxes_from_mask(
                                    mask_array, 
                                    label="Auto Brain Annotation",
                                    label_index=1
                                )
                            else:
                                # New format - multiple annotations per slice
                                for annotation_id, annotation_data in overlay_data.items():
                                    if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                                        mask_array = annotation_data['mask']
                                        shapes = create_annotation_boxes_from_mask(
                                            mask_array, 
                                            label=f"Auto Brain Annotation {annotation_id}",
                                            label_index=1
                                        )
                                        annotation_shapes.extend(shapes)
                            
                            logger.info(f"Converted automatic annotation mask to {len(annotation_shapes)} polygon shapes")
                        
                        # Create AnnotatedImageValue format with polygon shapes
                        annotated_result = {
                            "image": img_rgb,
                            "boxes": annotation_shapes,
                            "orientation": 0
                        }
                        
                    else:
                        status_msg += f"\nNote: Annotation completed but overlay could not be loaded: {load_msg}"
                        
                except Exception as e:
                    logger.warning(f"Could not load result for display: {e}")
                    status_msg += f"\nNote: Annotation completed but display loading failed: {str(e)}"
                
            else:
                status_msg = f"❌ Automatic brain annotation failed (return code: {result.returncode})\n"
                if result.stderr:
                    status_msg += f"Error output:\n{result.stderr[:500]}"
                if result.stdout:
                    status_msg += f"\nStandard output:\n{result.stdout[:500]}"
            
            return status_msg, annotated_result, success_flag
            
        except Exception as e:
            logger.error(f"Error running automatic MEDSAM2 annotation: {str(e)}")
            return f"Error: {str(e)}", None, False
    @log_exception
    def run_medsam2_annotation(self, dicom_folder: str, output_dir: str, 
                              save_visualizations: bool, device: str) -> str:
        """Run the MEDSAM2 annotation script"""
        if not self.selected_coordinates:
            return "Error: No coordinates selected. Please click on the image to select points."
        if not dicom_folder or not os.path.exists(dicom_folder):
            return f"Error: DICOM folder not found: {dicom_folder}"
        
        # Debug: Log the DICOM folder and check files
        logger.info(f"DICOM folder path: {dicom_folder}")
        try:
            dicom_files = [f for f in os.listdir(dicom_folder) if f.lower().endswith('.dcm')]
            logger.info(f"Found {len(dicom_files)} .dcm files in folder")
            if dicom_files:
                logger.info(f"Sample DICOM files: {dicom_files[:3]}")
            else:
                logger.warning("No .dcm files found in directory")
                # Check for any files
                all_files = os.listdir(dicom_folder)
                logger.info(f"All files in directory: {all_files[:10]}")
        except Exception as e:
            logger.error(f"Error listing DICOM folder contents: {str(e)}")
            return f"Error: Cannot access DICOM folder contents: {str(e)}"
        
        try:
            # Generate the prompt JSON file
            success, prompt_file_or_error = self.generate_prompt_json(dicom_folder)
            if not success:
                return f"Error generating prompts: {prompt_file_or_error}"
            
            prompt_file = prompt_file_or_error
            
            # Prepare the command with absolute paths
            script_path = os.path.join("models", "medsam2", "brain_mri_dicom_inference.py")
            if not os.path.exists(script_path):
                return f"Error: MEDSAM2 script not found at {script_path}"
            
            # Get absolute paths for checkpoint and config
            medsam2_dir = os.path.join("models", "medsam2")
            checkpoint_path = os.path.join(medsam2_dir, "checkpoints", "MedSAM2_latest.pt")
            config_path = os.path.join(medsam2_dir, "configs", "sam2.1_hiera_t512.yaml")
            #config_path = os.path.join(medsam2_dir, "configs", "sam2.1_hiera_tiny_finetune512.yaml")
            # Verify checkpoint and config exist
            if not os.path.exists(checkpoint_path):
                return f"Error: Checkpoint not found at {checkpoint_path}"
            if not os.path.exists(config_path):
                return f"Error: Config not found at {config_path}"
            cmd = [
                "python", script_path,
                "--dicom_folder", dicom_folder,
                "--output_dir", output_dir,
                "--prompt_points", prompt_file,
                "--checkpoint", checkpoint_path,
                "--config", config_path,
                "--device", device,
                "--single_slice",  # Only process the slice with prompts
                "--quiet"  # Reduce logging for better performance
            ]
            
            if save_visualizations:
                cmd.append("--save_visualizations")
                
            logger.info(f"Running MEDSAM2 command: {' '.join(cmd)}")
            logger.info(f"Annotated slice stored: {self.annotated_slice}")
            
            # Run the annotation
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )            
            if result.returncode == 0:
                logger.info("MEDSAM2 annotation completed successfully")
                # Use the same slice number for the output file
                expected_file_slice = self.annotated_slice  # Same as UI slice with our +1 adjustment
                slice_num_padded = str(expected_file_slice).zfill(4)
                logger.info(f"Will look for slice_{slice_num_padded}_mask.npy for UI slice {self.annotated_slice}")
                return f"Annotation completed successfully! Results saved to: {output_dir}"
            else:
                logger.error(f"MEDSAM2 failed: {result.stderr}")            
                return f"Error running MEDSAM2: {result.stderr}"
        except Exception as e:
            logger.error(f"Error running MEDSAM2 annotation: {str(e)}")
            return f"Error: {str(e)}"
    
    @log_exception
    def load_annotation_results(self, output_dir: str, processing_mode: str = "Single Slice") -> Tuple[str, Optional[np.ndarray]]:
        """Load and display annotation results from MEDSAM2 - combining .npy mask with original DICOM slice"""
        if not output_dir or not os.path.exists(output_dir):
            return "Error: Output directory not found", None
        
        try:
            # Check if we have the original DICOM data loaded
            if self.state.current_data is None:
                return "Error: No DICOM data loaded. Please load DICOM series first.", None
            
            if processing_mode == "Single Slice":
                return self.load_single_slice_results(output_dir)
            else:  # All Records mode
                return self.load_all_records_results(output_dir)
                
        except Exception as e:
            logger.error(f"Error loading annotation results: {str(e)}")
            return f"Error: {str(e)}", None    
        
    @log_exception
    def load_single_slice_results(self, output_dir: str) -> Tuple[str, Optional[np.ndarray]]:
        """Load results for single slice mode"""
        # Load the windowing metadata from the results
        windowing_metadata_path = os.path.join(output_dir, 'windowing_metadata.json')
        if os.path.exists(windowing_metadata_path):
            with open(windowing_metadata_path, 'r') as f:
                windowing_metadata = json.load(f)
            logger.info(f"Loaded windowing metadata: {windowing_metadata}")
        else:
            logger.warning("No windowing metadata found")
            windowing_metadata = {}
          # Check if we have an annotated slice stored
        if self.annotated_slice is None:
            return "Error: No annotated slice information found", None        # After our +1 adjustment to the prompt key, MEDSAM2 uses the same UI slice number for the output file
        expected_file_slice = self.annotated_slice  # Use the same UI slice number
        logger.info(f"UI slice: {self.annotated_slice}, expected file slice: {expected_file_slice}")

        masks_dir = os.path.join(output_dir, "masks")
        if not os.path.exists(masks_dir):
            return f"Error: Masks directory not found for slice {self.annotated_slice}", None

        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_mask.npy') and not f.endswith('_all_masks.npy')]
        logger.info(f"Available mask files: {mask_files}")

        # Check for the file with the same UI slice number
        slice_num_padded = str(expected_file_slice).zfill(4)
        mask_path = os.path.join(masks_dir, f"slice_{slice_num_padded}_mask.npy")
        logger.info(f"Checking for mask file: {mask_path}")        
        if mask_path and os.path.exists(mask_path):
            logger.info(f"SUCCESS: Found mask file for slice {self.annotated_slice}: {mask_path}")
            mask_array = np.load(mask_path)
            logger.info(f"Loaded mask with shape: {mask_array.shape}")
            # Use the same index for both DICOM and mask (+1 adjustment)
            dicom_slice_idx = self.annotated_slice  # Use UI slice directly with +1 adjustment
            logger.info(f"Using DICOM slice index: {dicom_slice_idx} (UI slice {self.annotated_slice} with +1 adjustment)")
            original_slice_img = display_slice(
                self.state.current_data,
                dicom_slice_idx,
                self.state.current_view,
                crosshair=self.state.crosshair_position
            )
            overlayed_img = overlay_segmentation(
                original_slice_img,
                mask_array,
                alpha=0.4,
                colormap={1: [255, 0, 0]}
            )
            from utils.visualization import make_image_for_gradio
            result_image = make_image_for_gradio(overlayed_img)
            self.annotation_overlays[self.annotated_slice] = {
                'mask': mask_array,
                'output_dir': output_dir,
                'slice_idx': dicom_slice_idx
            }
            return f"Successfully loaded annotation for slice {self.annotated_slice} (file: slice_{slice_num_padded}_mask.npy)", result_image
        else:
            return f"Error: No mask file found for UI slice {self.annotated_slice}. Expected file 'slice_{slice_num_padded}_mask.npy'. Available files: {mask_files}", None    @log_exception
    def load_all_records_results(self, output_dir: str) -> Tuple[str, Optional[np.ndarray]]:
        """Load results for all records mode - show current slice if it has valid annotation"""
        current_slice = self.state.current_slice_idx
        
        # Check if we have a stored overlay for the current slice
        if current_slice in self.annotation_overlays:
            overlay_data = self.annotation_overlays[current_slice]
            
            # Handle both old format (single mask) and new format (multiple annotations)
            if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                # Old format - single annotation
                mask_array = overlay_data['mask']
            else:
                # New format - multiple annotations per slice
                # Combine all masks for this slice
                combined_mask = None
                annotation_count = 0
                
                for annotation_id, annotation_data in overlay_data.items():
                    if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                        mask = annotation_data['mask']
                        if combined_mask is None:
                            combined_mask = mask.copy()
                        else:
                            # Combine masks (overlay them)
                            combined_mask = np.maximum(combined_mask, mask)
                        annotation_count += 1
                
                if combined_mask is not None:
                    mask_array = combined_mask
                else:
                    return f"No valid annotations found for slice {current_slice}", None
            
            # Get the original DICOM slice image
            original_slice_img = display_slice(
                self.state.current_data,
                current_slice,
                self.state.current_view,
                crosshair=self.state.crosshair_position
            )
            
            # Apply overlay
            overlayed_img = overlay_segmentation(
                original_slice_img,
                mask_array,
                alpha=0.4,
                colormap={1: [255, 0, 0]}  # Red overlay for annotations
            )            
            # Convert to PIL Image for gradio
            from utils.visualization import make_image_for_gradio
            result_image = make_image_for_gradio(overlayed_img)
            
            # Create informative message based on annotation structure
            if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                return f"Displaying annotation for slice {current_slice} (from all records processing)", result_image
            else:
                return f"Displaying {annotation_count} combined annotations for slice {current_slice} (from all records processing)", result_image
        else:
            # No annotation for current slice - show plain image
            original_slice_img = display_slice(
                self.state.current_data,
                current_slice,
                self.state.current_view,
                crosshair=self.state.crosshair_position
            )
            
            from utils.visualization import make_image_for_gradio
            result_image = make_image_for_gradio(original_slice_img)
            
            return f"No annotation available for slice {current_slice} (may not meet score threshold)", result_image
    def get_current_dicom_folder(self) -> Optional[str]:
        """Get the current DICOM folder path from application state"""
        if self.state.file_list and len(self.state.file_list) > 0:
            # Get the directory of the first file
            first_file = self.state.file_list[0]
            dicom_folder = os.path.dirname(first_file)
            logger.info(f"Determined DICOM folder from first file '{first_file}': {dicom_folder}")
            logger.info(f"File list contains {len(self.state.file_list)} files")
            return dicom_folder
        else:
            logger.warning("No files in state.file_list - cannot determine DICOM folder")
            return None
    @log_exception
    def run_full_annotation_workflow(self, output_dir: str, save_visualizations: bool, 
                                   device: str, processing_mode: str = "Single Slice", 
                                   score_threshold: float = 0.3) -> Tuple[str, Optional[np.ndarray]]:
        """Run the complete annotation workflow from coordinates to results"""
        # Validate that coordinates are selected for both modes
        if not self.selected_coordinates:
            return "Error: No coordinates selected. Please click on the image to select points first.", None
        
        # Store the score threshold for filtering
        self.score_threshold = score_threshold
        
        # Get current DICOM folder
        dicom_folder = self.get_current_dicom_folder()
        if not dicom_folder:
            return "Error: No DICOM data loaded. Please load DICOM files first.", None
        
        # Run annotation based on processing mode
        if processing_mode == "Single Slice":
            annotation_result = self.run_medsam2_annotation(
                dicom_folder, output_dir, save_visualizations, device
            )
        else:  # All Records mode
            annotation_result = self.run_medsam2_all_records(
                dicom_folder, output_dir, save_visualizations, device, score_threshold
            )        # Check if annotation was successful
        if "successfully" in annotation_result.lower():
            # Try to load results
            load_result, result_image = self.load_annotation_results(output_dir, processing_mode)
              # Convert result_image to annotated format if it exists
            if result_image is not None:
                # Get the original image without overlay for clean display
                from utils.visualization import display_slice, create_annotation_boxes_from_mask
                
                # Get clean image
                clean_img = display_slice(
                    self.state.current_data,
                    self.state.current_slice_idx,
                    self.state.current_view,
                    crosshair=self.state.crosshair_position
                )
                
                # Ensure it's RGB and uint8
                if len(clean_img.shape) == 2:
                    img_rgb = np.stack([clean_img] * 3, axis=-1)
                else:
                    img_rgb = clean_img
                if img_rgb.dtype != np.uint8:
                    img_rgb = (img_rgb * 255).astype(np.uint8)
                  # Convert MEDSAM2 masks to polygon shapes
                annotation_shapes = []
                if (hasattr(self, 'annotation_overlays') and 
                    self.state.current_slice_idx in self.annotation_overlays):
                    
                    overlay_data = self.annotation_overlays[self.state.current_slice_idx]
                    
                    # Handle both old format (single mask) and new format (multiple annotations)
                    if isinstance(overlay_data, dict) and 'mask' in overlay_data:
                        # Old format - single annotation
                        mask_array = overlay_data['mask']
                        annotation_shapes = create_annotation_boxes_from_mask(
                            mask_array, 
                            label="MEDSAM2 Annotation",
                            label_index=1
                        )
                    else:
                        # New format - multiple annotations per slice
                        for annotation_id, annotation_data in overlay_data.items():
                            if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                                mask_array = annotation_data['mask']
                                shapes = create_annotation_boxes_from_mask(
                                    mask_array, 
                                    label=f"MEDSAM2 Annotation {annotation_id}",
                                    label_index=1
                                )
                                annotation_shapes.extend(shapes)
                      
                    logger.info(f"Converted MEDSAM2 mask to {len(annotation_shapes)} polygon shapes")
                
                # Create AnnotatedImageValue format with polygon shapes
                annotated_result = {
                    "image": img_rgb,
                    "boxes": annotation_shapes,  # Now contains actual polygon shapes
                    "orientation": 0
                }
            else:
                annotated_result = None
            
            # Clear coordinates after successful annotation
            self.clear_coordinates()
            # Return tuple indicating success for UI updates
            return f"{annotation_result}\n{load_result}", annotated_result, True
        else:
            return annotation_result, None, False

    @log_exception
    def test_medsam2_setup(self) -> Tuple[bool, str]:
        """Test if MEDSAM2 can be initialized properly"""
        try:
            import sys
            import os
            
            # Add models path to sys.path
            models_path = os.path.join(os.getcwd(), "models", "medsam2")
            if models_path not in sys.path:
                sys.path.append(models_path)
            
            # Test imports
            try:
                from build_sam import build_sam2
                from sam2_image_predictor import SAM2ImagePredictor
            except ImportError as e:
                return False, f"Import error: {str(e)}"
            
            # Check if config and checkpoint files exist
            config_path = os.path.join("models", "medsam2", "configs", "sam2.1_hiera_t512.yaml")
            checkpoint_path = os.path.join("models", "medsam2", "checkpoints", "MedSAM2_latest.pt")
            
            if not os.path.exists(config_path):
                return False, f"Config file not found: {config_path}"
            
            if not os.path.exists(checkpoint_path):
                return False, f"Checkpoint file not found: {checkpoint_path}"
            
            # Try to initialize model (this might fail due to device/memory, but should pass import/config issues)
            try:
                logger.info("Testing SAM2 model initialization...")
                model = build_sam2(
                    config_file=config_path,
                    ckpt_path=checkpoint_path,
                    device="cpu"  # Use CPU for testing
                )
                return True, "MEDSAM2 setup successful!"
            except Exception as e:
                # If it's a Hydra error, that's what we're trying to fix
                if "GlobalHydra" in str(e) or "not initialized" in str(e):
                    return False, f"Hydra initialization error: {str(e)}"
                else:
                    # Other errors might be expected (memory, CUDA, etc.)
                    return True, f"MEDSAM2 imports work, model init error (may be expected): {str(e)}"
            
        except Exception as e:
            return False, f"Setup test failed: {str(e)}"
    
    @log_exception
    def validate_medsam2_environment(self) -> Tuple[bool, str]:
        """Validate that MEDSAM2 environment is properly set up"""
        try:
            import sys
            import os
            
            # Check if models directory exists
            models_dir = os.path.join(os.getcwd(), "models", "medsam2")
            if not os.path.exists(models_dir):
                return False, f"MEDSAM2 models directory not found: {models_dir}"
            
            # Check required files
            config_file = os.path.join(models_dir, "configs", "sam2.1_hiera_t512.yaml")
            checkpoint_file = os.path.join(models_dir, "checkpoints", "MedSAM2_latest.pt")
            build_sam_file = os.path.join(models_dir, "build_sam.py")
            predictor_file = os.path.join(models_dir, "sam2_image_predictor.py")
            
            missing_files = []
            for file_path, name in [
                (config_file, "Config file"),
                (checkpoint_file, "Checkpoint file"),
                (build_sam_file, "Build SAM script"),
                (predictor_file, "SAM2 predictor script")
            ]:
                if not os.path.exists(file_path):
                    missing_files.append(f"{name}: {file_path}")
            
            if missing_files:
                return False, f"Missing MEDSAM2 files:\n" + "\n".join(missing_files)
            
            return True, "MEDSAM2 environment validation passed"
            
        except Exception as e:
            return False, f"Error validating MEDSAM2 environment: {str(e)}"    @log_exception
    def run_medsam2_all_records(self, dicom_folder: str, output_dir: str, 
                               save_visualizations: bool, device: str, 
                               score_threshold: float) -> str:
        """Run MEDSAM2 annotation on all slices using user-selected coordinates"""
        if not self.selected_coordinates:
            return "Error: No coordinates selected. Please click on the image to select points."
        if not dicom_folder or not os.path.exists(dicom_folder):
            return f"Error: DICOM folder not found: {dicom_folder}"
        
        # Debug: Log the DICOM folder and check files
        logger.info(f"DICOM folder path for all records: {dicom_folder}")
        try:
            dicom_files = [f for f in os.listdir(dicom_folder) if f.lower().endswith('.dcm')]
            logger.info(f"Found {len(dicom_files)} .dcm files in folder for all records processing")
            if dicom_files:
                logger.info(f"Sample DICOM files: {dicom_files[:3]}")
            else:
                logger.warning("No .dcm files found in directory for all records")
                # Check for any files
                all_files = os.listdir(dicom_folder)
                logger.info(f"All files in directory: {all_files[:10]}")
        except Exception as e:
            logger.error(f"Error listing DICOM folder contents: {str(e)}")
            return f"Error: Cannot access DICOM folder contents: {str(e)}"
        
        try:
            # Generate the prompt JSON file for all slices using user coordinates
            success, prompt_file_or_error = self.generate_prompt_json_all_slices(dicom_folder)
            if not success:
                return f"Error generating prompts: {prompt_file_or_error}"
            
            prompt_file = prompt_file_or_error
            
            # Prepare the command with absolute paths
            script_path = os.path.join("models", "medsam2", "brain_mri_dicom_inference.py")
            if not os.path.exists(script_path):
                return f"Error: MEDSAM2 script not found at {script_path}"
            
            # Get absolute paths for checkpoint and config
            medsam2_dir = os.path.join("models", "medsam2")
            checkpoint_path = os.path.join(medsam2_dir, "checkpoints", "MedSAM2_latest.pt")
            config_path = os.path.join(medsam2_dir, "configs", "sam2.1_hiera_t512.yaml")
            
            # Verify checkpoint and config exist
            if not os.path.exists(checkpoint_path):
                return f"Error: Checkpoint not found at {checkpoint_path}"
            if not os.path.exists(config_path):
                return f"Error: Config not found at {config_path}"
            
            # Build command for all records using user prompts (NOT auto prompts)
            cmd = [
                "python", script_path,
                "--dicom_folder", dicom_folder,
                "--output_dir", output_dir,
                "--prompt_points", prompt_file,  # Use user-selected coordinates
                "--checkpoint", checkpoint_path,
                "--config", config_path,
                "--device", device,
                # NOTE: No --single_slice flag, so it processes all slices with prompts
                "--quiet"  # Reduce logging for better performance
            ]
            
            if save_visualizations:
                cmd.append("--save_visualizations")
                
            logger.info(f"Running MEDSAM2 all records command with user coordinates: {' '.join(cmd)}")
            logger.info(f"User coordinates: {self.selected_coordinates}")
            logger.info(f"Score threshold for filtering: {score_threshold}")            
            # Run the annotation
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logger.info("MEDSAM2 all records annotation completed successfully")
                # Apply score filtering to results using the new manual annotation filtering
                filtered_results = self.filter_manual_results_by_score(output_dir, score_threshold)
                return f"All records annotation completed successfully! {filtered_results}"
            else:
                logger.error(f"MEDSAM2 all records failed: {result.stderr}")            
                return f"Error running MEDSAM2 all records: {result.stderr}"
        except Exception as e:
            logger.error(f"Error running MEDSAM2 all records annotation: {str(e)}")
            return f"Error: {str(e)}"
    
    @log_exception
    def filter_manual_results_by_score(self, output_dir: str, score_threshold: float) -> str:
        """Filter manual annotation results based on score threshold - creates individual annotations for qualifying scores"""
        if not os.path.exists(output_dir):
            return "No results to filter."
        
        try:
            # Load summary.json to get scores for manual annotation
            summary_path = os.path.join(output_dir, "summary.json")
            if not os.path.exists(summary_path):
                return "No summary.json found for filtering."
            
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            
            slice_indices = summary_data.get('slice_indices', [])
            average_scores = summary_data.get('average_scores', [])
            
            if not slice_indices or not average_scores:
                return "No slice data found in summary.json"
            
            logger.info(f"Manual annotation: Processing {len(slice_indices)} slices with {len(average_scores)} scores")
            
            # Create individual annotations for each qualifying score
            valid_annotations = []
            filtered_annotations = []
            annotation_counter = 0
            
            # Check if we have multiple scores per slice
            if len(average_scores) > len(slice_indices):
                scores_per_slice = len(average_scores) // len(slice_indices)
                logger.info(f"Manual annotation: Detected {scores_per_slice} scores per slice")
                
                # Process each score individually, creating separate annotations
                for slice_idx in slice_indices:
                    slice_annotations = 0
                    for score_offset in range(scores_per_slice):
                        score_index = slice_indices.index(slice_idx) * scores_per_slice + score_offset
                        if score_index < len(average_scores):
                            score = average_scores[score_index]
                            
                            if score >= score_threshold:
                                annotation_id = f"slice_{slice_idx}_ann_{score_offset}"
                                valid_annotations.append((slice_idx, score, annotation_id))
                                # Store overlay with unique annotation ID
                                self.store_slice_overlay_with_id(output_dir, slice_idx, annotation_id, score_offset)
                                slice_annotations += 1
                                logger.info(f"Manual annotation: {annotation_id} passed with score {score:.6f}")
                            else:
                                filtered_annotations.append((slice_idx, score, f"slice_{slice_idx}_ann_{score_offset}"))
                                logger.info(f"Manual annotation: slice_{slice_idx}_ann_{score_offset} filtered with score {score:.6f}")
                    
                    if slice_annotations > 0:
                        logger.info(f"Manual annotation: Slice {slice_idx} has {slice_annotations} qualifying annotations")
            else:
                # Single score per slice - original behavior
                for slice_idx, score in zip(slice_indices, average_scores):
                    if score >= score_threshold:
                        annotation_id = f"slice_{slice_idx}_ann_0"
                        valid_annotations.append((slice_idx, score, annotation_id))
                        self.store_slice_overlay_with_id(output_dir, slice_idx, annotation_id, 0)
                        logger.info(f"Manual annotation: {annotation_id} passed with score {score:.6f}")
                    else:
                        filtered_annotations.append((slice_idx, score, f"slice_{slice_idx}_ann_0"))
                        logger.info(f"Manual annotation: slice_{slice_idx}_ann_0 filtered with score {score:.6f}")
            
            logger.info(f"Manual annotation: {len(valid_annotations)} annotations passed threshold (>= {score_threshold})")
            logger.info(f"Manual annotation: {len(filtered_annotations)} annotations filtered out")
            
            result_msg = f"Manual annotation processed {len(average_scores)} scores from {len(slice_indices)} slices. "
            result_msg += f"{len(valid_annotations)} annotations passed threshold (>= {score_threshold}), "
            result_msg += f"{len(filtered_annotations)} filtered out."
            
            # Show distribution by slice
            slice_counts = {}
            for slice_idx, _, _ in valid_annotations:
                slice_counts[slice_idx] = slice_counts.get(slice_idx, 0) + 1
            
            if slice_counts:
                result_msg += f"\nAnnotations per slice: {dict(sorted(slice_counts.items()))}"
            
            return result_msg
            
        except Exception as e:
            logger.error(f"Error in manual annotation filtering: {str(e)}")
            return f"Manual annotation filtering error: {str(e)}"    @log_exception
    def filter_automatic_results_by_score(self, output_dir: str, score_threshold: float) -> str:
        """Filter automatic annotation results based on score threshold - creates individual annotations for qualifying scores"""
        if not os.path.exists(output_dir):
            return "No results to filter."
        
        try:
            masks_dir = os.path.join(output_dir, "masks")
            if not os.path.exists(masks_dir):
                return "No masks directory found."
            
            # Get all score files directly
            score_files = [f for f in os.listdir(masks_dir) if f.endswith('_scores.npy')]
            
            if not score_files:
                return "No score files found in masks directory."
            
            valid_annotations = []
            filtered_annotations = []
            
            for score_file in score_files:
                # Extract slice number from filename (e.g., slice_0012_scores.npy -> 12)
                slice_num_str = score_file.split('_')[1]
                slice_idx = int(slice_num_str)
                
                # Load the scores for this slice
                score_path = os.path.join(masks_dir, score_file)
                scores = np.load(score_path)
                
                # Process each individual score in the slice
                for score_idx, score in enumerate(scores):
                    score_value = float(score)
                    annotation_id = f"slice_{slice_idx}_ann_{score_idx}"
                    
                    if score_value >= score_threshold:
                        valid_annotations.append((slice_idx, score_value, annotation_id))
                        # Store overlay with unique annotation ID
                        self.store_slice_overlay_with_id(output_dir, slice_idx, annotation_id, score_idx)
                        logger.info(f"Automatic annotation: {annotation_id} passed with score {score_value:.6f}")
                    else:
                        filtered_annotations.append((slice_idx, score_value, annotation_id))
                        logger.info(f"Automatic annotation: {annotation_id} filtered with score {score_value:.6f}")
            
            logger.info(f"Automatic annotation: {len(valid_annotations)} annotations passed threshold (>= {score_threshold})")
            logger.info(f"Automatic annotation: {len(filtered_annotations)} annotations filtered out")
            
            # Sort for display
            valid_annotations.sort(key=lambda x: (x[0], x[2]))  # Sort by slice, then annotation ID
            filtered_annotations.sort(key=lambda x: (x[0], x[2]))
            
            result_msg = f"Automatic annotation processed {len(score_files)} slices with {len(valid_annotations) + len(filtered_annotations)} total annotations. "
            result_msg += f"{len(valid_annotations)} annotations passed threshold (>= {score_threshold}), "
            result_msg += f"{len(filtered_annotations)} filtered out."
            
            # Show distribution by slice
            slice_counts = {}
            for slice_idx, _, _ in valid_annotations:
                slice_counts[slice_idx] = slice_counts.get(slice_idx, 0) + 1
            
            if slice_counts:
                result_msg += f"\nAnnotations per slice: {dict(sorted(slice_counts.items()))}"
                # Show sample of valid annotations
                if len(valid_annotations) <= 10:
                    result_msg += f"\nValid annotations: {[(f'#{a[0]}({a[1]:.3f})', a[2]) for a in valid_annotations]}"
                else:
                    result_msg += f"\nSample valid annotations: {[(f'#{a[0]}({a[1]:.3f})', a[2]) for a in valid_annotations[:10]]}"
                    result_msg += f" and {len(valid_annotations) - 10} more..."
            
            return result_msg
            
        except Exception as e:
            logger.error(f"Error filtering automatic results by score: {str(e)}")
            return f"Error filtering results: {str(e)}"              @log_exception
    def store_slice_overlay(self, output_dir: str, slice_idx: int) -> bool:
        """Store overlay data for a specific slice (legacy method for backward compatibility)"""
        return self.store_slice_overlay_with_id(output_dir, slice_idx, f"slice_{slice_idx}_default", 0)
    
    @log_exception
    def store_slice_overlay_with_id(self, output_dir: str, slice_idx: int, annotation_id: str, score_idx: int) -> bool:
        """Store overlay data for a specific slice with unique annotation ID"""
        try:
            # slice_idx is the UI slice number (1-based)
            # After our +1 adjustment to the prompt key, MEDSAM2 uses the same UI slice number for the output file
            slice_num_padded = str(slice_idx).zfill(4)
            mask_path = os.path.join(output_dir, "masks", f"slice_{slice_num_padded}_mask.npy")
            
            if os.path.exists(mask_path):
                mask_array = np.load(mask_path)
                
                ui_slice_number = slice_idx
                
                # Initialize slice overlay storage if it doesn't exist
                if ui_slice_number not in self.annotation_overlays:
                    self.annotation_overlays[ui_slice_number] = {}
                
                # Check if we already have an annotation for this slice to prevent duplicates
                existing_annotations = self.annotation_overlays[ui_slice_number]
                
                # For "All Records" mode, we should only store one annotation per slice
                # even if there are multiple qualifying scores
                if len(existing_annotations) > 0:
                    logger.info(f"Annotation already exists for slice {ui_slice_number}, skipping duplicate")
                    return True
                
                # Store with unique annotation ID (but avoid duplicates)
                self.annotation_overlays[ui_slice_number][annotation_id] = {
                    'mask': mask_array,
                    'output_dir': output_dir,
                    'slice_idx': slice_idx,
                    'annotation_id': annotation_id,
                    'score_idx': score_idx
                }
                
                logger.info(f"Stored overlay for UI slice {ui_slice_number} with annotation ID {annotation_id}")
                return True
            else:
                logger.warning(f"Mask file not found for UI slice {slice_idx}: {mask_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing overlay for slice {slice_idx} with ID {annotation_id}: {str(e)}")
            return False
    def enable_point_mode(self):
        """Enable point mode for coordinate capture"""
        # Ensure the attribute exists
        if not hasattr(self, 'point_mode_enabled'):
            self.point_mode_enabled = False
        self.point_mode_enabled = True
        logger.info("Point mode enabled")
    
    def disable_point_mode(self):
        """Disable point mode for coordinate capture"""
        # Ensure the attribute exists
        if not hasattr(self, 'point_mode_enabled'):
            self.point_mode_enabled = False
        self.point_mode_enabled = False
        logger.info("Point mode disabled")
    @log_exception
    def load_annotation_result(self, output_dir: str) -> Tuple[str, Optional[np.ndarray]]:
        """
        Convenience method - alias for load_annotation_results with Single Slice mode
        """
        return self.load_annotation_results(output_dir, "Single Slice")
    
    def set_annotation_workflow(self, workflow_type: str):
        """Set the annotation workflow type ('manual' or 'automatic')"""
        if workflow_type not in ['manual', 'automatic']:
            raise ValueError("Workflow type must be 'manual' or 'automatic'")
        
        self.workflow_type = workflow_type
        logger.info(f"Annotation workflow set to: {workflow_type}")
    
    def get_annotation_workflow(self) -> str:
        """Get the current annotation workflow type"""
        return getattr(self, 'workflow_type', 'manual')  # Default to manual for backward compatibility
    
    def filter_results_by_workflow(self, output_dir: str, score_threshold: float) -> str:
        """Filter results based on the current workflow type"""
        workflow = self.get_annotation_workflow()
        
        if workflow == 'automatic':
            return self.filter_automatic_results_by_score(output_dir, score_threshold)
        else:
            return self.filter_manual_results_by_score(output_dir, score_threshold)
    
    def get_slice_annotations(self, slice_idx: int) -> Dict[str, Any]:
        """Get all annotations for a specific slice"""
        if slice_idx not in self.annotation_overlays:
            return {}
        
        # Handle both old format (single annotation) and new format (multiple annotations)
        slice_data = self.annotation_overlays[slice_idx]
        
        if isinstance(slice_data, dict) and 'mask' in slice_data:
            # Old format - single annotation
            return {'default': slice_data}
        else:
            # New format - multiple annotations
            return slice_data
    
    def get_slice_annotation_count(self, slice_idx: int) -> int:
        """Get the number of annotations for a specific slice"""
        annotations = self.get_slice_annotations(slice_idx)
        return len(annotations)
    
    def get_all_slice_annotations(self) -> Dict[int, Dict[str, Any]]:
        """Get all annotations for all slices"""
        result = {}
        for slice_idx in self.annotation_overlays:
            result[slice_idx] = self.get_slice_annotations(slice_idx)
        return result
