"""
MEDSAM2 Integration Handlers

This module contains the handlers for integrating MEDSAM2 medical image annotation
into the SegMed-Pro application.

IMPORTANT NOTE ON INDEXING:
- UI slices are 1-based (starting from 1)
- MEDSAM2 expects prompt keys as strings but converts them to 0-based indices by subtracting 1
- We send prompt keys as (UI slice + 1) to compensate for MEDSAM2's internal subtraction
- MEDSAM2 outputs mask files with 0-based indices (e.g., slice_0009_mask.npy for UI slice 10)
- When loading masks, we need to use: mask index = UI slice - 1
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

logger = logging.getLogger(__name__)


class MEDSAM2Handlers:
    """Handlers for MEDSAM2 annotation operations"""
    
    def __init__(self, state):
        self.state = state
        self.selected_coordinates = []  # Store [(x, y), ...] coordinate pairs
        self.annotated_slice = None  # Store the slice number that was annotated
        self.annotation_overlays = {}  # Store overlays for each slice {slice_index: overlay_data}
        self.score_threshold = 0.3  # Default score threshold for filtering annotations
        
    def handle_image_click(self, evt: gr.SelectData) -> str:
        """Handle click events on the image to capture coordinates"""
        try:
            logger.info(f"Image click event received: {type(evt)}, data: {evt}")
            
            if evt is None:
                logger.warning("Event is None")
                return "No click data received (None event)"
                
            if not hasattr(evt, 'index'):
                logger.warning(f"Event missing index attribute. Available attributes: {dir(evt)}")
                return "No click data received (missing index)"
                
            x, y = evt.index
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
    def clear_annotation_overlays(self) -> Tuple[str, Optional[np.ndarray]]:
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
            
            # Convert to PIL Image for gradio
            from utils.visualization import make_image_for_gradio
            result_image = make_image_for_gradio(original_slice_img)
            
            return "Annotation overlays cleared", result_image
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
                labels.append(1)  # 1 for foreground point              # Create prompt structure for all slices using the same coordinates
            prompt_data = {}
            for slice_idx in range(total_slices):                # Use 0-based indexing for slice_idx, but MEDSAM2 will convert the string key to 0-based internally
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
            return f"Error: {str(e)}", None    @log_exception
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
            return f"Error: No mask file found for UI slice {self.annotated_slice}. Expected file 'slice_{slice_num_padded}_mask.npy'. Available files: {mask_files}", None

    @log_exception
    def load_all_records_results(self, output_dir: str) -> Tuple[str, Optional[np.ndarray]]:
        """Load results for all records mode - show current slice if it has valid annotation"""
        current_slice = self.state.current_slice_idx
        
        # Check if we have a stored overlay for the current slice
        if current_slice in self.annotation_overlays:
            overlay_data = self.annotation_overlays[current_slice]
            mask_array = overlay_data['mask']
            
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
            
            return f"Displaying annotation for slice {current_slice} (from all records processing)", result_image
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
            # Clear coordinates after successful annotation
            self.clear_coordinates()
            # Return tuple indicating success for UI updates
            return f"{annotation_result}\n{load_result}", result_image, True
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
                # Apply score filtering to results
                filtered_results = self.filter_results_by_score(output_dir, score_threshold)
                return f"All records annotation completed successfully! {filtered_results}"
            else:
                logger.error(f"MEDSAM2 all records failed: {result.stderr}")            
                return f"Error running MEDSAM2 all records: {result.stderr}"
        except Exception as e:
            logger.error(f"Error running MEDSAM2 all records annotation: {str(e)}")
            return f"Error: {str(e)}"

    @log_exception
    def filter_results_by_score(self, output_dir: str, score_threshold: float) -> str:
        """Filter annotation results based on score threshold and store valid overlays"""
        if not os.path.exists(output_dir):
            return "No results to filter."
        
        try:
            # Load summary.json to get average scores
            summary_path = os.path.join(output_dir, "summary.json")
            if not os.path.exists(summary_path):
                return "No summary.json found for filtering."
            
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            
            slice_indices = summary_data.get('slice_indices', [])
            average_scores = summary_data.get('average_scores', [])
            
            if len(slice_indices) != len(average_scores):
                return "Mismatch between slice indices and scores in summary."
            
            # Filter slices based on score threshold
            valid_slices = []
            filtered_slices = []
            
            for i, (slice_idx, avg_score) in enumerate(zip(slice_indices, average_scores)):
                if avg_score >= score_threshold:
                    valid_slices.append((slice_idx, avg_score))
                    # Store the overlay data for this slice
                    self.store_slice_overlay(output_dir, slice_idx)
                else:
                    filtered_slices.append((slice_idx, avg_score))
            
            logger.info(f"Valid slices (score >= {score_threshold}): {len(valid_slices)}")
            logger.info(f"Filtered slices (score < {score_threshold}): {len(filtered_slices)}")
            
            return f"Processed {len(slice_indices)} slices. {len(valid_slices)} passed threshold (>= {score_threshold}), {len(filtered_slices)} filtered out."
            
        except Exception as e:
            logger.error(f"Error filtering results by score: {str(e)}")
            return f"Error filtering results: {str(e)}"    @log_exception
    def store_slice_overlay(self, output_dir: str, slice_idx: int) -> bool:
        """Store overlay data for a specific slice"""
        try:
            # slice_idx is 0-based, used for the mask filename
            # UI slice is 1-based, used as the key in annotation_overlays
            slice_num_padded = str(slice_idx).zfill(4)
            mask_path = os.path.join(output_dir, "masks", f"slice_{slice_num_padded}_mask.npy")
            
            if os.path.exists(mask_path):
                mask_array = np.load(mask_path)
                # Store in our overlay cache using 1-based indexing to match UI
                ui_slice_number = slice_idx + 1
                self.annotation_overlays[ui_slice_number] = {
                    'mask': mask_array,
                    'output_dir': output_dir,
                    'slice_idx': slice_idx  # Keep 0-based for file operations
                }
                logger.info(f"Stored overlay for UI slice {ui_slice_number} (file slice {slice_idx})")
                return True
            else:
                logger.warning(f"Mask file not found for UI slice {slice_idx + 1} (file slice {slice_idx}): {mask_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing overlay for slice {slice_idx}: {str(e)}")
            return False
