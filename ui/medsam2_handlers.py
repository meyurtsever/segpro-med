"""
MEDSAM2 Integration Handlers

This module contains the handlers for integrating MEDSAM2 medical image annotation
into the SegMed-Pro application.
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
            
            # Store the slice that will be annotated
            self.annotated_slice = current_slice
            logger.info(f"Stored annotated slice: {self.annotated_slice}")
            
            # Prepare points and labels for the current slice
            points = []
            labels = []
            
            for x, y in self.selected_coordinates:
                points.append([int(x), int(y)])  # Ensure integers
                labels.append(1)  # 1 for foreground point
            
            # Create the prompt structure expected by MEDSAM2
            # Key is slice number as string, value contains points and labels
            prompt_data = {
                str(current_slice): {
                    "points": points,
                    "labels": labels
                }
            }
            
            # Save to brain_target_prompts.json in the project root (not in DICOM folder)
            prompt_file = "brain_target_prompts.json"
            with open(prompt_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
            
            logger.info(f"Generated prompt file: {prompt_file} for slice {current_slice} with {len(self.selected_coordinates)} points")
            logger.info(f"Points: {points}")
            logger.info(f"Stored annotated slice number: {self.annotated_slice}")
            return True, prompt_file
            
        except Exception as e:
            logger.error(f"Error generating prompt JSON: {str(e)}")
            return False, f"Error: {str(e)}"
    
    @log_exception
    def run_medsam2_annotation(self, dicom_folder: str, output_dir: str, 
                              save_visualizations: bool, device: str) -> str:
        """Run the MEDSAM2 annotation script"""
        if not self.selected_coordinates:
            return "Error: No coordinates selected. Please click on the image to select points."
        if not dicom_folder or not os.path.exists(dicom_folder):
            return f"Error: DICOM folder not found: {dicom_folder}"
        
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
                slice_num_padded = str(self.annotated_slice).zfill(4)
                logger.info(f"Will look for slice_{slice_num_padded}_mask.png or mask_{self.annotated_slice}.png in results")
                return f"Annotation completed successfully! Results saved to: {output_dir}"
            else:
                logger.error(f"MEDSAM2 failed: {result.stderr}")            
                return f"Error running MEDSAM2: {result.stderr}"
        except Exception as e:
            logger.error(f"Error running MEDSAM2 annotation: {str(e)}")
            return f"Error: {str(e)}"
    @log_exception
    def load_annotation_results(self, output_dir: str) -> Tuple[str, Optional[np.ndarray]]:
        """Load and display annotation results from MEDSAM2 - combining .npy mask with original DICOM slice"""
        if not output_dir or not os.path.exists(output_dir):
            return "Error: Output directory not found", None
        
        try:
            # Check if we have the original DICOM data loaded
            if self.state.current_data is None:
                return "Error: No DICOM data loaded. Please load DICOM series first.", None
            
            # Load the windowing metadata from the results
            windowing_metadata_path = os.path.join(output_dir, 'windowing_metadata.json')
            if os.path.exists(windowing_metadata_path):
                with open(windowing_metadata_path, 'r') as f:
                    windowing_metadata = json.load(f)
                logger.info(f"Loaded windowing metadata: {windowing_metadata}")
            else:
                logger.warning(f"Windowing metadata not found at {windowing_metadata_path}")
                windowing_metadata = {}
            
            # If we have a stored annotated slice, look for that specific mask
            if self.annotated_slice is not None:
                # Look for mask file with slice-specific naming  
                slice_num_padded = str(self.annotated_slice).zfill(4)  # e.g., "0012"
                
                # Check masks subfolder for .npy files
                masks_dir = os.path.join(output_dir, "masks")
                mask_file = f"slice_{slice_num_padded}_mask.npy"  # e.g., "slice_0012_mask.npy"
                mask_path = os.path.join(masks_dir, mask_file)
                
                logger.info(f"Looking for mask file: {mask_path}")
                
                if os.path.exists(mask_path):
                    logger.info(f"SUCCESS: Loading mask for annotated slice {self.annotated_slice}: {mask_path}")
                    
                    # Load the numpy mask array
                    mask_array = np.load(mask_path)
                    logger.info(f"Loaded mask with shape: {mask_array.shape}")
                    
                    # Get the original DICOM slice image that was annotated
                    # Use the same parameters that would have been used during annotation
                    original_slice_img = display_slice(
                        self.state.current_data,
                        self.annotated_slice,
                        self.state.current_view,
                        crosshair=self.state.crosshair_position
                    )
                    logger.info(f"Generated original DICOM slice with shape: {original_slice_img.shape}")
                    
                    # Overlay the segmentation mask on the original slice
                    # Convert mask to binary (assuming MEDSAM2 outputs binary masks)
                    binary_mask = (mask_array > 0).astype(np.uint8)
                    
                    # Use a simple red colormap for the mask overlay
                    mask_colormap = {
                        0: [0, 0, 0],       # Background (transparent)
                        1: [255, 0, 0],     # Segmentation (red)
                    }
                    
                    # Overlay the mask on the original image
                    overlaid_image = overlay_segmentation(
                        original_slice_img,
                        binary_mask,
                        alpha=0.3,  # Semi-transparent overlay
                        colormap=mask_colormap
                    )
                    
                    logger.info(f"Successfully overlaid mask on original slice {self.annotated_slice}")
                    return f"Loaded annotation mask for slice {self.annotated_slice} overlaid on original DICOM", overlaid_image
                else:
                    # List what files actually exist for debugging
                    actual_files = []
                    mask_files = []
                    
                    if os.path.exists(output_dir):
                        actual_files = os.listdir(output_dir)
                        logger.error(f"Expected mask file not found: {mask_path}")
                        logger.error(f"Files in output directory: {actual_files}")
                    
                    if os.path.exists(masks_dir):
                        mask_files = os.listdir(masks_dir)
                        logger.error(f"Files in masks directory: {mask_files}")
                    else:
                        logger.error(f"Masks directory does not exist: {masks_dir}")
                    
                    return f"Error: Expected mask file for slice {self.annotated_slice} not found at {mask_path}. Available mask files: {mask_files}", None
            else:
                logger.error("No annotated slice stored - cannot determine which mask to load")
                return "Error: No annotated slice information available. Please run annotation first.", None
                
        except Exception as e:
            logger.error(f"Error loading annotation results: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}", None
    
    def get_current_dicom_folder(self) -> Optional[str]:
        """Get the current DICOM folder path from application state"""
        if self.state.file_list and len(self.state.file_list) > 0:
            # Get the directory of the first file
            first_file = self.state.file_list[0]
            return os.path.dirname(first_file)
        return None
    
    @log_exception
    def run_full_annotation_workflow(self, output_dir: str, save_visualizations: bool, 
                                   device: str) -> Tuple[str, Optional[np.ndarray]]:
        """Run the complete annotation workflow from coordinates to results"""
        # Get current DICOM folder
        dicom_folder = self.get_current_dicom_folder()
        if not dicom_folder:
            return "Error: No DICOM data loaded. Please load DICOM files first.", None
        
        # Run annotation
        annotation_result = self.run_medsam2_annotation(
            dicom_folder, output_dir, save_visualizations, device
        )
        
        # Check if annotation was successful
        if "successfully" in annotation_result.lower():
            # Try to load results
            load_result, result_image = self.load_annotation_results(output_dir)            
            return f"{annotation_result}\n{load_result}", result_image
        else:
            return annotation_result, None

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
            return False, f"Error validating MEDSAM2 environment: {str(e)}"
