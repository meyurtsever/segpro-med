"""
Med-R1 Handler for SegMed-Pro

This module provides functionality to integrate Med-R1 visual language model
for medical image analysis in the Editor tab using a persistent service for fast inference.
Med-R1 is specifically optimized for medical imaging with MRI checkpoints and 384x384 image requirements.
"""

import logging
import os
import tempfile
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image

# Add models directory to path for import
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
if models_dir not in sys.path:
    sys.path.append(models_dir)

# Add med-r1 directory to path
med_r1_dir = os.path.join(models_dir, "med-r1")
if med_r1_dir not in sys.path:
    sys.path.append(med_r1_dir)

try:
    from med_r1_service import get_service, cleanup_service
except ImportError:
    # Fallback if service is not available
    logger = logging.getLogger(__name__)
    logger.warning("Med-R1 service not available - check installation")
    get_service = None
    cleanup_service = None

logger = logging.getLogger(__name__)


class MedR1Handlers:
    """Handlers for Med-R1 visual language model operations"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self._service = None
        
    def _get_service(self):
        """Get or initialize the Med-R1 service with meta tensor handling"""
        if self._service is None and get_service is not None:
            try:
                # Initialize service (it will auto-detect local MRI checkpoint)
                logger.info("Loading Med-R1 service (this may take a moment on first use)...")
                self._service = get_service(device="auto")
                logger.info("Med-R1 service initialized successfully")
            except Exception as e:
                error_msg = str(e)
                if "meta tensor" in error_msg.lower():
                    logger.error(f"Med-R1 meta tensor error (known PyTorch issue): {e}")
                    logger.info("Trying alternative loading method...")
                    try:
                        # Try with explicit device specification to avoid meta tensor issues
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        logger.info(f"Attempting Med-R1 load with explicit device: {device}")
                        self._service = get_service(device=device)
                        logger.info("Med-R1 service initialized successfully with explicit device")
                    except Exception as e2:
                        logger.error(f"Failed to initialize Med-R1 service even with explicit device: {e2}")
                        self._service = None
                else:
                    logger.error(f"Failed to initialize Med-R1 service: {e}")
                    self._service = None
        return self._service
        
    def run_med_r1_inference(self, image_annotator_value: Optional[dict], identify_anomalies: bool = True, describe_slice: bool = False, modality: str = "MRI") -> str:
        """
        Run Med-R1 VLM inference on the current image from image_annotator using persistent service
        
        Args:
            image_annotator_value: Current value from the image_annotator component
            identify_anomalies: Whether to use anomaly identification prompt
            describe_slice: Whether to use general description prompt
            modality: Imaging modality (MRI, MG for mammography, CT, etc.)
            
        Returns:
            str: Med-R1 generated caption or error message
        """
        try:
            if get_service is None:
                return "Error: Med-R1 service not available. Please check installation and requirements."
            
            if image_annotator_value is None or "image" not in image_annotator_value:
                return "Error: No image available for Med-R1 analysis"
            
            # Extract image from annotator value
            image_array = image_annotator_value["image"]
            if image_array is None:
                return "Error: Image data is None"
            
            # Create Med-R1 compatible prompts based on modality (following official script format)
            # Official script uses: "First output the thinking process in <think> </think> and final choice"
            if modality == "MG":
                if identify_anomalies:
                    base_prompt = ("Analyze this mammogram for any abnormal findings. "
                                  "Identify any masses, microcalcifications, architectural distortions, or asymmetries. "
                                  "Describe the location, characteristics (shape, margins, density), and BI-RADS category if applicable. "
                                  "If no abnormalities are detected, state that the mammogram appears normal.")
                    prompt = f"{base_prompt} First output your thinking process in <think> </think> tags and then provide your final analysis."
                elif describe_slice:
                    base_prompt = ("Provide a detailed analysis of this mammogram. "
                                  "Describe the breast tissue composition (density category), anatomical structures visible, "
                                  "imaging quality, positioning adequacy, and any notable features. "
                                  "Include information about tissue distribution and any visible anatomical landmarks.")
                    prompt = f"{base_prompt} First output your thinking process in <think> </think> tags and then provide your final analysis."
                else:
                    base_prompt = ("Analyze this mammogram image. "
                                  "Describe the breast tissue composition and any notable findings.")
                    prompt = f"{base_prompt} First output your thinking process in <think> </think> tags and then provide your final analysis."
            else:  # MRI or other
                if identify_anomalies:
                    base_prompt = ("Analyze this brain MRI scan for any abnormal findings. "
                                  "Identify any lesions, masses, hemorrhages, infarcts, or other pathological changes. "
                                  "Describe the location, size, and characteristics of any abnormalities found. "
                                  "If no abnormalities are detected, state that the scan appears normal.")
                    prompt = f"{base_prompt} First output your thinking process in <think> </think> tags and then provide your final analysis."
                elif describe_slice:
                    base_prompt = ("Provide a detailed medical analysis of this brain MRI slice. "
                                  "Describe the anatomical structures visible, imaging quality, "
                                  "slice level, and any notable features or findings. "
                                  "Include information about brain symmetry, ventricles, and tissue contrast.")
                    prompt = f"{base_prompt} First output your thinking process in <think> </think> tags and then provide your final analysis."
                else:
                    # Fallback prompt if neither is selected
                    base_prompt = ("Analyze this medical brain MRI image. "
                                  "Describe the visible anatomical structures and any notable findings.")
                    prompt = f"{base_prompt} First output your thinking process in <think> </think> tags and then provide your final analysis."
            
            logger.info(f"Running Med-R1 inference with prompt: {prompt[:50]}...")
            logger.info(f"Image shape: {image_array.shape}")
            
            # Convert numpy array to PIL Image
            if isinstance(image_array, np.ndarray):
                # Ensure the image is in the correct format (0-255, uint8)
                if image_array.dtype != np.uint8:
                    # Normalize to 0-255 if needed
                    if image_array.max() <= 1.0:
                        image_array = (image_array * 255).astype(np.uint8)
                    else:
                        image_array = image_array.astype(np.uint8)
                  
                # Convert to PIL Image
                if len(image_array.shape) == 3:
                    image = Image.fromarray(image_array, 'RGB')
                elif len(image_array.shape) == 2:
                    # Grayscale to RGB
                    image_rgb = np.stack([image_array] * 3, axis=-1)
                    image = Image.fromarray(image_rgb, 'RGB')
                else:
                    return "Error: Unsupported image shape for Med-R1 processing"
            else:
                return "Error: Image data is not a numpy array"
                # Save image to temporary file (Med-R1 needs file path)
            temp_image_path = self._save_temp_image(image)
            if temp_image_path is None:
                return "Error: Failed to save temporary image"
            
            # Also save a persistent copy for debugging/reference
            self._save_processed_image(image)
            
            try:
                # Get the persistent service
                service = self._get_service()
                if service is None:
                    return "Error: Failed to initialize Med-R1 service"
                
                # Run inference using the persistent service
                result = service.generate_caption(temp_image_path, prompt, max_tokens=256)
                
                if result["success"]:
                    timings = result["timings"]
                    logger.info(f"Med-R1 inference completed in {timings['total_time']:.2f}s "
                              f"(generation: {timings['text_generation']:.2f}s)")
                    
                    # Format the response nicely
                    caption = result["caption"]
                    
                    # Add timing information if in debug mode
                    if logger.isEnabledFor(logging.DEBUG):
                        caption += f"\n\n*Generated in {timings['total_time']:.2f}s*"
                    
                    return caption
                else:
                    logger.error(f"Med-R1 service error: {result['error']}")
                    return f"Error: {result['error']}"
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            logger.error(f"Error in Med-R1 inference: {e}")
            return f"Error: {str(e)}"
    
    def suggest_labels_for_annotations(self, image_annotator_value: Optional[dict]) -> str:
        """
        Suggest semantic labels for user-drawn annotations using Med-R1 VLM
        
        This method captures the current annotated image and asks Med-R1 to identify
        and suggest labels for the highlighted/segmented regions.
        
        Args:
            image_annotator_value: Current value from the image_annotator component
                                 including both the image and annotations
            
        Returns:
            str: Med-R1 generated label suggestions or error message
        """
        try:
            if get_service is None:
                return "Error: Med-R1 service not available. Please check installation and requirements."
            
            if image_annotator_value is None:
                return "Error: No image available for Med-R1 label analysis"
            
            # Check if there are any annotations using the same method as export handlers
            boxes = image_annotator_value.get("boxes", [])
            if not boxes:
                return "No annotations found. Please draw some annotations (polygons, boxes, etc.) on the image first, then try again."
            
            logger.info(f"Found {len(boxes)} annotation boxes for label suggestion")
            
            # Get the base image
            base_image = image_annotator_value.get("image", None)
            if base_image is None:
                return "Error: No base image found in image_annotator_value"
            
            logger.info(f"Base image shape: {base_image.shape}")
            logger.info(f"Annotation boxes: {[{k: v for k, v in box.items() if k not in ['points']} for box in boxes[:2]]}")  # Log first 2 boxes without points
            
            # Use the same method as export handlers to create composite image with annotations
            # This ensures we get the exact same result as "Include Overlays (PNG)" export
            composite_image = self._create_overlay_image_like_export(base_image, boxes)
            if composite_image is None:
                return "Error: Failed to create composite image with annotations"
            
            # Save debug images to understand what we're working with
            debug_base_path = self._save_processed_image(
                Image.fromarray((base_image * 255).astype(np.uint8) if base_image.max() <= 1 else base_image.astype(np.uint8)), 
                suffix="_debug_base"
            )
            debug_composite_path = self._save_processed_image(composite_image, suffix="_debug_composite")
            logger.info(f"Saved debug images: {debug_base_path}, {debug_composite_path}")
            
            # Create annotation info for the prompt
            annotation_info = self._format_annotation_info_from_boxes(boxes)
            
            # Create a specialized prompt for label suggestion that mentions the visible overlays
            '''
            prompt = (
                f"This is a brain MRI image with {len(boxes)} highlighted annotation region(s) overlaid in color. "
                f"The colored overlays mark specific areas: {annotation_info}. "
                f"Please analyze each colored/highlighted region and suggest appropriate semantic labels. "
                f"For each visible overlay, identify what anatomical structure, pathology, or tissue type "
                f"that region most likely represents. Consider: brain regions (frontal lobe, temporal lobe, "
                f"parietal lobe, occipital lobe, cerebellum, brainstem), tissue types (gray matter, white matter, CSF), "
                f"and potential pathologies (lesions, tumors, hemorrhages, infarcts). "
                f"Format your response as: Region 1 (color overlay): [specific label and brief explanation], "
                f"Region 2 (color overlay): [specific label and brief explanation], etc. "
                f"Use standard medical terminology and be specific about anatomical locations."
            )
            '''
            '''
            prompt = (
                f"This is a brain MRI image with {len(boxes)} annotated region(s). "
                f"The annotations are: {annotation_info}. "
                f"For each highlighted/marked region in this image, please provide specific semantic labels. "
                f"Identify what anatomical structure, pathology, or tissue type each marked region represents. "
                f"Consider: brain regions (frontal lobe, temporal lobe, etc.), pathologies (tumor, lesion, hemorrhage, etc.), "
                f"or tissue types (gray matter, white matter, CSF, etc.). "
                f"Format your response as: Region 1: [label], Region 2: [label], etc. "
                f"Be specific and use standard medical terminology."
            )
            '''
            
            prompt = (
    f"This is a brain MRI image with {len(boxes)} bounding box annotation(s), each highlighting a region of interest. "
    f"The image includes visible overlays. Your task is to identify what structure, tissue, or abnormality is inside each box. "
    f"For each region, suggest a single semantic label such as an anatomical part (e.g., eye, ventricle, corpus callosum), "
    f"a pathology (e.g., tumor, hemorrhage, lesion), or a tissue type (e.g., gray matter, white matter, CSF). "
    f"Use standard medical terminology and format your response like this: "
    f"Region 1: [label], Region 2: [label], etc. "
    f"First output the thinking process in <think> </think> and final choice in <answer> </answer> tags."
)



            logger.info(f"Running Med-R1 label suggestion with {len(boxes)} annotations")
            logger.info(f"Prompt: {prompt[:100]}...")
            
            # Save composite image to temporary file (Med-R1 needs file path)
            temp_image_path = self._save_temp_image(composite_image)
            if temp_image_path is None:
                return "Error: Failed to save temporary image"
            
            try:
                # Get the persistent service
                service = self._get_service()
                if service is None:
                    return "Error: Failed to initialize Med-R1 service"
                
                # Run inference using the persistent service
                result = service.generate_caption(temp_image_path, prompt, max_tokens=256)
                
                if result["success"]:
                    timings = result["timings"]
                    logger.info(f"Med-R1 label suggestion completed in {timings['total_time']:.2f}s "
                              f"(generation: {timings['text_generation']:.2f}s)")
                    
                    # Format the response nicely
                    caption = result["caption"]
                    
                    # Add some context about the analysis
                    formatted_response = (
                        f"🏷️ **Label Suggestions for {len(boxes)} Annotations:**\n\n"
                        f"{caption}\n\n"
                        f"📝 *Analysis based on colored overlay regions in the image.*"
                    )
                    
                    # Add timing information if in debug mode
                    if logger.isEnabledFor(logging.DEBUG):
                        formatted_response += f"\n\n*Generated in {timings['total_time']:.2f}s*"
                    
                    return formatted_response
                else:
                    logger.error(f"Med-R1 service error: {result['error']}")
                    return f"Error: {result['error']}"
                    
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass  # Ignore cleanup errors
                    
        except Exception as e:
            logger.error(f"Error in Med-R1 label suggestion: {e}")
            return f"Error: {str(e)}"
    
    def _create_composite_image_with_annotations(self, base_image: np.ndarray, annotations: list) -> Optional[Image.Image]:
        """
        Create a composite image with annotation overlays drawn on top
        
        Args:
            base_image: Base image array
            annotations: List of annotation objects
            
        Returns:
            PIL Image with annotations overlaid, or None if failed
        """
        try:
            import cv2
            
            # Ensure the image is in the correct format (0-255, uint8)
            if base_image.dtype != np.uint8:
                if base_image.max() <= 1.0:
                    image_array = (base_image * 255).astype(np.uint8)
                else:
                    image_array = base_image.astype(np.uint8)
            else:
                image_array = base_image.copy()
            
            # Convert to RGB if grayscale
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            # Define colors for different annotations (same as UI)
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            # Draw each annotation
            for i, annotation in enumerate(annotations):
                color = colors[i % len(colors)]
                annotation_type = annotation.get("type", "unknown")
                coordinates = annotation.get("coordinates", [])
                
                if not coordinates:
                    continue
                
                try:
                    if annotation_type == "box" and len(coordinates) >= 2:
                        # Draw bounding box
                        x1, y1 = int(coordinates[0][0]), int(coordinates[0][1])
                        x2, y2 = int(coordinates[1][0]), int(coordinates[1][1])
                        cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 3)
                        
                    elif annotation_type == "polygon" and len(coordinates) >= 3:
                        # Draw polygon
                        points = np.array([[int(coord[0]), int(coord[1])] for coord in coordinates], np.int32)
                        cv2.polylines(image_array, [points], True, color, 3)
                        # Fill with semi-transparent color
                        overlay = image_array.copy()
                        cv2.fillPoly(overlay, [points], color)
                        image_array = cv2.addWeighted(image_array, 0.7, overlay, 0.3, 0)
                        
                    elif annotation_type in ["brush", "point"]:
                        # Draw points/brush strokes
                        for coord in coordinates:
                            if len(coord) >= 2:
                                cv2.circle(image_array, (int(coord[0]), int(coord[1])), 5, color, -1)
                                
                except Exception as e:
                    logger.warning(f"Failed to draw annotation {i}: {e}")
                    continue
            
            # Convert back to PIL Image
            composite_image = Image.fromarray(image_array, 'RGB')
            return composite_image
            
        except Exception as e:
            logger.error(f"Error creating composite image: {e}")
            return None
    
    def _create_overlay_image_like_export(self, base_image: np.ndarray, boxes: list) -> Optional[Image.Image]:
        """
        Create a composite image with annotation overlays using the same method as export handlers
        This ensures the result matches exactly what "Include Overlays (PNG)" export produces
        
        Args:
            base_image: Base image array
            boxes: List of annotation box objects from image_annotator
            
        Returns:
            PIL Image with annotations overlaid, or None if failed
        """
        try:
            from PIL import ImageDraw
            
            # Ensure the image is in the correct format (0-255, uint8)
            if base_image.dtype != np.uint8:
                if base_image.max() <= 1.0:
                    img_array = (base_image * 255).astype(np.uint8)
                else:
                    img_array = base_image.astype(np.uint8)
            else:
                img_array = base_image.copy()
            
            # Convert to RGB if grayscale
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                img_array = np.stack([img_array.squeeze()] * 3, axis=-1)
            
            # Use the same drawing method as export handlers
            pil_img = Image.fromarray(img_array)
            
            # Create a transparent overlay layer for filled shapes (same as export)
            overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Also draw on the base image for outlines (same as export)
            base_draw = ImageDraw.Draw(pil_img)
            
            for box in boxes:
                color = tuple(box.get('color', (255, 0, 0)))
                # Create semi-transparent fill color (50% opacity) - same as export
                fill_color = color + (128,)  # Add alpha channel
                
                if box.get('type') == 'polygon' and 'points' in box:
                    # Draw FILLED polygon for overlay export (same as export)
                    points = [(p['x'], p['y']) for p in box['points']]
                    # Fill with semi-transparent color on overlay
                    overlay_draw.polygon(points, fill=fill_color)
                    # Draw outline on base image for visibility
                    base_draw.polygon(points, outline=color, width=2)
                    
                elif all(k in box for k in ('xmin','ymin','xmax','ymax')):
                    # Draw FILLED rectangle for overlay export (same as export)
                    coords = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
                    # Fill with semi-transparent color on overlay
                    overlay_draw.rectangle(coords, fill=fill_color)
                    # Draw outline on base image for visibility  
                    base_draw.rectangle(coords, outline=color, width=2)
            
            # Composite the overlay onto the base image (same as export)
            pil_img = pil_img.convert('RGBA')
            composite = Image.alpha_composite(pil_img, overlay)
            return composite.convert('RGB')
            
        except Exception as e:
            logger.error(f"Error creating overlay image like export: {e}")
            return None
    
    def _format_annotation_info(self, annotations: list) -> str:
        """
        Format annotation information for the prompt
        
        Args:
            annotations: List of annotation objects from image_annotator
            
        Returns:
            str: Formatted string describing the annotations
        """
        try:
            annotation_descriptions = []
            for i, annotation in enumerate(annotations, 1):
                # Extract annotation type and basic info
                annotation_type = annotation.get("type", "unknown")
                label = annotation.get("label", "unlabeled")
                
                # Format coordinates or bounding box info if available
                if "coordinates" in annotation:
                    coords = annotation["coordinates"]
                    if isinstance(coords, list) and len(coords) > 0:
                        if annotation_type == "box":
                            annotation_descriptions.append(f"box region '{label}'")
                        elif annotation_type == "polygon":
                            annotation_descriptions.append(f"polygon region '{label}' with {len(coords)} points")
                        else:
                            annotation_descriptions.append(f"{annotation_type} region '{label}'")
                    else:
                        annotation_descriptions.append(f"{annotation_type} region '{label}'")
                else:
                    annotation_descriptions.append(f"{annotation_type} region '{label}'")
            
            return ", ".join(annotation_descriptions)
        except Exception as e:
            logger.warning(f"Error formatting annotation info: {e}")
            return "annotated regions"

    def _format_annotation_info_from_boxes(self, boxes: list) -> str:
        """
        Format annotation information from boxes for the prompt
        
        Args:
            boxes: List of annotation box objects from image_annotator
            
        Returns:
            str: Formatted string describing the annotations
        """
        try:
            annotation_descriptions = []
            for i, box in enumerate(boxes, 1):
                # Extract annotation type and basic info
                annotation_type = box.get("type", "unknown")
                label = box.get("label", "unlabeled")
                
                # Format based on annotation type
                if annotation_type == "polygon" and "points" in box:
                    num_points = len(box["points"])
                    annotation_descriptions.append(f"polygon region '{label}' with {num_points} points")
                elif "xmin" in box and "ymin" in box and "xmax" in box and "ymax" in box:
                    annotation_descriptions.append(f"box region '{label}'")
                else:
                    annotation_descriptions.append(f"{annotation_type} region '{label}'")
            
            return ", ".join(annotation_descriptions)
        except Exception as e:
            logger.warning(f"Error formatting annotation info from boxes: {e}")
            return "annotated regions"

    def _save_processed_image(self, image: Image.Image, suffix: str = "") -> Optional[str]:
        """
        Save a persistent copy of the processed image for debugging/reference
        
        Args:
            image: PIL Image object
            suffix: Optional suffix to add to filename
            
        Returns:
            str: Path to saved image file or None if failed
        """
        try:
            # Create the filename in the current working directory
            output_path = f"latest_processed_med_r1{suffix}.jpg"
            
            # Convert image to RGB if it's not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 384x384 as required by Med-R1
            image_resized = image.resize((384, 384), Image.Resampling.LANCZOS)
            
            # Save image as JPG for maximum compatibility
            image_resized.save(output_path, 'JPEG', quality=95)
            logger.info(f"Saved processed Med-R1 image (384x384) to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed Med-R1 image: {e}")
            return None

    def _save_temp_image(self, image: Image.Image) -> Optional[str]:
        """
        Save PIL Image to temporary file with Med-R1 requirements (384x384)
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Path to temporary image file or None if failed
        """
        try:
            # Create temporary file with .jpg extension for better compatibility
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
            
            # Convert image to RGB if it's not already (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to 384x384 as required by Med-R1
            image_resized = image.resize((384, 384), Image.Resampling.LANCZOS)
              
            # Save image as JPG for maximum compatibility
            image_resized.save(temp_path, 'JPEG', quality=95)
            logger.info(f"Saved temporary Med-R1 image (384x384) to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            return None
    
    def cleanup(self):
        """Clean up the Med-R1 service resources"""
        if self._service is not None and cleanup_service is not None:
            try:
                cleanup_service()
                self._service = None
                logger.info("Med-R1 service cleaned up successfully")
            except Exception as e:
                logger.error(f"Error cleaning up Med-R1 service: {e}")

    def is_available(self) -> bool:
        """Check if Med-R1 service is available"""
        return get_service is not None
    
    def get_model_info(self) -> dict:
        """Get information about the Med-R1 model"""
        service = self._get_service()
        if service is not None:
            return {
                "model_name": "Med-R1 (Medical VLM)",
                "checkpoint": service.checkpoint_path,
                "device": service.device,
                "loaded": service.model_loaded,
                "image_size": "384x384",
                "modality": "MRI-optimized"
            }
        else:
            return {
                "model_name": "Med-R1 (Medical VLM)",
                "checkpoint": "Not available",
                "device": "N/A",
                "loaded": False,
                "image_size": "384x384",
                "modality": "MRI-optimized"
            }
