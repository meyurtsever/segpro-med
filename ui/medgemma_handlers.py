"""
MedGemma-4B Handler for SegMed-Pro

This module provides functionality to integrate MedGemma-4B visual language model
for medical image analysis in the Editor tab using a persistent service for fast inference.
MedGemma-4B is Google's specialized medical model optimized for medical imaging tasks.
"""

import logging
import os
import tempfile
import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image

# Add models directory to path for import
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
if models_dir not in sys.path:
    sys.path.append(models_dir)

# Add medgemma directory to path
medgemma_dir = os.path.join(models_dir, "medgemma")
if medgemma_dir not in sys.path:
    sys.path.append(medgemma_dir)

try:
    from medgemma.medgemma_service import get_service, cleanup_service, get_medical_prompt
except ImportError:
    # Fallback if service is not available
    logger = logging.getLogger(__name__)
    logger.warning("MedGemma service not available - check installation")
    get_service = None
    cleanup_service = None
    get_medical_prompt = None

# Behavioral analytics tracking
from analytics.tracking_integration import (
    track_vlm_analysis, track_label_suggestion, track_label_accepted
)

logger = logging.getLogger(__name__)

# Load modality-specific prompts
def load_vlm_prompts(modality: str = "MRI") -> Dict:
    """Load VLM prompts based on modality
    
    Args:
        modality: Imaging modality - "MRI", "MG" (mammography), "CT" (abdomen/chest)
    """
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Determine which prompt file to use based on modality
        modality_upper = modality.upper() if modality else "MRI"
        if modality_upper == "MG":
            prompt_file = "mg_vlm_prompts.json"
        elif modality_upper == "CT":
            prompt_file = "ct_vlm_prompts.json"
        else:
            # Default to MRI for brain and unknown modalities
            prompt_file = "mri_vlm_prompts.json"
        
        prompt_path = os.path.join(base_dir, "prompts", prompt_file)
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Prompt file not found: {prompt_path}, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        return {}


class MedGemmaHandlers:
    """Handlers for MedGemma-4B visual language model operations"""
    
    def __init__(self, state):
        """Initialize with application state"""
        self.state = state
        self._service = None
        
    def _get_service(self):
        """Get or initialize the MedGemma service"""
        if self._service is None and get_service is not None:
            try:
                self._service = get_service(device="auto")
                logger.info("MedGemma service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MedGemma service: {e}")
                self._service = None
        return self._service
        
    def run_vlm_inference(self, image_annotator_value: Optional[dict], identify_anomalies: bool = True, describe_slice: bool = False, modality: str = "MRI") -> str:
        """
        Run MedGemma VLM inference on the current image from image_annotator
        
        Args:
            image_annotator_value: Value from the image_annotator component containing image data
            identify_anomalies: Whether to focus on identifying anomalies
            describe_slice: Whether to provide general description
            modality: Imaging modality (MRI, MG for mammography, CT, etc.)
            
        Returns:
            VLM analysis result as string
        """
        try:
            # Check if service is available
            service = self._get_service()
            if service is None:
                return "MedGemma service not available. Please check installation and try again."
            
            # Validate input
            if not image_annotator_value:
                return "No image data available. Please load a DICOM or medical image first."
            
            # Extract image from image_annotator
            image_data = self._extract_image_from_annotator(image_annotator_value)
            if image_data is None:
                return "Could not extract image data from the annotator. Please ensure an image is loaded."
            
            # Build prompt based on user selections and modality
            prompt = self._build_prompt(identify_anomalies, describe_slice, modality)
            logger.info(f"Running MedGemma inference with prompt type: {self._get_prompt_type(identify_anomalies, describe_slice)} for modality: {modality}")
            
            # Run inference
            try:
                response = service.generate_response(
                    image=image_data,
                    prompt=prompt,
                    max_new_tokens=256
                )
                
                if response and len(response.strip()) > 0:
                    # Track VLM analysis for behavioral analytics
                    try:
                        analysis_type = self._get_prompt_type(identify_anomalies, describe_slice)
                        current_slice_idx = getattr(self.state, 'current_slice_idx', None)
                        track_vlm_analysis(
                            model="medgemma",
                            analysis_type=analysis_type,
                            slice_idx=current_slice_idx
                        )
                    except Exception as track_e:
                        logger.debug(f"Behavioral tracking skipped: {track_e}")
                    
                    return f"MedGemma-4B Analysis:\n\n{response}"
                else:
                    return "MedGemma generated an empty response. The image might not be suitable for analysis or the model encountered an issue."
                    
            except Exception as e:
                logger.error(f"MedGemma inference failed: {e}")
                
                # Track failed VLM analysis
                try:
                    current_slice_idx = getattr(self.state, 'current_slice_idx', None)
                    track_vlm_analysis(
                        model="medgemma",
                        analysis_type=self._get_prompt_type(identify_anomalies, describe_slice),
                        slice_idx=current_slice_idx
                    )
                except Exception as track_e:
                    logger.debug(f"Behavioral tracking skipped: {track_e}")
                
                return f"MedGemma inference failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in MedGemma VLM inference: {e}")
            return f"Error during MedGemma analysis: {str(e)}"
    
    def suggest_labels_for_annotations(self, image_annotator_value: Optional[dict], modality: str = "MRI") -> str:
        """
        Use MedGemma to suggest labels for anatomical/pathological structures in the image
        
        Args:
            image_annotator_value: Value from the image_annotator component
            modality: Imaging modality (MRI, MG, CT) for modality-specific prompts
            
        Returns:
            Suggested labels as a comma-separated string
        """
        try:
            # Check if service is available
            service = self._get_service()
            if service is None:
                return "MedGemma service not available for label suggestions."
            
            # Validate input
            if not image_annotator_value:
                return "No image data available for label suggestions."
            
            # Extract image from image_annotator
            image_data = self._extract_image_from_annotator(image_annotator_value)
            if image_data is None:
                return "Could not extract image for label suggestions."
            
            # Load modality-specific prompts
            prompts = load_vlm_prompts(modality)
            
            # Use modality-specific label suggestion prompt
            if prompts and "suggest_labels" in prompts:
                prompt = prompts["suggest_labels"]
                logger.info(f"Using {modality} specific prompt for label suggestions")
            elif get_medical_prompt:
                prompt = get_medical_prompt("suggest_labels")
            else:
                # Fallback generic prompt
                prompt = ("Please list the anatomical structures and any pathological findings that are visible in this medical image. "
                         "Return a comma-separated list of possible labels.")
            
            logger.info(f"Running MedGemma label suggestion inference for modality: {modality}")
            
            # Run inference
            try:
                response = service.generate_response(
                    image=image_data,
                    prompt=prompt,
                    max_new_tokens=128  # Shorter for label suggestions
                )
                
                if response and len(response.strip()) > 0:
                    # Track label suggestion for behavioral analytics
                    try:
                        # Parse suggested labels from response
                        # Handle both comma-separated and list-format responses
                        import re
                        suggested_labels = []
                        
                        # Try to find "item:" format first (list format)
                        item_matches = re.findall(r'item:\s*([^\n]+)', response, re.IGNORECASE)
                        if item_matches:
                            # Remove duplicates while preserving order
                            seen = set()
                            for item in item_matches:
                                item_clean = item.strip().strip('"\'')
                                if item_clean and item_clean.lower() not in seen:
                                    seen.add(item_clean.lower())
                                    suggested_labels.append(item_clean)
                        else:
                            # Fallback to comma-separated format
                            suggested_labels = [l.strip() for l in response.split(',') if l.strip()]
                        
                        track_label_suggestion(
                            model="medgemma",
                            labels=suggested_labels
                        )
                    except Exception as track_e:
                        logger.debug(f"Behavioral tracking skipped: {track_e}")
                    
                    return f"MedGemma Label Suggestions:\n{response}"
                else:
                    return "MedGemma could not generate label suggestions for this image."
                    
            except Exception as e:
                logger.error(f"MedGemma label suggestion failed: {e}")
                
                # Track failed label suggestion
                try:
                    track_label_suggestion(model="medgemma", labels=[])
                except Exception as track_e:
                    logger.debug(f"Behavioral tracking skipped: {track_e}")
                
                return f"Label suggestion failed: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in MedGemma label suggestion: {e}")
            return f"Error during label suggestion: {str(e)}"
    
    def _extract_image_from_annotator(self, annotator_value: dict) -> Optional[np.ndarray]:
        """
        Extract image data from the image_annotator component value
        
        Args:
            annotator_value: Dictionary containing image_annotator data
            
        Returns:
            numpy array of image data or None if extraction fails
        """
        try:
            if not annotator_value:
                return None
            
            # Debug: Log what keys are available in the annotator value
            logger.debug(f"Annotator value keys: {list(annotator_value.keys()) if annotator_value else 'None'}")
            
            # Try to get the image from the annotator (standard key is "image")
            image_data = annotator_value.get("image")
            if image_data is None:
                logger.warning("No image found in annotator value")
                # Try alternative keys as fallback
                image_data = annotator_value.get("background")
                if image_data is None:
                    return None
                logger.info("Found image data using 'background' key as fallback")
            
            # Debug: Log image data type and shape if possible
            if isinstance(image_data, np.ndarray):
                logger.debug(f"Image data type: {image_data.dtype}, shape: {image_data.shape}")
            else:
                logger.debug(f"Image data type: {type(image_data)}")
            
            # Handle different image formats
            if isinstance(image_data, str):
                # If it's a file path, load the image
                if os.path.exists(image_data):
                    image = Image.open(image_data)
                    return np.array(image)
                else:
                    logger.warning(f"Image path does not exist: {image_data}")
                    return None
            elif isinstance(image_data, np.ndarray):
                # Direct numpy array - this is the expected format
                return image_data
            elif hasattr(image_data, 'array'):
                # PIL Image or similar
                return np.array(image_data)
            else:
                logger.warning(f"Unknown image format: {type(image_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting image from annotator: {e}")
            return None
    
    def _build_prompt(self, identify_anomalies: bool, describe_slice: bool, modality: str = "MRI") -> str:
        """
        Build the appropriate prompt based on user selections and modality
        
        Args:
            identify_anomalies: Whether to focus on anomaly detection
            describe_slice: Whether to provide general description
            modality: Imaging modality (MRI, MG, CT, etc.)
            
        Returns:
            Formatted prompt string
        """
        # Load modality-specific prompts from JSON files
        prompts = load_vlm_prompts(modality)
        
        # If prompts are loaded from JSON, use them
        if prompts:
            if identify_anomalies and describe_slice:
                # Combined prompt
                anomaly_prompt = prompts.get("identify_anomalies", "")
                describe_prompt = prompts.get("describe_slice", "")
                if anomaly_prompt and describe_prompt:
                    return f"{anomaly_prompt} Additionally, {describe_prompt.lower()}"
            elif identify_anomalies:
                return prompts.get("identify_anomalies", "")
            elif describe_slice:
                return prompts.get("describe_slice", "")
        
        # Fallback to get_medical_prompt for MRI (original behavior)
        if get_medical_prompt and modality != "MG":
            if identify_anomalies and describe_slice:
                # Combined prompt
                return (get_medical_prompt("identify_anomalies") + " " +
                       "Additionally, " + get_medical_prompt("describe_slice").lower())
            elif identify_anomalies:
                return get_medical_prompt("identify_anomalies")
            elif describe_slice:
                return get_medical_prompt("describe_slice")
            else:
                # Default to description if neither is selected
                return get_medical_prompt("describe_slice")
        else:
            # Fallback prompts if service not available
            if modality == "MG":
                if identify_anomalies and describe_slice:
                    return ("Analyze this mammogram for any abnormalities or suspicious findings, "
                           "and also provide a general description of the breast tissue and structures visible.")
                elif identify_anomalies:
                    return ("Analyze this mammogram and identify any masses, calcifications, architectural distortions, "
                           "or other abnormalities that may be present. Use BI-RADS terminology.")
                elif describe_slice:
                    return ("Describe this mammogram, identifying the breast composition, anatomical structures "
                           "and overall characteristics visible in the image.")
                else:
                    return "Analyze and describe this mammogram."
            else:  # MRI or other
                if identify_anomalies and describe_slice:
                    return ("Analyze this brain MRI slice for any abnormalities or anomalies, "
                           "and also provide a general description of the anatomical structures visible.")
                elif identify_anomalies:
                    return ("Analyze this brain MRI slice and identify any abnormalities, anomalies, "
                           "or pathological regions that may be present.")
                elif describe_slice:
                    return ("Describe this brain MRI slice, identifying the anatomical structures "
                           "and overall characteristics visible in the image.")
                else:
                    return "Analyze and describe this brain MRI slice."
    
    def _get_prompt_type(self, identify_anomalies: bool, describe_slice: bool) -> str:
        """Get a readable description of the prompt type"""
        if identify_anomalies and describe_slice:
            return "anomaly detection + description"
        elif identify_anomalies:
            return "anomaly detection"
        elif describe_slice:
            return "general description"
        else:
            return "default analysis"
    
    def compute_visual_grounding_for_label(
        self, 
        image: np.ndarray, 
        target_label: str,
        xai_method: str = "input_gradient",
        use_cached_generation: bool = True
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Compute visual grounding (token-specific XAI) for a specific label.
        
        Uses the proper VLM XAI approach:
        1. Input×Gradient: Computes which image regions CAUSED the word generation
        2. Decoder Attention: Which image tokens the decoder attended to
        
        These methods analyze the DECODER behavior, not just the encoder features,
        which is the correct approach for VLMs.
        
        Args:
            image: RGB image as numpy array (H, W, 3), uint8
            target_label: The label text to compute attribution for
            xai_method: XAI method to use:
                - "input_gradient" (RECOMMENDED): Input×Gradient on projected image tokens
                - "decoder_attention": Decoder attention weights (SDPA workaround)
                - "attention": Legacy embedding similarity
                - "hirescam", "gradcam": Legacy encoder-only methods (slow)
            use_cached_generation: If True, tries to use cached generation for consistency
            
        Returns:
            Tuple of (heatmap: np.ndarray or None, metadata: dict)
            - heatmap: Spatial attribution heatmap (H, W) with values in [0, 1]
            - metadata: Dict with 'success', 'error', 'mean_attribution', 'max_attribution', etc.
        """
        try:
            # Get the service
            service = self._get_service()
            if service is None or not hasattr(service, 'compute_visual_grounding'):
                logger.warning("MedGemma service not available or visual grounding not supported")
                return None, {'success': False, 'error': 'Service not available'}
            
            # CRITICAL: Use the SAME prompt as label suggestion for consistent attribution
            # This is the key insight from HistoLens - different prompts = different attention
            if get_medical_prompt:
                prompt = get_medical_prompt("suggest_labels")
            else:
                prompt = (
                    "This is a brain MRI slice. Please list the anatomical or pathological structures that are visible. "
                    "Return a comma-separated list of possible labels (e.g., eye, lvent, tvent, tumor, lesion, etc.)."
                )
            
            logger.info(f"Computing visual grounding ({xai_method}) for label '{target_label}'")
            logger.info(f"Using label suggestion prompt for consistent attribution")
            
            # Compute visual grounding with selected method
            heatmap, metadata = service.compute_visual_grounding(
                image=image,
                prompt=prompt,
                target_token=target_label,
                xai_method=xai_method
            )
            
            if metadata.get('success', False):
                logger.info(f"Visual grounding success - Method: {metadata.get('method', xai_method)}, "
                           f"Mean: {metadata.get('mean_attribution', 0):.3f}, "
                           f"Max: {metadata.get('max_attribution', 0):.3f}")
            else:
                logger.warning(f"Visual grounding failed: {metadata.get('error', 'Unknown error')}")
            
            return heatmap, metadata
            
        except Exception as e:
            logger.error(f"Error computing visual grounding: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None, {'success': False, 'error': str(e)}
    
    def get_available_xai_methods(self) -> List[Tuple[str, str]]:
        """
        Get list of available XAI methods for visual grounding.
        
        Methods are ordered by recommendation:
        1. Input×Gradient: The standard VLM XAI approach - computes which image 
           regions CAUSED the word to be generated (gradient-based on projector output)
        2. Decoder Attention: Extracts attention from LLM decoder to see which
           image tokens the model "looked at" (SDPA workaround)
        3. Legacy methods: Previous approaches kept for comparison
        
        Returns:
            List of (method_key, display_name) tuples
        """
        return [
            ("input_gradient", "Input×Gradient (RECOMMENDED - Proper VLM XAI)"),
            ("decoder_attention", "Decoder Attention (SDPA Workaround)"),
            ("attention", "Embedding Similarity (Legacy, fast)"),
            ("hirescam", "HiResCAM (Legacy, encoder-only, slow)"),
            ("gradcam", "GradCAM (Legacy, encoder-only, slow)"),
        ]
    
    def cleanup(self):
        """Clean up MedGemma service resources"""
        try:
            if cleanup_service:
                cleanup_service()
                logger.info("MedGemma service cleaned up")
        except Exception as e:
            logger.error(f"Error during MedGemma cleanup: {e}")


# MRI Label Suggestion Prompt (for future use as specified in requirements)
MRI_LABEL_SUGGESTION_PROMPT = (
    "This is a brain MRI slice. Please list the anatomical or pathological structures that are visible. "
    "Return a comma-separated list of possible labels (e.g., eye, lvent, tvent, tumor, lesion, etc.)."
)
