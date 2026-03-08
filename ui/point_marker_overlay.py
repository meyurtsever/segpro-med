"""
Point Marker Overlay Module

Handles Dead-Eye style point marker visualization for MedSAM2 point-based prompts.
Draws red X markers on the image to show user click locations without requiring page refresh.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def draw_point_markers_on_image(
    medsam2_handlers,
    coords_text: str,
    include_xai: bool = True,
    include_uncertainty: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    Draw red X markers (Dead-Eye style) on the current image for all selected point coordinates.
    
    Args:
        medsam2_handlers: MedSAM2Handlers instance with current state and coordinates
        coords_text: Coordinate text from the click handler
        include_xai: Whether to include XAI overlay if enabled (default: True)
        include_uncertainty: Whether to include uncertainty overlay if enabled (default: True)
    
    Returns:
        Tuple of (coords_text, annotated_image_dict) for Gradio image_annotator
    """
    try:
        from utils.visualization import display_slice, create_annotation_boxes_from_mask
        
        # Get clean image from current state
        clean_img = display_slice(
            medsam2_handlers.state.current_data,
            medsam2_handlers.state.current_slice_idx,
            medsam2_handlers.state.current_view,
            window_level=medsam2_handlers.state.window_level,
            window_width=medsam2_handlers.state.window_width,
            crosshair=None
        )
        
        # Ensure RGB uint8
        if len(clean_img.shape) == 2:
            img_rgb = np.stack([clean_img] * 3, axis=-1)
        else:
            img_rgb = clean_img.copy()
        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        
        # Apply XAI overlay if enabled
        if include_xai and hasattr(medsam2_handlers, '_xai_show_overlay') and medsam2_handlers._xai_show_overlay:
            img_with_xai = medsam2_handlers.get_xai_display_image(img_rgb)
            if img_with_xai is not None:
                img_rgb = img_with_xai
        
        # Apply uncertainty overlay if enabled
        if include_uncertainty and hasattr(medsam2_handlers, 'uncertainty_overlay_enabled') and medsam2_handlers.uncertainty_overlay_enabled:
            img_rgb = _apply_uncertainty_overlay(medsam2_handlers, img_rgb)
        
        # Draw all point prompts as red X markers (Dead-Eye style!)
        if hasattr(medsam2_handlers, 'selected_coordinates') and medsam2_handlers.selected_coordinates:
            img_rgb = _draw_red_x_markers(img_rgb, medsam2_handlers.selected_coordinates)
        
        # Add existing annotation shapes
        annotation_shapes = _get_annotation_shapes(medsam2_handlers)
        
        return coords_text, {
            "image": img_rgb,
            "boxes": annotation_shapes,
            "orientation": 0
        }
        
    except Exception as e:
        logger.error(f"Error drawing point markers: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        import gradio as gr
        return coords_text, gr.update()


def _apply_uncertainty_overlay(medsam2_handlers, img_rgb: np.ndarray) -> np.ndarray:
    """Apply uncertainty map overlay to the image."""
    mask_logits = medsam2_handlers._xai_integration.get_mask_logits() if medsam2_handlers._xai_integration else None
    if mask_logits is None:
        return img_rgb
    
    try:
        from xai.segmentation.uncertainty_maps import UncertaintyMapGenerator
        
        generator = UncertaintyMapGenerator()
        uncertainty_map, stats = generator.compute_from_logits(mask_logits, method="margin")
        
        # Resize and create overlay
        target_h, target_w = img_rgb.shape[:2]
        if uncertainty_map.shape[:2] != (target_h, target_w):
            uncertainty_map_resized = cv2.resize(uncertainty_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        else:
            uncertainty_map_resized = uncertainty_map
        
        uncertainty_colored = generator.create_colored_overlay(uncertainty_map_resized, colormap="RdYlGn_r")
        
        # Get combined mask
        current_slice = medsam2_handlers.state.current_slice_idx
        combined_mask = _get_combined_segmentation_mask(medsam2_handlers, current_slice, target_h, target_w)
        
        # Blend with mask
        if combined_mask is not None:
            kernel = np.ones((15, 15), np.uint8)
            mask_dilated = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1)
            mask_3ch = np.stack([mask_dilated] * 3, axis=-1).astype(bool)
            alpha = 0.4
            img_rgb[mask_3ch] = ((1 - alpha) * img_rgb[mask_3ch] + alpha * uncertainty_colored[mask_3ch]).astype(np.uint8)
        
        return img_rgb
    except Exception as e:
        logger.warning(f"Failed to apply uncertainty overlay: {e}")
        return img_rgb


def _get_combined_segmentation_mask(
    medsam2_handlers,
    current_slice: int,
    target_h: int,
    target_w: int
) -> Optional[np.ndarray]:
    """Get combined segmentation mask from all annotations on current slice."""
    combined_mask = None
    
    if (hasattr(medsam2_handlers, 'annotation_overlays') and 
        current_slice in medsam2_handlers.annotation_overlays):
        overlay_data = medsam2_handlers.annotation_overlays[current_slice]
        masks = []
        
        if isinstance(overlay_data, dict):
            if 'mask' in overlay_data:
                masks.append(overlay_data['mask'])
            else:
                for annotation_id, annotation_data in overlay_data.items():
                    if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                        masks.append(annotation_data['mask'])
        
        if masks:
            combined_mask = np.zeros_like(masks[0], dtype=bool)
            for mask in masks:
                combined_mask = combined_mask | (mask > 0)
    
    # Resize if needed
    if combined_mask is not None and combined_mask.shape[:2] != (target_h, target_w):
        combined_mask = cv2.resize(combined_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
    
    return combined_mask


def _draw_red_x_markers(img_rgb: np.ndarray, coordinates: List[Tuple[int, int]]) -> np.ndarray:
    """
    Draw red X markers (Dead-Eye style) at specified coordinates.
    
    Args:
        img_rgb: RGB image array
        coordinates: List of (x, y) coordinate tuples
    
    Returns:
        Image with red X markers drawn
    """
    for point in coordinates:
        x, y = int(point[0]), int(point[1])
        
        # Dead-Eye style red X marker
        size = 5  # Size of the X
        thickness = 2
        color = (255, 0, 0)  # Red in RGB
        
        # Draw two diagonal lines forming an X
        cv2.line(img_rgb, (x - size, y - size), (x + size, y + size), color, thickness, cv2.LINE_AA)
        cv2.line(img_rgb, (x - size, y + size), (x + size, y - size), color, thickness, cv2.LINE_AA)
    
    return img_rgb


def _get_annotation_shapes(medsam2_handlers) -> List[Dict]:
    """Get annotation box shapes from current slice."""
    from utils.visualization import create_annotation_boxes_from_mask
    
    current_slice = medsam2_handlers.state.current_slice_idx
    annotation_shapes = []
    
    if (hasattr(medsam2_handlers, 'annotation_overlays') and 
        current_slice in medsam2_handlers.annotation_overlays):
        overlay_data = medsam2_handlers.annotation_overlays[current_slice]
        
        if isinstance(overlay_data, dict):
            if 'mask' in overlay_data:
                mask_array = overlay_data['mask']
                annotation_shapes = create_annotation_boxes_from_mask(
                    mask_array, label="MEDSAM2 Annotation", label_index=1
                )
            else:
                for annotation_id, annotation_data in overlay_data.items():
                    if isinstance(annotation_data, dict) and 'mask' in annotation_data:
                        mask_array = annotation_data['mask']
                        shapes = create_annotation_boxes_from_mask(
                            mask_array, label=f"MEDSAM2 Annotation {annotation_id}", label_index=1
                        )
                        annotation_shapes.extend(shapes)
    
    return annotation_shapes
