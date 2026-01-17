"""
Mask Confidence Visualization

Creates confidence heatmaps from segmentation mask logits.
This is a practical XAI approach that shows where the model is
confident vs uncertain about its segmentation predictions.

This is more reliable than attention extraction as it works directly
with the model's output logits without requiring internal hooks.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, binary_erosion
import logging

logger = logging.getLogger(__name__)


def compute_mask_confidence(
    mask_logits: np.ndarray,
    apply_smoothing: bool = True,
    sigma: float = 2.0
) -> np.ndarray:
    """
    Compute confidence map from mask logits.
    
    High confidence = model strongly predicts foreground OR background
    Low confidence = model is uncertain (logits near 0)
    
    Args:
        mask_logits: Raw logits from segmentation model (before sigmoid)
        apply_smoothing: Whether to apply Gaussian smoothing
        sigma: Sigma for Gaussian smoothing
        
    Returns:
        Confidence map normalized to [0, 1]
    """
    # Convert to numpy if tensor
    if isinstance(mask_logits, torch.Tensor):
        mask_logits = mask_logits.detach().cpu().numpy()
    
    # Squeeze extra dimensions
    while mask_logits.ndim > 2:
        mask_logits = mask_logits.squeeze(0)
    
    # Confidence is the absolute value of logits
    # High |logit| = confident, Low |logit| = uncertain
    confidence = np.abs(mask_logits)
    
    # Normalize to [0, 1]
    if confidence.max() > confidence.min():
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
    else:
        confidence = np.ones_like(confidence)
    
    # Apply smoothing
    if apply_smoothing and sigma > 0:
        confidence = gaussian_filter(confidence, sigma=sigma)
        # Re-normalize after smoothing
        if confidence.max() > confidence.min():
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
    
    return confidence.astype(np.float32)


def compute_boundary_uncertainty(
    mask: np.ndarray,
    mask_logits: Optional[np.ndarray] = None,
    boundary_width: int = 5,
    falloff_sigma: float = 3.0
) -> np.ndarray:
    """
    Compute uncertainty map focused on boundaries.
    
    Boundaries are where segmentation decisions are most critical,
    so highlighting uncertainty there is clinically meaningful.
    
    Args:
        mask: Binary segmentation mask
        mask_logits: Optional logits for confidence weighting
        boundary_width: Width of boundary region in pixels
        falloff_sigma: Sigma for smooth falloff from boundaries
        
    Returns:
        Boundary uncertainty map normalized to [0, 1]
    """
    # Convert to numpy and ensure binary
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    
    binary_mask = mask > 0.5
    
    # If mask is empty, return zeros
    if not binary_mask.any():
        return np.zeros_like(mask, dtype=np.float32)
    
    # Compute boundary as dilation - erosion
    dilated = binary_dilation(binary_mask, iterations=boundary_width)
    eroded = binary_erosion(binary_mask, iterations=boundary_width)
    boundary = dilated.astype(float) - eroded.astype(float)
    
    # Apply smooth falloff from boundary
    boundary_uncertainty = gaussian_filter(boundary, sigma=falloff_sigma)
    
    # Weight by logit confidence if available (invert confidence = uncertainty)
    if mask_logits is not None:
        confidence = compute_mask_confidence(mask_logits, apply_smoothing=False)
        # Uncertainty = 1 - confidence
        uncertainty = 1.0 - confidence
        # Combine with boundary
        boundary_uncertainty = boundary_uncertainty * (0.5 + 0.5 * uncertainty)
    
    # Normalize
    if boundary_uncertainty.max() > 0:
        boundary_uncertainty = boundary_uncertainty / boundary_uncertainty.max()
    
    return boundary_uncertainty.astype(np.float32)


def create_prompt_influence_map(
    mask: np.ndarray,
    prompt_points: Optional[List[Tuple[int, int]]] = None,
    prompt_box: Optional[Tuple[int, int, int, int]] = None,
    influence_sigma: float = 30.0
) -> np.ndarray:
    """
    Create a map showing how prompt points/boxes influenced the segmentation.
    
    This creates a gradient from prompt locations showing their "influence field".
    Very useful for understanding point-based and box-based segmentation.
    
    Args:
        mask: Binary segmentation mask
        prompt_points: List of (x, y) prompt point coordinates
        prompt_box: (x1, y1, x2, y2) bounding box prompt
        influence_sigma: Sigma for influence spread
        
    Returns:
        Influence map normalized to [0, 1]
    """
    # Convert to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    
    h, w = mask.shape
    influence_map = np.zeros((h, w), dtype=np.float32)
    
    # Add influence from prompt points
    if prompt_points:
        for px, py in prompt_points:
            # Ensure coordinates are within bounds
            px = max(0, min(w - 1, int(px)))
            py = max(0, min(h - 1, int(py)))
            
            # Create distance-based influence from point
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_sq = (x_coords - px) ** 2 + (y_coords - py) ** 2
            point_influence = np.exp(-dist_sq / (2 * influence_sigma ** 2))
            influence_map = np.maximum(influence_map, point_influence)
    
    # Add influence from prompt box
    if prompt_box:
        x1, y1, x2, y2 = prompt_box
        # Ensure coordinates are within bounds
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        
        # Create box influence - high inside, falling off outside
        box_mask = np.zeros((h, w), dtype=np.float32)
        box_mask[y1:y2, x1:x2] = 1.0
        
        # Distance transform from box edges
        dist_outside = distance_transform_edt(1 - box_mask)
        box_influence = np.exp(-dist_outside / influence_sigma)
        
        influence_map = np.maximum(influence_map, box_influence)
    
    # If no prompts provided, use mask centroid
    if not prompt_points and not prompt_box:
        binary_mask = mask > 0.5
        if binary_mask.any():
            # Find centroid of mask
            coords = np.argwhere(binary_mask)
            centroid_y = coords[:, 0].mean()
            centroid_x = coords[:, 1].mean()
            
            # Create distance-based influence from centroid
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_sq = (x_coords - centroid_x) ** 2 + (y_coords - centroid_y) ** 2
            influence_map = np.exp(-dist_sq / (2 * influence_sigma ** 2))
    
    # Combine with mask to show only relevant areas
    binary_mask = mask > 0.5
    if binary_mask.any():
        # Expand influence to cover nearby non-mask areas
        dist_from_mask = distance_transform_edt(~binary_mask)
        mask_proximity = np.exp(-dist_from_mask / influence_sigma)
        influence_map = influence_map * (0.3 + 0.7 * mask_proximity)
    
    # Normalize
    if influence_map.max() > influence_map.min():
        influence_map = (influence_map - influence_map.min()) / (influence_map.max() - influence_map.min())
    
    return influence_map.astype(np.float32)


def create_focus_map(
    mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    expansion_factor: float = 1.5
) -> np.ndarray:
    """
    Create a focus map showing where the model "looked" based on the mask.
    
    This creates a smooth gradient from the segmented region outward,
    providing an intuitive visualization of the model's area of interest.
    
    Args:
        mask: Binary segmentation mask
        image: Optional image for edge-aware focusing
        expansion_factor: How much to expand the focus region
        
    Returns:
        Focus map normalized to [0, 1]
    """
    # Convert to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    
    binary_mask = mask > 0.5
    
    # If mask is empty, return zeros
    if not binary_mask.any():
        return np.zeros_like(mask, dtype=np.float32)
    
    # Distance transform from mask boundary
    dist_outside = distance_transform_edt(~binary_mask)
    dist_inside = distance_transform_edt(binary_mask)
    
    # Combined distance (negative inside, positive outside)
    combined_dist = dist_outside - dist_inside
    
    # Create focus using sigmoid-like falloff
    # Regions inside mask have high focus (1.0)
    # Focus falls off with distance from mask
    max_dist = np.percentile(dist_outside[dist_outside > 0], 90) if (dist_outside > 0).any() else 1
    
    # Smooth falloff
    focus = 1.0 / (1.0 + np.exp(combined_dist / (max_dist * expansion_factor * 0.1 + 1)))
    
    # Smooth the result
    focus = gaussian_filter(focus, sigma=3)
    
    # Normalize
    if focus.max() > focus.min():
        focus = (focus - focus.min()) / (focus.max() - focus.min())
    
    return focus.astype(np.float32)


class MaskConfidenceExtractor:
    """
    Extracts confidence/uncertainty visualizations from segmentation outputs.
    
    This is a more practical XAI approach than attention extraction for
    transformer models that don't expose attention weights.
    """
    
    def __init__(self):
        self.last_logits: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_confidence_map: Optional[np.ndarray] = None
        self.last_boundary_map: Optional[np.ndarray] = None
        self.last_influence_map: Optional[np.ndarray] = None
        self.prompt_points: Optional[List[Tuple[int, int]]] = None
        self.prompt_box: Optional[Tuple[int, int, int, int]] = None
        
    def set_prompts(
        self,
        points: Optional[List[Tuple[int, int]]] = None,
        box: Optional[Tuple[int, int, int, int]] = None
    ) -> None:
        """Set prompt information for influence visualization"""
        self.prompt_points = points
        self.prompt_box = box
        
    def capture_outputs(
        self,
        mask: np.ndarray,
        mask_logits: Optional[np.ndarray] = None
    ) -> None:
        """
        Capture segmentation outputs for visualization.
        
        Args:
            mask: Binary segmentation mask
            mask_logits: Optional raw logits (before sigmoid)
        """
        self.last_mask = mask
        self.last_logits = mask_logits
        
        # Pre-compute visualizations
        if mask_logits is not None:
            self.last_confidence_map = compute_mask_confidence(mask_logits)
        else:
            self.last_confidence_map = None
        
        # Always compute boundary uncertainty (this is the most useful)
        self.last_boundary_map = compute_boundary_uncertainty(mask, mask_logits)
        
        # Compute influence map if we have prompts
        self.last_influence_map = create_prompt_influence_map(
            mask,
            prompt_points=self.prompt_points,
            prompt_box=self.prompt_box
        )
        
        logger.info("XAI: Captured segmentation outputs for visualization")
        
    def get_confidence_map(self) -> Optional[np.ndarray]:
        """Get the confidence visualization map"""
        return self.last_confidence_map
    
    def get_boundary_map(self) -> Optional[np.ndarray]:
        """Get the boundary uncertainty map"""
        return self.last_boundary_map
    
    def get_influence_map(self) -> Optional[np.ndarray]:
        """Get the prompt influence map"""
        return self.last_influence_map
        
    def get_focus_map(self) -> Optional[np.ndarray]:
        """Get the focus visualization map"""
        if self.last_mask is None:
            return None
        return create_focus_map(self.last_mask)
        
    def get_boundary_uncertainty(self) -> Optional[np.ndarray]:
        """Get boundary uncertainty map"""
        return self.last_boundary_map
    
    def get_best_visualization(self) -> Optional[np.ndarray]:
        """
        Get the best available visualization map.
        
        Priority:
        1. Boundary uncertainty (most clinically meaningful)
        2. Prompt influence (shows what drove segmentation)
        3. Confidence map (if logits available)
        4. Focus map (always available if mask exists)
        """
        if self.last_boundary_map is not None and self.last_boundary_map.max() > 0:
            return self.last_boundary_map
        if self.last_influence_map is not None and self.last_influence_map.max() > 0:
            return self.last_influence_map
        if self.last_confidence_map is not None:
            return self.last_confidence_map
        if self.last_mask is not None:
            return create_focus_map(self.last_mask)
        return None
        
    def clear(self) -> None:
        """Clear stored visualizations"""
        self.last_logits = None
        self.last_mask = None
        self.last_confidence_map = None
        self.last_boundary_map = None
        self.last_influence_map = None
        self.prompt_points = None
        self.prompt_box = None


# Global instance
_mask_confidence_extractor: Optional[MaskConfidenceExtractor] = None


def get_mask_confidence_extractor() -> MaskConfidenceExtractor:
    """Get or create the global mask confidence extractor"""
    global _mask_confidence_extractor
    if _mask_confidence_extractor is None:
        _mask_confidence_extractor = MaskConfidenceExtractor()
    return _mask_confidence_extractor
