"""
Heatmap Overlay Visualization

Creates attention heatmap overlays for explainability visualization.
Designed to be compatible with the existing visualization pipeline in
utils/visualization.py and the Gradio image_annotator component.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def create_attention_heatmap_overlay(
    base_image: np.ndarray,
    attention_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
    threshold: float = 0.1,
    normalize: bool = True,
    show_colorbar: bool = False
) -> np.ndarray:
    """
    Create an attention heatmap overlay on the base image.
    
    This function overlays an attention map as a semi-transparent heatmap
    on top of the original medical image, highlighting regions where the
    model focused attention.
    
    Args:
        base_image: Base image as numpy array (grayscale or RGB)
        attention_map: 2D attention map (should be same size or will be resized)
        colormap: Matplotlib colormap name ('jet', 'viridis', 'hot', 'plasma')
        alpha: Transparency of the overlay (0-1)
        threshold: Values below this threshold won't be shown (0-1)
        normalize: Whether to normalize attention_map to [0, 1]
        show_colorbar: Whether to add a colorbar legend
        
    Returns:
        RGB image with attention overlay as numpy array
    """
    # Ensure base image is RGB
    if len(base_image.shape) == 2:
        base_rgb = np.stack([base_image] * 3, axis=-1)
    elif base_image.shape[-1] == 1:
        base_rgb = np.stack([base_image[..., 0]] * 3, axis=-1)
    else:
        base_rgb = base_image.copy()
    
    # Ensure base image is uint8
    if base_rgb.dtype != np.uint8:
        if base_rgb.max() <= 1.0:
            base_rgb = (base_rgb * 255).astype(np.uint8)
        else:
            base_rgb = base_rgb.astype(np.uint8)
    
    # Handle attention map
    attention = attention_map.copy().astype(np.float32)
    
    # Resize attention map if needed
    target_size = (base_rgb.shape[1], base_rgb.shape[0])  # (W, H)
    if attention.shape[:2] != base_rgb.shape[:2]:
        pil_attn = Image.fromarray(attention)
        pil_attn = pil_attn.resize(target_size, Image.BILINEAR)
        attention = np.array(pil_attn)
    
    # Normalize attention map
    if normalize and attention.max() > attention.min():
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    # Apply threshold - set low values to 0
    attention[attention < threshold] = 0
    
    # Get colormap
    try:
        cmap = cm.get_cmap(colormap)
    except:
        logger.warning(f"Unknown colormap {colormap}, using 'jet'")
        cmap = cm.get_cmap('jet')
    
    # Apply colormap to attention (returns RGBA)
    colored_attention = cmap(attention)
    
    # Convert to uint8 RGB (drop alpha channel from colormap)
    heatmap_rgb = (colored_attention[:, :, :3] * 255).astype(np.uint8)
    
    # Create alpha mask from attention values
    # Higher attention = more visible overlay
    alpha_mask = (attention * alpha)[:, :, np.newaxis]
    
    # Blend: result = base * (1 - alpha_mask) + heatmap * alpha_mask
    result = base_rgb.astype(np.float32) * (1 - alpha_mask) + heatmap_rgb.astype(np.float32) * alpha_mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Add colorbar if requested
    if show_colorbar:
        result = _add_colorbar_to_image(result, colormap)
    
    return result


def create_uncertainty_overlay(
    base_image: np.ndarray,
    uncertainty_map: np.ndarray,
    colormap: str = "RdYlGn_r",  # Red=uncertain, Green=confident
    alpha: float = 0.4,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Create an uncertainty map overlay.
    
    Args:
        base_image: Base image as numpy array
        uncertainty_map: 2D uncertainty map (0=confident, 1=uncertain)
        colormap: Colormap for uncertainty visualization
        alpha: Transparency of overlay
        threshold: Don't show regions below this uncertainty
        
    Returns:
        RGB image with uncertainty overlay
    """
    return create_attention_heatmap_overlay(
        base_image=base_image,
        attention_map=uncertainty_map,
        colormap=colormap,
        alpha=alpha,
        threshold=threshold,
        normalize=True,
        show_colorbar=False
    )


def blend_heatmap_with_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Blend a heatmap with an image using alpha blending.
    
    Simpler version of create_attention_heatmap_overlay for general use.
    
    Args:
        image: Base image (grayscale or RGB)
        heatmap: Heatmap values (will be normalized and colormapped)
        alpha: Blend factor (0=only image, 1=only heatmap)
        colormap: Matplotlib colormap
        
    Returns:
        Blended RGB image
    """
    # Ensure image is RGB uint8
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
    
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        else:
            image_rgb = image_rgb.astype(np.uint8)
    
    # Normalize and colormap heatmap
    heatmap_norm = heatmap.astype(np.float32)
    if heatmap_norm.max() > heatmap_norm.min():
        heatmap_norm = (heatmap_norm - heatmap_norm.min()) / (heatmap_norm.max() - heatmap_norm.min())
    
    # Resize heatmap if needed
    if heatmap_norm.shape[:2] != image_rgb.shape[:2]:
        pil_hm = Image.fromarray(heatmap_norm)
        pil_hm = pil_hm.resize((image_rgb.shape[1], image_rgb.shape[0]), Image.BILINEAR)
        heatmap_norm = np.array(pil_hm)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_norm)[:, :, :3]
    colored_rgb = (colored * 255).astype(np.uint8)
    
    # Blend
    result = (image_rgb.astype(np.float32) * (1 - alpha) + 
              colored_rgb.astype(np.float32) * alpha)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def create_colorbar_legend(
    colormap: str = "jet",
    size: Tuple[int, int] = (30, 256),
    label_min: str = "Low",
    label_max: str = "High",
    title: str = "Attention",
    orientation: str = "vertical"
) -> np.ndarray:
    """
    Create a standalone colorbar legend image.
    
    Args:
        colormap: Matplotlib colormap name
        size: Size of colorbar (width, height) for vertical, (height, width) for horizontal
        label_min: Label for minimum value
        label_max: Label for maximum value
        title: Title for the colorbar
        orientation: 'vertical' or 'horizontal'
        
    Returns:
        RGB image of colorbar legend
    """
    fig, ax = plt.subplots(figsize=(1.5, 4) if orientation == 'vertical' else (4, 0.5))
    
    # Create gradient
    gradient = np.linspace(0, 1, 256)
    
    if orientation == 'vertical':
        gradient = gradient.reshape(-1, 1)
    else:
        gradient = gradient.reshape(1, -1)
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Display colorbar
    im = ax.imshow(gradient, aspect='auto', cmap=cmap, 
                   extent=[0, 1, 0, 1] if orientation == 'horizontal' else [0, 1, 0, 1])
    
    # Configure axes
    if orientation == 'vertical':
        ax.set_xticks([])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([label_min, label_max])
        ax.set_ylabel(title)
    else:
        ax.set_yticks([])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_min, label_max])
        ax.set_xlabel(title)
    
    # Remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img_array


def _add_colorbar_to_image(
    image: np.ndarray,
    colormap: str,
    position: str = "right",
    width: int = 30,
    padding: int = 10
) -> np.ndarray:
    """
    Add a colorbar legend to the side of an image.
    
    Args:
        image: RGB image
        colormap: Colormap used
        position: 'right', 'left', 'top', or 'bottom'
        width: Width of colorbar
        padding: Padding between image and colorbar
        
    Returns:
        Image with colorbar added
    """
    h, w = image.shape[:2]
    
    # Create colorbar
    if position in ['right', 'left']:
        cb_size = (width, h)
        colorbar = create_colorbar_legend(colormap=colormap, size=cb_size, orientation='vertical')
        # Resize to match height
        colorbar_pil = Image.fromarray(colorbar).resize((width, h), Image.BILINEAR)
        colorbar = np.array(colorbar_pil)
    else:
        cb_size = (w, width)
        colorbar = create_colorbar_legend(colormap=colormap, size=cb_size, orientation='horizontal')
        colorbar_pil = Image.fromarray(colorbar).resize((w, width), Image.BILINEAR)
        colorbar = np.array(colorbar_pil)
    
    # Create padding
    if position in ['right', 'left']:
        pad = np.ones((h, padding, 3), dtype=np.uint8) * 30  # Dark gray padding
    else:
        pad = np.ones((padding, w, 3), dtype=np.uint8) * 30
    
    # Combine
    if position == 'right':
        result = np.hstack([image, pad, colorbar])
    elif position == 'left':
        result = np.hstack([colorbar, pad, image])
    elif position == 'bottom':
        result = np.vstack([image, pad, colorbar])
    else:  # top
        result = np.vstack([colorbar, pad, image])
    
    return result


def overlay_attention_on_slice(
    slice_image: np.ndarray,
    attention_map: Optional[np.ndarray],
    enabled: bool = True,
    colormap: str = "jet",
    alpha: float = 0.5,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Convenience function to overlay attention on a medical image slice.
    
    This is the main entry point for integrating with the existing
    display_slice and overlay_segmentation workflow.
    
    Args:
        slice_image: The medical image slice (from display_slice or similar)
        attention_map: Attention map from SAM2AttentionExtractor
        enabled: Whether attention overlay is enabled
        colormap: Colormap for attention visualization
        alpha: Overlay transparency
        threshold: Minimum attention value to show
        
    Returns:
        Image with attention overlay (or original if disabled/no attention)
    """
    if not enabled or attention_map is None:
        return slice_image
    
    # Check if attention map is valid
    if attention_map.size == 0 or attention_map.max() == attention_map.min():
        logger.debug("Invalid attention map, returning original image")
        return slice_image
    
    return create_attention_heatmap_overlay(
        base_image=slice_image,
        attention_map=attention_map,
        colormap=colormap,
        alpha=alpha,
        threshold=threshold,
        normalize=True,
        show_colorbar=False
    )


def create_side_by_side_comparison(
    original: np.ndarray,
    with_attention: np.ndarray,
    attention_map: np.ndarray,
    colormap: str = "jet",
    add_labels: bool = True
) -> np.ndarray:
    """
    Create a side-by-side comparison of original and attention-overlaid images.
    
    Useful for debugging and educational purposes.
    
    Args:
        original: Original medical image
        with_attention: Image with attention overlay
        attention_map: The attention map (for pure heatmap display)
        colormap: Colormap used
        add_labels: Whether to add text labels
        
    Returns:
        Combined side-by-side image
    """
    # Ensure same height
    h = original.shape[0]
    
    # Create pure attention heatmap
    cmap = cm.get_cmap(colormap)
    
    attention_norm = attention_map.astype(np.float32)
    if attention_norm.max() > attention_norm.min():
        attention_norm = (attention_norm - attention_norm.min()) / (attention_norm.max() - attention_norm.min())
    
    # Resize if needed
    if attention_norm.shape[:2] != original.shape[:2]:
        pil_attn = Image.fromarray(attention_norm)
        pil_attn = pil_attn.resize((original.shape[1], original.shape[0]), Image.BILINEAR)
        attention_norm = np.array(pil_attn)
    
    pure_heatmap = (cmap(attention_norm)[:, :, :3] * 255).astype(np.uint8)
    
    # Ensure original is RGB
    if len(original.shape) == 2:
        original_rgb = np.stack([original] * 3, axis=-1)
    else:
        original_rgb = original
    
    # Resize with_attention if needed
    if with_attention.shape[0] != h:
        pil_wa = Image.fromarray(with_attention)
        pil_wa = pil_wa.resize((with_attention.shape[1], h), Image.BILINEAR)
        with_attention = np.array(pil_wa)
    
    # Create separator
    separator = np.ones((h, 5, 3), dtype=np.uint8) * 128
    
    # Combine
    combined = np.hstack([original_rgb, separator, with_attention, separator, pure_heatmap])
    
    # Add labels if requested
    if add_labels:
        combined = _add_comparison_labels(combined, original.shape[1])
    
    return combined


def _add_comparison_labels(image: np.ndarray, section_width: int) -> np.ndarray:
    """Add text labels to a comparison image"""
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    labels = ["Original", "With Attention", "Attention Map"]
    separator_width = 5
    
    for i, label in enumerate(labels):
        x = i * (section_width + separator_width) + section_width // 2
        draw.text((x - 40, 5), label, fill=(255, 255, 0), font=font)
    
    return np.array(pil_img)
