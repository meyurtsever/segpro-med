"""
SAM2 Attention Extractor

Extracts attention maps from SAM2/MedSAM2 model during inference
for explainability visualization.

This module hooks into the transformer attention layers to capture
attention weights that show which image regions the model focuses on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionMaps:
    """Container for extracted attention maps"""
    
    # Raw attention maps by layer name
    raw_maps: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Processed/aggregated attention maps
    aggregated_map: Optional[np.ndarray] = None
    
    # Metadata
    image_size: Tuple[int, int] = (512, 512)
    num_layers_captured: int = 0
    layer_names: List[str] = field(default_factory=list)
    
    def to_numpy(self, layer_name: Optional[str] = None) -> np.ndarray:
        """Convert attention map to numpy array"""
        if layer_name is not None and layer_name in self.raw_maps:
            return self.raw_maps[layer_name].cpu().numpy()
        elif self.aggregated_map is not None:
            return self.aggregated_map
        else:
            return np.zeros(self.image_size)
    
    def get_visualization_map(self) -> np.ndarray:
        """Get the best map for visualization (aggregated or first available)"""
        if self.aggregated_map is not None:
            return self.aggregated_map
        elif len(self.raw_maps) > 0:
            first_key = list(self.raw_maps.keys())[0]
            return self.raw_maps[first_key].cpu().numpy()
        else:
            return np.zeros(self.image_size)


class SAM2AttentionExtractor:
    """
    Extract attention maps from SAM2/MedSAM2 model.
    
    This class registers hooks on attention layers to capture attention weights
    during forward pass, then processes them for visualization.
    
    Usage:
        extractor = SAM2AttentionExtractor(model)
        extractor.register_hooks()
        
        # Run inference
        masks, scores = model.predict(image, prompts)
        
        # Get attention maps
        attention_maps = extractor.get_attention_maps()
        
        # Cleanup
        extractor.remove_hooks()
    """
    
    def __init__(
        self, 
        model: nn.Module,
        target_layers: Optional[List[str]] = None,
        max_layers: int = 4,
        normalize: bool = True,
        blur_sigma: float = 2.0
    ):
        """
        Initialize the attention extractor.
        
        Args:
            model: SAM2/MedSAM2 model
            target_layers: Specific layer names to hook (None = auto-detect)
            max_layers: Maximum number of layers to capture
            normalize: Whether to normalize attention maps
            blur_sigma: Gaussian blur sigma for smoothing
        """
        self.model = model
        self.target_layers = target_layers or ['attention', 'attn', 'self_attn', 'cross_attn']
        self.max_layers = max_layers
        self.normalize = normalize
        self.blur_sigma = blur_sigma
        
        # Storage for attention maps
        self._attention_maps: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._hooks_registered = False
        
        # For capturing intermediate outputs
        self._intermediate_outputs: Dict[str, torch.Tensor] = {}
        
        logger.info(f"SAM2AttentionExtractor initialized with max_layers={max_layers}")
    
    def register_hooks(self) -> int:
        """
        Register forward hooks on attention layers.
        
        Returns:
            Number of hooks registered
        """
        if self._hooks_registered:
            logger.warning("Hooks already registered. Call remove_hooks() first.")
            return len(self._hooks)
        
        hooks_count = 0
        
        # Try to hook into image encoder attention layers
        encoder = getattr(self.model, 'image_encoder', None)
        if encoder is not None:
            hooks_count += self._register_hooks_on_module(encoder, 'image_encoder')
        
        # Try to hook into mask decoder attention
        mask_decoder = getattr(self.model, 'mask_decoder', None)
        if mask_decoder is not None:
            hooks_count += self._register_hooks_on_module(mask_decoder, 'mask_decoder')
        
        # Try memory attention if exists (SAM2-specific)
        memory_attention = getattr(self.model, 'memory_attention', None)
        if memory_attention is not None:
            hooks_count += self._register_hooks_on_module(memory_attention, 'memory_attention')
        
        self._hooks_registered = True
        logger.info(f"Registered {hooks_count} attention hooks")
        
        return hooks_count
    
    def _register_hooks_on_module(self, module: nn.Module, prefix: str) -> int:
        """Register hooks on a specific module's attention layers"""
        hooks_count = 0
        
        for name, layer in module.named_modules():
            # Check if this is an attention-related layer
            if any(target in name.lower() for target in self.target_layers):
                if hooks_count >= self.max_layers:
                    break
                
                full_name = f"{prefix}.{name}"
                hook = layer.register_forward_hook(self._create_hook(full_name))
                self._hooks.append(hook)
                hooks_count += 1
                logger.debug(f"Hook registered on: {full_name}")
        
        return hooks_count
    
    def _create_hook(self, layer_name: str):
        """Create a hook function for capturing attention weights"""
        def hook_fn(module, input, output):
            try:
                # Different modules return attention weights differently
                attention_weights = None
                
                # Case 1: Module has explicit attention weights attribute
                if hasattr(module, 'attn_weights') and module.attn_weights is not None:
                    attention_weights = module.attn_weights.detach()
                
                # Case 2: Output is a tuple (output, attention_weights)
                elif isinstance(output, tuple) and len(output) >= 2:
                    potential_attn = output[1]
                    if isinstance(potential_attn, torch.Tensor) and len(potential_attn.shape) >= 2:
                        attention_weights = potential_attn.detach()
                
                # Case 3: Output is attention weights directly (some modules)
                elif isinstance(output, torch.Tensor):
                    # Check if it looks like attention weights
                    if len(output.shape) >= 3:  # [batch, heads/seq, seq]
                        attention_weights = output.detach()
                
                # Case 4: Check for Scaled Dot Product Attention pattern
                # We need to compute attention from Q, K, V if available
                if attention_weights is None and isinstance(input, tuple) and len(input) >= 2:
                    # Try to compute attention scores from input Q, K
                    q = input[0] if isinstance(input[0], torch.Tensor) else None
                    k = input[1] if len(input) > 1 and isinstance(input[1], torch.Tensor) else None
                    
                    if q is not None and k is not None and q.dim() >= 3:
                        try:
                            # Compute attention scores: softmax(Q @ K^T / sqrt(d))
                            scale = q.shape[-1] ** -0.5
                            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                            attention_weights = F.softmax(attn_scores, dim=-1).detach()
                        except Exception as e:
                            logger.debug(f"Could not compute attention from Q,K: {e}")
                
                if attention_weights is not None:
                    self._attention_maps[layer_name] = attention_weights
                    logger.debug(f"Captured attention from {layer_name}, shape: {attention_weights.shape}")
                else:
                    # Store the output for potential later processing
                    if isinstance(output, torch.Tensor):
                        self._intermediate_outputs[layer_name] = output.detach()
                    
            except Exception as e:
                logger.warning(f"Error in attention hook for {layer_name}: {e}")
        
        return hook_fn
    
    def get_attention_maps(
        self, 
        image_size: Optional[Tuple[int, int]] = None,
        aggregate: bool = True
    ) -> AttentionMaps:
        """
        Get collected attention maps, optionally aggregated.
        
        Args:
            image_size: Target image size for reshaping attention maps
            aggregate: Whether to aggregate multiple layers into one map
            
        Returns:
            AttentionMaps container with raw and processed maps
        """
        if image_size is None:
            image_size = (512, 512)
        
        result = AttentionMaps(
            image_size=image_size,
            num_layers_captured=len(self._attention_maps),
            layer_names=list(self._attention_maps.keys())
        )
        
        # Copy raw maps
        for name, attn in self._attention_maps.items():
            result.raw_maps[name] = attn.clone()
        
        # Aggregate if requested and we have maps
        if aggregate and len(self._attention_maps) > 0:
            result.aggregated_map = self._aggregate_attention_maps(image_size)
        
        return result
    
    def _aggregate_attention_maps(self, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Aggregate multiple attention maps into a single visualization-ready map.
        
        Args:
            target_size: Target (H, W) size
            
        Returns:
            Aggregated attention map as numpy array
        """
        if len(self._attention_maps) == 0:
            return np.zeros(target_size)
        
        aggregated_maps = []
        
        for name, attn in self._attention_maps.items():
            try:
                # Convert to numpy and handle different shapes
                attn_np = attn.cpu().numpy()
                
                # Handle different attention shapes
                # Shape could be: [batch, heads, seq, seq] or [batch, seq, seq] or others
                
                if len(attn_np.shape) == 4:
                    # [batch, heads, seq_q, seq_k] - average over batch and heads
                    attn_np = attn_np.mean(axis=(0, 1))
                elif len(attn_np.shape) == 3:
                    # [batch, seq_q, seq_k] or [heads, seq_q, seq_k]
                    attn_np = attn_np.mean(axis=0)
                elif len(attn_np.shape) == 2:
                    # Already [seq_q, seq_k]
                    pass
                else:
                    logger.warning(f"Unexpected attention shape {attn_np.shape} for {name}")
                    continue
                
                # For self-attention, we need to reshape to spatial dimensions
                # Assuming attention is over patches, compute spatial map
                seq_len = attn_np.shape[0]
                
                # Try to infer spatial dimensions
                # SAM2 typically uses 64x64 = 4096 or 32x32 = 1024 patches
                possible_sizes = [(64, 64), (32, 32), (16, 16), (128, 128)]
                spatial_size = None
                
                for ps in possible_sizes:
                    if ps[0] * ps[1] == seq_len:
                        spatial_size = ps
                        break
                
                if spatial_size is None:
                    # Try square root
                    sqrt_len = int(np.sqrt(seq_len))
                    if sqrt_len * sqrt_len == seq_len:
                        spatial_size = (sqrt_len, sqrt_len)
                    else:
                        # Use mean over sequence dimension
                        attn_np = attn_np.mean(axis=-1) if len(attn_np.shape) > 1 else attn_np
                        sqrt_len = int(np.sqrt(len(attn_np)))
                        if sqrt_len * sqrt_len == len(attn_np):
                            spatial_size = (sqrt_len, sqrt_len)
                        else:
                            logger.warning(f"Cannot reshape {name} attention of shape {attn_np.shape}")
                            continue
                
                # Aggregate attention (sum over keys, keep queries which represent spatial locations)
                if len(attn_np.shape) == 2:
                    spatial_attn = attn_np.sum(axis=-1)  # Sum over keys
                else:
                    spatial_attn = attn_np
                
                # Reshape to spatial dimensions
                try:
                    spatial_attn = spatial_attn.reshape(spatial_size)
                except ValueError:
                    logger.warning(f"Could not reshape {name} to {spatial_size}")
                    continue
                
                # Resize to target size
                from PIL import Image
                pil_img = Image.fromarray(spatial_attn.astype(np.float32))
                pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
                resized_attn = np.array(pil_img)
                
                aggregated_maps.append(resized_attn)
                
            except Exception as e:
                logger.warning(f"Error processing attention from {name}: {e}")
                continue
        
        if len(aggregated_maps) == 0:
            return np.zeros(target_size)
        
        # Average all maps
        aggregated = np.mean(aggregated_maps, axis=0)
        
        # Normalize
        if self.normalize and aggregated.max() > aggregated.min():
            aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())
        
        # Apply Gaussian blur for smoother visualization
        if self.blur_sigma > 0:
            aggregated = gaussian_filter(aggregated, sigma=self.blur_sigma)
            # Re-normalize after blur
            if aggregated.max() > aggregated.min():
                aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())
        
        return aggregated
    
    def clear(self):
        """Clear stored attention maps"""
        self._attention_maps.clear()
        self._intermediate_outputs.clear()
        logger.debug("Cleared attention maps")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._hooks_registered = False
        logger.info("Removed all attention hooks")
    
    def __del__(self):
        """Cleanup hooks on deletion"""
        self.remove_hooks()


def create_attention_extractor(model: nn.Module, config: Optional[Any] = None) -> SAM2AttentionExtractor:
    """
    Factory function to create an attention extractor with config.
    
    Args:
        model: SAM2/MedSAM2 model
        config: XAI config object (optional)
        
    Returns:
        Configured SAM2AttentionExtractor
    """
    # Use config if provided, otherwise defaults
    if config is not None:
        return SAM2AttentionExtractor(
            model=model,
            target_layers=getattr(config, 'attention_layer_names', None),
            max_layers=getattr(config, 'max_attention_layers', 4),
            normalize=getattr(config, 'attention_normalize', True),
            blur_sigma=getattr(config, 'blur_sigma', 2.0)
        )
    else:
        return SAM2AttentionExtractor(model=model)
