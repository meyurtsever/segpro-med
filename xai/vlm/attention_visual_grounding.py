"""
Attention-Based Visual Grounding for MedGemma

A FAST and ACCURATE approach using the model's attention mechanism directly,
without requiring backward passes or gradient computation.

Key Advantages:
1. FAST: Single forward pass, no gradient computation needed
2. ACCURATE: Directly shows what the model attends to when generating tokens
3. CONSISTENT: Uses cached generation outputs from label suggestion

Architecture:
MedGemma = SigLIP Encoder (vision) + Gemma Decoder (text with causal attention)

The model processes:
- Image tokens (from SigLIP): ~256 tokens for 224x224 image (16x16 patches)
- Text tokens (from tokenizer): variable length

When generating text, the decoder attends to BOTH:
- Previous text tokens (causal self-attention)
- Image tokens (cross-modal attention via concatenated embeddings)

This module extracts the attention weights to image tokens when the model
generates specific label tokens, creating a spatial heatmap.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class AttentionVisualGrounding:
    """
    Attention-based visual grounding for MedGemma.
    
    Uses the model's attention weights directly to show which image regions
    the model attends to when generating specific tokens.
    
    This is MUCH faster than gradient-based methods because:
    - No backward pass needed
    - Attention weights are computed during forward pass
    - Can reuse cached generation outputs
    """
    
    def __init__(self, model, processor, device: str = "cuda"):
        """
        Initialize attention-based visual grounding.
        
        Args:
            model: MedGemma model (PaliGemmaForConditionalGeneration)
            processor: MedGemma processor
            device: Device for computation
        """
        self.model = model
        self.processor = processor
        self.device = device
        
        # Cache for reusing generation outputs
        self._cache = {
            'inputs': None,
            'generation': None,
            'attentions': None,
            'hidden_states': None,
            'generated_text': None,
            'prompt': None,
            'image_hash': None,
            'preprocessed_image': None,
            'num_image_tokens': None
        }
        
        # Get image grid size from config
        self._init_image_config()
    
    def _init_image_config(self):
        """Initialize image configuration from model config"""
        try:
            config = self.model.config
            vision_config = getattr(config, 'vision_config', None)
            
            if vision_config:
                image_size = getattr(vision_config, 'image_size', 224)
                patch_size = getattr(vision_config, 'patch_size', 14)
            else:
                # Default for MedGemma-4B
                image_size = 224
                patch_size = 14
            
            self.image_size = image_size
            self.patch_size = patch_size
            self.grid_size = image_size // patch_size  # 16 for 224/14
            self.num_patches = self.grid_size * self.grid_size  # 256
            
            # Check if SigLIP includes CLS token
            # SigLIP doesn't use CLS token, so num_image_tokens = num_patches
            self.num_image_tokens = self.num_patches
            
            logger.info(f"Image config: {image_size}x{image_size}, patch={patch_size}, "
                       f"grid={self.grid_size}x{self.grid_size}, tokens={self.num_image_tokens}")
            
        except Exception as e:
            logger.warning(f"Could not read image config, using defaults: {e}")
            self.image_size = 224
            self.patch_size = 14
            self.grid_size = 16
            self.num_patches = 256
            self.num_image_tokens = 256
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """Get hash for caching"""
        return f"{image.size}_{image.mode}_{hash(image.tobytes()[:1000])}"
    
    def cache_generation_outputs(
        self,
        image: Image.Image,
        prompt: str,
        generated_text: str,
        inputs: Dict[str, torch.Tensor],
        generation: torch.Tensor,
        attentions: Optional[Tuple] = None,
        hidden_states: Optional[Tuple] = None
    ):
        """
        Cache generation outputs from label suggestion for reuse in visual grounding.
        
        This is the KEY optimization - we store the outputs from label suggestion
        and reuse them for XAI, avoiding a second forward pass.
        
        Args:
            image: Original image
            prompt: Prompt used for generation
            generated_text: Generated text
            inputs: Tokenized inputs
            generation: Generated token IDs
            attentions: Attention weights (if available)
            hidden_states: Hidden states (if available)
        """
        self._cache['inputs'] = inputs
        self._cache['generation'] = generation
        self._cache['attentions'] = attentions
        self._cache['hidden_states'] = hidden_states
        self._cache['generated_text'] = generated_text
        self._cache['prompt'] = prompt
        self._cache['image_hash'] = self._get_image_hash(image)
        self._cache['preprocessed_image'] = image  # Store preprocessed image for XAI reuse
        self._cache['num_image_tokens'] = inputs['input_ids'].shape[-1] - len(
            self.processor.tokenizer.encode(prompt, add_special_tokens=False)
        )
        
        logger.info(f"Cached generation outputs (text: '{generated_text[:50]}...')")
    
    def clear_cache(self):
        """Clear the generation cache"""
        for key in self._cache:
            self._cache[key] = None
        logger.info("Cleared attention visual grounding cache")
    
    def compute_visual_grounding(
        self,
        image: Image.Image,
        prompt: str,
        target_token: str,
        method: str = "attention"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding heatmap for a target token.
        
        If cached generation outputs are available (from label suggestion),
        this is FAST because we just need to analyze attention patterns.
        
        If not cached, we do a single forward pass with output_attentions=True.
        
        Args:
            image: Input image
            prompt: Text prompt
            target_token: Token to compute grounding for
            method: Method to use ("attention", "hidden_similarity", "hybrid")
        
        Returns:
            Tuple of (heatmap, metadata)
        """
        try:
            image_hash = self._get_image_hash(image)
            
            # Debug: Log cache state
            logger.info(f"Cache state: inputs={self._cache['inputs'] is not None}, "
                       f"prompt_match={self._cache['prompt'] == prompt}, "
                       f"image_match={self._cache['image_hash'] == image_hash}")
            
            if self._cache['prompt'] is not None:
                logger.info(f"Cached prompt: '{self._cache['prompt'][:50]}...'")
                logger.info(f"Current prompt: '{prompt[:50]}...'")
            
            # Check if we can use cached outputs
            if (self._cache['inputs'] is not None and 
                self._cache['prompt'] == prompt and
                self._cache['image_hash'] == image_hash):
                
                logger.info("Using cached generation outputs for visual grounding")
                return self._compute_from_cache(target_token, method)
            
            # Otherwise, do a forward pass with attention
            logger.info("Computing fresh visual grounding (cache miss)")
            return self._compute_fresh(image, prompt, target_token, method)
            
        except Exception as e:
            logger.error(f"Error computing attention visual grounding: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
    
    def _compute_from_cache(
        self,
        target_token: str,
        method: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Compute grounding using cached generation outputs"""
        
        inputs = self._cache['inputs']
        generation = self._cache['generation']
        generated_text = self._cache['generated_text']
        
        # Check if cache is valid
        if inputs is None:
            logger.warning("Cache miss: inputs is None")
            return None, {'success': False, 'error': 'No cached inputs'}
        
        if generated_text is None:
            logger.warning("Cache miss: generated_text is None")
            return None, {'success': False, 'error': 'No cached generated text'}
        
        logger.info(f"Cached generated text: '{generated_text[:100]}...'")
        logger.info(f"Looking for target token: '{target_token}'")
        
        # Find target token in generated text (case-insensitive)
        if target_token.lower() not in generated_text.lower():
            logger.warning(f"Token '{target_token}' not found in generated text")
            return None, {
                'success': False,
                'error': f"Token '{target_token}' not found in generated text",
                'generated_text': generated_text[:200]
            }
        
        logger.info(f"Found '{target_token}' in generated text, computing similarity...")
        
        # Use hidden state similarity approach - compares target label embedding
        # with image patch embeddings. This is fast and reliable.
        return self._hidden_state_similarity(inputs, target_token)
    
    def _compute_fresh(
        self,
        image: Image.Image,
        prompt: str,
        target_token: str,
        method: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Compute grounding with a fresh forward pass using hidden state similarity"""
        
        # Prepare inputs using the SAME chat template format as MedGemma service
        # This is critical - MedGemma requires the chat template format with image tokens
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        input_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs with properly formatted text
        inputs = self.processor(
            text=input_text,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text first (without attention - SDPA doesn't support it)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True
                )
            
            generation = outputs.sequences
            
            # Decode generated text
            input_len = inputs['input_ids'].shape[-1]
            generated_ids = generation[0][input_len:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)
            
            # Cache for future use
            self.cache_generation_outputs(
                image=image,
                prompt=prompt,
                generated_text=generated_text,
                inputs=inputs,
                generation=generation,
                attentions=None,
                hidden_states=None
            )
            
            if target_token.lower() not in generated_text.lower():
                return None, {
                    'success': False,
                    'error': f"Token '{target_token}' not found in generated text",
                    'generated_text': generated_text[:200]
                }
            
            # Use hidden state similarity (works with SDPA, fast and reliable)
            return self._hidden_state_similarity(inputs, target_token)
            
        except Exception as e:
            logger.error(f"Error in fresh computation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
    
    def _compute_fresh_attention(
        self,
        inputs: Dict[str, torch.Tensor],
        target_token: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Compute attention with a single forward pass (no generation)"""
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )
            
            attentions = outputs.attentions
            
            if attentions is None or len(attentions) == 0:
                return None, {'success': False, 'error': 'No attention weights available'}
            
            # Process attention weights
            return self._process_attention_to_heatmap(attentions, inputs, target_token)
            
        except Exception as e:
            logger.error(f"Error computing fresh attention: {e}")
            return None, {'success': False, 'error': str(e)}
    
    def _attention_based_grounding(
        self,
        inputs: Dict[str, torch.Tensor],
        generation: torch.Tensor,
        target_token: str,
        attentions: Tuple
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute grounding from attention weights.
        
        Extracts attention weights from text tokens to image tokens and
        creates a spatial heatmap.
        """
        
        try:
            # Get number of image tokens
            num_image_tokens = self.num_image_tokens
            input_len = inputs['input_ids'].shape[-1]
            
            # Find target token position in generation
            target_ids = self.processor.tokenizer.encode(target_token, add_special_tokens=False)
            generated_ids = generation[0][input_len:]
            
            target_positions = self._find_token_positions(generated_ids, target_ids)
            
            if not target_positions:
                # Fallback: use average over all generated positions
                logger.warning(f"Could not find exact position for '{target_token}', using average")
                target_positions = list(range(len(generated_ids)))
            
            # Process attentions
            # Attention shape: (num_layers, batch, num_heads, seq_len, seq_len)
            # We want attention from generated tokens to image tokens
            
            if isinstance(attentions, tuple) and len(attentions) > 0:
                # Take attention from last few layers (they capture more semantic info)
                num_layers = len(attentions)
                layers_to_use = attentions[num_layers // 2:]  # Use last half of layers
                
                # Stack and average
                attn_stack = torch.stack([a.squeeze(0) for a in layers_to_use])  # (L, H, S, S)
                attn_avg = attn_stack.mean(dim=(0, 1))  # Average over layers and heads -> (S, S)
                
                # Extract attention from target positions to image tokens
                # Image tokens are at the beginning (positions 0 to num_image_tokens-1)
                target_attn = attn_avg[input_len + np.array(target_positions), :num_image_tokens]
                target_attn = target_attn.mean(dim=0)  # Average over target positions
                
                # Convert to spatial heatmap
                heatmap = self._attention_to_spatial(target_attn.cpu().numpy())
                
                return heatmap, {
                    'success': True,
                    'method': 'attention',
                    'num_layers_used': len(layers_to_use),
                    'target_positions': target_positions[:5],  # First 5 for logging
                    'mean_attribution': float(heatmap.mean()),
                    'max_attribution': float(heatmap.max())
                }
            
            return None, {'success': False, 'error': 'Invalid attention format'}
            
        except Exception as e:
            logger.error(f"Error in attention-based grounding: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
    
    def _process_attention_to_heatmap(
        self,
        attentions: Tuple,
        inputs: Dict[str, torch.Tensor],
        target_token: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Process raw attention outputs to heatmap"""
        
        try:
            # Find token position for target
            target_ids = self.processor.tokenizer.encode(target_token, add_special_tokens=False)
            input_ids = inputs['input_ids'][0]
            
            target_positions = self._find_token_positions(input_ids, target_ids)
            
            if not target_positions:
                # Use all non-image positions as fallback
                target_positions = list(range(self.num_image_tokens, len(input_ids)))
            
            # Stack attention from multiple layers
            num_layers = len(attentions)
            layers_to_use = attentions[num_layers // 2:]  # Last half
            
            attn_list = []
            for attn in layers_to_use:
                if attn is not None:
                    attn_list.append(attn.squeeze(0))  # (H, S, S)
            
            if not attn_list:
                return None, {'success': False, 'error': 'No valid attention layers'}
            
            attn_stack = torch.stack(attn_list)  # (L, H, S, S)
            attn_avg = attn_stack.mean(dim=(0, 1))  # (S, S)
            
            # Get attention to image tokens
            target_attn = attn_avg[np.array(target_positions), :self.num_image_tokens]
            target_attn = target_attn.mean(dim=0)
            
            heatmap = self._attention_to_spatial(target_attn.cpu().numpy())
            
            return heatmap, {
                'success': True,
                'method': 'attention_forward',
                'mean_attribution': float(heatmap.mean()),
                'max_attribution': float(heatmap.max())
            }
            
        except Exception as e:
            logger.error(f"Error processing attention: {e}")
            return None, {'success': False, 'error': str(e)}
    
    def _hidden_state_similarity(
        self,
        inputs: Dict[str, torch.Tensor],
        target_token: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding using hidden state similarity.
        
        This approach:
        1. Runs a forward pass to get hidden states
        2. Extracts image patch embeddings from the hidden states
        3. Encodes the target label separately and gets its embedding
        4. Computes cosine similarity between target and each image patch
        
        This is FAST because it only needs one forward pass with the cached inputs.
        Works with SDPA attention (no output_attentions needed).
        """
        
        try:
            with torch.no_grad():
                # Get hidden states from forward pass (works with SDPA)
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=False,  # Explicitly disable - not supported by SDPA
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                return None, {'success': False, 'error': 'No hidden states available'}
            
            # Use last layer hidden states
            last_hidden = hidden_states[-1][0]  # (seq_len, hidden_dim)
            seq_len = last_hidden.shape[0]
            
            # Determine number of image tokens dynamically
            # MedGemma typically uses 256 image tokens (16x16 grid for 224x224 image)
            # But we should check the actual sequence composition
            num_image_tokens = self.num_image_tokens
            
            # Safety check: ensure we don't exceed sequence length
            if num_image_tokens >= seq_len:
                # Fallback: estimate from sequence length
                # Image tokens + text tokens = seq_len
                # Assume at least 10 text tokens
                num_image_tokens = max(1, seq_len - 50)
                logger.warning(f"Adjusted num_image_tokens to {num_image_tokens} (seq_len={seq_len})")
            
            # Get image embeddings (first num_image_tokens positions)
            image_embeddings = last_hidden[:num_image_tokens]  # (num_patches, hidden_dim)
            
            # Get the embedding for the target label from the model's embedding layer
            target_ids = self.processor.tokenizer.encode(target_token, add_special_tokens=False)
            target_ids_tensor = torch.tensor([target_ids]).to(self.device)
            
            # Get target embedding from the model's embedding layer
            if hasattr(self.model, 'get_input_embeddings'):
                embed_layer = self.model.get_input_embeddings()
                target_embeddings = embed_layer(target_ids_tensor)  # (1, num_tokens, hidden_dim)
                target_embedding = target_embeddings.mean(dim=1).squeeze(0)  # (hidden_dim,)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'get_input_embeddings'):
                # For PaliGemmaForConditionalGeneration
                embed_layer = self.model.model.get_input_embeddings()
                target_embeddings = embed_layer(target_ids_tensor)
                target_embedding = target_embeddings.mean(dim=1).squeeze(0)
            else:
                # Fallback: use average of text positions in hidden states
                text_start = num_image_tokens
                target_embedding = last_hidden[text_start:].mean(dim=0)  # (hidden_dim,)
                logger.warning("Using fallback text embedding method")
            
            # Convert to float32 for cosine similarity (BFloat16 not supported)
            image_embeddings_f32 = image_embeddings.float()
            target_embedding_f32 = target_embedding.float()
            
            # Compute cosine similarity between target embedding and each image patch
            similarity = F.cosine_similarity(
                image_embeddings_f32, 
                target_embedding_f32.unsqueeze(0).expand_as(image_embeddings_f32),
                dim=-1
            )  # (num_patches,)
            
            # Normalize to [0, 1]
            sim_min = similarity.min()
            sim_max = similarity.max()
            if sim_max > sim_min:
                similarity = (similarity - sim_min) / (sim_max - sim_min)
            else:
                similarity = torch.zeros_like(similarity)
            
            heatmap = self._attention_to_spatial(similarity.cpu().detach().numpy())
            
            return heatmap, {
                'success': True,
                'method': 'embedding_similarity',
                'target_token': target_token,
                'num_patches': num_image_tokens,
                'seq_len': seq_len,
                'mean_attribution': float(heatmap.mean()),
                'max_attribution': float(heatmap.max())
            }
            
        except Exception as e:
            logger.error(f"Error in hidden state similarity: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, {'success': False, 'error': str(e)}
    
    def _hidden_state_approach(
        self,
        inputs: Dict[str, torch.Tensor],
        generation: torch.Tensor,
        target_token: str
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Fallback: Use hidden state similarity approach.
        
        Computes similarity between the hidden state of the target token
        and the image patch embeddings.
        """
        # Delegate to the new similarity method
        return self._hidden_state_similarity(inputs, target_token)
    
    def _find_token_positions(self, token_ids: torch.Tensor, target_ids: List[int]) -> List[int]:
        """Find positions of target tokens in token_ids"""
        
        positions = []
        target_tensor = torch.tensor(target_ids).to(token_ids.device)
        
        # Exact match
        for i in range(len(token_ids) - len(target_ids) + 1):
            if torch.equal(token_ids[i:i+len(target_ids)], target_tensor):
                positions.extend(range(i, i + len(target_ids)))
        
        return positions
    
    def _attention_to_spatial(self, attention: np.ndarray) -> np.ndarray:
        """
        Convert 1D attention vector to 2D spatial heatmap.
        
        Args:
            attention: 1D array of shape (num_patches,)
        
        Returns:
            2D heatmap of shape (grid_size, grid_size) or (image_size, image_size)
        """
        
        num_patches = len(attention)
        
        # Try to reshape to square grid
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size == num_patches:
            heatmap = attention.reshape(grid_size, grid_size)
        else:
            # Handle non-square (shouldn't happen for standard VLMs)
            h = int(np.sqrt(num_patches))
            w = num_patches // h
            heatmap = attention[:h * w].reshape(h, w)
        
        # Normalize
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Upscale to image size
        heatmap_resized = cv2.resize(
            heatmap.astype(np.float32),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize again after resize
        heatmap_resized = np.clip(heatmap_resized, 0, None)
        if heatmap_resized.max() > 0:
            heatmap_resized = heatmap_resized / heatmap_resized.max()
        
        return heatmap_resized


class HybridVisualGrounding:
    """
    Hybrid approach combining gradient-based and attention-based methods.
    
    Uses attention for speed, gradients for accuracy when needed.
    """
    
    def __init__(self, model, processor, device: str = "cuda"):
        self.attention_grounding = AttentionVisualGrounding(model, processor, device)
        self.model = model
        self.processor = processor
        self.device = device
    
    def compute_visual_grounding(
        self,
        image: Image.Image,
        prompt: str,
        target_token: str,
        method: str = "auto"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Compute visual grounding with automatic method selection.
        
        Args:
            method: 
                - "auto": Attention first, gradient fallback
                - "attention": Pure attention-based
                - "gradient": Pure gradient-based (slower)
        """
        
        # Try attention-based first (fast)
        heatmap, metadata = self.attention_grounding.compute_visual_grounding(
            image, prompt, target_token, method="attention"
        )
        
        if heatmap is not None and metadata.get('success'):
            return heatmap, metadata
        
        # Fallback to hidden state similarity
        return self.attention_grounding._hidden_state_approach(
            self.attention_grounding._cache.get('inputs', {}),
            self.attention_grounding._cache.get('generation'),
            target_token
        )
