#!/usr/bin/env python3
"""
Attention Visualization for TAM-ViT

Provides tools to visualize:
- Self-attention patterns across layers
- Attention rollout for interpretability
- Skin tone conditioning effects

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import cv2


class AttentionVisualizer:
    """
    Visualizes attention patterns from TAM-ViT model.
    
    Supports:
    - Per-layer attention maps
    - Attention rollout (accumulated attention)
    - CLS token attention highlighting
    """
    
    def __init__(
        self,
        model: nn.Module,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        self.model = model
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Storage for attention weights
        self.attention_weights: List[torch.Tensor] = []
        self._hooks = []
        
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self._clear_hooks()
        self.attention_weights = []
        
        def hook_fn(module, input, output):
            # For standard attention: output is (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_weights.append(attn_weights.detach().cpu())
        
        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(hook_fn)
                self._hooks.append(hook)
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.attention_weights = []
    
    @torch.no_grad()
    def get_attention_maps(
        self,
        image: torch.Tensor,
        return_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps from model forward pass.
        
        Args:
            image: Input image tensor (B, C, H, W)
            return_attention: Whether to return attention from model
            
        Returns:
            Dictionary containing attention weights per layer
        """
        self.model.eval()
        
        # Forward pass with attention output
        outputs = self.model(image, return_attention=return_attention)
        
        if 'attentions' in outputs:
            return {
                'attentions': outputs['attentions'],
                'tone_probs': outputs.get('tone_probs'),
            }
        
        return outputs
    
    def attention_rollout(
        self,
        attentions: List[torch.Tensor],
        discard_ratio: float = 0.1,
        head_fusion: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute attention rollout - accumulated attention through layers.
        
        This provides a more interpretable view of what the model attends to.
        
        Args:
            attentions: List of attention tensors, one per layer
            discard_ratio: Fraction of lowest attention to discard
            head_fusion: How to fuse heads ('mean', 'max', 'min')
            
        Returns:
            Rolled-out attention map (B, num_patches)
        """
        result = None
        
        for attention in attentions:
            # attention shape: (B, num_heads, seq_len, seq_len)
            
            # Fuse attention heads
            if head_fusion == 'mean':
                attention_fused = attention.mean(dim=1)  # (B, seq, seq)
            elif head_fusion == 'max':
                attention_fused = attention.max(dim=1)[0]
            elif head_fusion == 'min':
                attention_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head_fusion: {head_fusion}")
            
            # Discard lowest attention values
            if discard_ratio > 0:
                flat = attention_fused.view(attention_fused.size(0), -1)
                threshold = torch.quantile(flat, discard_ratio, dim=1, keepdim=True)
                threshold = threshold.view(-1, 1, 1)
                attention_fused = torch.where(
                    attention_fused > threshold,
                    attention_fused,
                    torch.zeros_like(attention_fused)
                )
            
            # Add identity (residual connection)
            I = torch.eye(attention_fused.size(-1)).to(attention_fused.device)
            attention_fused = attention_fused + I
            
            # Normalize rows
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True)
            
            # Accumulate
            if result is None:
                result = attention_fused
            else:
                result = torch.bmm(attention_fused, result)
        
        # Extract CLS token attention to patches
        # Assuming CLS token is at position 0
        cls_attention = result[:, 0, 1:]  # (B, num_patches)
        
        return cls_attention
    
    def create_attention_heatmap(
        self,
        attention: torch.Tensor,
        image: Optional[np.ndarray] = None,
        alpha: float = 0.6,
        colormap: str = 'jet',
    ) -> np.ndarray:
        """
        Create attention heatmap overlaid on image.
        
        Args:
            attention: Attention weights (num_patches,) or (H, W)
            image: Original image as numpy array (H, W, C)
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            Heatmap image as numpy array
        """
        # Reshape attention to grid
        if attention.dim() == 1:
            attention = attention.view(self.grid_size, self.grid_size)
        
        attention = attention.cpu().numpy()
        
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        # Resize to image size
        attention_resized = cv2.resize(
            attention,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap = cmap(attention_resized)[:, :, :3]  # RGB only
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay on image if provided
        if image is not None:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Blend
            overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
            return overlay
        
        return heatmap
    
    def visualize_layer_attention(
        self,
        attentions: List[torch.Tensor],
        image: np.ndarray,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Visualize attention from a specific layer and head.
        
        Args:
            attentions: List of attention tensors
            image: Original image
            layer_idx: Which layer to visualize (-1 for last)
            head_idx: Which head to visualize (None for mean)
            
        Returns:
            Visualization as numpy array
        """
        attention = attentions[layer_idx]  # (B, heads, seq, seq)
        
        if head_idx is not None:
            attention = attention[:, head_idx]  # (B, seq, seq)
        else:
            attention = attention.mean(dim=1)  # (B, seq, seq)
        
        # Get CLS attention to patches
        cls_attention = attention[0, 0, 1:]  # (num_patches,)
        
        return self.create_attention_heatmap(cls_attention, image)
    
    def create_multi_head_visualization(
        self,
        attentions: List[torch.Tensor],
        image: np.ndarray,
        layer_idx: int = -1,
        num_heads_to_show: int = 12,
    ) -> plt.Figure:
        """
        Create grid visualization of multiple attention heads.
        
        Args:
            attentions: List of attention tensors
            image: Original image
            layer_idx: Which layer to visualize
            num_heads_to_show: Number of heads to display
            
        Returns:
            Matplotlib figure
        """
        attention = attentions[layer_idx][0]  # (heads, seq, seq)
        num_heads = min(attention.size(0), num_heads_to_show)
        
        # Calculate grid dimensions
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for i in range(num_heads):
            cls_attention = attention[i, 0, 1:]  # CLS -> patches
            heatmap = self.create_attention_heatmap(cls_attention, image.copy())
            axes[i].imshow(heatmap)
            axes[i].set_title(f'Head {i}')
            axes[i].axis('off')
        
        # Hide unused axes
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


def visualize_attention_maps(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to visualize attention maps.
    
    Args:
        model: TAM-ViT model
        image: Preprocessed input tensor
        original_image: Original image as numpy array
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary of visualization images
    """
    visualizer = AttentionVisualizer(model)
    
    # Get attention maps
    outputs = visualizer.get_attention_maps(image)
    attentions = outputs.get('attentions', [])
    
    if not attentions:
        print("Warning: No attention maps returned. Ensure model returns attention.")
        return {}
    
    results = {}
    
    # Attention rollout
    rollout = visualizer.attention_rollout(attentions)
    results['rollout'] = visualizer.create_attention_heatmap(
        rollout[0], original_image
    )
    
    # Last layer attention
    results['last_layer'] = visualizer.visualize_layer_attention(
        attentions, original_image, layer_idx=-1
    )
    
    # Multi-head visualization
    fig = visualizer.create_multi_head_visualization(
        attentions, original_image
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-head visualization to: {save_path}")
    
    plt.close(fig)
    
    return results
