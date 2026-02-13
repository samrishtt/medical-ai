"""
GradCAM and GradCAM++ for TAM-ViT

Provides gradient-based class activation mapping for:
- Understanding model decisions
- Identifying diagnostic regions
- Validating clinical relevance
- Uncertainty-aware visualization

Supports multi-scale patch attention visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    import cv2
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Vision Transformers.
    
    Adapted for ViT architecture where we use attention-weighted gradients
    instead of traditional CNN activation maps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        img_size: int = 224,
        patch_size: int = 16,
    ):
        """
        Args:
            model: TAM-ViT model
            target_layer: Name of layer to compute GradCAM on.
                         If None, uses last transformer block.
            img_size: Input image size
            patch_size: Patch size used in model
        """
        self.model = model
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        # Find target layer
        self.target_layer = self._find_target_layer(target_layer)
        
        # Storage for gradients and activations
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self, layer_name: Optional[str] = None) -> nn.Module:
        """Find the target layer for GradCAM computation."""
        if layer_name is not None:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    return module
            raise ValueError(f"Layer '{layer_name}' not found in model")
        
        # Default: find last transformer block or attention layer
        target = None
        for name, module in self.model.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                target = module
        
        if target is None:
            # Fallback to last named module
            modules = list(self.model.named_modules())
            if modules:
                target = modules[-1][1]
        
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        retain_graph: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute GradCAM for the input image.
        
        Args:
            image: Input tensor (B, C, H, W)
            target_class: Target class for GradCAM. If None, uses predicted class.
            retain_graph: Whether to retain computation graph
            
        Returns:
            Tuple of (cam_map, target_class)
        """
        self.model.eval()
        
        # Enable gradients for input
        image.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(image, return_uncertainty=False)
        logits = outputs['logits']
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=retain_graph)
        
        # Compute GradCAM
        cam = self._compute_cam()
        
        return cam, target_class
    
    def _compute_cam(self) -> np.ndarray:
        """Compute the class activation map."""
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. "
                             "Ensure hooks are registered correctly.")
        
        # For ViT: activations shape is typically (B, seq_len, hidden_dim)
        # We need to handle the CLS token and patch tokens differently
        
        gradients = self.gradients
        activations = self.activations
        
        # Handle different tensor shapes
        if len(activations.shape) == 3:
            # Transformer output: (B, seq_len, hidden_dim)
            # Remove CLS token if present (assuming position 0)
            if activations.size(1) == self.grid_size ** 2 + 1:
                gradients = gradients[:, 1:, :]  # Remove CLS
                activations = activations[:, 1:, :]
            
            # Global average pooling of gradients for weights
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, hidden_dim)
            
            # Weighted combination
            cam = (weights * activations).sum(dim=-1)  # (B, seq_len)
            
            # Reshape to grid
            cam = cam.view(1, self.grid_size, self.grid_size)
            
        elif len(activations.shape) == 4:
            # CNN-like output: (B, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1)
        
        else:
            raise ValueError(f"Unexpected activation shape: {activations.shape}")
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to image size
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        return cam[0, 0].cpu().numpy()
    
    def create_visualization(
        self,
        cam: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.5,
        colormap: str = 'jet',
    ) -> np.ndarray:
        """
        Create CAM visualization overlaid on image.
        
        Args:
            cam: Class activation map
            image: Original image (H, W, C)
            alpha: Overlay transparency
            colormap: Matplotlib colormap
            
        Returns:
            Visualization as numpy array
        """
        # Ensure proper scaling
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        cam_colored = cmap(cam)[:, :, :3]  # RGB
        cam_colored = (cam_colored * 255).astype(np.uint8)
        
        # Resize if needed
        if cam_colored.shape[:2] != image.shape[:2]:
            cam_colored = cv2.resize(cam_colored, (image.shape[1], image.shape[0]))
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, cam_colored, alpha, 0)
        
        return overlay


class UncertaintyAwareGradCAM(GradCAM):
    """
    GradCAM weighted by model uncertainty.
    
    Regions with higher model uncertainty get darker visualization,
    indicating less confident areas.
    """
    
    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        uncertainty_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Compute uncertainty-weighted GradCAM.
        
        Args:
            image: Input tensor
            target_class: Target class for visualization
            uncertainty_map: Model uncertainty estimates (H, W)
            
        Returns:
            Weighted CAM and target class
        """
        self.model.eval()
        image.requires_grad_(True)
        
        # Forward pass with uncertainty
        outputs = self.model(image, return_uncertainty=True)
        logits = outputs['logits']
        
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Get uncertainty if not provided
        if uncertainty_map is None and 'variance' in outputs:
            uncertainty = outputs['variance'].detach().cpu().numpy()
            uncertainty_map = uncertainty.mean(axis=1)  # Average across classes
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute standard CAM
        cam = self._compute_cam()
        
        # Weight by uncertainty (inverse: high uncertainty -> darker)
        if uncertainty_map is not None:
            uncertainty_normalized = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min() + 1e-8)
            uncertainty_resized = cv2.resize(uncertainty_normalized, (self.img_size, self.img_size))
            
            # Darker regions where model is uncertain
            cam = cam * (1 - uncertainty_resized) if VISUALIZATION_AVAILABLE else cam
        
        return cam, target_class


def extract_attention_maps(
    model: nn.Module,
    image: torch.Tensor,
    layer_idx: int = -1,
) -> Dict[str, np.ndarray]:
    """
    Extract attention maps from Vision Transformer blocks.
    
    Args:
        model: TAM-ViT model
        image: Input tensor
        layer_idx: Which block to extract from (-1 for last)
        
    Returns:
        Dictionary with attention weights for each head
    """
    with torch.no_grad():
        outputs = model.forward_features(image, return_attention=True)
        
        if 'attention' not in outputs:
            return {}
        
        attention = outputs['attention']  # (depth, B, num_heads, seq_len, seq_len)
        
        # Get attention from specified layer
        if attention.ndim == 5:
            layer_attention = attention[layer_idx]  # (B, num_heads, seq_len, seq_len)
        else:
            layer_attention = attention
        
        # Average over batch if needed
        if layer_attention.shape[0] == 1:
            layer_attention = layer_attention[0]  # (num_heads, seq_len, seq_len)
        
        attention_maps = {}
        for head_idx in range(layer_attention.shape[0]):
            # Skip CLS token attention
            head_attn = layer_attention[head_idx, 1:, 1:]  # Remove CLS
            
            # Reshape to grid
            grid_size = int(np.sqrt(head_attn.shape[0]))
            if grid_size ** 2 == head_attn.shape[0]:
                head_attn = head_attn.view(grid_size, grid_size, grid_size, grid_size)
                # Average spatial dims
                head_attn = head_attn.sum(dim=[0, 1]) / (grid_size * grid_size)
                attention_maps[f'head_{head_idx}'] = head_attn.cpu().numpy()
        
        return attention_maps


def create_attention_visualization(
    attention_maps: Dict[str, np.ndarray],
    image: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize attention maps overlaid on image.
    
    Args:
        attention_maps: Dictionary of attention maps per head
        image: Original image
        save_path: Optional path to save figure
    """
    if not VISUALIZATION_AVAILABLE or not attention_maps:
        return
    
    n_heads = len(attention_maps)
    fig_size = (4 * n_heads, 4)
    
    fig, axes = plt.subplots(1, n_heads + 1, figsize=fig_size)
    if n_heads == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention maps
    for idx, (name, attn_map) in enumerate(attention_maps.items(), 1):
        attn_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
        
        axes[idx].imshow(image)
        axes[idx].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to: {save_path}")
    
    return fig


def apply_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    method: str = 'gradcam',
    uncertainty_aware: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Convenience function to apply GradCAM visualization.
    
    Args:
        model: TAM-ViT model
        image: Preprocessed input tensor
        original_image: Original image as numpy array (H, W, C) in [0, 255]
        target_class: Target class (None for predicted)
        method: 'gradcam' or 'gradcam++'
        uncertainty_aware: Whether to weight by model uncertainty
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary with 'cam', 'visualization', and 'target_class'
    """
    # Select CAM method
    if uncertainty_aware:
        cam_extractor = UncertaintyAwareGradCAM(model)
    elif method == 'gradcam++':
        cam_extractor = GradCAMPlusPlus(model)
    elif method == 'gradcam':
        cam_extractor = GradCAM(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute CAM
    cam, pred_class = cam_extractor(image, target_class)
    
    # Create visualization
    viz = cam_extractor.create_visualization(cam, original_image)
    
    # Save if requested
    if save_path and VISUALIZATION_AVAILABLE:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        if original_image.max() <= 1.0:
            plt.imshow((original_image * 255).astype(np.uint8))
        else:
            plt.imshow(original_image.astype(np.uint8))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Class Activation Map')
        plt.colorbar(fraction=0.046)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(viz.astype(np.uint8))
        plt.title(f'GradCAM (Class: {pred_class})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved GradCAM visualization to: {save_path}")
    
    result = {
        'cam': cam,
        'visualization': viz,
        'target_class': pred_class,
        'method': method,
    }
    
    if uncertainty_aware:
        result['uncertainty_weighted'] = True
    
    return result


def generate_model_explanation(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    output_dir: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Generate comprehensive model explanation with multiple visualization methods.
    
    Args:
        model: TAM-ViT model
        image: Preprocessed input tensor
        original_image: Original image
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with all explanations
    """
    explanations = {}
    output_dir = output_dir or "./explanations"
    
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # GradCAM
    save_path = f"{output_dir}/gradcam.png" if output_dir else None
    explanations['gradcam'] = apply_gradcam(
        model, image, original_image,
        method='gradcam',
        save_path=save_path
    )
    
    # GradCAM++
    save_path = f"{output_dir}/gradcam_plus.png" if output_dir else None
    try:
        explanations['gradcam++'] = apply_gradcam(
            model, image, original_image,
            method='gradcam++',
            save_path=save_path
        )
    except Exception as e:
        print(f"⚠️  GradCAM++ failed: {e}")
    
    # Uncertainty-aware GradCAM
    save_path = f"{output_dir}/gradcam_uncertainty.png" if output_dir else None
    try:
        explanations['gradcam_uncertainty'] = apply_gradcam(
            model, image, original_image,
            method='gradcam',
            uncertainty_aware=True,
            save_path=save_path
        )
    except Exception as e:
        print(f"⚠️  Uncertainty-aware GradCAM failed: {e}")
    
    # Attention maps
    try:
        attention_maps = extract_attention_maps(model, image, layer_idx=-1)
        if attention_maps:
            save_path = f"{output_dir}/attention.png" if output_dir else None
            create_attention_visualization(attention_maps, original_image, save_path)
            explanations['attention'] = attention_maps
    except Exception as e:
        print(f"⚠️  Attention visualization failed: {e}")
    
    return explanations
