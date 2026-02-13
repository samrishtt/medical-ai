#!/usr/bin/env python3
"""
GradCAM and GradCAM++ for TAM-ViT

Provides gradient-based class activation mapping for:
- Understanding model decisions
- Identifying diagnostic regions
- Validating clinical relevance

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import cv2


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


class GradCAMPlusPlus(GradCAM):
    """
    GradCAM++ provides better localization by using weighted gradients.
    
    Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based
    Visual Explanations for Deep Convolutional Networks"
    """
    
    def _compute_cam(self) -> np.ndarray:
        """Compute GradCAM++ activation map."""
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured.")
        
        gradients = self.gradients
        activations = self.activations
        
        # Handle transformer outputs
        if len(activations.shape) == 3:
            # Remove CLS token
            if activations.size(1) == self.grid_size ** 2 + 1:
                gradients = gradients[:, 1:, :]
                activations = activations[:, 1:, :]
            
            # GradCAM++ weights
            grad_2 = gradients ** 2
            grad_3 = gradients ** 3
            
            # Sum over sequence dimension
            sum_activations = activations.sum(dim=1, keepdim=True)
            
            # Alpha weights
            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
            alpha = alpha_num / alpha_denom
            
            # Weight by ReLU of gradients
            weights = (alpha * F.relu(gradients)).sum(dim=1, keepdim=True)
            
            # Weighted combination
            cam = (weights * activations).sum(dim=-1)
            cam = cam.view(1, self.grid_size, self.grid_size)
            
        else:
            # Fallback to standard GradCAM
            return super()._compute_cam()
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        
        return cam[0, 0].cpu().numpy()


def apply_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    method: str = 'gradcam',
    save_path: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Convenience function to apply GradCAM visualization.
    
    Args:
        model: TAM-ViT model
        image: Preprocessed input tensor
        original_image: Original image as numpy array
        target_class: Target class (None for predicted)
        method: 'gradcam' or 'gradcam++'
        save_path: Optional path to save visualization
        
    Returns:
        Dictionary with 'cam', 'visualization', and 'target_class'
    """
    if method == 'gradcam':
        cam_extractor = GradCAM(model)
    elif method == 'gradcam++':
        cam_extractor = GradCAMPlusPlus(model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute CAM
    cam, pred_class = cam_extractor(image, target_class)
    
    # Create visualization
    viz = cam_extractor.create_visualization(cam, original_image)
    
    # Save if requested
    if save_path:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap='jet')
        plt.title('Class Activation Map')
        plt.colorbar(fraction=0.046)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(viz)
        plt.title(f'GradCAM (Class: {pred_class})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved GradCAM visualization to: {save_path}")
    
    return {
        'cam': cam,
        'visualization': viz,
        'target_class': pred_class,
    }
