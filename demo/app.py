#!/usr/bin/env python3
"""
DERM-EQUITY Interactive Demo

Gradio-based web interface for:
- Image upload and prediction
- Uncertainty visualization
- Skin tone analysis
- Fairness explanation

Usage:
    python demo/app.py --model checkpoints/best.ckpt

Author: [Your Name]
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.tam_vit import create_tam_vit_base


# Global model
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
CLASS_NAMES = [
    'Melanoma', 'Melanocytic Nevus', 'Basal Cell Carcinoma',
    'Actinic Keratosis', 'Benign Keratosis', 'Dermatofibroma',
    'Vascular Lesion', 'Squamous Cell Carcinoma', 'Unknown'
]

# Fitzpatrick type descriptions
FITZPATRICK_NAMES = [
    'Type I: Very Fair', 'Type II: Fair', 'Type III: Medium',
    'Type IV: Olive', 'Type V: Brown', 'Type VI: Dark Brown/Black'
]


def load_model(checkpoint_path: str = None):
    """Load trained model."""
    global MODEL
    
    print(f"Loading model on {DEVICE}...")
    
    MODEL = create_tam_vit_base(num_classes=9)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            # PyTorch Lightning checkpoint
            state_dict = {k.replace('model.', ''): v 
                         for k, v in checkpoint['state_dict'].items()}
            MODEL.load_state_dict(state_dict, strict=False)
        else:
            MODEL.load_state_dict(checkpoint, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Using randomly initialized weights (demo mode)")
    
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    return MODEL


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize
    image = image.resize((224, 224))
    
    # Convert to tensor
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    return tensor


def create_uncertainty_plot(mean_probs, epistemic, aleatoric):
    """Create uncertainty visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Class probabilities
    ax1 = axes[0]
    top_k = 5
    top_indices = np.argsort(mean_probs)[-top_k:][::-1]
    top_probs = mean_probs[top_indices]
    top_names = [CLASS_NAMES[i] for i in top_indices]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_k))[::-1]
    bars = ax1.barh(range(top_k), top_probs, color=colors)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels(top_names)
    ax1.set_xlabel('Probability')
    ax1.set_title('Top 5 Predictions')
    ax1.set_xlim(0, 1)
    
    # Add percentage labels
    for bar, prob in zip(bars, top_probs):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center')
    
    # Right: Uncertainty breakdown
    ax2 = axes[1]
    uncertainties = [epistemic, aleatoric, epistemic + aleatoric]
    labels = ['Epistemic\n(Model)', 'Aleatoric\n(Data)', 'Total']
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    
    bars = ax2.bar(labels, uncertainties, color=colors)
    ax2.set_ylabel('Uncertainty')
    ax2.set_title('Uncertainty Decomposition')
    
    # Add value labels
    for bar, unc in zip(bars, uncertainties):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{unc:.3f}', ha='center')
    
    plt.tight_layout()
    return fig


def create_skin_tone_plot(tone_probs):
    """Create skin tone distribution visualization."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Color mapping for Fitzpatrick types
    colors = ['#FFDBAC', '#F1C27D', '#E0AC69', '#C68642', '#8D5524', '#5C3836']
    
    bars = ax.bar(range(6), tone_probs, color=colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'Type {i+1}' for i in range(6)])
    ax.set_ylabel('Probability')
    ax.set_title('Estimated Fitzpatrick Skin Type Distribution')
    ax.set_ylim(0, 1)
    
    # Highlight most likely type
    max_idx = np.argmax(tone_probs)
    bars[max_idx].set_edgecolor('red')
    bars[max_idx].set_linewidth(3)
    
    # Add legend
    ax.text(0.02, 0.98, f'Most likely: {FITZPATRICK_NAMES[max_idx]}',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


def predict(image):
    """Run prediction on uploaded image."""
    if MODEL is None:
        return None, None, "Model not loaded. Please load a model first."
    
    if image is None:
        return None, None, "Please upload an image."
    
    # Preprocess
    tensor = preprocess_image(Image.fromarray(image))
    tensor = tensor.to(DEVICE)
    
    # Predict with MC Dropout
    with torch.no_grad():
        mc_outputs = MODEL.predict_with_mc_dropout(tensor, n_samples=30)
    
    # Extract results
    mean_probs = mc_outputs['mean_probs'][0].cpu().numpy()
    epistemic = mc_outputs['epistemic_uncertainty'][0].item()
    aleatoric = mc_outputs['aleatoric_uncertainty'][0].item()
    
    # Get skin tone prediction
    outputs = MODEL(tensor, return_uncertainty=True)
    tone_probs = outputs['tone_probs'][0].cpu().numpy()
    
    # Create visualizations
    uncertainty_fig = create_uncertainty_plot(mean_probs, epistemic, aleatoric)
    skin_tone_fig = create_skin_tone_plot(tone_probs)
    
    # Generate text summary
    top_class = CLASS_NAMES[np.argmax(mean_probs)]
    top_prob = np.max(mean_probs) * 100
    total_uncertainty = epistemic + aleatoric
    
    # Determine confidence level
    if total_uncertainty < 0.3 and top_prob > 70:
        confidence = "HIGH CONFIDENCE ✓"
        confidence_color = "green"
    elif total_uncertainty < 0.5 and top_prob > 50:
        confidence = "MODERATE CONFIDENCE"
        confidence_color = "orange"
    else:
        confidence = "LOW CONFIDENCE - Recommend Expert Review ⚠️"
        confidence_color = "red"
    
    summary = f"""
## 🩺 Prediction Summary

**Primary Diagnosis:** {top_class}
**Probability:** {top_prob:.1f}%

### Confidence Assessment
**Status:** <span style="color: {confidence_color}">{confidence}</span>

- Epistemic Uncertainty: {epistemic:.3f}
- Aleatoric Uncertainty: {aleatoric:.3f}
- Total Uncertainty: {total_uncertainty:.3f}

### Skin Tone Analysis
**Estimated Fitzpatrick Type:** {FITZPATRICK_NAMES[np.argmax(tone_probs)]}

---
⚠️ **Disclaimer:** This is a research tool, not a diagnostic device. Always consult a qualified dermatologist for medical decisions.
"""
    
    return uncertainty_fig, skin_tone_fig, summary


def create_demo():
    """Create Gradio demo interface."""
    
    with gr.Blocks(
        title="DERM-EQUITY: Equitable Skin Cancer Detection",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown("""
        # 🏥 DERM-EQUITY
        ## Equitable Skin Cancer Detection via Uncertainty-Aware Multi-Scale Vision Transformers
        
        Upload a dermoscopy image to get an AI-powered analysis with:
        - **Multi-class diagnosis prediction**
        - **Uncertainty quantification** for safe clinical deployment
        - **Skin tone estimation** for fairness assessment
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Dermoscopy Image",
                    type="numpy",
                    height=300,
                )
                
                predict_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg",
                )
                
                gr.Markdown("""
                ### 📋 Instructions
                1. Upload a dermoscopy image (JPEG or PNG)
                2. Click "Analyze Image"
                3. Review the predictions and uncertainty estimates
                
                ### 🖼️ Example Images
                Try uploading images from:
                - [ISIC Archive](https://www.isic-archive.com/)
                - [DermNet NZ](https://dermnetnz.org/)
                """)
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📊 Predictions"):
                        uncertainty_plot = gr.Plot(label="Prediction Probabilities & Uncertainty")
                    
                    with gr.Tab("🎨 Skin Tone"):
                        skin_tone_plot = gr.Plot(label="Fitzpatrick Skin Type Analysis")
                
                summary_output = gr.Markdown(label="Summary")
        
        # Wire up the prediction
        predict_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[uncertainty_plot, skin_tone_plot, summary_output],
        )
        
        gr.Markdown("""
        ---
        ### 📚 About This Project
        
        DERM-EQUITY addresses the critical issue of AI bias in dermatology. Current AI systems 
        often perform worse on darker skin tones due to training data imbalances. Our approach:
        
        1. **Tone-Aware Architecture:** Conditions the model on estimated skin tone for equitable performance
        2. **Uncertainty Quantification:** Identifies when the model is unsure, enabling safe deferral
        3. **Counterfactual Fairness:** Regularization ensures predictions don't change with skin tone
        
        **📄 Paper:** [Coming Soon]  
        **💻 Code:** [GitHub Repository]  
        **🤗 Model:** [Hugging Face Hub]
        
        ---
        *Built with PyTorch, Gradio, and ❤️ for healthcare equity*
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='DERM-EQUITY Demo')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    parser.add_argument('--share', action='store_true', help='Create public link')
    args = parser.parse_args()
    
    # Load model
    load_model(args.model)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
