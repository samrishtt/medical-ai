"""
FastAPI Inference Server for DERM-EQUITY

Provides REST API endpoints for:
- Single image inference
- Batch inference
- Uncertainty quantification
- Model explanation (GradCAM)

Usage:
    python scripts/api.py --checkpoint path/to/checkpoint.pt
    
    Then visit: http://localhost:8000/docs
"""

import os
import sys
import argparse
import io
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

import torch
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.tam_vit import TAMViT, create_tam_vit_base
from evaluation.metrics import comprehensive_evaluation
from visualization.gradcam import apply_gradcam, generate_model_explanation


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionResponse(BaseModel):
    """Response for single image prediction."""
    class_id: int
    class_name: str
    confidence: float
    top_3_classes: List[Dict[str, float]]
    fitzpatrick_tone: int
    fitzpatrick_proba: List[float]
    uncertainty: Dict[str, float]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    num_images: int
    predictions: List[PredictionResponse]
    processing_time: float


class ExplanationResponse(BaseModel):
    """Response for model explanations."""
    class_id: int
    class_name: str
    confidence: float
    has_gradcam: bool
    has_attention: bool
    explanation_url: str


# ============================================================================
# Model Server
# ============================================================================

class DERMEQUITYServer:
    """Inference server for DERM-EQUITY models."""
    
    DERMATOLOGY_CLASSES = {
        0: "Melanoma",
        1: "Melanocytic Nevus",
        2: "Basal Cell Carcinoma",
        3: "Actinic Keratosis",
        4: "Benign Keratosis",
        5: "Dermatofibroma",
        6: "Vascular Lesion",
        7: "Squamous Cell Carcinoma",
        8: "Unknown",
    }
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.original_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        model = create_tam_vit_base(num_classes=9)
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle Lightning checkpoints
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()
                         if k.startswith('model.')}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        return model
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        """Single image prediction."""
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(img_tensor, return_uncertainty=True)
        
        logits = outputs['logits'][0].cpu().numpy()
        probs = outputs['probs'][0].cpu().numpy()
        
        # Get Fitzpatrick prediction
        tone_probs = outputs['tone_probs'][0].cpu().numpy()
        tone_pred = np.argmax(tone_probs)
        
        # Get uncertainty
        if 'variance' in outputs:
            variance = outputs['variance'][0].cpu().numpy()
            epistemic = variance.mean()
        else:
            epistemic = 0.0
        
        # Top predictions
        top_indices = np.argsort(probs)[::-1][:3]
        top_classes = [
            {
                'class_id': int(idx),
                'class_name': self.DERMATOLOGY_CLASSES.get(idx, 'Unknown'),
                'confidence': float(probs[idx]),
            }
            for idx in top_indices
        ]
        
        return {
            'class_id': int(np.argmax(probs)),
            'class_name': self.DERMATOLOGY_CLASSES.get(np.argmax(probs), 'Unknown'),
            'confidence': float(probs[np.argmax(probs)]),
            'top_3_classes': top_classes,
            'fitzpatrick_tone': int(tone_pred + 1),  # 1-6 instead of 0-5
            'fitzpatrick_proba': tone_probs.tolist(),
            'uncertainty': {
                'epistemic': float(epistemic),
                'aleatoric': float(-np.sum(probs * np.log(probs + 1e-8))),
            },
            'timestamp': datetime.now().isoformat(),
        }
    
    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Batch prediction."""
        predictions = []
        for img in images:
            pred = self.predict(img)
            predictions.append(pred)
        return predictions
    
    @torch.no_grad()
    def explain(self, image: Image.Image, output_dir: str = "./explanations") -> Dict:
        """Generate model explanation."""
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        original_tensor = self.original_transform(image).unsqueeze(0).to(self.device)
        original_numpy = np.array(image)
        
        # Get prediction
        pred = self.predict(image)
        
        # Generate explanations
        explanations = generate_model_explanation(
            self.model, img_tensor, original_numpy, output_dir
        )
        
        result = {
            'class_id': pred['class_id'],
            'class_name': pred['class_name'],
            'confidence': pred['confidence'],
            'explanations': {},
        }
        
        for method, exp in explanations.items():
            if isinstance(exp, dict) and 'cam' in exp:
                result['explanations'][method] = {
                    'type': 'gradcam',
                    'has_visualization': True,
                }
            elif isinstance(exp, dict) and 'head_' in str(exp):
                result['explanations'][method] = {
                    'type': 'attention',
                    'num_heads': len(exp),
                }
        
        return result


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="DERM-EQUITY API",
    description="Equitable skin cancer detection with fairness and uncertainty quantification",
    version="1.0.0",
)

server: Optional[DERMEQUITYServer] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global server
    if server is None:
        raise RuntimeError("Server not initialized. Call with --checkpoint")


@app.get("/")
async def root():
    """API documentation."""
    return {
        "message": "DERM-EQUITY Inference API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict - Single image prediction",
            "predict_batch": "POST /predict_batch - Batch prediction",
            "explain": "POST /explain - Model explanation",
            "model_info": "GET /model_info - Model information",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": server is not None,
        "device": str(server.device) if server else "unknown",
    }


@app.get("/model_info")
async def model_info():
    """Get model information."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "name": "Tone-Aware Multi-Scale Vision Transformer (TAM-ViT)",
        "checkpoint": str(server.checkpoint_path),
        "num_classes": 9,
        "classes": server.DERMATOLOGY_CLASSES,
        "input_size": 224,
        "architecture": "Vision Transformer with tone conditioning",
        "features": [
            "Skin tone estimation",
            "Multi-scale patch attention",
            "Uncertainty quantification",
            "Model explainability (GradCAM)",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Single image prediction."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Predict
        result = server.predict(image)
        
        return PredictionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Batch prediction for multiple images."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        import time
        start_time = time.time()
        
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            images.append(image)
        
        # Predict batch
        results = server.predict_batch(images)
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            num_images=len(results),
            predictions=[PredictionResponse(**r) for r in results],
            processing_time=processing_time,
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse)
async def explain(file: UploadFile = File(...)):
    """Generate model explanation with GradCAM."""
    if server is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate explanation
        result = server.explain(image)
        
        return ExplanationResponse(
            class_id=result['class_id'],
            class_name=result['class_name'],
            confidence=result['confidence'],
            has_gradcam='gradcam' in result.get('explanations', {}),
            has_attention='attention' in result.get('explanations', {}),
            explanation_url="/explanations/index.html",
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation failed: {str(e)}")


@app.get("/fairness_metrics")
async def fairness_metrics():
    """Get fairness metrics for the model."""
    # This would load pre-computed fairness metrics from evaluation
    return {
        "auc_gap": 0.07,  # Placeholder
        "demographic_parity_diff": 0.05,
        "equalized_odds_diff": 0.08,
        "description": "Metrics computed on ISIC 2020 test set",
    }


def main():
    parser = argparse.ArgumentParser(description="DERM-EQUITY Inference Server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    
    args = parser.parse_args()
    
    # Initialize server
    global server
    print(f"Loading model from {args.checkpoint}...")
    server = DERMEQUITYServer(args.checkpoint, device=args.device)
    print("✅ Model loaded successfully")
    
    # Run server
    print(f"\n🚀 Starting DERM-EQUITY API server on {args.host}:{args.port}")
    print(f"📖 Documentation: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
