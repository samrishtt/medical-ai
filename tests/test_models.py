#!/usr/bin/env python3
"""
Unit Tests for DERM-EQUITY Models
Run with: pytest tests/test_models.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly from modules to avoid data package import
from models.tam_vit import (
    TAMViT, create_tam_vit_base, create_tam_vit_small,
    SkinToneEstimator, ToneAdaptiveLayerNorm,
)
from models.losses import (
    FocalLoss, UncertaintyAwareLoss, DermEquityLoss, compute_class_weights,
)


@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def img_size():
    return 224

@pytest.fixture
def num_classes():
    return 9

@pytest.fixture
def sample_batch(batch_size, img_size):
    return torch.randn(batch_size, 3, img_size, img_size)

@pytest.fixture
def sample_labels(batch_size, num_classes):
    return torch.randint(0, num_classes, (batch_size,))


class TestTAMViT:
    def test_create_model(self, num_classes):
        model = create_tam_vit_small(num_classes=num_classes)
        assert model is not None
        assert isinstance(model, TAMViT)
    
    def test_forward_pass(self, sample_batch, num_classes):
        model = create_tam_vit_small(num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            outputs = model(sample_batch, return_uncertainty=True)
        
        assert 'logits' in outputs
        assert 'probs' in outputs
        assert outputs['logits'].shape == (sample_batch.size(0), num_classes)
    
    def test_mc_dropout(self, sample_batch, num_classes):
        model = create_tam_vit_small(num_classes=num_classes)
        with torch.no_grad():
            outputs = model.predict_with_mc_dropout(sample_batch, n_samples=3)
        
        assert 'mean_probs' in outputs
        assert 'epistemic_uncertainty' in outputs
    
    def test_gradient_flow(self, sample_batch, sample_labels, num_classes):
        model = create_tam_vit_small(num_classes=num_classes)
        model.train()
        outputs = model(sample_batch)
        loss = nn.CrossEntropyLoss()(outputs['logits'], sample_labels)
        loss.backward()
        
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


class TestSkinToneEstimator:
    def test_output_shape(self, sample_batch):
        estimator = SkinToneEstimator(num_tones=6, hidden_dim=256)
        with torch.no_grad():
            tone_probs, tone_embed = estimator(sample_batch)
        
        assert tone_probs.shape == (sample_batch.size(0), 6)
        assert tone_embed.shape == (sample_batch.size(0), 256)
    
    def test_probability_sum(self, sample_batch):
        estimator = SkinToneEstimator(num_tones=6, hidden_dim=256)
        with torch.no_grad():
            tone_probs, _ = estimator(sample_batch)
        
        assert torch.allclose(tone_probs.sum(dim=-1), torch.ones(sample_batch.size(0)), atol=1e-5)


class TestLossFunctions:
    def test_focal_loss(self, sample_labels, num_classes):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(len(sample_labels), num_classes)
        loss = loss_fn(logits, sample_labels)
        assert loss.dim() == 0 and not torch.isnan(loss)
    
    def test_uncertainty_loss(self, sample_labels, num_classes):
        loss_fn = UncertaintyAwareLoss()
        batch_size = len(sample_labels)
        logits = torch.randn(batch_size, num_classes)
        variance = torch.rand(batch_size, num_classes).abs() + 0.1
        loss = loss_fn(logits, sample_labels, variance)
        assert loss.dim() == 0 and not torch.isnan(loss)
    
    def test_derm_equity_loss(self, sample_labels, num_classes):
        loss_fn = DermEquityLoss(num_classes=num_classes)
        batch_size = len(sample_labels)
        outputs = {
            'logits': torch.randn(batch_size, num_classes),
            'variance': torch.rand(batch_size, num_classes).abs() + 0.1,
        }
        total_loss, loss_dict = loss_fn(outputs, sample_labels, None)
        assert total_loss.dim() == 0
        assert 'focal' in loss_dict


class TestClassWeights:
    def test_compute_weights(self, num_classes):
        labels = torch.tensor([0, 0, 0, 0, 1, 2, 2, 2, 2, 2])
        weights = compute_class_weights(labels, num_classes)
        assert weights.shape == (num_classes,)
        assert torch.all(weights > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
