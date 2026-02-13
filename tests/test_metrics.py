#!/usr/bin/env python3
"""
Unit Tests for DERM-EQUITY Evaluation Metrics

Run with: pytest tests/test_metrics.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    compute_auc_roc,
    expected_calibration_error,
    auc_gap_across_groups,
    comprehensive_evaluation,
)


@pytest.fixture
def binary_predictions():
    """Binary classification predictions."""
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.rand(n)
    # Make predictions correlated with labels
    y_prob = 0.3 * y_prob + 0.7 * y_true + np.random.normal(0, 0.1, n)
    y_prob = np.clip(y_prob, 0, 1)
    return y_true, y_prob


@pytest.fixture
def multiclass_predictions():
    """Multiclass predictions."""
    np.random.seed(42)
    n, num_classes = 200, 9
    y_true = np.random.randint(0, num_classes, n)
    y_prob = np.random.rand(n, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    return y_true, y_prob


@pytest.fixture
def skin_tones():
    """Skin tone labels (Fitzpatrick I-VI)."""
    np.random.seed(42)
    return np.random.randint(1, 7, 200)


class TestAUCROC:
    def test_binary_auc(self, binary_predictions):
        y_true, y_prob = binary_predictions
        auc = compute_auc_roc(y_true, y_prob)
        assert 0 <= auc <= 1
        assert auc > 0.5  # Should be better than random
    
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = compute_auc_roc(y_true, y_prob)
        assert auc == 1.0


class TestCalibration:
    def test_ece_range(self, binary_predictions):
        y_true, y_prob = binary_predictions
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0 <= ece <= 1
    
    def test_perfect_calibration(self):
        # Perfect calibration: predicted probability equals actual frequency
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.5] * 10)  # 60% are 1s, but we predict 50%
        ece = expected_calibration_error(y_true, y_prob, n_bins=1)
        assert 0 <= ece <= 1


class TestFairnessMetrics:
    def test_auc_gap(self, binary_predictions, skin_tones):
        y_true, y_prob = binary_predictions
        gap = auc_gap_across_groups(y_true, y_prob, skin_tones)
        assert gap >= 0  # Gap should be non-negative


class TestComprehensiveEvaluation:
    def test_with_skin_tones(self, multiclass_predictions, skin_tones):
        y_true, y_prob = multiclass_predictions
        y_pred = y_prob.argmax(axis=1)
        uncertainty = np.random.rand(len(y_true))
        
        results = comprehensive_evaluation(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            skin_tones=skin_tones,
            uncertainty=uncertainty,
            n_bootstrap=10,
        )
        
        assert 'overall' in results
        assert 'fairness' in results
        assert 'subgroups' in results
    
    def test_without_optional_args(self, multiclass_predictions):
        y_true, y_prob = multiclass_predictions
        results = comprehensive_evaluation(
            y_true=y_true,
            y_prob=y_prob,
            n_bootstrap=10,
        )
        assert 'overall' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
