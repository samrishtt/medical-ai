"""
Comprehensive Evaluation Metrics for DERM-EQUITY

Includes:
- Standard classification metrics (AUC, F1, Accuracy)
- Fairness metrics (Demographic Parity, Equalized Odds)
- Calibration metrics (ECE, MCE)
- Uncertainty metrics
- Subgroup analysis
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# Core Classification Metrics
# =============================================================================

def compute_auc_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    multi_class: str = 'ovr',
) -> float:
    """Compute macro-averaged AUC-ROC."""
    try:
        return roc_auc_score(y_true, y_prob, multi_class=multi_class, average='macro')
    except ValueError:
        return 0.0


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1,
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity.
    
    Sensitivity = TP / (TP + FN) - ability to detect positives
    Specificity = TN / (TN + FP) - ability to detect negatives
    """
    # Binary case
    if len(np.unique(y_true)) == 2:
        tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
        tn = np.sum((y_true != positive_class) & (y_pred != positive_class))
        fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
        fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
        
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
    else:
        # Multi-class: compute per-class and average
        cm = confusion_matrix(y_true, y_pred)
        sensitivities = []
        specificities = []
        
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivities.append(tp / (tp + fn + 1e-8))
            specificities.append(tn / (tn + fp + 1e-8))
        
        sensitivity = np.mean(sensitivities)
        specificity = np.mean(specificities)
    
    return sensitivity, specificity


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred_or_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)
        try:
            score = metric_fn(y_true[indices], y_pred_or_prob[indices])
            bootstrap_scores.append(score)
        except ValueError:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    point_estimate = metric_fn(y_true, y_pred_or_prob)
    
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrap_scores, alpha * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)
    
    return point_estimate, ci_lower, ci_upper


# =============================================================================
# Calibration Metrics
# =============================================================================

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match true frequencies.
    Lower is better; 0 means perfectly calibrated.
    """
    # For multi-class, use predicted class probabilities
    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    
    # Compute ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error (MCE) - worst-case calibration."""
    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        predictions = (y_prob > 0.5).astype(int)
        accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_error = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            avg_confidence_in_bin = confidences[in_bin].mean()
            avg_accuracy_in_bin = accuracies[in_bin].mean()
            error = np.abs(avg_accuracy_in_bin - avg_confidence_in_bin)
            max_error = max(max_error, error)
    
    return max_error


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram.
    
    Returns:
        bin_centers, bin_accuracies, bin_counts
    """
    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        confidences = y_prob
        accuracies = (y_true == 1).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_counts.append(in_bin.sum())
    
    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


# =============================================================================
# Fairness Metrics
# =============================================================================

def demographic_parity_difference(
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """
    Demographic Parity Difference.
    
    Measures if positive prediction rates are equal across groups.
    DP = max(P(Y=1|A=a)) - min(P(Y=1|A=a)) for all groups a
    
    Lower is fairer; 0 means perfect parity.
    """
    groups = np.unique(sensitive_attr)
    positive_rates = []
    
    for group in groups:
        mask = sensitive_attr == group
        if mask.sum() > 0:
            rate = y_pred[mask].mean()
            positive_rates.append(rate)
    
    if len(positive_rates) < 2:
        return 0.0
    
    return max(positive_rates) - min(positive_rates)


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """
    Equalized Odds Difference.
    
    Measures if TPR and FPR are equal across groups.
    
    EO = max_g(|TPR_g - TPR_overall|) + max_g(|FPR_g - FPR_overall|)
    
    Lower is fairer.
    """
    groups = np.unique(sensitive_attr)
    
    # Overall TPR and FPR
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    overall_tpr = tp / (tp + fn + 1e-8)
    overall_fpr = fp / (fp + tn + 1e-8)
    
    tpr_diffs = []
    fpr_diffs = []
    
    for group in groups:
        mask = sensitive_attr == group
        if mask.sum() < 10:
            continue
        
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        tp_g = np.sum((y_true_g == 1) & (y_pred_g == 1))
        fn_g = np.sum((y_true_g == 1) & (y_pred_g == 0))
        fp_g = np.sum((y_true_g == 0) & (y_pred_g == 1))
        tn_g = np.sum((y_true_g == 0) & (y_pred_g == 0))
        
        tpr_g = tp_g / (tp_g + fn_g + 1e-8)
        fpr_g = fp_g / (fp_g + tn_g + 1e-8)
        
        tpr_diffs.append(abs(tpr_g - overall_tpr))
        fpr_diffs.append(abs(fpr_g - overall_fpr))
    
    if not tpr_diffs:
        return 0.0
    
    return max(tpr_diffs) + max(fpr_diffs)


def auc_gap_across_groups(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attr: np.ndarray,
) -> Tuple[float, Dict[Any, float]]:
    """
    Compute AUC gap across demographic groups.
    
    Returns:
        auc_gap: max(AUC) - min(AUC) across groups
        group_aucs: Dictionary of AUC per group
    """
    groups = np.unique(sensitive_attr)
    group_aucs = {}
    
    for group in groups:
        mask = sensitive_attr == group
        if mask.sum() < 10:
            continue
        
        try:
            if y_prob.ndim == 2:
                auc = roc_auc_score(
                    y_true[mask], y_prob[mask],
                    multi_class='ovr', average='macro'
                )
            else:
                auc = roc_auc_score(y_true[mask], y_prob[mask])
            group_aucs[group] = auc
        except ValueError:
            continue
    
    if len(group_aucs) < 2:
        return 0.0, group_aucs
    
    aucs = list(group_aucs.values())
    gap = max(aucs) - min(aucs)
    
    return gap, group_aucs


# =============================================================================
# Uncertainty Metrics
# =============================================================================

def uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Evaluate if uncertainty correlates with errors.
    
    High-uncertainty predictions should have higher error rates.
    """
    is_correct = (y_true == y_pred).astype(float)
    
    # Spearman correlation (uncertainty vs error)
    correlation, p_value = stats.spearmanr(uncertainty, 1 - is_correct)
    
    # Binned analysis
    bin_boundaries = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_error_rates = []
    
    for i in range(n_bins):
        in_bin = (uncertainty >= bin_boundaries[i]) & (uncertainty < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_error_rates.append(1 - is_correct[in_bin].mean())
    
    # Monotonicity score (higher uncertainty should mean higher error)
    if len(bin_error_rates) > 1:
        increases = sum(bin_error_rates[i] < bin_error_rates[i + 1] 
                       for i in range(len(bin_error_rates) - 1))
        monotonicity = increases / (len(bin_error_rates) - 1)
    else:
        monotonicity = 0.0
    
    return {
        'uncertainty_error_correlation': correlation,
        'correlation_p_value': p_value,
        'monotonicity_score': monotonicity,
    }


def selective_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    coverage_targets: List[float] = [0.7, 0.8, 0.9, 0.95],
) -> Dict[str, float]:
    """
    Compute selective prediction metrics.
    
    Evaluate accuracy when model can defer on high-uncertainty cases.
    """
    n = len(y_true)
    is_correct = (y_true == y_pred)
    
    # Sort by uncertainty (low to high)
    sorted_indices = np.argsort(uncertainty)
    
    results = {}
    
    for coverage in coverage_targets:
        n_keep = int(n * coverage)
        kept_indices = sorted_indices[:n_keep]
        
        accuracy_at_coverage = is_correct[kept_indices].mean()
        results[f'accuracy@{int(coverage*100)}%'] = accuracy_at_coverage
        
        # Risk at coverage (error rate)
        results[f'risk@{int(coverage*100)}%'] = 1 - accuracy_at_coverage
    
    # Area under risk-coverage curve (AURC)
    coverages = np.arange(1, n + 1) / n
    cumulative_correct = np.cumsum(is_correct[sorted_indices])
    risks = 1 - cumulative_correct / np.arange(1, n + 1)
    
    aurc = np.trapz(risks, coverages)
    results['aurc'] = aurc
    
    return results


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def comprehensive_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    skin_tones: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation suite.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        y_pred: Predicted labels (computed from y_prob if not provided)
        skin_tones: Fitzpatrick skin type (0-5)
        uncertainty: Model uncertainty estimates
        n_bootstrap: Number of bootstrap iterations for CIs
        
    Returns:
        Dictionary with all metrics
    """
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim == 2 else (y_prob > 0.5).astype(int)
    
    results = {}
    
    # =================================
    # 1. Overall Classification Metrics
    # =================================
    results['overall'] = {
        'auc_roc': compute_auc_roc(y_true, y_prob),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    
    sens, spec = compute_sensitivity_specificity(y_true, y_pred)
    results['overall']['sensitivity'] = sens
    results['overall']['specificity'] = spec
    
    # Bootstrap confidence intervals
    auc_point, auc_lower, auc_upper = bootstrap_confidence_interval(
        y_true, y_prob,
        lambda y, p: roc_auc_score(y, p, multi_class='ovr', average='macro'),
        n_bootstrap=n_bootstrap,
    )
    results['overall']['auc_roc_ci'] = (auc_lower, auc_upper)
    
    # =================================
    # 2. Calibration Metrics
    # =================================
    results['calibration'] = {
        'ece': expected_calibration_error(y_true, y_prob),
        'mce': maximum_calibration_error(y_true, y_prob),
    }
    
    # =================================
    # 3. Fairness Metrics (if skin tones provided)
    # =================================
    if skin_tones is not None:
        auc_gap, group_aucs = auc_gap_across_groups(y_true, y_prob, skin_tones)
        
        results['fairness'] = {
            'auc_gap': auc_gap,
            'demographic_parity_diff': demographic_parity_difference(y_pred, skin_tones),
            'equalized_odds_diff': equalized_odds_difference(y_true, y_pred, skin_tones),
            'group_aucs': group_aucs,
        }
        
        # Detailed subgroup metrics
        results['subgroups'] = {}
        for tone in range(6):
            mask = skin_tones == tone
            if mask.sum() >= 10:
                try:
                    tone_auc = compute_auc_roc(y_true[mask], y_prob[mask])
                    tone_sens, tone_spec = compute_sensitivity_specificity(
                        y_true[mask], y_pred[mask]
                    )
                    results['subgroups'][f'fitzpatrick_{tone+1}'] = {
                        'auc_roc': tone_auc,
                        'sensitivity': tone_sens,
                        'specificity': tone_spec,
                        'n_samples': int(mask.sum()),
                    }
                except ValueError:
                    pass
    
    # =================================
    # 4. Uncertainty Metrics
    # =================================
    if uncertainty is not None:
        results['uncertainty'] = uncertainty_calibration(y_true, y_pred, uncertainty)
        results['uncertainty'].update(
            selective_prediction_metrics(y_true, y_pred, uncertainty)
        )
    
    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("DERM-EQUITY EVALUATION REPORT")
    print("=" * 60)
    
    print("\n🎯 OVERALL METRICS")
    print("-" * 40)
    for metric, value in results['overall'].items():
        if isinstance(value, tuple):
            print(f"  {metric}: [{value[0]:.4f}, {value[1]:.4f}]")
        else:
            print(f"  {metric}: {value:.4f}")
    
    print("\n📊 CALIBRATION METRICS")
    print("-" * 40)
    for metric, value in results['calibration'].items():
        print(f"  {metric}: {value:.4f}")
    
    if 'fairness' in results:
        print("\n⚖️ FAIRNESS METRICS")
        print("-" * 40)
        print(f"  AUC Gap: {results['fairness']['auc_gap']:.4f}")
        print(f"  Demographic Parity Diff: {results['fairness']['demographic_parity_diff']:.4f}")
        print(f"  Equalized Odds Diff: {results['fairness']['equalized_odds_diff']:.4f}")
        
        print("\n  📈 AUC by Fitzpatrick Type:")
        for group, auc in results['fairness']['group_aucs'].items():
            print(f"    Type {group+1}: {auc:.4f}")
    
    if 'subgroups' in results:
        print("\n👥 SUBGROUP ANALYSIS")
        print("-" * 40)
        for group, metrics in results['subgroups'].items():
            print(f"\n  {group.replace('_', ' ').title()}: (n={metrics['n_samples']})")
            print(f"    AUC: {metrics['auc_roc']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
    
    if 'uncertainty' in results:
        print("\n🔮 UNCERTAINTY METRICS")
        print("-" * 40)
        for metric, value in results['uncertainty'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    n_classes = 9
    
    y_true = np.random.randint(0, n_classes, n)
    y_prob = np.random.rand(n, n_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    y_pred = np.argmax(y_prob, axis=1)
    skin_tones = np.random.randint(0, 6, n)
    uncertainty = np.random.rand(n)
    
    results = comprehensive_evaluation(
        y_true, y_prob, y_pred, skin_tones, uncertainty
    )
    
    print_evaluation_report(results)
