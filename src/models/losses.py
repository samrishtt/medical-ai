"""
Loss Functions for DERM-EQUITY

Implements:
1. Focal Loss - handles severe class imbalance in skin lesion datasets
2. Uncertainty-Aware Loss - calibrated predictions with learned variance
3. Counterfactual Fairness Loss - ensures predictions invariant to skin tone
4. Combined DermEquity Loss - weighted combination of all components
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, computed from data if None)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits (B, C)
            targets: Ground truth labels (B,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class UncertaintyAwareLoss(nn.Module):
    """
    Negative Log-Likelihood with learned variance (heteroscedastic).
    
    Encourages the model to predict higher uncertainty for difficult samples.
    
    Paper: "What Uncertainties Do We Need in Bayesian Deep Learning for 
           Computer Vision?" (Kendall & Gal, 2017)
    
    Formula: L = 0.5 * (log(σ²) + (y - μ)² / σ²)
    
    Args:
        reduction: Reduction method
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, C)
            targets: Ground truth labels (B,)
            variance: Predicted variance per class (B, C)
        """
        # Get cross-entropy loss per sample
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get variance for the target class
        target_variance = variance[torch.arange(len(targets)), targets]
        
        # NLL with learned variance
        nll = 0.5 * (torch.log(target_variance + 1e-8) + ce_loss / (target_variance + 1e-8))
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


class CounterfactualFairnessLoss(nn.Module):
    """
    Counterfactual Fairness Regularization.
    
    Ensures predictions remain similar when skin tone is hypothetically changed.
    This encourages the model to focus on lesion features rather than skin tone.
    
    Uses the approach from "Ensuring Algorithmic Fairness" by computing
    prediction variance across counterfactual skin tone variations.
    
    Formula: L_cf = E[Var_t(f(x, t))] - minimize prediction variance across tones
    
    Args:
        num_tones: Number of Fitzpatrick skin types (6)
        temperature: Temperature for softening predictions
        reduction: Reduction method
    """
    
    def __init__(
        self,
        num_tones: int = 6,
        temperature: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.num_tones = num_tones
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        logits_per_tone: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute counterfactual fairness loss from predictions under different tones.
        
        Args:
            logits_per_tone: Logits per counterfactual tone (num_tones, B, num_classes)
                            or (B, num_tones, num_classes)
                            
        Returns:
            Scalar loss penalizing variance across tone-conditioned predictions
        """
        # Ensure shape is (num_tones, B, num_classes)
        if logits_per_tone.ndim == 3 and logits_per_tone.shape[1] == self.num_tones:
            logits_per_tone = logits_per_tone.transpose(0, 1)
        
        # Softmax over class dimension with temperature
        probs_per_tone = F.softmax(logits_per_tone / self.temperature, dim=-1)
        
        # Compute variance of predictions across tones
        # (num_tones, B, num_classes) -> (B, num_classes)
        prob_mean = probs_per_tone.mean(dim=0)
        prob_var = ((probs_per_tone - prob_mean) ** 2).mean(dim=0)
        
        # Loss: penalize variance (want predictions consistent across tones)
        loss = prob_var.mean()
        
        return loss


class AdversarialDemographicParityLoss(nn.Module):
    """
    Adversarial approach to enforce demographic parity.
    
    Uses a discriminator to detect Fitzpatrick type from model predictions.
    Loss penalizes predictions that allow discriminator to distinguish skin tones.
    
    This encourages the model's predictions to be "blind" to skin tone.
    
    Approach:
    1. Train discriminator: tone ~ predictions (standard binary classification)
    2. Train model: minimize discriminator's ability to infer tone from predictions
    """
    
    def __init__(
        self,
        num_tones: int = 6,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_tones = num_tones
        
        # Simple discriminator: predicts tone from class predictions
        self.discriminator = nn.Sequential(
            nn.Linear(9, hidden_dim),  # 9 classes
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tones),
        )
    
    def forward(
        self,
        predictions: torch.Tensor,  # (B, num_classes)
        tones: torch.Tensor,  # (B,) - tone labels
        step: str = 'D',  # 'D' for discriminator, 'G' for generator
    ) -> torch.Tensor:
        """
        Compute adversarial demographic parity loss.
        
        Args:
            predictions: Model's class predictions (B, num_classes)
            tones: True Fitzpatrick tone labels (B,)
            step: Whether training discriminator ('D') or generator/model ('G')
            
        Returns:
            Loss for the respective step
        """
        # Discriminator logits: can it predict tone from predictions?
        tone_logits = self.discriminator(predictions.detach() if step == 'D' else predictions)
        
        if step == 'D':
            # Train discriminator to predict tone accurately
            loss = F.cross_entropy(tone_logits, tones)
        else:  # step == 'G'
            # Train model to fool discriminator (make tone unpredictable)
            # Maximize discriminator loss (i.e., make it confused)
            # This is equivalent to minimizing the log of discriminator accuracy
            tone_probs = F.softmax(tone_logits, dim=-1)
            # Uniform distribution over tones (discriminator fooled)
            uniform_dist = torch.ones_like(tone_probs) / self.num_tones
            # KL divergence from uniform (symmetric cross-entropy as approximation)
            loss = F.kl_div(F.log_softmax(tone_logits, dim=-1), uniform_dist, reduction='batchmean')
        
        return loss


class DermEquityLoss(nn.Module):
    """
    Combined loss function for DERM-EQUITY.
    
    Combines:
    1. Focal Loss - main classification objective
    2. Uncertainty Loss - calibrated uncertainty estimates
    3. Fairness Loss - equitable performance across skin tones
    
    Args:
        num_classes: Number of output classes
        gamma: Focal loss gamma parameter
        lambda_unc: Weight for uncertainty loss
        lambda_fair: Weight for fairness loss
        class_weights: Optional class weights for imbalance
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        gamma: float = 2.0,
        lambda_unc: float = 0.1,
        lambda_fair: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.focal_loss = FocalLoss(gamma=gamma, alpha=class_weights)
        self.uncertainty_loss = UncertaintyAwareLoss()
        self.fairness_loss = CounterfactualFairnessLossV2()
        
        self.lambda_unc = lambda_unc
        self.lambda_fair = lambda_fair
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        counterfactual_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dict with 'logits', 'variance', etc.
            targets: Ground truth labels
            counterfactual_logits: Logits for counterfactual tones (optional)
            
        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of individual loss components
        """
        logits = outputs['logits']
        
        # 1. Focal Loss (main classification)
        focal = self.focal_loss(logits, targets)
        
        loss_dict = {'focal': focal.item()}
        total = focal
        
        # 2. Uncertainty Loss (if variance is provided)
        if 'variance' in outputs:
            unc = self.uncertainty_loss(logits, targets, outputs['variance'])
            total = total + self.lambda_unc * unc
            loss_dict['uncertainty'] = unc.item()
        
        # 3. Fairness Loss (if counterfactual logits provided)
        if counterfactual_logits is not None:
            fair = self.fairness_loss(counterfactual_logits)
            total = total + self.lambda_fair * fair
            loss_dict['fairness'] = fair.item()
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing for better calibration.
    
    Prevents overconfident predictions by distributing some probability
    mass to non-target classes.
    """
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 9):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return (-true_dist * log_probs).sum(dim=-1).mean()


class AUCMaxLoss(nn.Module):
    """
    AUC Maximization Loss (for direct optimization of AUC).
    
    Useful when the primary metric is AUC-ROC rather than accuracy.
    Based on pairwise ranking formulation.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pairwise AUC loss.
        
        For each positive-negative pair, penalize if negative ranked higher.
        """
        # Get probabilities for positive class (assuming binary or using max)
        probs = F.softmax(logits, dim=-1)
        
        # For multi-class, use logit of correct class
        scores = logits[torch.arange(len(targets)), targets]
        
        # Find positive and negative indices
        # This is simplified - full implementation needs class-specific handling
        pos_mask = targets == 1  # Example for binary
        neg_mask = targets == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        
        # Pairwise loss: want pos_score > neg_score + margin
        loss = 0.0
        for pos in pos_scores:
            for neg in neg_scores:
                loss += F.relu(self.margin - (pos - neg))
        
        n_pairs = pos_mask.sum() * neg_mask.sum()
        return loss / (n_pairs + 1e-8)


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: All training labels
        num_classes: Number of classes
        
    Returns:
        Class weights tensor (normalized to sum to num_classes)
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * num_classes
    return weights


if __name__ == "__main__":
    # Test losses
    batch_size = 8
    num_classes = 9
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    variance = torch.abs(torch.randn(batch_size, num_classes)) + 0.1
    
    # Test Focal Loss
    focal = FocalLoss(gamma=2.0)
    fl = focal(logits, targets)
    print(f"Focal Loss: {fl.item():.4f}")
    
    # Test Uncertainty Loss
    unc_loss = UncertaintyAwareLoss()
    ul = unc_loss(logits, targets, variance)
    print(f"Uncertainty Loss: {ul.item():.4f}")
    
    # Test Combined Loss
    outputs = {'logits': logits, 'variance': variance}
    combined = DermEquityLoss(num_classes=num_classes)
    total, loss_dict = combined(outputs, targets)
    print(f"Combined Loss: {loss_dict}")
    
    # Test class weights computation
    all_labels = torch.randint(0, num_classes, (1000,))
    weights = compute_class_weights(all_labels, num_classes)
    print(f"Class weights: {weights}")
