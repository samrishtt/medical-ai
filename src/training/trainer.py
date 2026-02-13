"""
Training Pipeline for DERM-EQUITY

PyTorch Lightning-based training with:
- Mixed precision training
- Gradient accumulation
- Cosine annealing with warm restarts
- Comprehensive logging (W&B, TensorBoard)
- Model checkpointing
- Early stopping
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.tam_vit import TAMViT, create_tam_vit_base
from models.losses import DermEquityLoss, compute_class_weights


class DermEquityModule(pl.LightningModule):
    """
    PyTorch Lightning module for DERM-EQUITY training.
    
    Handles:
    - Forward pass with tone conditioning
    - Loss computation with uncertainty and fairness
    - Optimizer and scheduler configuration
    - Validation metrics computation
    - MC Dropout inference for uncertainty
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        train_config: Dict[str, Any],
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build model
        self.model = TAMViT(
            img_size=model_config.get('img_size', 224),
            patch_sizes=model_config.get('patch_sizes', [16, 8]),
            num_classes=model_config.get('num_classes', 9),
            embed_dim=model_config.get('embed_dim', 768),
            depth=model_config.get('depth', 12),
            num_heads=model_config.get('num_heads', 12),
            mlp_ratio=model_config.get('mlp_ratio', 4.0),
            drop_rate=model_config.get('dropout', 0.1),
            drop_path_rate=model_config.get('drop_path', 0.1),
        )
        
        # Loss function
        self.criterion = DermEquityLoss(
            num_classes=model_config.get('num_classes', 9),
            gamma=train_config.get('focal_gamma', 2.0),
            lambda_unc=train_config.get('lambda_unc', 0.1),
            lambda_fair=train_config.get('lambda_fair', 0.5),
            class_weights=class_weights,
        )
        
        # Training config
        self.train_config = train_config
        self.model_config = model_config
        
        # Validation predictions storage
        self.val_preds = []
        self.val_labels = []
        self.val_tones = []
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x, return_uncertainty=True)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self.model(images, return_uncertainty=True)
        
        # Compute loss
        loss, loss_dict = self.criterion(outputs, labels)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, on_step=False, on_epoch=True)
        
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        images = batch['image']
        labels = batch['label']
        fitzpatrick = batch.get('fitzpatrick', torch.zeros_like(labels) - 1)
        
        # Forward pass
        outputs = self.model(images, return_uncertainty=True)
        
        # Compute loss
        loss, loss_dict = self.criterion(outputs, labels)
        
        # Log loss
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions for epoch-end metrics
        probs = outputs['probs'].detach().cpu()
        self.val_preds.append(probs)
        self.val_labels.append(labels.cpu())
        self.val_tones.append(fitzpatrick.cpu())
    
    def on_validation_epoch_end(self) -> None:
        # Aggregate predictions
        all_probs = torch.cat(self.val_preds, dim=0).numpy()
        all_labels = torch.cat(self.val_labels, dim=0).numpy()
        all_tones = torch.cat(self.val_tones, dim=0).numpy()
        
        all_preds = np.argmax(all_probs, axis=1)
        
        # Overall metrics
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        
        self.log('val/auc_roc', auc, prog_bar=True)
        self.log('val/f1_macro', f1, prog_bar=True)
        self.log('val/accuracy', acc)
        
        # Subgroup metrics by Fitzpatrick type
        for tone in range(6):  # 0-5 (Fitzpatrick I-VI)
            mask = all_tones == tone
            if mask.sum() > 10:  # Minimum samples for meaningful metric
                try:
                    tone_auc = roc_auc_score(
                        all_labels[mask], all_probs[mask],
                        multi_class='ovr', average='macro'
                    )
                    self.log(f'val/auc_fitz_{tone+1}', tone_auc)
                except ValueError:
                    pass
        
        # Log fairness gap (max - min AUC across tones)
        tone_aucs = []
        for tone in range(6):
            mask = all_tones == tone
            if mask.sum() > 10:
                try:
                    tone_auc = roc_auc_score(
                        all_labels[mask], all_probs[mask],
                        multi_class='ovr', average='macro'
                    )
                    tone_aucs.append(tone_auc)
                except ValueError:
                    pass
        
        if len(tone_aucs) >= 2:
            fairness_gap = max(tone_aucs) - min(tone_aucs)
            self.log('val/fairness_gap', fairness_gap, prog_bar=True)
        
        # Clear stored predictions
        self.val_preds.clear()
        self.val_labels.clear()
        self.val_tones.clear()
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # Same as validation but with MC Dropout
        images = batch['image']
        labels = batch['label']
        
        # MC Dropout for uncertainty
        mc_outputs = self.model.predict_with_mc_dropout(images, n_samples=30)
        
        # Store for epoch-end processing
        self.val_preds.append(mc_outputs['mean_probs'].cpu())
        self.val_labels.append(labels.cpu())
        
        # Log uncertainty statistics
        epistemic = mc_outputs['epistemic_uncertainty'].mean()
        aleatoric = mc_outputs['aleatoric_uncertainty'].mean()
        
        self.log('test/epistemic_uncertainty', epistemic)
        self.log('test/aleatoric_uncertainty', aleatoric)
    
    def configure_optimizers(self):
        # Get parameters with layer-wise learning rate decay
        parameters = self._get_parameter_groups()
        
        optimizer = AdamW(
            parameters,
            lr=self.train_config.get('lr', 1e-4),
            weight_decay=self.train_config.get('weight_decay', 0.05),
            betas=tuple(self.train_config.get('betas', [0.9, 0.999])),
        )
        
        # Warmup + Cosine Annealing
        warmup_epochs = self.train_config.get('warmup_epochs', 5)
        total_epochs = self.train_config.get('epochs', 100)
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.train_config.get('T_0', 10),
            T_mult=self.train_config.get('T_mult', 2),
            eta_min=self.train_config.get('eta_min', 1e-6),
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }
    
    def _get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Create parameter groups with layer-wise learning rate decay (LLRD).
        
        Earlier layers get smaller learning rates since they capture
        more general features that don't need as much fine-tuning.
        """
        lr = self.train_config.get('lr', 1e-4)
        decay = self.train_config.get('layer_decay', 0.75)
        
        # Get all named parameters
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'norm']
        
        # Group by layer depth
        param_groups = []
        
        # Skin tone estimator (learns faster)
        tone_params = [
            p for n, p in self.model.named_parameters()
            if 'tone_estimator' in n or 'tone_embed' in n
        ]
        if tone_params:
            param_groups.append({
                'params': tone_params,
                'lr': lr * 2,  # Higher LR for tone-specific parts
                'weight_decay': self.train_config.get('weight_decay', 0.05),
            })
        
        # Transformer blocks (layer-wise decay)
        num_layers = self.model_config.get('depth', 12)
        for i in range(num_layers):
            layer_lr = lr * (decay ** (num_layers - i - 1))
            
            layer_params_decay = [
                p for n, p in self.model.named_parameters()
                if f'blocks.{i}.' in n and not any(nd in n for nd in no_decay)
            ]
            layer_params_no_decay = [
                p for n, p in self.model.named_parameters()
                if f'blocks.{i}.' in n and any(nd in n for nd in no_decay)
            ]
            
            if layer_params_decay:
                param_groups.append({
                    'params': layer_params_decay,
                    'lr': layer_lr,
                    'weight_decay': self.train_config.get('weight_decay', 0.05),
                })
            if layer_params_no_decay:
                param_groups.append({
                    'params': layer_params_no_decay,
                    'lr': layer_lr,
                    'weight_decay': 0.0,
                })
        
        # Classification head (full learning rate)
        head_params = [
            p for n, p in self.model.named_parameters()
            if 'cls_head' in n or 'uncertainty_head' in n
        ]
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': lr,
                'weight_decay': self.train_config.get('weight_decay', 0.05),
            })
        
        # Remaining parameters
        assigned = set()
        for group in param_groups:
            for p in group['params']:
                assigned.add(id(p))
        
        remaining = [p for p in self.model.parameters() if id(p) not in assigned]
        if remaining:
            param_groups.append({
                'params': remaining,
                'lr': lr,
                'weight_decay': self.train_config.get('weight_decay', 0.05),
            })
        
        return param_groups


def create_callbacks(config: Dict[str, Any]) -> List[pl.Callback]:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('checkpoint_dir', './checkpoints'),
        filename='derm-equity-{epoch:02d}-{val/auc_roc:.4f}',
        monitor='val/auc_roc',
        mode='max',
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val/auc_roc',
        patience=config.get('patience', 15),
        mode='max',
        min_delta=0.001,
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    return callbacks


def create_loggers(config: Dict[str, Any]) -> List[pl.loggers.Logger]:
    """Create experiment loggers."""
    loggers = []
    
    # Weights & Biases
    if config.get('use_wandb', True):
        wandb_logger = WandbLogger(
            project=config.get('wandb_project', 'derm-equity'),
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_dir=config.get('log_dir', './logs'),
        )
        loggers.append(wandb_logger)
    
    # TensorBoard
    tb_logger = TensorBoardLogger(
        save_dir=config.get('log_dir', './logs'),
        name='tensorboard',
    )
    loggers.append(tb_logger)
    
    return loggers


def train(
    train_dataloader,
    val_dataloader,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
) -> DermEquityModule:
    """
    Main training function.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        model_config: Model configuration dict
        train_config: Training configuration dict
        class_weights: Optional class weights for imbalance
        
    Returns:
        Trained model module
    """
    # Create model
    model = DermEquityModule(model_config, train_config, class_weights)
    
    # Create callbacks and loggers
    callbacks = create_callbacks(train_config)
    loggers = create_loggers(train_config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=train_config.get('epochs', 100),
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if train_config.get('use_fp16', True) else 32,
        accumulate_grad_batches=train_config.get('accumulate_grad_batches', 2),
        gradient_clip_val=train_config.get('gradient_clip_val', 1.0),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=50,
        val_check_interval=1.0,
        deterministic=False,
    )
    
    # Train
    trainer.fit(model, train_dataloader, val_dataloader)
    
    return model


if __name__ == "__main__":
    print("DERM-EQUITY Training Module")
    print("Run scripts/train.py to start training")
