#!/usr/bin/env python3
"""
DERM-EQUITY Training Script

Usage:
    python scripts/train.py --config configs/train_config.yaml
    
    # Override config values:
    python scripts/train.py --config configs/train_config.yaml \
        training.batch_size=64 \
        training.epochs=50

Author: [Your Name]
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tam_vit import create_tam_vit_base
from src.models.losses import compute_class_weights
from src.data.datasets import (
    ISIC2020Dataset,
    Fitzpatrick17kDataset,
    get_train_transforms,
    get_val_transforms,
    get_val_transforms,
    create_dataloaders,
    MILK10kDataset,
)
from src.training.trainer import DermEquityModule, create_callbacks, create_loggers


def parse_args():
    parser = argparse.ArgumentParser(description='Train DERM-EQUITY model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (small batches, no logging)'
    )
    
    # Allow config overrides via command line
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in key=value format'
    )
    
    return parser.parse_args()


def setup_environment(seed: int):
    """Set up reproducibility and environment."""
    pl.seed_everything(seed, workers=True)
    
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    
    # Set float32 matmul precision for better performance
    torch.set_float32_matmul_precision('high')
    
    print(f"\n🔧 Environment Setup:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_data(config):
    """Load datasets and create dataloaders."""
    print("\n📂 Loading datasets...")
    
    data_config = config.data
    
    # Get transforms
    train_transform = get_train_transforms(config.model.img_size)
    val_transform = get_val_transforms(config.model.img_size)
    
    # Check if data exists
    train_dir = Path(data_config.train_data_dir)
    if not train_dir.exists():
        print(f"\n⚠️  Training data not found at {train_dir}")
        print("Please download the ISIC 2020 dataset:")
        print("  1. Visit https://www.isic-archive.com/")
        print("  2. Download the ISIC 2020 Challenge data")
        print("  3. Extract to ./data/isic2020/")
        print("\nAlternatively, run: python scripts/download_data.py")
        
        # Create dummy data for testing
        print("\n🧪 Creating dummy data for testing...")
        return create_dummy_dataloaders(config)
    
    # Load datasets
    train_csv = Path(data_config.train_data_dir).parent / 'train.csv'
    val_csv = Path(data_config.val_data_dir).parent / 'val.csv'
    
    if data_config.dataset == 'milk10k':
        print(f"   Using MILK10k Dataset (Dual-Image Input)")
        train_dataset = MILK10kDataset(
            root_dir=data_config.train_data_dir,
            csv_file=str(train_csv),
            transform=train_transform,
            phase='train',
        )
        val_dataset = MILK10kDataset(
            root_dir=data_config.val_data_dir,
            csv_file=str(val_csv),
            transform=val_transform,
            phase='val',
        )
    else:
        # Default to ISIC 2020
        train_dataset = ISIC2020Dataset(
            root_dir=data_config.train_data_dir,
            csv_file=str(train_csv),
            transform=train_transform,
            phase='train',
            return_metadata=True,
        )
        
        val_dataset = ISIC2020Dataset(
            root_dir=data_config.val_data_dir,
            csv_file=str(val_csv),
            transform=val_transform,
            phase='val',
            return_metadata=True,
        )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        use_weighted_sampler=True,
    )
    
    # Compute class weights
    labels = torch.tensor([train_dataset[i]['label'] for i in range(len(train_dataset))])
    class_weights = compute_class_weights(labels, len(data_config.classes))
    
    return dataloaders, class_weights


def create_dummy_dataloaders(config):
    """Create dummy dataloaders for testing without real data."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # Dummy data
    n_train, n_val = 320, 64
    img_size = config.model.img_size
    num_classes = config.model.num_classes
    
    train_images = torch.randn(n_train, 3, img_size, img_size)
    train_labels = torch.randint(0, num_classes, (n_train,))
    train_tones = torch.randint(0, 6, (n_train,))
    
    val_images = torch.randn(n_val, 3, img_size, img_size)
    val_labels = torch.randint(0, num_classes, (n_val,))
    val_tones = torch.randint(0, 6, (n_val,))
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels, tones):
            self.images = images
            self.labels = labels
            self.tones = tones
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'image': self.images[idx],
                'label': self.labels[idx],
                'fitzpatrick': self.tones[idx],
            }
    
    train_dataset = DummyDataset(train_images, train_labels, train_tones)
    val_dataset = DummyDataset(val_images, val_labels, val_tones)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=False),
    }
    
    class_weights = torch.ones(num_classes)
    
    return dataloaders, class_weights


def train(config, dataloaders, class_weights, resume_from=None, debug=False):
    """Main training function."""
    print("\n🚀 Starting training...")
    
    # Model configuration
    model_config = OmegaConf.to_container(config.model, resolve=True)
    train_config = OmegaConf.to_container(config.training, resolve=True)
    train_config.update(OmegaConf.to_container(config.optimizer, resolve=True))
    train_config.update(OmegaConf.to_container(config.scheduler, resolve=True))
    train_config.update(OmegaConf.to_container(config.loss, resolve=True))
    train_config['use_wandb'] = not debug and config.logging.wandb.enabled
    train_config['wandb_project'] = config.logging.wandb.project
    
    # Create model
    model = DermEquityModule(
        model_config=model_config,
        train_config=train_config,
        class_weights=class_weights,
    )
    
    # Callbacks
    callbacks = create_callbacks(train_config)
    
    # Loggers
    if debug:
        loggers = []
    else:
        loggers = create_loggers(train_config)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        log_every_n_steps=10 if debug else config.logging.log_every_n_steps,
        val_check_interval=1.0,
        enable_checkpointing=not debug,
        enable_progress_bar=True,
        fast_dev_run=5 if debug else False,
    )
    
    # Train
    trainer.fit(
        model,
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val'],
        ckpt_path=resume_from,
    )
    
    print("\n✅ Training complete!")
    print(f"   Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"   Best validation AUC: {trainer.checkpoint_callback.best_model_score:.4f}")
    
    return model, trainer


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🏥 DERM-EQUITY: Equitable Skin Cancer Detection")
    print("=" * 60)
    
    # Load and merge config
    config = OmegaConf.load(args.config)
    
    # Apply command-line overrides
    if args.overrides:
        override_config = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, override_config)
    
    # Setup
    setup_environment(args.seed)
    
    # Create output directories
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = Path(config.paths.output_dir) / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    OmegaConf.save(config, config_save_path)
    print(f"\n📋 Config saved to: {config_save_path}")
    
    # Load data
    dataloaders, class_weights = load_data(config)
    
    # Train
    model, trainer = train(
        config,
        dataloaders,
        class_weights,
        resume_from=args.resume,
        debug=args.debug,
    )
    
    # Final evaluation on validation set
    print("\n📊 Final Evaluation...")
    trainer.validate(model, dataloaders=dataloaders['val'])


if __name__ == "__main__":
    main()
