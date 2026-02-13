"""
Dataset classes for DERM-EQUITY

Provides unified interface for:
- ISIC 2020 (primary dataset)
- Fitzpatrick17k (external validation with skin tone labels)
- DDI (Diverse Dermatology Images)
- PAD-UFES-20
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Callable, Any
import json

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# ISIC 2020 Dataset
# =============================================================================
class ISIC2020Dataset(Dataset):
    """
    ISIC 2020 Challenge Dataset for skin lesion classification.
    
    Classes:
        0: melanoma
        1: melanocytic nevus  
        2: basal cell carcinoma
        3: actinic keratosis
        4: benign keratosis
        5: dermatofibroma
        6: vascular lesion
        7: squamous cell carcinoma
        8: unknown
    
    Args:
        root_dir: Path to dataset root
        csv_file: Path to metadata CSV
        transform: Albumentations transform
        phase: 'train', 'val', or 'test'
        return_metadata: Whether to return additional metadata
    """
    
    CLASS_NAMES = [
        'melanoma', 'melanocytic_nevus', 'basal_cell_carcinoma',
        'actinic_keratosis', 'benign_keratosis', 'dermatofibroma',
        'vascular_lesion', 'squamous_cell_carcinoma', 'unknown'
    ]
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[A.Compose] = None,
        phase: str = 'train',
        return_metadata: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.phase = phase
        self.return_metadata = return_metadata
        
        # Load metadata
        self.df = pd.read_csv(csv_file)
        
        # Filter by phase if split column exists
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == phase].reset_index(drop=True)
        
        # Create label mapping
        self.label_map = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        
        # Estimate skin tone if not provided (placeholder)
        if 'fitzpatrick' not in self.df.columns:
            self.df['fitzpatrick'] = -1  # Unknown
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.root_dir / f"{row['image_name']}.jpg"
        if not img_path.exists():
            img_path = self.root_dir / f"{row['image_name']}.png"
        
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        diagnosis = row.get('diagnosis', row.get('dx', 'unknown'))
        if diagnosis in self.label_map:
            label = self.label_map[diagnosis]
        elif 'target' in row:
            label = int(row['target'])
        else:
            label = 8  # unknown
        
        result = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_name': row['image_name'],
        }
        
        if self.return_metadata:
            result['fitzpatrick'] = row.get('fitzpatrick', -1)
            result['age'] = row.get('age_approx', -1)
            result['sex'] = row.get('sex', 'unknown')
            result['location'] = row.get('anatom_site_general_challenge', 'unknown')
        
        return result


# =============================================================================
# Fitzpatrick17k Dataset
# =============================================================================
class Fitzpatrick17kDataset(Dataset):
    """
    Fitzpatrick17k Dataset with ground truth skin tone labels.
    
    Critical for evaluating fairness across Fitzpatrick types I-VI.
    
    Args:
        root_dir: Path to dataset root
        csv_file: Path to annotations CSV
        transform: Albumentations transform
        fitzpatrick_types: List of Fitzpatrick types to include (1-6)
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[A.Compose] = None,
        fitzpatrick_types: Optional[List[int]] = None,
        label_column: str = 'three_partition_label',
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_column = label_column
        
        # Load metadata
        self.df = pd.read_csv(csv_file)
        
        # Filter by Fitzpatrick types if specified
        if fitzpatrick_types:
            self.df = self.df[
                self.df['fitzpatrick'].isin(fitzpatrick_types)
            ].reset_index(drop=True)
        
        # Create label mapping
        self.classes = sorted(self.df[label_column].unique())
        self.label_map = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.root_dir / row['md5hash']
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            # Return a placeholder if image fails to load
            print(f"Warning: Failed to load {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label and Fitzpatrick type
        label = self.label_map[row[self.label_column]]
        fitzpatrick = int(row['fitzpatrick'])
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'fitzpatrick': torch.tensor(fitzpatrick - 1, dtype=torch.long),  # 0-indexed
            'image_id': row['md5hash'],
        }


# =============================================================================
# DDI (Diverse Dermatology Images) Dataset
# =============================================================================
class DDIDataset(Dataset):
    """
    DDI Dataset - specifically curated for diverse skin tones.
    
    Created by Stanford for evaluating dermatology AI fairness.
    Contains 656 images with clinical labels and Fitzpatrick annotations.
    """
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[A.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.df = pd.read_csv(csv_file)
        
        # Map diagnoses to binary (malignant vs benign)
        self.df['is_malignant'] = self.df['malignant'].astype(int)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        img_path = self.root_dir / row['DDI_file']
        image = np.array(Image.open(img_path).convert('RGB'))
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(row['is_malignant'], dtype=torch.long),
            'fitzpatrick': torch.tensor(row['skin_tone'] - 1, dtype=torch.long),
            'diagnosis': row['diagnosis'],
        }


# =============================================================================
# Transforms
# =============================================================================
def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Get training augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.GaussianBlur(blur_limit=(3, 5)),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Get validation/test augmentation pipeline."""
    return A.Compose([
        A.Resize(int(img_size * 1.14), int(img_size * 1.14)),  # 256 for 224
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size: int = 224) -> List[A.Compose]:
    """
    Get Test-Time Augmentation transforms.
    
    Returns multiple transforms for TTA during inference.
    """
    base_transform = A.Compose([
        A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transforms = [base_transform]
    
    # Add flipped versions
    for flip in [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]:
        transforms.append(A.Compose([
            A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
            A.CenterCrop(img_size, img_size),
            flip,
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]))
    
    return transforms


# =============================================================================
# Data Loading Utilities
# =============================================================================
def create_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler:
    """
    Create weighted sampler for class-imbalanced dataset.
    
    Ensures each batch has roughly balanced class representation.
    """
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(item['label'].item())
    
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    
    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    return sampler


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 8,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """Create train, validation, and test dataloaders."""
    
    train_sampler = None
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_dataset)
    
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
        ),
    }
    
    if test_dataset is not None:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders


# =============================================================================
# Synthetic Data Generation (for rare conditions/skin tones)
# =============================================================================
class SyntheticAugmentedDataset(Dataset):
    """
    Wrapper that adds synthetic samples for underrepresented groups.
    
    Uses style transfer / color augmentation to generate samples
    that simulate different skin tones while preserving lesion features.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        augmentation_factor: Dict[int, float],  # {fitzpatrick_type: factor}
        color_transfer_fn: Optional[Callable] = None,
    ):
        self.base_dataset = base_dataset
        self.augmentation_factor = augmentation_factor
        self.color_transfer_fn = color_transfer_fn
        
        # Build index with synthetic samples
        self._build_index()
    
    def _build_index(self):
        """Create index mapping including synthetic samples."""
        self.index = []
        
        for i in range(len(self.base_dataset)):
            item = self.base_dataset[i]
            fitz = item.get('fitzpatrick', torch.tensor(-1)).item()
            
            # Original sample
            self.index.append((i, False, fitz))
            
            # Add synthetic copies for underrepresented groups
            if fitz in self.augmentation_factor:
                n_copies = int(self.augmentation_factor[fitz])
                for _ in range(n_copies):
                    self.index.append((i, True, fitz))
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx, is_synthetic, fitz = self.index[idx]
        item = self.base_dataset[base_idx]
        
        if is_synthetic and self.color_transfer_fn is not None:
            # Apply synthetic augmentation
            item['image'] = self.color_transfer_fn(item['image'])
            item['is_synthetic'] = True
        else:
            item['is_synthetic'] = False
        
        return item


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset classes...")
    
    # Test transforms
    train_tf = get_train_transforms(224)
    val_tf = get_val_transforms(224)
    
    # Create dummy data for testing
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    train_result = train_tf(image=dummy_img)
    val_result = val_tf(image=dummy_img)
    
    print(f"Train transform output shape: {train_result['image'].shape}")
    print(f"Val transform output shape: {val_result['image'].shape}")
    
    print("\nDataset classes ready for use!")
    print("Download datasets from:")
    print("  - ISIC 2020: https://www.isic-archive.com/")
    print("  - Fitzpatrick17k: https://github.com/mattgroh/fitzpatrick17k")
    print("  - DDI: https://ddi-dataset.github.io/")
