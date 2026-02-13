"""
MILK10k Dataset for ISIC Challenge.

Handles dual-image inputs (Clinical + Dermoscopic) and 11 diagnostic categories.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A

class MILK10kDataset(Dataset):
    """
    MILK10k Benchmark Dataset.
    
    Features:
    - Dual-input: Clinical (close-up) + Dermoscopic images
    - 11 Diagnostic Categories
    - Metadata: Age, Sex, Site, Skin Tone
    
    Args:
        root_dir: Path to dataset root (containing 'images' folder)
        csv_file: Path to metadata CSV
        transform: Albumentations transform (applied to both images)
        phase: 'train' or 'test'
    """
    
    CLASS_NAMES = [
        'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 
        'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
    ]
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[A.Compose] = None,
        phase: str = 'train',
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.phase = phase
        
        # Load metadata
        self.df = pd.read_csv(csv_file)
        
        # Create label mapping
        self.label_map = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        lesion_id = row['lesion_id']
        
        # Load images (Clinical + Dermoscopic)
        # Assuming file naming convention or column names: 'image_clinical', 'image_dermoscopic'
        # Adjust based on actual CSV structure. Most likely separate columns or predictable suffixes.
        # For now, let's assume columns 'img_clinical' and 'img_dermoscopic' exist, 
        # or we try to find them by ID if the dataset structure suggests it.
        # BASED ON MILK10k description: "Input data are image pairs... composed of one clinical close-up image and one dermoscopic image."
        # We'll assume the CSV has paths or filenames.
        
        # Check for potential column names based on common practices or valid guesses
        # If specific column names aren't known yet, we'll try to look for them.
        # Let's assume standard names for now and handle key errors if needed.
        
        clinical_path = self.root_dir / row.get('clinical_image_name', f"{lesion_id}_clin.jpg") 
        dermoscopic_path = self.root_dir / row.get('dermoscopic_image_name', f"{lesion_id}_derm.jpg")
        
        # Fallback for extensions
        if not clinical_path.exists(): clinical_path = clinical_path.with_suffix('.png')
        if not dermoscopic_path.exists(): dermoscopic_path = dermoscopic_path.with_suffix('.png')
            
        try:
            img_clin = np.array(Image.open(clinical_path).convert('RGB'))
            img_derm = np.array(Image.open(dermoscopic_path).convert('RGB'))
        except FileNotFoundError:
             # Create dummy images if files are missing (for testing/safety)
            img_clin = np.zeros((224, 224, 3), dtype=np.uint8)
            img_derm = np.zeros((224, 224, 3), dtype=np.uint8)
            print(f"Warning: Missing images for {lesion_id}")

        # Apply transforms
        # We apply the same geometric transform to both to maintain correspondence? 
        # Actually, they are different views, so independent augmentation might be better, 
        # OR inconsistent augmentation could hurt if the model learns correspondence.
        # Usually, standard practice matches spatial augmentations if they are registered, 
        # but these are different views (close-up vs dermoscopy). Independent is likely fine/better.
        
        if self.transform:
            # Independent transforms for robustness
            img_clin = self.transform(image=img_clin)['image']
            img_derm = self.transform(image=img_derm)['image']

        # Stack images: (3, H, W) + (3, H, W) -> (6, H, W)
        # Concatenate along channel dimension (dim=0 for torch tensors)
        image_stacked = torch.cat([img_clin, img_derm], dim=0)
        
        # Get label (multi-class one-hot or index)
        # If training, we expect a diagnosis.
        label = -1
        if self.phase != 'test':
            # Check if diagnosis column exists
            for col in ['diagnosis', 'pathology', 'dx']:
                if col in row:
                    diagnosis = row[col]
                    if diagnosis in self.label_map:
                        label = self.label_map[diagnosis]
                        break
        
        # Metadata
        meta = {
            'age': float(row.get('age', -1)),
            'sex': 1 if row.get('sex') == 'male' else 0, # Simple encoding
            'fitzpatrick': int(row.get('skin_tone', -1)),
            'anatom_site': row.get('anatom_site', 'unknown')
        }

        return {
            'image': image_stacked,
            'label': torch.tensor(label, dtype=torch.long),
            'fitzpatrick': torch.tensor(meta['fitzpatrick'], dtype=torch.long), # For tone-aware model
            'lesion_id': lesion_id,
            'metadata': meta
        }
