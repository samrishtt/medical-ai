#!/usr/bin/env python3
"""
Dataset Download Utility for DERM-EQUITY

Downloads and prepares:
- ISIC 2020 Challenge Dataset
- Fitzpatrick17k
- DDI (Diverse Dermatology Images)

Usage:
    python scripts/download_data.py --dataset isic2020
    python scripts/download_data.py --dataset fitzpatrick17k
    python scripts/download_data.py --all
"""

import os
import sys
import argparse
import zipfile
import tarfile
from pathlib import Path
import urllib.request
import shutil
from tqdm import tqdm

# Data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = None):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_isic2020():
    """
    Download ISIC 2020 Challenge Dataset.
    
    Note: The full dataset requires registration and is ~30GB.
    This script provides instructions for manual download.
    """
    print("\n" + "="*60)
    print("📂 ISIC 2020 Challenge Dataset")
    print("="*60)
    
    isic_dir = DATA_DIR / "isic2020"
    isic_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
The ISIC 2020 Challenge Dataset requires registration.

📋 Manual Download Instructions:
1. Visit https://www.isic-archive.com/
2. Create an account and log in
3. Navigate to: Challenges → ISIC 2020
4. Download:
   - ISIC_2020_Training_JPEG.zip (~30GB)
   - ISIC_2020_Training_GroundTruth.csv
5. Extract to: {isic_dir}/

Alternative (smaller sample):
- Visit: https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-256x256
- Download the 256x256 resized version

Expected structure after download:
{isic_dir}/
├── train/
│   ├── ISIC_0000001.jpg
│   ├── ISIC_0000002.jpg
│   └── ...
├── train.csv
└── test.csv
""".format(isic_dir=isic_dir))
    
    # Create placeholder files
    (isic_dir / "train").mkdir(exist_ok=True)
    
    # Create sample CSV
    sample_csv = isic_dir / "train.csv"
    if not sample_csv.exists():
        with open(sample_csv, 'w') as f:
            f.write("image_name,target,diagnosis,age_approx,sex,anatom_site_general_challenge\n")
            f.write("ISIC_0000000,0,melanocytic_nevus,45,male,torso\n")
            f.write("ISIC_0000001,1,melanoma,55,female,lower extremity\n")
        print(f"✓ Created sample CSV: {sample_csv}")
    
    return isic_dir


def download_fitzpatrick17k():
    """
    Download Fitzpatrick17k dataset.
    
    Contains ~17K images with Fitzpatrick skin type labels.
    """
    print("\n" + "="*60)
    print("📂 Fitzpatrick17k Dataset")
    print("="*60)
    
    fitz_dir = DATA_DIR / "fitzpatrick17k"
    fitz_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
The Fitzpatrick17k dataset is available on GitHub.

📋 Download Instructions:
1. Visit: https://github.com/mattgroh/fitzpatrick17k
2. Follow the download instructions in the README
3. The dataset requires agreeing to terms of use

Alternative access via Hugging Face:
- https://huggingface.co/datasets/mattgroh/fitzpatrick17k

Expected structure:
{fitz_dir}/
├── images/
│   ├── <md5hash1>.jpg
│   ├── <md5hash2>.jpg
│   └── ...
└── fitzpatrick17k.csv
""".format(fitz_dir=fitz_dir))
    
    # Create placeholder
    (fitz_dir / "images").mkdir(exist_ok=True)
    
    return fitz_dir


def download_ddi():
    """
    Download DDI (Diverse Dermatology Images) dataset.
    
    Contains 656 images curated for skin tone diversity.
    """
    print("\n" + "="*60)
    print("📂 DDI (Diverse Dermatology Images) Dataset")
    print("="*60)
    
    ddi_dir = DATA_DIR / "ddi"
    ddi_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
The DDI dataset is from Stanford.

📋 Download Instructions:
1. Visit: https://ddi-dataset.github.io/
2. Request access through the form
3. Download the images and annotations

Citation:
Daneshjou et al., "Disparities in Dermatology AI Performance..."
Lancet Digital Health, 2022

Expected structure:
{ddi_dir}/
├── images/
│   ├── DDI_0001.jpg
│   ├── DDI_0002.jpg
│   └── ...
└── ddi_metadata.csv
""".format(ddi_dir=ddi_dir))
    
    # Create placeholder
    (ddi_dir / "images").mkdir(exist_ok=True)
    
    return ddi_dir


def download_milk10k():
    """
    Download MILK10k Challenge Dataset.
    """
    print("\n" + "="*60)
    print("📂 MILK10k Challenge Dataset")
    print("="*60)
    
    milk_dir = DATA_DIR / "milk10k"
    milk_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
    The MILK10k Benchmark requires ISIC Archive login.
    
    📋 Manual Download Instructions:
    1. Visit: https://challenge.isic-archive.com/landing/milk10k/
    2. Click 'Download Data' (requires free account)
    3. You need:
       - Images (zip files mostly)
       - Metadata CSV
       
    structure:
    {milk_dir}/
    ├── train/
    │   ├── <lesion_id>_clin.jpg
    │   ├── <lesion_id>_derm.jpg
    │   └── ...
    ├── train.csv
    └── val.csv (if split manually)
    """.format(milk_dir=milk_dir))
    
    return milk_dir


def create_sample_data():
    """Create small sample data for testing."""
    print("\n" + "="*60)
    print("🧪 Creating Sample Data for Testing")
    print("="*60)
    
    import numpy as np
    from PIL import Image
    
    sample_dir = DATA_DIR / "sample"
    (sample_dir / "train").mkdir(parents=True, exist_ok=True)
    (sample_dir / "val").mkdir(parents=True, exist_ok=True)
    
    # Create random images
    for split in ["train", "val"]:
        n_images = 100 if split == "train" else 20
        for i in range(n_images):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img).save(sample_dir / split / f"sample_{i:04d}.jpg")
    
    # Create metadata CSV
    import pandas as pd
    
    train_data = []
    for i in range(100):
        train_data.append({
            'image_name': f'sample_{i:04d}',
            'target': np.random.randint(0, 9),
            'diagnosis': np.random.choice(['melanoma', 'nevus', 'bcc']),
            'fitzpatrick': np.random.randint(1, 7),
            'split': 'train',
        })
    
    val_data = []
    for i in range(20):
        val_data.append({
            'image_name': f'sample_{i:04d}',
            'target': np.random.randint(0, 9),
            'diagnosis': np.random.choice(['melanoma', 'nevus', 'bcc']),
            'fitzpatrick': np.random.randint(1, 7),
            'split': 'val',
        })
    
    pd.DataFrame(train_data).to_csv(sample_dir / "train.csv", index=False)
    pd.DataFrame(val_data).to_csv(sample_dir / "val.csv", index=False)
    
    print(f"✓ Created sample dataset at: {sample_dir}")
    print(f"  - Train: 100 images")
    print(f"  - Val: 20 images")
    
    return sample_dir


def verify_data():
    """Verify downloaded datasets."""
    print("\n" + "="*60)
    print("🔍 Verifying Data")
    print("="*60)
    
    datasets = {
        'ISIC 2020': DATA_DIR / "isic2020",
        'Fitzpatrick17k': DATA_DIR / "fitzpatrick17k", 
        'DDI': DATA_DIR / "ddi",
        'Sample': DATA_DIR / "sample",
    }
    
    for name, path in datasets.items():
        if path.exists():
            n_images = len(list(path.rglob("*.jpg"))) + len(list(path.rglob("*.png")))
            n_csv = len(list(path.glob("*.csv")))
            print(f"✓ {name}: {n_images} images, {n_csv} CSV files")
        else:
            print(f"✗ {name}: Not found")


def main():
    parser = argparse.ArgumentParser(description='Download DERM-EQUITY datasets')
    parser.add_argument('--dataset', type=str, choices=['isic2020', 'fitzpatrick17k', 'ddi', 'milk10k', 'sample'],
                        help='Dataset to download')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--verify', action='store_true', help='Verify existing downloads')
    args = parser.parse_args()
    
    print("🏥 DERM-EQUITY Data Download Utility")
    print(f"📁 Data directory: {DATA_DIR}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.verify:
        verify_data()
        return
    
    if args.all or args.dataset == 'isic2020':
        download_isic2020()
    
    if args.all or args.dataset == 'fitzpatrick17k':
        download_fitzpatrick17k()
    
    if args.all or args.dataset == 'ddi':
        download_ddi()
        
    if args.all or args.dataset == 'milk10k':
        download_milk10k()
    
    if args.dataset == 'sample':
        create_sample_data()
    
    if not args.dataset and not args.all:
        print("\nNo dataset specified. Use --dataset <name> or --all")
        print("Available datasets: isic2020, fitzpatrick17k, ddi, sample")
        print("\nFor testing, run: python scripts/download_data.py --dataset sample")
    
    verify_data()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
