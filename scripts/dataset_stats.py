"""
Dataset Statistics and Profiling for DERM-EQUITY

Analyzes downloaded datasets for:
- Class distribution
- Skin tone distribution
- Image dimensions and quality
- Data splits

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --dataset isic2020
    python scripts/dataset_stats.py --analyze-fairness
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DatasetProfiler:
    """Analyzes dataset statistics."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.stats = {}
        
    def profile_isic2020(self) -> Dict:
        """Profile ISIC 2020 dataset."""
        dataset_dir = self.data_dir / "isic2020"
        
        if not dataset_dir.exists():
            print(f"⚠️  Dataset not found at {dataset_dir}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"📊 ISIC 2020 Dataset Profile")
        print(f"{'='*60}")
        
        stats = {
            'name': 'ISIC 2020',
            'path': str(dataset_dir),
            'dataset_size': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'num_classes': 9,
            'class_distribution': {},
            'fitzpatrick_distribution': {},
            'image_stats': {},
            'split_info': {},
        }
        
        # Look for CSV metadata
        csv_files = list(dataset_dir.glob("*.csv"))
        
        if not csv_files:
            print("⚠️  No CSV metadata found")
            return stats
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                print(f"\n📋 File: {csv_file.name}")
                print(f"   Shape: {df.shape}")
                
                # Class distribution
                if 'target' in df.columns:
                    class_dist = df['target'].value_counts().sort_index().to_dict()
                    stats['class_distribution'].update(class_dist)
                    print(f"   Classes: {list(class_dist.keys())}")
                    print(f"   Distribution:")
                    for cls, count in sorted(class_dist.items()):
                        pct = 100.0 * count / len(df)
                        print(f"      Class {cls}: {count:,} ({pct:.1f}%)")
                
                # Skin tone if available
                if 'fitzpatrick' in df.columns or 'skin_tone' in df.columns:
                    tone_col = 'fitzpatrick' if 'fitzpatrick' in df.columns else 'skin_tone'
                    tone_dist = df[tone_col].value_counts().sort_index().to_dict()
                    stats['fitzpatrick_distribution'].update(tone_dist)
                    print(f"   Fitzpatrick distribution:")
                    for tone, count in sorted(tone_dist.items()):
                        pct = 100.0 * count / len(df)
                        print(f"      Type {tone}: {count:,} ({pct:.1f}%)")
                
                # Split info
                split_name = csv_file.stem
                stats['split_info'][split_name] = {'samples': len(df)}
                
                stats['dataset_size'] += len(df)
                
                if 'train' in split_name:
                    stats['train_samples'] += len(df)
                elif 'val' in split_name:
                    stats['val_samples'] += len(df)
                elif 'test' in split_name:
                    stats['test_samples'] += len(df)
            
            except Exception as e:
                print(f"❌ Error reading {csv_file}: {e}")
        
        # Image analysis
        if PIL_AVAILABLE:
            img_dir = dataset_dir / "train"
            if img_dir.exists():
                self._analyze_images(img_dir, stats)
        
        return stats
    
    def profile_milk10k(self) -> Dict:
        """Profile MILK10k dataset."""
        dataset_dir = self.data_dir / "milk10k"
        
        if not dataset_dir.exists():
            print(f"⚠️  Dataset not found at {dataset_dir}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"📊 MILK10k Dataset Profile")
        print(f"{'='*60}")
        
        stats = {
            'name': 'MILK10k',
            'path': str(dataset_dir),
            'dataset_size': 0,
            'train_samples': 0,
            'val_samples': 0,
            'num_classes': 11,
            'class_distribution': {},
            'fitzpatrick_distribution': {},
            'image_stats': {},
            'dual_image_pairs': 0,
        }
        
        # Look for dual images (clinical + dermoscopic)
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                clin = list(split_dir.glob("*_clin.jpg")) + list(split_dir.glob("*_clinical.jpg"))
                derm = list(split_dir.glob("*_derm.jpg")) + list(split_dir.glob("*_dermoscopic.jpg"))
                
                print(f"\n   {split.upper()}:")
                print(f"   - Clinical images: {len(clin)}")
                print(f"   - Dermoscopic images: {len(derm)}")
                print(f"   - Pairs: {min(len(clin), len(derm))}")
                
                stats['dual_image_pairs'] += min(len(clin), len(derm))
                
                if split == 'train':
                    stats['train_samples'] = len(clin)
                elif split == 'val':
                    stats['val_samples'] = len(derm)
        
        # Read metadata CSV
        csv_file = dataset_dir / "metadata.csv" or dataset_dir / "train.csv"
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                
                print(f"\n   Metadata: {csv_file.name}")
                print(f"   Rows: {len(df)}")
                
                # Class distribution
                if 'diagnosis' in df.columns:
                    class_dist = df['diagnosis'].value_counts().to_dict()
                    stats['class_distribution'] = class_dist
                    print(f"   Diagnoses:")
                    for diag, count in sorted(class_dist.items(), key=lambda x: x[1], reverse=True):
                        print(f"      {diag}: {count}")
                
                # Fitzpatrick
                if 'fitzpatrick_skin_type' in df.columns:
                    tone_dist = df['fitzpatrick_skin_type'].value_counts().sort_index().to_dict()
                    stats['fitzpatrick_distribution'] = tone_dist
                    print(f"   Fitzpatrick distribution:")
                    for tone, count in sorted(tone_dist.items()):
                        print(f"      Type {tone}: {count}")
                
                stats['dataset_size'] = len(df)
            
            except Exception as e:
                print(f"❌ Error reading metadata: {e}")
        
        return stats
    
    def _analyze_images(self, img_dir: Path, stats: Dict) -> None:
        """Analyze image properties."""
        if not PIL_AVAILABLE:
            return
        
        print(f"\n   Image Analysis:")
        
        img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))
        
        if not img_paths:
            print(f"   No images found in {img_dir}")
            return
        
        heights, widths, sizes = [], [], []
        
        for img_path in img_paths[:min(100, len(img_paths))]:  # Sample up to 100
            try:
                img = Image.open(img_path)
                w, h = img.size
                size_mb = img_path.stat().st_size / (1024*1024)
                
                heights.append(h)
                widths.append(w)
                sizes.append(size_mb)
            
            except Exception as e:
                continue
        
        stats['image_stats'] = {
            'total_images': len(img_paths),
            'sampled_images': len(heights),
            'height': {'mean': np.mean(heights), 'std': np.std(heights)},
            'width': {'mean': np.mean(widths), 'std': np.std(widths)},
            'size_mb': {'mean': np.mean(sizes), 'std': np.std(sizes)},
        }
        
        print(f"   Total images: {len(img_paths)}")
        print(f"   Resolution: {np.mean(heights):.0f}×{np.mean(widths):.0f} ± "
              f"{np.std(heights):.0f}×{np.std(widths):.0f}")
        print(f"   Size: {np.mean(sizes):.2f} ± {np.std(sizes):.2f} MB")
    
    def analyze_fairness(self, stats: Dict) -> None:
        """Analyze fairness representation in dataset."""
        print(f"\n{'='*60}")
        print(f"⚖️  Fairness Analysis")
        print(f"{'='*60}")
        
        fitz_dist = stats.get('fitzpatrick_distribution', {})
        
        if not fitz_dist:
            print("No Fitzpatrick distribution data available")
            return
        
        print(f"\nFitzpatrick Representation:")
        total = sum(fitz_dist.values())
        
        for tone in range(1, 7):
            count = fitz_dist.get(tone, 0)
            if count > 0:
                pct = 100.0 * count / total
                bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                print(f"  Type {tone}: {bar} {pct:5.1f}% ({count:,})")
        
        # Check for representation gaps
        if fitz_dist:
            light_skin = sum([fitz_dist.get(i, 0) for i in [1, 2]])
            dark_skin = sum([fitz_dist.get(i, 0) for i in [5, 6]])
            
            if dark_skin > 0:
                gap_ratio = light_skin / dark_skin if dark_skin > 0 else float('inf')
                print(f"\n  Light skin (I-II) vs Dark skin (V-VI) ratio: {gap_ratio:.2f}x")
                print(f"  Light: {light_skin:,} | Dark: {dark_skin:,}")
                
                if gap_ratio > 2.0:
                    print(f"  ⚠️  Significant representation gap detected!")
                else:
                    print(f"  ✅ Reasonable representation balance")
    
    def generate_report(self, stats_dict: Dict) -> str:
        """Generate HTML report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Statistics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background: #007bff; color: white; }}
                .stat-box {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>🏥 DERM-EQUITY Dataset Statistics</h1>
            <p>Generated: {timestamp}</p>
        """
        
        for dataset_name, stats in stats_dict.items():
            html += f"""
            <div class="stat-box">
                <h2>{stats.get('name', 'Unknown')}</h2>
                <p><strong>Path:</strong> {stats.get('path', 'N/A')}</p>
                <p><strong>Total Samples:</strong> {stats.get('dataset_size', 0):,}</p>
                
                <h3>Split Distribution</h3>
                <table>
                    <tr>
                        <th>Split</th>
                        <th>Samples</th>
                        <th>Percentage</th>
                    </tr>
            """
            
            total = stats.get('dataset_size', 1)
            for split, info in stats.get('split_info', {}).items():
                samples = info.get('samples', 0)
                pct = 100.0 * samples / total if total > 0 else 0
                html += f"<tr><td>{split}</td><td>{samples:,}</td><td>{pct:.1f}%</td></tr>"
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def profile_all(self) -> Dict:
        """Profile all available datasets."""
        all_stats = {}
        
        all_stats['ISIC2020'] = self.profile_isic2020()
        all_stats['MILK10k'] = self.profile_milk10k()
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(description='Analyze DERM-EQUITY datasets')
    parser.add_argument('--dataset', choices=['isic2020', 'fitzpatrick17k', 'ddi', 'milk10k'],
                        help='Dataset to analyze')
    parser.add_argument('--analyze-fairness', action='store_true',
                        help='Detailed fairness analysis')
    parser.add_argument('--output', default='results',
                        help='Output directory for reports')
    
    args = parser.parse_args()
    
    profiler = DatasetProfiler('data')
    
    print("\n" + "="*60)
    print("📊 DERM-EQUITY Dataset Profiler")
    print("="*60)
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    if args.dataset == 'isic2020' or not args.dataset:
        all_stats['ISIC2020'] = profiler.profile_isic2020()
    
    if args.dataset == 'milk10k' or not args.dataset:
        all_stats['MILK10k'] = profiler.profile_milk10k()
    
    # Fairness analysis
    if args.analyze_fairness:
        for name, stats in all_stats.items():
            if stats:
                profiler.analyze_fairness(stats)
    
    # Generate report
    report_html = profiler.generate_report(all_stats)
    report_file = Path(args.output) / f"dataset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    with open(report_file, 'w') as f:
        f.write(report_html)
    
    print(f"\n✅ Report saved to: {report_file}")
    
    # Save JSON stats
    json_file = Path(args.output) / "dataset_stats.json"
    with open(json_file, 'w') as f:
        # Make serializable
        for key, val in all_stats.items():
            if isinstance(val.get('image_stats'), dict):
                val['image_stats'] = {k: float(v) if isinstance(v, (int, float)) else v 
                                     for k, v in val['image_stats'].items()}
        json.dump(all_stats, f, indent=2, default=str)
    
    print(f"✅ JSON stats saved to: {json_file}")


if __name__ == "__main__":
    main()
