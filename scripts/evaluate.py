"""
Comprehensive Evaluation Script for DERM-EQUITY

Evaluates trained models on test sets with:
- Classification metrics (AUC, F1, Accuracy, Sensitivity, Specificity)
- Fairness metrics (AUC gap, demographic parity, equalized odds)
- Calibration metrics (ECE, MCE) 
- Uncertainty metrics (if MC Dropout enabled)
- Per-skin-tone subgroup analysis
- Automatic fairness report generation

Usage:
    python scripts/evaluate.py \
        --checkpoint path/to/checkpoint.pt \
        --dataset isic2020 \
        --output results/evaluation_report.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.metrics import classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.tam_vit import TAMViT, create_tam_vit_base
from data.datasets import ISIC2020Dataset, Fitzpatrick17kDataset, MILK10kDataset
from data.milk10k_dataset import MILK10kDatasetFlexible
from evaluation.metrics import comprehensive_evaluation, print_evaluation_report
import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    checkpoint_path: str
    dataset_name: str = "isic2020"  # isic2020, fitzpatrick17k, milk10k, ddi
    dataset_path: str = ""
    batch_size: int = 32
    num_workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mc_dropout: bool = False
    mc_samples: int = 30
    output_dir: str = "results"
    verbose: bool = True


class DERMEQUITYEvaluator:
    """Evaluation pipeline for DERM-EQUITY models."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load dataset
        self.test_loader = self._load_dataset()
        
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.config.checkpoint_path}")
        
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # Handle both direct checkpoint and Lightning checkpoint
        if 'state_dict' in checkpoint:  # Lightning checkpoint
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix from Lightning checkpoints
            state_dict = {
                k.replace('model.', ''): v for k, v in state_dict.items()
                if k.startswith('model.')
            }
        else:  # Direct model checkpoint
            state_dict = checkpoint
        
        # Create model
        model = create_tam_vit_base(
            num_classes=9,
            pretrained=False,
            img_size=224,
        )
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        return model
    
    def _load_dataset(self) -> DataLoader:
        """Load evaluation dataset."""
        dataset_name = self.config.dataset_name.lower()
        
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == "isic2020":
            dataset = ISIC2020Dataset(
                root=self.config.dataset_path or "data/ISIC_2020",
                split='test',
                transform=None,  # Use default val transform
            )
        elif dataset_name == "fitzpatrick17k":
            dataset = Fitzpatrick17kDataset(
                root=self.config.dataset_path or "data/fitzpatrick17k",
                split='test',
                transform=None,
            )
        elif dataset_name == "milk10k":
            dataset = MILK10kDatasetFlexible(
                root=self.config.dataset_path or "data/MILK10k",
                split='test',
                transform=None,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        return loader
    
    @torch.no_grad()
    def predict_single_pass(self) -> Dict[str, np.ndarray]:
        """Standard single-pass inference."""
        all_probs = []
        all_labels = []
        all_tones = []
        all_variances = []
        
        for batch_idx, batch in enumerate(self.test_loader):
            if self.config.verbose and batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{len(self.test_loader)}")
            
            images = batch['image'].to(self.device)
            labels = batch['label'].cpu().numpy()
            
            # Forward pass
            outputs = self.model(images)
            
            # Extract outputs
            if isinstance(outputs, dict):
                probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
                if 'variance' in outputs:
                    all_variances.append(outputs['variance'].cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels)
            
            # Fitzpatrick if available
            if 'fitzpatrick' in batch:
                all_tones.append(batch['fitzpatrick'].cpu().numpy())
        
        results = {
            'probs': np.concatenate(all_probs, axis=0),
            'labels': np.concatenate(all_labels, axis=0),
        }
        
        if all_tones:
            results['tones'] = np.concatenate(all_tones, axis=0)
        
        if all_variances:
            results['variance'] = np.concatenate(all_variances, axis=0)
        
        return results
    
    @torch.no_grad()
    def predict_mc_dropout(self) -> Dict[str, np.ndarray]:
        """MC Dropout inference for uncertainty estimation."""
        print(f"\nRunning MC Dropout inference ({self.config.mc_samples} samples)...")
        
        # Enable dropout
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        all_probs_mc = []
        all_labels = []
        all_tones = []
        
        for sample in range(self.config.mc_samples):
            if self.config.verbose and sample % 5 == 0:
                print(f"  MC sample {sample + 1}/{self.config.mc_samples}")
            
            sample_probs = []
            
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                # Forward pass (dropout enabled)
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                
                sample_probs.append(probs)
                
                if sample == 0:
                    all_labels.append(labels)
                    if 'fitzpatrick' in batch:
                        all_tones.append(batch['fitzpatrick'].cpu().numpy())
            
            sample_probs = np.concatenate(sample_probs, axis=0)
            all_probs_mc.append(sample_probs)
        
        # Aggregate MC samples
        all_probs_mc = np.array(all_probs_mc)  # (mc_samples, N, num_classes)
        
        mean_probs = all_probs_mc.mean(axis=0)
        epistemic_uncertainty = all_probs_mc.var(axis=0).mean(axis=-1)  # Average variance per sample
        
        # Disable dropout
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
        
        results = {
            'probs': mean_probs,
            'labels': np.concatenate(all_labels, axis=0),
            'epistemic_uncertainty': epistemic_uncertainty,
            'mc_probs': all_probs_mc,  # Store all MC samples
        }
        
        if all_tones:
            results['tones'] = np.concatenate(all_tones, axis=0)
        
        return results
    
    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation."""
        print("\n" + "=" * 60)
        print("DERM-EQUITY MODEL EVALUATION")
        print("=" * 60)
        
        # Get predictions
        if self.config.mc_dropout:
            print("\nUsing MC Dropout for uncertainty estimation...")
            pred_results = self.predict_mc_dropout()
            uncertainty = pred_results.get('epistemic_uncertainty')
        else:
            print("\nRunning standard inference...")
            pred_results = self.predict_single_pass()
            uncertainty = pred_results.get('variance')
        
        # Compute evaluation metrics
        results = comprehensive_evaluation(
            y_true=pred_results['labels'],
            y_prob=pred_results['probs'],
            skin_tones=pred_results.get('tones'),
            uncertainty=uncertainty,
        )
        
        # Print report
        print_evaluation_report(results)
        
        return {
            'predictions': pred_results,
            'metrics': results,
        }
    
    def save_results(self, eval_results: Dict[str, Any]) -> str:
        """Save evaluation results to JSON and generate report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for JSON serialization
        metrics_json = self._prepare_for_json(eval_results['metrics'])
        
        # Save metrics
        metrics_file = Path(self.config.output_dir) / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        print(f"\n✅ Metrics saved to {metrics_file}")
        
        # Save predictions
        predictions_file = Path(self.config.output_dir) / f"predictions_{timestamp}.npz"
        np.savez_compressed(
            predictions_file,
            probs=eval_results['predictions']['probs'],
            labels=eval_results['predictions']['labels'],
            tones=eval_results['predictions'].get('tones', np.array([])),
        )
        print(f"✅ Predictions saved to {predictions_file}")
        
        # Generate HTML report
        report_html = self._generate_html_report(eval_results['metrics'], timestamp)
        report_file = Path(self.config.output_dir) / f"report_{timestamp}.html"
        with open(report_file, 'w') as f:
            f.write(report_html)
        print(f"✅ Report saved to {report_file}")
        
        return str(metrics_file)
    
    def _prepare_for_json(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to JSON-serializable types."""
        json_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                json_metrics[key] = self._prepare_for_json(value)
            elif isinstance(value, (np.floating, np.integer)):
                json_metrics[key] = float(value)
            elif isinstance(value, (tuple, list)):
                json_metrics[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for v in value]
            elif value is None:
                json_metrics[key] = None
            else:
                json_metrics[key] = value
        
        return json_metrics
    
    def _generate_html_report(self, metrics: Dict[str, Any], timestamp: str) -> str:
        """Generate HTML report of evaluation results."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DERM-EQUITY Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #007bff; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-value {{ font-weight: bold; color: #007bff; }}
                .warning {{ color: #ff6b6b; }}
                .success {{ color: #51cf66; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 DERM-EQUITY Model Evaluation Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                
                <h2>📊 Overall Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        for metric, value in metrics.get('overall', {}).items():
            if isinstance(value, tuple):
                value_str = f"{value[0]:.4f} [{value[1]:.4f}, {value[2]:.4f}]"
            else:
                value_str = f"{value:.4f}"
            html += f"<tr><td>{metric}</td><td class='metric-value'>{value_str}</td></tr>"
        
        html += """
                </table>
                
                <h2>⚖️ Fairness Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Assessment</th>
                    </tr>
        """
        
        fairness = metrics.get('fairness', {})
        auc_gap = fairness.get('auc_gap', 0.5)
        assessment = "✅ Good" if auc_gap < 0.10 else "⚠️ Needs improvement" if auc_gap < 0.20 else "❌ Poor"
        
        html += f"""
                    <tr>
                        <td>AUC Gap (Fitzpatrick Gap)</td>
                        <td class='metric-value'>{auc_gap:.4f}</td>
                        <td>{assessment}</td>
                    </tr>
                    <tr>
                        <td>Demographic Parity Diff</td>
                        <td class='metric-value'>{fairness.get('demographic_parity_diff', 0):.4f}</td>
                        <td></td>
                    </tr>
                    <tr>
                        <td>Equalized Odds Diff</td>
                        <td class='metric-value'>{fairness.get('equalized_odds_diff', 0):.4f}</td>
                        <td></td>
                    </tr>
                </table>
                
                <h2>👥 Subgroup Analysis (Fitzpatrick Types)</h2>
                <table>
                    <tr>
                        <th>Skin Tone</th>
                        <th>AUC</th>
                        <th>Sensitivity</th>
                        <th>Specificity</th>
                        <th>Samples</th>
                    </tr>
        """
        
        for group, metrics_subgroup in metrics.get('subgroups', {}).items():
            html += f"""
                    <tr>
                        <td>{group.replace('fitzpatrick_', 'Type ').replace('_', ' ')}</td>
                        <td class='metric-value'>{metrics_subgroup.get('auc_roc', 0):.4f}</td>
                        <td>{metrics_subgroup.get('sensitivity', 0):.4f}</td>
                        <td>{metrics_subgroup.get('specificity', 0):.4f}</td>
                        <td>{metrics_subgroup.get('n_samples', 0)}</td>
                    </tr>
            """
        
        html += """
                </table>
                
                <p style="margin-top: 40px; font-size: 12px; color: #666;">
                    Report generated by DERM-EQUITY evaluation pipeline.
                </p>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    parser = argparse.ArgumentParser(description="DERM-EQUITY Evaluation Script")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", default="isic2020", help="Dataset name (isic2020, fitzpatrick17k, milk10k)")
    parser.add_argument("--dataset-path", default="", help="Path to dataset root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-dropout", action="store_true", help="Enable MC Dropout for uncertainty")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of MC Dropout samples")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        mc_dropout=args.mc_dropout,
        mc_samples=args.mc_samples,
        output_dir=args.output,
        device=args.device,
        verbose=args.verbose,
    )
    
    # Run evaluation
    evaluator = DERMEQUITYEvaluator(config)
    eval_results = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(eval_results)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
