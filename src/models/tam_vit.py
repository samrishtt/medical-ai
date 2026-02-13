"""
Tone-Aware Multi-Scale Vision Transformer (TAM-ViT)

Novel architecture that conditions attention on skin tone for equitable
performance across Fitzpatrick skin types I-VI.

Key innovations:
1. Multi-scale patch embedding for fine-grained lesion features
2. Skin tone estimation branch with differentiable conditioning
3. Tone-adaptive layer normalization (modulates based on skin tone)
4. Dual uncertainty head (epistemic + aleatoric)
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_


class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings.
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x


class SkinToneEstimator(nn.Module):
    """
    Estimates Fitzpatrick skin type from dermoscopy image.
    
    Uses a lightweight CNN to predict skin tone distribution,
    which is then used to condition the main transformer.
    """
    
    def __init__(self, num_tones: int = 6, hidden_dim: int = 256):
        super().__init__()
        
        # Lightweight feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_tones),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, H, W)
            
        Returns:
            tone_logits: Raw logits for Fitzpatrick types (B, 6)
            tone_probs: Softmax probabilities (B, 6)
        """
        features = self.features(x)
        features = features.flatten(1)
        tone_logits = self.classifier(features)
        tone_probs = F.softmax(tone_logits, dim=-1)
        return tone_logits, tone_probs


class ToneAdaptiveLayerNorm(nn.Module):
    """
    Layer normalization conditioned on skin tone embedding.
    
    Modulates the normalized output using tone-dependent scale (gamma)
    and shift (beta) parameters, enabling the model to adapt its
    representations based on predicted skin tone.
    
    Formula: output = gamma(tone) * LayerNorm(x) + beta(tone)
    """
    
    def __init__(self, dim: int, tone_dim: int = 768):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # Learnable tone-dependent modulation
        self.gamma_proj = nn.Sequential(
            nn.Linear(tone_dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(tone_dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
        )
        
        # Initialize to identity transformation
        nn.init.zeros_(self.gamma_proj[-1].weight)
        nn.init.zeros_(self.gamma_proj[-1].bias)
        nn.init.zeros_(self.beta_proj[-1].weight)
        nn.init.zeros_(self.beta_proj[-1].bias)
    
    def forward(self, x: torch.Tensor, tone_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, D)
            tone_embed: Tone embedding (B, tone_dim)
            
        Returns:
            Modulated output (B, N, D)
        """
        normalized = self.norm(x)
        
        # Compute tone-dependent modulation
        gamma = 1 + self.gamma_proj(tone_embed).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_proj(tone_embed).unsqueeze(1)  # (B, 1, D)
        
        return gamma * normalized + beta


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional relative position bias."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn


class ToneModulatedMLP(nn.Module):
    """
    MLP with tone-dependent gating mechanism.
    
    The output is modulated by a learned gating function of the skin tone,
    allowing the network to apply different transformations for different
    skin tones.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        tone_dim: int = 768,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # Tone-dependent gating
        self.gate = nn.Sequential(
            nn.Linear(tone_dim, hidden_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features // 2, out_features),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor, tone_embed: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        # Apply tone-dependent gating
        gate = self.gate(tone_embed).unsqueeze(1)  # (B, 1, D)
        x = x * gate
        
        x = self.drop(x)
        return x


class ToneConditionedBlock(nn.Module):
    """
    Transformer block with tone-aware components.
    
    Components:
    - Tone-adaptive layer normalization
    - Multi-head self-attention
    - Tone-modulated MLP
    - Stochastic depth for regularization
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        tone_dim: int = 768,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = ToneAdaptiveLayerNorm(dim, tone_dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = ToneAdaptiveLayerNorm(dim, tone_dim)
        self.mlp = ToneModulatedMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            tone_dim=tone_dim,
            drop=drop,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        tone_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x, tone_embed))
        x = x + self.drop_path(attn_out)
        
        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x, tone_embed), tone_embed))
        
        return x, attn_weights


class MultiScalePatchMerger(nn.Module):
    """
    Merges patch embeddings from multiple scales using cross-attention.
    
    This allows the model to capture both coarse (16x16 patches) and
    fine-grained (8x8 patches) features for better lesion analysis.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(
        self,
        coarse: torch.Tensor,
        fine: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coarse: Coarse-scale patches (B, N_coarse, D)
            fine: Fine-scale patches (B, N_fine, D)
            
        Returns:
            Merged patches (B, N_coarse, D)
        """
        # Cross-attention: coarse queries fine
        attended, _ = self.cross_attn(
            self.norm1(coarse),
            self.norm2(fine),
            fine
        )
        
        # Concatenate and project
        merged = torch.cat([coarse, attended], dim=-1)
        merged = self.proj(merged)
        
        return merged


class UncertaintyHead(nn.Module):
    """
    Dual uncertainty estimation head.
    
    Estimates both:
    - Epistemic uncertainty (model uncertainty, reducible with more data)
    - Aleatoric uncertainty (data noise, irreducible)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Variance head (for aleatoric uncertainty)
        self.variance_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),  # Ensure positive variance
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted variance for each class."""
        return self.variance_head(x) + 1e-6  # Numerical stability


class TAMViT(nn.Module):
    """
    Tone-Aware Multi-Scale Vision Transformer.
    
    A novel architecture for equitable skin lesion classification that:
    1. Estimates skin tone and conditions all layers on it
    2. Uses multi-scale patch embedding for fine-grained features
    3. Provides uncertainty estimates for safe clinical deployment
    
    Args:
        img_size: Input image size (default: 224)
        patch_sizes: List of patch sizes for multi-scale (default: [16, 8])
        in_chans: Number of input channels (default: 3)
        num_classes: Number of output classes (default: 9)
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP hidden dim ratio (default: 4.0)
        qkv_bias: Use bias in QKV projection (default: True)
        drop_rate: Dropout rate (default: 0.0)
        attn_drop_rate: Attention dropout rate (default: 0.0)
        drop_path_rate: Stochastic depth rate (default: 0.1)
        num_tones: Number of Fitzpatrick skin types (default: 6)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_sizes: List[int] = [16, 8],
        in_chans: int = 3,
        num_classes: int = 9,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        num_tones: int = 6,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_tones = num_tones
        
        # =================================================================
        # Skin Tone Estimation Branch
        # =================================================================
        self.tone_estimator = SkinToneEstimator(num_tones=num_tones)
        self.tone_embed = nn.Sequential(
            nn.Linear(num_tones, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # =================================================================
        # Multi-Scale Patch Embedding
        # =================================================================
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(img_size, ps, in_chans, embed_dim)
            for ps in patch_sizes
        ])
        
        # Use coarse scale as primary
        self.num_patches = self.patch_embeds[0].num_patches
        
        # Patch merger for multi-scale fusion
        self.patch_merger = MultiScalePatchMerger(embed_dim, num_heads)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # =================================================================
        # Transformer Blocks
        # =================================================================
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            ToneConditionedBlock(
                dim=embed_dim,
                num_heads=num_heads,
                tone_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # =================================================================
        # Classification Head
        # =================================================================
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes),
        )
        
        # =================================================================
        # Uncertainty Head
        # =================================================================
        self.uncertainty_head = UncertaintyHead(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        
        self.apply(self._init_weights_module)
    
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward_features(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features with skin tone conditioning.
        
        Returns dictionary with:
        - features: CLS token features
        - tone_probs: Predicted skin tone distribution
        - attention: Attention weights (if requested)
        """
        B = x.shape[0]
        
        # Step 1: Estimate skin tone
        # If input is 6 channels (stacked), use the last 3 (dermoscopic) for tone estimation
        # If input is 3 channels, use all of them
        if x.shape[1] == 6:
            img_for_tone = x[:, 3:, :, :]
        else:
            img_for_tone = x
            
        tone_logits, tone_probs = self.tone_estimator(img_for_tone)
        tone_embedding = self.tone_embed(tone_probs)  # (B, embed_dim)
        
        # Step 2: Multi-scale patch embedding
        coarse_patches = self.patch_embeds[0](x)  # 16x16 patches
        
        if len(self.patch_embeds) > 1:
            fine_patches = self.patch_embeds[1](x)  # 8x8 patches
            # Merge scales using cross-attention
            patches = self.patch_merger(coarse_patches, fine_patches)
        else:
            patches = coarse_patches
        
        # Step 3: Add CLS token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Step 4: Transformer blocks with tone conditioning
        attentions = []
        for block in self.blocks:
            x, attn = block(x, tone_embedding)
            if return_attention:
                attentions.append(attn)
        
        x = self.norm(x)
        
        result = {
            'features': x[:, 0],  # CLS token
            'all_tokens': x,
            'tone_logits': tone_logits,
            'tone_probs': tone_probs,
            'tone_embedding': tone_embedding,
        }
        
        if return_attention:
            result['attention'] = torch.stack(attentions, dim=1)
        
        return result
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_uncertainty: Whether to return uncertainty estimates
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with logits, probabilities, uncertainties, and attention
        """
        # Extract features
        feat_dict = self.forward_features(x, return_attention)
        features = feat_dict['features']
        
        # Classification
        logits = self.cls_head(features)
        probs = F.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probs': probs,
            'tone_logits': feat_dict['tone_logits'],
            'tone_probs': feat_dict['tone_probs'],
        }
        
        if return_uncertainty:
            variance = self.uncertainty_head(features)
            result['variance'] = variance
            result['uncertainty'] = variance.sum(dim=-1)  # Total uncertainty
        
        if return_attention:
            result['attention'] = feat_dict['attention']
        
        return result
    
    @torch.no_grad()
    def predict_with_mc_dropout(
        self,
        x: torch.Tensor,
        n_samples: int = 30,
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction with Monte Carlo Dropout for epistemic uncertainty.
        
        Args:
            x: Input images
            n_samples: Number of MC samples
            
        Returns:
            Mean prediction, epistemic uncertainty, aleatoric uncertainty
        """
        was_training = self.training
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            out = self.forward(x, return_uncertainty=True)
            predictions.append(out['probs'])
        
        preds = torch.stack(predictions)  # (n_samples, B, num_classes)
        
        # Mean prediction
        mean_pred = preds.mean(dim=0)
        
        # Epistemic uncertainty (variance across samples)
        epistemic = preds.var(dim=0).sum(dim=-1)
        
        # Aleatoric uncertainty (entropy of mean prediction)
        aleatoric = -torch.sum(
            mean_pred * torch.log(mean_pred + 1e-8), dim=-1
        )
        
        if not was_training:
            self.eval()
        
        return {
            'mean_probs': mean_pred,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': epistemic + aleatoric,
            'predictions': preds,
        }


def create_tam_vit_base(num_classes: int = 9, pretrained: bool = False, in_chans: int = 3) -> TAMViT:
    """Create TAM-ViT Base model."""
    model = TAMViT(
        img_size=224,
        patch_sizes=[16, 8],
        in_chans=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )
    
    if pretrained:
        # Load pretrained ViT weights (excluding tone-specific parts)
        # This would require downloading ViT-B/16 weights
        pass
    
    return model


def create_tam_vit_small(num_classes: int = 9) -> TAMViT:
    """Create TAM-ViT Small model (for faster experiments)."""
    return TAMViT(
        img_size=224,
        patch_sizes=[16],  # Single scale for efficiency
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )


if __name__ == "__main__":
    # Quick test
    model = create_tam_vit_base(num_classes=9)
    x = torch.randn(2, 3, 224, 224)
    
    out = model(x, return_uncertainty=True, return_attention=True)
    
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Probs shape: {out['probs'].shape}")
    print(f"Tone probs shape: {out['tone_probs'].shape}")
    print(f"Variance shape: {out['variance'].shape}")
    print(f"Uncertainty shape: {out['uncertainty'].shape}")
    print(f"Attention shape: {out['attention'].shape}")
    
    # MC Dropout test
    mc_out = model.predict_with_mc_dropout(x, n_samples=10)
    print(f"\nMC Dropout - Epistemic uncertainty: {mc_out['epistemic_uncertainty']}")
    print(f"MC Dropout - Aleatoric uncertainty: {mc_out['aleatoric_uncertainty']}")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,} ({params/1e6:.1f}M)")
