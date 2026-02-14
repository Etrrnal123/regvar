"""
RegVAR model with advanced alphagenome integration and attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegVAR_DNA(nn.Module):
    def __init__(self, nt_feat_dim: int, ag_feat_dim: int, pcawg_feat_dim: int, 
                 hidden_dim: int = None, dropout: float = 0.1, modal_dropout_p: float = 0.2, attn_temperature: float = 1.0, fusion_mode: str = "attention"):
        super().__init__()
        # 固定使用三个模态
        self.use_alphagenome = True
        self.use_pcawg = True
        self.fusion_mode = fusion_mode
        
        # 特征维度调整
        hid = hidden_dim or nt_feat_dim
        
        # 为每个模态添加独立的特征处理层
        # NT模态处理层
        self.nt_proj = nn.Sequential(
            nn.LayerNorm(nt_feat_dim),  # 添加层归一化
            nn.Linear(nt_feat_dim, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # AlphaGenome模态处理层
        self.ag_proj = nn.Sequential(
            nn.LayerNorm(ag_feat_dim),  # 添加层归一化
            nn.Linear(ag_feat_dim, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # PCAWG模态处理层
        self.pcawg_proj = nn.Sequential(
            nn.LayerNorm(pcawg_feat_dim),  # 添加层归一化
            nn.Linear(pcawg_feat_dim, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        modal_count = 3
        self.modal_dropout_p = modal_dropout_p
        self.attn_temperature = attn_temperature
        self.attention_mlp = nn.Sequential(
            nn.Linear(hid * modal_count, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, modal_count),
        )
        
        # 融合后的特征处理层
        self.fusion = nn.Sequential(
            nn.Linear(hid * modal_count, hid),  # 考虑所有模态的特征
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # 分类器层
        self.classifier = nn.Linear(hid, 1)

    def forward(self, sequence_features: torch.Tensor, ag_feats: torch.Tensor, pcawg_feats: torch.Tensor):
        nt_emb = self.nt_proj(sequence_features.float())
        ag_emb = self.ag_proj(ag_feats.float())
        pcawg_emb = self.pcawg_proj(pcawg_feats.float())
        
        embeddings = [nt_emb, ag_emb, pcawg_emb]
        
        # Attention fusion with modal dropout
        if self.training and self.modal_dropout_p > 0:
            b = nt_emb.size(0)
            mask = torch.rand(b, 3, device=nt_emb.device) > self.modal_dropout_p
            ensure = (mask.sum(dim=1) == 0)
            if ensure.any():
                idx = torch.randint(0, 3, (ensure.sum(),), device=nt_emb.device)
                mask[ensure, idx] = True
            nt_emb = nt_emb * mask[:, 0].unsqueeze(-1)
            ag_emb = ag_emb * mask[:, 1].unsqueeze(-1)
            pcawg_emb = pcawg_emb * mask[:, 2].unsqueeze(-1)
            embeddings = [nt_emb, ag_emb, pcawg_emb]
            
        combined = torch.cat(embeddings, dim=-1)
        logits_attn = self.attention_mlp(combined) / max(self.attn_temperature, 1e-6)
        weights = torch.softmax(logits_attn, dim=-1)
        weighted_embeddings = []
        for i, emb in enumerate(embeddings):
            weighted_emb = weights[:, i].unsqueeze(-1) * emb
            weighted_embeddings.append(weighted_emb)
        fused = torch.cat(weighted_embeddings, dim=-1)
        
        x = self.fusion(fused)
        
        # 分类
        logits = self.classifier(x).squeeze(-1)  # [B]
        return logits
