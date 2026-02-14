"""
PyTorch Dataset for DNA features with alphagenome support.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import sys

# 使用绝对导入而不是相对导入
sys.path.append(os.path.dirname(__file__))

ID_COLS = {"variant_id","CHROM","POS","REF","ALT","chr","pos","ref","alt"}

def _norm_chr(c):
    c = str(c).upper().replace("CHR","")
    return {"M":"MT"}.get(c, c)

def _make_vid(df, chr_c, pos_c, ref_c, alt_c):
    return (df[chr_c].map(_norm_chr).astype(str) + ":" +
            df[pos_c].astype(int).astype(str) + ":" +
            df[ref_c].astype(str).str.upper() + ">" +
            df[alt_c].astype(str).str.upper())

class RegVARDataset(Dataset):
    def __init__(self, features_npy, labels_npy, genomic_csv,
                 ag_parquet, pcawg_npy, filter_zero_nt=True):
        self.X = np.load(features_npy).astype("float32")        # [N, nt_dim]
        self.y = np.load(labels_npy).reshape(-1).astype("int64")
        assert len(self.X) == len(self.y), "features / labels 长度不一致"

        g = pd.read_csv(genomic_csv)
        g["variant_id"] = _make_vid(g, "chr","pos","ref","alt")
        if g["variant_id"].duplicated().sum() > 0:
            dup = int(g["variant_id"].duplicated().sum())
            keep_mask = ~g["variant_id"].duplicated()
            self.X = self.X[keep_mask.values] if len(self.X) == len(g) else self.X
            self.y = self.y[keep_mask.values] if len(self.y) == len(g) else self.y
            g = g.loc[keep_mask].reset_index(drop=True)
            print(f"[Data] 去重重复 variant_id: {dup}")
        self.variant_ids = g["variant_id"].tolist()

        # 可选：过滤全零 NT 行
        if filter_zero_nt:
            row_norm = np.linalg.norm(self.X, axis=1)
            mask = row_norm > 0
            if mask.sum() < len(mask):
                drop = int((~mask).sum())
                self.X = self.X[mask]
                self.y = self.y[mask]
                g = g.loc[mask].reset_index(drop=True)
                self.variant_ids = g["variant_id"].tolist()
                print(f"[Data] 过滤全零 NT 样本: {drop}")

        self.nt_dim = self.X.shape[1]
        # 固定使用三个模态
        self.use_alphagenome = True
        self.ag_dim = 0
        self.ag_mat = None
        
        self.use_pcawg = True
        self.pcawg_dim = 0
        self.pcawg_mat = None

        # 加载AlphaGenome特征（固定使用）
        if ag_parquet is None:
            raise ValueError("必须提供AlphaGenome特征文件")
        
        ag = pd.read_parquet(ag_parquet)
        if "variant_id" not in ag.columns:
            ag["variant_id"] = _make_vid(ag, "CHROM","POS","REF","ALT")
        ag = ag.drop_duplicates("variant_id").set_index("variant_id")

        num_cols = [c for c in ag.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(ag[c])]
        assert len(num_cols) > 0, "AlphaGenome parquet 没有数值列，请先更新 precompute.py 重新生成！"
        self.ag_dim = len(num_cols)

        # 按训练样本顺序对齐
        ag_vals = ag.reindex(self.variant_ids)[num_cols].to_numpy(dtype="float32")
        coverage = np.isfinite(ag_vals).any(axis=1).mean()
        print(f"[AG] coverage（任一数值存在）= {coverage:.2%}")
        if coverage < 0.95:
            raise ValueError(f"AlphaGenome 覆盖率过低：{coverage:.2%}")

        self.ag_mat = np.nan_to_num(ag_vals, nan=0.0, posinf=0.0, neginf=0.0)
        ag_mean = self.ag_mat.mean(axis=0)
        ag_std = self.ag_mat.std(axis=0)
        ag_std[ag_std == 0] = 1.0
        self.ag_mat = (self.ag_mat - ag_mean) / ag_std
        
        # 加载PCAWG特征（固定使用）
        if pcawg_npy is None:
            raise ValueError("必须提供PCAWG特征文件")
        
        try:
            pcawg_feats = np.load(pcawg_npy).astype("float32")
            ids_dir = os.path.dirname(pcawg_npy)
            ids_file = os.path.dirname(pcawg_npy)
            ids_file = os.path.join(ids_dir, "pcawg_variant_ids.txt")
            if not os.path.exists(ids_file):
                alt_ids_file = os.path.join(ids_dir, "omics_variant_ids.txt")
                if os.path.exists(alt_ids_file):
                    ids_file = alt_ids_file
            if os.path.exists(ids_file):
                with open(ids_file, "r", encoding="utf-8") as f:
                    pcawg_ids = [line.strip() for line in f if line.strip()]
                def _norm_vid(s):
                    try:
                        if ":" in s:
                            parts = s.split(":")
                            if len(parts) != 3:
                                return s
                            ch = _norm_chr(parts[0])
                            pos = int(parts[1])
                            ref_alt = parts[2]
                            ra = ref_alt.split(">")
                            if len(ra) != 2:
                                return s
                            ref, alt = ra[0].upper(), ra[1].upper()
                            return f"{ch}:{pos}:{ref}>{alt}"
                        elif "_" in s:
                            parts = s.split("_")
                            if len(parts) != 4:
                                return s
                            ch = _norm_chr(parts[0])
                            pos = int(parts[1])
                            ref = parts[2].upper()
                            alt = parts[3].upper()
                            return f"{ch}:{pos}:{ref}>{alt}"
                        else:
                            return s
                    except Exception:
                        return s
                id_to_idx = {_norm_vid(vid): i for i, vid in enumerate(pcawg_ids)}
                dim = pcawg_feats.shape[1]
                aligned = np.zeros((len(self.y), dim), dtype="float32")
                hit = 0
                for i, vid in enumerate(self.variant_ids):
                    j = id_to_idx.get(vid)
                    if j is not None:
                        aligned[i] = pcawg_feats[j]
                        hit += 1
                self.pcawg_dim = dim
                self.pcawg_mat = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"[PCAWG] 已按variant_id对齐，命中: {hit}/{len(self.y)}")
                pc_mean = self.pcawg_mat.mean(axis=0)
                pc_std = self.pcawg_mat.std(axis=0)
                pc_std[pc_std == 0] = 1.0
                self.pcawg_mat = (self.pcawg_mat - pc_mean) / pc_std
            else:
                if len(pcawg_feats) == len(self.y):
                    self.pcawg_dim = pcawg_feats.shape[1]
                    self.pcawg_mat = np.nan_to_num(pcawg_feats, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f"[PCAWG] 已加载PCAWG特征，维度: {self.pcawg_dim}")
                    pc_mean = self.pcawg_mat.mean(axis=0)
                    pc_std = self.pcawg_mat.std(axis=0)
                    pc_std[pc_std == 0] = 1.0
                    self.pcawg_mat = (self.pcawg_mat - pc_mean) / pc_std
                else:
                    print(f"[PCAWG] 特征数量不匹配（{len(pcawg_feats)} vs {len(self.y)}），创建默认特征矩阵")
                    self.pcawg_dim = min(pcawg_feats.shape[1], 10)
                    self.pcawg_mat = np.zeros((len(self.y), self.pcawg_dim), dtype="float32")
                    print(f"[PCAWG] 已创建默认PCAWG特征矩阵，维度: {self.pcawg_dim}")
        except Exception as e:
            print(f"[PCAWG] 加载特征时出错: {e}，创建默认特征矩阵")
            self.pcawg_dim = 10
            self.pcawg_mat = np.zeros((len(self.y), self.pcawg_dim), dtype="float32")
            print(f"[PCAWG] 已创建默认PCAWG特征矩阵，维度: {self.pcawg_dim}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 总是返回三个模态的数据
        return {
            "sequence_features": torch.from_numpy(self.X[idx]),  # float32
            "label": int(self.y[idx]),
            "ag_feats": torch.from_numpy(self.ag_mat[idx]),  # float32
            "pcawg_feats": torch.from_numpy(self.pcawg_mat[idx]),  # float32
            "variant_id": str(self.variant_ids[idx])
        }
