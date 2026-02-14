"""
Training and evaluation script for RegVAR using K-Fold Cross-Validation with alphagenome support.
Usage:
  python src/task.py --config config.json
"""
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset

# 使用绝对导入而不是相对导入
import sys
import os
sys.path.append(os.path.dirname(__file__))

import fea_loader
from data_loader import RegVARDataset
import model
from ag_batch import precompute

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    batch_count = 0
    total_batches = len(loader)
    
    for batch in loader:
        batch_count += 1
        xb = batch['sequence_features'].to(device)
        yb = batch['label'].float().to(device)
        ag = batch['ag_feats'].to(device)  # 总是存在
        pcawg = batch['pcawg_feats'].to(device)  # 总是存在

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, ag, pcawg)  # [B]
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item() * xb.size(0)
        
        # 添加进度信息
        if batch_count % 50 == 0 or batch_count == total_batches:
            print(f"  Processed {batch_count}/{total_batches} batches ({100*batch_count/total_batches:.1f}%)")
            
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    probs_all, labels_all, vids_all = [], [], []
    batch_count = 0
    total_batches = len(loader)
    
    for batch in loader:
        batch_count += 1
        xb = batch['sequence_features'].to(device)
        yb = batch['label'].float().to(device)
        ag = batch['ag_feats'].to(device)  # 总是存在
        pcawg = batch['pcawg_feats'].to(device)  # 总是存在
        vids = batch['variant_id']

        logits = model(xb, ag, pcawg)
        probs = torch.sigmoid(logits)
        probs_all.append(probs.cpu())
        labels_all.append(yb.cpu())
        vids_all.extend(vids)
        
        # 添加进度信息
        if batch_count % 50 == 0 or batch_count == total_batches:
            print(f"  Evaluated {batch_count}/{total_batches} batches ({100*batch_count/total_batches:.1f}%)")
    
    # 合并所有结果
    p = torch.cat(probs_all).numpy()
    y = torch.cat(labels_all).numpy()
    
    # 打印概率标准差进行自检
    probs_std = torch.cat(probs_all).std().item()
    print(f"[Eval] probs std: {probs_std:.6f}")
    
    # 计算评估指标
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
    aupr = average_precision_score(y, p)
                
    return y, p, vids_all

def save_results(vids, labels, scores, output_path):
    import pandas as pd
    chrs, pos, refs, alts = [], [], [], []
    for vid in vids:
        try:
            # Expected format: chr:pos:REF>ALT
            parts = vid.split(':')
            c = parts[0]
            p = parts[1]
            ra = parts[2].split('>')
            r = ra[0]
            a = ra[1]
            chrs.append(c)
            pos.append(p)
            refs.append(r)
            alts.append(a)
        except Exception:
            # Fallback if format is unexpected
            chrs.append(vid)
            pos.append(0)
            refs.append('N')
            alts.append('N')
            
    df = pd.DataFrame({
        'Chr': chrs,
        'Pos': pos,
        'Ref': refs,
        'Alt': alts,
        'Label': labels,
        'Score': scores
    })
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved results to {output_path}")

def _norm_chr_value(x):
    s = str(x).upper().replace("CHR", "")
    return {"M": "MT"}.get(s, s)

def _make_vid_from_cols(chr_v, pos_v, ref_v, alt_v):
    try:
        p = int(pos_v)
    except Exception:
        p = int(float(pos_v))
    return f"{_norm_chr_value(chr_v)}:{p}:{str(ref_v).upper()}>{str(alt_v).upper()}"

def merge_kfold_results_to_train_info(output_dir: str, train_info_tsv: str, out_path: str | None = None):
    import glob
    import pandas as pd

    if out_path is None:
        out_path = os.path.join(output_dir, "train_info_results.tsv")

    fold_paths = sorted(glob.glob(os.path.join(output_dir, "fold*_results.tsv")))
    if not fold_paths:
        raise FileNotFoundError(f"no fold results found under: {output_dir}")
    if not os.path.exists(train_info_tsv):
        raise FileNotFoundError(f"train_info.tsv not found: {train_info_tsv}")

    fold_dfs = []
    for p in fold_paths:
        df = pd.read_csv(p, sep="\t")
        need = {"Chr", "Pos", "Ref", "Alt", "Score"}
        if not need.issubset(set(df.columns)):
            raise ValueError(f"unexpected columns in {p}: {list(df.columns)}")
        df = df.copy()
        df["variant_id"] = [
            _make_vid_from_cols(c, pos, r, a)
            for c, pos, r, a in zip(df["Chr"], df["Pos"], df["Ref"], df["Alt"])
        ]
        fold_dfs.append(df[["variant_id", "Score"]])

    all_preds = pd.concat(fold_dfs, axis=0, ignore_index=True)
    all_preds["Score"] = pd.to_numeric(all_preds["Score"], errors="coerce")
    all_preds = all_preds.dropna(subset=["variant_id", "Score"])
    pred_by_vid = all_preds.groupby("variant_id", as_index=False)["Score"].mean()

    ti = pd.read_csv(train_info_tsv, sep="\t")
    orig_cols = list(ti.columns)
    cols = {c.lower(): c for c in ti.columns}
    def pick(*cands):
        for k in cands:
            if k in cols:
                return cols[k]
        raise KeyError(f"missing columns: tried {cands} in {list(ti.columns)}")
    chr_c = pick("chr", "chrom", "chromosome")
    pos_c = pick("pos", "position", "start_position", "start")
    ref_c = pick("ref", "reference", "reference_allele")
    alt_c = pick("alt", "alternate", "tumor_seq_allele2", "alt_allele")

    ti = ti.copy()
    ti["variant_id"] = [
        _make_vid_from_cols(c, pos, r, a)
        for c, pos, r, a in zip(ti[chr_c], ti[pos_c], ti[ref_c], ti[alt_c])
    ]

    merged = ti.merge(pred_by_vid, on="variant_id", how="left")
    if "variant_id" not in orig_cols and "variant_id" in merged.columns:
        merged = merged.drop(columns=["variant_id"])
    covered = int(merged["Score"].notna().sum())
    total = int(len(merged))
    print(f"[OOF Merge] wrote {out_path}")
    print(f"[OOF Merge] covered {covered}/{total} rows with Score")
    merged.to_csv(out_path, sep="\t", index=False)
    return out_path


def main(config_path, merge_oof_only: bool = False):
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)

    if merge_oof_only:
        out_dir = config.get('output_dir', './output')
        train_info_tsv = os.path.join(config.get('raw_dir', './data/raw'), 'train_info.tsv')
        merge_kfold_results_to_train_info(out_dir, train_info_tsv)
        return

    # 设置随机种子
    set_seed(config.get('seed', 42))

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保输出目录存在
    os.makedirs(config.get('output_dir', './output'), exist_ok=True)
    os.makedirs(config.get('fea_dir', './data/fea'), exist_ok=True)
    dna_dir = os.path.join(config['fea_dir'], 'DNA')
    ag_dir = os.path.join(config['fea_dir'], 'AlphaGenome')
    pcawg_dir = os.path.join(config['fea_dir'], 'PCAWG')
    omics_dir = os.path.join(config['fea_dir'], 'Omics')
    omics_tag = 'train_info'
    omics_subdir = os.path.join(omics_dir, omics_tag)
    os.makedirs(dna_dir, exist_ok=True)
    os.makedirs(ag_dir, exist_ok=True)
    os.makedirs(pcawg_dir, exist_ok=True)
    os.makedirs(omics_dir, exist_ok=True)
    os.makedirs(omics_subdir, exist_ok=True)

    # 强制固定融合方式为 attention
    if config.get('fusion_mode') != 'attention':
        print(f"Overriding fusion_mode from {config.get('fusion_mode')} to attention")
        config['fusion_mode'] = 'attention'

    # 检查是否需要预计算 AlphaGenome 特征（固定使用）
    ag_features_file = os.path.join(ag_dir, 'alphagenome_features.parquet')
    if not os.path.exists(ag_features_file):
        ag_found = None
        if os.path.exists(ag_dir):
            for sub in os.listdir(ag_dir):
                p = os.path.join(ag_dir, sub)
                if not os.path.isdir(p):
                    continue
                files = set(os.listdir(p))
                cands = [f for f in files if f.endswith('_alphagenome.parquet')]
                pref = [f for f in cands if f.startswith('train_info')]
                if pref:
                    ag_found = os.path.join(p, pref[0])
                    break
                if cands:
                    ag_found = os.path.join(p, cands[0])
                    break
        if ag_found:
            ag_features_file = ag_found
            print(f"Using precomputed AlphaGenome features: {ag_features_file}")
        elif config.get('alphagenome_api_key'):
            print("Precomputing AlphaGenome features...")
            import subprocess
            import sys
            subprocess.run([
                sys.executable, '-m', 'src.ag_batch.precompute',
                '--input-tsv', os.path.join(config['raw_dir'], 'clinvar_cancer_noncoding_GRCh37.tsv'),
                '--output-parquet', ag_features_file,
                '--api-key', config['alphagenome_api_key'],
                '--cache-dir', 'ag_cache',
                '--max-workers', str(config.get('ag_max_workers', 12)),
                '--rate-limit', str(config.get('ag_rate_limit', 8.0)),
                '--scorers', 'RNA_SEQ', 'CAGE', 'PROCAP', 'DNASE', 'ATAC', 'CHIP_HISTONE', 'CHIP_TF', 'SPLICE_SITES', 'SPLICE_JUNCTIONS', 'SPLICE_SITE_USAGE', 'CONTACT_MAPS'
            ])
        else:
            print("AlphaGenome features not found and no API key provided; proceeding without precompute.")
    else:
        print("Using precomputed AlphaGenome features")
    
    # 预计算 PCAWG 特征（固定使用，Linux/跨平台，现在用 Python 直接生成）
    pcawg_features_file = os.path.join(pcawg_dir, 'pcawg_features.npy')
    if not config.get('use_omics_encoder', False):
        if not os.path.exists(pcawg_features_file):
            print("Precomputing PCAWG features...")
            import pandas as pd
            from src.pcawg_features.processor import _build_config_from_dir, compute_pcawg_features_for_variants
            train_tsv = os.path.join(config['raw_dir'], 'train_info_no_varid.tsv')
            df = pd.read_csv(train_tsv, sep='\t', header=None)
            if df.shape[1] < 4:
                raise ValueError("train_info_no_varid.tsv 至少需要四列：chr pos ref alt")
            df = df.iloc[:, :4]
            df.columns = ['chr', 'pos', 'ref', 'alt']
            df['chr'] = df['chr'].astype(str).str.replace('chr', '', case=False)
            df['pos'] = pd.to_numeric(df['pos'], errors='coerce').astype(int)
            df['ref'] = df['ref'].astype(str)
            df['alt'] = df['alt'].astype(str)
            vars_df = df.dropna().reset_index(drop=True)
            cfg = _build_config_from_dir(os.path.join(os.path.dirname(__file__), 'pcawg_features', 'PCAWG'), batch_size=1000)
            feats = compute_pcawg_features_for_variants(vars_df, cfg).reset_index()
            num_cols = [c for c in feats.columns if c.startswith('pcawg_') and pd.api.types.is_numeric_dtype(feats[c])]
            X = feats[num_cols].to_numpy(dtype='float32')
            np.save(pcawg_features_file, X.astype('float32'))
            with open(os.path.join(pcawg_dir, 'pcawg_variant_ids.txt'), 'w', encoding='utf-8') as f:
                for vid in feats['VariationID'].astype(str).tolist():
                    f.write(vid + '\n')
        else:
            print("Using precomputed PCAWG features")

    # 表格encoder预训练嵌入
    omics_emb_path = os.path.join(omics_subdir, 'omics_embeddings.npy')
    if config.get('use_omics_encoder', False):
        if not os.path.exists(omics_emb_path):
            saint_ckpt = os.path.join(config['fea_dir'], 'saint_pretrain.pth')
            if os.path.exists(saint_ckpt):
                print("Encoding omics embeddings from pretrained SAINT checkpoint...")
                import subprocess
                import sys
                
                # 确保输出目录存在
                os.makedirs(omics_subdir, exist_ok=True)
                
                # 构建命令行参数
                # 注意：saint_pretrain.py 已经修改为根据 fea_tag 自动构建输出路径 data/fea/Omics/{tag}
                # 所以我们只需要传入 output_dir=config['fea_dir'] 和 fea_tag=omics_tag
                subprocess.run([
                    sys.executable, '-m', 'models.tab_pretrain.saint_pretrain',
                    '--source', 'train',
                    '--train_tsv', os.path.join(config['raw_dir'], 'train_info.tsv'),
                    '--pca_dir', os.path.join(os.path.dirname(__file__), 'pcawg_features', 'PCAWG'),
                    '--output_dir', config['fea_dir'],
                    '--batch_size', str(config.get('omics_batch_size', 64)),
                    '--fea_tag', omics_tag,
                    '--encode_only',
                    '--ckpt', saint_ckpt
                ], check=True)
            else:
                raise FileNotFoundError(f"Missing SAINT checkpoint at {saint_ckpt}; please provide the pretrained model.")
        
        if os.path.exists(omics_emb_path):
            print(f"Using omics embeddings: {omics_emb_path}")
        else:
            print(f"Warning: Expected omics embeddings at {omics_emb_path} but not found even after running encoder.")

    # 评估模式（在测试集上评估已训练模型）
    if bool(config.get('eval_only', False)):
        print("[Mode] Evaluation-only")
        # 允许指定单独的测试特征目录，否则复用 fea_dir
        fea_dir_eval = config.get('fea_dir_test', config.get('fea_dir', './data/fea'))
        dna_dir_eval = os.path.join(fea_dir_eval, 'DNA')
        ag_dir_eval = os.path.join(fea_dir_eval, 'AlphaGenome')
        pcawg_dir_eval = os.path.join(fea_dir_eval, 'PCAWG')
        omics_dir_eval = os.path.join(fea_dir_eval, 'Omics')
        # 选择使用的第三模态
        pcawg_path_eval = os.path.join(pcawg_dir_eval, "pcawg_features.npy")
        if config.get('use_omics_encoder', False):
            omics_eval_subdir = os.path.join(omics_dir_eval, omics_tag, 'omics_embeddings.npy')
            if os.path.exists(omics_eval_subdir):
                pcawg_path_eval = omics_eval_subdir
        dataset_dir_eval = dna_dir_eval
        feat_path_eval = os.path.join(dataset_dir_eval, "features.npy")
        lab_path_eval = os.path.join(dataset_dir_eval, "labels.npy")
        gen_path_eval = os.path.join(dataset_dir_eval, "genomic_data.csv")
        if not os.path.exists(feat_path_eval):
            for sub in os.listdir(dna_dir_eval):
                p = os.path.join(dna_dir_eval, sub)
                if not os.path.isdir(p):
                    continue
                files = set(os.listdir(p))
                if "features.npy" in files and "labels.npy" in files and "genomic_data.csv" in files:
                    dataset_dir_eval = p
                    feat_path_eval = os.path.join(p, "features.npy")
                    lab_path_eval = os.path.join(p, "labels.npy")
                    gen_path_eval = os.path.join(p, "genomic_data.csv")
                    break
                for fx in files:
                    if fx.endswith("_features.npy"):
                        base = fx[: -len("_features.npy")]
                        if (base + "_labels.npy") in files and (base + "_genomic.csv") in files:
                            dataset_dir_eval = p
                            feat_path_eval = os.path.join(p, fx)
                            lab_path_eval = os.path.join(p, base + "_labels.npy")
                            gen_path_eval = os.path.join(p, base + "_genomic.csv")
                            break
        ag_parq_eval = os.path.join(ag_dir_eval, "alphagenome_features.parquet")
        if not os.path.exists(ag_parq_eval):
            ag_found_eval = None
            if os.path.exists(ag_dir_eval):
                for sub in os.listdir(ag_dir_eval):
                    p = os.path.join(ag_dir_eval, sub)
                    if not os.path.isdir(p):
                        continue
                    files = set(os.listdir(p))
                    cands = [f for f in files if f.endswith('_alphagenome.parquet')]
                    pref = [f for f in cands if f.startswith('train_info')]
                    if pref:
                        ag_found_eval = os.path.join(p, pref[0])
                        break
                    if cands:
                        ag_found_eval = os.path.join(p, cands[0])
                        break
            if ag_found_eval:
                ag_parq_eval = ag_found_eval
        eval_dataset = RegVARDataset(
            features_npy=feat_path_eval,
            labels_npy=lab_path_eval,
            genomic_csv=gen_path_eval,
            ag_parquet=ag_parq_eval,
            pcawg_npy=pcawg_path_eval,
            filter_zero_nt=True
        )
        eval_loader = DataLoader(eval_dataset, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=0)
        # 构建模型并加载权重
        regvar_model = model.RegVAR_DNA(
            nt_feat_dim=eval_dataset.nt_dim,
            ag_feat_dim=eval_dataset.ag_dim,
            pcawg_feat_dim=eval_dataset.pcawg_dim,
            hidden_dim=max(256, eval_dataset.nt_dim),
            dropout=float(config.get('dropout', 0.1)),
            modal_dropout_p=float(config.get('modal_dropout_p', 0.2)),
            attn_temperature=float(config.get('attn_temperature', 1.0)),
            fusion_mode=str(config.get('fusion_mode', 'attention'))
        ).to(device)
        ckpt = config.get('checkpoint', os.path.join(config.get('output_dir', './output'), 'best_model_fold1.pth'))
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        regvar_model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[Eval] loaded checkpoint: {ckpt}")
        ys, yps, vids = eval_model(regvar_model, eval_loader, device)
        auc = roc_auc_score(ys, yps) if len(set(ys)) == 2 else float("nan")
        aupr = average_precision_score(ys, yps)
        print(f"[Eval] AUC: {auc:.4f}, AUPR: {aupr:.4f}")
        
        # Save results
        out_file = os.path.join(config.get('output_dir', './output'), 'eval_results.tsv')
        save_results(vids, ys, yps, out_file)
        return

    # 加载数据（用于K折交叉验证的索引划分）
    # 先构建完整数据集（在这里就完成"过滤全零样本"）
    # 固定使用三个模态，必需提供所有特征文件
    pcawg_path = os.path.join(pcawg_dir, "pcawg_features.npy")
    if config.get('use_omics_encoder', False) and os.path.exists(omics_emb_path):
        pcawg_path = omics_emb_path
    dataset_dir = dna_dir
    feat_path = os.path.join(dataset_dir, "features.npy")
    lab_path = os.path.join(dataset_dir, "labels.npy")
    gen_path = os.path.join(dataset_dir, "genomic_data.csv")
    if not os.path.exists(feat_path):
        for sub in os.listdir(dna_dir):
            p = os.path.join(dna_dir, sub)
            if not os.path.isdir(p):
                continue
            files = set(os.listdir(p))
            if "features.npy" in files and "labels.npy" in files and "genomic_data.csv" in files:
                dataset_dir = p
                feat_path = os.path.join(p, "features.npy")
                lab_path = os.path.join(p, "labels.npy")
                gen_path = os.path.join(p, "genomic_data.csv")
                break
            for fx in files:
                if fx.endswith("_features.npy"):
                    base = fx[: -len("_features.npy")]
                    if (base + "_labels.npy") in files and (base + "_genomic.csv") in files:
                        dataset_dir = p
                        feat_path = os.path.join(p, fx)
                        lab_path = os.path.join(p, base + "_labels.npy")
                        gen_path = os.path.join(p, base + "_genomic.csv")
                        break
    ag_parq = os.path.join(ag_dir, "alphagenome_features.parquet")
    if not os.path.exists(ag_parq):
        ag_found = None
        if os.path.exists(ag_dir):
            for sub in os.listdir(ag_dir):
                p = os.path.join(ag_dir, sub)
                if not os.path.isdir(p):
                    continue
                files = set(os.listdir(p))
                cands = [f for f in files if f.endswith('_alphagenome.parquet')]
                pref = [f for f in cands if f.startswith('train_info')]
                if pref:
                    ag_found = os.path.join(p, pref[0])
                    break
                if cands:
                    ag_found = os.path.join(p, cands[0])
                    break
        if ag_found:
            ag_parq = ag_found
    full_dataset = RegVARDataset(
        features_npy=feat_path,
        labels_npy=lab_path,
        genomic_csv=gen_path,
        ag_parquet=ag_parq,
        pcawg_npy=pcawg_path,
        filter_zero_nt=True
    )
    
    # 现在是过滤后的长度，不是原始长度
    N = len(full_dataset)
    # 直接用数据集里已对齐/已过滤后的标签
    y_for_split = full_dataset.y

    auc_scores = []
    aupr_scores = []

    # K折交叉验证
    kfold = StratifiedKFold(n_splits=config.get('k_folds', 5), shuffle=True, random_state=config.get('seed', 42))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(N), y_for_split)):
        print(f"Training fold {fold + 1}")

        # 分割数据
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # 批量大小
        batch_size = config.get('batch_size', 32)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 初始化模型（固定使用三个模态）
        regvar_model = model.RegVAR_DNA(
            nt_feat_dim=full_dataset.nt_dim,
            ag_feat_dim=full_dataset.ag_dim,
            pcawg_feat_dim=full_dataset.pcawg_dim,
            hidden_dim=max(256, full_dataset.nt_dim),
            dropout=float(config.get('dropout', 0.1)),
            modal_dropout_p=float(config.get('modal_dropout_p', 0.2)),
            attn_temperature=float(config.get('attn_temperature', 1.0)),
            fusion_mode=str(config.get('fusion_mode', 'attention'))
        ).to(device)
        
        # 优化器（固定包含所有三个模态的参数）
        optimizer_params = [
            {"params": regvar_model.fusion.parameters()},
            {"params": regvar_model.classifier.parameters()},
            {"params": regvar_model.nt_proj.parameters()},
            {"params": regvar_model.ag_proj.parameters()},
            {"params": regvar_model.pcawg_proj.parameters()},
            {"params": regvar_model.attention_mlp.parameters()}
        ]
        
        optimizer = optim.Adam(optimizer_params, lr=config.get('lr', 1e-4), weight_decay=1e-4)
        
        # 学习率调度器
        total_steps = len(train_loader) * config.get('epochs', 10)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # 早停机制参数
        patience = 5
        best_val_auc = 0.0
        counter = 0
        best_model_state = None
        
        epochs = config.get('epochs', 30)
        if epochs < 30:
             print(f"Increasing epochs from {epochs} to 30 for better convergence.")
             epochs = 30
        
        # 训练
        for epoch in range(epochs):
            print(f"Fold {fold + 1}, Epoch {epoch + 1}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.6f}")
            train_loss = train_one_epoch(regvar_model, train_loader, optimizer, scheduler, device)
            print(f"  Train Loss: {train_loss:.4f}")
            
            # 评估
            ys, yps, vids = eval_model(regvar_model, val_loader, device)
            val_auc = roc_auc_score(ys, yps)
            val_aupr = average_precision_score(ys, yps)
            print(f"  Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                counter = 0
                best_model_state = regvar_model.state_dict()
                # 保存最佳模型到文件
                torch.save(best_model_state, os.path.join(config['output_dir'], f'best_model_fold{fold+1}.pth'))
                # 保存当前最佳结果
                save_results(vids, ys, yps, os.path.join(config['output_dir'], f'fold{fold+1}_results.tsv'))
                print(f"  New best model saved! AUC: {val_auc:.4f}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
        # 使用最佳模型进行最终评估
        regvar_model.load_state_dict(torch.load(os.path.join(config['output_dir'], f'best_model_fold{fold+1}.pth')))
        ys, yps, vids = eval_model(regvar_model, val_loader, device)
        auc = roc_auc_score(ys, yps)
        aupr = average_precision_score(ys, yps)

        auc_scores.append(auc)
        aupr_scores.append(aupr)

        print(f"Fold {fold + 1} - AUC: {auc:.4f}, AUPR: {aupr:.4f}")

    # 输出最终结果
    print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"Mean AUPR: {np.mean(aupr_scores):.4f} ± {np.std(aupr_scores):.4f}")
    train_info_tsv = os.path.join(config.get('raw_dir', './data/raw'), 'train_info.tsv')
    try:
        merge_kfold_results_to_train_info(config.get('output_dir', './output'), train_info_tsv)
    except Exception as e:
        print(f"[OOF Merge] skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--merge_oof_only', action='store_true', default=False)
    args = parser.parse_args()

    main(args.config, merge_oof_only=bool(args.merge_oof_only))
