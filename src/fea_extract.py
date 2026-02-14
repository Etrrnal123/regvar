"""
Generate DNA features using NT model (or deterministic random fallback).
Input TSV format: ref_seq,alt_seq,label
Outputs: features.npy (N,D) and labels.npy (N,) saved to config['fea_dir'].
"""
import os
import json
import csv
import hashlib
import argparse
import traceback
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import gc
import sys


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_nt_model(config):
    try:
        model_path = config.get("model_path", "./models/NT_model")
        print(f"Loading NT model/tokenizer from {model_path}...", flush=True)
        # Clear any existing models from memory
        gc.collect()
        torch.cuda.empty_cache()
        
        print("Loading tokenizer...", flush=True)
        # NT模型使用EsmTokenizer，需要trust_remote_code以支持自定义模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("Loaded tokenizer", flush=True)
        print("Loading model...", flush=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        print("Loaded model", flush=True)
        print("Loaded NT model/tokenizer.", flush=True)
        return tokenizer, model
    except Exception as e:
        print(f"Could not load NT model: {e}", flush=True)
        print("Full traceback:", flush=True)
        traceback.print_exc()
        return None, None



def deterministic_random_vector(seq: str, dim: int) -> np.ndarray:
    md5 = hashlib.md5(seq.encode("utf-8")).hexdigest()
    seed = int(md5[:8], 16)
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dim,)).astype(np.float32)


def seq_to_embedding(seq: str, model, tokenizer, config: dict) -> np.ndarray:
    emb_dim = int(config.get("embedding_dim", 1024))  # NT模型默认使用1024维嵌入
    device = torch.device(config.get("device", "cpu"))
    if model is None or tokenizer is None:
        print(f"Using deterministic random vector for sequence: {seq[:20]}...")
        return deterministic_random_vector(seq, emb_dim)

    # 确保序列是大写的
    seq = seq.strip().upper()
    print(f"Processing sequence: {seq[:20]}...")
    
    # NT模型最大序列长度为2048
    max_length = min(tokenizer.model_max_length, int(config.get("seq_len", 2048)))
    
    # 如果序列太长，截断到最大长度
    if len(seq) > max_length:
        seq = seq[:max_length]
        print(f"Sequence truncated to {max_length} characters")
    
    # 编码序列
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_length)
    
    # 确保输入张量和模型在同一设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 获取隐藏状态
        outputs = model(**inputs, output_hidden_states=True)
        # 获取最后一层隐藏状态
        last_hidden_states = outputs.hidden_states[-1]
        # 计算序列的平均嵌入（所有标记的平均值）
        emb = last_hidden_states.mean(dim=1)
    
    result = emb.squeeze().cpu().numpy()
    print(f"Generated embedding with shape: {result.shape}")
    return result


def generate_features(variants: List[Dict], model, tokenizer, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    use_delta = bool(config.get("use_delta", True))
    feats = []
    labels = []
    for i, v in enumerate(variants):
        if (i + 1) % 10 == 0:
            print(f"Processing variant {i+1}/{len(variants)}...")
        ref_seq = v["ref_seq"]
        alt_seq = v["alt_seq"]
        ref_emb = seq_to_embedding(ref_seq, model, tokenizer, config)
        alt_emb = seq_to_embedding(alt_seq, model, tokenizer, config)
        if use_delta:
            feature = alt_emb - ref_emb
        else:
            feature = np.concatenate([ref_emb, alt_emb])
        feats.append(feature)
        labels.append(int(v["label"]))
    return np.stack(feats, axis=0), np.array(labels)

def read_variants_from_tsv(path: str) -> List[Dict]:
    variants = []
    with open(path, "r", encoding="utf-8") as f:
        # Use delimiter='\t' for TSV files
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if "ref_seq" not in row or "alt_seq" not in row or "label" not in row:
                raise ValueError("TSV must contain columns: ref_seq, alt_seq, label")
            variants.append(row)
    return variants

def read_genomic_variants_from_tsv(path: str) -> pd.DataFrame:
    """
    读取基因组变异信息（染色体、位置、参考和替代碱基）
    """
    df = pd.read_csv(path, sep='\t')
    return df[['chr', 'pos', 'ref', 'alt', 'label']]

def save_features(features: np.ndarray, labels: np.ndarray, genomic_data: Optional[pd.DataFrame], out_dir: str, prefix: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    if prefix:
        np.save(os.path.join(out_dir, f"{prefix}_features.npy"), features)
        np.save(os.path.join(out_dir, f"{prefix}_labels.npy"), labels)
    else:
        np.save(os.path.join(out_dir, "features.npy"), features)
        np.save(os.path.join(out_dir, "labels.npy"), labels)
    
    # 如果提供了基因组数据，也保存下来
    if genomic_data is not None:
        genomic_data.to_csv(os.path.join(out_dir, "genomic_data.csv"), index=False)
    
    print(f"Saved features and labels to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input_tsv", required=True, help="Input TSV file with ref_seq, alt_seq, label columns")
    parser.add_argument("--genomic_tsv", help="TSV file with chr, pos, ref, alt, label columns for alphagenome")
    parser.add_argument("--out_dir", default=None, help="Output directory for features and labels")
    parser.add_argument("--prefix", default=None, help="Filename prefix, e.g., train_info or val_cosmic_info")
    args = parser.parse_args()

    config = load_config(args.config)
    # 确保配置中use_alphagenome为false
    config["use_alphagenome"] = False
    
    out_dir = args.out_dir or config.get("fea_dir", "./data/fea")
    os.makedirs(out_dir, exist_ok=True)
    base = args.prefix
    if base is None:
        try:
            base = os.path.splitext(os.path.basename(args.input_tsv))[0]
        except Exception:
            base = "dataset"

    print("About to load NT model...", flush=True)
    tokenizer, model = load_nt_model(config)
    print(f"Model loading result - Tokenizer: {'Loaded' if tokenizer else 'None'}, Model: {'Loaded' if model else 'None'}", flush=True)
    
    # Force flush stdout to see the messages
    sys.stdout.flush()
    
    if model is not None:
        device = torch.device(config.get("device", "cpu"))
        print(f"Moving model to device: {device}", flush=True)
        model.to(device)
        
    if model is None or tokenizer is None:
        print("NT model not loaded; using deterministic random embeddings.", flush=True)
    else:
        print("Setting model to evaluation mode...", flush=True)
        model.eval()
        print("Model is ready.", flush=True)

    # Force flush stdout to see the messages
    sys.stdout.flush()

    print("Reading variants from TSV...", flush=True)
    variants = read_variants_from_tsv(args.input_tsv)
    print(f"Read {len(variants)} variants.", flush=True)

    print("Generating features...", flush=True)
    features, labels = generate_features(variants, model, tokenizer, config)
    print(f"Generated features shape: {features.shape}", flush=True)

    # 读取基因组变异数据（如果提供了alphagenome输入文件）
    genomic_data = None
    if args.genomic_tsv:
        print("Reading genomic variants from TSV...", flush=True)
        genomic_data = read_genomic_variants_from_tsv(args.genomic_tsv)
        print(f"Read {len(genomic_data)} genomic variants for alphagenome.", flush=True)
        # 确保两种数据的数量一致
        if len(genomic_data) != len(variants):
            raise ValueError(f"Number of sequence variants ({len(variants)}) does not match number of genomic variants ({len(genomic_data)})")

    print("Saving features...", flush=True)
    save_features(features, labels, genomic_data, out_dir, prefix=base)
    print("Feature extraction completed.", flush=True)

if __name__ == "__main__":
    main()
