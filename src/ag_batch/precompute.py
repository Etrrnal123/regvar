"""
CLI tool to precompute AlphaGenome features from input TSV.
"""
import argparse
import os
import pandas as pd
import numpy as np
import json
import re
from typing import List
from dataclasses import is_dataclass, asdict
from .client import AlphaGenomeBatchScorer


def load_tsv_chunks(filepath: str, chunk_size: int = 1000):
    """
    Load TSV file in chunks.
    
    Args:
        filepath: Path to the TSV file
        chunk_size: Size of each chunk
        
    Yields:
        DataFrames of chunks
    """
    for chunk in pd.read_csv(filepath, sep='\t', chunksize=chunk_size):
        yield chunk


def process_chunk(chunk: pd.DataFrame, scorer: AlphaGenomeBatchScorer) -> List[dict]:
    """
    Process a chunk of variants.
    
    Args:
        chunk: DataFrame chunk
        scorer: AlphaGenomeBatchScorer instance
        
    Returns:
        List of results
    """
    # Convert chunk to list of dictionaries
    rows = chunk.to_dict('records')
    
    # Score variants
    results = scorer.score_many(rows)
    
    return results


# ------------- precompute.py 补丁开始 -------------
ASSAYS = ["rna_seq", "atac", "cage", "dnase", "chip_histone", "chip_tf", "procap", "splice_sites", "splice_junctions", "splice_site_usage", "contact_maps"]  # 可按需扩展

def _to_plain(obj):
    """尽力把对象转 dict/list/标量。"""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    if is_dataclass(obj):
        try:
            return asdict(obj)
        except Exception:
            pass
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj  # 标量或字符串

_num_pat = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _flatten_numbers(seq):
    """把任意嵌套 list/ndarray/标量 -> 1D float ndarray（过滤非数）。"""
    if seq is None:
        return np.array([], dtype=float)
    if isinstance(seq, str):
        # 从字符串里抓所有数字
        vals = [float(x) for x in _num_pat.findall(seq)]
        return np.asarray(vals, dtype=float)
    try:
        arr = np.asarray(seq, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        return arr
    except Exception:
        return np.array([], dtype=float)

def _stats_1d(arr: np.ndarray):
    """输出固定4维统计：mean/max/std/count（空则全0）。"""
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0
    return float(arr.mean()), float(arr.max()), float(arr.std()), int(arr.size)

def _extract_section_block(text: str, section_names=("reference", "alternate", "mutated")):
    """
    从 repr 字符串里粗定位 'reference=Output(...)' 和 'alternate/mutated=Output(...)' 的片段。
    返回 dict: {'reference': '...','alternate':'...'}（如缺失则为空串）。
    """
    out = {k: "" for k in ["reference", "alternate"]}
    # 按先后顺序切分，避免跨段误配
    for key in section_names:
        m = re.search(rf"{key}\s*=\s*Output\(", text)
        if m:
            start = m.end()
            # 近似匹配到下一个 '), <下一段key>=Output(' 或 '))]'
            tail = text[start:]
            # 用括号计数更严谨，这里用非贪婪直到 '), <word>=Output(' 或 '))]'
            m2 = re.search(r"\)\s*,\s*(reference|alternate|mutated)\s*=\s*Output\(|\)\)\]", tail)
            end = start + (m2.start() if m2 else len(tail))
            out["reference" if key == "reference" else "alternate"] = text[m.start():end]
    return out

def _extract_assay_values_from_block(block: str, assay: str):
    """
    在一个 section 片段（'reference=Output(...)'）中提取某个 assay 的 TrackData(values=array([...])...) 数字。
    """
    # 匹配： assay=TrackData(values=array(  [ ... ]  , dtype=...
    pat = re.compile(rf"{assay}\s*=\s*TrackData\(\s*values\s*=\s*array\(\s*(\[[\s\S]*?\])\s*,\s*dtype=", re.IGNORECASE)
    m = pat.search(block or "")
    if not m:
        return np.array([], dtype=float)
    return _flatten_numbers(m.group(1))

def _flatten_variant_output(obj):
    """
    将一个 VariantOutput（dict/obj/字符串 repr）扁平成固定维度数值特征。
    特征设计：对每个 assay 计算 ref/alt 的 mean/max/std/count 以及三种 delta。
    """
    feats = {}
    # 1) 若是 dict 结构，尽可能走结构化路径
    plain = _to_plain(obj)
    if isinstance(plain, dict) and ("reference" in plain or "alternate" in plain or "mutated" in plain):
        ref = plain.get("reference") or {}
        alt = plain.get("alternate") or plain.get("mutated") or {}
        for assay in ASSAYS:
            ref_vals = None
            alt_vals = None
            # 常见结构：{"rna_seq": {"values": [...]} } 或 {"rna_seq": [...]} 等
            if isinstance(ref, dict) and assay in ref:
                v = ref[assay]
                ref_vals = v.get("values") if isinstance(v, dict) else v
            if isinstance(alt, dict) and assay in alt:
                v = alt[assay]
                alt_vals = v.get("values") if isinstance(v, dict) else v
            r = _flatten_numbers(ref_vals)
            a = _flatten_numbers(alt_vals)
            r_mean, r_max, r_std, r_cnt = _stats_1d(r)
            a_mean, a_max, a_std, a_cnt = _stats_1d(a)
            feats[f"{assay}_ref_mean"]  = r_mean
            feats[f"{assay}_ref_max"]   = r_max
            feats[f"{assay}_ref_std"]   = r_std
            feats[f"{assay}_ref_count"] = r_cnt
            feats[f"{assay}_alt_mean"]  = a_mean
            feats[f"{assay}_alt_max"]   = a_max
            feats[f"{assay}_alt_std"]   = a_std
            feats[f"{assay}_alt_count"] = a_cnt
            feats[f"{assay}_delta_mean"] = float(a_mean - r_mean)
            feats[f"{assay}_delta_max"]  = float(a_max  - r_max)
            feats[f"{assay}_delta_std"]  = float(a_std  - r_std)
            feats[f"{assay}_delta_count"]= int(a_cnt - r_cnt)
        return feats

    # 2) 若是字符串 repr（你的样例），从文本中切块再抓数值
    if isinstance(obj, str):
        blocks = _extract_section_block(obj, section_names=("reference", "alternate", "mutated"))
        for assay in ASSAYS:
            r = _extract_assay_values_from_block(blocks.get("reference",""), assay)
            a = _extract_assay_values_from_block(blocks.get("alternate",""), assay)
            # 某些接口用 mutated 命名
            if a.size == 0:
                a = _extract_assay_values_from_block(blocks.get("alternate",""), assay) \
                    if "alternate" in blocks else _extract_assay_values_from_block(blocks.get("mutated",""), assay)
            r_mean, r_max, r_std, r_cnt = _stats_1d(r)
            a_mean, a_max, a_std, a_cnt = _stats_1d(a)
            feats[f"{assay}_ref_mean"]  = r_mean
            feats[f"{assay}_ref_max"]   = r_max
            feats[f"{assay}_ref_std"]   = r_std
            feats[f"{assay}_ref_count"] = r_cnt
            feats[f"{assay}_alt_mean"]  = a_mean
            feats[f"{assay}_alt_max"]   = a_max
            feats[f"{assay}_alt_std"]   = a_std
            feats[f"{assay}_alt_count"] = a_cnt
            feats[f"{assay}_delta_mean"] = float(a_mean - r_mean)
            feats[f"{assay}_delta_max"]  = float(a_max  - r_max)
            feats[f"{assay}_delta_std"]  = float(a_std  - r_std)
            feats[f"{assay}_delta_count"]= int(a_cnt - r_cnt)
        return feats

    # 3) 兜底：提取到的所有数字给一组总统计（避免空列）
    arr = _flatten_numbers(plain)
    mean_, max_, std_, cnt_ = _stats_1d(arr)
    feats["all_ref_mean"] = mean_
    feats["all_ref_max"]  = max_
    feats["all_ref_std"]  = std_
    feats["all_ref_count"]= cnt_
    feats["all_alt_mean"] = 0.0
    feats["all_alt_max"]  = 0.0
    feats["all_alt_std"]  = 0.0
    feats["all_alt_count"]= 0
    feats["all_delta_mean"] = 0.0
    feats["all_delta_max"]  = 0.0
    feats["all_delta_std"]  = 0.0
    feats["all_delta_count"]= 0
    return feats

def save_results(results: list, output_file: str, failures_file: str):
    """把 AlphaGenome 返回的 results 扁平成数值 parquet。"""
    successes, failures = [], []
    for r in results:
        if r and "error" not in r: successes.append(r)
        elif r: failures.append(r)

    rows = []
    for r in successes:
        base = {
            "variant_id": r.get("variant_id"),
            "CHROM":      str(r.get("CHROM")) if r.get("CHROM") is not None else None,
            "POS":        r.get("POS"),
            "REF":        r.get("REF"),
            "ALT":        r.get("ALT"),
        }
        vo = r.get("result", None)
        feats = _flatten_variant_output(vo)
        rows.append({**base, **feats})

    if rows:
        df = pd.DataFrame(rows)
        # 数值列探测（除标识列外）
        non_feat = {"variant_id","CHROM","POS","REF","ALT"}
        num_cols = [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]
        # 缺失填 0
        df[num_cols] = df[num_cols].fillna(0.0)
        # 去重
        df = df.drop_duplicates("variant_id")
        df.to_parquet(output_file, index=False)
        print(f"[AG] saved {len(df)} rows to {output_file} with {len(num_cols)} numeric feature cols")
    else:
        pd.DataFrame(columns=["variant_id","CHROM","POS","REF","ALT"]).to_parquet(output_file, index=False)
        print(f"[AG] created empty file: {output_file}")

    if failures:
        pd.json_normalize(failures).to_csv(failures_file, sep="\t", index=False)
        print(f"[AG] saved {len(failures)} failures to {failures_file}")
    else:
        with open(failures_file, "w", encoding="utf-8") as f:
            f.write("variant_id\tCHROM\tPOS\tREF\tALT\terror\n")
        print(f"[AG] created empty failures file: {failures_file}")
# ------------- precompute.py 补丁结束 -------------


def main():
    parser = argparse.ArgumentParser(description="Precompute AlphaGenome features")
    parser.add_argument("--input-tsv", required=True, help="Input TSV file")
    parser.add_argument("--output-parquet", default="alphagenome_features.parquet", 
                        help="Output parquet file")
    parser.add_argument("--failures-tsv", default="failures.tsv", 
                        help="Failures TSV file")
    parser.add_argument("--api-key", required=True, help="AlphaGenome API key")
    parser.add_argument("--organism", default="HOMO_SAPIENS", help="Organism")
    parser.add_argument("--sequence-length", type=int, default=131072, 
                        help="Sequence length")
    parser.add_argument("--scorers", nargs='+', default=["RNA_SEQ"], 
                        help="List of scorers to use")
    parser.add_argument("--cache-dir", default="ag_cache", help="Cache directory")
    parser.add_argument("--max-workers", type=int, default=12, 
                        help="Maximum number of workers")
    parser.add_argument("--rate-limit", type=float, default=8.0, 
                        help="Rate limit (requests per second)")
    parser.add_argument("--max-retries", type=int, default=5, 
                        help="Maximum number of retries")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                        help="Chunk size for processing")
    
    args = parser.parse_args()
    
    # Create scorer
    print(f"[DEBUG] 开始创建批量评分器...")
    scorer = AlphaGenomeBatchScorer(
        api_key=args.api_key,
        organism=args.organism,
        seq_len=args.sequence_length,
        scorers=args.scorers,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit,
        max_retries=args.max_retries
    )
    print(f"[DEBUG] 批量评分器创建成功")

    # Process input file in chunks
    all_results = []
    
    for i, chunk in enumerate(load_tsv_chunks(args.input_tsv, args.chunk_size)):
        print(f"Processing chunk {i+1}")
        chunk_results = process_chunk(chunk, scorer)
        all_results.extend(chunk_results)
    
    # Save results
    save_results(all_results, args.output_parquet, args.failures_tsv)
    print("Feature precomputation completed!")


if __name__ == "__main__":
    main()