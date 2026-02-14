import os
import pandas as pd
import numpy as np
from intervaltree import IntervalTree

def add_prefix(d, prefix):
    return {f"{prefix}{k}": v for k, v in d.items()}

def _norm_chr(c):
    c = str(c).upper().replace("CHR", "")
    return {"M": "MT"}.get(c, c)

def _make_vid(df, chr_c, pos_c, ref_c, alt_c):
    return (
        df[chr_c].map(_norm_chr).astype(str)
        + ":"
        + df[pos_c].astype(int).astype(str)
        + ":"
        + df[ref_c].astype(str).str.upper()
        + ">"
        + df[alt_c].astype(str).str.upper()
    )

def build_donor_info(
    project_code_donor_path: str,
    pcawg_donor_clinical_path: str,
    pcawg_specimen_histology_path: str,
    purity_ploidy_path: str,
    donor_wgs_exclusion_path: str,
    sub_signatures_path: str,
) -> pd.DataFrame:
    project_code_df = pd.read_csv(project_code_donor_path, sep="\t")
    if "icgc_donor_id" in project_code_df.columns:
        project_code_df = project_code_df.rename(columns={"icgc_donor_id": "donor_id"})
    if "dcc_project_code" in project_code_df.columns:
        project_code_df = project_code_df.rename(columns={"dcc_project_code": "project_code"})
    clinical_df = pd.read_csv(pcawg_donor_clinical_path, sep="\t")
    if "icgc_donor_id" in clinical_df.columns:
        clinical_df = clinical_df.rename(columns={"icgc_donor_id": "donor_id"})
    purity_ploidy_df = pd.read_csv(purity_ploidy_path, sep="\t")
    if "icgc_donor_id" in purity_ploidy_df.columns:
        purity_ploidy_df = purity_ploidy_df.rename(columns={"icgc_donor_id": "donor_id"})
    histology_df = pd.read_csv(pcawg_specimen_histology_path, sep="\t")
    if "icgc_donor_id" in histology_df.columns:
        histology_df = histology_df.rename(columns={"icgc_donor_id": "donor_id"})
    exclusion_df = pd.read_csv(donor_wgs_exclusion_path, sep="\t")
    if "icgc_donor_id" in exclusion_df.columns:
        exclusion_df = exclusion_df.rename(columns={"icgc_donor_id": "donor_id"})
    if "donor_wgs_exclusion_white_gray" in exclusion_df.columns:
        exclusion_df = exclusion_df.rename(columns={"donor_wgs_exclusion_white_gray": "qc_status"})
    if "qc_status" in exclusion_df.columns:
        q = exclusion_df["qc_status"].astype(str).str.lower()
        q = q.replace({"whitelist": "white", "graylist": "gray"})
        exclusion_df["qc_status"] = q
    signatures_df = pd.read_csv(sub_signatures_path, sep="\t")
    if "icgc_donor_id" in signatures_df.columns:
        signatures_df = signatures_df.rename(columns={"icgc_donor_id": "donor_id"})
    elif "donor_id" not in signatures_df.columns and "Sample" in signatures_df.columns:
        signatures_df = signatures_df.rename(columns={"Sample": "donor_id"})
    donor_info = project_code_df.copy()
    if "donor_id" in clinical_df.columns:
        donor_info = donor_info.merge(clinical_df, on="donor_id", how="left")
    if "donor_id" in histology_df.columns:
        donor_info = donor_info.merge(histology_df, on="donor_id", how="left")
    if "donor_id" in purity_ploidy_df.columns:
        donor_info = donor_info.merge(purity_ploidy_df, on="donor_id", how="left")
    if "qc_status" in exclusion_df.columns:
        exclusion_map = dict(zip(exclusion_df["donor_id"], exclusion_df["qc_status"]))
    else:
        exclusion_map = {}
        for _, row in exclusion_df.iterrows():
            vals = list(row.values)
            if len(vals) >= 2:
                v = str(vals[1]).lower()
                if v.startswith("white"):
                    v = "white"
                elif v.startswith("gray"):
                    v = "gray"
                elif v.startswith("excl"):
                    v = "excluded"
                exclusion_map[vals[0]] = v
    donor_info["qc_status"] = donor_info["donor_id"].map(exclusion_map).fillna("white").str.lower()
    if "donor_id" in signatures_df.columns:
        sig_cols = [c for c in signatures_df.columns if c.startswith("Signature_")]
        if sig_cols:
            sig_df = signatures_df[["donor_id"] + sig_cols].copy()
            rename_map = {c: "sig_" + c.replace("Signature_", "").upper() for c in sig_cols}
            sig_df = sig_df.rename(columns=rename_map)
            donor_info = donor_info.merge(sig_df, on="donor_id", how="left")
    donor_info["tumor_type"] = donor_info["project_code"].astype(str).str.split("-").str[0]
    if "purity" not in donor_info.columns:
        donor_info["purity"] = np.nan
    if "ploidy" not in donor_info.columns:
        donor_info["ploidy"] = np.nan
    donor_info = donor_info.drop_duplicates(subset=["donor_id"]).reset_index(drop=True)
    return donor_info

def load_pcawg_cnv(cnv_path: str, donor_info: pd.DataFrame) -> pd.DataFrame:
    cnv_df = pd.read_csv(cnv_path, sep="\t")
    rename = {}
    if "sampleID" in cnv_df.columns:
        rename["sampleID"] = "donor_id"
    if "chr" in cnv_df.columns:
        rename["chr"] = "chrom"
    cnv_df = cnv_df.rename(columns=rename)
    for c in ["chrom"]:
        if c in cnv_df.columns and cnv_df[c].dtype == object:
            cnv_df[c] = cnv_df[c].astype(str).str.replace("chr", "", case=False)
    white = donor_info[donor_info["qc_status"] == "white"]["donor_id"].tolist()
    if "donor_id" in cnv_df.columns:
        cnv_df = cnv_df[cnv_df["donor_id"].isin(white)]
    for c in ["start", "end", "total_cn", "major_cn", "minor_cn"]:
        if c in cnv_df.columns:
            cnv_df[c] = pd.to_numeric(cnv_df[c], errors="coerce")
    cnv_df = cnv_df.dropna(subset=["donor_id", "chrom", "start", "end"])
    return cnv_df

def build_cnv_index(cnv_df: pd.DataFrame):
    index = {}
    for chrom, g in cnv_df.groupby("chrom"):
        t = IntervalTree()
        for _, r in g.iterrows():
            t[r["start"]:r["end"]] = {
                "donor_id": r.get("donor_id"),
                "total_cn": r.get("total_cn", np.nan),
                "major_cn": r.get("major_cn", np.nan),
                "minor_cn": r.get("minor_cn", np.nan),
            }
        index[chrom] = t
    return index

def pcawg_cnv_features_for_variant(
    chrom: str,
    pos: int,
    cnv_index,
    donor_info: pd.DataFrame,
    main_tumor_types: list | None = None,
) -> dict:
    out = {
        "total_cn_mean_all": 2.0,
        "total_cn_std_all": 0.0,
        "amp_freq_all": 0.0,
        "gain_freq_all": 0.0,
        "loss_freq_all": 0.0,
        "homdel_freq_all": 0.0,
        "loh_freq_all": 0.0,
    }
    if chrom not in cnv_index:
        return out
    segs = cnv_index[chrom][pos]
    if not segs:
        return out
    donors = []
    for seg in segs:
        d = seg.data
        donors.append((d.get("donor_id"), d.get("total_cn"), d.get("minor_cn")))
    if not donors:
        return out
    a = np.array([[x[1], x[2]] for x in donors], dtype=float)
    total = a[:, 0]
    minor = a[:, 1]
    out["total_cn_mean_all"] = float(np.mean(total))
    out["total_cn_std_all"] = float(np.std(total)) if len(total) > 1 else 0.0
    n = len(total)
    out["amp_freq_all"] = float(np.mean(total >= 4)) if n else 0.0
    out["gain_freq_all"] = float(np.mean(total > 2)) if n else 0.0
    out["loss_freq_all"] = float(np.mean(total < 2)) if n else 0.0
    out["homdel_freq_all"] = float(np.mean(total == 0)) if n else 0.0
    out["loh_freq_all"] = float(np.mean((minor == 0) & (total >= 1))) if n else 0.0
    if main_tumor_types:
        dmap = donor_info.set_index("donor_id")["tumor_type"].to_dict()
        by_type = {}
        for (donor_id, tcn, mcn) in donors:
            tt = dmap.get(donor_id)
            if tt in main_tumor_types:
                by_type.setdefault(tt, []).append((tcn, mcn))
        for tt in main_tumor_types:
            vals = by_type.get(tt, [])
            if vals:
                arr = np.array(vals, dtype=float)
                tc = arr[:, 0]
                mn = arr[:, 1]
                out[f"amp_freq_{tt}"] = float(np.mean(tc >= 4))
                out[f"gain_freq_{tt}"] = float(np.mean(tc > 2))
                out[f"loss_freq_{tt}"] = float(np.mean(tc < 2))
                out[f"homdel_freq_{tt}"] = float(np.mean(tc == 0))
                out[f"loh_freq_{tt}"] = float(np.mean((mn == 0) & (tc >= 1)))
            else:
                out[f"amp_freq_{tt}"] = 0.0
                out[f"gain_freq_{tt}"] = 0.0
                out[f"loss_freq_{tt}"] = 0.0
                out[f"homdel_freq_{tt}"] = 0.0
                out[f"loh_freq_{tt}"] = 0.0
    return out

def load_pcawg_whitelist_snv(maf_path: str, donor_info: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(maf_path, sep="\t")
    rename = {}
    if "Sample" in df.columns:
        rename["Sample"] = "donor_id"
    if "chr" in df.columns:
        rename["chr"] = "chrom"
    if "reference" in df.columns:
        rename["reference"] = "ref"
    df = df.rename(columns=rename)
    if "chrom" in df.columns:
        df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", case=False)
    df["effect"] = df.get("effect", pd.Series([np.nan] * len(df))).astype(str).str.lower()
    noncoding = {"intron", "igr", "rna", "utr", "intergenic"}
    df["is_noncoding"] = df["effect"].isin(noncoding)
    white = donor_info[donor_info["qc_status"] == "white"]["donor_id"].tolist()
    if "donor_id" in df.columns:
        df = df[df["donor_id"].isin(white)]
    for c in ["start", "end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["chrom", "start"])
    return df

def build_whitelist_index(whitelist_df: pd.DataFrame):
    idx = {}
    for chrom, g in whitelist_df.groupby("chrom"):
        g2 = g.sort_values("start").reset_index(drop=True)
        idx[chrom] = g2
    return idx

def pcawg_whitelist_features_for_variant(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    whitelist_index,
    windows=(10, 100, 1000),
) -> dict:
    out = {
        "exact_match_any": 0,
        "exact_match_noncoding": 0,
        "min_dist_any": 1_000_000,
        "min_dist_noncoding": 1_000_000,
    }
    for w in windows:
        out[f"count_pm{w}bp_all"] = 0
        out[f"count_pm{w}bp_noncoding"] = 0
    g = whitelist_index.get(chrom)
    if g is None or len(g) == 0:
        return out
    starts = g["start"].values
    lo = pos - max(windows)
    hi = pos + max(windows)
    mask = (starts >= lo) & (starts <= hi)
    cand = g.loc[mask]
    if len(cand) == 0:
        return out
    exact = cand[(cand["start"] == pos) & (cand.get("ref", "") == ref) & (cand.get("alt", "") == alt)]
    out["exact_match_any"] = int(len(exact) > 0)
    out["exact_match_noncoding"] = int(len(exact[exact["is_noncoding"]]) > 0) if "is_noncoding" in exact.columns else 0
    dists = np.abs(cand["start"].values - pos)
    out["min_dist_any"] = int(np.min(dists)) if len(dists) else out["min_dist_any"]
    if "is_noncoding" in cand.columns:
        nc = cand[cand["is_noncoding"]]
        d2 = np.abs(nc["start"].values - pos)
        out["min_dist_noncoding"] = int(np.min(d2)) if len(d2) else out["min_dist_noncoding"]
    for w in windows:
        m2 = (cand["start"] >= pos - w) & (cand["start"] <= pos + w)
        out[f"count_pm{w}bp_all"] = int(np.sum(m2))
        if "is_noncoding" in cand.columns:
            out[f"count_pm{w}bp_noncoding"] = int(np.sum(m2 & cand["is_noncoding"]))
    return out

def load_gene_annotation(probemap_path: str):
    df = pd.read_csv(probemap_path, sep="\t")
    cols = set(df.columns)
    chrom_col = "chrom" if "chrom" in cols else ("chr" if "chr" in cols else ("chromosome" if "chromosome" in cols else None))
    if chrom_col is None:
        raise ValueError("missing chrom column in probemap")
    df = df.rename(columns={chrom_col: "chrom"})
    if "start" not in df.columns or "end" not in df.columns:
        ren = {}
        if "chromStart" in df.columns:
            ren["chromStart"] = "start"
        if "chromEnd" in df.columns:
            ren["chromEnd"] = "end"
        df = df.rename(columns=ren)
    gene_col = "gene" if "gene" in df.columns else ("gene_name" if "gene_name" in df.columns else ("name" if "name" in df.columns else None))
    if gene_col is None:
        raise ValueError("missing gene column in probemap")
    df = df.rename(columns={gene_col: "gene_name"})
    df["chrom"] = df["chrom"].astype(str).str.replace("chr", "", case=False)
    for c in ["start", "end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["chrom", "start", "end"]).reset_index(drop=True)
    index = {}
    for chrom, g in df.groupby("chrom"):
        t = IntervalTree()
        for _, r in g.iterrows():
            t[r["start"]:r["end"]] = {"gene_name": r["gene_name"], "start": r["start"], "end": r["end"]}
        index[chrom] = t
    return df, index

def map_variant_to_gene(
    chrom: str,
    pos: int,
    gene_index,
    upstream: int = 100000,
    downstream: int = 100000,
):
    if chrom not in gene_index:
        return None, "intergenic", None
    segs = gene_index[chrom][pos]
    if segs:
        g = next(iter(segs)).data
        dist = int(pos - g["start"]) if g.get("start") is not None else None
        return g["gene_name"], "genic", dist
    lo = pos - upstream
    hi = pos + downstream
    candidates = [iv.data for iv in gene_index[chrom][lo:hi]]
    if not candidates:
        return None, "intergenic", None
    arr = np.array([[c["gene_name"], c["start"]] for c in candidates], dtype=object)
    dists = np.abs(arr[:, 1].astype(int) - pos)
    i = int(np.argmin(dists))
    gene = arr[i, 0]
    dist = int(arr[i, 1]) - pos
    rel = "upstream" if dist > 0 else "downstream"
    return gene, rel, -dist if rel == "upstream" else abs(dist)

def load_pcawg_expression(expr_path: str, donor_info: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(expr_path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    white = donor_info[donor_info["qc_status"] == "white"]["donor_id"].astype(str).tolist()
    keep_cols = [c for c in df.columns if c in white]
    if len(keep_cols) > 0:
        df = df[keep_cols]
    df = df.replace(-9.9658, np.nan)
    return df

def harmonize_expression_gene_symbols(expr_df: pd.DataFrame, gene_df: pd.DataFrame) -> pd.DataFrame:
    genes_idx = expr_df.index.astype(str).tolist()
    gene_series = gene_df.get("gene_name", pd.Series(dtype=str)).astype(str)
    up_to_canon = {s.upper(): s for s in gene_series.tolist()}
    id_col = None
    for cand in ["id", "#id", "gene_id", "geneId"]:
        if cand in gene_df.columns:
            id_col = cand
            break
    id_to_gene = {}
    if id_col is not None:
        ids = gene_df[id_col].astype(str).str.replace("\.\d+$", "", regex=True)
        id_to_gene = dict(zip(ids.tolist(), gene_df["gene_name"].astype(str).tolist()))
    def normalize_id(x: str) -> str:
        x = str(x)
        x = x.split("|")[0]
        x = x.strip()
        if x.startswith("ENSG"):
            x = x.split(".")[0]
        return x
    new_index = []
    for g in genes_idx:
        if g in up_to_canon.values():
            new_index.append(g)
        else:
            nid = normalize_id(g)
            if nid in id_to_gene:
                new_index.append(id_to_gene[nid])
            else:
                gu = g.upper()
                new_index.append(up_to_canon.get(gu, g))
    expr2 = expr_df.copy()
    expr2.index = new_index
    if len(set(expr2.index)) < len(expr2.index):
        expr2 = expr2.groupby(expr2.index).mean()
    return expr2

def build_gene_expression_stats(
    expr_df: pd.DataFrame,
    donor_info: pd.DataFrame,
    main_tumor_types: list | None = None,
) -> dict:
    stats = {}
    if expr_df.empty:
        return stats
    donors = expr_df.columns.astype(str).tolist()
    d2t = donor_info.set_index("donor_id")["tumor_type"].astype(str).to_dict()
    for gene in expr_df.index.astype(str).tolist():
        vals = expr_df.loc[gene].dropna().values
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        q25 = float(np.quantile(vals, 0.25))
        q75 = float(np.quantile(vals, 0.75))
        over = float(np.mean(vals > mean + 2 * std)) if std > 0 else 0.0
        under = float(np.mean(vals < mean - 2 * std)) if std > 0 else 0.0
        s = {
            "expr_mean_all": mean,
            "expr_std_all": std,
            "expr_q25_all": q25,
            "expr_q75_all": q75,
            "expr_overexpr_freq_all": over,
            "expr_underexpr_freq_all": under,
        }
        if main_tumor_types:
            for tt in main_tumor_types:
                donors_tt = [d for d in donors if d2t.get(d) == tt]
                if donors_tt:
                    vtt = expr_df.loc[gene, donors_tt].dropna().values
                    if len(vtt):
                        s[f"expr_overexpr_freq_{tt}"] = float(np.mean(vtt > mean + 2 * std)) if std > 0 else 0.0
                        s[f"expr_underexpr_freq_{tt}"] = float(np.mean(vtt < mean - 2 * std)) if std > 0 else 0.0
                    else:
                        s[f"expr_overexpr_freq_{tt}"] = 0.0
                        s[f"expr_underexpr_freq_{tt}"] = 0.0
                else:
                    s[f"expr_overexpr_freq_{tt}"] = 0.0
                    s[f"expr_underexpr_freq_{tt}"] = 0.0
        stats[gene] = s
    return stats

def load_pcawg_coding_drivers(coding_driver_path: str, donor_info: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(coding_driver_path, sep="\t")
    if "Sample" in df.columns and "donor_id" not in df.columns:
        df = df.rename(columns={"Sample": "donor_id"})
    if "sample" in df.columns and "donor_id" not in df.columns:
        df = df.rename(columns={"sample": "donor_id"})
    if "Gene" in df.columns and "gene" not in df.columns:
        df = df.rename(columns={"Gene": "gene"})
    df["driver_flag"] = 1
    white = donor_info[donor_info["qc_status"] == "white"]["donor_id"].tolist()
    if "donor_id" in df.columns:
        df = df[df["donor_id"].isin(white)]
    df = df.dropna(subset=["donor_id", "gene"]).reset_index(drop=True)
    return df

def build_gene_coding_driver_stats(
    coding_driver_df: pd.DataFrame,
    donor_info: pd.DataFrame,
) -> dict:
    if len(coding_driver_df) == 0:
        return {}
    proj_map = donor_info.set_index("donor_id")["project_code"].to_dict()
    stats = {}
    total_white = int((donor_info["qc_status"] == "white").sum())
    for gene, g in coding_driver_df.groupby("gene"):
        donors = set(g["donor_id"].astype(str).tolist())
        count = len(donors)
        freq = float(count) / total_white if total_white > 0 else 0.0
        projects = set([proj_map.get(d, None) for d in donors])
        projects = {p for p in projects if p is not None}
        stats[gene] = {
            "coding_driver_donor_count": int(count),
            "coding_driver_donor_freq": float(freq),
            "coding_driver_project_diversity": int(len(projects)),
        }
    return stats

def pcawg_methylation_features_for_variant(
    chrom: str,
    pos: int,
    gene: str | None,
    methylation_index,
    donor_info: pd.DataFrame,
) -> dict:
    return {}

def compute_pcawg_features_with_batch_processing(
    variants_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    donor_info = build_donor_info(
        config["project_code_donor_path"],
        config["pcawg_donor_clinical_path"],
        config["pcawg_specimen_histology_path"],
        config["purity_ploidy_path"],
        config["donor_wgs_exclusion_path"],
        config["sub_signatures_path"],
    )
    cnv_df = load_pcawg_cnv(config["cnv_path"], donor_info)
    cnv_index = build_cnv_index(cnv_df)
    wl_df = load_pcawg_whitelist_snv(config["whitelist_maf_path"], donor_info)
    wl_index = build_whitelist_index(wl_df)
    coding_df = load_pcawg_coding_drivers(config["coding_driver_path"], donor_info)
    gene_driver_stats = build_gene_coding_driver_stats(coding_df, donor_info)
    gene_df, gene_index = load_gene_annotation(config["probemap_path"])
    expr_df = load_pcawg_expression(config["expression_path"], donor_info)
    expr_df = harmonize_expression_gene_symbols(expr_df, gene_df)
    gene_expr_stats = build_gene_expression_stats(expr_df, donor_info, config.get("main_tumor_types"))
    batch_size = int(config.get("batch_size", 1000))
    rows = []
    total = len(variants_df)
    for i in range(0, total, batch_size):
        batch = variants_df.iloc[i : min(i + batch_size, total)].copy()
        for _, r in batch.iterrows():
            var_id = r["VariationID"]
            chrom = str(r["chr"]) if not pd.isna(r["chr"]) else str(r["chr"])
            pos = int(r["pos"]) if not pd.isna(r["pos"]) else int(r["pos"]) 
            ref = str(r["ref"]) if not pd.isna(r["ref"]) else str(r["ref"]) 
            alt = str(r["alt"]) if not pd.isna(r["alt"]) else str(r["alt"]) 
            gname, grel, dist = map_variant_to_gene(chrom, pos, gene_index)
            f_cnv = pcawg_cnv_features_for_variant(chrom, pos, cnv_index, donor_info, config.get("main_tumor_types"))
            f_wl = pcawg_whitelist_features_for_variant(chrom, pos, ref, alt, wl_index)
            f_gene = gene_driver_stats.get(gname, {
                "coding_driver_donor_count": 0,
                "coding_driver_donor_freq": 0.0,
                "coding_driver_project_diversity": 0,
            })
            f_meth = pcawg_methylation_features_for_variant(chrom, pos, gname, None, donor_info)
            default_expr = {
                "expr_mean_all": np.nan,
                "expr_std_all": np.nan,
                "expr_q25_all": np.nan,
                "expr_q75_all": np.nan,
                "expr_overexpr_freq_all": 0.0,
                "expr_underexpr_freq_all": 0.0,
            }
            f_expr = add_prefix(gene_expr_stats.get(gname, default_expr), "pcawg_expr_")
            row = {
                "VariationID": var_id,
                "pcawg_gene": gname,
                "pcawg_gene_relation": grel,
                "pcawg_dist_to_tss": dist,
                **add_prefix(f_cnv, "pcawg_cnv_"),
                **add_prefix(f_wl, "pcawg_wl_"),
                **add_prefix(f_gene, "pcawg_gene_"),
                **add_prefix(f_meth, "pcawg_meth_"),
                **f_expr,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if "VariationID" in df.columns:
        df = df.set_index("VariationID")
    return df

def compute_pcawg_features_for_variants(variants_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = variants_df.copy()
    if "chr" not in df.columns or "pos" not in df.columns or "ref" not in df.columns or "alt" not in df.columns:
        raise ValueError("variants_df missing required columns")
    df["chr"] = df["chr"].astype(str).str.replace("chr", "", case=False)
    df["pos"] = df["pos"].astype(int)
    if "VariationID" not in df.columns:
        df["VariationID"] = df.apply(lambda r: f"{r['chr']}_{r['pos']}_{r['ref']}_{r['alt']}", axis=1)
    return compute_pcawg_features_with_batch_processing(df, config)

def _build_config_from_dir(pca_dir: str, batch_size: int | None = None):
    base = pca_dir
    cfg = {
        "project_code_donor_path": os.path.join(base, "project_code_donor"),
        "pcawg_donor_clinical_path": os.path.join(base, "pcawg_donor_clinical_August2016_v9"),
        "pcawg_specimen_histology_path": os.path.join(base, "pcawg_specimen_histology_August2016_v9_donor"),
        "purity_ploidy_path": os.path.join(base, "consensus.20170217.purity.ploidy_donor"),
        "donor_wgs_exclusion_path": os.path.join(base, "donor_wgs_exclusion_white_gray"),
        "sub_signatures_path": os.path.join(base, "PCAWG_sub_signatures_in_samples_beta2.20170320.donor"),
        "cnv_path": os.path.join(base, "20170119_final_consensus_copynumber_donor"),
        "whitelist_maf_path": os.path.join(base, "October_2016_whitelist_2583.snv_mnv_indel.maf.xena.nonUS"),
        "coding_driver_path": os.path.join(base, "pcawg_whitelist_coding_drivers_v1_sep302016.donor.xena"),
        "probemap_path": os.path.join(base, "gencode.v19.annotation.gene.probemap"),
        "expression_path": os.path.join(base, "tophat_star_fpkm_uq.v2_aliquot_gl.donor.log"),
    }
    if batch_size is not None:
        cfg["batch_size"] = int(batch_size)
    return cfg

if __name__ == "__main__":
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--pca_dir", type=str, default=os.path.join("src", "pcawg_features", "PCAWG"))
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--main_tumor_types", type=str, default="")
    parser.add_argument("--prefix", type=str, default=None)
    args = parser.parse_args()
    sep = "\t" if args.input.lower().endswith(".tsv") else ","
    variants_df = pd.read_csv(args.input, sep=sep)
    variants_df["variant_id"] = _make_vid(variants_df, "chr", "pos", "ref", "alt")
    cfg = _build_config_from_dir(args.pca_dir, args.batch_size)
    if args.main_tumor_types:
        cfg["main_tumor_types"] = [t.strip() for t in args.main_tumor_types.split(",") if t.strip()]
    features = compute_pcawg_features_for_variants(variants_df, cfg)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    features.reset_index().to_csv(args.output, sep="\t", index=False)
    num_cols = [c for c in features.columns if c.startswith("pcawg_") and pd.api.types.is_numeric_dtype(features[c])]
    base = args.prefix
    if base is None:
        try:
            base = os.path.splitext(os.path.basename(args.input))[0]
        except Exception:
            base = "dataset"
    npy_path = os.path.join(os.path.dirname(args.output), f"{base}_pcawg_features.npy")
    np.save(npy_path, features[num_cols].to_numpy(dtype="float32"))
    ids_path = os.path.join(os.path.dirname(args.output), f"{base}_pcawg_variant_ids.txt")
    with open(ids_path, "w", encoding="utf-8") as f:
        for vid in variants_df["variant_id"].tolist():
            f.write(str(vid) + "\n")

