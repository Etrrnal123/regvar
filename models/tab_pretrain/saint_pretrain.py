import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.tab_pretrain.saint_model import SAINT
from src.pcawg_features.processor import _build_config_from_dir, compute_pcawg_features_for_variants

def _pcawg_cache_paths(output_dir: str, tag: str):
    tag_dir = os.path.join(output_dir, "PCAWG", tag)
    npy_tag = os.path.join(tag_dir, f"{tag}_pcawg_features.npy")
    ids_tag = os.path.join(tag_dir, f"{tag}_pcawg_variant_ids.txt")

    legacy_dir = os.path.join(output_dir, "PCAWG")
    npy_legacy = os.path.join(legacy_dir, f"pcawg_features_{tag}.npy")
    ids_legacy = os.path.join(legacy_dir, f"pcawg_variant_ids_{tag}.txt")

    return (tag_dir, npy_tag, ids_tag, npy_legacy, ids_legacy)

class DataSetCatCon(Dataset):
    def __init__(self, X):
        self.X2 = X['data'].astype(np.float32)
        self.X2_mask = X['mask'].astype(np.int64)
        self.labels = X.get('labels')
        self.cls = np.zeros((self.X2.shape[0], 1), dtype=int)
        self.cls_mask = np.ones((self.X2.shape[0], 1), dtype=int)
    def __len__(self):
        return self.X2.shape[0]
    def __getitem__(self, idx):
        y = 0 if self.labels is None else int(self.labels[idx])
        return np.concatenate((self.cls[idx], self.X2[idx])), np.concatenate((self.cls_mask[idx], self.X2_mask[idx])), y

def embed_data_mask(x_cont, con_mask, model):
    device = x_cont.device
    n1, n2 = x_cont.shape
    x_cont_enc = torch.empty(n1, n2, model.dim, device=device)
    for i in range(model.num_continuous):
        x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i].unsqueeze(-1))
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]
    return x_cont_enc

def saint_pretrain(model, X_train, device, num_epoch=200, batch_size=64, patience=20, min_delta=1e-4):
    pt_aug = ['cutmix']
    pt_tasks = ['contrastive', 'denoising']
    pt_projhead_style = 'diff'
    nce_temp = 0.7
    lam0 = 0.5
    train_ds = DataSetCatCon(X_train)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    clf_head = nn.Linear(model.dim, 1).to(device)
    params = list(model.parameters()) + list(clf_head.parameters())
    optimizer = optim.AdamW(params, lr=1e-4)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    criterion_sup = nn.BCEWithLogitsLoss()
    lam_sup = 1.0
    print(f"device {device}")
    print(f"epochs {num_epoch} batches {len(trainloader)} batch_size {batch_size}")
    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(num_epoch):
        print(f"epoch {epoch+1}/{num_epoch}")
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        for step, data in enumerate(trainloader, start=1):
            optimizer.zero_grad()
            x_cont, con_mask = data[0].to(device), data[1].to(device)
            yb = None
            if X_train.get("labels") is not None:
                yb = torch.tensor(data[2], device=device).float()
            x_cont = torch.nan_to_num(x_cont, nan=0.0, posinf=0.0, neginf=0.0)
            x_cont = x_cont.float()
            x_cont = torch.clamp(x_cont, -10.0, 10.0)
            con_mask = con_mask.int()
            x_cont_corr = x_cont.clone().detach()
            if 'cutmix' in pt_aug:
                index = torch.randperm(x_cont.size(0)).to(device)
                lam = 0.1
                corr = torch.from_numpy(np.random.choice(2, x_cont.shape, p=[lam, 1 - lam])).to(device)
                x2 = x_cont[index, :]
                x_cont_corr[corr == 0] = x2[corr == 0]
            x_cont_enc = embed_data_mask(x_cont, con_mask, model)
            x_cont_enc_2 = embed_data_mask(x_cont_corr, con_mask, model)
            loss = 0
            if 'contrastive' in pt_tasks:
                aug1 = model.transformer(x_cont_enc)
                aug2 = model.transformer(x_cont_enc_2)
                aug1 = torch.nan_to_num(aug1, nan=0.0, posinf=0.0, neginf=0.0)
                aug2 = torch.nan_to_num(aug2, nan=0.0, posinf=0.0, neginf=0.0)
                aug1 = (aug1 / (aug1.norm(dim=-1, keepdim=True) + 1e-12)).flatten(1, 2)
                aug2 = (aug2 / (aug2.norm(dim=-1, keepdim=True) + 1e-12)).flatten(1, 2)
                if pt_projhead_style == 'diff':
                    aug1 = model.pt_mlp(aug1)
                    aug2 = model.pt_mlp2(aug2)
                else:
                    aug1 = model.pt_mlp(aug1)
                    aug2 = model.pt_mlp(aug2)
                logits1 = aug1 @ aug2.t() / nce_temp
                logits2 = aug2 @ aug1.t() / nce_temp
                logits1 = torch.nan_to_num(logits1, nan=0.0, posinf=0.0, neginf=0.0)
                logits2 = torch.nan_to_num(logits2, nan=0.0, posinf=0.0, neginf=0.0)
                targets = torch.arange(logits1.size(0)).to(logits1.device)
                loss_1 = criterion1(logits1, targets)
                loss_2 = criterion1(logits2, targets)
                loss = lam0 * (loss_1 + loss_2) / 2
            if 'denoising' in pt_tasks:
                con_outs = model(x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs, dim=1)
                    con_outs = torch.nan_to_num(con_outs, nan=0.0, posinf=0.0, neginf=0.0)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                loss = loss + l2
            if yb is not None:
                enc_main = embed_data_mask(x_cont, con_mask, model)
                z_main = model.transformer(enc_main)
                z_main = z_main.mean(dim=1)
                logits = clf_head(z_main).squeeze(-1)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                ls = criterion_sup(logits, yb)
                loss = loss + lam_sup * ls
            if not torch.isfinite(loss):
                print(f"step {step}/{len(trainloader)} invalid_loss")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * x_cont.size(0)
            epoch_count += x_cont.size(0)
            if step % 50 == 0 or step == len(trainloader):
                print(f"step {step}/{len(trainloader)} loss {loss.item():.4f}")
        if epoch_count > 0:
            avg = epoch_loss/epoch_count
            print(f"epoch_loss {avg:.4f}")
            if best_loss - avg > float(min_delta):
                best_loss = avg
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
                    print("early_stop")
                    break
    return model

def build_variants_df_from_maf(maf_path):
    df = pd.read_csv(maf_path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    chr_col = cols.get("chromosome", "Chromosome")
    pos_col = cols.get("start_position", "Start_Position")
    ref_col = cols.get("reference_allele", "Reference_Allele")
    alt_col = cols.get("tumor_seq_allele2", "Tumor_Seq_Allele2")
    out = pd.DataFrame({
        "chr": df[chr_col].astype(str).str.replace("chr", "", case=False),
        "pos": pd.to_numeric(df[pos_col], errors="coerce").astype(int),
        "ref": df[ref_col].astype(str),
        "alt": df[alt_col].astype(str),
    })
    out = out.dropna().reset_index(drop=True)
    return out

def build_variants_df_from_train_tsv(train_tsv):
    df = pd.read_csv(train_tsv, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for k in cands:
            if k in cols:
                return cols[k]
        raise KeyError(f"missing columns: tried {cands} in {list(df.columns)}")
    chr_col = pick("chr", "chrom", "chromosome")
    pos_col = pick("pos", "position", "start_position", "start")
    ref_col = pick("ref", "reference", "reference_allele")
    alt_col = pick("alt", "alternate", "tumor_seq_allele2", "alt_allele")
    out = pd.DataFrame({
        "chr": df[chr_col].astype(str).str.replace("chr", "", case=False),
        "pos": pd.to_numeric(df[pos_col], errors="coerce").astype(int),
        "ref": df[ref_col].astype(str),
        "alt": df[alt_col].astype(str),
    })
    out = out.dropna().reset_index(drop=True)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maf_path", type=str, default=os.path.join("data", "raw", "pcawg_noncoding_50k.maf"))
    parser.add_argument("--pca_dir", type=str, default=os.path.join("src", "pcawg_features", "PCAWG"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "fea"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--supervised", action="store_true", default=False)
    parser.add_argument("--labels_path", type=str, default=os.path.join("data", "fea", "DNA", "labels.npy"))
    parser.add_argument("--genomic_csv", type=str, default=os.path.join("data", "fea", "DNA", "genomic_data.csv"))
    parser.add_argument("--source", type=str, default="maf", choices=["maf","train"])
    parser.add_argument("--encode_train", action="store_true", default=False)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--encode_only", action="store_true", default=False)
    parser.add_argument("--train_tsv", type=str, default=os.path.join("data", "raw", "train_info_no_varid.tsv"))
    parser.add_argument("--fea_tag", type=str, default="")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.fea_tag:
        args.fea_tag = "maf" if args.source == "maf" else "train"
    if args.source == "maf":
        pcawg_dir, precomp_npy, precomp_ids, legacy_npy, legacy_ids = _pcawg_cache_paths(args.output_dir, args.fea_tag)
        if os.path.exists(precomp_npy) and os.path.exists(precomp_ids):
            X = np.load(precomp_npy).astype("float32")
            with open(precomp_ids, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip()]
        elif os.path.exists(legacy_npy) and os.path.exists(legacy_ids):
            X = np.load(legacy_npy).astype("float32")
            with open(legacy_ids, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            os.makedirs(pcawg_dir, exist_ok=True)
            variants_df = build_variants_df_from_maf(args.maf_path)
            cfg = _build_config_from_dir(args.pca_dir, batch_size=1000)
            feats = compute_pcawg_features_for_variants(variants_df, cfg)
            feats = feats.reset_index()
            ids = feats["VariationID"].astype(str).tolist()
            num_cols = [c for c in feats.columns if c.startswith("pcawg_") and pd.api.types.is_numeric_dtype(feats[c])]
            X = feats[num_cols].to_numpy(dtype="float32")
            np.save(precomp_npy, X.astype("float32"))
            with open(precomp_ids, "w", encoding="utf-8") as f:
                for vid in ids:
                    f.write(str(vid) + "\n")
    else:
        pcawg_dir, precomp_npy, precomp_ids, legacy_npy, legacy_ids = _pcawg_cache_paths(args.output_dir, args.fea_tag)
        if os.path.exists(precomp_npy) and os.path.exists(precomp_ids):
            X = np.load(precomp_npy).astype("float32")
            with open(precomp_ids, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip()]
        elif os.path.exists(legacy_npy) and os.path.exists(legacy_ids):
            X = np.load(legacy_npy).astype("float32")
            with open(legacy_ids, "r", encoding="utf-8") as f:
                ids = [line.strip() for line in f if line.strip()]
        else:
            os.makedirs(pcawg_dir, exist_ok=True)
            variants_df = build_variants_df_from_train_tsv(args.train_tsv)
            cfg = _build_config_from_dir(args.pca_dir, batch_size=1000)
            feats = compute_pcawg_features_for_variants(variants_df, cfg)
            feats = feats.reset_index()
            ids = feats["VariationID"].astype(str).tolist()
            num_cols = [c for c in feats.columns if c.startswith("pcawg_") and pd.api.types.is_numeric_dtype(feats[c])]
            X = feats[num_cols].to_numpy(dtype="float32")
            np.save(precomp_npy, X.astype("float32"))
            with open(precomp_ids, "w", encoding="utf-8") as f:
                for vid in ids:
                    f.write(str(vid) + "\n")
    X_raw = X.copy()
    col_means = np.nanmean(X_raw, axis=0)
    col_stds = np.nanstd(X_raw, axis=0)
    X = (X_raw - col_means) / (col_stds + 1e-6)
    mask = np.ones_like(X_raw, dtype=np.int64)
    mask[np.isnan(X_raw)] = 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = {"data": X, "mask": mask}
    if args.supervised:
        labels = np.load(args.labels_path).reshape(-1).astype("int64")
        g = pd.read_csv(args.genomic_csv)
        g["variant_id"] = g["chr"].astype(str).str.upper().str.replace("CHR","") + ":" + g["pos"].astype(int).astype(str) + ":" + g["ref"].astype(str).str.upper() + ">" + g["alt"].astype(str).str.upper()
        id_to_idx = {vid: i for i, vid in enumerate(g["variant_id"].tolist())}
        y_aligned = []
        for vid in ids:
            j = id_to_idx.get(vid, None)
            y_aligned.append(labels[j] if j is not None else 0)
        X_train["labels"] = np.array(y_aligned, dtype="int64")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAINT(
        num_continuous=X_train["data"].shape[1] + 1,
        dim=32,
        dim_out=1,
        depth=2,
        heads=8,
        attn_dropout=0.5,
        ff_dropout=0.5,
        mlp_hidden_mults=(4, 2),
        cont_embeddings="MLP",
        attentiontype="colrow",
        final_mlp_style="sep",
        y_dim=2
    )
    model.to(device)
    model.float()
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading pretrained SAINT model from {args.ckpt}")
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd)
    elif args.encode_only:
        raise FileNotFoundError("encode_only requires a valid --ckpt path")

    if not args.encode_only:
        model = saint_pretrain(model, X_train, device, num_epoch=args.epochs, batch_size=args.batch_size, patience=args.patience, min_delta=args.min_delta)
        tag = args.fea_tag or ("maf" if args.source == "maf" else "train")
        omics_dir = os.path.join(args.output_dir, "Omics", tag)
        os.makedirs(omics_dir, exist_ok=True)
        ckpt_path = os.path.join(omics_dir, "saint_pretrain.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"saved_checkpoint {ckpt_path}")
    with torch.no_grad():
        ds = DataSetCatCon(X_train)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        outs = []
        for i, data in enumerate(loader, start=1):
            x_cont, con_mask = data[0].to(device), data[1].to(device)
            x_cont = x_cont.float()
            con_mask = con_mask.int()
            enc = embed_data_mask(x_cont, con_mask, model)
            z = model.transformer(enc)
            z = z.mean(dim=1)
            outs.append(z.cpu().numpy())
            if i % 50 == 0 or i == len(loader):
                print(f"export {i}/{len(loader)}")
        embeds = np.concatenate(outs, axis=0).astype("float32")
    tag = args.fea_tag or ("maf" if args.source == "maf" else "train")
    omics_dir = os.path.join(args.output_dir, "Omics", tag)
    os.makedirs(omics_dir, exist_ok=True)
    np.save(os.path.join(omics_dir, "omics_embeddings.npy"), embeds)
    with open(os.path.join(omics_dir, "omics_variant_ids.txt"), "w", encoding="utf-8") as f:
        for vid in ids:
            f.write(str(vid) + "\n")
    if args.encode_train and args.source == "maf":
        print("encode_train_from_pretrained")
        train_tag = "train"
        pcawg_dir, precomp_npy, precomp_ids, legacy_npy, legacy_ids = _pcawg_cache_paths(args.output_dir, train_tag)
        if (not os.path.exists(precomp_npy) or not os.path.exists(precomp_ids)) and os.path.exists(legacy_npy) and os.path.exists(legacy_ids):
            precomp_npy = legacy_npy
            precomp_ids = legacy_ids
        X2 = np.load(precomp_npy).astype("float32")
        with open(precomp_ids, "r", encoding="utf-8") as f:
            ids2 = [line.strip() for line in f if line.strip()]
        mask2 = np.ones_like(X2, dtype=np.int64)
        mask2[np.isnan(X2)] = 0
        X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)
        ds2 = DataSetCatCon({"data": X2, "mask": mask2})
        loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False)
        outs2 = []
        with torch.no_grad():
            for i, data in enumerate(loader2, start=1):
                x_cont, con_mask = data[0].to(device), data[1].to(device)
                x_cont = x_cont.float()
                con_mask = con_mask.int()
                enc = embed_data_mask(x_cont, con_mask, model)
                z = model.transformer(enc)
                z = z.mean(dim=1)
                outs2.append(z.cpu().numpy())
                if i % 50 == 0 or i == len(loader2):
                    print(f"encode_train_export {i}/{len(loader2)}")
        embeds2 = np.concatenate(outs2, axis=0).astype("float32")
        omics_dir2 = os.path.join(args.output_dir, "Omics", train_tag)
        os.makedirs(omics_dir2, exist_ok=True)
        np.save(os.path.join(omics_dir2, "omics_embeddings.npy"), embeds2)
        with open(os.path.join(omics_dir2, "omics_variant_ids.txt"), "w", encoding="utf-8") as f:
            for vid in ids2:
                f.write(str(vid) + "\n")

if __name__ == "__main__":
    main()
