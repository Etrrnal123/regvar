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

class DataSetCatCon(Dataset):
    def __init__(self, X):
        self.X2 = X['data'].astype(np.float32)
        self.X2_mask = X['mask'].astype(np.int64)
        self.cls = np.zeros((self.X2.shape[0], 1), dtype=int)
        self.cls_mask = np.ones((self.X2.shape[0], 1), dtype=int)
    def __len__(self):
        return self.X2.shape[0]
    def __getitem__(self, idx):
        return np.concatenate((self.cls[idx], self.X2[idx])), np.concatenate((self.cls_mask[idx], self.X2_mask[idx]))

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
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    best_loss = float("inf")
    bad_epochs = 0
    for epoch in range(num_epoch):
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            x_cont, con_mask = data[0].to(device), data[1].to(device)
            x_cont = x_cont.float()
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
                aug1 = (aug1 / aug1.norm(dim=-1, keepdim=True)).flatten(1, 2)
                aug2 = (aug2 / aug2.norm(dim=-1, keepdim=True)).flatten(1, 2)
                if pt_projhead_style == 'diff':
                    aug1 = model.pt_mlp(aug1)
                    aug2 = model.pt_mlp2(aug2)
                else:
                    aug1 = model.pt_mlp(aug1)
                    aug2 = model.pt_mlp(aug2)
                logits1 = aug1 @ aug2.t() / nce_temp
                logits2 = aug2 @ aug1.t() / nce_temp
                targets = torch.arange(logits1.size(0)).to(logits1.device)
                loss_1 = criterion1(logits1, targets)
                loss_2 = criterion1(logits2, targets)
                loss = lam0 * (loss_1 + loss_2) / 2
            if 'denoising' in pt_tasks:
                con_outs = model(x_cont_enc_2)
                if len(con_outs) > 0:
                    con_outs = torch.cat(con_outs, dim=1)
                    l2 = criterion2(con_outs, x_cont)
                else:
                    l2 = 0
                loss = loss + l2
            loss.backward()
            optimizer.step()
        if trainloader.dataset is not None and len(trainloader) > 0:
            avg = 0.0
            # approximate using last batch loss as proxy
            avg = float(loss.item())
            if best_loss - avg > float(min_delta):
                best_loss = avg
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(patience):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maf_path", type=str, default=os.path.join("data", "raw", "pcawg_noncoding_50k.maf"))
    parser.add_argument("--pca_dir", type=str, default=os.path.join("src", "pcawg_features", "PCAWG"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("data", "fea"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pcawg_dir = os.path.join(args.output_dir, "PCAWG")
    os.makedirs(pcawg_dir, exist_ok=True)
    precomp_npy = os.path.join(pcawg_dir, "pcawg_features.npy")
    precomp_ids = os.path.join(pcawg_dir, "pcawg_variant_ids.txt")
    if os.path.exists(precomp_npy) and os.path.exists(precomp_ids):
        X = np.load(precomp_npy).astype("float32")
        with open(precomp_ids, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        variants_df = build_variants_df_from_maf(args.maf_path)
        cfg = _build_config_from_dir(args.pca_dir, batch_size=1000)
        feats = compute_pcawg_features_for_variants(variants_df, cfg)
        feats = feats.reset_index()
        ids = feats["VariationID"].astype(str).tolist()
        num_cols = [c for c in feats.columns if c.startswith("pcawg_") and pd.api.types.is_numeric_dtype(feats[c])]
        X = feats[num_cols].to_numpy(dtype="float32")
    X_raw = X.copy()
    col_means = np.nanmean(X_raw, axis=0)
    col_stds = np.nanstd(X_raw, axis=0)
    X = (X_raw - col_means) / (col_stds + 1e-6)
    mask = np.ones_like(X_raw, dtype=np.int64)
    mask[np.isnan(X_raw)] = 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = {"data": X, "mask": mask}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAINT(
        num_continuous=X_train["data"].shape[1] + 1,
        dim=4,
        dim_out=1,
        depth=1,
        heads=4,
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
    model = saint_pretrain(model, X_train, device, num_epoch=args.epochs, batch_size=args.batch_size)
    with torch.no_grad():
        ds = DataSetCatCon(X_train)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        outs = []
        for data in loader:
            x_cont, con_mask = data[0].to(device), data[1].to(device)
            x_cont = torch.nan_to_num(x_cont, nan=0.0, posinf=0.0, neginf=0.0).float()
            x_cont = torch.clamp(x_cont, -10.0, 10.0)
            con_mask = con_mask.int()
            enc = embed_data_mask(x_cont, con_mask, model)
            z = model.transformer(enc)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z = z.mean(dim=1)
            outs.append(z.cpu().numpy())
        embeds = np.concatenate(outs, axis=0).astype("float32")
    omics_dir = os.path.join(args.output_dir, "Omics")
    os.makedirs(omics_dir, exist_ok=True)
    np.save(os.path.join(omics_dir, "omics_embeddings.npy"), embeds)
    with open(os.path.join(omics_dir, "omics_variant_ids.txt"), "w", encoding="utf-8") as f:
        for vid in ids:
            f.write(str(vid) + "\n")

if __name__ == "__main__":
    main()
