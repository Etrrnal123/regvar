# RegVAR

RegVAR 是一个三模态变异致病性预测管线：将 DNA 序列特征（NT）、AlphaGenome 功能组学特征（AG）与 PCAWG 癌症组学衍生表格特征（PCAWG/Omics）对齐到同一变异 ID 上，然后训练融合模型输出致病性概率。

本仓库默认按训练集 `data/raw/train_info.tsv` 与 `data/raw/train_seq.tsv` 组织数据，并将所有特征产物落在 `data/fea/` 下。

## 模态与产物

- DNA/NT：从 `ref_seq/alt_seq` 提取序列特征，产物在 `data/fea/DNA/<tag>/`
- AlphaGenome：调用 AlphaGenome API 批量打分，产物在 `data/fea/AlphaGenome/<tag>/`
- 第三模态（两种方式二选一）
  - PCAWG 数值特征：产物 `data/fea/PCAWG/*`
  - Omics（推荐）：用预训练 SAINT 将 PCAWG 数值特征编码成嵌入，产物 `data/fea/Omics/<tag>/omics_embeddings.npy`

## 环境与安装

推荐使用 conda 环境名 `regvar`。

```powershell
conda create -n regvar python=3.11 -y
conda activate regvar
pip install -r requirements.txt
pip install -e .\models\alphagenome
```

说明：
- `requirements.txt` 固定了依赖版本，其中 `torch` 为 CUDA 版本；如你的机器无 GPU，可改装 CPU 版 torch。
- 若 `pip install -r requirements.txt` 提示找不到 `torch==...+cu128`，按 PyTorch 官方安装命令先安装 torch/torchvision/torchaudio，再重新执行 requirements（或从 requirements 中移除 torch 三行）。
- AlphaGenome 代码以子目录形式放在 `models/alphagenome/`，建议用 `pip install -e` 方式安装。

## 数据准备

最小可跑通训练需要两份 TSV：

- `data/raw/train_info.tsv`：至少包含列 `chr pos ref alt label`
- `data/raw/train_seq.tsv`：至少包含列 `ref_seq alt_seq label`

变异 ID 在代码内部会统一为 `chr:pos:REF>ALT`，同时兼容 `chr_pos_REF_ALT` 等形式。

## 快速流程（从零生成 PCAWG + Omics）

确保 `data/fea/saint_pretrain.pth` 存在（SAINT 预训练权重），然后运行：

```powershell
conda activate regvar
.\scripts\extract_pcawg_features.ps1
```

默认行为（当 `data/fea/PCAWG/<tag>/` 与 `data/fea/Omics/<tag>/` 为空时）：
- 先生成 PCAWG 缓存：
  - `data/fea/PCAWG/train_info/train_info_pcawg_features.npy`
  - `data/fea/PCAWG/train_info/train_info_pcawg_variant_ids.txt`
- 再用 SAINT 输出 Omics：
  - `data/fea/Omics/train_info/omics_embeddings.npy`
  - `data/fea/Omics/train_info/omics_variant_ids.txt`

可选：导出可读的 PCAWG TSV（会额外计算一次 PCAWG，用于人工检查）

```powershell
.\scripts\extract_pcawg_features.ps1 -ExportPCAWGTSV
```

## 生成 DNA/NT 特征

```powershell
conda activate regvar
.\scripts\extract_features.ps1 -SequenceTSV .\data\raw\train_seq.tsv -GenomicTSV .\data\raw\train_info.tsv -Config .\config.json
```

产物位于 `data/fea/DNA/train_info/`（包含 `*_features.npy`、`*_labels.npy`、`*_genomic.csv` 等）。

## 预计算 AlphaGenome 特征（可选）

需要 AlphaGenome API Key。

```powershell
conda activate regvar
.\scripts\precompute_alphagenome.ps1 -InputTSV .\data\raw\train_info.tsv -Config .\config.json
```

产物位于 `data/fea/AlphaGenome/train_info/`。

## 训练（K 折）

训练入口是 [task.py](file:///e:/compare/RegVAR/src/task.py)：

```powershell
conda activate regvar
python .\src\task.py --config .\config.json
```

输出：
- `output/best_model_fold{K}.pth`
- `output/fold{K}_results.tsv`

## 评估（单 checkpoint）

在 `config.json` 中设置：

- `eval_only: true`
- `checkpoint: "./output/best_model_fold1.pth"`

然后运行：

```powershell
conda activate regvar
python .\src\task.py --config .\config.json
```

## 配置说明（config.json）

常用字段（非完整列表）：

- `fea_dir`：特征目录（默认 `./data/fea`）
- `raw_dir`：原始数据目录（默认 `./data/raw`）
- `output_dir`：模型输出目录（默认 `./output`）
- `use_omics_encoder`：是否用 Omics 替代 PCAWG（推荐 `true`）
- `omics_batch_size`：SAINT 编码批大小
- `k_folds / epochs / lr / batch_size`：训练超参
- `alphagenome_api_key`：AlphaGenome API key（建议只在本地保存，不要提交）

## 目录结构

```
RegVAR/
  src/                     # 训练与特征对齐主逻辑
  models/tab_pretrain/     # SAINT（PCAWG->Omics）
  models/NT_model/         # NT 序列模型与权重
  models/alphagenome/      # AlphaGenome SDK（本地 vendoring）
  scripts/                 # PowerShell 一键脚本
  data/raw/                # 输入 TSV / PCAWG 资源 / MAF
  data/fea/                # 特征缓存与嵌入
  output/                  # 训练输出
```
