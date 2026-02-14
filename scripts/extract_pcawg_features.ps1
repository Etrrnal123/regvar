# RegVAR PCAWG 特征提取脚本
# 用法: ./scripts/extract_pcawg_features.ps1

param(
    [string]$GenomicTSV = ".\data\raw\train_info.tsv",
    [string]$PCAWG_DIR = ".\src\pcawg_features\PCAWG",
    [string]$OutputDir = ".\data\fea\PCAWG",
    [int]$Epochs = 200,
    [int]$BatchSize = 128,
    [string]$Ckpt = ".\data\fea\saint_pretrain.pth",
    [switch]$UsePretrained,
    [switch]$ExportPCAWGTSV
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
cd $ProjectRoot

python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) { echo "ERROR: torch not available. Please run: conda activate regvar"; exit 1 }

$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($GenomicTSV)
if (-not (Test-Path $PCAWG_DIR)) {
    echo "ERROR: PCAWG resource directory not found: $PCAWG_DIR"
    echo "Please set -PCAWG_DIR to your PCAWG data directory (e.g., .\src\pcawg_features\PCAWG)."
    exit 1
}

$PcawgSubdir = Join-Path $OutputDir $BaseName
$PcawgNpy = Join-Path $PcawgSubdir ($BaseName + "_pcawg_features.npy")
$PcawgIds = Join-Path $PcawgSubdir ($BaseName + "_pcawg_variant_ids.txt")
$PcawgTsv = Join-Path $PcawgSubdir ($BaseName + "_pcawg_features.tsv")

if (-not (Test-Path $PcawgSubdir)) {
    New-Item -ItemType Directory -Force -Path $PcawgSubdir | Out-Null
}

if (-not (Test-Path $PcawgNpy) -or -not (Test-Path $PcawgIds)) {
    echo "PCAWG cache not found; generating PCAWG features first..."
    python -m src.pcawg_features.processor --input $GenomicTSV --output $PcawgTsv --pca_dir $PCAWG_DIR --batch_size 1000 --prefix $BaseName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

echo "Generating omics embeddings using pretrained SAINT model..."
if (Test-Path $Ckpt) {
    python -m models.tab_pretrain.saint_pretrain --source train --train_tsv $GenomicTSV --pca_dir $PCAWG_DIR --output_dir ".\data\fea" --batch_size $BatchSize --fea_tag $BaseName --encode_only --ckpt $Ckpt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    echo "ERROR: Pretrained SAINT model not found at $Ckpt. Training from scratch is disabled."
    exit 1
}

if ($ExportPCAWGTSV) {
    echo "Exporting PCAWG feature table (this will compute PCAWG features for TSV export)..."
    python -m src.pcawg_features.processor --input $GenomicTSV --output $PcawgTsv --pca_dir $PCAWG_DIR --batch_size 1000 --prefix $BaseName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

echo "Omics embedding completed!"
