# RegVAR 特征提取脚本
# 用法: ./scripts/extract_features.ps1

param(
    [string]$SequenceTSV = ".\data\raw\train_seq.tsv",
    [string]$GenomicTSV = ".\data\raw\train_info.tsv",
    [string]$OutDir = ".\data\fea\DNA",
    [string]$Config = ".\config.json"
)

# 激活conda环境
conda activate regvar

# 提取特征
echo "Extracting NT features..."
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($GenomicTSV)
$TargetDir = Join-Path $OutDir $BaseName
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
}
python C:/Users/fs201/Downloads/RegVAR/src/fea_extract.py --input_tsv $SequenceTSV --genomic_tsv $GenomicTSV --config $Config --out_dir $TargetDir --prefix $BaseName

echo "Feature extraction completed!"
