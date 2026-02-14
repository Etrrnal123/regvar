# RegVAR AlphaGenome 特征预计算脚本
# 用法: .\precompute_alphagenome.ps1

param(
    [string]$InputTSV = "C:\Users\fs201\Downloads\RegVAR\data\raw\train_info.tsv",
    [string]$OutputRoot = ".\data\fea\AlphaGenome",
    [string]$Config = ".\config.json"
)

# 设置代理环境变量（如果需要）
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"

# 激活conda环境
conda activate regvar

# 从配置文件读取API密钥
$ConfigContent = Get-Content $Config | ConvertFrom-Json
$ApiKey = $ConfigContent.alphagenome_api_key

# 创建输出目录
$BaseName = [System.IO.Path]::GetFileNameWithoutExtension($InputTSV)
$OutDir = Join-Path $OutputRoot $BaseName
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}
$OutputParquet = Join-Path $OutDir ($BaseName + "_alphagenome.parquet")
$FailuresTSV = Join-Path $OutDir ($BaseName + "_alphagenome_failures.tsv")

# 预计算 AlphaGenome 特征
echo "Precomputing AlphaGenome features..."
python -m src.ag_batch.precompute --input-tsv $InputTSV --output-parquet $OutputParquet --failures-tsv $FailuresTSV --api-key "$ApiKey" --scorers RNA_SEQ CAGE PROCAP DNASE ATAC CHIP_HISTONE CHIP_TF SPLICE_SITES SPLICE_JUNCTIONS SPLICE_SITE_USAGE CONTACT_MAPS --max-workers 8 --rate-limit 2.0

echo "AlphaGenome feature precomputation completed!"
