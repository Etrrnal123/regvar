import subprocess
from pathlib import Path

def run_pcawg_and_omics():
    cmd = [
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", "C:/Users/fs201/Downloads/RegVAR/scripts/extract_pcawg_features.ps1",
        "-ExportPCAWGTSV"
    ]
    subprocess.run(cmd, check=True)

def run_dna_features(seq_tsv='C:/Users/fs201/Downloads/RegVAR/data/raw/train_seq.tsv', info_tsv ='C:/Users/fs201/Downloads/RegVAR/data/raw/train_info.tsv', config='C:/Users/fs201/Downloads/RegVAR/config.json'):
    cmd = [
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", "C:/Users/fs201/Downloads/RegVAR/scripts/extract_features.ps1",
        "-SequenceTSV", seq_tsv,
        "-GenomicTSV", info_tsv,
        "-Config", config
    ]
    subprocess.run(cmd, check=True)

def run_alphagenome_features(input_tsv: str='C:/Users/fs201/Downloads/RegVAR/data/raw/train_info.tsv', config: str='C:/Users/fs201/Downloads/RegVAR/config.json'):
    """
    调用 AlphaGenome 特征预计算脚本
    产物位于 data/fea/AlphaGenome/<tag>/
    """
    cmd = [
        "powershell",
        "-ExecutionPolicy", "Bypass",
        "-File", "C:/Users/fs201/Downloads/RegVAR/scripts/precompute_alphagenome.ps1",
        "-InputTSV", input_tsv,
        "-Config", config
    ]
    subprocess.run(cmd, check=True)

def extract_features():
    run_dna_features()