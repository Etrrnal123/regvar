import time

import pandas as pd
from fastapi import APIRouter, UploadFile, File


from backend.services.dna_umap import dnaumap
from backend.services.feature_runner import (
    run_pcawg_and_omics,
    run_dna_features,
    run_alphagenome_features
)

router = APIRouter(prefix="/api/features")

@router.post("/extract")
async def extract_features(trainInfoFile: UploadFile = File(...),
        trainSeqFile: UploadFile = File(...)):
    text = await trainInfoFile.read()

    run_dna_features()
    return {"success":True}

@router.get("/alphagenome/index/{index}")
def generate_alphagenome(index: int):
    try:
        # run_alphagenome_features()
        time.sleep(1)
    except Exception as e:
        return {"success": False, "error": str(e)}
    data = pd.read_parquet(
        r"C:\Users\fs201\Downloads\RegVAR\data\fea\AlphaGenome\train_info\train_info_alphagenome.parquet")
    
    data.fillna(0, inplace=True)
    data = data.iloc[index].to_dict()
    data["id"] = index
    data["category"] = "alphagenome"
    return {"success":True,"message":"获取成功","feature":data}

@router.get("/dna")
def generate_dna():
    try:
        time.sleep(1)
        # dnaumap()
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"success":True,"message":"DNA特征图片","imageUrl":r"http://localhost:8000/static/umap_labels.png"}

@router.get("/pcawg/index/{index}")
def generate_pcawg(index: int):
    try:
        # run_pcawg_and_omics()
        time.sleep(2)
    except Exception as e:
        return {"success": False, "error": str(e)}
    data = pd.read_csv(r'C:\Users\fs201\Downloads\RegVAR\data\fea\PCAWG\train_info\train_info_pcawg_features.tsv', sep='\t')
    data.fillna(0, inplace=True)
    data=data.iloc[index].to_dict()
    data["id"] = index
    data["category"]="pcawg"

    return {"success":True,"message":"获取成功","feature":data}

@router.get("/counts")
def totalcount():
    dna = 1
    pcawg = pd.read_csv(r'C:\Users\fs201\Downloads\RegVAR\data\fea\PCAWG\train_info\train_info_pcawg_features.tsv', sep='\t').shape[0]
    alphagenome = pd.read_parquet(
        r"C:\Users\fs201\Downloads\RegVAR\data\fea\AlphaGenome\train_info\train_info_alphagenome.parquet").shape[0]
    return{"success": True,"message": "成功获取总数",
    "counts":{
    "dna": dna,
    "alphagenome": alphagenome,
    "pcawg": pcawg,
    }
}