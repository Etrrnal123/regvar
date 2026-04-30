import os
import shutil
import time

import pandas as pd
from fastapi import APIRouter, UploadFile, File
from starlette.responses import JSONResponse

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
    # 指定保存目录
    save_dir = "C:\\Users\\fs201\\Downloads\\RegVAR\\backend\\data\\raw"
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件保存路径
    trainInfo_path = os.path.join(save_dir, trainInfoFile.filename)
    trainSeq_path = os.path.join(save_dir, trainSeqFile.filename)

    try:
        # 保存 trainInfoFile
        trainInfoFile.file.seek(0)
        with open(trainInfo_path, "wb") as buffer:
            shutil.copyfileobj(trainInfoFile.file, buffer)

        # 保存 trainSeqFile
        with open(trainSeq_path, "wb") as buffer:
            shutil.copyfileobj(trainSeqFile.file, buffer)

        # 调用处理函数
        run_dna_features()

        return {"success": True, "message": "文件保存成功"}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        # 关闭文件流
        await trainInfoFile.close()
        await trainSeqFile.close()

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
        dnaumap()
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