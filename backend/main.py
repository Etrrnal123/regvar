import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api import features
import predict

app = FastAPI(title="非编码突变风险评估系统")
app.mount("/static", StaticFiles(directory=r"C:\Users\fs201\Downloads\RegVAR\backend"), name="umap_labels.png")
app.add_middleware(
    CORSMiddleware,

    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(features.router)
app.include_router(predict.router)

# 如果需要，在文件最后添加以下代码
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
