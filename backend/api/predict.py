# import subprocess
#
# from fastapi import APIRouter
# from pandas import DataFrame
#
# router = APIRouter(prefix="/api/model")
# # python .\src\task.py --config .\config.json
# def predict():
#     cmd = [
#         "python",
#         r"C:\Users\fs201\Downloads\RegVAR\src\task.py",
#         "--config", r"C:\Users\fs201\Downloads\RegVAR\config.json"
#     ]
#     subprocess.run(cmd, check=True)
#
# @router.post("/inference")
# def inference():
#     predict()
#     data = DataFrame.from_csv(r"output/eval_results.tsv")
#
