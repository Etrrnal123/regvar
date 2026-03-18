import os
import shutil
import subprocess
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import pymysql
from fastapi import APIRouter
from pandas import DataFrame

router = APIRouter(prefix="/api/model")
# python .\src\task.py --config .\config.json
def predict():
    cmd = [
        "python",
        r"C:\Users\fs201\Downloads\RegVAR\src\task.py",
        "--config", r"C:\Users\fs201\Downloads\RegVAR\config.json"
    ]
    subprocess.run(cmd, check=True)


@router.post("/inference")
def inference():
    try:
        # 读取TSV文件
        df = pd.read_csv(r'C:\Users\fs201\Downloads\RegVAR\output\eval_results.tsv', sep='\t', header=None)
        source_path = r"C:\Users\fs201\Downloads\RegVAR\output\eval_results.tsv"
        rows = len(df)
        # 生成随机文件名
        random_name = f"eval_results_{uuid.uuid4().hex[:8]}.tsv"  # 取前8位UUID
        # 目标路径（复制到data）
        dest_path = os.path.join(r"C:\Users\fs201\Downloads\RegVAR\backend\data", random_name)

        # 复制文件
        shutil.copy2(source_path, dest_path)
        from backend.db import db
        def get_cursor():
            return db.cursor(pymysql.cursors.DictCursor)

        cursor = get_cursor()
        cursor.execute('SELECT userId FROM current LIMIT 1')
        user_result = cursor.fetchone()
        if not user_result:
            return {"success": False, "message": "用户未登录"}

        userId = user_result['userId']
        cursor.execute('SELECT id FROM currentproject LIMIT 1')
        project_result = cursor.fetchone()
        projectId = project_result['id']
        sql = "UPDATE projects SET updatedAt = %s, dataCount = %s, result = %s, complete = %s WHERE id = %s"
        params = (datetime.now(), rows, dest_path, 1,projectId, )
        cursor.execute(sql, params)
        db.commit()


        # 安全地处理第5列（索引5）
        def safe_float_conversion(x):
            try:
                # 如果是字符串'Score'，返回默认值
                if isinstance(x, str) and x.strip().lower() == 'score':
                    return 0.5
                return float(x)
            except (ValueError, TypeError):
                return 0.5

        # 应用安全转换
        df[5] = df[5].apply(safe_float_conversion)

        # 创建合并ID
        df['final'] = (
                df[0].astype(str) + ':' +
                df[1].astype(str) + '_' +
                df[2].astype(str) + '>' +
                df[3].astype(str)
        )

        # 构建结果
        results = []
        for _, row in df.iloc[1:].iterrows():
            result_item = {
                "id": str(row['final']),
                "riskScore": float(row[5])  # 现在这里已经是安全的float值
            }
            results.append(result_item)

        final_result = {
            "success": True,
            "message": f"成功处理 {len(results)} 条数据",
            "results": results
        }
        return final_result

    except Exception as e:
        return {
            "success": False,
            "message": f"处理错误: {str(e)}",
            "results": []
        }


