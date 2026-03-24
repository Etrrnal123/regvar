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
        from backend.db import db
        def get_cursor():
            return db.cursor(pymysql.cursors.DictCursor)

        cursor = get_cursor()

        # 1. 上下文获取
        cursor.execute('SELECT userId FROM current LIMIT 1')
        user_result = cursor.fetchone()
        if not user_result:
            return {"success": False, "message": "用户未登录"}

        cursor.execute('SELECT id FROM currentproject LIMIT 1')
        project_result = cursor.fetchone()
        if not project_result:
            return {"success": False, "message": "未找到项目"}
        projectId = project_result['id']

        # 2. 读取 train_info 构建过滤标准
        # 建议路径从数据库动态获取，这里保留你的默认路径
        train_info_path = r"C:\Users\fs201\Downloads\RegVAR\backend\data\raw\train_info.tsv"
        info_df = pd.read_csv(train_info_path, sep=None, engine='python')
        info_df.columns = [c.lower().strip() for c in info_df.columns]

        # 构建合法组合的唯一标识集合 (Chr:Pos:Ref:Alt)
        # 这样比字符串拼接后再 in 判断更快
        info_df['match_key'] = (
                info_df['chr'].astype(str) + ":" +
                info_df['pos'].astype(str) + ":" +
                info_df['ref'].astype(str) + ":" +
                info_df['alt'].astype(str)
        )
        valid_keys = set(info_df['match_key'])

        # 3. 读取模型生成的原始全量结果
        source_path = r'C:\Users\fs201\Downloads\RegVAR\output\eval_results.tsv'
        # 假设原始文件没有表头，列索引为 0:Chr, 1:Pos, 2:Ref, 3:Alt, 5:Score
        raw_df = pd.read_csv(source_path, sep='\t', header=None)

        # 4. 【核心修改】在保存文件前进行 Pandas 级别过滤
        # 创建临时匹配列
        raw_df['tmp_key'] = (
                raw_df[0].astype(str) + ":" +
                raw_df[1].astype(str) + ":" +
                raw_df[2].astype(str) + ":" +
                raw_df[3].astype(str)
        )

        # 仅保留在 valid_keys 中的行
        filtered_df = raw_df[raw_df['tmp_key'].isin(valid_keys)].copy()

        # 移除临时列
        filtered_df.drop(columns=['tmp_key'], inplace=True)

        # 5. 将过滤后的“干净”数据持久化到磁盘
        random_name = f"filtered_results_{uuid.uuid4().hex[:8]}.tsv"
        dest_path = os.path.join(r"C:\Users\fs201\Downloads\RegVAR\backend\data", random_name)
        # 保存为不带表头的 tsv
        filtered_df.to_csv(dest_path, sep='\t', index=False, header=False)

        # 6. 更新数据库
        rows = len(filtered_df)
        sql = "UPDATE projects SET updatedAt = %s, dataCount = %s, result = %s, complete = %s WHERE id = %s"
        cursor.execute(sql, (datetime.now(), rows, dest_path, 1, projectId))
        db.commit()

        # 7. 格式化返回给前端的 JSON
        def safe_float(x):
            try:
                return float(x)
            except:
                return 0.5

        results = []
        for _, row in filtered_df.iterrows():
            results.append({
                "id": f"{row[0]}:{row[1]}_{row[2]}>{row[3]}",
                "riskScore": safe_float(row[5])
            })

        return {
            "success": True,
            "message": f"分析完成。原始 {len(raw_df)} 条，过滤并保存 {len(results)} 条匹配数据",
            "results": results
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # 方便调试
        return {"success": False, "message": f"处理错误: {str(e)}", "results": []}


