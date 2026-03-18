from datetime import datetime
from tkinter.constants import INSERT
from typing import Dict

import pandas as pd
from fastapi import APIRouter, HTTPException

import pymysql
from backend.db import db
router = APIRouter(prefix="/api")


def get_cursor():
    return db.cursor(pymysql.cursors.DictCursor)


@router.get("/projects", response_model=dict)
def get_projects():
    """
    获取所有项目列表
    数据库列名是 createdAt 和 updatedAt（驼峰命名）
    """
    try:
        cursor = get_cursor()

        cursor.execute('select userId from current')
        result = cursor.fetchone()
        userId = result['userId']
        print(userId)
        # 方法1：使用驼峰命名的列名
        query = """
            SELECT 
                id, 
                name, 
                COALESCE(description, '') as description,
                createdAt,
                updatedAt,
                COALESCE(dataCount, 0) as dataCount
            FROM projects
            WHERE user_id = %s
            ORDER BY updatedAt DESC
        """

        cursor.execute(query,(userId,))
        projects_data = cursor.fetchall()
        cursor.close()

        # 格式化响应数据
        formatted_projects = []
        for project in projects_data:
            # 处理时间字段
            created_at = project.get("createdAt") or ""
            updated_at = project.get("updatedAt") or ""

            # 如果是datetime对象，转换为ISO格式字符串
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()

            formatted_projects.append({
                "id": str(project["id"]),
                "name": project["name"],
                "description": project["description"],
                "createdAt": created_at,
                "updatedAt": updated_at,
                "dataCount": int(project["dataCount"])
            })

        return {
            "success": True,
            "message": f"成功获取 {len(formatted_projects)} 个项目",
            "projects": formatted_projects
        }



    except Exception as e2:
        print(f"备用查询也失败: {e2}")
        return {
            "success": False,
            "message": f"获取项目列表失败: {str(e2)}",
            "projects": []
        }


from datetime import datetime
import json


@router.post("/projects", response_model=dict)
def create_project(data: Dict[str, str]):
    name = data.get("name")
    description = data.get("description")
    cursor = get_cursor()
    cursor.execute('select userId from current')
    result = cursor.fetchone()
    userId = result['userId']

    cursor.execute('SELECT id from projects WHERE name = %s AND user_id = %s', (name, userId,))
    if cursor.fetchone() is None:

        current_time = datetime.now()

        cursor.execute(
            'INSERT INTO projects (name, description, dataCount, user_id, createdAt, updatedAt, complete) VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (name, description, 0, userId, current_time, current_time, 0))
        db.commit()

        # 获取刚插入的项目ID
        project_id = cursor.lastrowid

        # 将datetime对象转换为字符串
        time_str = current_time.isoformat()

        return {
            "success": True,
            "message": "创建成功",
            "project": {
                "id": project_id,  # 使用正确的ID
                "name": name,
                "description": description,
                "createdAt": time_str,
                "updatedAt": time_str,
                "dataCount": 0
            }
        }
    else:
        return {
            "success": False,
            "message": "重复的名称"
        }

@router.delete("/projects/{id}")
def delete_project(id: str):
    cursor = get_cursor()

    cursor.execute('select userId from current')
    result = cursor.fetchone()
    userId = result['userId']
    cursor.execute('DELETE FROM projects WHERE id = %s AND user_id = %s', (id, userId,))
    db.commit()
    return {
        "success": True,
        "message": "删除成功"
        }


@router.get("/projects/{id}/status")
def get_project_status(id: str):
    cursor = get_cursor()
    try:
        # 1. 获取当前登录用户 ID
        cursor.execute('SELECT userId FROM current LIMIT 1')
        user_result = cursor.fetchone()
        if not user_result:
            return {"success": False, "message": "用户未登录", "isCompleted": False}

        userId = user_result['userId']

        # 2. 查询项目状态
        # 建议加上 user_id 校验，确保用户只能查询自己的项目
        cursor.execute('SELECT complete FROM projects WHERE id = %s AND user_id = %s', (id, userId))
        project_result = cursor.fetchone()

        # 3. 检查项目是否存在
        if not project_result:
            return {
                "success": False,
                "isCompleted": False,
                "message": "找不到该项目"
            }



        is_done = False
        if project_result and project_result['complete']:
            # 兼容 1, "1", True 等情况
            if str(project_result['complete']) in ['1', 'True']:
                is_done = True

        # 更新当前项目 ID
        cursor.execute('DELETE FROM currentproject')
        cursor.execute('INSERT INTO currentproject (id) VALUES (%s)', (id,))
        db.commit()

        return {
            "success": True,
            "isCompleted": is_done,  # 确保这里传给前端的是 True 或 False
            "status": "completed" if is_done else "processing",
            "message": "项目已就绪" if is_done else "项目分析尚未完成"
        }
    except Exception as e:
        print(f"数据库错误: {e}")
        return {"success": False, "message": str(e), "isCompleted": False}

@router.get("/projects/{projectId}/results")
def get_project_results(projectId: str):
    cursor = get_cursor()
    cursor.execute('select result from projects WHERE id = %s', (projectId,))
    result = cursor.fetchone()
    if not result:
        return {
            "success": False,
            "message": "未查询到结果"
        }
    df = pd.read_csv(result['result'], sep='\t', header=None)

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