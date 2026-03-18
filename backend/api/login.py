from typing import Dict
from fastapi import APIRouter, HTTPException

import pymysql
from backend.db import db

router = APIRouter(prefix="/api/auth")
cursor = db.cursor(pymysql.cursors.DictCursor)  # 使用DictCursor获取字典格式



@router.post("/login")
def login(data: Dict[str, str]):
    try:
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            raise HTTPException(status_code=400, detail="邮箱和密码不能为空")

        # 查询用户
        cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
        result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code=401, detail="用户不存在")

        # 验证密码（如果数据库存储的是加密密码）
        # 假设数据库密码是明文存储，实际情况应该加密
        stored_password = result.get("password")

        # 如果是明文存储
        if stored_password != password:
            # 如果是加密存储，使用bcrypt检查
            # if not bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            raise HTTPException(status_code=401, detail="密码错误")

        # 获取用户ID
        userId = result.get("id")

        # 将用户ID插入current表
        try:
            # 先删除旧的记录（如果有的话）
            cursor.execute("DELETE FROM current WHERE userId = %s", (userId,))
            cursor.execute("INSERT INTO current (userId) VALUES (%s)", (userId,))

            db.commit()
            print(f"成功插入current表: userId={userId}")
        except Exception as db_error:
            db.rollback()
            print(f"插入current表失败: {db_error}")
            # 不返回错误，继续登录流程



        return {
            "success": True,
            "message": "登录成功",
            "token": "1234",  # 应该生成真正的token
            "user": {
                "email": email
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@router.post("/register")
def register(data: Dict[str, str]):
    try:
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            raise HTTPException(status_code=400, detail="邮箱和密码不能为空")

        # 检查邮箱是否已存在
        cursor.execute("SELECT id FROM user WHERE email = %s", (email,))
        if cursor.fetchone() is not None:
            raise HTTPException(status_code=400, detail="邮箱已存在")

        # 对密码进行加密（推荐）
        # salt = bcrypt.gensalt()
        # hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        # 暂时先使用明文，后续可以改为加密


        cursor.execute(
            "INSERT INTO user (email, password) VALUES (%s, %s)",
            (email, password)  # 这里password应该是hashed_password.decode('utf-8')
        )

        # 必须提交事务
        db.commit()

        # 获取插入的用户ID
        user_id = cursor.lastrowid

        return {
            "success": True,
            "message": "注册成功",
            "user_id": user_id
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()  # 回滚事务
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")