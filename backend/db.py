import pymysql
# 创建数据库连接
db = pymysql.connect(
   host="localhost",
   user="root",
   password="1234",
   database="regvar"
)
print("数据库连接成功!")