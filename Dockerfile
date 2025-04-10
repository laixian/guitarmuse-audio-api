FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY ./app ./app

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV APP_DEBUG=False
ENV APP_FRONTEND_URL=http://localhost:3000

# 启动应用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 