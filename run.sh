#!/bin/bash

# 激活虚拟环境
if [ -d "venv" ]; then
  echo "激活虚拟环境..."
  source venv/bin/activate
else
  echo "未找到虚拟环境，安装依赖..."
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
fi

# 设置环境变量
export PYTHONPATH=$(pwd)
export LOG_LEVEL=INFO
export ENABLE_LLM_ENHANCEMENT=false

# Music AI相关配置 - 请替换为您的实际API密钥
export MUSIC_AI_API_KEY="94398d48-f56c-48a1-9cd2-cf3c6eea78af"
export MUSIC_AI_API_URL="https://api.music.ai"

# 启动应用
echo "启动音频分析API服务..."
python -m app.main
