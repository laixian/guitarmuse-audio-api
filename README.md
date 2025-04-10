<<<<<<< HEAD
# Python音频分析API

本项目是一个基于FastAPI的Python音频分析API，使用librosa库进行音频特征提取，并结合大型语言模型(LLM)进行高级音乐结构分析。

## 主要功能

- 音频文件分析
  - 调性检测
  - 速度(BPM)检测
  - 和弦检测
  - 歌曲结构分析
  - 节拍分析
- LLM增强分析
  - 优化和弦进行
  - 改进歌曲结构
  - 基于音乐理论的调性验证

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置LLM API

在使用LLM增强功能前，需要配置OpenAI API密钥：

1. 在`run.sh`中设置环境变量：
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. 或者在运行前设置环境变量：
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   python -m uvicorn app.main:app --reload
   ```

## 启动API服务

### 直接启动

```bash
./run.sh
```

### Docker启动

```bash
docker-compose up --build
```

## API使用

### 音频分析接口

`POST /api/audio-analysis`

**参数**:
- `audio`: 音频文件
- `key`: (可选) 首选调性
- `use_llm`: (可选) 是否使用LLM增强分析，默认使用配置中的设置

**返回示例**:
```json
{
  "result": {
    "key": "C",
    "tempo": 120.5,
    "structures": [
      {
        "type": "Intro",
        "startTime": 0.0,
        "endTime": 12.5,
        "measures": [
          {
            "number": 1,
            "chord": "C",
            "startTime": 0.0,
            "endTime": 2.0
          },
          ...
        ]
      },
      ...
    ],
    "beats": {
      "positions": [0.0, 0.5, 1.0, ...],
      "bpm": 120.5,
      "timeSignature": "4/4",
      "beatsPerBar": 4
    }
  },
  "warning": null
}
```

## LLM增强配置

LLM增强功能提供了更智能的音频分析能力，可通过以下配置项进行控制：

- `APP_ENABLE_LLM_ENHANCEMENT`: 是否启用LLM增强（默认：true）
- `APP_LLM_MODEL`: 使用的模型名称（默认：gpt-4）
- `APP_USE_LLM_CACHE`: 是否启用缓存以减少API调用（默认：true）

## 缓存管理

LLM响应会被缓存到`.llm_cache`目录，如需清除缓存：

```bash
rm -rf .llm_cache
```

## 技术栈

- FastAPI: 高性能API框架
- Librosa: 专业音频分析库
- Pydantic: 数据验证
- Uvicorn: ASGI服务器

## 系统要求

- Python 3.9+
- FFmpeg
- libsndfile

## 安装

### 本地安装

```bash
# 克隆仓库
git clone <repository_url>
cd python-audio-api

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 使用Docker

```bash
# 构建并启动容器
docker-compose up -d
```

## 运行

### 本地运行

```bash
# 启动开发服务器
uvicorn app.main:app --reload
```

### 使用已安装的Docker

```bash
# 如果已经安装了Docker
docker-compose up -d
```

## API文档

启动服务器后，可以在以下URL访问API文档：

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 使用方法

### 分析音频文件

```python
import requests

url = "http://localhost:8000/api/audio-analysis"

# 可选：指定音频调性
data = {"key": "E"}  # 如果不提供，API将自动检测

files = {"audio": open("your_audio_file.mp3", "rb")}

response = requests.post(url, data=data, files=files)
result = response.json()
print(result)
```

## 前端集成

GuitarMuse前端可以通过以下方式与API集成：

```typescript
// 在前端代码中使用Fetch API
async function analyzeAudio(file, key = null) {
  const formData = new FormData();
  formData.append('audio', file);
  
  if (key) {
    formData.append('key', key);
  }
  
  const response = await fetch('http://localhost:8000/api/audio-analysis', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}
```

## 许可证

MIT 
=======
# guitarmuse-audio-api
>>>>>>> c6cd094ec32ca5908a49a471c210d60fae67bd70
