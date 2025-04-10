from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.api.endpoints import analysis
from app.core.config import settings
from app.core.logging import get_logger

# 创建logger
logger = get_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="GuitarMuse音频分析API",
    description="基于Librosa的音频分析API，为GuitarMuse提供调性和速度分析、歌曲结构分析和和弦检测功能",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 配置CORS允许前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(analysis.router)

@app.get("/", tags=["健康检查"])
async def root():
    """API根路径，用于健康检查"""
    return {"status": "online", "message": "GuitarMuse音频分析API正在运行"}

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    logger.info("GuitarMuse音频分析API启动成功")

# 应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("GuitarMuse音频分析API关闭")

# 直接运行应用
if __name__ == "__main__":
    logger.info("使用uvicorn启动应用")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 