import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from app.models.schemas import AnalysisResponse
from app.services.audio_analyzer import analyze_audio_file
from app.core.logging import get_logger
from app.core.config import settings

# 创建logger
logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["audio"])

# 存储进行中的分析任务状态
analysis_tasks = {}

@router.post("/audio-analysis", response_model=AnalysisResponse)
async def analyze_audio_file_endpoint(
    audio: UploadFile = File(...),
    key: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    分析上传的音频文件，检测调性、速度和结构
    使用Music AI第三方服务进行高质量分析
    """
    
    logger.info(f"收到音频分析请求: 文件={audio.filename}, 大小={audio.size}, 类型={audio.content_type}, 请求调性={key}")
    
    # 验证文件格式
    if not audio.content_type.startswith("audio/"):
        logger.warning(f"文件格式验证失败: {audio.content_type}")
        raise HTTPException(
            status_code=400, 
            detail="上传的文件不是有效的音频格式"
        )
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as temp_file:
        try:
            logger.info(f"保存临时文件: {temp_file.name}")
            # 将上传的文件内容写入临时文件
            contents = await audio.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
            
            # 分析音频
            logger.info(f"开始使用Music AI分析音频: {temp_file_path}")
            result, warning = analyze_audio_file(temp_file_path, preferred_key=key)
            logger.info(f"音频分析完成: key={result.key}, tempo={result.tempo}, structures={len(result.structures)}")
            
            # 构建响应
            return AnalysisResponse(
                result=result,
                warning=warning
            )
            
        except Exception as e:
            logger.error(f"分析音频时出错: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"分析音频时出错: {str(e)}")
        finally:
            # 清理临时文件
            if os.path.exists(temp_file_path):
                logger.info(f"删除临时文件: {temp_file_path}")
                os.unlink(temp_file_path) 