import os
import json
import logging
import requests
from typing import Tuple, Optional, Dict, Any, List
from app.models.schemas import AudioAnalysisResult, SongStructure, Measure, BeatInfo
from app.core.config import settings

# 导入Music AI官方SDK - 正确的导入方式
from musicai_sdk import MusicAiClient

# 获取logger
logger = logging.getLogger(__name__)

class MusicAIAnalyzer:
    """Music AI 第三方分析服务封装，使用官方SDK"""
    
    def __init__(self):
        """初始化Music AI分析器"""
        # 获取API密钥，优先使用环境变量
        self.api_key = os.environ.get("MUSIC_AI_API_KEY", "") or settings.MUSIC_AI_API_KEY
        # 获取API基础URL - 保存用于日志但不传递给客户端
        self.api_base_url = os.environ.get("MUSIC_AI_API_URL", "") or settings.MUSIC_AI_API_URL
        # 预先创建的workflow ID
        self.workflow_id = "chords-and-beat-mapping"
        
        if not self.api_key:
            logger.warning("未设置MUSIC_AI_API_KEY，Music AI分析将无法使用")
            
        # 初始化SDK客户端 - 只传递api_key参数
        self.client = MusicAiClient(
            api_key=self.api_key
        )
        
        logger.info(f"初始化Music AI分析器: API URL={self.api_base_url}, Workflow ID={self.workflow_id}")
    
    def analyze_audio(self, file_path: str, preferred_key: Optional[str] = None) -> Tuple[AudioAnalysisResult, Optional[str]]:
        """使用Music AI分析音频文件"""
        warning = None
        
        if not self.api_key:
            warning = "未配置Music AI API密钥，无法使用Music AI分析"
            logger.error(warning)
            # 如果没有API密钥，返回一个空的分析结果
            return self._create_empty_result(preferred_key), warning
        
        try:
            logger.info(f"使用Music AI分析音频: {file_path}")
            
            # 1. 上传文件 - 返回的是file_url而非file_id
            logger.info("步骤1: 上传音频文件到Music AI...")
            file_url = self.client.upload_file(file_path=file_path)
            logger.info(f"文件上传成功，file_url: {file_url}")
            
            # 2. 创建分析任务
            logger.info("步骤2: 创建音频分析任务...")
            
            # 根据官方示例构造参数
            workflow_params = {
                'inputUrl': file_url,
            }
            
            # 如果有指定的调性，添加到参数中
            if preferred_key:
                workflow_params['preferred_key'] = preferred_key
            
            # 使用正确的参数创建任务
            job_name = f"analyze_{os.path.basename(file_path)}"
            create_job_info = self.client.create_job(
                job_name=job_name,
                workflow_id=self.workflow_id,
                params=workflow_params
            )
            
            # 从返回值中获取job_id
            job_id = create_job_info['id']
            logger.info(f"分析任务创建成功，job_id: {job_id}")
            
            # 3. 等待任务完成
            logger.info("步骤3: 等待分析任务完成...")
            job_info = self.client.wait_for_job_completion(job_id)
            
            # 从job_info中获取结果
            if job_info.get('status') != 'SUCCEEDED':
                raise Exception(f"任务状态: {job_info.get('status')}, 未能成功完成")
            
            logger.info(f"任务完成: {job_id}，结果类型: {type(job_info.get('result'))}")
            
            # 4. 下载并处理结果链接
            raw_result = job_info.get('result', {})
            logger.info(f"原始结果: {raw_result}")
            
            # 处理字段名称，将BPM改为bpm
            processed_result = self._process_field_names(raw_result)
            
            # 下载JSON链接内容
            logger.info("步骤4: 下载chords map和beat map JSON数据...")
            processed_result = self._download_json_links(processed_result)
            
            # 将原始结果转换为AudioAnalysisResult对象，但保留处理后的原始数据
            result = AudioAnalysisResult(
                key=processed_result.get('root key', preferred_key or 'C'),
                tempo=float(processed_result.get('bpm', 120.0)),
                structures=[],  # 空结构，前端将处理原始数据
                beats=BeatInfo(
                    positions=[],
                    bpm=float(processed_result.get('bpm', 120.0)),
                    timeSignature="4/4",
                    beatsPerBar=4
                ),
                raw_data=processed_result  # 添加处理后的原始数据字段
            )
            
            return result, warning
            
        except Exception as e:
            error_msg = f"Music AI分析出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_empty_result(preferred_key), error_msg
    
    def _process_field_names(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理字段名称，将BPM改为bpm"""
        result = {}
        
        for key, value in data.items():
            if key == 'BPM':
                result['bpm'] = value
            else:
                result[key] = value
                
        return result
        
    def _download_json_links(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """下载JSON链接内容"""
        processed_data = data.copy()
        
        # 下载chords map数据
        chords_map_url = data.get('chords map')
        if chords_map_url and isinstance(chords_map_url, str):
            logger.info(f"下载chords map数据: {chords_map_url}")
            try:
                chords_data = self._download_json_from_url(chords_map_url)
                processed_data['chords map'] = chords_data
            except Exception as e:
                logger.error(f"下载chords map数据失败: {str(e)}")
        
        # 下载beat map数据
        beat_map_url = data.get('beat map')
        if beat_map_url and isinstance(beat_map_url, str):
            logger.info(f"下载beat map数据: {beat_map_url}")
            try:
                beat_data = self._download_json_from_url(beat_map_url)
                processed_data['beat map'] = beat_data
            except Exception as e:
                logger.error(f"下载beat map数据失败: {str(e)}")
        
        return processed_data
    
    def _download_json_from_url(self, url: str) -> Any:
        """从URL下载JSON数据"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"下载JSON数据失败: {str(e)}")
            raise
    
    def _create_empty_result(self, preferred_key: Optional[str] = None) -> AudioAnalysisResult:
        """创建一个空的分析结果"""
        key = preferred_key or "C"
        tempo = 120.0
        
        # 创建一个基本的结构
        structures = [
            SongStructure(
                type="Verse",
                startTime=0.0,
                endTime=30.0,
                measures=[
                    Measure(number=1, chord="C", startTime=0.0, endTime=2.0, bars=1),
                    Measure(number=2, chord="G", startTime=2.0, endTime=4.0, bars=1),
                    Measure(number=3, chord="Am", startTime=4.0, endTime=6.0, bars=1),
                    Measure(number=4, chord="F", startTime=6.0, endTime=8.0, bars=1)
                ]
            )
        ]
        
        # 创建基本节拍信息
        beats_info = BeatInfo(
            positions=[],
            bpm=tempo,
            timeSignature="4/4",
            beatsPerBar=4
        )
        
        # 创建空的原始数据
        raw_data = {
            "root key": key,
            "bpm": tempo,
            "chords map": [],
            "beat map": []
        }
        
        return AudioAnalysisResult(
            key=key,
            tempo=tempo,
            structures=structures,
            beats=beats_info,
            raw_data=raw_data
        )

# 创建一个单例实例
_music_ai_analyzer = None

def get_music_ai_analyzer() -> MusicAIAnalyzer:
    """获取Music AI分析器单例"""
    global _music_ai_analyzer
    if _music_ai_analyzer is None:
        _music_ai_analyzer = MusicAIAnalyzer()
    return _music_ai_analyzer 