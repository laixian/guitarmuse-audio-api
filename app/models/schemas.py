from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Measure(BaseModel):
    number: int
    chord: str
    startTime: float
    endTime: float
    barSpan: int = 1  # 和弦跨越的小节数量，默认为1
    bars: int = 1     # 同barSpan，显示用，对应end_bar - start_bar

class SongStructure(BaseModel):
    type: str  # 'Intro', 'Verse', 'Chorus', 'Bridge', 'Outro'
    startTime: float
    endTime: float
    measures: List[Measure]

class BeatInfo(BaseModel):
    positions: List[float]  # 拍点的时间位置（秒）
    bpm: float  # 每分钟拍数
    timeSignature: str  # 例如 "4/4", "3/4" 等
    beatsPerBar: int  # 每小节拍数

class AudioAnalysisResult(BaseModel):
    """音频分析结果模型"""
    key: str
    tempo: float
    structures: List[SongStructure]
    beats: BeatInfo
    raw_data: Optional[Dict[str, Any]] = None  # 添加原始数据字段，用于存储API返回的原始数据

class AnalysisResponse(BaseModel):
    result: AudioAnalysisResult
    warning: Optional[str] = None 