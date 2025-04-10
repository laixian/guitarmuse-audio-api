import librosa
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from app.models.schemas import AudioAnalysisResult, SongStructure, Measure, BeatInfo
from app.services.chord_detector import detect_chords
from app.core.logging import get_logger
from app.core.config import settings
import scipy
import os
import tempfile
import hashlib
import time

# 导入LLM增强模块
from app.services.llm_enhancement import get_llm_enhancer

# 导入Music AI分析器
from app.services.music_ai_analyzer import get_music_ai_analyzer

# 创建logger
logger = get_logger(__name__)

# SciPy兼容性工具函数
def get_hann_window(window_length):
    """
    获取Hann窗口，兼容各版本的SciPy
    """
    try:
        # 尝试使用scipy.signal.hann (旧版本)
        return scipy.signal.hann(window_length)
    except AttributeError:
        try:
            # 尝试使用scipy.signal.windows.hann (新版本)
            return scipy.signal.windows.hann(window_length)
        except (AttributeError, ImportError):
            # 如果都失败，使用numpy创建hann窗口
            logger.warning("无法使用scipy的hann窗口，改用numpy实现")
            n = np.arange(0, window_length)
            return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (window_length - 1))

# 修补librosa的窗口函数
original_stft = librosa.core.stft
def patched_stft(y, *args, **kwargs):
    """包装librosa的STFT函数使其使用我们的窗口兼容函数"""
    if 'window' not in kwargs and len(args) < 4:
        n_fft = args[0] if len(args) > 0 else kwargs.get('n_fft', 2048)
        kwargs['window'] = get_hann_window(n_fft)
    return original_stft(y, *args, **kwargs)

# 应用补丁
librosa.core.stft = patched_stft

# 简单的内存缓存
# 结构: {'file_hash': {'timestamp': timestamp, 'result': analysis_result}}
_analysis_cache: Dict[str, Dict[str, Any]] = {}
# 缓存过期时间: 7天（单位秒）
CACHE_EXPIRY = 7 * 24 * 60 * 60

class AudioAnalyzer:
    """音频分析器类"""
    
    def __init__(self, use_llm: bool = None):
        """初始化分析器"""
        # 如果未指定，使用配置中的默认值
        self.use_llm = settings.ENABLE_LLM_ENHANCEMENT if use_llm is None else use_llm
        logger.info(f"初始化音频分析器: use_llm={self.use_llm}")
    
    def analyze_audio(self, file_path: str, preferred_key: Optional[str] = None) -> Tuple[AudioAnalysisResult, Optional[str]]:
        """
        分析音频文件，返回分析结果
        
        参数:
            file_path: 音频文件路径
            preferred_key: 首选调性，如果为None则自动检测
            
        返回:
            (分析结果, 警告信息)
        """
        warning = None
        
        try:
            # 使用Music AI分析器
            music_ai_analyzer = get_music_ai_analyzer()
            result, music_ai_warning = music_ai_analyzer.analyze_audio(file_path, preferred_key)
            
            # 如果有警告，添加到警告列表
            if music_ai_warning:
                warning = music_ai_warning
            
            # 直接返回Music AI分析器的结果，不做任何处理
            return result, warning
            
        except Exception as e:
            error_msg = f"音频分析失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_empty_result(preferred_key), error_msg

    def _basic_analysis(self, file_path: str, preferred_key: Optional[str] = None) -> Tuple[AudioAnalysisResult, Optional[str]]:
        """基础音频分析"""
        warning = None
        try:
            # 加载音频文件
            # sr=None使用文件原始采样率，后面再重采样
            logger.info(f"开始加载音频文件: {file_path}")
            y, sr = librosa.load(file_path, sr=None)
            
            # 重采样到22050Hz以加快分析速度
            if sr != 22050:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
            
            logger.info(f"音频加载完成: 采样率={sr}Hz, 长度={len(y)/sr:.2f}秒")
            
            # 1. 检测调性 (音高)
            detected_key = detect_key(y, sr)
            logger.info(f"检测到的调性: {detected_key}")
            
            # 如果用户提供了首选调性，优先使用它
            key = preferred_key if preferred_key else detected_key
            logger.info(f"使用的调性: {key} (用户指定: {preferred_key != None})")
            
            # 2. 检测速度 (BPM)
            tempo = detect_tempo(y, sr)
            logger.info(f"检测到的速度: {tempo} BPM")
            
            # 3. 分析歌曲结构
            structures = analyze_structure(y, sr, tempo)
            logger.info(f"分析了 {len(structures)} 个段落")
            
            # 4. 检测和弦
            structures = detect_song_chords(y, sr, structures, key)
            
            # 5. 基于常用和弦走向优化结构和和弦
            structures = optimize_structure_with_chord_patterns(structures, key)
            logger.info("已完成基于和弦走向的优化")
            
            # 6. 分析节拍信息
            beats_info = analyze_beats(y, sr, tempo)
            logger.info(f"节拍分析: 检测到 {len(beats_info.positions)} 个拍点, BPM={beats_info.bpm}")
            
            # 构建结果
            result = AudioAnalysisResult(
                key=key,
                tempo=tempo,
                structures=structures,
                beats=beats_info
            )
            
            return result, warning
        
        except Exception as e:
            logger.error(f"分析音频时出错: {str(e)}", exc_info=True)
            # 出错时使用备用方法
            result = generate_fallback_analysis(file_path, preferred_key)
            warning = "使用了简化的音频分析方法，分析结果可能不够准确"
            return result, warning

def detect_key(y: np.ndarray, sr: int) -> str:
    """检测音频的调性"""
    # 使用Librosa的调性估计功能
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_index = np.argmax(np.sum(librosa.feature.tonnetz(y=y, sr=sr), axis=1))
    
    # 定义音高映射
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 根据调性检测结果判断是大调还是小调
    is_minor = False
    
    # 分析和旋律特征来判断是否为小调
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    if np.mean(centroid) < 2000:  # 简单启发式方法
        is_minor = True
    
    key = keys[key_index]
    if is_minor:
        # 计算相对小调
        minor_index = (key_index + 9) % 12
        key = keys[minor_index] + 'm'
    
    return key

def detect_tempo(y: np.ndarray, sr: int) -> float:
    """检测音频的速度 (BPM)"""
    # 使用Librosa的节奏检测功能
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return round(tempo, 1)

def analyze_structure(y: np.ndarray, sr: int, tempo: float) -> List[SongStructure]:
    """
    分析歌曲结构，简化版本
    不再进行段落标记，只返回基本节奏单位（小节）的连续序列
    由LLM负责进行段落规划
    """
    # 计算拍子间距
    beat_duration = 60 / tempo
    measure_duration = 4 * beat_duration  # 假设4/4拍
    
    # 获取音频时长
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 计算总小节数 (确保是4的倍数)
    raw_n_measures = max(1, int(duration / measure_duration))
    n_measures = make_measures_multiple_of_four(raw_n_measures)
    
    logger.info(f"音频时长: {duration:.2f}秒, 小节数: {n_measures}")
    
    # 创建小节序列
    measures = []
    for i in range(n_measures):
        measure_start = i * measure_duration
        measure_end = min((i + 1) * measure_duration, duration)
        
        measures.append(Measure(
            number=i+1,
            chord="",  # 和弦将在后续步骤中添加
            startTime=measure_start,
            endTime=measure_end
        ))
    
    # 创建单个结构，包含所有小节
    # 这里使用Undefined作为临时类型，由LLM来分配实际的段落
    structures = [SongStructure(
        type="Undefined",
        startTime=0,
        endTime=duration,
        measures=measures
    )]
    
    return structures

def make_measures_multiple_of_four(n_measures: int) -> int:
    """
    确保小节数是4的整数倍
    """
    if n_measures % 4 == 0:
        return n_measures  # 已经是4的倍数
    
    # 向上调整到最近的4的倍数
    return ((n_measures + 3) // 4) * 4

def verify_measures_multiple_of_four(structures: List[SongStructure]) -> None:
    """
    验证所有段落的小节数是否为4的整数倍
    如果不是，记录警告信息
    """
    for i, structure in enumerate(structures):
        if len(structure.measures) % 4 != 0:
            logger.warning(
                f"段落验证失败: 第{i+1}个段落({structure.type})的小节数({len(structure.measures)})不是4的整数倍"
            )
        else:
            logger.info(
                f"段落验证通过: 第{i+1}个段落({structure.type})的小节数({len(structure.measures)})是4的整数倍"
            )

def detect_song_chords(y: np.ndarray, sr: int, structures: List[SongStructure], key: str) -> List[SongStructure]:
    """为歌曲中的每个小节检测和弦"""
    logger.info("开始检测和弦...")

    # 导入chord_detector模块中的函数
    from app.services.chord_detector import detect_chords
    
    # 尝试进行Bass频率分离以提高和弦检测准确度
    try:
        # 使用数字信号处理方法分离低频(Bass)
        from scipy import signal
        
        # 设计截止频率为400Hz的低通滤波器
        nyquist = sr / 2
        cutoff = 400 / nyquist
        b, a = signal.butter(4, cutoff, btype='low')
        
        # 应用滤波器分离Bass频率
        y_bass = signal.filtfilt(b, a, y)
        logger.info("成功分离Bass频率，用于和弦检测")
    except Exception as e:
        logger.warning(f"Bass频率分离失败: {str(e)}，使用原始音频")
        y_bass = y
    
    for structure in structures:
        for measure in structure.measures:
            # 提取小节的时间段
            start_sample = int(measure.startTime * sr)
            end_sample = int(measure.endTime * sr)
            
            # 确保索引在有效范围内
            if start_sample >= len(y_bass):
                start_sample = max(0, len(y_bass) - 1)
            if end_sample >= len(y_bass):
                end_sample = len(y_bass) - 1
                
            # 提取小节音频
            if start_sample < end_sample:
                measure_audio = y_bass[start_sample:end_sample]
                
                # 检测和弦，使用chord_detector模块
                chord = detect_chords(measure_audio, sr, key)
                measure.chord = chord
            else:
                # 如果时间范围无效，使用上一个和弦或默认值
                measure.chord = key  # 默认使用调性的主和弦
    
    logger.info("和弦检测完成")
    return structures

def analyze_beats(y: np.ndarray, sr: int, tempo: float) -> BeatInfo:
    """分析音频的节拍信息"""
    logger.info("开始分析节拍信息...")
    
    try:
        # 使用librosa检测拍点位置
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # 确定时间签名和每小节拍数
        time_signature = "4/4"  # 默认为4/4拍
        beats_per_bar = 4
        
        # 尝试检测每小节拍数
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # 使用自相关来尝试检测节奏模式
        # 这可能有助于确定是否是3/4拍或其他节奏模式
        ac = librosa.autocorrelate(onset_env, max_size=sr // 2)
        peak_idx = np.argmax(ac[1:]) + 1
        
        if 2 < peak_idx < 4:
            # 可能是3/4拍
            time_signature = "3/4"
            beats_per_bar = 3
        
        logger.info(f"拍点分析完成: 找到 {len(beat_times)} 个拍点, 节拍型: {time_signature}")
        
        return BeatInfo(
            positions=beat_times.tolist(),
            bpm=tempo,
            timeSignature=time_signature,
            beatsPerBar=beats_per_bar
        )
    except Exception as e:
        logger.warning(f"节拍分析出错: {str(e)}，使用简化方法")
        
        # 如果分析失败，使用基于tempo的简化方法
        duration = librosa.get_duration(y=y, sr=sr)
        beat_duration = 60 / tempo  # 秒/拍
        beat_count = int(duration / beat_duration)
        
        # 生成等间距拍点
        beat_times = np.arange(0, duration, beat_duration)
        
        return BeatInfo(
            positions=beat_times.tolist(),
            bpm=tempo,
            timeSignature="4/4",
            beatsPerBar=4
        )

def generate_fallback_analysis(file_path: str, preferred_key: Optional[str] = None) -> AudioAnalysisResult:
    """
    生成简化的备用分析结果
    使用和弦级数和模式生成更自然的和弦进行
    """
    # 获取文件时长（如果可能）
    try:
        y, sr = librosa.load(file_path, sr=None, duration=10)  # 只加载前10秒以快速估计
        duration = librosa.get_duration(y=y, sr=sr)
    except:
        # 如果无法读取，假设3分钟时长
        duration = 180  
    
    # 默认调性和速度
    key = preferred_key if preferred_key else 'C'
    tempo = 120
    
    # 获取和弦走向库
    chord_library = get_chord_progressions_library()
    key_type = 'minor' if key.endswith('m') else 'major'
    
    # 为不同段落随机选择不同的和弦模式
    import random
    
    # 生成简单的结构
    section_times = [
        (0, duration * 0.1, 'Intro'),
        (duration * 0.1, duration * 0.35, 'Verse'),
        (duration * 0.35, duration * 0.55, 'Chorus'),
        (duration * 0.55, duration * 0.7, 'Verse'),
        (duration * 0.7, duration * 0.85, 'Bridge'),
        (duration * 0.85, duration, 'Outro')
    ]
    
    structures = []
    for start, end, section_type in section_times:
        section_duration = end - start
        
        # 估算小节数
        beats_per_measure = 4
        seconds_per_beat = 60 / tempo
        measure_duration = seconds_per_beat * beats_per_measure
        raw_n_measures = max(1, int(section_duration / measure_duration))
        
        # 确保小节数是4的整数倍
        n_measures = make_measures_multiple_of_four(raw_n_measures)
        
        # 为此段落随机选择一个和弦模式
        progression_pattern = random.choice(chord_library[key_type])
        
        # 选择一个变体（优先选择与小节数匹配的变体）
        matching_variants = [v for v in progression_pattern['variants'] if len(v) == n_measures]
        if not matching_variants:
            # 没有完全匹配的变体，选择一个重复的变体
            variant = random.choice(progression_pattern['variants'])
            # 扩展变体以匹配小节数
            chord_degrees = (variant * (n_measures // len(variant) + 1))[:n_measures]
        else:
            chord_degrees = random.choice(matching_variants)
        
        # 将和弦级数转换为具体和弦
        chord_progression = [degree_to_chord(degree, key) for degree in chord_degrees]
        
        measures = []
        for i in range(n_measures):
            measure_start = start + (i * measure_duration)
            measure_end = min(measure_start + measure_duration, end)
            
            # 使用生成的和弦
            chord_idx = i % len(chord_progression)
            chord = chord_progression[chord_idx]
            
            measures.append(Measure(
                number=i+1,
                chord=chord,
                startTime=measure_start,
                endTime=measure_end
            ))
        
        # 调整段落终点时间以匹配小节总数
        adjusted_end = start + (n_measures * measure_duration)
        
        structures.append(SongStructure(
            type=section_type,
            startTime=start,
            endTime=min(adjusted_end, end),  # 确保不超过原始结束时间
            measures=measures
        ))
    
    # 验证所有段落的小节数是否为4的倍数
    verify_measures_multiple_of_four(structures)
    
    # 生成拍点信息
    beat_duration = 60 / tempo
    beat_times = np.arange(0, duration, beat_duration).tolist()
    
    beats_info = BeatInfo(
        positions=beat_times,
        bpm=tempo,
        timeSignature="4/4",
        beatsPerBar=4
    )
    
    return AudioAnalysisResult(
        key=key,
        tempo=tempo,
        structures=structures,
        beats=beats_info
    )

def get_chord_progressions_library() -> Dict[str, List[Dict]]:
    """
    创建常用和弦走向库，使用罗马数字级数表示
    返回格式: {调性类型: [走向模式列表]}
    每个走向模式是一个字典，包含和弦级数和模式描述
    """
    # 定义大调常用和弦级数走向
    major_progressions = [
        {
            'name': '流行音乐最常见走向 (I-V-vi-IV)',
            'degrees': ['I', 'V', 'vi', 'IV'],
            'variants': [
                # 标准形式（每个和弦1小节）
                ['I', 'V', 'vi', 'IV'],
                # 每个和弦2小节
                ['I', 'I', 'V', 'V', 'vi', 'vi', 'IV', 'IV'],
                # 混合形式（不同和弦持续时间不同）
                ['I', 'I', 'V', 'vi', 'IV']
            ]
        },
        {
            'name': '50年代和弦进行 (I-vi-IV-V)',
            'degrees': ['I', 'vi', 'IV', 'V'],
            'variants': [
                ['I', 'vi', 'IV', 'V'],
                ['I', 'I', 'vi', 'vi', 'IV', 'IV', 'V', 'V'],
                ['I', 'vi', 'vi', 'IV', 'V', 'V']
            ]
        },
        {
            'name': '基本和弦进行 (I-IV-V-I)',
            'degrees': ['I', 'IV', 'V', 'I'],
            'variants': [
                ['I', 'IV', 'V', 'I'],
                ['I', 'I', 'IV', 'IV', 'V', 'V', 'I', 'I']
            ]
        },
        {
            'name': '常见变体 (I-iii-IV-V)',
            'degrees': ['I', 'iii', 'IV', 'V'],
            'variants': [
                ['I', 'iii', 'IV', 'V'],
                ['I', 'I', 'iii', 'iii', 'IV', 'IV', 'V', 'V']
            ]
        },
        {
            'name': '民谣常用 (I-IV-I-V)',
            'degrees': ['I', 'IV', 'I', 'V'],
            'variants': [
                ['I', 'IV', 'I', 'V'],
                ['I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'V']
            ]
        },
        {
            'name': '摇滚常用 (I-V-IV-V)',
            'degrees': ['I', 'V', 'IV', 'V'],
            'variants': [
                ['I', 'V', 'IV', 'V'],
                ['I', 'I', 'V', 'V', 'IV', 'IV', 'V', 'V']
            ]
        },
        {
            'name': '爵士常用 (I-IV-ii-V)',
            'degrees': ['I', 'IV', 'ii', 'V'],
            'variants': [
                ['I', 'IV', 'ii', 'V'],
                ['I', 'I', 'IV', 'IV', 'ii', 'ii', 'V', 'V']
            ]
        },
        {
            'name': '爵士标准进行 (ii-V-I)',
            'degrees': ['ii', 'V', 'I', 'I'],
            'variants': [
                ['ii', 'V', 'I', 'I'],
                ['ii', 'ii', 'V', 'V', 'I', 'I', 'I', 'I']
            ]
        },
        {
            'name': '基本交替 (I-V-I-V)',
            'degrees': ['I', 'V', 'I', 'V'],
            'variants': [
                ['I', 'V', 'I', 'V'],
                ['I', 'I', 'V', 'V', 'I', 'I', 'V', 'V']
            ]
        },
        {
            'name': '四度上行走向 (I-IV-vii-iii-vi-ii-V-I)',
            'degrees': ['I', 'IV', 'vii', 'iii', 'vi', 'ii', 'V', 'I'],
            'variants': [
                ['I', 'IV', 'vii', 'iii', 'vi', 'ii', 'V', 'I']
            ]
        }
    ]
    
    # 定义小调常用和弦级数走向
    minor_progressions = [
        {
            'name': '小调最常见 (i-VI-III-VII)',
            'degrees': ['i', 'VI', 'III', 'VII'],
            'variants': [
                ['i', 'VI', 'III', 'VII'],
                ['i', 'i', 'VI', 'VI', 'III', 'III', 'VII', 'VII']
            ]
        },
        {
            'name': '小调基本进行 (i-iv-VII-i)',
            'degrees': ['i', 'iv', 'VII', 'i'],
            'variants': [
                ['i', 'iv', 'VII', 'i'],
                ['i', 'i', 'iv', 'iv', 'VII', 'VII', 'i', 'i']
            ]
        },
        {
            'name': '下行走向 (i-VII-VI-V)',
            'degrees': ['i', 'VII', 'VI', 'V'],
            'variants': [
                ['i', 'VII', 'VI', 'V'],
                ['i', 'i', 'VII', 'VII', 'VI', 'VI', 'V', 'V']
            ]
        },
        {
            'name': '交替式 (i-VII-i-VII)',
            'degrees': ['i', 'VII', 'i', 'VII'],
            'variants': [
                ['i', 'VII', 'i', 'VII'],
                ['i', 'i', 'VII', 'VII', 'i', 'i', 'VII', 'VII']
            ]
        },
        {
            'name': '流行小调 (i-III-VII-iv)',
            'degrees': ['i', 'III', 'VII', 'iv'],
            'variants': [
                ['i', 'III', 'VII', 'iv'],
                ['i', 'i', 'III', 'III', 'VII', 'VII', 'iv', 'iv']
            ]
        },
        {
            'name': '小调终止 (i-VI-VII-i)',
            'degrees': ['i', 'VI', 'VII', 'i'],
            'variants': [
                ['i', 'VI', 'VII', 'i'],
                ['i', 'i', 'VI', 'VI', 'VII', 'VII', 'i', 'i']
            ]
        },
        {
            'name': '小调交替 (i-V-i-V)',
            'degrees': ['i', 'V', 'i', 'V'],
            'variants': [
                ['i', 'V', 'i', 'V'],
                ['i', 'i', 'V', 'V', 'i', 'i', 'V', 'V']
            ]
        },
        {
            'name': '和声小调 (i-iv-V-i)',
            'degrees': ['i', 'iv', 'V', 'i'],
            'variants': [
                ['i', 'iv', 'V', 'i'],
                ['i', 'i', 'iv', 'iv', 'V', 'V', 'i', 'i']
            ]
        },
        {
            'name': '爵士小调 (i-VI-ii-V)',
            'degrees': ['i', 'VI', 'ii', 'V'],
            'variants': [
                ['i', 'VI', 'ii', 'V'],
                ['i', 'i', 'VI', 'VI', 'ii', 'ii', 'V', 'V']
            ]
        }
    ]
    
    return {
        'major': major_progressions,
        'minor': minor_progressions
    }

def chord_to_degree(chord: str, key: str) -> str:
    """
    将具体和弦转换为相对于调性的和弦级数
    例如: 在C大调中，G = V，Am = vi
    """
    # 定义大调和弦级数
    major_degrees = {
        0: 'I',
        1: 'bII',
        2: 'II',
        3: 'bIII',
        4: 'III',
        5: 'IV',
        6: 'bV',
        7: 'V',
        8: 'bVI',
        9: 'VI',
        10: 'bVII',
        11: 'VII'
    }
    
    # 定义小调和弦级数
    minor_degrees = {
        0: 'i',
        1: 'bII',
        2: 'ii',
        3: 'III',
        4: 'iv',
        5: 'v',
        6: 'bVI',
        7: 'VI',
        8: 'bVII',
        9: 'VII',
        10: 'bVIII',
        11: 'VIII'
    }
    
    # 定义音高映射
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 检查和弦是否是小调
    is_chord_minor = chord.endswith('m') and not chord.endswith('dim')
    
    # 获取和弦根音
    chord_root = chord[:-1] if is_chord_minor else chord
    
    # 获取调性根音和是否为小调
    is_key_minor = key.endswith('m')
    key_root = key[:-1] if is_key_minor else key
    
    # 计算和弦根音与调性根音的半音距离
    try:
        root_index = notes.index(chord_root)
        key_index = notes.index(key_root)
        semitones = (root_index - key_index) % 12
        
        # 根据调性和和弦类型选择合适的级数表示
        if is_key_minor:
            # 小调中的级数
            degree = minor_degrees[semitones]
            # 如果和弦是大三和弦，并且小调中该位置应该是小三和弦，则标记为大三和弦
            if not is_chord_minor and semitones in [0, 4, 7]:
                degree = degree.upper()
        else:
            # 大调中的级数
            degree = major_degrees[semitones]
            # 如果和弦是小三和弦，并且大调中该位置应该是大三和弦，则转换为小写
            if is_chord_minor and semitones in [0, 5, 7]:
                degree = degree.lower()
        
        return degree
    except (ValueError, KeyError):
        # 无法识别的和弦，返回原始和弦
        return chord

def degree_to_chord(degree: str, key: str) -> str:
    """
    将和弦级数转换为具体和弦
    例如: 在C大调中，V = G，vi = Am
    """
    # 定义音高映射
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 定义大调级数对应的半音距离
    major_degree_semitones = {
        'I': 0, 'bII': 1, 'II': 2, 'bIII': 3, 'III': 4, 'IV': 5, 
        'bV': 6, 'V': 7, 'bVI': 8, 'VI': 9, 'bVII': 10, 'VII': 11,
        'i': 0, 'ii': 2, 'iii': 4, 'iv': 5, 'v': 7, 'vi': 9, 'vii': 11
    }
    
    # 定义小调级数对应的半音距离
    minor_degree_semitones = {
        'i': 0, 'bII': 1, 'ii': 2, 'III': 3, 'iv': 5, 'v': 7, 
        'bVI': 8, 'VI': 8, 'bVII': 10, 'VII': 10, 'bVIII': 11, 'VIII': 11,
        'I': 0, 'II': 2, 'IV': 5, 'V': 7, 'VI': 8, 'VII': 10
    }
    
    # 检查调性是否是小调
    is_key_minor = key.endswith('m')
    key_root = key[:-1] if is_key_minor else key
    
    try:
        # 获取调性根音在音高列表中的索引
        key_index = notes.index(key_root)
        
        # 获取级数对应的半音数
        if is_key_minor:
            semitones = minor_degree_semitones.get(degree)
        else:
            semitones = major_degree_semitones.get(degree)
        
        if semitones is None:
            # 无法识别的级数
            return key  # 默认返回调性主和弦
        
        # 计算和弦根音
        root_index = (key_index + semitones) % 12
        root_note = notes[root_index]
        
        # 判断和弦类型（大三和弦或小三和弦）
        is_minor_chord = False
        
        if is_key_minor:
            # 小调中，i, iv, v 是小三和弦
            is_minor_chord = degree in ['i', 'iv', 'v'] or (degree.islower() and degree not in ['bII', 'bIII', 'bVI', 'bVII'])
        else:
            # 大调中，ii, iii, vi 是小三和弦
            is_minor_chord = degree in ['ii', 'iii', 'vi'] or (degree.islower() and degree not in ['bII', 'bIII', 'bVI', 'bVII'])
        
        # 构建和弦
        chord = root_note + ('m' if is_minor_chord else '')
        
        return chord
    except (ValueError, KeyError):
        # 无法转换时返回原始级数
        return key  # 默认返回调性主和弦

def optimize_structure_with_chord_patterns(structures: List[SongStructure], key: str) -> List[SongStructure]:
    """
    基于常用和弦走向优化歌曲结构和和弦
    使用级数表示法进行匹配，支持不同长度的和弦持续模式
    """
    logger.info("开始基于和弦走向优化歌曲结构...")
    
    # 获取和弦走向库
    chord_library = get_chord_progressions_library()
    
    # 确定调性类型
    key_type = 'minor' if key.endswith('m') else 'major'
    progressions = chord_library[key_type]
    
    # 为每个段落优化和弦走向
    for structure_idx, structure in enumerate(structures):
        if len(structure.measures) < 4:
            logger.warning(f"段落 {structure.type} 小节数 ({len(structure.measures)}) 小于4，跳过和弦优化")
            continue
        
        # 收集段落中的和弦序列，并转换为级数表示
        chord_sequence = [measure.chord for measure in structure.measures]
        degree_sequence = [chord_to_degree(chord, key) for chord in chord_sequence]
        
        logger.info(f"段落 {structure.type} 的和弦级数序列: {degree_sequence[:16]}...")
        
        # 识别段落的和弦模式
        best_pattern, best_variant, confidence = identify_chord_pattern(degree_sequence, progressions)
        
        if best_pattern and confidence > 0.5:
            logger.info(f"发现和弦模式: {best_pattern['name']}, 置信度: {confidence:.2f}")
            
            # 使用识别出的模式优化和弦序列
            optimized_degrees = optimize_using_pattern(degree_sequence, best_pattern, best_variant)
            
            # 将优化后的级数转换回具体和弦
            optimized_chords = [degree_to_chord(degree, key) for degree in optimized_degrees]
            
            # 更新段落中的和弦
            for i, chord in enumerate(optimized_chords[:len(structure.measures)]):
                structure.measures[i].chord = chord
            
            logger.info(f"段落 {structure.type} 的和弦已优化为模式: {best_pattern['name']}")
        else:
            # 如果未找到明确的模式，使用局部优化
            logger.info(f"未找到明确的和弦模式，应用局部优化")
            
            # 分割成更小的片段（4小节一组）进行优化
            for i in range(0, len(degree_sequence), 4):
                segment = degree_sequence[i:i+4]
                if len(segment) < 4:
                    continue
                
                # 寻找最匹配的4小节模式
                local_pattern, local_variant, local_confidence = identify_chord_pattern(segment, progressions)
                
                if local_pattern and local_confidence > 0.6:
                    # 局部优化这4小节
                    optimized_segment = local_variant
                    
                    # 更新这部分小节的和弦
                    for j in range(min(4, len(structure.measures) - i)):
                        chord = degree_to_chord(optimized_segment[j % len(optimized_segment)], key)
                        structure.measures[i + j].chord = chord
    
    # 优化段落结构，确保总小节数是4的倍数
    optimized_structures = optimize_section_structure(structures, key)
    
    return optimized_structures

def identify_chord_pattern(degree_sequence: List[str], progressions: List[Dict]) -> Tuple[Optional[Dict], Optional[List[str]], float]:
    """
    识别和弦级数序列最匹配的模式
    返回最匹配的模式、具体变体和置信度
    """
    best_pattern = None
    best_variant = None
    best_confidence = 0.0
    
    for pattern in progressions:
        # 检查每个模式的所有变体
        for variant in pattern['variants']:
            # 计算匹配度
            confidence = compute_pattern_match(degree_sequence, variant)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_pattern = pattern
                best_variant = variant
    
    return best_pattern, best_variant, best_confidence

def compute_pattern_match(sequence: List[str], pattern: List[str]) -> float:
    """
    计算序列与模式的匹配度，考虑重复和部分匹配
    返回0-1之间的置信度
    """
    # 如果序列太短，直接返回低置信度
    if len(sequence) < 4:
        return 0.0
    
    # 扩展模式以覆盖整个序列
    extended_pattern = pattern * (len(sequence) // len(pattern) + 1)
    extended_pattern = extended_pattern[:len(sequence)]
    
    # 计算完全匹配的数量
    exact_matches = sum(1 for a, b in zip(sequence, extended_pattern) if a == b)
    
    # 计算主要和弦功能的匹配（如I、IV、V级）
    function_matches = sum(1 for a, b in zip(sequence, extended_pattern) 
                        if (a in ['I', 'IV', 'V', 'i', 'iv', 'v', 'VI'] and a == b))
    
    # 计算级数相似性（不考虑大小写和修饰符）
    similar_matches = sum(1 for a, b in zip(sequence, extended_pattern) 
                         if a.upper().replace('B', '').replace('#', '') == b.upper().replace('B', '').replace('#', ''))
    
    # 计算加权置信度
    confidence = (0.6 * exact_matches + 0.3 * function_matches + 0.1 * similar_matches) / len(sequence)
    
    # 对于更长的匹配，给予额外奖励
    if len(sequence) >= 8:
        confidence *= 1.1
    
    # 对于模式重复至少两次的情况，给予额外奖励
    if len(sequence) >= len(pattern) * 2:
        # 检查重复的模式是否一致
        pattern_repetitions = len(sequence) // len(pattern)
        consistent_repetition = True
        
        for i in range(pattern_repetitions):
            if i > 0:
                segment1 = sequence[0:len(pattern)]
                segment2 = sequence[i*len(pattern):(i+1)*len(pattern)]
                
                # 检查两个片段的相似度
                similarity = sum(1 for a, b in zip(segment1, segment2) if a == b) / len(pattern)
                if similarity < 0.5:
                    consistent_repetition = False
                    break
        
        if consistent_repetition:
            confidence *= 1.2
    
    return min(confidence, 1.0)  # 确保置信度不超过1

def optimize_using_pattern(sequence: List[str], pattern: Dict, variant: List[str]) -> List[str]:
    """
    使用识别的模式优化和弦序列
    """
    # 如果序列太短，不进行优化
    if len(sequence) < 4:
        return sequence
    
    # 扩展变体模式以覆盖整个序列
    extended_pattern = variant * (len(sequence) // len(variant) + 1)
    extended_pattern = extended_pattern[:len(sequence)]
    
    # 将不匹配的位置替换为模式中的和弦
    optimized = []
    for i, (orig, patt) in enumerate(zip(sequence, extended_pattern)):
        # 保留序列中已经匹配模式的和弦
        if orig == patt:
            optimized.append(orig)
        else:
            # 检查周围和弦的一致性
            context_match = False
            if i > 0 and i < len(sequence) - 1:
                if sequence[i-1] == extended_pattern[i-1] and sequence[i+1] == extended_pattern[i+1]:
                    context_match = True
            
            # 根据上下文决定是保留原和弦还是使用模式和弦
            if context_match or orig not in ['I', 'IV', 'V', 'i', 'iv', 'v']:  # 非主要和弦功能更容易替换
                optimized.append(patt)
            else:
                optimized.append(orig)
    
    return optimized

def optimize_section_structure(structures: List[SongStructure], key: str) -> List[SongStructure]:
    """
    优化歌曲段落结构，确保总小节数是4的倍数
    可能会调整段落边界或合并短段落
    """
    # 如果结构为空，直接返回
    if not structures:
        return structures
    
    # 计算总小节数
    total_measures = sum(len(structure.measures) for structure in structures)
    
    # 如果总小节数已经是4的倍数，不需要调整
    if total_measures % 4 == 0:
        logger.info(f"总小节数 {total_measures} 已经是4的倍数，不需要调整段落结构")
        return structures
    
    # 需要添加或删除的小节数
    measures_to_adjust = 4 - (total_measures % 4)
    logger.info(f"需要调整 {measures_to_adjust} 个小节使总数为4的倍数")
    
    # 根据调性获取和弦进行
    chord_progression = get_common_chord_progression(key)
    
    # 策略1: 找出最适合调整的段落
    # 通常，Intro或Outro是最容易调整的
    target_section_types = ['Outro', 'Intro', 'Bridge']
    target_section = None
    
    for type_name in target_section_types:
        for structure in structures:
            if structure.type == type_name:
                target_section = structure
                break
        if target_section:
            break
    
    # 如果没找到目标段落，选择最后一个段落
    if not target_section:
        target_section = structures[-1]
    
    # 如果需要添加小节
    if measures_to_adjust > 0:
        section_idx = structures.index(target_section)
        
        # 获取段落中最后一个小节的时间信息
        last_measure = target_section.measures[-1]
        measure_duration = last_measure.endTime - last_measure.startTime
        
        # 创建新的小节
        new_measures = []
        for i in range(measures_to_adjust):
            start_time = last_measure.endTime + i * measure_duration
            end_time = start_time + measure_duration
            
            # 使用适当的和弦
            chord_idx = (len(target_section.measures) + i) % len(chord_progression)
            chord = chord_progression[chord_idx]
            
            new_measure = Measure(
                number=len(target_section.measures) + i + 1,
                chord=chord,
                startTime=start_time,
                endTime=end_time
            )
            new_measures.append(new_measure)
        
        # 更新段落
        target_section.measures.extend(new_measures)
        target_section.endTime = new_measures[-1].endTime
        
        # 更新后续段落的开始时间
        for j in range(section_idx + 1, len(structures)):
            time_shift = measures_to_adjust * measure_duration
            structures[j].startTime += time_shift
            structures[j].endTime += time_shift
            
            # 更新该段落中所有小节的时间
            for measure in structures[j].measures:
                measure.startTime += time_shift
                measure.endTime += time_shift
    
    # 如果需要删除小节 (较少见的情况)
    elif measures_to_adjust < 0:
        measures_to_remove = abs(measures_to_adjust)
        
        # 从目标段落中移除小节
        if len(target_section.measures) > measures_to_remove:
            # 移除末尾的小节
            target_section.measures = target_section.measures[:-measures_to_remove]
            target_section.endTime = target_section.measures[-1].endTime
            
            # 更新后续段落的开始时间
            section_idx = structures.index(target_section)
            for j in range(section_idx + 1, len(structures)):
                time_shift = -measures_to_remove * (
                    structures[j-1].measures[1].startTime - structures[j-1].measures[0].startTime 
                    if len(structures[j-1].measures) > 1 else 0
                )
                structures[j].startTime += time_shift
                structures[j].endTime += time_shift
                
                # 更新该段落中所有小节的时间
                for measure in structures[j].measures:
                    measure.startTime += time_shift
                    measure.endTime += time_shift
        else:
            # 如果段落小节数不足，考虑移除整个段落
            logger.warning(f"段落 {target_section.type} 小节数不足以删除，考虑调整其他段落")
            
            # 这里可以添加更复杂的逻辑来处理这种情况
    
    # 再次验证小节数是否为4的倍数
    new_total = sum(len(structure.measures) for structure in structures)
    logger.info(f"优化后总小节数: {new_total}，是否为4的倍数: {new_total % 4 == 0}")
    
    return structures 

def _get_from_cache(file_hash: str) -> Optional[AudioAnalysisResult]:
    """从缓存中获取分析结果"""
    try:
        cache_data = _analysis_cache.get(file_hash)
        if not cache_data:
            return None
            
        # 检查缓存是否过期
        if time.time() - cache_data['timestamp'] > CACHE_EXPIRY:
            logger.info(f"缓存已过期: {file_hash}")
            _analysis_cache.pop(file_hash, None)
            return None
            
        return cache_data['result']
    except Exception as e:
        logger.error(f"从缓存获取分析结果失败: {str(e)}")
        return None

# 获取分析器实例（单例模式）
def get_analyzer_instance():
    """获取AudioAnalyzer的单例实例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AudioAnalyzer()
    return _analyzer_instance

# 初始化单例实例
_analyzer_instance = None

# 原有的analyze_audio函数保留，但不直接在API中调用
def analyze_audio(file_path: str, preferred_key: Optional[str] = None) -> Tuple[AudioAnalysisResult, Optional[str]]:
    """
    使用librosa分析音频文件并生成分析结果
    """
    warning = None
    try:
        # 加载音频文件
        # sr=None使用文件原始采样率，后面再重采样
        logger.info(f"开始加载音频文件: {file_path}")
        y, sr = librosa.load(file_path, sr=None)
        
        # 重采样到22050Hz以加快分析速度
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        
        logger.info(f"音频加载完成: 采样率={sr}Hz, 长度={len(y)/sr:.2f}秒")
        
        # 1. 检测调性 (音高)
        detected_key = detect_key(y, sr)
        logger.info(f"检测到的调性: {detected_key}")
        
        # 如果用户提供了首选调性，优先使用它
        key = preferred_key if preferred_key else detected_key
        logger.info(f"使用的调性: {key} (用户指定: {preferred_key != None})")
        
        # 2. 检测速度 (BPM)
        tempo = detect_tempo(y, sr)
        logger.info(f"检测到的速度: {tempo} BPM")
        
        # 3. 分析歌曲结构
        structures = analyze_structure(y, sr, tempo)
        logger.info(f"分析了 {len(structures)} 个段落")
        
        # 4. 检测和弦
        structures = detect_song_chords(y, sr, structures, key)
        
        # 5. 基于常用和弦走向优化结构和和弦
        structures = optimize_structure_with_chord_patterns(structures, key)
        logger.info("已完成基于和弦走向的优化")
        
        # 6. 分析节拍信息
        beats_info = analyze_beats(y, sr, tempo)
        logger.info(f"节拍分析: 检测到 {len(beats_info.positions)} 个拍点, BPM={beats_info.bpm}")
        
        # 构建结果
        result = AudioAnalysisResult(
            key=key,
            tempo=tempo,
            structures=structures,
            beats=beats_info
        )
        
        return result, warning
        
    except Exception as e:
        logger.error(f"分析音频时出错: {str(e)}", exc_info=True)
        # 出错时使用备用方法
        result = generate_fallback_analysis(file_path, preferred_key)
        warning = "使用了简化的音频分析方法，分析结果可能不够准确"
        return result, warning 

def clean_short_duration_chords(analysis_result: AudioAnalysisResult, min_duration: float = 1.0) -> AudioAnalysisResult:
    """
    此功能在API直接传递数据的模式下已弃用
    仅保留函数签名以兼容现有代码
    
    参数:
        analysis_result: 音频分析结果
        min_duration: 最小和弦持续时间（秒），默认为1.0秒
        
    返回:
        原样返回传入的音频分析结果
    """
    logger.info("clean_short_duration_chords函数在API直接传递数据模式下已弃用")
    return analysis_result

def analyze_audio_file(file_path: str, preferred_key: Optional[str] = None, use_llm: bool = None) -> Tuple[AudioAnalysisResult, Optional[str]]:
    """分析音频文件，检测调性、速度和结构，返回原始API数据"""
    start_time = time.time()
    
    try:
        # 计算文件哈希值作为缓存键
        file_hash = _get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        
        # 尝试从缓存中获取分析结果
        cached_result = _get_from_cache(file_hash)
        if cached_result:
            logger.info(f"从缓存中获取分析结果: {file_name} ({file_hash})")
            return cached_result, None
            
        logger.info(f"开始分析音频文件: {file_path}")
        
        # 使用Music AI分析器
        music_ai_analyzer = get_music_ai_analyzer()
        result, warning = music_ai_analyzer.analyze_audio(file_path, preferred_key)
        
        # 直接将Music AI的结果返回给前端，不做任何处理
        
        # 将结果保存到缓存
        _save_to_cache(file_hash, result)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        logger.info(f"音频分析完成: {file_name}, 耗时: {processing_time:.2f}秒")
        
        return result, warning
    
    except Exception as e:
        logger.error(f"音频分析失败: {str(e)}", exc_info=True)
        # 计算处理时间
        processing_time = time.time() - start_time
        logger.info(f"音频分析失败, 耗时: {processing_time:.2f}秒")
        
        # 返回空的分析结果
        # 创建一个包含原始数据格式的空结果
        empty_result = AudioAnalysisResult(
            key="C", 
            tempo=120.0,
            structures=[],
            beats=BeatInfo(
                positions=[],
                bpm=120.0,
                timeSignature="4/4",
                beatsPerBar=4
            ),
            raw_data={
                "root key": "C",
                "bpm": 120.0,
                "chords map": [],
                "beat map": []
            }
        )
        return empty_result, f"音频分析失败: {str(e)}"

def _get_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # 只读取文件前1MB来计算哈希值（提高性能）
            buf = f.read(1024 * 1024)
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"计算文件哈希值失败: {str(e)}")
        # 如果计算失败，返回基于文件名和大小的哈希
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        return hashlib.md5(f"{file_name}_{file_size}".encode()).hexdigest()

def _save_to_cache(file_hash: str, result: AudioAnalysisResult) -> None:
    """保存分析结果到缓存"""
    try:
        _analysis_cache[file_hash] = {
            'timestamp': time.time(),
            'result': result
        }
        logger.info(f"分析结果已保存到缓存: {file_hash}")
    except Exception as e:
        logger.error(f"保存分析结果到缓存失败: {str(e)}") 