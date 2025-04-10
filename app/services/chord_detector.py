import numpy as np
import librosa
from typing import Optional, Dict, List
import logging

# 获取logger
logger = logging.getLogger(__name__)

# 定义常见和弦的音符集
CHORD_PROFILES = {
    # 大三和弦
    'C': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # C, E, G
    'C#': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'D#': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'E': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    'F': [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'F#': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    'G': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    'G#': [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    'A': [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#': [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'B': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    
    # 小三和弦
    'Cm': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # C, Eb, G
    'C#m': [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'Dm': [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    'D#m': [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    'Em': [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'Fm': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    'F#m': [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    'Gm': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    'G#m': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    'Am': [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'A#m': [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    'Bm': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
}

def detect_chords(y: np.ndarray, sr: int, key: Optional[str] = None) -> str:
    """
    使用色度特征检测音频片段中的主要和弦
    """
    if len(y) == 0:
        return key if key else "C"
        
    # 优先尝试分离Bass音轨以提高和弦检测准确度
    try:
        y_bass = isolate_bass_frequencies(y, sr)
        # 提取色度特征 (12维向量表示12个音高类)
        chroma = librosa.feature.chroma_cqt(y=y_bass, sr=sr)
    except Exception as e:
        # 如果Bass分离失败，使用原始音频
        logger.warning(f"Bass频率分离失败: {str(e)}，使用原始音频")
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # 计算每个和弦模板的相似度
    chord_scores = {}
    for chord_name, profile in CHORD_PROFILES.items():
        # 计算色度特征与和弦模板的相似度
        similarity = cosine_similarity(np.mean(chroma, axis=1), profile)
        chord_scores[chord_name] = similarity
    
    # 如果提供了调性，增加该调性中常见和弦的权重
    if key:
        chord_weights = get_key_chord_weights(key)
        for chord, weight in chord_weights.items():
            if chord in chord_scores:
                chord_scores[chord] *= weight
    
    # 选择最高分的和弦
    best_chord = max(chord_scores.items(), key=lambda x: x[1])[0]
    
    return best_chord

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a = np.array(a)
    b = np.array(b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
        
    return dot_product / (norm_a * norm_b)

def get_key_chord_weights(key: str) -> Dict[str, float]:
    """
    获取特定调性中常见和弦的权重
    这使得和弦检测在特定调性中更加准确
    """
    weights = {}
    
    # 大调和弦权重
    major_weights = {
        'C': {'C': 1.5, 'F': 1.3, 'G': 1.3, 'Am': 1.2, 'Dm': 1.1, 'Em': 1.1},
        'G': {'G': 1.5, 'C': 1.3, 'D': 1.3, 'Em': 1.2, 'Am': 1.1, 'Bm': 1.1},
        'D': {'D': 1.5, 'G': 1.3, 'A': 1.3, 'Bm': 1.2, 'Em': 1.1, 'F#m': 1.1},
        'A': {'A': 1.5, 'D': 1.3, 'E': 1.3, 'F#m': 1.2, 'Bm': 1.1, 'C#m': 1.1},
        'E': {'E': 1.5, 'A': 1.3, 'B': 1.3, 'C#m': 1.2, 'F#m': 1.1, 'G#m': 1.1},
        'F': {'F': 1.5, 'Bb': 1.3, 'C': 1.3, 'Dm': 1.2, 'Gm': 1.1, 'Am': 1.1},
    }
    
    # 小调和弦权重
    minor_weights = {
        'Am': {'Am': 1.5, 'Dm': 1.3, 'Em': 1.3, 'F': 1.2, 'G': 1.2, 'C': 1.1},
        'Em': {'Em': 1.5, 'Am': 1.3, 'Bm': 1.3, 'C': 1.2, 'D': 1.2, 'G': 1.1},
        'Bm': {'Bm': 1.5, 'Em': 1.3, 'F#m': 1.3, 'G': 1.2, 'A': 1.2, 'D': 1.1},
    }
    
    # 根据调性类型选择权重映射
    if key.endswith('m'):
        weights = minor_weights.get(key, {})
    else:
        weights = major_weights.get(key, {})
    
    # 对于任何没有指定的和弦，使用默认权重1.0
    return {chord: weights.get(chord, 1.0) for chord in CHORD_PROFILES.keys()}

def isolate_bass_frequencies(y: np.ndarray, sr: int) -> np.ndarray:
    """
    使用数字信号处理方法分离Bass频率
    """
    # 使用低通滤波器分离Bass频率范围 (40-400Hz)
    from scipy import signal
    
    # 设计截止频率为400Hz的低通滤波器
    nyquist = sr / 2
    cutoff = 400 / nyquist
    b, a = signal.butter(4, cutoff, btype='low')
    
    # 应用滤波器
    y_bass = signal.filtfilt(b, a, y)
    
    return y_bass 