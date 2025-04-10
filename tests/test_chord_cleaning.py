import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.schemas import AudioAnalysisResult, SongStructure, Measure, BeatInfo
from app.services.audio_analyzer import clean_short_duration_chords
from app.services.music_ai_analyzer import MusicAIAnalyzer

class TestChordCleaning(unittest.TestCase):
    """测试和弦清洗功能"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试用的音频分析结果
        self.measures = [
            Measure(number=1, chord="C", startTime=0.0, endTime=2.0, barSpan=1, bars=1),
            Measure(number=2, chord="G", startTime=2.0, endTime=2.5, barSpan=1, bars=1),  # 短和弦，应被合并
            Measure(number=3, chord="Am", startTime=2.5, endTime=4.0, barSpan=1, bars=1),
            Measure(number=4, chord="F", startTime=4.0, endTime=4.8, barSpan=1, bars=1),  # 短和弦，应被合并
            Measure(number=5, chord="G", startTime=4.8, endTime=6.0, barSpan=1, bars=1)
        ]
        
        self.structure = SongStructure(
            type="Verse",
            startTime=0.0,
            endTime=6.0,
            measures=self.measures
        )
        
        self.beats_info = BeatInfo(
            positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            bpm=120.0,
            timeSignature="4/4",
            beatsPerBar=4
        )
        
        self.analysis_result = AudioAnalysisResult(
            key="C",
            tempo=120.0,
            structures=[self.structure],
            beats=self.beats_info
        )
    
    def test_clean_short_duration_chords(self):
        """测试清洗短和弦函数"""
        # 清洗和弦
        cleaned_result = clean_short_duration_chords(self.analysis_result, min_duration=1.0)
        
        # 验证结果
        # 1. 应该只剩三个小节
        self.assertEqual(len(cleaned_result.structures[0].measures), 3, "应该只有三个小节")
        
        # 2. 验证合并后的小节信息
        measures = cleaned_result.structures[0].measures
        
        # 第一个小节应该是C，但结束时间延长了
        self.assertEqual(measures[0].chord, "C")
        self.assertEqual(measures[0].startTime, 0.0)
        self.assertEqual(measures[0].endTime, 2.5)  # 合并了G
        self.assertEqual(measures[0].barSpan, 2)  # 跨度增加了
        
        # 第二个小节应该是Am，但结束时间延长了
        self.assertEqual(measures[1].chord, "Am")
        self.assertEqual(measures[1].startTime, 2.5)
        self.assertEqual(measures[1].endTime, 4.8)  # 合并了F
        self.assertEqual(measures[1].barSpan, 2)  # 跨度增加了
        
        # 第三个小节应该是G
        self.assertEqual(measures[2].chord, "G")
        self.assertEqual(measures[2].startTime, 4.8)
        self.assertEqual(measures[2].endTime, 6.0)
        self.assertEqual(measures[2].barSpan, 1)
    
    def test_music_ai_analyzer_merge_short_chords(self):
        """测试MusicAIAnalyzer中的短和弦合并"""
        # 创建分析器实例
        analyzer = MusicAIAnalyzer()
        
        # 调用合并函数
        merged_structures = analyzer._merge_short_chords([self.structure], min_duration=1.0)
        
        # 验证结果
        # 1. 应该只剩三个小节
        self.assertEqual(len(merged_structures[0].measures), 3, "应该只有三个小节")
        
        # 2. 验证合并后的小节信息
        measures = merged_structures[0].measures
        
        # 第一个小节应该是C，但结束时间延长了
        self.assertEqual(measures[0].chord, "C")
        self.assertEqual(measures[0].startTime, 0.0)
        self.assertEqual(measures[0].endTime, 2.5)  # 合并了G
        self.assertEqual(measures[0].barSpan, 2)  # 跨度增加了
        
        # 第二个小节应该是Am，但结束时间延长了
        self.assertEqual(measures[1].chord, "Am")
        self.assertEqual(measures[1].startTime, 2.5)
        self.assertEqual(measures[1].endTime, 4.8)  # 合并了F
        self.assertEqual(measures[1].barSpan, 2)  # 跨度增加了
        
        # 第三个小节应该是G
        self.assertEqual(measures[2].chord, "G")
        self.assertEqual(measures[2].startTime, 4.8)
        self.assertEqual(measures[2].endTime, 6.0)
        self.assertEqual(measures[2].barSpan, 1)


if __name__ == "__main__":
    unittest.main() 