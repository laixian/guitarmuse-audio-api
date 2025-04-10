import json
import os
import hashlib
import pickle
import re
import requests
from typing import Dict, Any, Optional, List, Tuple
from app.models.schemas import AudioAnalysisResult, SongStructure, Measure
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

class LLMCache:
    """LLM响应缓存"""
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, prompt: str) -> str:
        """生成缓存键"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """获取缓存的响应"""
        key = self.get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                logger.info(f"从缓存获取LLM响应: {key}")
                return cached_data
            except Exception as e:
                logger.error(f"读取缓存失败: {str(e)}")
                return None
        return None
    
    def set(self, prompt: str, response: str) -> None:
        """缓存响应"""
        key = self.get_cache_key(prompt)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(response, f)
            logger.info(f"缓存LLM响应: {key}")
        except Exception as e:
            logger.error(f"缓存LLM响应失败: {str(e)}")


class LLMEnhancer:
    """使用LLM增强音频分析的组件"""
    
    def __init__(self, use_cache: bool = True):
        """初始化增强器"""
        self.api_key = os.getenv("OPENAI_API_KEY", getattr(settings, "OPENAI_API_KEY", ""))
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = os.getenv("LLM_MODEL", getattr(settings, "LLM_MODEL", "gpt-4"))
        self.use_cache = use_cache
        self.cache = LLMCache() if use_cache else None
    
    def enhance_analysis(self, analysis: AudioAnalysisResult) -> AudioAnalysisResult:
        """使用LLM增强音频分析结果"""
        try:
            if not self.api_key:
                logger.warning("未配置API密钥，跳过LLM增强")
                return analysis
                
            # 创建提示词 - 使用内部定义的简化版提示词函数
            prompt = self._create_simple_analysis_prompt(analysis)
            
            # 检查缓存
            cached_response = None
            if self.use_cache and self.cache:
                cached_response = self.cache.get(prompt)
            
            # 如果有缓存结果，直接使用
            if cached_response:
                llm_response = cached_response
            else:
                # 调用LLM API
                llm_response = self._call_llm_api(prompt)
                
                # 缓存结果
                if llm_response and self.use_cache and self.cache:
                    self.cache.set(prompt, llm_response)
            
            # 解析结果
            if llm_response:
                enhanced_result = self._parse_response(llm_response, analysis)
                # 记录增强前后的变化
                self._log_enhancement_changes(analysis, enhanced_result)
                logger.info("LLM增强分析完成")
                return enhanced_result
            else:
                logger.warning("LLM返回空响应，使用原始分析结果")
                return analysis
                
        except Exception as e:
            logger.error(f"LLM增强失败: {str(e)}", exc_info=True)
            # 发生错误时返回原始分析结果
            return analysis
    
    def _create_simple_analysis_prompt(self, analysis: AudioAnalysisResult) -> str:
        """
        创建高质量的音乐结构分析提示词，专注于让LLM进行专业的段落规划
        """
        # 格式化小节数据
        measures_text = ""
        if analysis.structures and len(analysis.structures) > 0:
            measures = analysis.structures[0].measures[:16]
            for m in measures:
                measures_text += f"小节{m.number}: 和弦={m.chord}, 开始={m.startTime:.2f}s, 结束={m.endTime:.2f}s\n"
            
            if len(analysis.structures[0].measures) > 16:
                measures_text += f"... (共{len(analysis.structures[0].measures)}个小节)"
                
            # 添加和弦列表，便于LLM分析
            all_chords = [m.chord for m in analysis.structures[0].measures]
            chord_sequence = ", ".join(all_chords[:32])
            if len(all_chords) > 32:
                chord_sequence += "..."
        
        # 计算小节总数
        total_measures = len(analysis.structures[0].measures) if analysis.structures and len(analysis.structures) > 0 else 0
        
        prompt = f"""
        作为专业音乐制作人和音乐理论专家，请对以下音频分析数据进行专业分析和优化。你拥有丰富的流行音乐、摇滚、爵士等多种风格的编曲和制作经验。

        ## 基本信息
        - 调性: {analysis.key}
        - 速度: {analysis.tempo} BPM
        - 小节总数: {total_measures}
        
        ## 和弦序列
        {chord_sequence}
        
        ## 小节详细信息
        {measures_text}
        
        ## 背景知识
        流行音乐通常遵循以下结构模式:
        1. Intro (前奏): 通常4-8小节，建立歌曲的和声基础和氛围
        2. Verse (主歌): 通常8-16小节，是歌词讲述故事的部分
        3. Pre-Chorus (副歌前): 通常4-8小节，作为主歌到副歌的过渡
        4. Chorus (副歌): 通常8-16小节，是歌曲的高潮和主题部分
        5. Bridge (桥段): 通常4-8小节，提供对比和变化
        6. Outro (尾声): 通常4-8小节，结束歌曲

        每个段落的小节数通常是4的倍数，这是西方音乐中的常见结构单位。

        常见的段落组合:
        - 标准流行: Intro → Verse → Chorus → Verse → Chorus → Bridge → Chorus → Outro
        - 简约结构: Intro → Verse → Chorus → Verse → Chorus → Outro
        - AABA结构: Verse → Verse → Bridge → Verse → Outro
        
        ## 分析任务
        请基于音乐理论和专业判断，完成以下工作:
        
        1. 验证调性判断是否准确，如果不准确，请给出更合适的调性。

        2. 如果发现和弦序列中存在不属于当前调性顺阶和弦的和弦，请将其替换为符合规律的和弦。
        
        3. 将小节序列划分为音乐段落(Intro, Verse, Chorus, Bridge, Outro等)，考虑:
           - 和弦进行的变化和重复
           - 典型的段落长度(4的倍数)
           - 主流流行音乐的结构模式
           - 段落间的和声逻辑过渡
        
        4. 确保每个段落的小节数是4的倍数，并给出合理的结构划分。
        
        5. 优化和弦进行:
           - 检查并纠正可能的和弦误判，比如通常段落开始第一个和弦应该是主和弦
           - 使每个段落的和弦进行符合该调性下的常见进行模式
           - 确保段落之间的和声衔接自然流畅
           - 利用和弦功能理论(主和弦、下属和弦、属和弦等)确保和声进行符合音乐理论
        
        ## 输出格式要求
        请以标准JSON格式返回分析结果，包含:

        ```json
        {{
          "key": "调性，如C或Am",
          "structures": [
            {{
              "type": "段落类型，如Intro/Verse/Chorus/Bridge/Outro",
              "startMeasure": 起始小节编号(从1开始),
              "endMeasure": 结束小节编号,
              "measures": [
                "小节1和弦",
                "小节2和弦",
                ... (该段落所有小节的和弦)
              ]
            }},
            ... (更多段落)
          ]
        }}
        ```

        重要提示:
        - 请确保JSON格式有效，可以被直接解析
        - 结构划分要合理，符合流行音乐制作规范
        - 要考虑音乐的叙事流程和情感递进
        - 每个段落长度应是4的倍数(4、8、12、16等)
        - 所有小节编号必须连续且不重叠
        - 确保返回的总小节数与原始小节数相匹配
        """
        
        return prompt
    
    def _create_prompt(self, analysis: AudioAnalysisResult) -> str:
        """创建提示词"""
        # 格式化结构数据
        structures_text = ""
        for i, struct in enumerate(analysis.structures):
            chords = [f"{m.number}: {m.chord}" for m in struct.measures[:8]]
            chords_text = ", ".join(chords)
            if len(struct.measures) > 8:
                chords_text += f", ... (共{len(struct.measures)}小节)"
            
            structures_text += f"段落{i+1}: 类型={struct.type}, 起始时间={struct.startTime:.2f}s, "
            structures_text += f"结束时间={struct.endTime:.2f}s, 小节数={len(struct.measures)}, 和弦=[{chords_text}]\n"
        
        prompt = f"""
        你是一位专业的音频处理专家和音乐理论专家，你很擅长发现和弦进行编码规律，请分析并优化以下音频分析结果。

        ## 基本信息
        - 调性: {analysis.key}
        - 速度: {analysis.tempo} BPM
        
        ## 当前识别的结构
        {structures_text}
        
        ## 任务
        请基于音乐理论和作曲惯例，提供以下优化：
        
        1. 验证调性判断是否合适，必要时建议更合适的调性
        2. 确保每个段落的小节数是4的倍数，符合常规音乐创作习惯
        3. 检查并改进每个段落的和弦进行，使其更符合该调性下的常见和弦进行模式
        4. 确保和弦进行在音乐上是连贯的，特别注意段落之间的衔接
        5. 根据段落类型(Intro/Verse/Chorus/Bridge/Outro等)建议更适合的和弦走向
        6. 如果发现段落中有不属于当前调性顺阶和弦的和弦，请将其替换为符合规律的和弦

        ## 输出格式
        请以JSON格式返回优化结果，包含以下字段：
        - key: 建议的调性
        - structures: 优化后的段落数组，每个段落包含type, measures字段
        
        确保只返回JSON格式，不要有其他解释性文字。
        """
        
        return prompt
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API"""
        if not self.api_key:
            logger.error("缺少API密钥，无法调用LLM")
            return None
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # 降低随机性
            "max_tokens": 2000
        }
        
        try:
            logger.info(f"调用LLM API: 模型={self.model}")
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info("LLM API调用成功")
                return content
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"API请求异常: {str(e)}")
            return None
    
    def _parse_response(self, response: str, original: AudioAnalysisResult) -> AudioAnalysisResult:
        """解析LLM响应并生成增强的分析结果"""
        try:
            # 尝试提取JSON部分
            data = self._extract_structured_data(response)
            
            if not data:
                logger.warning("无法从LLM响应中提取结构化数据")
                return original
                
            logger.info("成功解析LLM响应数据")
            logger.info(f"LLM响应数据: {data}")
            
            # 更新调性
            enhanced = AudioAnalysisResult(
                key=data.get("key", original.key),
                tempo=original.tempo,
                structures=[],  # 将重建结构
                beats=original.beats
            )
            
            # 如果返回了结构信息，构建新的结构
            if "structures" in data and isinstance(data["structures"], list):
                new_structures = []
                
                # 获取所有原始小节
                all_measures = []
                for struct in original.structures:
                    all_measures.extend(struct.measures)
                
                # 根据LLM规划的段落构建新结构
                for struct_data in data["structures"]:
                    try:
                        # 获取段落类型
                        section_type = struct_data.get("type", "Unknown")
                        
                        # 获取起始和结束小节编号
                        start_measure = int(struct_data.get("startMeasure", 1))
                        end_measure = int(struct_data.get("endMeasure", len(all_measures)))
                        
                        # 确保范围有效
                        start_idx = max(0, start_measure - 1)  # 转为0索引
                        end_idx = min(len(all_measures) - 1, end_measure - 1)
                        
                        if start_idx > end_idx or start_idx >= len(all_measures):
                            continue
                        
                        # 获取段落中的小节
                        section_measures = all_measures[start_idx:end_idx+1]
                        
                        # 如果LLM提供了和弦信息，更新和弦
                        if "measures" in struct_data and isinstance(struct_data["measures"], list):
                            chords = struct_data["measures"]
                            for i, chord in enumerate(chords):
                                if i < len(section_measures):
                                    if isinstance(chord, dict) and "chord" in chord:
                                        section_measures[i].chord = chord["chord"]
                                    elif isinstance(chord, str):
                                        section_measures[i].chord = chord
                        
                        # 创建段落结构
                        new_struct = SongStructure(
                            type=section_type,
                            startTime=section_measures[0].startTime,
                            endTime=section_measures[-1].endTime,
                            measures=section_measures
                        )
                        
                        new_structures.append(new_struct)
                        
                    except Exception as e:
                        logger.error(f"处理段落数据时出错: {str(e)}")
                        continue
                
                if new_structures:
                    enhanced.structures = new_structures
                else:
                    # 如果无法创建新结构，保留原始结构
                    enhanced.structures = original.structures
            else:
                # 如果没有结构信息，保留原始结构
                enhanced.structures = original.structures
            
            return enhanced
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {str(e)}", exc_info=True)
            return original
    
    def _extract_structured_data(self, text: str) -> Optional[Dict]:
        """
        从文本中提取结构化数据，优先尝试提取JSON，
        如果失败则尝试从文本中提取关键信息
        """
        # 首先尝试提取JSON
        json_data = self._extract_json(text)
        if json_data:
            try:
                return json.loads(json_data)
            except json.JSONDecodeError:
                logger.warning("无法解析提取的JSON")
        
        # 如果无法提取JSON，尝试从文本中提取结构信息
        logger.info("尝试从文本中提取结构信息")
        try:
            structured_data = self._extract_from_text(text)
            if structured_data:
                return structured_data
        except Exception as e:
            logger.error(f"从文本提取结构失败: {str(e)}")
        
        # 所有方法都失败
        return None
    
    def _extract_from_text(self, text: str) -> Optional[Dict]:
        """
        从非JSON格式的文本中提取结构化信息
        """
        result = {"structures": []}
        
        # 尝试提取调性信息
        key_match = re.search(r'调性[：:]?\s*([A-G](#|b)?m?)', text)
        if key_match:
            result["key"] = key_match.group(1)
        
        # 尝试提取段落信息
        # 寻找类似 "Intro (小节1-8)" 或 "Verse: 9-16" 的模式
        section_patterns = [
            r'(Intro|Verse|Chorus|Bridge|Outro)[^0-9]*?(\d+)[^0-9]*?(\d+)',
            r'(前奏|主歌|副歌|间奏|尾声)[^0-9]*?(\d+)[^0-9]*?(\d+)'
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                section_type = match.group(1)
                # 将中文段落名转换为英文
                if section_type == "前奏":
                    section_type = "Intro"
                elif section_type == "主歌":
                    section_type = "Verse"
                elif section_type == "副歌":
                    section_type = "Chorus"
                elif section_type == "间奏":
                    section_type = "Bridge"
                elif section_type == "尾声":
                    section_type = "Outro"
                
                start_measure = int(match.group(2))
                end_measure = int(match.group(3))
                
                # 提取这个段落附近的和弦信息
                # 假设和弦信息可能跟在段落信息后面
                chords = []
                section_text = text[match.end():match.end() + 500]  # 查看后面的500个字符
                chord_matches = re.finditer(r'[A-G](#|b)?m?(7|maj7|min7|aug|dim)?', section_text)
                for chord_match in chord_matches:
                    chords.append(chord_match.group(0))
                
                # 限制和弦数量与小节数量匹配
                measure_count = end_measure - start_measure + 1
                if len(chords) > measure_count:
                    chords = chords[:measure_count]
                
                # 如果找不到足够的和弦，填充空字符串
                while len(chords) < measure_count:
                    chords.append("")
                
                section = {
                    "type": section_type,
                    "startMeasure": start_measure,
                    "endMeasure": end_measure,
                    "measures": chords
                }
                
                result["structures"].append(section)
        
        # 如果找到结构，返回结果
        if result["structures"]:
            return result
        
        return None
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON部分"""
        try:
            # 尝试找到JSON开始和结束的位置
            start = text.find("{")
            end = text.rfind("}")
            
            if start >= 0 and end > start:
                json_str = text[start:end+1]
                # 验证是否为有效JSON
                json.loads(json_str)
                return json_str
            else:
                logger.warning("无法在响应中找到JSON")
                return None
        except json.JSONDecodeError:
            logger.error("提取的文本不是有效的JSON")
            return None
    
    def _log_enhancement_changes(self, original: AudioAnalysisResult, enhanced: AudioAnalysisResult):
        """记录LLM增强前后的变化"""
        if original.key != enhanced.key:
            logger.info(f"调性变更: {original.key} -> {enhanced.key}")
        
        if len(original.structures) != len(enhanced.structures):
            logger.info(f"段落数量变更: {len(original.structures)} -> {len(enhanced.structures)}")
        
        # 记录和弦变化
        chord_changes = 0
        for i, (orig, enh) in enumerate(zip(original.structures, enhanced.structures)):
            if orig.type != enh.type:
                logger.info(f"段落类型变更: 段落{i+1} {orig.type} -> {enh.type}")
                
            for j, (om, em) in enumerate(zip(orig.measures, enh.measures)):
                if om.chord != em.chord:
                    chord_changes += 1
                    
        logger.info(f"和弦变更: 共{chord_changes}处")


# 单例模式封装
_llm_enhancer = None

def get_llm_enhancer() -> LLMEnhancer:
    """获取LLM增强器实例"""
    global _llm_enhancer
    if _llm_enhancer is None:
        _llm_enhancer = LLMEnhancer()
    return _llm_enhancer 