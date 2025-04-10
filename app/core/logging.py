import logging
import sys
from typing import List

# 日志格式
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    配置并返回一个logger实例

    Args:
        name: Logger名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        配置好的logger实例
    """
    # 解析日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # 避免重复处理器
    if not logger.handlers:
        # 控制台处理器
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(handler)
        
    return logger

# 根logger
logger = get_logger("audio_api") 