import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    应用配置设置
    """
    API_PREFIX: str = "/api"
    DEBUG: bool = Field(default=False)
    FRONTEND_URL: str = Field(default="http://localhost:3000")
    
    # LLM API配置
    ENABLE_LLM_ENHANCEMENT: bool = Field(default=False)  # 默认禁用LLM增强
    OPENAI_API_KEY: str = Field(default="")
    LLM_MODEL: str = Field(default="gpt-4") 
    USE_LLM_CACHE: bool = Field(default=True)
    
    # Music AI配置
    MUSIC_AI_API_KEY: str = Field(default="")
    MUSIC_AI_API_URL: str = Field(default="https://api.musicai.example.com")
    
    # 生产环境中从环境变量加载
    class Config:
        env_file = ".env"
        env_prefix = "APP_"
        case_sensitive = True

# 创建单例实例
settings = Settings(
    # 可以在开发环境中显式覆盖设置
    DEBUG=os.getenv("APP_DEBUG", "True").lower() in ("true", "1", "t"),
    ENABLE_LLM_ENHANCEMENT=os.getenv("APP_ENABLE_LLM_ENHANCEMENT", "False").lower() in ("true", "1", "t"),
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", ""),
    LLM_MODEL=os.getenv("APP_LLM_MODEL", "gpt-4"),
    USE_LLM_CACHE=os.getenv("APP_USE_LLM_CACHE", "True").lower() in ("true", "1", "t"),
    # Music AI配置
    MUSIC_AI_API_KEY=os.getenv("MUSIC_AI_API_KEY", ""),
    MUSIC_AI_API_URL=os.getenv("MUSIC_AI_API_URL", "https://api.musicai.example.com")
) 