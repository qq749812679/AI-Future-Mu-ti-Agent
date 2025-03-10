"""
配置管理模块 - 管理系统的全局配置参数
支持从环境变量、配置文件和默认值加载配置
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json
import toml

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        # 默认配置
        self.defaults = {
            # LLM配置
            "llm": {
                "model": "gpt-3.5-turbo",
                "base_url": "https://api.openai.com/v1",
                "api_key": "",
                "temperature": 0.0,
                "max_tokens": 4096,
                "api_type": "openai"  # openai 或 azure
            },
            
            # RAG系统配置
            "rag": {
                "vector_db_type": "chroma",
                "embedding_model": "openai",
                "embedding_model_name": "text-embedding-ada-002",
                "collection_name": "enterprise_knowledge",
                "persist_directory": "./data/vector_db",
                "search_k": 5,
                "hybrid_search": True
            },
            
            # MCP系统配置
            "mcp": {
                "message_queue_size": 1000,
                "message_workers": 3,
                "default_timeout": 30,
                "task_retry_count": 2
            },
            
            # 日志配置
            "logging": {
                "level": "INFO",
                "log_file": "app.log",
                "console_log": True,
                "file_log": True,
                "max_log_size_mb": 10,
                "backup_count": 3
            },
            
            # 系统配置
            "system": {
                "data_dir": "./data",
                "max_parallel_tasks": 10,
                "agent_heartbeat_interval": 30,
                "analytics_enabled": True
            }
        }
        
        # 存储最终配置
        self.config = {}
        
        # 加载配置
        self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str] = None):
        """加载配置，按优先级：默认值 < 配置文件 < 环境变量"""
        # 加载默认配置
        self.config = self.defaults.copy()
        
        # 加载配置文件
        if config_path:
            loaded_config = self._load_from_file(config_path)
            if loaded_config:
                self._deep_update(self.config, loaded_config)
        
        # 加载环境变量（优先级最高）
        self._load_from_env()
        
        logger.info("Configuration loaded successfully")
    
    def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """从配置文件加载配置"""
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}")
                return {}
            
            ext = os.path.splitext(config_path)[1].lower()
            
            if ext == ".json":
                with open(config_path, 'r') as f:
                    return json.load(f)
            
            elif ext == ".toml":
                return toml.load(config_path)
            
            else:
                logger.warning(f"Unsupported config file format: {ext}")
                return {}
        
        except Exception as e:
            logger.error(f"Error loading config from file: {str(e)}")
            return {}
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # LLM配置
        self._update_from_env("llm.model", "LLM_MODEL")
        self._update_from_env("llm.base_url", "LLM_BASE_URL")
        self._update_from_env("llm.api_key", "LLM_API_KEY")
        self._update_from_env("llm.temperature", "LLM_TEMPERATURE", float)
        self._update_from_env("llm.max_tokens", "LLM_MAX_TOKENS", int)
        self._update_from_env("llm.api_type", "LLM_API_TYPE")
        
        # RAG配置
        self._update_from_env("rag.vector_db_type", "RAG_VECTOR_DB_TYPE")
        self._update_from_env("rag.embedding_model", "RAG_EMBEDDING_MODEL")
        self._update_from_env("rag.collection_name", "RAG_COLLECTION_NAME")
        self._update_from_env("rag.persist_directory", "RAG_PERSIST_DIRECTORY")
        self._update_from_env("rag.search_k", "RAG_SEARCH_K", int)
        self._update_from_env("rag.hybrid_search", "RAG_HYBRID_SEARCH", bool)
        
        # 系统配置
        self._update_from_env("system.data_dir", "SYSTEM_DATA_DIR")
        self._update_from_env("system.max_parallel_tasks", "SYSTEM_MAX_PARALLEL_TASKS", int)
    
    def _update_from_env(self, config_key: str, env_var: str, type_cast=None):
        """从环境变量更新指定配置项"""
        value = os.environ.get(env_var)
        if value is not None:
            # 类型转换
            if type_cast == int:
                value = int(value)
            elif type_cast == float:
                value = float(value)
            elif type_cast == bool:
                value = value.lower() in ("true", "1", "yes", "y", "t")
            
            # 更新配置
            parts = config_key.split(".")
            conf = self.config
            for part in parts[:-1]:
                if part not in conf:
                    conf[part] = {}
                conf = conf[part]
            
            conf[parts[-1]] = value
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """深度更新字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def get(self, key: str, default=None) -> Any:
        """获取配置项"""
        parts = key.split(".")
        conf = self.config
        
        for part in parts:
            if part not in conf:
                return default
            conf = conf[part]
        
        return conf
    
    def set(self, key: str, value: Any):
        """设置配置项"""
        parts = key.split(".")
        conf = self.config
        
        for part in parts[:-1]:
            if part not in conf:
                conf[part] = {}
            conf = conf[part]
        
        conf[parts[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置区段"""
        if section in self.config:
            return self.config[section]
        return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """返回完整配置字典"""
        return self.config.copy()
    
    def save_to_file(self, file_path: str) -> bool:
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == ".json":
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            elif ext == ".toml":
                with open(file_path, 'w') as f:
                    toml.dump(self.config, f)
            
            else:
                logger.warning(f"Unsupported config file format: {ext}")
                return False
            
            logger.info(f"Configuration saved to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving config to file: {str(e)}")
            return False


# 创建一个全局配置实例
config = Config()

def get_config() -> Config:
    """获取全局配置实例"""
    return config

def load_config(config_path: str) -> Config:
    """加载指定配置文件并返回配置实例"""
    global config
    config = Config(config_path)
    return config 