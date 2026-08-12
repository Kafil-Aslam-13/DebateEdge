# Settings , env vars
"""Central Configuration for DebateEdge.
Single place for all settings.
Every module calls get_settings() -  and never reads YAML or ENV directly"""

import os
from functools import lru_cache
import yaml
from dotenv import load_dotenv
from src.core.exceptions import ConfigurationError
from src.core.logger import get_logger
load_dotenv()
logger = get_logger(__name__)

def _load_yaml(path:str)->dict:
    if not os.path.exists(path):
        raise ConfigurationError(f"Config file not found: {path}")
    with open(path,"r") as f:
        return yaml.safe_load(f)

class Settings:
    def __init__(self)->None:
        _config = _load_yaml("configs/config.yaml")
        _models = _load_yaml("configs/models.yaml")

        # App
        self.app_name = _config["app"]["name"]
        self.version = _config["app"]["version"]
        self.environment = _config["app"]["environment"]

        # debate
        self.max_turns = _config["debate"]["max_turns"]
        self.default_side = _config["debate"]["default_side"]
        self.valid_sides = _config["debate"]["supported_sides"]

        # llm
        self.default_model = _config["llm"]["default_model"]
        self.complex_model = _config["llm"]["complex_model"]
        self.temperature = _config["llm"]["temperature"]
        self.max_tokens = _config["llm"]["max_tokens"]

        # Model routing
        self.model_config = _models["models"]
        self.fallback_models = _models.get("fallbacks", [])

        # Gateway Config 
        self.task_routing=_models.get("task_routing",{})
        self.gateway_config=_models.get("gateway",{}) 

        # Observability
        self.langsmith_enabled = _config["observability"]["langsmith"]["enabled"]
        self.langsmith_project = _config["observability"]["langsmith"]["project"]
        self.logfire_enabled = _config["observability"]["logfire"]["enabled"]
        self.logfire_service = _config["observability"]["logfire"]["service_name"]

        # Memory
        self.buffer_max_messages = _config["memory"]["buffer_max_messages"]
        self.summary_max_tokens = _config["memory"]["summary_max_tokens"]

        # Retrieval
        self.chroma_collection = _config["retrieval"]["chroma"]["collection_name"]
        self.chroma_top_k = _config["retrieval"]["chroma"]["top_k"]
        self.pinecone_index = _config["retrieval"]["pinecone"]["index_name"]
        self.pinecone_top_k = _config["retrieval"]["pinecone"]["top_k"]
        self.pinecone_namespace = _config["retrieval"]["pinecone"]["namespace"]

        # API Keys from env
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.langsmith_api_key = os.getenv("LANGCHAIN_API_KEY", "")
        self.logfire_token = os.getenv("LOGFIRE_TOKEN", "")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.cohere_api_key = os.getenv("COHERE_API_KEY", "")

        # Validate critical keys
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY not found in .env. "
                "Get a free key at console.groq.com"
            )


    def __repr__(self) -> str:
        return(
            f"Settings(app={self.app_name}),"
            f"env={self.environment},"
            f"model={self.default_model}"
        )

@lru_cache
def get_settings()->Settings:
    """return cached singleton settings instance"""
    return Settings()