"""Embeddings Model for RAG.
TWO MODELS USED:
1. HUGGINGFACEEMBEDDINGS(LOCAL FREE)
2.COHEREEMBEDDINGS(API,Richer semantic understanding)
Different retrieval contexts need different embedding quality.
pinecone is presistant and quality matters more than speed

Both are Singletons:
Build once and reused. Embedding models are heavy to load"""

from functools import lru_cache
from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

_HF_MODEL="sentence-transformers/all-MiniLM-L6-v2"
_COHERE_MODEL = "embed-english-v3.0"

@lru_cache(maxsize=1)
def get_hf_embeddings()->HuggingFaceEmbeddings:
    """Return hf embeddings . no api key - free local
    Used with Chroma db for fast in session retrieval
    Cached: model loads once , reused for all chromabd operations"""
    logger.info(f"Loading Huggingface embeddings | model = {_HF_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=_HF_MODEL,
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@lru_cache(maxsize=1)
def get_cohere_embeddings()-> CohereEmbeddings:
    """Return Cohere Embeddings - API - BASED , RICHER UNDERSTANDING
    Used with Piconebfor presistant cross-session retrieval
    Cached: 1 Instance per process 
    Requires api key in .env . fallbacks if key not set - returns HFD Embeddings"""

    settings=get_settings()
    if not settings.cohere_api_key:
        logger.warning("Cohere Api Key Not set- fallback to HFEmbeddings")
        return get_hf_embeddings()

    logger.info(F"loading cohere embeddings | model = {_COHERE_MODEL}")
    return CohereEmbeddings(
        model=_COHERE_MODEL,
        cohere_api_key=settings.cohere_api_key
    )

