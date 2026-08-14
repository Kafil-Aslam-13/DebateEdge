# langsmith tracing config
"""LangSmith setup — Sprint 8.

LangSmith traces every LangChain chain and LLM call automatically
when the right environment variables are set.

WHAT LANGSMITH TRACES AUTOMATICALLY:
-  full chain trace
- Every ChatGroq call via litellm    → token counts, latency

WHAT YOU SEE IN LANGSMITH UI:
- Input/output for every chain run
- Token usage per call
- Latency per step
- Error traces with full context
- Runs grouped by session
Enough for development and demo purposes.
"""

import os

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith() -> bool:
    """Set LangSmith environment variables to enable tracing.

    Must be called before any LangChain imports in the process.
    Returns True if setup succeeded, False if key not set.
    Degrades gracefully.
    """
    settings = get_settings()

    if not settings.langsmith_enabled:
        logger.info("LangSmith disabled in config.yaml — skipping.")
        return False

    api_key = os.getenv("LANGSMITH_API_KEY", "")

    if not api_key:
        logger.warning(
            "LANGCHAIN_API_KEY not set — LangSmith tracing disabled. "
            "Get a free key at smith.langchain.com"
        )
        return False

    # LangSmith is configured entirely via environment variables
    # LangChain reads these automatically on import
    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
    os.environ["LANGSMITH_API_KEY"]     = api_key
    os.environ["LANGCHAIN_PROJECT"]     = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

    logger.info(
        f"LangSmith tracing enabled | "
        f"project={settings.langsmith_project}"
    )
    return True
