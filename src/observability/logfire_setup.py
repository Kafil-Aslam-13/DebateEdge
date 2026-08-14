# logfire config + instrumentation
"""Logfire setup
Its Pydantics observability platform

WHAT GETS INSTRUMENTED AUTOMATICALLY:
- logfire.instrument_httpx()    → all outbound HTTP calls
                                   includes every Groq API call
                                   you see latency, status, tokens
- logfire.instrument_pydantic() → all Pydantic model validations
                                   see FallacyDetectionResult parsing
                                   see guardrail schema validation
                                   
SETUP RULE:
logfire.configure() MUST be called before instrument_*().
instrument_*() MUST be called before the app starts.
Call setup_logfire() in main.py before anything else.
"""

import os , logfire
from src.core.config import get_settings
from src.core.logger import get_logger

logger=get_logger(__name__)

def setup_logfire()->bool:
    """configure logfire and instrument libraries.
    """
    settings=get_settings()

    if not settings.logfire_enabled:
        logger.info("Logfire disabled in config.yaml - skipping setup")
        return False

    token = os.getenv("LOGFIRE_TOKEN","")

    if not token:
        logger.warning("Logfire token not set ")
        return False
    try:
        scrub_fields=getattr(settings,"logfire_scrub_fields",[])

        logfire.configure(token=token,service_name=settings.logfire_service,
                          environment=settings.environment,scrubbing=logfire.ScrubbingOptions(extra_patterns=scrub_fields),)
        

        # 2. Instrument libraries AFTER configure
        # httpx: captures every outbound HTTP call
        # including all Groq API calls via litellm
        logfire.instrument_httpx()

        # pydantic: captures all model validations
        # FallacyDetectionResult, guardrail schemas etc
        logfire.instrument_pydantic()

        logger.info(
            f"Logfire configured | "
            f"service={settings.logfire_service} | "
            f"env={settings.environment}"
        )
        return True
    except Exception as e:
        logger.warning(f"Logfire setup failed: {e} — continuing without it.")
        return False

def get_logfire_span(name: str, **attributes):
    """Return a logfire span context manager.

    Usage:
        with get_logfire_span("classify_argument", topic=topic):
            result = classifier.invoke(...)

    If Logfire is not configured, returns a no-op context manager
    so the rest of the code never needs to know.
    """
    try:
        return logfire.span(name, **attributes)
    except Exception:
        # Return no-op context manager if Logfire not available
        from contextlib import nullcontext
        return nullcontext()