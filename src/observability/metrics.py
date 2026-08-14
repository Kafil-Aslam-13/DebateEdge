# custom debate metrics -> debate_score , fallacies_caught etc

"""Custom Logfire metrics — Sprint 8.

Business-level metrics for DebateEdge.
These appear as time-series charts in the Logfire dashboard.

METRIC TYPES USED:
- metric_gauge    → current value at a point in time
                    use for: argument score per turn
- metric_counter  → cumulative count that only goes up
                    use for: total fallacies caught, cache hits
- metric_histogram → distribution of values
                     use for: latency, token counts

WHY CUSTOM METRICS BEYOND AUTO-INSTRUMENTATION:
Auto-instrumentation gives you HTTP latency and error rates.
Custom metrics give you BUSINESS insight:
- Are users improving? (score trend per session)
- Which fallacies are most common? (fallacy_name counter)
- Is RAG helping? (rag_docs_retrieved per turn)
- Is the cache saving cost? (cache_hit_rate)

These are the metrics a real ML product team would track.
"""

from src.core.constants import (
    METRIC_DEBATE_SCORE,
    METRIC_FALLACY_COUNT,
    METRIC_LLM_COST,
    METRIC_LLM_TOKENS,
    METRIC_SESSION_TURNS,
)
from src.core.logger import get_logger

logger = get_logger(__name__)

_logfire_available = False

try:
    import logfire
    _logfire_available = True
except ImportError:
    pass


def _safe_metric(fn_name: str, *args, **kwargs) -> None:
    """Call a logfire metric function safely.

    No-op if logfire not available or not configured.
    """
    if not _logfire_available:
        return
    try:
        fn = getattr(logfire, fn_name)
        fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"Metric {fn_name} failed: {e}")


def record_argument_score(
    score: int,
    quality: str,
    turn_number: int,
    topic: str,
) -> None:
    """Record argument score as a gauge metric.

    Shows score trend across turns in Logfire dashboard.
    Useful for tracking if user is improving.
    """
    _safe_metric(
        "metric_gauge",
        METRIC_DEBATE_SCORE,
        score,
        attributes={
            "quality":     quality,
            "turn_number": turn_number,
            "topic":       topic[:50],
        },
    )


def record_fallacy_detected(
    fallacy_name: str,
    severity: str,
    turn_number: int,
) -> None:
    """Increment fallacy counter.

    Shows which fallacies are most common across all sessions.
    """
    _safe_metric(
        "metric_counter",
        METRIC_FALLACY_COUNT,
        attributes={
            "fallacy_name": fallacy_name,
            "severity":     severity,
            "turn_number":  turn_number,
        },
    )


def record_llm_cost(
    cost_usd: float,
    tokens: int,
    task_type: str,
    model: str,
    cached: bool,
) -> None:
    """Record LLM token usage and estimated cost.

    Feeds into cost management (Sprint 10).
    """
    _safe_metric(
        "metric_gauge",
        METRIC_LLM_COST,
        cost_usd,
        attributes={
            "task_type": task_type,
            "model":     model,
            "cached":    cached,
        },
    )
    _safe_metric(
        "metric_counter",
        METRIC_LLM_TOKENS,
        attributes={
            "task_type": task_type,
            "model":     model,
        },
    )


def record_session_turn(
    turn_number: int,
    topic: str,
) -> None:
    """Record debate turn count per session."""
    _safe_metric(
        "metric_gauge",
        METRIC_SESSION_TURNS,
        turn_number,
        attributes={"topic": topic[:50]},
    )


def record_rag_retrieval(
    chroma_count: int,
    pinecone_count: int,
) -> None:
    """Record how many RAG docs were retrieved per turn."""
    _safe_metric(
        "metric_gauge",
        "debate.rag.chroma_docs",
        chroma_count,
    )
    _safe_metric(
        "metric_gauge",
        "debate.rag.pinecone_docs",
        pinecone_count,
    )


def record_guard_result(
    guard_name: str,
    action: str,
    guard_type: str,
) -> None:
    """Record guardrail check result."""
    _safe_metric(
        "metric_counter",
        "debate.guardrail.check",
        attributes={
            "guard_name": guard_name,
            "action":     action,
            "guard_type": guard_type,
        },
    )