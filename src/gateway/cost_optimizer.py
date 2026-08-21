"""Some cost optimization techniques

1 TOKEN BUDGET ENFORCEMENT
2 PROMPT COMPRESSION
3 SMART MODEL ROUTING
4 COST REPORTING """

from src.core.logger import get_logger
from src.gateway.llm_gateway import get_gateway

logger = get_logger(__name__)

# token budgets
MAX_TOKENS_PER_ARGUMENT=500  # user argument trunkated if longer
MAX_TOKENS_PER_SESSION=50000 # session HARD CAP
MAX_HISTORY_TOKENS=800 # DEBATE HISTORY CONTEXT CAP

#  smart routing threshold
STRONG_SCORE_THRESHOLD=7 # if recent avg score is above this use fast model
WEAK_SCORE_THRESHOLD=4

class CostOptimizer:
    """Applies cost optimization strategies to reduce token spend."""
    def __init__(self):
        self._session_tokens=0
        logger.info("CostOptimizer initialised")


    #  TOKEN BUDGET OPTIMISATION

    def enforce_argument_budget(self,argument:str)->str:
        """Truncate argument if it exceeds limit 
        rough token estimate 1 token = 4 characters"""

        max_chars=MAX_TOKENS_PER_ARGUMENT * 4
        if len(argument)<=max_chars:
            return argument
        truncated = argument[:max_chars]
        # cut at last word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:
            truncated = truncated[:last_space]

        logger.warning(
            f"Argument truncated: {len(argument)} → {len(truncated)} chars"
        )

        return truncated + "...[truncated for length]"

    def check_session_budget(self)-> bool:
        """Return True if session is within token budget."""
        summary = get_gateway().get_cost_summary()
        total = summary.get("total_tokens",0)
        if total >=MAX_TOKENS_PER_SESSION:
            logger.warning(
                f"Session token budget exceeded: "
                f"{total}/{MAX_TOKENS_PER_SESSION}"
            )
            return False
        return True

    # Prompt compression

    def compress_history(
            self,
            history:list[tuple[str,str]],
            
    )-> list[tuple[str,str]]:
        """compress debate history to fit within token budget . """
        if not history:
            return history

        total_chars = sum(len(content) for _,content in history)
        budget_chars = MAX_HISTORY_TOKENS * 4

        if total_chars <= budget_chars:
            return history

        compressed = history[-6:]
        logger.info(
            f"History compressed: {len(history)} → {len(compressed)} messages"
        )
        return compressed

    #  Smart Model Routing

    def get_debate_model_alias(
        self,
        recent_scores: list[int],
    ) -> str:
        """Choose model alias based on recent argument quality.

        Strong debater → fast model (cheaper, still sufficient)
        Weak debater   → powerful model (better coaching needed)
        Mixed/unknown  → powerful model (default)
        """
        if len(recent_scores) < 2:
            return "powerful"   # not enough data — default to powerful

        recent_avg = sum(recent_scores[-3:]) / len(recent_scores[-3:])

        if recent_avg >= STRONG_SCORE_THRESHOLD:
            logger.info(
                f"Smart routing: strong debater "
                f"(avg={recent_avg:.1f}) → fast model"
            )
            return "fast"

        if recent_avg <= WEAK_SCORE_THRESHOLD:
            logger.info(
                f"Smart routing: struggling debater "
                f"(avg={recent_avg:.1f}) → powerful model"
            )
            return "powerful"

        return "powerful"

    #  COST REPORTING

    def get_turn_cost_report(self)->dict:
        """Return cost breakdown for most recent turn"""
        summary = get_gateway().get_cost_summary()
        return {
            "total_tokens":   summary.get("total_tokens", 0),
            "total_cost_usd": summary.get("total_cost_usd", 0.0),
            "cache_hits":     summary.get("cache_hits", 0),
            "tokens_by_task": summary.get("tokens_by_task", {}),
        }

    def get_session_cost_report(self) -> dict:
        """Return full session cost summary with savings calculation."""
        summary   = get_gateway().get_cost_summary()
        total     = summary.get("total_tokens", 0)
        cost      = summary.get("total_cost_usd", 0.0)
        hits      = summary.get("cache_hits", 0)
        calls     = summary.get("total_calls", 0)

        hit_rate  = (hits / calls * 100) if calls > 0 else 0.0

        # Estimate savings from cache
        avg_tokens_per_call = (total / (calls - hits)) if (calls - hits) > 0 else 0
        saved_tokens = int(avg_tokens_per_call * hits)
        saved_usd    = saved_tokens / 1000 * 0.0003  # rough avg rate

        return {
            "total_tokens":      total,
            "total_cost_usd":    cost,
            "total_calls":       calls,
            "cache_hits":        hits,
            "cache_hit_rate_pct": round(hit_rate, 1),
            "estimated_saved_tokens": saved_tokens,
            "estimated_saved_usd":    round(saved_usd, 6),
        }

    def reset(self) -> None:
        """Reset session tracking."""
        self._session_tokens = 0
        logger.info("CostOptimizer reset.")