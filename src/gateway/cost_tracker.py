"""Cost and Token tracker
Tracks every LLM Call made through the gateway:
- tokens used (prompt + completion)
- estimated cost in usd
-calls per task type
- calls per session

WHY TRACK COST:
- BCZ OF RATE LIMITS ,
- For cost optimisation
- for logfire metrics

"""

from dataclasses import dataclass,field
from datetime import datetime
from src.core.logger import get_logger
logger = get_logger(__name__)

# Approximate cost per 1000 tokens (USD)
# Groq free tier = $0 but tracking for when you scale to paid
_COST_PER_1K={
    "llama-3.1-8b-instant": {"prompt":0.00005,"completion":0.00008},
    "qwen/qwen3.6-27b": {"prompt":0.00024,"completion":0.00024},
    "llama-3.3-70b-versatile": {"prompt":0.00059,"completion":0.00079},
    "default":                 {"prompt": 0.00010, "completion": 0.00010}
}

@dataclass
class CallRecord:
    """Record of a single LLM Call."""
    task_type: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str = field(default_factory= lambda: datetime.now().isoformat())
    cached: bool = False

class CostTracker:
    """tracks token usage and estimated cost across all gateway calls.
    Session-scoped - reset when debate session resets"""

    def __init__(self)->None:
        self._records: list[CallRecord] = []
        self._total_tokens = 0
        self._total_cost = 0.0
        self._cache_hits = 0
        logger.info("CostTracker Initialised")

    def record(self,task_type:str,
               model: str,
               prompt_tokens:int,
               completion_tokens:int,
               cached: bool = False)->CallRecord:
        """Record one LLM call and compute its cost."""
        total_tokens = prompt_tokens+completion_tokens
        #  look for cost rates for this model 
        rates = _COST_PER_1K.get(model,_COST_PER_1K["default"])
        cost = (
            (prompt_tokens/1000) * rates["prompt"]
            + (completion_tokens/1000) * rates["completion"]
        )

        record = CallRecord(
            task_type=task_type,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=round(cost,8),
            cached=cached,

        )

        self._records.append(record)
        self._total_tokens +=total_tokens
        self._total_cost += cost

        if cached:
            self._cache_hits +=1

        logger.info(f"Cost tracker | task = {task_type} | model = {model} |" 
                    f"tokens = {total_tokens} | cost=${cost:.6f} | Cached ={cached}")
        return record

    def get_summary(self) -> dict:
        """Return session cost summary."""
        by_task: dict[str, int] = {}
        for r in self._records:
            by_task[r.task_type] = by_task.get(r.task_type, 0) + r.total_tokens

        return {
            "total_calls":   len(self._records),
            "total_tokens":  self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "cache_hits":    self._cache_hits,
            "tokens_by_task": by_task,
        }

    def reset(self) -> None:
        """Clear all records — call at new debate session."""
        self._records.clear()
        self._total_tokens = 0
        self._total_cost   = 0.0
        self._cache_hits   = 0
        logger.info("CostTracker reset.")