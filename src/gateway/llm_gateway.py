"""LLM Gateway
Single entry point for all LLM calls in the project.
WHat gateway does:
1-> Routing of llms based on model alias  i.e fast / powerful
2-> litellm router handles automatically 
fast fails -> tries powerfvul and vbice versa . All fail -> raise gateway error
3 -> Caching -> identical prompts return cached response ,
Zero api calls , zero cost , instant response. cache TTL: 1 hour
4-> retries :  3 retries with backoff on transient faliures handles groq 429 rate limits autoomatically
5-> Cost tracking -> Every Call Recorded in CostTracker tokens , cost , task type , model used

Why litellm.Router:
- single interface Across any provider
Build-in fallback chains
BBuild in retry logic
if Switching providers = change modals.yaml file not code

MODERN PATTERN:
Route is Build once (singleton) from models.yaml file . 
Every Graph Node calls Gateway.complete(messages,task_type).
Node no longer immport ChatGroq Directly.
"""

from functools import lru_cache
from typing import Literal
from typing import Literal, TypeVar, Type
from pydantic import BaseModel
import hashlib , json

from litellm.router import Router

from src.core.config import get_settings
from src.core.exceptions import GatewayError
from src.core.logger import get_logger
from  src.gateway.cost_tracker import CostTracker
from src.observability.logfire_setup import get_logfire_span
from src.observability.metrics import record_llm_cost
logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)
TaskType = Literal[
    "classification",
    "scoring",
    "debate",
    "fallacy",
    "summary",
    "explanation"

]

class LLMGateway:
    """Single entry point for all LLM calls."""
    def __init__(self):
        settings=get_settings()
        self._cfg = settings.model_config
        self._task_routing=settings.task_routing
        self._gateway_cfg=settings.gateway_config

        # Build llm router from models.yaml list

        self._router = Router(
            model_list=self._cfg,
            fallbacks=settings.fallback_models,
            num_retries=self._gateway_cfg.get("num_retries",3),
            retry_after=self._gateway_cfg.get("retry_after",2),
            set_verbose=False,
        )

        # IN memory cache : hash(msg + model ) -> response context
        self._cache: dict[str,str] = {}
        self._cache_enabled =  self._gateway_cfg.get("cache_enabled",True)
        self._cache_ttl = self._gateway_cfg.get("cache_ttl",3600)

        self.cost_tracker = CostTracker()

        logger.info(F"LLM GATEWAY Initialised"
                    f"cache={self._cache_enabled} | "
            f"retries={self._gateway_cfg.get('num_retries', 3)}"
            )

    def _get_model_alias(self,task_type:TaskType)-> str:
        """Map task type to model alias defined in models.yaml file"""
        return self._task_routing.get(task_type,"fast")

    def _cache_key(
    self,
    messages: list[dict],
    model_alias: str,
    response_model: Type[BaseModel] | None = None,
) -> str:
        

        payload = {
            "messages": messages,
            "model": model_alias,
            "response_model": (
                response_model.model_json_schema()
                if response_model is not None
                else None
            ),
        }

        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def complete(
    self,
    messages: list[dict],
    task_type: TaskType = "debate",
    response_model: Type[T] | None = None,
) -> str | T:
        

        model_alias = self._get_model_alias(task_type)

        cache_key = self._cache_key(
            messages=messages,
            model_alias=model_alias,
            response_model=response_model,)

        # ── CACHE CHECK ───────────────────────────────────────────────────────
        if self._cache_enabled and cache_key in self._cache:
            logger.info(
                f"Gateway cache HIT | "
                f"task={task_type} | "
                f"model={model_alias} | "
                f"structured={response_model is not None}")

            cached_content = self._cache[cache_key]

            self.cost_tracker.record(
                task_type=task_type,
                model=model_alias,
                prompt_tokens=0,
                completion_tokens=0,
                cached=True,)

            #  log cache hit metric
            record_llm_cost(
                cost_usd=0.0,
                tokens=0,
                task_type=task_type,
                model=model_alias,
                cached=True,)

            if response_model is not None:
                return response_model.model_validate_json(cached_content)

            return cached_content

    #  wrap entire LLM call in logfire span
        with get_logfire_span(
            "llm_gateway_call",
            task_type=task_type,
            model_alias=model_alias,
            structured=response_model is not None,):

            try:

                logger.info(
                    f"Gateway call | "
                    f"task={task_type} | "
                    f"model={model_alias} | "
                    f"structured={response_model is not None}")

                completion_kwargs = {
                    "model":    model_alias,
                    "messages": messages,}
                

            # ── STRUCTURED OUTPUT ─────────────────────────────────────────
                if response_model is not None:
                    completion_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name":   response_model.__name__,
                            "schema": response_model.model_json_schema(),},}

                response = self._router.completion(**completion_kwargs)
                content  = response.choices[0].message.content

            # ── COST TRACKING ─────────────────────────────────────────────
                usage  = response.usage
                record = self.cost_tracker.record(
                task_type=task_type,
                model=response.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached=False,
                )

            # Sprint 8: log cost to Logfire metrics
                record_llm_cost(
                cost_usd=record.cost_usd,
                tokens=record.total_tokens,
                task_type=task_type,
                model=response.model,
                cached=False,
                )

            # ── CACHE RAW CONTENT ─────────────────────────────────────────
                if self._cache_enabled:
                    self._cache[cache_key] = content

            # ── PYDANTIC VALIDATION ───────────────────────────────────────
                if response_model is not None:
                    return response_model.model_validate_json(content)

                return content

            except Exception as e:
                logger.error(
                f"Gateway failed | "
                f"task={task_type} | "
                f"error={e}"
                )
                raise GatewayError(
                f"All models failed for task '{task_type}': {e}"
                ) from e

    def get_cost_summary(self) -> dict:
        """Return session cost and token summary."""
        return self.cost_tracker.get_summary()

    def reset(self) -> None:
        """Clear cache and cost tracker for new session."""
        self._cache.clear()
        self.cost_tracker.reset()
        logger.info("Gateway reset — cache and cost tracker cleared.")

@lru_cache(maxsize=1)
def get_gateway() -> LLMGateway:
    """Return singleton LLMGateway instance.

    Built once, reused for every graph node call.
    Cache cleared on new debate session via reset_gateway().
    """
    return LLMGateway()


def reset_gateway() -> None:
    """Reset gateway cache and cost tracker.

    Call at start of each new debate session alongside reset_memory().
    """
    get_gateway().reset()




