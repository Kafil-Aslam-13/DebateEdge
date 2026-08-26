"""Output guardrails
Runs after graph generates AI argument.
4 checksmbefore response reaches user

THREE layers:
1-> rule based:
legal disclaimer injection(always for sensitive topics)
Sensitive toopic detection

2->Guardrails hub
- toxicity filter on AI response

3-> Layer3 LLM-as-a judge
-response relevance. is ai response actually about debate
Even with a well-prompted AI, the model can occasionally:
- Produce aggressive/toxic language
- Go off-topic
- Hallucinate statistics
- Fail to maintain debate framing
Output guardrails catch these before the user sees them"""

from pydantic import BaseModel

from dataclasses import dataclass
from src.core.constants import(
    GUARD_BLOCK,GUARD_DISCLAIMER,GUARD_PASS,
    GUARD_RELEVANCE,GUARD_SENSITIVE,GUARD_TOXICITY,
    GUARD_WARN,SENSITIVE_TOPICS,TASK_CLASSIFICATION
)
from src.core.logger import get_logger
from src.gateway.llm_gateway import get_gateway
from src.guardrails.direct_validators import DirectToxicGuard


logger=get_logger(__name__)

class ResponseRelevanceOutput(BaseModel):
    relevant: bool
    reason: str

    

_DISCLAIMER = (
    "\n\n⚠️ Note: This is an AI-generated debate argument for practice purposes. "
    "It does not constitute professional advice."
)

@dataclass
class OutputGuardResult:
    """Resukt from a single output guardrail check"""
    guard_name:   str
    action:       str
    reason:       str
    passed:       bool
    rewritten:    str = ""  # if action=rewrite, contains cleaned response

    @classmethod
    def passed_result(cls, name: str) -> "OutputGuardResult":
        return cls(guard_name=name, action=GUARD_PASS,
                   reason="Check passed.", passed=True)
    @classmethod
    def warned_result(cls, name: str, reason: str) -> "OutputGuardResult":
        return cls(guard_name=name, action=GUARD_WARN,
                   reason=reason, passed=True)
    @classmethod
    def blocked_result(cls, name: str, reason: str) -> "OutputGuardResult":
        return cls(guard_name=name, action=GUARD_BLOCK,
                   reason=reason, passed=False)

class OutputGuard:
    """runs all output guardrails on Ai resopnse before showing to user."""

    def __init__(self)->None:
        self._gateway=get_gateway()
        self._toxic_guard=self._build_toxic_guard()
        logger.info("output guard initialised")

    def _build_toxic_guard(self):
        """Toxic language validator for AI output — lightweight profanity filter."""
        try:
            return DirectToxicGuard(threshold=0.5)
        except Exception as e:
            logger.warning(f"Output toxic guard build failed: {e}")
            return None

    def _check_sensitive_topic(self,response:str,topic:str)->OutputGuardResult:
        """Detect sensitive topics and flag for disclaimer injection.

        Rule-based: regex match against known sensitive topic list.
        No LLM needed — keyword detection is sufficient here.
        """
        combined = (response + " " + topic).lower()

        for sensitive in SENSITIVE_TOPICS:
            if sensitive in combined:
                return OutputGuardResult.warned_result(
                    GUARD_SENSITIVE,
                    f"Sensitive topic detected: {sensitive}"
                    "Disclaimer will be appended"
                )

        return OutputGuardResult.passed_result(GUARD_SENSITIVE)


    def _inject_disclaimer(
        self,
        response: str,
        needed: bool,
    ) -> tuple[str, OutputGuardResult]:
        """Append disclaimer to response if sensitive topic detected."""
        if needed:
            return (
                response + _DISCLAIMER,
                OutputGuardResult.passed_result(GUARD_DISCLAIMER),
            )
        return response, OutputGuardResult.passed_result(GUARD_DISCLAIMER)

    #  guardrails hub

    def _check_toxicity(self, response: str) -> OutputGuardResult:
        """Detect toxic language in AI response using guardrails hub."""

        if self._toxic_guard is not None:
            try:
                self._toxic_guard.validate(response)
                return OutputGuardResult.passed_result(GUARD_TOXICITY)
            except Exception:
                return OutputGuardResult.blocked_result(
                    GUARD_TOXICITY,
                    "AI response contains inappropriate language. "
                    "Response blocked — please try again.",
                )

        logger.warning("Output toxic guard unavailable — skipping toxicity check")
        return OutputGuardResult.passed_result(GUARD_TOXICITY)

    # llm as a judge

    def _check_response_relevance(
        self,
        response: str,
        topic: str,
        user_argument: str,
    ) -> OutputGuardResult:
        """Check AI response is actually about the debate topic.

        WHY LLM NOT GUARDRAILS HUB:
        Guardrails Hub cannot evaluate whether an AI debate response
        is contextually relevant to a specific user argument and topic.
        Only an LLM judge can understand this contextual relevance.

        Caching: gateway caches this call, so if the same response
        appears twice (rare) it costs zero tokens.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You evaluate AI debate responses for quality. "
                        "Respond with JSON only: "
                        '{"relevant": true/false, "reason": "one sentence"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Debate topic: {topic}\n"
                        f"User's argument: {user_argument}\n"
                        f"AI's response: {response}\n\n"
                        f"Is the AI response relevant to the topic "
                        f"and does it address the user's argument? "
                        f"Respond JSON only."
                    ),
                },
            ]

            result = self._gateway.complete(
                messages=messages, task_type=TASK_CLASSIFICATION,response_model=ResponseRelevanceOutput
            )

            assert isinstance(result, ResponseRelevanceOutput)
            relevant = result.relevant
            reason   = result.reason

            if not relevant:
                return OutputGuardResult.warned_result(
                    GUARD_RELEVANCE,
                    f"AI response may be off-topic. {reason}",
                )

            return OutputGuardResult.passed_result(GUARD_RELEVANCE)

        except Exception as e:
            logger.warning(f"Response relevance check failed: {e}")
            return OutputGuardResult.passed_result(GUARD_RELEVANCE)

    # ── Main run method ────────────────────────────────────────────────────

    def run(
        self,
        response: str,
        topic: str,
        user_argument: str,
    ) -> tuple[str, list[OutputGuardResult]]:
        """Run all output guardrails. Returns (final_response, results).

        Unlike input guardrails, output guardrails don't fully block —
        they either warn or modify the response (e.g. append disclaimer).
        Only a BLOCK from toxicity actually stops the response.

        Returns:
            (final_response, all_results)
            final_response may be modified (disclaimer appended)
        """
        all_results: list[OutputGuardResult] = []
        final_response = response


        sensitive_result = self._check_sensitive_topic(response, topic)
        all_results.append(sensitive_result)

        needs_disclaimer = (sensitive_result.action == GUARD_WARN)
        final_response, disclaimer_result = self._inject_disclaimer(
            final_response, needs_disclaimer
        )
        all_results.append(disclaimer_result)

        toxicity_result = self._check_toxicity(response)
        all_results.append(toxicity_result)

        if toxicity_result.action == GUARD_BLOCK:
            logger.warning("Output blocked by toxicity guardrail")
            fallback = (
                "I need to rephrase my response. "
                "Let me argue this point more constructively: "
                "the evidence suggests the opposite position has "
                "significant merit that hasn't been addressed."
            )
            return fallback, all_results
        
        relevance_result = self._check_response_relevance(
            response, topic, user_argument
        )
        all_results.append(relevance_result)

        for r in all_results:
            logger.info(
                f"OutputGuard | {r.guard_name} → {r.action}"
            )

        return final_response, all_results
    
        