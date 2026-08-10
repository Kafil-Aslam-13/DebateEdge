"""
LangGraph debate workflow — Sprint 2.

GRAPH STRUCTURE:
                START
                  |
        classify_argument
                  |
        score_argument
                  |
        route_by_quality (conditional)
         /        |        \
    strong       weak    fallacy
         /        |        /
      generate_counterargument
                  |
                 END
"""

import json

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src.core.config import get_settings
from src.core.constants import (
    QUALITY_STRONG, QUALITY_WEAK, QUALITY_FALLACY,
    NODE_CLASSIFY, NODE_COUNTER, NODE_FALLACY,
    NODE_SCORE, NODE_STRONG, NODE_WEAK, NODE_FALLACY_DETECT
)
from src.core.exceptions import GraphError
from src.core.logger import get_logger
from src.graphs.state import DebateState
from src.parsers.scoring_parsers import (
    get_structured_classifier,
    get_structured_scorer,
    handler_response_parser
)
from src.prompts.scoring_prompts import (
    strong_handler_prompt,
    weak_handler_prompt,
    fallacy_handler_prompt,
    classification_prompt,
    scoring_prompt
)
from src.agents.fallacy_agent import FallacyDetectionService
logger = get_logger(__name__)


# ── LLM instances ─────────────────────────────────────────────────────────────

def _get_fast_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.default_model,
        temperature=0.2,
        max_tokens=256,
        api_key=settings.groq_api_key,
    )


def _get_powerful_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.complex_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        api_key=settings.groq_api_key,
    )


# ── Node functions ─────────────────────────────────────────────────────────────

def classify_argument_node(state: DebateState) -> dict:
    """Classify argument quality: strong, weak, or fallacy."""
    logger.info(f"[Node: classify] Classifying argument (turn {state['turn_number']})")

    if state.get("has_error"):
        return {}

    try:
        structured_llm = get_structured_classifier()

        result = structured_llm.invoke(
            classification_prompt.format(
                argument=state['user_argument'],
                topic=state['topic']
            )
        )

        quality   = result.quality   if hasattr(result, 'quality')   else result.get('quality', 'weak')
        reasoning = result.reasoning if hasattr(result, 'reasoning') else result.get('reasoning', '')

        if quality not in [QUALITY_STRONG, QUALITY_WEAK, QUALITY_FALLACY]:
            logger.warning(f"Unknown quality '{quality}', defaulting to weak")
            quality = QUALITY_WEAK

        logger.info(f"[Node: classify] Quality={quality} | Reasoning={reasoning[:60]}")

        return {
            "argument_quality":  quality,
            "quality_reasoning": reasoning,
        }

    except Exception as e:
        logger.error(f"[Node: classify] Failed: {e}")
        return {
            "argument_quality":  QUALITY_WEAK,
            "quality_reasoning": "Classification failed, defaulting to weak",
            "error":             str(e),
            "has_error":         False,
        }


def score_argument_node(state: DebateState) -> dict:
    """Score argument on logic, evidence, clarity (0-10 each)."""
    logger.info("[Node: score] Scoring argument")

    if state.get("has_error"):
        return {}

    try:
        structured_scorer = get_structured_scorer()
        result = structured_scorer.invoke(
    scoring_prompt.format(
        topic=state["topic"],
        user_side=state["user_side"],
        argument=state["user_argument"],
        quality=state.get("argument_quality", "weak"),
    )
)

        logic    = result.logic    if hasattr(result, 'logic')    else result.get('logic', 5)
        evidence = result.evidence if hasattr(result, 'evidence') else result.get('evidence', 5)
        clarity  = result.clarity  if hasattr(result, 'clarity')  else result.get('clarity', 5)
        overall = (
    logic * 0.4
    + evidence * 0.4
    + clarity * 0.2
)
        

        logger.info(
            f"[Node: score] Overall={overall}/10 | "
            f"Logic={logic} | Evidence={evidence} | Clarity={clarity}"
        )

        return {
    "argument_score": round(overall, 1),
    "score_breakdown": {
        "logic": logic,
        "evidence": evidence,
        "clarity": clarity,
    },
}

    except Exception as e:
        logger.error(f"[Node: score] Failed: {e}")
        return {
            "argument_score":  5,
            "score_breakdown": {"logic": 5, "evidence": 5, "clarity": 5},
        }


def handle_strong_node(state: DebateState) -> dict:
    """Handle strong argument: brief acknowledgment + hard challenge."""
    logger.info("[Node: handle_strong] Handling strong argument")

    return {
        "handler_note": (               # Bug 2 fix: was handler_node
            f"Strong argument (score={state.get('argument_score', '?')}/10). "
            "Acknowledging briefly then attacking hardest point."
        )
    }


def handle_weak_node(state: DebateState) -> dict:
    """Handle weak argument: expose weakness + guide improvement."""
    logger.info("[Node: handle_weak] Handling weak argument")

    return {
        "handler_note": (
            f"Weak argument (score={state.get('argument_score', '?')}/10). "
            "Exposing specific weakness and guiding toward stronger reasoning."
        )
    }


def handle_fallacy_node(state: DebateState) -> dict:
    """Handle fallacy: name it, explain it, show how to fix it."""
    logger.info("[Node: handle_fallacy] Handling fallacy argument")

    return {
        "handler_note": (
            f"Logical fallacy detected. "
            f"Reasoning: {state.get('quality_reasoning', '')}. "
            "Naming, explaining, and correcting."
        )
    }

def detect_fallacy_details_node(state: DebateState) -> dict:
    """Deep fallacy analysis — only runs when classifier flagged fallacy.

    WHY SEPARATE FROM classify_argument_node:
    classify_argument_node: quick classification (strong/weak/fallacy)
    This node: deep analysis (which fallacy, severity, correction)

    Separating them means:
    - Quick classify runs every turn (cheap, fast)
    - Deep detection only when needed (expensive, skipped otherwise)
    """
    logger.info("[Node: fallacy_detect] Running detailed fallacy detection")

    if state.get("argument_quality") != QUALITY_FALLACY:
        logger.info("[Node: fallacy_detect] No fallacy flagged — skipping.")
        return {
            "contains_fallacy":    False,
            "fallacy_name":        "none",
            "fallacy_type":        "none",
            "fallacy_severity":    "none",
            "fallacy_explanation": "",
            "fallacy_correction":  "none",
        }

    try:
        service = FallacyDetectionService()

        result = service.detect(
            argument=state["user_argument"],
            topic=state["topic"],
            user_side=state["user_side"],
        )

        explanation = ""
        if result.contains_fallacy and result.fallacy_name != "none":
            explanation = service.explain(
                fallacy_name=result.fallacy_name,
                argument=state["user_argument"],
            )

        logger.info(
            f"[Node: fallacy_detect] "
            f"name={result.fallacy_name} | "
            f"severity={result.severity}"
        )

        return {
            "contains_fallacy":    result.contains_fallacy,
            "fallacy_name":        result.fallacy_name,
            "fallacy_type":        result.fallacy_type.value,
            "fallacy_severity":    result.severity.value,
            "fallacy_explanation": explanation,
            "fallacy_correction":  result.correction,
        }

    except Exception as e:
        logger.error(f"[Node: fallacy_detect] Failed: {e}")
        return {
            "contains_fallacy":    False,
            "fallacy_name":        "none",
            "fallacy_type":        "none",
            "fallacy_severity":    "none",
            "fallacy_explanation": "",
            "fallacy_correction":  "none",
            "error":               str(e),
        }


def generate_counterargument_node(state: DebateState) -> dict:
    """Generate AI counterargument adapted to argument quality."""
    logger.info("[Node: counter] Generating counterargument")

    if state.get("has_error"):
        return {"ai_response": "I encountered an issue. Please try again."}

    try:
        powerful_llm = _get_powerful_llm()
        ai_side = "against" if state["user_side"] == "for" else "for"
        quality = state.get("argument_quality", QUALITY_WEAK)

        prompt_map = {
            QUALITY_STRONG:  strong_handler_prompt,
            QUALITY_WEAK:    weak_handler_prompt,
            QUALITY_FALLACY: fallacy_handler_prompt,
        }
        selected_prompt = prompt_map.get(quality, weak_handler_prompt)

        chain = selected_prompt | powerful_llm | handler_response_parser

        response = chain.invoke({
            "topic":             state["topic"],
            "argument":          state["user_argument"],
            "quality":           quality,
            "score":             state.get("argument_score", 5),
            "quality_reasoning": state.get("quality_reasoning", ""),
            "ai_side":           ai_side,
        })

        logger.info("[Node: counter] Counterargument generated successfully.")
        return {"ai_response": response}

    except Exception as e:
        logger.error(f"[Node: counter] Failed: {e}")
        return {
            "ai_response": "I encountered an issue generating a response. Please try again.",
            "error":       str(e),
            "has_error":   True,
        }


# ── Router ─────────────────────────────────────────────────────────────────────

def route_by_quality(state: DebateState) -> str:
    """Conditional edge: route to handler based on argument quality."""
    quality = state.get("argument_quality", QUALITY_WEAK)

    route_map = {
        QUALITY_STRONG:  NODE_STRONG,
        QUALITY_WEAK:    NODE_WEAK,
        QUALITY_FALLACY: NODE_FALLACY,
    }

    next_node = route_map.get(quality, NODE_WEAK)
    logger.info(f"[Router] Quality={quality} -> Node={next_node}")
    return next_node


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_debate_graph() -> StateGraph:
    """Build and compile the debate LangGraph."""
    graph = StateGraph(DebateState)

    graph.add_node(NODE_CLASSIFY, classify_argument_node)
    graph.add_node(NODE_SCORE,    score_argument_node)
    graph.add_node(NODE_FALLACY_DETECT,detect_fallacy_details_node)
    graph.add_node(NODE_STRONG,   handle_strong_node)
    graph.add_node(NODE_WEAK,     handle_weak_node)
    graph.add_node(NODE_FALLACY,  handle_fallacy_node)
    graph.add_node(NODE_COUNTER,  generate_counterargument_node)

    graph.set_entry_point(NODE_CLASSIFY)

    graph.add_edge(NODE_CLASSIFY, NODE_SCORE)

    graph.add_conditional_edges(
        NODE_SCORE,
        route_by_quality,
        {
            NODE_STRONG:  NODE_STRONG,
            NODE_WEAK:    NODE_WEAK,
            NODE_FALLACY: NODE_FALLACY_DETECT,
        }
    )
    graph.add_edge(NODE_FALLACY_DETECT,NODE_FALLACY)

    graph.add_edge(NODE_STRONG,  NODE_COUNTER)
    graph.add_edge(NODE_WEAK,    NODE_COUNTER)
    graph.add_edge(NODE_FALLACY, NODE_COUNTER)
    graph.add_edge(NODE_COUNTER, END)

    compiled = graph.compile()
    logger.info("Debate graph compiled successfully.")
    return compiled


# ── Singleton ──────────────────────────────────────────────────────────────────

_debate_graph = None


def get_debate_graph():
    """Return compiled graph (built once, cached)."""
    global _debate_graph
    if _debate_graph is None:
        _debate_graph = build_debate_graph()
    return _debate_graph