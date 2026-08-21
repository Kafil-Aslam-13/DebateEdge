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
from langgraph.graph import StateGraph, END
import re
from src.core.config import get_settings
from src.gateway.llm_gateway import get_gateway, reset_gateway
from src.core.constants import (
    QUALITY_STRONG, QUALITY_WEAK, QUALITY_FALLACY,
    NODE_CLASSIFY, NODE_COUNTER, NODE_FALLACY,
    NODE_SCORE, NODE_STRONG, NODE_WEAK, NODE_FALLACY_DETECT,
    NODE_MEMORY_UPDATE , NODE_RAG_RETRIEVE ,
    TASK_CLASSIFICATION,TASK_DEBATE,TASK_SCORING,
    NODE_INPUT_GUARD,NODE_OUTPUT_GUARD,GUARD_BLOCK,GUARD_PASS,
    NODE_EVALUATE
)

from src.core.exceptions import GraphError
from src.core.logger import get_logger
from src.graphs.state import DebateState
from src.parsers.scoring_parsers import (
    ClassificationOutput,
    ScoreOutput
)
from src.prompts.scoring_prompts import (
    strong_handler_prompt,
    weak_handler_prompt,
    fallacy_handler_prompt,
    classification_prompt,
    scoring_prompt,
)

from src.memory.buffer_memory import DebateBufferMemory
from src.memory.summary_memory import DebateSummaryMemory
from src.memory.vector_memory import DebateVectorMemory
from src.agents.fallacy_agent import FallacyDetectionService


from src.retrieval.chroma_store import DebateChromaStore
from src.retrieval.pinecone_db import DebatePineconeDB
from langchain_core.documents import Document



from src.guardrails.input_guard import InputGuard
from src.guardrails.output_guard import OutputGuard


from src.observability.logfire_setup import get_logfire_span
from src.observability.metrics import (
    record_argument_score,
    record_fallacy_detected,
    record_rag_retrieval,
    record_session_turn,
    record_guard_result,
)

from src.evaluation.evaluator import DebateEvaluator


from src.gateway.cost_optimizer import CostOptimizer



logger = get_logger(__name__)
#  memory instance
_buffer_memory  = DebateBufferMemory(window_size=6)
_summary_memory = DebateSummaryMemory()
_vector_memory  = DebateVectorMemory(top_k=3)

# module-level retrieval instance
_chroma_store  = DebateChromaStore()
_pinecone_db   = DebatePineconeDB()

# module level guardrail instances
_input_guard  = InputGuard()
_output_guard = OutputGuard()

evaluator=DebateEvaluator()


_cost_optimizer=CostOptimizer()




def clean_ai_response(content: str) -> str:
    """Remove leaked reasoning/meta sections before returning AI response."""

    if not content:
        return ""

    text = content.strip()

    # Remove complete <think>...</think> blocks
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove an unmatched opening <think>
    text = re.sub(
        r"<think>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove an unmatched closing tag
    text = re.sub(
        r"</think>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove common leaked reasoning prefixes
    text = re.sub(
        r"^\s*(Thinking Process|Chain of Thought|Reasoning|Analysis)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()

def _prompt_to_messages(prompt_value) -> list[dict]:
    """Convert LangChain prompt messages to OpenAI-compatible messages."""

    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
    }

    messages = []

    for message in prompt_value.messages:
        role = role_map.get(message.type)

        if role is None:
            raise ValueError(
                f"Unsupported LangChain message type: {message.type!r}"
            )

        messages.append({
            "role": role,
            "content": message.content,
        })

    return messages

def input_blocked_node(state: DebateState) -> dict:
    """Handle an argument rejected by input guardrails."""
    reason = state.get(
        "input_guard_reason",
        "Your argument was rejected by the input guard."
    )

    logger.warning(
        f"[Node: input_blocked] Request rejected | reason={reason}"
    )

    return {
        "ai_response": reason,
        "has_error": True,
    }

def _format_docs(docs:list[Document])->str:
    """Format retrieved documents into a single context string"""
    if not docs:
        return "no relevance evidence retrieved"
    return "\n\n".join(
        f"[source {i+1}]: {doc.page_content}"
        for i ,  doc in enumerate(docs)
    )

def reset_memory() -> None:
    """Clear all memory — call at start of each new debate session."""
    _buffer_memory.clear()
    _summary_memory.clear()
    _vector_memory.clear()
    reset_gateway()
    evaluator.reset() 
    _cost_optimizer.reset()
    logger.info("All memory cleared for new debate session.")

def get_buffer_memory() -> DebateBufferMemory:
    """Expose buffer memory so debate_service can read history from it."""
    return _buffer_memory



# ── Node functions ─────────────────────────────────────────────────────────────

def classify_argument_node(state: DebateState) -> dict:
    """Classify argument quality: strong, weak, or fallacy."""
    logger.info(f"[Node: classify] Classifying argument (turn {state['turn_number']})")

    if state.get("has_error"):
        return {}

    with get_logfire_span(
        "node.classify_argument",
        turn=state["turn_number"],
        topic=state["topic"][:50],
    ):
        try:
            gateway = get_gateway()

            prompt_value= classification_prompt.invoke({
            "argument":state['user_argument'],
            "topic": state["topic"],
            })
            messages = _prompt_to_messages(prompt_value)


            result = gateway.complete(messages=messages,task_type=TASK_CLASSIFICATION,response_model=ClassificationOutput)
            assert isinstance(result,ClassificationOutput)
        
            logger.info(f"[Node: classify] Quality={result.quality} | Reasoning={result.reasoning[:60]}")

            return {
                "argument_quality":  result.quality,
                "quality_reasoning": result.reasoning,
            }

        except Exception as e:
            logger.error(f"[Node: classify] Failed:{e}")
            logger.exception(e)
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

    with get_logfire_span(
        "node.score_argument",
        turn=state["turn_number"],
        quality=state.get("argument_quality", ""),
    ):
        try:
            gateway = get_gateway()

            prompt_value = scoring_prompt.invoke({
                "topic": state["topic"],
                "user_side": state["user_side"],
                "argument": state["user_argument"],
                "quality": state.get(
                    "argument_quality",
                    QUALITY_WEAK,
                ),
            })

            messages = _prompt_to_messages(prompt_value)
 
            result=gateway.complete(messages,task_type=TASK_SCORING,response_model=ScoreOutput)

            assert isinstance(result, ScoreOutput)

        
            # Application calculates overall.
            overall = round(
                result.logic * 0.4
                + result.evidence * 0.4
                + result.clarity * 0.2,2
            )
            score_breakdown = {
                "logic": result.logic,
                "evidence": result.evidence,
                "clarity": result.clarity,
            }
            record_argument_score(
                score=overall,
                quality=state.get("argument_quality",""),
                turn_number=state["turn_number"],
                topic=state["topic"]
            )

            logger.info(
                f"[Node: score] "
                f"Overall={overall:.2f}/10 | "
                f"Logic={result.logic} | "
                f"Evidence={result.evidence} | "
                f"Clarity={result.clarity}"
            )

            return {
                "argument_score":  overall,
                "score_breakdown": score_breakdown,
            }

        except Exception as e:
            logger.error(f"[Node: score] Failed: {e}")
            logger.exception(e)
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

    with get_logfire_span(
        "node.detect_fallacy",
        turn=state["turn_number"],
    ):

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


            record_fallacy_detected(
                    fallacy_name=result.fallacy_name,
                    severity=result.severity.value,
                    turn_number=state["turn_number"],
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




def input_guardrail_node(state: DebateState) -> dict:
    """Run all input guardrails before processing argument.

    If BLOCKED: set has_error=True so all subsequent nodes skip.
    The generate_counterargument_node checks has_error and returns
    the block reason as the ai_response instead.

    WHY AT START OF GRAPH:
    Guardrails must run before any LLM call is made.
    Injecting them as the first node means the rest of the graph
    is protected automatically — no changes needed in other nodes.
    """
    logger.info(
        f"[Node: input_guard] Running input guardrails | "
        f"turn={state['turn_number']}"
    )

    with get_logfire_span(
        "node.input_guardrail",
        turn=state["turn_number"],
    ):
        

        result, all_results = _input_guard.run(
            argument=state["user_argument"],
            topic=state["topic"],
        )
        for r in all_results:
            record_guard_result(
                guard_name=r.guard_name,
                action=r.action,
                guard_type="input",
            )

        if result.action == GUARD_BLOCK:
            logger.warning(
                f"[Node: input_guard] BLOCKED | reason={result.reason}"
            )
            return {
                "input_guard_passed": False,
                "input_guard_action": GUARD_BLOCK,
                "input_guard_reason": result.reason,
                "has_error":          True,
                "error":              f"Input blocked: {result.reason}",
            }

        logger.info("[Node: input_guard] All input checks passed.")
        return {
            "input_guard_passed": True,
            "input_guard_action": result.action,
            "input_guard_reason": result.reason,
            "has_error":          False,
        }







def generate_counterargument_node(state: DebateState) -> dict:
    """Generate AI counterargument adapted to argument quality."""
    logger.info("[Node: counter] Generating counterargument")

    if state.get("has_error"):
        return {"ai_response": "I encountered an issue. Please try again."}

    with get_logfire_span(
        "node.generate_counterargument",
        turn=state["turn_number"],
        quality=state.get("argument_quality", ""),
    ):
        

        try:
            gateway=get_gateway()
            ai_side = "against" if state["user_side"] == "for" else "for"
            quality = state.get("argument_quality", QUALITY_WEAK)

            raw_history = state.get("debate_history",[])
            history_tuples=[]
            if raw_history:
                from langchain_core.messages import HumanMessage ,AIMessage
                for msg in raw_history:
                    if isinstance(msg,HumanMessage):
                        history_tuples.append(("human",msg.content))
                    elif isinstance(msg,AIMessage):
                        history_tuples.append(("assistant",msg.content))
            compressed_history = _cost_optimizer.compress_history(
                history_tuples
            )
            user_argument = _cost_optimizer.enforce_argument_budget(
                state["user_argument"]

            )


            prompt_map = {
                QUALITY_STRONG:  strong_handler_prompt,
                QUALITY_WEAK:    weak_handler_prompt,
                QUALITY_FALLACY: fallacy_handler_prompt,
            }
            selected_prompt = prompt_map.get(quality, weak_handler_prompt)
            prompt_value = selected_prompt.invoke({
                "quality": quality,
                "topic": state["topic"],
                "argument": state["user_argument"],
                "score": state.get("argument_score", 0),
                "quality_reasoning": state.get(
                    "quality_reasoning",
                    "",
                ),
                "ai_side": ai_side,
                "rag_context": state.get(
                    "rag_context",
                    "No relevant evidence retrieved.",
                ),
            })

            messages = _prompt_to_messages(prompt_value)
            logger.debug(
                "[Node: counter] Messages: %s",
                messages,
            )

            recent_scores = [e.argument_score
                             for e in evaluator.get_turn_history()[-3:]]
            smart_alias= _cost_optimizer.get_debate_model_alias(
                recent_scores=recent_scores
            )
            task = "debate" if smart_alias == "powerful" else "scoring"



            content = gateway.complete(
                messages=messages,
                task_type=task,
            )

            logger.info("[Node: counter] Counterargument generated successfully.")
            cleaned_content=clean_ai_response(content)
            logger.info(
                "[Node: counter] Counterargument generated successfully | chars=%d",
                len(cleaned_content),
            )
            return {"ai_response": cleaned_content}

        except Exception as e:
            logger.error(f"[Node: counter] Failed: {e}")
            return {
                "ai_response": "I encountered an issue generating a response. Please try again.",
                "error":       str(e),
                "has_error":   True,
            }






def output_guardrail_node(state: DebateState) -> dict:
    """Run all output guardrails on AI response before returning.

    Modifies ai_response in place if disclaimer needed.
    Replaces ai_response if toxicity detected.
    Logs all check results to state.
    """
    logger.info("[Node: output_guard] Running output guardrails")
    with get_logfire_span(
        "node.output_guardrail",
        turn=state["turn_number"],
    ):
        

        response = state.get("ai_response", "")

        if not response:
            return {
                "output_guard_results": [],
            }
    

        final_response, results = _output_guard.run(
            response=response,
            topic=state["topic"],
            user_argument=state["user_argument"],
        )

    # Serialize results for state (TypedDict needs serializable types)
        serialized = [
            {
                "guard_name": r.guard_name,
                "action":     r.action,
                "reason":     r.reason,
                "passed":     r.passed,
            }
            for r in results
        ]
        for r in results:
            record_guard_result(
                guard_name=r.guard_name,
                action=r.action,
                guard_type="output",
            )

        logger.info("[Node: output_guard] Output guardrails complete.")

        return {
            "ai_response":          final_response,
            "output_guard_results": serialized,
        }









def rag_retrieval_node(state:DebateState)-> dict:
    """retrieves relevant evidence before generating counterargument.
    TWO-STAGE RETRIEVAL:
    Stage 1: ChromaDB (in-session, HuggingFace, fast)
             → similarity search for most relevant evidence
    Stage 2: Pinecone (persistent, Cohere, richer)
             → MMR search for diverse additional evidence
             → only runs if Pinecone is configured

    COMBINED CONTEXT:
    Results from both are merged into one rag_context string.
    The counterargument node injects this into the debate prompt.
    """
    logger.info(f"[NODE: rag_retrieval] Retrieving relevant evidance")
    query=state.get("user_argument","")
    topic=state.get("topic","")

    if not query.strip():
        return {"rag_context":"No relevant evidence retrieved."}

    with get_logfire_span(
        "node.rag_retrieval",
        turn=state["turn_number"],
        topic=topic[:50],
    ):
        try:
            chroma_docs=_chroma_store.retrieve_similar(query=f"{topic}: {query}",k=2)
            pinecone_docs=_pinecone_db.retrieve_mmr(
                query=f"{topic}:{query}",
                k=2,
                lambda_mult=0.5
            )

            all_docs = chroma_docs + pinecone_docs
            rag_context = _format_docs(all_docs)

            record_rag_retrieval(
                chroma_count=len(chroma_docs),
                pinecone_count=len(pinecone_docs),
            )
            logger.info(
                f"[Node: rag_retrieve] Retrieved "
                f"{len(chroma_docs)} chroma + "
                f"{len(pinecone_docs)} pinecone docs"
            )
            return {"rag_context": rag_context}
        except Exception as e:
            logger.warning(f"[Node: rag_retrieve] Failed: {e}")
            return {"rag_context": "Evidence retrieval unavailable."}
    






def update_memory_node(state: DebateState) -> dict:
    """Update all three memory types after each debate turn.

    ORDER MATTERS:
    1. Check buffer — is it near limit?
    2. If yes: pop oldest messages → pass to summary → summary updates
    3. Add this turn to buffer
    4. Store argument in vector memory with fallacy metadata
    5. Search vector memory for similar past arguments

    WHY fallacy_name IN VECTOR STORE:
    Your DebateVectorMemory supports find_previous_fallacies().
    This only works if we store the fallacy_name in metadata.
    Sprint 5 coaching uses this to say:
    'You committed ad_hominem in turn 2 and again in turn 5.'
    """
    logger.info(
        f"[Node: memory_update] Updating memory | turn={state['turn_number']}"
    )

    with get_logfire_span(
        "node.update_memory",
        turn=state["turn_number"],
    ):
        

        user_argument = state.get("user_argument", "")
        ai_response   = state.get("ai_response", "")
        quality       = state.get("argument_quality", "weak")
        score         = state.get("argument_score", 0)
        turn_number   = state.get("turn_number", 1)
        fallacy_name  = state.get("fallacy_name", "none")

        if _buffer_memory.is_near_limit():
            logger.info(
                "[Node: memory_update] Buffer near limit — "
                "popping oldest messages for summary"
            )
            # Pop 2 oldest messages (1 turn = human + AI)
            oldest_messages = _buffer_memory.pop_oldest(2)

            if oldest_messages:
                _summary_memory.update(oldest_messages)
                logger.info("[Node: memory_update] Summary updated from popped messages")

        # ── 2. Buffer: add this turn 
        _buffer_memory.add_turn(user_argument, ai_response)

    
        _vector_memory.store_argument(
            argument=user_argument,
            turn_number=turn_number,
            quality=quality,
            score=float(score),
            fallacy_name=fallacy_name,  
        )
        similar_past_args = []
        if turn_number > 1:
            similar_past_args = _vector_memory.find_similar(user_argument)

            if similar_past_args:
                logger.info(
                    f"[Node: memory_update] "
                    f"Found {len(similar_past_args)} similar past arguments"
                )

        debate_summary = _summary_memory.get_summary()

        record_session_turn(
            turn_number=turn_number,
            topic=state["topic"],
        )

        logger.info("[Node: memory_update] All memory updated.")

        return {
            "debate_summary":    debate_summary,
            "similar_past_args": similar_past_args,
            "memory_updated":    True,
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





def evaluate_turn_node(state: DebateState) -> dict:
    """Evaluate turn quality after counterargument is generated.

    Full session evaluation runs on debate end (called from service).
    """
    logger.info(
        f"[Node: evaluate] Evaluating turn {state['turn_number']}"
    )

    with get_logfire_span(
        "node.evaluate_turn",
        turn=state["turn_number"],
    ):
        try:
            turn_eval = evaluator.evaluate_turn(
                turn_number=state["turn_number"],
                user_argument=state.get("user_argument", ""),
                ai_response=state.get("ai_response", ""),
                argument_score=state.get("argument_score", 0),
                argument_quality=state.get("argument_quality", "weak"),
                topic=state.get("topic", ""),
            )

            logger.info(
                f"[Node: evaluate] "
                f"AI score={turn_eval.ai_response_score} | "
                f"grade={turn_eval.grade}"
            )

            return {
                "turn_eval_score":    turn_eval.ai_response_score,
                "turn_eval_grade":    turn_eval.grade,
                "turn_eval_feedback": turn_eval.ai_feedback,
            }

        except Exception as e:
            logger.warning(f"[Node: evaluate] Failed: {e}")
            return {
                "turn_eval_score":    5,
                "turn_eval_grade":    "average",
                "turn_eval_feedback": "Evaluation unavailable.",
            }

# ── Graph builder ──────────────────────────────────────────────────────────────

def build_debate_graph() -> StateGraph:
    """Build and compile the debate LangGraph."""
    graph = StateGraph(DebateState)
    graph.add_node(NODE_INPUT_GUARD,    input_guardrail_node)
    graph.add_node(NODE_CLASSIFY, classify_argument_node)
    graph.add_node(NODE_SCORE,    score_argument_node)
    graph.add_node(NODE_FALLACY_DETECT,detect_fallacy_details_node)
    graph.add_node(NODE_STRONG,   handle_strong_node)
    graph.add_node(NODE_WEAK,     handle_weak_node)
    graph.add_node(NODE_FALLACY,  handle_fallacy_node)
    graph.add_node(NODE_RAG_RETRIEVE,rag_retrieval_node)
    graph.add_node(NODE_COUNTER,  generate_counterargument_node)
    graph.add_node(NODE_OUTPUT_GUARD,output_guardrail_node)
    graph.add_node("input_blocked", input_blocked_node)
    graph.add_node(NODE_EVALUATE,evaluate_turn_node)
    graph.add_node(NODE_MEMORY_UPDATE,update_memory_node)

    graph.set_entry_point(NODE_INPUT_GUARD)
    graph.add_conditional_edges(

        NODE_INPUT_GUARD,
        lambda state: (
            NODE_CLASSIFY
            if state.get("input_guard_passed")
            else "input_blocked"
        ),
        {
            NODE_CLASSIFY: NODE_CLASSIFY,
           "input_blocked": "input_blocked",
        }
    )

    graph.add_edge("input_blocked", END)

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

    graph.add_edge(NODE_STRONG,  NODE_RAG_RETRIEVE)
    graph.add_edge(NODE_WEAK,    NODE_RAG_RETRIEVE)
    graph.add_edge(NODE_FALLACY, NODE_RAG_RETRIEVE)
    graph.add_edge(NODE_RAG_RETRIEVE,NODE_COUNTER)



    graph.add_edge(NODE_COUNTER,       NODE_OUTPUT_GUARD)
    graph.add_edge(NODE_OUTPUT_GUARD,NODE_EVALUATE)
    graph.add_edge(NODE_EVALUATE,NODE_MEMORY_UPDATE)
    
    
    graph.add_edge(NODE_MEMORY_UPDATE, END)

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