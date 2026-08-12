"""Debate service — Sprint 2.

Sprint 1: used a simple LCEL chain (prompt | llm | parser)
Sprint 2: replaced by LangGraph workflow with:
          - Classification node
          - Scoring node
          - Conditional routing
          - Three handler nodes
          - Counterargument node

The service is the only layer that knows about both the graph
AND the outside world (CLI, API). It translates between them.
"""

from langchain_groq import ChatGroq

from src.core.config import get_settings
from src.core.exceptions import DebateError, GraphError
from src.core.logger import get_logger
from src.graphs.debate_graph import get_debate_graph,reset_memory,get_buffer_memory
from src.graphs.state import DebateState
from src.parsers.debate_parsers import opening_statement_parser
from src.prompts.debate_prompts import opening_prompt

logger = get_logger(__name__)

def _summary_memory_snapshot() -> str:
    """Get current summary from module-level memory for initial state.

    We inject the existing summary into initial_state so the graph
    has it available at the start of each turn, not just at the end.
    """
    from src.graphs.debate_graph import _summary_memory
    return _summary_memory.get_summary()

class DebateService:

    def __init__(self) -> None:
        settings = get_settings()

        # Opening statement still uses simple chain
        # (no classification needed for AI's own opening)
        self.llm = ChatGroq(
            model=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            groq_api_key=settings.groq_api_key,
        )
        self.opening_chain = opening_prompt | self.llm | opening_statement_parser

        # LangGraph for all user argument processing
        self.graph = get_debate_graph()

        logger.info(
            f"DebateService (Sprint 2) initialised | "
            f"model={settings.default_model}"
        )

    def open_debate(self, topic: str, user_side: str) -> str:
        """Generate AI opening statement + reset all memory for new session"""
        reset_memory()
        ai_side = "against" if user_side == "for" else "for"
        try:
            logger.info(f"Opening debate | topic='{topic}' | ai_side={ai_side}")
            return self.opening_chain.invoke({"topic": topic, "side": ai_side})
        except Exception as e:
            raise DebateError(f"Opening statement failed: {e}") from e
    def process_argument(
        self,
        topic: str,
        user_side: str,
        user_argument: str,
        debate_history: list,
        turn_number: int = 1,
    ) -> dict:
        """Process user argument through the full LangGraph workflow.
        """
        try:
            logger.info(
                f"Processing argument | turn={turn_number} | "
                f"topic='{topic}'"
            )

            # Build initial state for this turn
            buffer = get_buffer_memory()
            initial_state: DebateState = {
                "topic": topic,
                "user_side": user_side,
                "user_argument": user_argument,
                "turn_number": turn_number,
                "debate_history": buffer.get_as_tuples(),
                "argument_quality": "",
                "quality_reasoning": "",
                "argument_score": 0,
                "score_breakdown": {},
                "handler_note": "",
                "ai_response": "",
                "error": "",
                "has_error": False,

                "contains_fallacy":    False,
                "fallacy_name":        "none",
                "fallacy_type":        "none",
                "fallacy_severity":    "none",
                "fallacy_explanation": "",
                "fallacy_correction":  "none",
                "debate_summary": _summary_memory_snapshot(),
                "similar_past_args":[],
                "memory_updated": False,
                "rag_context":""
            }

            # Run through graph
            final_state = self.graph.invoke(initial_state)

            logger.info(
                f"Argument processed | "
                f"quality={final_state.get('argument_quality')} | "
                f"score={final_state.get('argument_score')}/10"
            )

            return {
                "ai_response":       final_state.get("ai_response", ""),
                "argument_quality":  final_state.get("argument_quality", ""),
                "argument_score":    final_state.get("argument_score", 0),
                "score_breakdown":   final_state.get("score_breakdown", {}),
                "quality_reasoning": final_state.get("quality_reasoning", ""),
                "handler_note":      final_state.get("handler_note", ""),
                "error":             final_state.get("error", ""),
                "contains_fallacy":    final_state.get("contains_fallacy", False),
                "fallacy_name":        final_state.get("fallacy_name", "none"),
                "fallacy_severity":    final_state.get("fallacy_severity", "none"),
                "fallacy_explanation": final_state.get("fallacy_explanation", ""),
                "fallacy_correction":  final_state.get("fallacy_correction", "none"),

                "debate_summary":    final_state.get("debate_summary", ""),
                "similar_past_args": final_state.get("similar_past_args", []),
                "memory_updated":    final_state.get("memory_updated", False),
                "rag_context":       final_state.get("rag_context","")
            }

        except Exception as e:
            raise GraphError(f"Graph execution failed: {e}") from e