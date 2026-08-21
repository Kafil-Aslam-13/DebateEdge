"""Debate API routes"""

from fastapi import APIRouter ,HTTPException

from app.api.schemas.request import ArgueRequest , StartDebateRequest
from app.api.schemas.response import (
    ArgueResponse,
    CostInfo,
    EvalInfo,
    FallacyInfo,
    GuardInfo,
    ScoreBreakdown,
    SessionEvalResponse,
    StartDebateResponse
)

from src.core.config import get_settings
from src.core.exceptions import DebateEdgeError
from src.core.logger import get_logger
from src.gateway.cost_optimizer import CostOptimizer
from src.services.debate_service import DebateService

logger  = get_logger(__name__)

router = APIRouter(prefix="/api/v1/debate",tags=["Debate"])

# module level build once reused across requests
_service=DebateService()
_optimizer=CostOptimizer()
_settings=get_settings()

@router.post("/start",response_model=StartDebateResponse)
def start_debate(req:StartDebateRequest)->StartDebateResponse:
    """starts new debate session
    resets all memory and returns ai opening statement"""

    try:
        opening=_service.open_debate(
            topic=req.topic,
            user_side=req.user_side
        )

        ai_side = "against" if req.user_side == "for" else "for"
        return StartDebateResponse(
            opening_statement=opening,
            ai_side=ai_side,
            topic=req.topic
        )
    except DebateEdgeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in start_debate")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/argue",response_model=ArgueResponse)
def argue(req:ArgueRequest)->ArgueResponse:
    try:
        result = _service.process_argument(
            topic=req.topic,
            user_side=req.user_side,
            user_argument=req.argument,
            debate_history=[],
            turn_number=req.turn_number
        )
        

        cost = _optimizer.get_turn_cost_report()

        return ArgueResponse(
            ai_response=result.get("ai_response", ""),
            argument_quality=result.get("argument_quality", ""),
            argument_score=result.get("argument_score", 0),
            score_breakdown=ScoreBreakdown(
                **(
                    result.get("score_breakdown")
                    or {"logic": 0, "evidence": 0, "clarity": 0}
                )

            ),
            quality_reasoning=result.get("quality_reasoning", ""),
            fallacy=FallacyInfo(
                detected=result.get("contains_fallacy", False),
                name=result.get("fallacy_name", "none"),
                severity=result.get("fallacy_severity", "none"),
                explanation=result.get("fallacy_explanation", ""),
                correction=result.get("fallacy_correction", "none"),
            ),
            input_guard=GuardInfo(
                passed=result.get("input_guard_passed", True),
                reason=result.get("input_guard_reason", ""),
            ),
            evaluation=EvalInfo(
                score=result.get("turn_eval_score", 0),
                grade=result.get("turn_eval_grade", ""),
                feedback=result.get("turn_eval_feedback", ""),
            ),
            debate_summary=result.get("debate_summary", ""),
            similar_past_args=result.get("similar_past_args", []),
            cost=CostInfo(
                total_tokens=cost.get("total_tokens", 0),
                total_cost_usd=cost.get("total_cost_usd", 0.0),
                cache_hits=cost.get("cache_hits", 0),
            ),
            rag_context=result.get("rag_context", ""),
        )
    except DebateEdgeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in argue")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/evaluate",response_model=SessionEvalResponse)
def evaluate_session()->SessionEvalResponse:
    try:
        session_eval=_service.evaluate_session()
        if session_eval is None:
            raise HTTPException(
                status_code=400,
                detail="Insufficient debate turns for evaluation (need >= 2)",
            )
        return SessionEvalResponse(
            total_turns=session_eval.total_turns,
            user_improvement=session_eval.user_improvement,
            avg_score_first_half=session_eval.avg_score_first_half,
            avg_score_second_half=session_eval.avg_score_second_half,
            score_trend=session_eval.score_trend,
            fallacy_count=session_eval.fallacy_count,
            strong_count=session_eval.strong_count,
            weak_count=session_eval.weak_count,
            best_turn=session_eval.best_turn,
            worst_turn=session_eval.worst_turn,
            overall_grade=session_eval.overall_grade,
            improvement_advice=session_eval.improvement_advice,
            total_tokens=session_eval.total_tokens,
            total_cost_usd=session_eval.total_cost_usd,
            cache_hits=session_eval.cache_hits,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in evaluate_session")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/reset")
def reset_debate()->dict:
    """reset debate session - clears all memory and state"""
    try:
        _service.reset_debate()
        return {"status": "reset", "message": "Debate session cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost")
def get_cost() -> dict:
    """Get session cost report."""
    return _optimizer.get_session_cost_report()