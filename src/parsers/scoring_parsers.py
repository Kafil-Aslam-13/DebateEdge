"""Parsers for Scoring and classification results

we add:
-JsonOutputParser - why?
Classification and scoring return structures data 
(quality label , reasoning , scores). Json O/P parser
 converts LLM's Json string into a python dict automatically.
 - with_structured_output(Pydantic) → when you have a defined schema
and want guaranteed valid output
 """
import json
from typing import Optional
from langchain_core.output_parsers import JsonOutputParser , StrOutputParser
from pydantic import BaseModel , Field
from langchain_groq import ChatGroq
from typing import Literal
from src.core.config import get_settings
from src.core.logger import get_logger
logger = get_logger(__name__)


class ClassificationOutput(BaseModel):
    """Schema for argument classification result."""
    quality: Literal["strong", "weak", "fallacy"] = Field(
        description="Argument quality: must be 'strong', 'weak', or 'fallacy'"
    )
    reasoning: str = Field(
        description="One sentence explaining why this quality was assigned"
    )


class ScoreOutput(BaseModel):
    """Schema for argument scoring result."""
    logic: int = Field(description="Logic score 0-10",ge=0,le=10)
    evidence: int = Field(description="Evidence score 0-10",ge=0,le=10)
    clarity: int = Field(description="Clarity score 0-10",ge=0,le=10)
    # overall: int = Field(description="Overall weighted score 0-10",ge=0,le=10)
    feedback: str = Field(description="One sentence of specific actionable feedback")




handler_response_parser = StrOutputParser()

# structured o/p builders

def get_structured_classifier():
    """Return LLM constrained to return ClassificationOutput.
    
    Uses Groq's native JSON mode to guarantee valid output.
    
    Usage:
        classifier = get_structured_classifier()
        result = classifier.invoke("classify this argument...")
        # result is always a valid ClassificationOutput dict
    """
    settings=get_settings()

    llm = ChatGroq(
        model=settings.default_model,
        temperature=0,
        api_key=settings.groq_api_key
    )
    return llm.with_structured_output(ClassificationOutput,method="json_mode")

def get_structured_scorer():
    """Return LLM constrained to return ScoreOutput.
    
    Usage:
        scorer = get_structured_scorer()
        result = scorer.invoke("score this argument...")
        # result is always a valid ScoreOutput dict
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.default_model,
        temperature=0,
        api_key=settings.groq_api_key,
    )
    return llm.with_structured_output(
        ScoreOutput,
        method="json_mode",
    )

def safe_parse_json(text: str, fallback: dict) -> dict:
    """Parse JSON string safely, returning fallback on failure.
    
    Used as last resort when structured output isn't available.
    with_structured_output() should eliminate the need for this
    in most cases — but kept as a safety net.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"JSON parse failed, using fallback. Text: {str(text)[:100]}")
        return fallback

