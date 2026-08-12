"""Parsers for Scoring and classification results

we add:
-JsonOutputParser - why?
Classification and scoring return structures data 
(quality label , reasoning , scores). Json O/P parser
 converts LLM's Json string into a python dict automatically.
 - with_structured_output(Pydantic) → when you have a defined schema
and want guaranteed valid output
 """
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser


class ClassificationOutput(BaseModel):
    """Structured result from argument classification."""

    quality: Literal["strong", "weak", "fallacy"] = Field(
        description="Argument quality"
    )

    reasoning: str = Field(
        description="One sentence explaining the classification"
    )


class ScoreOutput(BaseModel):
    """Structured result from argument scoring."""

    logic: int = Field(
        description="Logic score from 0 to 10",
        ge=0,
        le=10,
    )

    evidence: int = Field(
        description="Evidence score from 0 to 10",
        ge=0,
        le=10,
    )

    clarity: int = Field(
        description="Clarity score from 0 to 10",
        ge=0,
        le=10,
    )

    feedback: str = Field(
        description="One sentence of actionable feedback"
    )


handler_response_parser = StrOutputParser()

