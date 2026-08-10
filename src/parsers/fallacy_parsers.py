"""Fallacy detection parsers — Sprint 3.

Uses pydantic v2 (not deprecated langchain_core.pydantic_v1).

PARSER TYPES SO FAR:
Sprint 1: StrOutputParser          → plain string
Sprint 2: JsonOutputParser         → dict (no schema)
          with_structured_output() → constrained at generation time
Sprint 3: PydanticOutputParser     → Pydantic model with validators

WHY PydanticOutputParser HERE (not with_structured_output):
Fallacy detection needs cross-field validation:
- if contains_fallacy=False → force fallacy_name, severity, correction to "none"
This validator logic lives in the Pydantic model itself.
with_structured_output() can't run Python validators — it only
constrains the JSON shape at generation time.
PydanticOutputParser runs AFTER generation and executes validators.
Both are valid modern approaches — choose based on whether you need
Python-level validation logic.
"""

import json
from enum import Enum
from typing import Optional


from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class FallacyType(str, Enum):
    FORMAL   = "formal"
    INFORMAL = "informal"
    NONE     = "none"


class FallacySeverity(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"
    NONE   = "none"


# ── Pydantic model ────────────────────────────────────────────────────────────

class FallacyDetectionResult(BaseModel):
    """Structured result of fallacy detection with cross-field validation."""

    contains_fallacy: bool = Field(
        description="Whether the argument contains a logical fallacy"
    )
    fallacy_name: str = Field(
        description="Name of the fallacy or 'none'"
    )
    fallacy_type: FallacyType = Field(
        description="Type: formal, informal, or none"
    )
    explanation: str = Field(
        description="One sentence explaining why"
    )
    severity: FallacySeverity = Field(
        description="Severity: high, medium, low, or none"
    )
    correction: str = Field(
        description="How to fix the argument or 'none'"
    )

    @field_validator("fallacy_name")
    @classmethod
    def normalise_name(cls, v: str) -> str:
        """Lowercase, strip, underscore the fallacy name."""
        return v.lower().strip().replace(" ", "_")

    @model_validator(mode="after")
    def enforce_consistency(self) -> "FallacyDetectionResult":
        """If no fallacy found force all related fields to none."""
        if not self.contains_fallacy:
            self.fallacy_name = "none"
            self.fallacy_type = FallacyType.NONE
            self.severity     = FallacySeverity.NONE
            self.correction   = "none"
        return self


# ── Parser instances ──────────────────────────────────────────────────────────

# PydanticOutputParser — third parser type introduced in Sprint 3
fallacy_parser = PydanticOutputParser(pydantic_object=FallacyDetectionResult)

# StrOutputParser — for plain English explanations
fallacy_explanation_parser = StrOutputParser()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_format_instructions() -> str:
    """Return format instructions generated from the Pydantic schema."""
    return fallacy_parser.get_format_instructions()


def parse_fallacy_safe(text: str) -> FallacyDetectionResult:
    """Parse LLM output into FallacyDetectionResult.

    Falls back to a safe default if parsing fails.
    Tries json.loads first (cleaner) then PydanticOutputParser.
    """
    try:
        # Try direct JSON parse first
        data = json.loads(text)
        return FallacyDetectionResult(**data)
    except Exception:
        pass

    try:
        # Try PydanticOutputParser (handles markdown fences etc)
        return fallacy_parser.parse(text)
    except Exception:
        pass

    # Safe default — no fallacy
    return FallacyDetectionResult(
        contains_fallacy=False,
        fallacy_name="none",
        fallacy_type=FallacyType.NONE,
        explanation="Fallacy parsing failed — defaulting to no fallacy.",
        severity=FallacySeverity.NONE,
        correction="none",
    )