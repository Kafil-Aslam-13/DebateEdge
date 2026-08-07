# Custom exceptions will be here
""" Custom Exception hierarchy for DebateEdge.
Trace your pipeline mentally:
topic-> input -> validate -> debate -> detect fallacy -> score ->coach
Each failure mode gets its own exception class"""

class DebateEdgeError(Exception):
    """Base exception - catch in this any project error"""

class ConfigurationError(DebateEdgeError):
    """ config file missing , malformed , or missing required keys"""

class ValidationError(DebateEdgeError):
    """Input Failed guardrail validation"""

class DebateError(DebateEdgeError):
    """Debate agent failed to generate response"""

class FallacyDetectionError(DebateEdgeError):
    """Fallacy detection agent failed"""

class ScoringError(DebateEdgeError):
    """Argument scoring failed."""

class CoachingError(DebateEdgeError):
    """Coaching report generation failed."""

class GatewayError(DebateEdgeError):
    """LLM Gateway - All models failed or rate limit hit."""

class MemoryError(DebateEdgeError):
    """Memory read write failed"""

class RetrievalError(DebateEdgeError):
    """Vector store or database retrieval failed"""

class EmbeddingError(DebateEdgeError):
    """embedding generation failed"""

class ObservabilityError(DebateEdgeError):
    """Logfire or langSmith setup/logging failed - non-fatal."""
    