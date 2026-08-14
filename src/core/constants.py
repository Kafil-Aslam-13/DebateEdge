# fixed values will be here
"""Project-wide constants. No logic, just named fixed values."""

PROJECT_NAME = "DebateEdge"
VERSION = "0.1.0"

# Debate sides
SIDE_FOR = "for"
SIDE_AGAINST = "against"
VALID_SIDES = [SIDE_FOR, SIDE_AGAINST]

# Argument quality levels
QUALITY_STRONG = "strong"
QUALITY_WEAK = "weak"
QUALITY_FALLACY = "fallacy"
QUALITY_LEVELS = [QUALITY_STRONG, QUALITY_WEAK, QUALITY_FALLACY]

# Fallacy types
FALLACY_STRAWMAN = "strawman"
FALLACY_AD_HOMINEM = "ad_hominem"
FALLACY_FALSE_DICHOTOMY = "false_dichotomy"
FALLACY_SLIPPERY_SLOPE = "slippery_slope"
FALLACY_APPEAL_TO_EMOTION = "appeal_to_emotion"
FALLACY_HASTY_GENERALIZATION = "hasty_generalization"
FALLACY_CIRCULAR_REASONING = "circular_reasoning"
FALLACY_RED_HERRING = "red_herring"
FALLACY_NONE = "none"

# Score boundaries
SCORE_MIN = 0
SCORE_MAX = 10
SCORE_STRONG_THRESHOLD = 7
SCORE_WEAK_THRESHOLD = 4

# Logfire metric names (used in Sprint 9)
METRIC_DEBATE_SCORE = "debate.argument.score"
METRIC_FALLACY_COUNT = "debate.fallacy.count"
METRIC_SESSION_TURNS = "debate.session.turns"
METRIC_LLM_COST = "debate.llm.cost_usd"
METRIC_LLM_TOKENS = "debate.llm.tokens"

# Memory
BUFFER_MEMORY_KEY = "debate_history"
SUMMARY_MEMORY_KEY = "debate_summary"
VECTOR_MEMORY_KEY = "past_arguments"

# Default config paths
CONFIG_PATH = "configs/config.yaml"
MODELS_CONFIG_PATH = "configs/models.yaml"
PROMPTS_CONFIG_PATH = "configs/prompts.yaml"

#  Different nodes in a graph
NODE_CLASSIFY = "classify_argument"
NODE_SCORE    = "score_argument"
NODE_ROUTE    = "route_by_quality"
NODE_STRONG   = "handle_strong"
NODE_WEAK     = "handle_weak"
NODE_FALLACY  = "handle_fallacy"
NODE_COUNTER  = "generate_counterargument"
NODE_FALLACY_DETECT = "detect_fallacy_details"
NODE_MEMORY_UPDATE = "update_memory"
NODE_RAG_RETRIEVE = "rag_retrieval"

#  Task types for LLM GATEWAY
TASK_CLASSIFICATION = "classification"
TASK_SCORING        = "scoring"
TASK_DEBATE         = "debate"
TASK_FALLACY        = "fallacy"
TASK_SUMMARY        = "summary"
TASK_EXPLANATION    = "explanation"


# Guardrails
NODE_INPUT_GUARD = "input_guardrail"
NODE_OUTPUT_GUARD= "output_guardrail"

# Guard actions
GUARD_PASS = "pass"
GUARD_BLOCK = "block"
GUARD_WARN = "warn"

# Guard names
GUARD_LENGTH="length_check"
GUARD_PII="pii_detection"
GUARD_PROMPT_INJECTION="prompt_injection"
GUARD_HATE_SPEECH="hate_speech"
GUARD_TOPIC_RELEVANCE="topic_relevance"
GUARD_TOXICITY="toxicity_filter"
GUARD_RELEVANCE="response_relevance"
GUARD_DISCLAIMER="legal_disclaimer"
GUARD_SENSITIVE="sensitive_topic"

SENSITIVE_TOPICS = [
    "suicide","self-harm","drugs","voilence",
    "weapons","terrorism","abuse", "mental health"
]