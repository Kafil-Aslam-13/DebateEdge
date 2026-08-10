"""Fallacy detection prompts

Uses only ChatPromptTemplate (no deprecated FewShotChatMessagePromptTemplate).

WHY PLAIN ChatPromptTemplate WITH BAKED-IN EXAMPLES:
FewShotChatMessagePromptTemplate is messy and brittle with
newer LangChain versions. The cleanest modern approach is to
bake the few-shot examples directly into the system message
as formatted text. Same effect, zero deprecated imports,
easier to read and maintain.

TWO PROMPTS — TWO DIFFERENT TECHNIQUES:

1. fallacy_detection_prompt
   → System message contains few-shot examples as formatted text
   → Human message contains the actual argument to classify
   → WHY: Few-shot in system + task in human is the most
     reliable pattern for chat models

2. fallacy_explanation_prompt
   → Pure instruction prompt, no examples
   → WHY: Explanation is a generative task not a classification task.
     Examples would constrain the output — we want natural language.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Few-shot examples baked into system message ───────────────────────────────
_FALLACY_EXAMPLES ="""
EXAMPLE 1:
Argument: "You can't trust John's opinion on climate change — he's not a scientist."
Result: contains_fallacy=true, fallacy_name=ad_hominem, severity=high
Reason: Attacks the person's credentials rather than the argument itself.

EXAMPLE 2:
Argument: "If we allow same-sex marriage, next people will want to marry animals."
Result: contains_fallacy=true, fallacy_name=slippery_slope, severity=high
Reason: Assumes extreme chain of events without evidence for each step.

EXAMPLE 3:
Argument: "Studies from 50 countries consistently show stricter gun laws lower homicide rates."
Result: contains_fallacy=false, fallacy_name=none, severity=none
Reason: Evidence-based, specific, verifiable claim. No fallacy present.

EXAMPLE 4:
Argument: "You either support this war or you hate our soldiers."
Result: contains_fallacy=true, fallacy_name=false_dichotomy, severity=high
Reason: Only two options presented when many positions exist.

EXAMPLE 5:
Argument: "Think of the children — how can you support this policy?"
Result: contains_fallacy=true, fallacy_name=appeal_to_emotion, severity=medium
Reason: Emotional manipulation used instead of logical argument.

EXAMPLE 6:
Argument: "I met three rude French people, so French people are generally rude."
Result: contains_fallacy=true, fallacy_name=hasty_generalization, severity=medium
Reason: Broad conclusion drawn from too small a sample size.
"""


# ── Prompt 2: Plain explanation — no examples needed ─────────────────────────
fallacy_explanation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a debate coach explaining logical fallacies clearly "
        "and encouragingly to students who want to improve.",
    ),
    (
        "human",
        """The debater used a {fallacy_name} fallacy in this argument:
"{argument}"

Explain in under 80 words:
1. What this fallacy is
2. Why their argument is an example of it
3. How to make the same point without the fallacy

Be encouraging and specific. No jargon.""",
    ),
])


FALLACY_SYSTEM_PROMPT = f"""You are an expert logician specialising in identifying logical fallacies in debate arguments.

You have access to two tools:
- lookup_fallacy: look up the definition of a specific fallacy before classifying
- list_fallacies: see all detectable fallacies when unsure which applies

Your process:
1. Read the argument carefully
2. If you suspect a fallacy, use lookup_fallacy to verify it matches the definition
3. If unsure which fallacy, use list_fallacies to compare options
4. Make your final classification

Known fallacies: ad_hominem, strawman, false_dichotomy, slippery_slope, 
appeal_to_emotion, hasty_generalization, circular_reasoning, red_herring

Examples: {_FALLACY_EXAMPLES}

Always respond with valid JSON only — no text before or after:
{{
    "contains_fallacy": true or false,
    "fallacy_name": "fallacy name or none",
    "fallacy_type": "formal or informal or none",
    "explanation": "one sentence explaining why",
    "severity": "high or medium or low or none",
    "correction": "how to fix the argument or none"
}}"""