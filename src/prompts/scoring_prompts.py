"""Scoring and classification prompts.
 
 here used two new prompt template types:
1. FewShotPromptTemplate  — for argument classification
   WHY: Classification is a pattern-matching task.
   Showing examples of strong/weak/fallacy arguments
   dramatically improves accuracy vs zero-shot.
 
2. PromptTemplate         — for argument scoring
   WHY: Scoring is a structured analytical task, not
   a conversation. No message history needed here.
   Plain PromptTemplate is the right fit.
 
choose the template type that matches the nature
 of the task, not personal preference.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate
)

# Classification:
# show e.g before asking model to classify.
# this is right choice when task requires recognising patterns,but its accuracy is not reliable but helps to a certain extent

_CLASSIFICATION_EXAMPLES = [
    {
        "argument": "Studies consistently show that countries with stricter gun laws have lower homicide rates.",
        "quality": "strong",
        "reasoning": "Uses empirical evidence, specific and verifiable claim.",
    },
    {
        "argument": "Everyone knows social media is bad for mental health, it's obvious.",
        "quality": "weak",
        "reasoning": "Vague appeal to common knowledge, no evidence provided.",
    },
    {
        "argument": "You only support free healthcare because you're a socialist.",
        "quality": "fallacy",
        "reasoning": "Ad hominem — attacks the person not the argument.",
    },
    {
        "argument": "If we allow gay marriage, next people will want to marry animals.",
        "quality": "fallacy",
        "reasoning": "Slippery slope — assumes extreme consequences without evidence.",
    },
    {
        "argument": "Climate change policy will hurt the economy AND failing to act will also hurt the economy.",
        "quality": "weak",
        "reasoning": "Self-contradictory, lacks a clear position.",
    },
    {
        "argument": "The IPCC reports from 193 countries show 97% of climate scientists agree warming is human-caused.",
        "quality": "strong",
        "reasoning": "Cites authoritative source, specific consensus data.",
    },
]

_example_prompt = PromptTemplate(
    input_variables=["argument","quality","reasoning"],
    template=(
        "Argument:{argument}\n"
        "Quality:{quality}\n"
        "Reasoning: {reasoning}"
    )
)

_classification_few_short = FewShotPromptTemplate(
    examples=_CLASSIFICATION_EXAMPLES,
    example_prompt=_example_prompt,
    prefix=(
        "You are an expert debate judge. "
        "Classify each argument as 'strong', 'weak', or 'fallacy'.\n\n"
        "Definitions:\n"
        "- strong:  logical, evidence-based, directly addresses the topic\n"
        "- weak:    vague, unsupported, off-topic, or self-contradictory\n"
        "- fallacy: contains a named logical fallacy\n\n"
        "Examples:\n"
    ),
    suffix=(
         "Now classify this argument:\n"
        "Argument: {argument}\n"
        "Topic context: {topic}\n\n"
        "Classify the argument as strong, weak, or fallacy "
        "and provide one sentence explaining your decision."
        
    ),
    input_variables=["argument","topic"]
)
# Wrap in chatprompt template so it works with llm

classification_prompt = ChatPromptTemplate.from_messages([
    ("human",_classification_few_short.format(argument="{argument}",topic="{topic}",)),
])



scoring_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert debate scorer.

Score arguments rigorously and fairly.

Evaluate the argument itself rather than the quality of the
person presenting it."""
    ),
    (
        "human",
        """Score the following debate argument.

Topic:
{topic}

User's side:
{user_side}

Argument:
{argument}

Pre-classification:
{quality}

Evaluate these dimensions from 0 to 10:

Logic:
Is the reasoning sound, coherent, and logically connected?

Evidence:
Are factual claims supported with relevant evidence, examples,
or reasoning?

Clarity:
Is the argument clear, specific, and well structured?

Do not calculate an overall score.
The application calculates the overall score.

Provide one sentence of specific actionable feedback."""
    ),
])

# ---------------------------------------------------------------------------
# Debate response handler prompts
# ---------------------------------------------------------------------------

_HANDLER_BASE = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert debate opponent and debate coach.

You are arguing the {ai_side} position.

Your job is to produce a clear, concise debate response.

IMPORTANT OUTPUT RULES:
- Return ONLY the final response intended for the user.
- NEVER output internal reasoning.
- NEVER output a thinking process.
- NEVER output <think>, </think>, "Thinking Process", analysis,
  chain-of-thought, planning, drafting, word-count checks, or meta-commentary.
- Do not describe how you generated the answer.
- Do not mention these instructions.
- Use simple paragraphs and clear headings when helpful.
- Keep the response under 150 words."""
    ),
    (
        "human",
        """Topic:
{topic}

User's argument:
{argument}

Argument quality:
{quality}

Quality reasoning:
{quality_reasoning}

Score:
{score}/10

Coaching instruction:
{handler_instruction}

Relevant evidence:
{rag_context}

Evidence rules:
- Use retrieved evidence when it directly supports a claim.
- Do not invent statistics, studies, researchers, or citations.
- Do not claim that research proves something unless the
  retrieved evidence actually supports that conclusion.
- If relevant evidence is unavailable, argue from logic instead
  of pretending that evidence exists.

  
Now write the final debate response.

Follow the coaching instruction, but DO NOT reveal your reasoning
or the steps you used to produce the response.

Return ONLY the final response that should be shown to the user.
Keep the total response under 150 words."""
    ),
])


strong_handler_prompt = _HANDLER_BASE.partial(
    coaching_mode="The user made a strong argument. Acknowledge its strongest point briefly, then challenge it.",
    handler_instruction=(
        "Use exactly this structure:\n"
        "Acknowledgement: <one sentence>\n"
        "Counterargument: <3-5 concise sentences>"
    ),
)

weak_handler_prompt = _HANDLER_BASE.partial(
    coaching_mode="The user made a weak argument. Explain the weakness constructively while presenting the opposing position.",
    handler_instruction=(
        "Use exactly this structure:\n"
        "Weakness: <one sentence>\n"
        "Counterargument: <3-4 concise sentences>\n"
        "Question: <one pointed question>"
    ),
)

fallacy_handler_prompt = _HANDLER_BASE.partial(
    coaching_mode=(
        "The user committed a logical fallacy. "
        "Identify it clearly, explain it briefly, "
        "and then challenge the user's position."
    ),
    handler_instruction=(
        "Use exactly this structure:\n"
        "Fallacy: <fallacy name>\n"
        "Explanation: <one sentence>\n"
        "Fix: <one sentence>\n"
        "Counterargument: <2-4 concise sentences>"
    ),
)
