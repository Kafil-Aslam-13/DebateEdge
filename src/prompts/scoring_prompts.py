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
        "Return only valid JSON.\n"
        "Do not include markdown or explanations outside the JSON.\n"
        "Respond in this exact JSON format:\n"
        '{{{{"quality": "strong|weak|fallacy", "reasoning": "one sentence explanation"}}}}'
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
        "You are an expert debate scorer. Score arguments rigorously and fairly.",
    ),
    (
        "human",
        """Score the following debate argument on a scale of 0-10.
 
Topic: {topic}
User's side: {user_side}
Argument: {argument}
Pre-classification: {quality} (use this as context)
 
Score these three dimensions (0-10 each):
- Logic:    Is the reasoning sound and coherent?
- Evidence: Are claims supported with facts or examples?
- Clarity:  Is the argument clear and well-structured?
 
do not calculate an overall score.
The application will calculate the overall score.

Return only valid JSON.
Do not include markdown or explanations outside the JSON.


Return only valid JSON with:
{{
  "logic": <0-10>,
  "evidence": <0-10>,
  "clarity": <0-10>,
  "feedback": "<one sentence of specific actionable feedback>"
}}
"""
    ),
])

#  Handler prompt - Using Partial technique .partial

_HANDLER_BASE = ChatPromptTemplate.from_messages([
    ("system","You are a  debate coach giving targetted feedback. {coaching_mode}"),
    ("human",
     """ the user made a {quality} argument about: {topic}
     Their argument: {argument}
     Score: {score}/10
     Reasoning: {quality_reasoning}
     
     {handler_instruction}
     
     Then give your counterargument representingh the {ai_side} position.
     Keep total response Under 150 words"""),
])

#  Pre Fill the coach mode and handler instructions for each handler type:

strong_handler_prompt=_HANDLER_BASE.partial(
    coaching_mode="The user made a strong argument. Acknowledge briefly then challenge hard.",
    handler_instruction="Give One sentence of genuine acklodgement. Then attack their strongest point.",
)

weak_handler_prompt = _HANDLER_BASE.partial(
    coaching_mode="The user made a weak argument. Point out the weakness constructively.",
    handler_instruction="Name specifically what was weak. Ask one pointed question to guide them.",
)

fallacy_handler_prompt = _HANDLER_BASE.partial(
    coaching_mode="The user committed a logical fallacy. Name it and explain it clearly.",
    handler_instruction="Name the fallacy. Explain it in one sentence. Show them how to fix it.",
)