import json
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent 
from langchain_core.messages import SystemMessage

from src.core.config import get_settings
from src.core.constants import (
    FALLACY_AD_HOMINEM,
    FALLACY_APPEAL_TO_EMOTION,
    FALLACY_CIRCULAR_REASONING,
    FALLACY_FALSE_DICHOTOMY,
    FALLACY_HASTY_GENERALIZATION,
    FALLACY_RED_HERRING,
    FALLACY_SLIPPERY_SLOPE,
    FALLACY_STRAWMAN,
    QUALITY_FALLACY,
)
from src.core.exceptions import FallacyDetectionError
from src.core.logger import get_logger
from src.parsers.fallacy_parsers import(
    FallacyDetectionResult,
    FallacySeverity,
    FallacyType,fallacy_explanation_parser,
    parse_fallacy_safe
)
from src.prompts.fallacy_prompts import (

    fallacy_explanation_prompt,
    FALLACY_SYSTEM_PROMPT
)

logger = get_logger(__name__)


_FALLACY_KB = {
    FALLACY_STRAWMAN: {
        "definition": "Misrepresenting an argument to make it easier to attack.",
        "example": "Person A: reduce military spending. Person B: A wants us defenceless.",
        "type": "informal",
    },
    FALLACY_AD_HOMINEM: {
        "definition": "Attacking the person rather than their argument.",
        "example": "You can't trust his economic policy — he went bankrupt twice.",
        "type": "informal",
    },
    FALLACY_FALSE_DICHOTOMY: {
        "definition": "Presenting only two options when more exist.",
        "example": "You're either with us or against us.",
        "type": "informal",
    },
    FALLACY_SLIPPERY_SLOPE: {
        "definition": "Claiming one event leads to extreme consequences without evidence.",
        "example": "Legalise marijuana and soon all drugs will be legal.",
        "type": "informal",
    },
    FALLACY_APPEAL_TO_EMOTION: {
        "definition": "Using emotional manipulation instead of logical argument.",
        "example": "Think of the poor children who will suffer.",
        "type": "informal",
    },
    FALLACY_HASTY_GENERALIZATION: {
        "definition": "Drawing a broad conclusion from too small a sample.",
        "example": "I met two rude Americans so Americans are rude.",
        "type": "informal",
    },
    FALLACY_CIRCULAR_REASONING: {
        "definition": "Using the conclusion as a premise.",
        "example": "The Bible is true because it says so.",
        "type": "formal",
    },
    FALLACY_RED_HERRING: {
        "definition": "Introducing irrelevant information to distract.",
        "example": "Why worry about climate change when there's poverty?",
        "type": "informal",
    },
}

# tools
@tool 
def lookup_fallacy(fallacy_name:str)-> str:
    """Look up definition and example of a specific logical fallacy.
    Call this before making your final classification to verify the argument actually matches the fallacy definition.
    Args:
    fallacy_name: e.g 'ad_hominen','strawman' 
    """
    name= fallacy_name.lower().strip().replace(" ","_")
    if name in _FALLACY_KB:
        info = _FALLACY_KB[name]
        return (
            f"Fallacy: {name}\n"
            f"Definition: {info['definition']}\n"
            f"Example: {info['example']}\n"
            f"Type: {info['type']}"
        )
    known = list(_FALLACY_KB.keys())
    return f"{fallacy_name} not found. Known fallacies: {known}"

@tool
def list_fallacies()->str:
    """List all detectable fallacies .
    call this when unsure which fallacy applies and want to compare
    """
    return "\n".join(
        f" - {name}:{info['definition']}"
        for name , info in _FALLACY_KB.items()
    )



def _build_agent():
    """Build agent using langgraph."""
    settings = get_settings()
    llm=ChatGroq(
        model=settings.complex_model,
        temperature=0.1,
        api_key=settings.groq_api_key
    )

    tools = [lookup_fallacy,list_fallacies]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=FALLACY_SYSTEM_PROMPT
    )

    return agent

class FallacyDetectionService:
    """Orchestrates detection and explanation
    detect uses agent with tools
    explain uses simple lecl chain"""

    def __init__(self)->None:
        settings=get_settings()
        self._agent=None
        explanation_llm = ChatGroq(
            model=settings.default_model,
            temperature=0.5,
            api_key=settings.groq_api_key
        )
        self.explanation_chain=(
            fallacy_explanation_prompt | explanation_llm | fallacy_explanation_parser
        )

        logger.info("Fallacy detection service initialised")

    def _get_agent(self):
        if self._agent is None:
            self._agent = _build_agent()
            logger.info("Fallacy agent build")
        return self._agent

    def detect(self,argument:str,topic:str,user_side:str)->FallacyDetectionResult:
        """Detect fallacy using agent with tools
        agent can call lookup fallacy and list fallacies 
        before making its final classification."""

        try:
            logger.info(f"detect fallacy | topic='{topic[:50]}'")

            agent = self._get_agent()
            # build a prompt message for agent
            user_message=(
                f"Analyse this debate argument for logical fallacies.\n\n"
                f"Topic: {topic}\n"
                f"Argument: {argument}\n"
                f"Arguer's side: {user_side}\n\n"
                f"Use your tools to look up relevant fallacy definitions "
                f"before making your classification.\n\n"
                f"Respond with valid JSON only:\n"
            )
            result = agent.invoke({
                "messages": [("user", user_message)]
            })

            
            last_message = result["messages"][-1]
            output_text = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message))

            parsed = parse_fallacy_safe(output_text)
            logger.info(
                f"Fallacy detection done"
                f"contains={parsed.contains_fallacy}"
                f"name={parsed.fallacy_name}"
            )

            return parsed
        except Exception as e:
            logger.error(f"Fallacy detection failed: {e}")
            raise FallacyDetectionError(f"Detection failed {e}") from e

    def explain(self,fallacy_name:str,argument:str)->str:
        """Explain a fallacy in plain english using lecl chain.
        """
        try:
            return self.explanation_chain.invoke({
                "fallacy_name":fallacy_name,
                "argument":argument,
            })
        except Exception as e:
            logger.error(f"Fallacy explanation failed: {e}")
            return f"Your argument contains a {fallacy_name} fallacy."