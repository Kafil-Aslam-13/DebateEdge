""" Summary memory
Maintains a compact summary of old debate turns

the memory is explicitly stored as string.
A lightweight LLM updates the summary when requested.
RRecent messages remain in beief memory. and
older context is compressed into this summary."""

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You maintain a concise summary of an ongoing debate.

Preserve:
- the user's main arguments
- important counterarguments
- changes in the user's position
- important evidence or claims
- weaknesses or unresolved points

Do not invent information.

Keep the summary concise enough to be reused as context
for future debate turns.""",
        ),
        (
            "human",
            """Existing summary:

{existing_summary}

New debate messages:

{messages}

Create an updated summary that combines the existing summary
with the new messages.

Return only the updated summary.""",
        ),
    ]
)


class DebateSummaryMemory:
    """maintains a running summary of older debate context"""

    def __init__(self)-> None:
        settings=get_settings()

        self._llm=ChatGroq(
            model=settings.default_model,
            temperature=0.3,
            api_key=settings.groq_api_key,
        )

        self._summary = ""
        logger.info("summary initialized")

    def update(
            self,
            messages:list[BaseMessage],
    )-> None:
        """Update the running summary using messages"""

        if not messages:
            return

        formatted_messages = "\n".join(
            f"{message.type}: {message.content}"
            for message in messages
        )

        prompt = _SUMMARY_PROMPT.invoke({
            "existing_summary": self._summary or "(none)",
            "messages":formatted_messages
        })

        response = self._llm.invoke(prompt)
        self._summary = response.content.strip()
        logger.info(f"Summarised memory updated")

    def get_summary(self)-> str:
        return self._summary

    def has_summary(self)-> bool:
        return bool (self._summary.strip())

    def clear(self)->None:
        """clear running summary"""
        self._summary = ""
        logger.info("SummaryMemory Cleared")

        
