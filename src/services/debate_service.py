"""Debate service

Orchestrates the basic debate flow:
topic + side + user argument -> AI counterargument

This is the layer debate_graph.py and the API both call into.
No LangGraph and No memory yet
Just a clean chain that works end to end.
"""

from langchain_groq import ChatGroq

from src.core.config import get_settings
from src.core.exceptions import DebateError
from src.core.logger import get_logger
from src.parsers.debate_parsers import (
    debate_response_parser,
    opening_statement_parser,
)
from src.prompts.debate_prompts import debate_prompt, opening_prompt

logger = get_logger(__name__)


class DebateService:

    def __init__(self) -> None:
        settings = get_settings()

        self.llm = ChatGroq(
            model=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            groq_api_key=settings.groq_api_key,
        )

        # LCEL chains — prompt | llm | parser
        self.debate_chain = debate_prompt | self.llm | debate_response_parser
        self.opening_chain = opening_prompt | self.llm | opening_statement_parser

        logger.info(f"DebateService initialised with model: {settings.default_model}")

    def open_debate(self, topic: str, user_side: str) -> str:
        """Generate AI opening statement.

        AI takes the opposite side of whatever the user chose.

        Args:
            topic:     The debate topic
            user_side: "for" or "against" (user's side)

        Returns:
            AI opening statement string
        """
        # AI argues opposite side to user
        ai_side = "against" if user_side == "for" else "for"

        try:
            logger.info(f"Opening debate | topic='{topic}' | ai_side={ai_side}")

            response = self.opening_chain.invoke({
                "topic": topic,
                "side": ai_side,
            })

            logger.info("Opening statement generated successfully.")
            return response

        except Exception as e:
            raise DebateError(f"Failed to generate opening statement: {e}") from e

    def argue(
        self,
        topic: str,
        user_side: str,
        user_argument: str,
        debate_history: list,
    ) -> str:
        """Generate AI counterargument to user's argument.

        Args:
            topic:          The debate topic
            user_side:      User's chosen side ("for" or "against")
            user_argument:  What the user just argued
            debate_history: List of previous messages in this debate

        Returns:
            AI counterargument string
        """
        ai_side = "against" if user_side == "for" else "for"

        try:
            logger.info(
                f"Generating counterargument | "
                f"topic='{topic}' | "
                f"turns={len(debate_history)}"
            )

            response = self.debate_chain.invoke({
                "topic": topic,
                "side": ai_side,
                "user_argument": user_argument,
                "debate_history": debate_history,
            })

            logger.info("Counterargument generated successfully.")
            return response

        except Exception as e:
            raise DebateError(
                f"Failed to generate counterargument: {e}"
            ) from e