"""Debate output parsers.

StrOutputParser for conversational debate responses.
 in Future we can add PydanticOutputParser, JsonOutputParser etc.
"""

from langchain_core.output_parsers import StrOutputParser


debate_response_parser = StrOutputParser()

opening_statement_parser = StrOutputParser()