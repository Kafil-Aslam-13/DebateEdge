"""buffer window memory.

Keeps most recent N debate messages.

Modern LangChain/Langgraph approach:
conversation history is represented with langchain message objects.
the sliding window is managed explicitly.
Messages remain Humanmessage , AImessage so they can 
be passed directly to langchain"""

from langchain_core.messages import HumanMessage , AIMessage,BaseMessage

from src.core.logger import get_logger
logger=get_logger(__name__)

class DebateBufferMemory:
    """Manages a window of recent debate messages.
    6 messages= abou 3 debate turns"""

    def __init__(self, window_size: int=6)-> None:
        self.window_size = window_size
        self._messages:list[BaseMessage] = []

        logger.info(f"Buffer memory initialized | window ={window_size} messages")

    def add_turn(self,user_argument:str,ai_response:str)->None:
        """store one debate turn and enforce the sliding window."""
        self._messages.extend([
            HumanMessage(content=user_argument),
            AIMessage(content=ai_response),
        ])

        # keep most recent n messages
        self._messages = self._messages[-self.window_size:]
        logger.info(
            f"BufferMemory: turn saved | "
            f"total messages={self.get_message_count()}"
        )

    def get_messages(self)->list[BaseMessage]:
        """returns recent messages ready for langchain prompts."""
        return list(self._messages)

    def get_message_count(self) ->int:
        """Return number of messages currently in buffer"""
        return len(self._messages)

    def get_as_tuples(self)->list[tuple[str,str]]:
        """return messages as (role ,content) tuples"""
        tuples=[]
        for msg in self._messages:
            if isinstance(msg,HumanMessage):
                tuples.append(("human",msg.content))
            elif isinstance(msg,AIMessage):
                tuples.append(("assistant",msg.content))

        return tuples

    def clear(self) -> None:
        """Clear all recent conversation memory."""

        self._messages.clear()

        logger.info("BufferMemory cleared.")

    def is_near_limit(self, threshold: float = 0.8) -> bool:
        """Check whether the message window is near capacity.

        This is message-count based, not token-count based.
        """

        return self.get_message_count() >= int(
            self.window_size * threshold
        )
    
    def pop_oldest(self, count: int) -> list[BaseMessage]:
        """Remove and return the oldest messages."""

        count = min(count, len(self._messages))

        old_messages = self._messages[:count]
        self._messages = self._messages[count:]

        return old_messages