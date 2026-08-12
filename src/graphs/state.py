"""LangGraph state definition for DebateEdge.
 
The state is the single object that flows through every node
in the graph. Every node reads from it and writes back to it.
"""

from typing import Annotated , Any , Literal
from typing_extensions import TypedDict
from langgraph.graph import add_messages

class DebateState(TypedDict):
    """
    State that flows through the debate graph

    Every field has a clear owner:
    - Input fields: Set by caller before the graph runs
    - Node fields: set by specific nodes during graph execution
    - Output fields: read by the caller after graph completes
    """

    # Input Fields
    topic: str
    user_side: str
    user_argument:str
    turn_number: int

    # conversation history - grows each turn 
    # Annotated  with add messages so langgraph appends instead of overwrite
    debate_history: Annotated[list,add_messages]

    # Classification node Output
    argument_quality:Literal["strong","weak","fallacy"]  # strong | weak | fallacy
    quality_reasoning:str 

    #  Scoring node output
    argument_score:float
    score_breakdown:dict 

    # handler node output
    handler_note:str 

    # counterargument node o/p
    ai_response:str

    # Fallacy
    contains_fallacy:bool
    fallacy_name:str
    fallacy_type:str
    fallacy_severity:str
    fallacy_explanation:str
    fallacy_correction:str


    #   Memory
    debate_summary: str
    similar_past_args: list
    memory_updated:bool

    # rag
    rag_context:str


    # Error Tracking
    error:str
    has_error: bool
