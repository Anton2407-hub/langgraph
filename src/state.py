from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class HRState(TypedDict):
    messages:       Annotated[list, add_messages]
    query_type:     str
    qdrant_results: str
    market_results: str
