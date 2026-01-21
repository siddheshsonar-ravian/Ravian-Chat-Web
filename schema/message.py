from typing import TypedDict, Annotated

from langgraph.graph import add_messages


class MessagesState(TypedDict):
    """State for the chat agent"""
    messages: Annotated[list, add_messages]
