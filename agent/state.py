"""
Shared state schema for the research agent graph.
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single page the agent read and used as grounding for its answer."""

    title: str = Field(description="Page title")
    url: str = Field(description="Page URL")
    snippet: str = Field(default="", description="Why this source was useful")


def _merge_sources(existing: List[Source], new: List[Source]) -> List[Source]:
    """Reducer for the sources list: append, de-duplicated by URL."""
    seen = {s.url for s in existing}
    merged = list(existing)
    for s in new:
        if s.url not in seen:
            merged.append(s)
            seen.add(s.url)
    return merged


class AgentState(TypedDict):
    """
    messages:       full conversation trail (Human/AI/Tool messages),
                     accumulated automatically via add_messages.
    research_notes: flat log of tool outputs gathered along the way,
                     kept separate from `messages` so the synthesizer
                     has a clean, de-duplicated trail to work from.
    sources:        structured, de-duplicated list of pages actually
                     read (via read_page), used to render a clickable
                     "Sources" section -- this is the structured-output
                     piece, kept separate from prose so the UI can
                     render it without re-parsing the final answer.
    final_answer:   set once the synthesizer node runs.
    iterations:     planner loop counter, used by the guardrail that
                     forces synthesis after MAX_ITERATIONS.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    research_notes: Annotated[List[str], operator.add]
    sources: Annotated[List[Source], _merge_sources]
    final_answer: Optional[str]
    iterations: int