"""
Shared state schema for the research agent graph.
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Source(TypedDict):
    """A single page the agent read and used as grounding for its answer.

    Plain TypedDict (not a Pydantic model) so it serializes cleanly
    through LangGraph's checkpointer without any custom-type
    registration -- checkpoints are just JSON/msgpack-friendly dicts.
    """

    title: str
    url: str
    snippet: str


def _merge_sources(existing: List[Source], new: List[Source]) -> List[Source]:
    """Reducer for the sources list: append, de-duplicated by URL."""
    seen = {s["url"] for s in existing}
    merged = list(existing)
    for s in new:
        if s["url"] not in seen:
            merged.append(s)
            seen.add(s["url"])
    return merged


class AgentState(TypedDict):
    """
    messages:       full conversation trail (Human/AI/Tool messages),
                     accumulated automatically via add_messages. This
                     is also the basis for multi-turn memory: with a
                     checkpointer + thread_id, this list persists and
                     grows across separate run_agent() calls instead
                     of resetting each time.
    research_notes: flat log of tool outputs gathered along the way,
                     kept separate from `messages` so the synthesizer
                     has a clean, de-duplicated trail to work from.
                     Also accumulates across turns via the checkpointer.
    sources:        structured, de-duplicated list of pages actually
                     read (via read_page), used to render a clickable
                     "Sources" section -- kept separate from prose so
                     the UI can render it without re-parsing the final
                     answer. Also accumulates across turns.
    final_answer:   set once the synthesizer node runs; reset to None
                     at the start of every new turn.
    iterations:     planner loop counter for the current turn only,
                     used by the guardrail that forces synthesis after
                     MAX_ITERATIONS; reset to 0 at the start of every
                     new turn.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    research_notes: Annotated[List[str], operator.add]
    sources: Annotated[List[Source], _merge_sources]
    final_answer: Optional[str]
    iterations: int