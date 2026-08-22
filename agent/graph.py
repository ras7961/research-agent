r"""
Web research agent — plan → act (tools) → observe → decide loop,
built as a LangGraph StateGraph.

Graph shape:

    START -> planner --(tool_calls?)--> researcher -> planner (loop)
                    \--(no tool_calls, or max iters)--> synthesizer -> END

- planner:     a cheap/fast LLM bound to [web_search, read_page]. It
               looks at the conversation + notes so far and either
               (a) emits tool calls to gather more info, or (b) emits
               a plain message signalling it has enough to answer.
               THIS is the node that makes the "when to call tools"
               decision — a model decision, not a hardcoded rule.
- researcher:  a prebuilt ToolNode that actually executes whatever
               tool calls the planner emitted, appends results as
               ToolMessages ("observe"), and extracts structured
               Source records from any read_page calls.
- synthesizer: a stronger LLM (no tools bound) that reads the
               accumulated research notes (NOT the raw message
               history) and writes the final, cited answer. It
               deliberately does not see raw ToolMessages -- some
               providers (Gemini included) can return an empty
               response when a tools-unbound model is handed
               function-response messages it doesn't recognize.

Cost control: planning happens many times per run (one call per loop
iteration) but is a comparatively easy "decide next action" task, so
it uses a cheaper/faster model (PLANNER_MODEL). Synthesis happens
exactly once per run and is what the user actually reads, so it uses
a stronger model (SYNTHESIZER_MODEL). Override either via env vars.

A max_iterations guardrail forces a move to synthesizer even if the
planner keeps wanting to search, so the loop can't run forever.
"""

import os
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState, Source
from agent.tools import TOOLS
from agent.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "6"))

# Cheaper/faster model for the high-frequency "what do I do next" decision;
# stronger model for the one-shot final write-up the user actually reads.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gemini-3.6-flash")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "gemini-3.6-flash")

planner_llm = ChatGoogleGenerativeAI(model=PLANNER_MODEL, temperature=0)
planner_llm_with_tools = planner_llm.bind_tools(TOOLS)

synthesizer_llm = ChatGoogleGenerativeAI(model=SYNTHESIZER_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_text(content) -> str:
    """Normalize a model response's .content into a plain string.

    Anthropic models return a plain str. Gemini (via langchain_google_genai)
    can return a list of content-part dicts instead, e.g.
    [{"type": "text", "text": "..."}] -- this flattens either shape into
    a single string so downstream code doesn't have to care which
    provider produced it.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content) if content else ""


def _parse_read_page_source(tool_message: ToolMessage) -> Source | None:
    """Pull a structured Source out of a read_page ToolMessage, which is
    formatted as 'TITLE: ...\\nURL: ...\\n\\n<body>'. Returns None for
    non-read_page messages or ones that don't match the expected shape."""
    if tool_message.name != "read_page":
        return None
    content = _extract_text(tool_message.content)
    title_match = re.match(r"TITLE: (.*)\nURL: (.*)\n", content)
    if not title_match:
        return None
    title, url = title_match.group(1).strip(), title_match.group(2).strip()
    if not url or "[fetch failed]" in title:
        return None
    snippet = content[title_match.end():].strip().replace("\n", " ")[:200]
    return Source(title=title, url=url, snippet=snippet)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def planner_node(state: AgentState) -> dict:
    """Decide the next action: call a tool, or declare readiness to answer."""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + messages

    response = planner_llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
    }


def researcher_node(state: AgentState) -> dict:
    """Execute whatever tool calls the planner just emitted (the 'act' step),
    and extract structured Source records from any pages read."""
    tool_node = ToolNode(TOOLS)
    result = tool_node.invoke(state)

    notes = []
    new_sources = []
    for msg in result["messages"]:
        content = _extract_text(getattr(msg, "content", ""))
        if content:
            notes.append(content[:2000])
        if isinstance(msg, ToolMessage):
            source = _parse_read_page_source(msg)
            if source:
                new_sources.append(source)

    return {
        "messages": result["messages"],
        "research_notes": notes,
        "sources": new_sources,
    }


def synthesizer_node(state: AgentState) -> dict:
    """Compose the final answer from the accumulated research notes.

    Deliberately builds a fresh, clean prompt instead of forwarding the
    raw message history: passing ToolMessages to a model with no tools
    bound has caused empty responses on some providers (observed with
    Gemini). research_notes already holds everything the synthesizer
    needs, in plain text.
    """
    original_question = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "",
    )
    notes_block = "\n\n---\n\n".join(state.get("research_notes", [])) or "(no research notes gathered)"

    prompt = (
        f"Original question: {original_question}\n\n"
        f"Research notes gathered:\n\n{notes_block}"
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    response = synthesizer_llm.invoke(messages)
    final_text = _extract_text(response.content)

    if not final_text.strip():
        final_text = (
            "I gathered research but wasn't able to generate a final "
            "write-up. Try re-running the question, or check the Debug "
            "panel for what was found."
        )

    return {
        "messages": [response],
        "final_answer": final_text,
    }


# ---------------------------------------------------------------------------
# Conditional edge: "decide next step"
# ---------------------------------------------------------------------------
def route_after_planner(state: AgentState) -> str:
    last_message = state["messages"][-1]

    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "synthesizer"

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "researcher"

    return "synthesizer"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "researcher": "researcher",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_edge("researcher", "planner")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def run_agent(question: str) -> dict:
    """Convenience entrypoint used by app.py and evaluation scripts."""
    app = build_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "research_notes": [],
        "sources": [],
        "final_answer": None,
        "iterations": 0,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    result = run_agent(
        "What is LangGraph and how is it different from LangChain's AgentExecutor?"
    )
    print("\n=== FINAL ANSWER ===\n")
    print(result["final_answer"])
    print("\n=== SOURCES ===\n")
    for s in result["sources"]:
        print(f"- {s.title}: {s.url}")