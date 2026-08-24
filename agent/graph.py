r"""
Web research agent — plan → act (tools) → observe → decide loop,
built as a LangGraph StateGraph, with persistent multi-turn memory.

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
               accumulated research notes plus prior conversation
               turns (NOT the raw message history) and writes a
               long, detailed final answer. It deliberately does not
               see raw ToolMessages -- some providers (Gemini
               included) can return an empty response when a
               tools-unbound model is handed function-response
               messages it doesn't recognize.

Multi-turn memory: the graph is compiled with a checkpointer keyed by
a `thread_id`. Every call to run_agent() with the same thread_id
resumes from where the conversation left off -- messages, research
notes, and sources all accumulate across turns via each field's
reducer, instead of starting fresh each time. iterations and
final_answer are explicitly reset at the start of every turn since
they're per-turn concerns (the planner-loop guardrail, and the
answer to whichever question was just asked).

Cost / rate-limit control: planning happens many times per run (one
call per loop iteration), which is exactly the pattern that burns
through a hosted API's rate limit fastest -- so the planner runs on
Groq (PLANNER_MODEL), which has a generous free-tier rate limit and
very fast inference well suited to a high-frequency step like this.
Synthesis happens once per turn and is what the user actually reads,
so it stays on a stronger model (SYNTHESIZER_MODEL, Gemini). Override
either via env vars.

A max_iterations guardrail forces a move to synthesizer even if the
planner keeps wanting to search, so the loop can't run forever.
"""

import os
import re
import sqlite3
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import AgentState, Source
from agent.tools import TOOLS
from agent.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT, TITLE_PROMPT

from dotenv import load_dotenv
load_dotenv()

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "6"))

# Hybrid setup: the planner runs multiple times per question (once per
# loop iteration), so it's the part that burns through a rate limit
# fastest -- it runs on Groq instead, which is hosted (no local server
# to run) but has a generous free-tier rate limit and very fast
# inference, well suited to the high-frequency planning step. The
# synthesizer runs once per turn and is what the user actually reads,
# so it stays on the stronger Gemini model.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen/qwen3.6-27b")
SYNTHESIZER_MODEL = os.getenv("SYNTHESIZER_MODEL", "models/gemini-3.6-flash")

planner_llm = ChatGroq(model=PLANNER_MODEL, temperature=0)
planner_llm_with_tools = planner_llm.bind_tools(TOOLS)

# max_output_tokens raised so long, detailed synthesized answers don't
# get cut off mid-explanation.
synthesizer_llm = ChatGoogleGenerativeAI(
    model=SYNTHESIZER_MODEL, temperature=0.3, max_output_tokens=4096
)

# Long-term memory: a SQLite-backed checkpointer, so the agent's
# reasoning state (messages, research notes, sources) survives app
# restarts, not just the lifetime of one Streamlit process. One
# connection is opened once per process and reused -- check_same_thread
# is disabled because Streamlit can touch it from more than one thread.
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
_CHECKPOINT_DB_PATH = os.path.join(DATA_DIR, "checkpoints.sqlite")
_conn = sqlite3.connect(_CHECKPOINT_DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)
_checkpointer.setup()


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
    return {"title": title, "url": url, "snippet": snippet}


def _build_conversation_history(messages, latest_question: str) -> str:
    """Reconstruct a plain-text transcript of PRIOR turns (not the current
    question) from the accumulated message list, for the synthesizer to
    use as context on follow-up questions. Skips ToolMessages entirely
    and skips the planner's terse "READY: ..." notes, keeping only real
    user questions and real synthesizer answers from earlier turns."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            text = _extract_text(m.content)
            if text and text != latest_question:
                lines.append(f"User asked: {text}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            text = _extract_text(m.content)
            if text and not text.startswith("READY:"):
                lines.append(f"Assistant answered: {text}")
    return "\n\n".join(lines)


def generate_title(question: str) -> str:
    """One-off, cheap call to produce a short conversation title from the
    first question of a new conversation. Uses the fast planner model
    directly (no tools bound), called once per new conversation -- not
    once per turn like the planner loop itself."""
    try:
        response = planner_llm.invoke(
            [SystemMessage(content=TITLE_PROMPT), HumanMessage(content=question)]
        )
        title = _extract_text(response.content).strip().strip('"').strip()
        return title[:60] if title else question[:50]
    except Exception:
        return question[:50]


def seed_research_from_thread(graph, new_thread_id: str, source_thread_id: str) -> bool:
    """Copy research_notes/sources from an existing, unrelated conversation
    thread into a brand-new one. Used when a "new" conversation turns out
    to be about a topic already researched in an earlier, separate
    conversation (see agent/conversations.py::find_related_conversation),
    so the agent doesn't have to re-search from scratch. Returns True if
    anything was actually seeded."""
    source_state = graph.get_state({"configurable": {"thread_id": source_thread_id}})
    if not source_state.values:
        return False
    notes = source_state.values.get("research_notes", [])
    sources = source_state.values.get("sources", [])
    if not notes and not sources:
        return False
    graph.update_state(
        {"configurable": {"thread_id": new_thread_id}},
        {"research_notes": notes, "sources": sources},
    )
    return True


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
    """Compose the final answer from the accumulated research notes plus
    prior-turn conversation context.

    Deliberately builds a fresh, clean prompt instead of forwarding the
    raw message history: passing ToolMessages to a model with no tools
    bound has caused empty responses on some providers (observed with
    Gemini). research_notes already holds everything the synthesizer
    needs, in plain text, and accumulates across the whole conversation.
    """
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    latest_question = _extract_text(human_msgs[-1].content) if human_msgs else ""

    history_block = _build_conversation_history(state["messages"], latest_question)
    notes_block = "\n\n---\n\n".join(state.get("research_notes", [])) or "(no research notes gathered)"

    prompt_parts = []
    if history_block:
        prompt_parts.append(f"Earlier in this conversation:\n\n{history_block}")
    prompt_parts.append(f"Current question: {latest_question}")
    prompt_parts.append(
        f"Research notes gathered so far in this conversation:\n\n{notes_block}"
    )
    prompt = "\n\n".join(prompt_parts)

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
    """Compile the graph with the shared in-process checkpointer, so any
    caller using the same thread_id resumes the same conversation."""
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

    return graph.compile(checkpointer=_checkpointer)


def run_agent(question: str, thread_id: str = "default") -> dict:
    """Run one turn of the conversation on the given thread_id.

    iterations and final_answer are reset per turn (they're per-question
    concerns); messages/research_notes/sources are merged onto whatever
    is already checkpointed for this thread_id via their reducers, so
    repeated calls with the same thread_id form a continuous, memory-
    carrying conversation.
    """
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    turn_input: AgentState = {
        "messages": [HumanMessage(content=question)],
        "research_notes": [],
        "sources": [],
        "final_answer": None,
        "iterations": 0,
    }
    return app.invoke(turn_input, config=config)


if __name__ == "__main__":
    result = run_agent(
        "What is LangGraph and how is it different from LangChain's AgentExecutor?"
    )
    print("\n=== FINAL ANSWER ===\n")
    print(result["final_answer"])
    print("\n=== SOURCES ===\n")
    for s in result["sources"]:
        print(f"- {s['title']}: {s['url']}")