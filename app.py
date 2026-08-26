"""
Streamlit front-end for the research agent -- a persistent, multi-
conversation chat, not a one-shot Q&A form.

Long-term memory has two layers, both on disk (see README for details):
- agent/graph.py's SQLite checkpointer holds what the AGENT needs to
  reason with (raw messages, research notes, sources), keyed by
  thread_id, and survives app restarts.
- agent/conversations.py holds what the UI needs to display (titles,
  timestamps, a clean turn-by-turn transcript), also on disk, so a
  page reload always shows exactly what you saw before -- nothing
  here lives only in ephemeral browser session state.

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import build_graph, generate_title, seed_research_from_thread
from agent.state import AgentState
from agent.conversations import (
    create_conversation,
    add_turn,
    get_turns,
    list_conversations,
    find_related_conversation,
)

load_dotenv()


st.set_page_config(page_title="Research Agent", page_icon="🕵️", layout="centered")


@st.cache_resource
def get_graph():
    """Build the compiled graph once per server process (not per browser
    session) -- it wraps a persistent SQLite connection, so it should be
    a true singleton rather than being recreated on every rerun."""
    return build_graph()


graph = get_graph()

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None  # None = not-yet-started new conversation

# --- sidebar: conversation list, grouped new vs. past --------------------
with st.sidebar:
    st.subheader("🕵️ Research Agent")
    st.caption("Conversations are saved to disk and persist across restarts.")

    if st.button("➕ New conversation", use_container_width=True):
        st.session_state.active_thread_id = None
        st.rerun()

    st.divider()
    st.caption("Previous conversations")

    past_conversations = list_conversations()
    if not past_conversations:
        st.caption("_None yet — ask something to start._")
    else:
        for convo in past_conversations:
            is_active = convo["thread_id"] == st.session_state.active_thread_id
            label = f"**{convo['title']}**" if is_active else convo["title"]
            if st.button(label, key=f"convo_{convo['thread_id']}", use_container_width=True):
                st.session_state.active_thread_id = convo["thread_id"]
                st.rerun()

    if st.session_state.active_thread_id:
        session_sources = []
        seen_urls = set()
        for turn in get_turns(st.session_state.active_thread_id):
            for s in turn.get("sources", []):
                if s["url"] not in seen_urls:
                    session_sources.append(s)
                    seen_urls.add(s["url"])
        if session_sources:
            st.divider()
            st.caption("Sources in this conversation")
            for s in session_sources:
                st.markdown(f"- [{s['title']}]({s['url']})")

# --- main area -------------------------------------------------------------
st.title(" 🕵️ Research Agent")

if st.session_state.active_thread_id is None:
    st.caption("Ask a research question to start a conversation.")
else:
    active_title = next(
        (c["title"] for c in list_conversations() if c["thread_id"] == st.session_state.active_thread_id),
        "Conversation",
    )
    st.caption(f"Continuing: **{active_title}**")

# Replay the ACTIVE conversation from disk every rerun -- this is the
# source of truth for what's displayed, so a page reload or app restart
# always reconstructs the same visible history instead of relying on
# session state that can reset.
if st.session_state.active_thread_id:
    for turn in get_turns(st.session_state.active_thread_id):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for s in turn["sources"]:
                        st.markdown(f"- [{s['title']}]({s['url']})")

# --- handle a new message ---------------------------------------------------
question = st.chat_input("Ask a research question or a follow-up...")

if question:
    is_new_conversation = st.session_state.active_thread_id is None
    related_note = None

    if is_new_conversation:
        # Check whether this "new" question is actually a continuation of
        # a topic already covered in a separate, earlier conversation --
        # if so, link back to it by seeding this thread's research.
        related = find_related_conversation(question)
        title = generate_title(question)
        thread_id = create_conversation(title)
        st.session_state.active_thread_id = thread_id
        if related:
            seeded = seed_research_from_thread(graph, thread_id, related["thread_id"])
            if seeded:
                related_note = (
                    f"This looks related to your earlier conversation "
                    f"*\u201c{related['title']}\u201d* — I'll build on what was already found there."
                )
    else:
        thread_id = st.session_state.active_thread_id

    add_turn(thread_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"thread_id": thread_id}}
    turn_input: AgentState = {
        "messages": [{"role": "user", "content": question}],
        "research_notes": [],
        "sources": [],
        "final_answer": None,
        "iterations": 0,
    }

    # snapshot sources BEFORE this turn runs, so we can show only the
    # NEW ones found this turn afterward (sources accumulate across the
    # whole thread via the state reducer).
    prior_snapshot = graph.get_state(config)
    urls_before_this_turn = {
        s["url"] for s in (prior_snapshot.values.get("sources", []) if prior_snapshot.values else [])
    }

    with st.chat_message("assistant"):
        if related_note:
            st.info(related_note)

        step_area = st.container()
        final_state = None

        with st.spinner("Researching..."):
            for step in graph.stream(turn_input, config=config, stream_mode="values"):
                final_state = step
                last_msg = step["messages"][-1]

                if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                    for call in last_msg.tool_calls:
                        with step_area:
                            if call["name"] == "web_search":
                                st.markdown(f"🕵️ **Searching:** `{call['args'].get('query', '')}`")
                            elif call["name"] == "read_page":
                                st.markdown(f"📖 **Reading:** {call['args'].get('url', '')}")
                            else:
                                st.markdown(f"🛠️ **Calling `{call['name']}`** — `{call['args']}`")

                elif isinstance(last_msg, ToolMessage):
                    with step_area:
                        with st.expander(f"↳ Result from `{last_msg.name}`"):
                            st.text(str(last_msg.content)[:2000])

        answer = final_state.get("final_answer") if final_state else None
        if not answer:
            answer = "I gathered research but wasn't able to generate a final write-up. Try asking again."
            st.warning(answer)
        else:
            st.markdown(answer)

        all_sources_so_far = final_state.get("sources", []) if final_state else []
        turn_sources = [s for s in all_sources_so_far if s["url"] not in urls_before_this_turn]
        if turn_sources:
            with st.expander("Sources"):
                for s in turn_sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

    add_turn(thread_id, "assistant", answer, sources=turn_sources)

    with st.expander(f"Debug: {final_state.get('iterations', 0)} planner iterations this turn"):
        st.json(
            [
                {"type": type(m).__name__, "content": str(getattr(m, "content", ""))[:500]}
                for m in final_state["messages"]
            ]
        )

    st.rerun()  # redraw from disk so the sidebar's new/renamed conversation shows immediately