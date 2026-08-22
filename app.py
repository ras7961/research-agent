"""
Streamlit front-end for the research agent.

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, ToolMessage

from agent.graph import build_graph
from agent.state import AgentState

load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered")
st.title("🔎 Research Agent")
st.caption("plan → act (web_search / read_page) → observe → decide, via LangGraph")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

question = st.text_input(
    "Ask a research question",
    placeholder="e.g. What is LangGraph and how does it differ from AgentExecutor?",
)

run = st.button("Run agent", type="primary", disabled=not question)

if run and question:
    initial_state: AgentState = {
        "messages": [{"role": "user", "content": question}],
        "research_notes": [],
        "sources": [],
        "final_answer": None,
        "iterations": 0,
    }

    st.subheader("Steps")
    trace_container = st.container()

    with st.spinner("Researching..."):
        final_state = None
        for step in st.session_state.graph.stream(initial_state, stream_mode="values"):
            final_state = step
            last_msg = step["messages"][-1]

            if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                for call in last_msg.tool_calls:
                    with trace_container:
                        if call["name"] == "web_search":
                            st.markdown(f"🔍 **Searching:** `{call['args'].get('query', '')}`")
                        elif call["name"] == "read_page":
                            st.markdown(f"📖 **Reading:** {call['args'].get('url', '')}")
                        else:
                            st.markdown(f"🛠️ **Calling `{call['name']}`** — `{call['args']}`")

            elif isinstance(last_msg, ToolMessage):
                with trace_container:
                    with st.expander(f"↳ Result from `{last_msg.name}`"):
                        st.text(str(last_msg.content)[:2000])

    st.divider()

    st.subheader("Final Answer")
    if final_state and final_state.get("final_answer"):
        st.markdown(final_state["final_answer"])
    else:
        st.warning("The agent didn't produce a final answer.")

    sources = final_state.get("sources", []) if final_state else []
    if sources:
        st.subheader("Sources")
        for s in sources:
            st.markdown(f"- [{s.title}]({s.url})")

    with st.expander(f"Debug: {final_state.get('iterations', 0)} planner iterations"):
        st.json(
            [
                {"type": type(m).__name__, "content": str(getattr(m, "content", ""))[:500]}
                for m in final_state["messages"]
            ]
        )