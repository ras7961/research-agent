![alt text](image-1.png)

# Research Agent

A tool-calling web research agent built with LangGraph. The agent
follows a **plan → act → observe → decide** loop: it searches the web,
reads pages it finds useful, and decides for itself when it has enough
information to answer — there's no hardcoded number of search steps.

## How it works

```
START -> planner --(tool calls?)--> researcher -> planner   (loop)
                 \--(no tool calls, or max iters)--> synthesizer -> END
```

- **`planner`** — an LLM bound to `web_search` and `read_page`. Each
  turn it decides whether it needs more information (and calls a tool)
  or is ready to answer (replies with plain text starting `READY:`).
  This is the core agentic decision — it's made by the model, not a
  fixed loop count.
- **`researcher`** — executes whatever tool calls the planner just
  requested, feeds the results back in as observations, and extracts
  a structured `Source(title, url, snippet)` record for every page
  actually read (deduplicated by URL).
- **`synthesizer`** — once the planner is satisfied (or a
  `MAX_ITERATIONS` guardrail kicks in), a stronger LLM call reads the
  whole research trail and writes a concise, bullet-first final
  answer, grounded only in what was actually retrieved.

## Skills this demonstrates

- **Tool / function calling** — `web_search` and `read_page` bound to
  the planner LLM via `bind_tools`.
- **Stateful agent with loops** — `AgentState` persists across the
  planner ⇄ researcher loop; `iterations` + `MAX_ITERATIONS` bound it.
- **Multi-step reasoning** — the planner searches, reads, and decides
  whether to search again before answering.
- **Grounding in real data** — the synthesizer is explicitly told to
  answer only from retrieved research notes, not prior knowledge.
- **Structured outputs** — `Source` is a Pydantic model; `sources` is
  a typed, deduplicated list in state, rendered as clickable links,
  independent of the free-text final answer.
- **Evaluation mindset** — `evaluation/test_questions.py` checks
  whether tool use matched expectation per question (e.g. the agent
  should NOT search for pure arithmetic).

### Standout details

- **Visible intermediate steps** — the Streamlit UI shows each search
  query and each URL read live, with tool output in an expander, not
  just the final answer.
- **Clickable Sources section** — rendered separately from the answer
  text, from the structured `sources` list, not parsed out of prose.
- **Graceful failure handling** — `web_search` retries once with a
  simplified/reformulated query if the first attempt errors or comes
  back empty, and the planner is instructed not to retry a dead query
  verbatim.
- **Concise, structured answers** — the synthesizer prompt requires a
  1-2 sentence summary up front, then bullets — no dense paragraphs.
- **Cost control** — planning (called once per loop iteration) uses a
  cheaper/faster model (`PLANNER_MODEL`, default `claude-haiku-4-5`);
  synthesis (called once per run, and what the user actually reads)
  uses a stronger model (`SYNTHESIZER_MODEL`, default `claude-sonnet-4-6`).
  Override either via `.env`.

## Project structure

```
research-agent/
├── app.py                 # Streamlit UI
├── agent/
│   ├── graph.py            # LangGraph StateGraph definition
│   ├── tools.py             # web_search + read_page tools
│   ├── state.py              # AgentState schema
│   └── prompts.py             # planner / synthesizer system prompts
├── evaluation/
│   └── test_questions.py       # a handful of test cases
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your ANTHROPIC_API_KEY
```

## Run

**Streamlit UI:**

```bash
streamlit run app.py
```

**CLI (single question):**

```bash
python -m agent.graph
```

**Evaluation suite:**

```bash
python -m evaluation.test_questions
```

## Notes / things to tune

- `web_search` uses DuckDuckGo (`ddgs`) — free, no API key. Swap it
  for Tavily or SerpAPI in `agent/tools.py` if you want higher-quality
  or more reliable results; the function signature
  (`str -> list[{title, url, snippet}]`) is all the graph depends on.
- `read_page` truncates page text to 8000 characters to keep context
  usage bounded — raise/lower this in `agent/tools.py` if needed.
- `MAX_ITERATIONS` (default 6, set via `.env`) is a hard guardrail
  against the planner looping forever; tune it if you see the agent
  getting cut off mid-research on multi-hop questions.
- Planner prompt discourages re-running the same search query — worth
  strengthening into actual dedup logic if you see repeats in practice.
- `PLANNER_MODEL` / `SYNTHESIZER_MODEL` (both in `.env`) control the
  cost/quality tradeoff — drop `SYNTHESIZER_MODEL` to a cheaper model
  too if you're optimizing for cost over answer quality.
- Source extraction in `agent/graph.py::_parse_read_page_source` relies
  on `read_page`'s `TITLE: ...\nURL: ...\n` output prefix — if you
  change that format in `agent/tools.py`, update the parser too.
