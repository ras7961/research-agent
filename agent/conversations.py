"""
Long-term, disk-persisted conversation registry.

This is deliberately SEPARATE from LangGraph's own checkpointer
(agent/graph.py). The two serve different jobs:

- LangGraph's checkpointer persists what the AGENT needs to reason:
  raw messages, research notes, sources -- keyed by thread_id.
- This module persists what the UI needs to display: conversation
  titles, timestamps, and a clean turn-by-turn transcript -- so a
  page reload or app restart can always reconstruct exactly what the
  user saw, not just what the agent remembers internally.

Both are plain SQLite files on disk, so this is real long-term memory:
it survives closing the browser tab, restarting the Streamlit app, or
rebooting the machine -- nothing here lives only in st.session_state.
"""

import json
import os
import re
import sqlite3
import time
import uuid
from typing import List, Optional, TypedDict

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "conversations.sqlite")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
    "when", "where", "who", "do", "does", "did", "in", "on", "at", "to",
    "for", "of", "and", "or", "it", "its", "this", "that", "tell", "me",
    "about", "explain", "can", "you", "your", "i", "my", "please",
    "with", "from", "be", "have", "has", "will", "would", "could",
}


class Turn(TypedDict):
    role: str            # "user" or "assistant"
    content: str
    sources: list          # [{"title","url","snippet"}, ...] -- [] for user turns


class Conversation(TypedDict):
    thread_id: str
    title: str
    created_at: float


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return conn


def create_conversation(title: str) -> str:
    """Register a brand-new conversation and return its thread_id."""
    thread_id = str(uuid.uuid4())
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO conversations (thread_id, title, created_at) VALUES (?, ?, ?)",
            (thread_id, title, time.time()),
        )
    conn.close()
    return thread_id


def add_turn(thread_id: str, role: str, content: str, sources: Optional[list] = None) -> None:
    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO turns (thread_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(sources or []), time.time()),
        )
    conn.close()


def get_turns(thread_id: str) -> List[Turn]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, sources FROM turns WHERE thread_id = ? ORDER BY id ASC",
        (thread_id,),
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "sources": json.loads(r[2])} for r in rows]


def list_conversations() -> List[Conversation]:
    """Most recent first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT thread_id, title, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def _keywords(text: str) -> set:
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def find_related_conversation(
    question: str, exclude_thread_id: Optional[str] = None, threshold: float = 0.3
) -> Optional[Conversation]:
    """Lightweight keyword-overlap check against past conversation titles
    (plus each conversation's first user question) to catch when a
    "new" question is actually a continuation of a topic already
    covered in an earlier, separate conversation.

    Deliberately simple (Jaccard word overlap, no embeddings/vector DB)
    so the match is transparent and explainable to the user, not a
    black box -- appropriate for a project of this scope.
    """
    q_words = _keywords(question)
    if not q_words:
        return None

    best_match, best_score = None, 0.0
    for convo in list_conversations():
        if convo["thread_id"] == exclude_thread_id:
            continue
        turns = get_turns(convo["thread_id"])
        first_question = next((t["content"] for t in turns if t["role"] == "user"), "")
        candidate_words = _keywords(convo["title"]) | _keywords(first_question)
        if not candidate_words:
            continue
        # Overlap coefficient (intersection / smaller set size) rather
        # than Jaccard -- better suited to short texts like titles and
        # single questions, where Jaccard over-penalizes size mismatch.
        overlap = len(q_words & candidate_words) / min(len(q_words), len(candidate_words))
        if overlap > best_score:
            best_score, best_match = overlap, convo

    return best_match if best_score >= threshold else None