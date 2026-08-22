"""
Tool implementations for the web research agent.

web_search: hits DuckDuckGo (no API key needed) and returns
            {title, url, snippet} dicts. Retries once with a
            reformulated query if the first attempt errors or
            comes back empty, so a single bad query doesn't kill
            the whole research step.
read_page:  fetches a URL and returns cleaned, readable text,
            prefixed with a TITLE/URL header so the title survives
            the round-trip through the tool call for structured
            source tracking downstream.

Swap web_search's backend for Tavily/SerpAPI/Bing later if you need
higher-quality results — the tool's input/output contract
(str -> list[dict]) is what the graph depends on, not the backend.
"""

import re
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from ddgs import DDGS


def _run_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    with DDGS() as ddgs:
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in ddgs.text(query, max_results=max_results)
        ]


def _simplify_query(query: str) -> str:
    """Fallback reformulation: strip punctuation/quotes and trim to the
    first few words, on the theory that an overly specific or malformed
    query is why the first search came back empty."""
    stripped = re.sub(r'["\'?!:,]', "", query)
    words = stripped.split()
    return " ".join(words[:6]) if len(words) > 6 else stripped


@tool
def web_search(query: str) -> List[Dict[str, str]]:
    """Search the web and return a list of results.

    Each result has: title, url, snippet.
    Use this to find candidate pages before reading them in full.
    If a search returns nothing useful, try again with a simpler or
    differently-phrased query rather than giving up.
    """
    try:
        results = _run_search(query)
    except Exception:
        results = []

    if not results:
        # Graceful failure handling: retry once with a simplified query
        # before surfacing an empty/error result to the planner.
        fallback_query = _simplify_query(query)
        if fallback_query and fallback_query != query:
            try:
                results = _run_search(fallback_query)
            except Exception as e:
                return [
                    {
                        "title": "SEARCH_ERROR",
                        "url": "",
                        "snippet": (
                            f"Both '{query}' and fallback '{fallback_query}' "
                            f"failed: {e}. Try a substantially different query."
                        ),
                    }
                ]

    if not results:
        return [
            {
                "title": "NO_RESULTS",
                "url": "",
                "snippet": (
                    f"No results for '{query}' (also tried a simplified "
                    "version). Try a broader or differently-phrased query."
                ),
            }
        ]

    return results


@tool
def read_page(url: str) -> str:
    """Fetch a URL and return its cleaned, readable text content.

    Use this after web_search to read the full content of a
    promising result before citing or summarizing it. Output is
    prefixed with "TITLE: ..." and "URL: ..." lines.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"TITLE: [fetch failed]\nURL: {url}\n\n[ERROR fetching {url}: {e}]"

    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else url

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    max_chars = 8000
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n[...truncated...]"

    return f"TITLE: {page_title}\nURL: {url}\n\n{cleaned}"


TOOLS = [web_search, read_page]