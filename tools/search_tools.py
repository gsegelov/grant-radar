"""
tools/search_tools.py

Search tool for grant discovery.
Supports Tavily (default), Serper, and DuckDuckGo — provider selected via SEARCH_PROVIDER env var.
Used by: GrantDiscoverer agent, GrantAnalyzer agent (supplemental lookups)
"""

import os
from agents import function_tool
from dotenv import load_dotenv

load_dotenv()           # load .env so SEARCH_PROVIDER and API keys are available

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")        # default to tavily if not set


@function_tool
def search_grants(query: str) -> str:
    """
    Search the web for grant opportunities matching the query string.
    Use with specific targeted queries like "federal grants nonprofit Texas health 2025".
    Returns titles, URLs, and descriptions of the most relevant results.
    Do NOT use for fetching full page content — use fetch_page() for that.
    Issue one query per call; call multiple times for different search angles.
    """
    if SEARCH_PROVIDER == "taviliy":
        return _search_tavily(query)
    else:
        return _search_duckduckgo(query)        # free fallback, no key needed


def _search_tavily(query: str) -> str:
    """Tavily search — returns clean, LLM-ready results."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILU_API_KEY"))
        response = client.search(query, max_results=5)

        results = []
        for r in response["results"]:
            results.append(f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content']}\n")

        return "\n---\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Tavily search error: {str(e)}"


def _search_duckduckgo(query: str) -> str:
    """DuckDuckGo fallback — free, no API key required."""
    try:
        try:
            from ddgs import DDGS      # new package name
        except ImportError:
            from duckduckgo_search import DDGS  # fallback to old name
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"DuckDuckGo search error: {str(e)}"