"""
pipeline/discoverer.py

GrantDiscoverer agent — executes one search query and returns grant candidates.
Called with staggered start times in app.py to avoid Tavily rate limiting.

Each call returns a list of GrantCandidate objects. Results from all calls
are merged and deduplicated before Phase 4 begins.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (Agent, Runner, set_default_openai_client,
                    set_default_openai_api, set_tracing_disabled)
from models.schemas import GrantCandidate, OrgProfile
from pipeline.context import PipelineContext
from tools.search_tools import search_grants
from tools.parse_utils import parse_json_output

load_dotenv()

# ── GEMINI SETUP ──────────────────────────────────────────────────────────
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ── AGENT: GrantDiscoverer ────────────────────────────────────────────────
# One instance handles one query. Called N times in parallel.
# Uses Flash — search and extraction, not reasoning.

grant_discoverer = Agent(
    name="GrantDiscoverer",
    instructions="""You are a grant researcher. Your job is to find grant opportunities
    using the search_grants tool.

    Steps:
    1. Call search_grants() with the query string provided.
    2. Review the results and extract legitimate grant opportunities.
    3. Ignore results that are: news articles, blog posts, directories, or aggregator
       sites that don't link to a specific grant program.

    Return ONLY a JSON array with no markdown fences — one object per grant found:
    [
        {
            "name": "grant program name",
            "funder": "foundation or agency name",
            "url": "direct URL to the grant page",
            "description": "brief description of the grant",
            "source_query": "the exact query string you searched with"
        }
    ]

    Return an empty array [] if no legitimate grants are found.""",
    tools=[search_grants],
    model="gemini-2.5-flash"
)


def parse_grant_candidates(raw_output: str, query: str) -> list[GrantCandidate]:
    """
    Parse JSON array output into a list of GrantCandidate objects.
    Injects source_query from the function parameter if the agent omitted it.
    Skips individual items that fail validation rather than dropping the whole list.
    """
    try:
        data = parse_json_output(raw_output)
        if not isinstance(data, list):
            return []
        candidates = []
        for item in data:
            item.setdefault("source_query", query)  # agent often omits this required field
            try:
                candidates.append(GrantCandidate(**item))
            except Exception:
                continue                            # skip bad items, keep good ones
        return candidates
    except Exception:
        return []


def deduplicate_candidates(all_candidates: list[GrantCandidate]) -> list[GrantCandidate]:
    """Remove duplicate grants by URL from merged results."""
    seen_urls = set()
    unique = []
    for candidate in all_candidates:
        if candidate.url not in seen_urls:
            seen_urls.add(candidate.url)
            unique.append(candidate)
    return unique


async def run_discovery_parallel(
    search_queries: list[str],
    ctx: PipelineContext,
    on_progress=None
) -> list[GrantCandidate]:
    """
    Run GrantDiscoverer with staggered start times to avoid Tavily rate limiting.
    Returns a deduplicated list of all candidates found.
    Called by app.py after Phase 2 completes.
    """
    all_candidates = []
    tasks = []

    # stagger task starts by 0.5s to prevent rate limiting on Tavily free tier
    for i, query in enumerate(search_queries):
        await asyncio.sleep(0.5 * i)
        task = asyncio.create_task(
            Runner.run(grant_discoverer, query, context=ctx)
        )
        tasks.append((task, query))

    # wait for all tasks to complete and collect results
    for idx, (task, query) in enumerate(tasks):
        try:
            result = await task
            candidates = parse_grant_candidates(result.final_output, query)
            all_candidates.extend(candidates)
        except Exception as e:
            print(f"Discovery query failed: {e}")
        if on_progress:
            on_progress(idx + 1, len(tasks))

    return deduplicate_candidates(all_candidates)