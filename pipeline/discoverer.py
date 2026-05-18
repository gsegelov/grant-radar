"""
pipeline/discoverer.py

GrantDiscoverer agent — executes one search query and returns grant candidates.
Called in parallel via asyncio.gather() in app.py, once per search query.

Each call returns a list of GrantCandidate objects. Results from all parallel
calls are merged and deduplicated before Phase 4 begins.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import GrantCandidate, OrgProfile
from pipeline.context import PipelineContext
from tools.search_tools import search_grants
from tools.parse_utils import parse_json_output

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


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
    model=LitellmModel(model="gemini/gemini-2.5-flash")
)


def parse_grant_candidates(raw_output: str, query: str) -> list[GrantCandidate]:
    """
    Parse GrantDiscoverer's JSON array output into a list of GrantCandidate objects.
    Falls back to empty list if parsing fails — one bad query shouldn't stop discovery.
    """
    try:
        data = parse_json_output(raw_output)
        if not isinstance(data, list):
            return []   # agent returned an object instead of array — skip it
        return [GrantCandidate(**item) for item in data]
    except Exception:
        return []       # parse failure on one query — return empty, continue pipeline


def deduplicate_candidates(all_candidates: list[GrantCandidate]) -> list[GrantCandidate]:
    """
    Remove duplicate grants from merged parallel results.
    Deduplication is by URL — same grant page = same grant.
    """
    seen_urls = set()
    unique = []
    for candidate in all_candidates:
        if candidate.url not in seen_urls:
            seen_urls.add(candidate.url)
            unique.append(candidate)
    return unique


async def run_discovery_parallel(
    search_queries: list[str],
    ctx: PipelineContext
) -> list[GrantCandidate]:
    """
    Run GrantDiscoverer in parallel across all search queries.
    Returns a deduplicated list of all candidates found.
    Called by app.py after Phase 2 completes.
    """
    # create one coroutine per query — none start executing yet
    tasks = [
        Runner.run(grant_discoverer, query, context=ctx)
        for query in search_queries
    ]

    # fire all tasks simultaneously — wait for all to finish
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # collect all candidates from all queries
    all_candidates = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Query {i} failed: {result}")   # log failure, don't crash
            continue
        candidates = parse_grant_candidates(result.final_output, search_queries[i])
        all_candidates.extend(candidates)

    return deduplicate_candidates(all_candidates)