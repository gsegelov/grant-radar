"""
diagnose.py — Time each pipeline phase to find browser-timeout culprits.
Usage: python diagnose.py
"""

import asyncio
import time
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled

load_dotenv()

# Gemini setup (must happen before importing pipeline modules that also set it)
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

from agents import Runner
from pipeline.context import PipelineContext
from pipeline.scraper import web_scraper, parse_scraped_site
from pipeline.profiler import org_profiler, build_profiler_input, parse_org_profile
from pipeline.discoverer import run_discovery_parallel
from pipeline.analyzer import run_analysis_parallel
from pipeline.scorer import run_scoring_parallel

TARGET_URL = "https://www.cobblehilllifecare.org/"
ORG_TYPE = "nonprofit"
USER_CONTEXT = ""
MAX_GRANTS = 5
RESEARCH_DEPTH = "standard"


async def main():
    ctx = PipelineContext(
        org_url=TARGET_URL,
        org_type=ORG_TYPE,
        user_context=USER_CONTEXT,
        max_grants_to_analyze=MAX_GRANTS,
        research_depth=RESEARCH_DEPTH,
    )

    timings = {}

    # Phase 1: Scrape
    print("Phase 1 — Scraping...")
    t0 = time.monotonic()
    scrape_result = await Runner.run(web_scraper, TARGET_URL, context=ctx)
    site = parse_scraped_site(scrape_result.final_output)
    elapsed = time.monotonic() - t0
    timings["Phase 1 (scrape)"] = elapsed
    print(f"  Done: {len(site.subpage_texts) + 1} pages in {elapsed:.1f}s")

    # Phase 2: Profile
    print("Phase 2 — Profiling...")
    t0 = time.monotonic()
    profile_result = await Runner.run(
        org_profiler, build_profiler_input(site, USER_CONTEXT), context=ctx
    )
    ctx.org_profile = parse_org_profile(profile_result.final_output)
    elapsed = time.monotonic() - t0
    timings["Phase 2 (profile)"] = elapsed
    print(f"  Done: {len(ctx.org_profile.search_queries)} queries in {elapsed:.1f}s")

    # Phase 3: Discover
    print("Phase 3 — Discovering...")
    t0 = time.monotonic()
    ctx.all_candidates = await run_discovery_parallel(
        ctx.org_profile.search_queries, ctx,
        on_progress=lambda done, total: print(f"    [progress] Discovery: {done}/{total}")
    )
    elapsed = time.monotonic() - t0
    timings["Phase 3 (discover)"] = elapsed
    print(f"  Done: {len(ctx.all_candidates)} candidates in {elapsed:.1f}s")

    # Phase 4: Analyze
    print(f"Phase 4 — Analyzing top {MAX_GRANTS}...")
    t0 = time.monotonic()
    ctx.all_grant_data = await run_analysis_parallel(
        ctx.all_candidates, ctx, max_grants=MAX_GRANTS,
        on_progress=lambda done, total: print(f"    [progress] Analysis: {done}/{total}")
    )
    elapsed = time.monotonic() - t0
    timings["Phase 4 (analyze)"] = elapsed
    print(f"  Done: {len(ctx.all_grant_data)} grants in {elapsed:.1f}s")

    # Phase 5: Score
    print("Phase 5 — Scoring...")
    t0 = time.monotonic()
    scored_grants = await run_scoring_parallel(
        ctx.all_grant_data, ctx.org_profile, ctx,
        on_progress=lambda done, total: print(f"    [progress] Scoring: {done}/{total}")
    )
    elapsed = time.monotonic() - t0
    timings["Phase 5 (score)"] = elapsed
    print(f"  Done: {len(scored_grants)} scored in {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 50)
    print("PHASE TIMING SUMMARY")
    print("=" * 50)
    total = 0
    for phase, secs in timings.items():
        flag = " *** OVER 30s ***" if secs > 30 else ""
        print(f"  {phase:25s} {secs:6.1f}s{flag}")
        total += secs
    print(f"  {'TOTAL':25s} {total:6.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
