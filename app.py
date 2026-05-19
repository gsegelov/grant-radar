"""
app.py

GrantRadar pipeline orchestrator and Gradio UI.
Built in two passes:
  Pass 1 (this file): pipeline logic — phase sequencing, HITL gate, output assembly
  Pass 2: Gradio UI wired to the pipeline functions

Python controls all phase transitions. No agent decides what runs next.
Pipeline sequence:
  Phase 1 (scrape) → Phase 2 (profile) → Phase 3 (discover) →
  Phase 4 (analyze) → Phase 5 (score) → HITL gate →
  Phase 6 (brief) → Phase 7 (draft) → Phase 8 (compliance) → output assembly
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (set_default_openai_client, set_default_openai_api,
                    set_tracing_disabled, Runner)

# pipeline phases
from pipeline.context import PipelineContext
from pipeline.scraper import web_scraper, parse_scraped_site
from pipeline.profiler import org_profiler, build_profiler_input, parse_org_profile
from pipeline.discoverer import run_discovery_parallel
from pipeline.analyzer import run_analysis_parallel
from pipeline.scorer import run_scoring_parallel
from pipeline.brief_writer import run_brief_writing
from pipeline.drafter import run_drafting
from pipeline.compliance import run_compliance_checks

# output assembly
from tools.export_tools import build_csv, build_docx

load_dotenv()

# ── GEMINI SETUP ──────────────────────────────────────────────────────────
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ── PIPELINE: Phases 1–5 (scrape through score) ───────────────────────────
# Returns scored grants sorted by score descending.
# Called first — before the HITL gate.

async def run_pipeline_phase1_to_5(
    org_url: str,
    org_type: str,
    user_context: str,
    max_grants: int,
    research_depth: str,
    progress_callback=None       # optional function(message) for Gradio status updates
) -> tuple[PipelineContext, list]:
    """
    Run phases 1–5 of the pipeline and return context + scored grants.
    Stops before the HITL gate so the user can review scores before drafting begins.
    Returns (ctx, scored_grants) — both needed by the HITL handler in app.py.
    """

    def log(msg: str):
        """Send progress update to UI if callback provided, else print."""
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # create shared context — passed to every agent run
    ctx = PipelineContext(
        org_url=org_url,
        org_type=org_type,
        user_context=user_context,
        max_grants_to_analyze=max_grants,
        research_depth=research_depth
    )

    # ── PHASE 1: Scrape ───────────────────────────────────────────────────
    log("Phase 1/5 — Scraping organization website...")
    scrape_result = await Runner.run(web_scraper, org_url, context=ctx)
    site = parse_scraped_site(scrape_result.final_output)
    log(f"  Scraped {len(site.subpage_texts) + 1} pages")

    # ── PHASE 2: Profile ──────────────────────────────────────────────────
    log("Phase 2/5 — Building organization profile...")
    profile_result = await Runner.run(
        org_profiler,
        build_profiler_input(site, user_context),
        context=ctx
    )
    ctx.org_profile = parse_org_profile(profile_result.final_output)
    log(f"  Profile built — {len(ctx.org_profile.search_queries)} search queries generated")

    # ── PHASE 3: Discover ─────────────────────────────────────────────────
    log("Phase 3/5 — Discovering grant opportunities...")
    ctx.all_candidates = await run_discovery_parallel(
        ctx.org_profile.search_queries, ctx
    )
    log(f"  Found {len(ctx.all_candidates)} unique candidates")

    # ── PHASE 4: Analyze ──────────────────────────────────────────────────
    log(f"Phase 4/5 — Analyzing top {max_grants} grants...")
    ctx.all_grant_data = await run_analysis_parallel(
        ctx.all_candidates, ctx, max_grants=max_grants
    )
    log(f"  Analyzed {len(ctx.all_grant_data)} grants")

    # ── PHASE 5: Score ────────────────────────────────────────────────────
    log("Phase 5/5 — Scoring grant fit...")
    scored_grants = await run_scoring_parallel(
        ctx.all_grant_data, ctx.org_profile, ctx
    )
    log(f"  Scored {len(scored_grants)} grants — top score: {scored_grants[0].score if scored_grants else 0}")

    return ctx, scored_grants


# ── PIPELINE: Phases 6–8 (brief through compliance) ──────────────────────
# Called after the HITL gate once the user has selected grants to draft.

async def run_pipeline_phase6_to_8(
    ctx: PipelineContext,
    selected_grants: list,
    progress_callback=None
) -> tuple[bytes, bytes]:
    """
    Run phases 6–8 and assemble final outputs.
    Receives the PipelineContext from phase 1–5 so all pipeline state is preserved.
    Returns (csv_bytes, docx_bytes) — Gradio serves these as downloads.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # store selected grants in context for downstream reference
    ctx.selected_grants = selected_grants

    # ── PHASE 6: Strategic Briefs ─────────────────────────────────────────
    log(f"Phase 6/8 — Writing strategic briefs for {len(selected_grants)} grants...")
    briefs = await run_brief_writing(selected_grants, ctx.org_profile, ctx)
    log(f"  {len(briefs)} briefs written")

    # ── PHASE 7: Proposal Drafts ──────────────────────────────────────────
    log("Phase 7/8 — Drafting proposals...")
    drafts = await run_drafting(selected_grants, briefs, ctx.org_profile, ctx)
    log(f"  {len(drafts)} drafts completed")

    # ── PHASE 8: Compliance Checks ────────────────────────────────────────
    log("Phase 8/8 — Running compliance checks...")

    # get grant data only for selected grants — compliance needs the requirements
    selected_names = {sg.grant.candidate.name for sg in selected_grants}
    selected_grant_data = [
        gd for gd in ctx.all_grant_data
        if gd.candidate.name in selected_names
    ]

    reports = await run_compliance_checks(drafts, selected_grant_data, ctx)
    log(f"  {len(reports)} compliance reports generated")

    # ── OUTPUT ASSEMBLY ───────────────────────────────────────────────────
    log("Assembling outputs...")

    # CSV covers all scored grants — full picture for the user
    csv_bytes = build_csv(ctx.scored_grants)

    # DOCX covers only selected grants — brief + draft + compliance per grant
    docx_bytes = build_docx(
        ctx.org_profile,
        ctx.scored_grants,
        briefs,
        drafts,
        reports
    )

    log("Done.")
    return csv_bytes, docx_bytes


# ── HITL HELPERS ──────────────────────────────────────────────────────────
# These functions sit between phase 1–5 and phase 6–8.
# format_scores_for_display() renders the scored grants as a readable table.
# parse_user_selection() converts the user's checkbox choices back into
# ScoredGrant objects that phase 6–8 can process.

def format_scores_for_display(scored_grants: list) -> list[list]:
    """
    Format scored grants as a list of rows for Gradio's DataFrame component.
    Each row = one grant. Columns match the HITL table headers defined in the UI.
    """
    rows = []
    for i, sg in enumerate(scored_grants):
        rows.append([
            i + 1,                              # rank
            sg.grant.candidate.name,            # grant name
            sg.grant.candidate.funder,          # funder
            sg.score,                           # fit score
            "Yes" if sg.recommended else "No",  # recommended
            "Yes" if sg.disqualified else "No", # disqualified
            f"${sg.grant.award_min:,}–${sg.grant.award_max:,}",  # award range
            sg.grant.deadline,                  # deadline
            sg.rationale[:120] + "..." if len(sg.rationale) > 120 else sg.rationale
        ])
    return rows


def parse_user_selection(
    scored_grants: list,
    selected_indices: list[int]
) -> list:
    """
    Convert user-selected row indices into ScoredGrant objects.
    selected_indices = list of 0-based row positions the user checked.
    Returns the selected ScoredGrant objects in score order.
    Caps at 5 selections — more than 5 grants is too many to draft well.
    """
    selected = []
    for i in selected_indices:
        if 0 <= i < len(scored_grants):
            selected.append(scored_grants[i])

    if len(selected) > 5:
        print(f"User selected {len(selected)} grants — capping at 5")
        selected = selected[:5]

    return selected


# ── MAIN: end-to-end pipeline test ───────────────────────────────────────
# Temporary test runner — replaced by Gradio UI in Pass 2.
# Runs the full pipeline with hardcoded inputs and saves outputs to disk.

async def main():
    """
    TEMPORARY: Pipeline test runner with hardcoded inputs.
    Replaced by Gradio UI in Step 15.
    Run with: python app.py
    """
    print("=== GrantRadar Pipeline Test ===\n")

    # ── PHASE 1–5 ─────────────────────────────────────────────────────────
    ctx, scored_grants = await run_pipeline_phase1_to_5(
        org_url="https://www.habitat.org",
        org_type="nonprofit",
        user_context="Focus on housing construction and community development grants",
        max_grants=3,               # keep low for testing
        research_depth="standard"
    )

    if not scored_grants:
        print("No grants found — check search provider and API keys")
        return

    # ── HITL GATE ─────────────────────────────────────────────────────────
    print("\n=== SCORED GRANTS ===")
    rows = format_scores_for_display(scored_grants)
    for row in rows:
        print(f"  [{row[0]}] {row[1]} | Score: {row[3]} | Recommended: {row[4]}")

    # auto-select recommended grants for testing — in UI the user picks manually
    selected_indices = [
        i for i, sg in enumerate(scored_grants) if sg.recommended
    ]
    if not selected_indices:
        selected_indices = [0]      # fallback — take top grant if none recommended

    selected_grants = parse_user_selection(scored_grants, selected_indices)
    print(f"\nAuto-selected {len(selected_grants)} grant(s) for drafting:")
    for sg in selected_grants:
        print(f"  - {sg.grant.candidate.name}")

    # ── PHASE 6–8 ─────────────────────────────────────────────────────────
    print("\nRunning phases 6–8...")
    csv_bytes, docx_bytes = await run_pipeline_phase6_to_8(ctx, selected_grants)

    # save outputs to disk for inspection
    with open("test_output.csv", "wb") as f:
        f.write(csv_bytes)
    with open("test_output.docx", "wb") as f:
        f.write(docx_bytes)

    print("\n=== DONE ===")
    print(f"CSV saved: test_output.csv ({len(csv_bytes)} bytes)")
    print(f"DOCX saved: test_output.docx ({len(docx_bytes)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())