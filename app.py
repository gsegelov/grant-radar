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
    timeout=60.0    # HTTP-level timeout — httpx respects this
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ── PIPELINE: Phases 1–5 (scrape through score) ───────────────────────────

async def run_pipeline_phase1_to_5(
    org_url: str,
    org_type: str,
    user_context: str,
    max_grants: int,
    research_depth: str,
    progress_callback=None
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
        ctx.org_profile.search_queries, ctx,
        on_progress=lambda done, total: log(f"  Discovery: {done}/{total} queries complete")
    )
    log(f"  Found {len(ctx.all_candidates)} unique candidates")

    # ── PHASE 4: Analyze ──────────────────────────────────────────────────
    log(f"Phase 4/5 — Analyzing top {max_grants} grants...")
    ctx.all_grant_data = await run_analysis_parallel(
        ctx.all_candidates, ctx, max_grants=max_grants,
        on_progress=lambda done, total: log(f"  Analysis: {done}/{total} grants analyzed")
    )
    log(f"  Analyzed {len(ctx.all_grant_data)} grants")

    # ── PHASE 5: Score ────────────────────────────────────────────────────
    log("Phase 5/5 — Scoring grant fit...")
    scored_grants = await run_scoring_parallel(
        ctx.all_grant_data, ctx.org_profile, ctx,
        on_progress=lambda done, total: log(f"  Scoring: {done}/{total} grants scored")
    )
    log(f"  Scored {len(scored_grants)} grants — top score: {scored_grants[0].score if scored_grants else 0}")

    return ctx, scored_grants


# ── PIPELINE: Phases 6–8 (brief through compliance) ──────────────────────

async def run_pipeline_phase6_to_8(
    ctx: PipelineContext,
    selected_grants: list,
    progress_callback=None
) -> tuple[bytes, bytes]:
    """
    Run phases 6–8 and assemble final outputs.
    Returns (csv_bytes, docx_bytes) — Gradio serves these as downloads.
    """

    def log(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

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

    selected_names = {sg.grant.candidate.name for sg in selected_grants}
    selected_grant_data = [
        gd for gd in ctx.all_grant_data
        if gd.candidate.name in selected_names
    ]

    reports = await run_compliance_checks(drafts, selected_grant_data, ctx)
    log(f"  {len(reports)} compliance reports generated")

    log("Assembling outputs...")
    csv_bytes = build_csv(ctx.scored_grants)
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

def format_scores_for_display(scored_grants: list) -> list[list]:
    """Format scored grants as rows for Gradio DataFrame."""
    rows = []
    for i, sg in enumerate(scored_grants):
        rows.append([
            i + 1,
            sg.grant.candidate.name,
            sg.grant.candidate.funder,
            sg.score,
            "Yes" if sg.recommended else "No",
            "Yes" if sg.disqualified else "No",
            f"${sg.grant.award_min:,}–${sg.grant.award_max:,}",
            sg.grant.deadline,
            sg.rationale[:120] + "..." if len(sg.rationale) > 120 else sg.rationale
        ])
    return rows


def build_checkbox_choices(scored_grants: list) -> list[str]:
    """
    Build checkbox labels from non-disqualified scored grants.
    Format: "Score | Grant Name" so user can see fit score at a glance.
    """
    choices = []
    for sg in scored_grants:
        if not sg.disqualified:
            label = f"{sg.score} | {sg.grant.candidate.name}"
            choices.append(label)
    return choices


def parse_checkbox_selection(
    selected_labels: list[str],
    scored_grants: list
) -> list:
    """
    Convert selected checkbox labels back into ScoredGrant objects.
    Matches by grant name embedded in the label. Caps at 5 selections.
    """
    selected = []
    for label in selected_labels:
        # label format: "score | grant name" — extract name after the pipe
        grant_name = label.split(" | ", 1)[1] if " | " in label else label
        for sg in scored_grants:
            if sg.grant.candidate.name == grant_name:
                selected.append(sg)
                break

    if len(selected) > 5:
        print(f"User selected {len(selected)} grants — capping at 5")
        selected = selected[:5]

    return selected


# ── GRADIO UI ─────────────────────────────────────────────────────────────
import gradio as gr
import tempfile


# ── ASYNC HANDLERS ────────────────────────────────────────────────────────
# Gradio supports async handlers natively — no threading bridge needed.
# Awaiting the pipeline keeps the event loop free so the browser stays connected.

async def handle_phase1_to_5(
    org_url: str,
    org_type: str,
    user_context: str,
    max_grants: int,
    research_depth: str,
    progress=gr.Progress()
) -> tuple:
    """
    Gradio handler for Stage 1 — runs pipeline phases 1–5.
    Called when user clicks Find Grants.
    Returns updated UI components + state for Stage 2.
    """
    if not org_url or not org_url.startswith("http"):
        yield (
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            gr.update(choices=[], visible=False),
            None,
            None,
            "⚠️ Please enter a valid URL starting with http:// or https://"
        )
        return

    try:
        progress(0, desc="Starting pipeline...")

        ctx, scored_grants = await run_pipeline_phase1_to_5(
            org_url=org_url,
            org_type=org_type,
            user_context=user_context,
            max_grants=int(max_grants),
            research_depth=research_depth,
            progress_callback=lambda msg: progress(0.5, desc=msg)
        )

        if not scored_grants:
            yield (
                gr.update(visible=False),
                gr.update(visible=False),
                [],
                gr.update(choices=[], visible=False),
                None,
                None,
                "⚠️ No grants found. Try a different URL or check your API keys."
            )
            return

        progress(0.9, desc="Preparing results...")

        table_rows = format_scores_for_display(scored_grants)
        checkbox_choices = build_checkbox_choices(scored_grants)

        yield (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(value=table_rows),
            gr.update(choices=checkbox_choices, value=[], visible=True),
            ctx,
            scored_grants,
            f"✅ Found {len(scored_grants)} grants — {len(checkbox_choices)} eligible for drafting. Select below."
        )

    except Exception as e:
        yield (
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            gr.update(choices=[], visible=False),
            None,
            None,
            f"❌ Pipeline error: {str(e)}"
        )


async def handle_phase6_to_8(
    selected_labels: list,
    ctx,
    scored_grants: list,
    progress=gr.Progress()
) -> tuple:
    """
    Gradio handler for Stage 2 — runs pipeline phases 6–8.
    Called when user clicks Draft Selected Grants.
    Returns download file paths for CSV and DOCX.
    """
    if ctx is None or scored_grants is None:
        yield (
            gr.update(visible=False),
            None,
            None,
            "⚠️ Please run Find Grants first."
        )
        return

    if not selected_labels:
        yield (
            gr.update(visible=False),
            None,
            None,
            "⚠️ Please select at least one grant to draft."
        )
        return

    try:
        progress(0, desc="Starting drafting pipeline...")

        selected_grants = parse_checkbox_selection(selected_labels, scored_grants)

        progress(0.1, desc="Writing strategic briefs...")

        csv_bytes, docx_bytes = await run_pipeline_phase6_to_8(
            ctx=ctx,
            selected_grants=selected_grants,
            progress_callback=lambda msg: progress(0.5, desc=msg)
        )

        progress(0.9, desc="Assembling downloads...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
            f.write(csv_bytes)
            csv_path = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(docx_bytes)
            docx_path = f.name

        yield (
            gr.update(visible=True),
            csv_path,
            docx_path,
            f"✅ Done — {len(selected_grants)} grant(s) drafted. Download your files below."
        )

    except Exception as e:
        yield (
            gr.update(visible=False),
            None,
            None,
            f"❌ Drafting error: {str(e)}"
        )


# ── GRADIO LAYOUT ─────────────────────────────────────────────────────────

with gr.Blocks(title="GrantRadar") as demo:

    # ── HEADER ────────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🎯 GrantRadar
    **AI-powered grant discovery and proposal drafting for nonprofits.**
    Enter your organization's website URL and GrantRadar will find matching
    grants, score them for fit, and draft proposal sections for the best opportunities.
    """)

    # ── STAGE 1: INPUT FORM ───────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            org_url = gr.Textbox(
                label="Organization Website URL",
                placeholder="https://www.yourorg.org",
                info="The homepage of the grant-seeking organization"
            )
        with gr.Column(scale=1):
            org_type = gr.Dropdown(
                choices=["nonprofit", "for-profit", "government"],
                value="nonprofit",
                label="Organization Type"
            )

    user_context = gr.Textbox(
        label="Additional Context (optional)",
        placeholder="e.g. Focus on youth programs, we serve Dallas-Fort Worth, target grants $25K–$100K",
        info="Any priorities, program names, or constraints to factor into the search",
        lines=2
    )

    with gr.Row():
        with gr.Column(scale=1):
            max_grants = gr.Slider(
                minimum=3,
                maximum=15,
                value=5,
                step=1,
                label="Max Grants to Analyze",
                info="Higher = more thorough but slower"
            )
        with gr.Column(scale=1):
            research_depth = gr.Dropdown(
                choices=["quick", "standard", "deep"],
                value="standard",
                label="Research Depth"
            )

    find_btn = gr.Button("🔍 Find Grants", variant="primary", size="lg")
    status_msg = gr.Markdown("")

    # ── STAGE 1: RESULTS TABLE ────────────────────────────────────────────
    with gr.Column(visible=False) as results_section:
        gr.Markdown("## Scored Grant Opportunities")
        scores_table = gr.DataFrame(
            headers=["Rank", "Grant Name", "Funder", "Score", "Recommended",
                     "Disqualified", "Award Range", "Deadline", "Rationale"],
            datatype=["number", "str", "str", "number", "str", "str", "str", "str", "str"],
            wrap=True
        )

    # ── STAGE 2: HITL SELECTION ───────────────────────────────────────────
    with gr.Column(visible=False) as hitl_section:
        gr.Markdown("## Select Grants to Draft")
        gr.Markdown("Check the grants you want proposals drafted for, then click **Draft Selected Grants**.")

        # populated dynamically after phase 1-5 completes
        grant_checkboxes = gr.CheckboxGroup(
            choices=[],
            label="Eligible Grants (Score | Grant Name)",
            visible=False
        )

        draft_btn = gr.Button("✍️ Draft Selected Grants", variant="primary", size="lg")

    # ── STAGE 2: DOWNLOADS ────────────────────────────────────────────────
    with gr.Column(visible=False) as download_section:
        gr.Markdown("## Downloads")
        with gr.Row():
            csv_download = gr.File(label="Grant Opportunities (CSV)")
            docx_download = gr.File(label="Full Report with Proposals (DOCX)")

    # ── STATE ─────────────────────────────────────────────────────────────
    ctx_state = gr.State(None)          # holds PipelineContext between stages
    grants_state = gr.State(None)       # holds scored_grants list between stages

    # ── EVENT WIRING ──────────────────────────────────────────────────────
    # show immediate status, then run async pipeline
    find_btn.click(
        fn=lambda: "⏳ Pipeline running — this takes 2–3 minutes. Please wait...",
        outputs=[status_msg]
    ).then(
        fn=handle_phase1_to_5,
        inputs=[org_url, org_type, user_context, max_grants, research_depth],
        outputs=[results_section, hitl_section, scores_table,
                 grant_checkboxes, ctx_state, grants_state, status_msg]
    )

    draft_btn.click(
        fn=lambda: "⏳ Drafting proposals — this takes 3–5 minutes. Please wait...",
        outputs=[status_msg]
    ).then(
        fn=handle_phase6_to_8,
        inputs=[grant_checkboxes, ctx_state, grants_state],
        outputs=[download_section, csv_download, docx_download, status_msg]
    )


# ── LAUNCH ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue()                        # required for .then() chaining and async handlers
    demo.launch(theme=gr.themes.Soft())