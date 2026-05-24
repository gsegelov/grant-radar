"""
app.py — GrantRadar
Simple UI: URL in → DOCX + CSV out. Full 8-agent pipeline runs underneath.
"""

import os
import asyncio
import threading
import tempfile
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (set_default_openai_client, set_default_openai_api,
                    set_tracing_disabled)
import gradio as gr

from pipeline.context import PipelineContext
from pipeline.scraper import web_scraper, parse_scraped_site
from pipeline.profiler import org_profiler, build_profiler_input, parse_org_profile
from pipeline.discoverer import run_discovery_parallel
from pipeline.analyzer import run_analysis_parallel
from pipeline.scorer import run_scoring_parallel
from pipeline.brief_writer import run_brief_writing
from pipeline.drafter import run_drafting
from pipeline.compliance import run_compliance_checks
from tools.export_tools import build_csv, build_docx
from agents import Runner
from models.schemas import OrgProfile

load_dotenv()

# ── GEMINI SETUP ──────────────────────────────────────────────────────────
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60.0
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MOCK_PIPELINE = os.getenv("MOCK_PIPELINE", "false").lower() == "true"


# ── PIPELINE FUNCTIONS ────────────────────────────────────────────────────

async def run_pipeline_phase1_to_5(org_url, org_type, user_context,
                                    max_grants, research_depth,
                                    progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    ctx = PipelineContext(
        org_url=org_url, org_type=org_type, user_context=user_context,
        max_grants_to_analyze=max_grants, research_depth=research_depth
    )

    log("Phase 1/5 — Scraping organization website...")
    scrape_result = await Runner.run(web_scraper, org_url, context=ctx)
    site = parse_scraped_site(scrape_result.final_output)

    log("Phase 2/5 — Building organization profile...")
    profile_result = await Runner.run(
        org_profiler, build_profiler_input(site, user_context), context=ctx
    )
    ctx.org_profile = parse_org_profile(profile_result.final_output)

    log("Phase 3/5 — Discovering grant opportunities...")
    ctx.all_candidates = await run_discovery_parallel(
        ctx.org_profile.search_queries, ctx
    )

    log(f"Phase 4/5 — Analyzing top {max_grants} grants...")
    ctx.all_grant_data = await run_analysis_parallel(
        ctx.all_candidates, ctx, max_grants=max_grants
    )

    log("Phase 5/5 — Scoring grant fit...")
    scored_grants = await run_scoring_parallel(
        ctx.all_grant_data, ctx.org_profile, ctx
    )
    ctx.scored_grants = scored_grants

    return ctx, scored_grants


async def run_pipeline_phase6_to_8(ctx, selected_grants,
                                    progress_callback=None):
    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    ctx.selected_grants = selected_grants

    log("Phase 6/8 — Writing strategic briefs...")
    briefs = await run_brief_writing(selected_grants, ctx.org_profile, ctx)

    log("Phase 7/8 — Drafting proposals...")
    drafts = await run_drafting(selected_grants, briefs, ctx.org_profile, ctx)

    log("Phase 8/8 — Running compliance checks...")
    selected_names = {sg.grant.candidate.name for sg in selected_grants}
    selected_grant_data = [
        gd for gd in ctx.all_grant_data
        if gd.candidate.name in selected_names
    ]
    reports = await run_compliance_checks(drafts, selected_grant_data, ctx)

    csv_bytes = build_csv(ctx.scored_grants)
    docx_bytes = build_docx(ctx.org_profile, ctx.scored_grants,
                            briefs, drafts, reports)
    return csv_bytes, docx_bytes


# ── MAIN HANDLER ─────────────────────────────────────────────────────────

def run_grantradar(org_url, org_type, user_context,
                   max_grants, research_depth, progress=gr.Progress()):
    """Single handler: runs full pipeline, returns files."""

    if MOCK_PIPELINE:
        import time
        progress(0.3, desc="[MOCK] Running pipeline...")
        time.sleep(1)
        progress(1.0, desc="[MOCK] Done!")
        return None, None, "✅ [MOCK] Pipeline complete — in real mode, files appear here."

    if not org_url or not org_url.startswith("http"):
        return None, None, "⚠️ Please enter a valid URL starting with http:// or https://"

    progress(0.05, desc="Starting pipeline...")
    result = {}

    def _run():
        async def _async():
            try:
                ctx, scored_grants = await run_pipeline_phase1_to_5(
                    org_url=org_url, org_type=org_type,
                    user_context=user_context,
                    max_grants=int(max_grants),
                    research_depth=research_depth,
                    progress_callback=print
                )
                if not scored_grants:
                    result["error"] = "No grants found. Try a different URL."
                    return

                selected = [sg for sg in scored_grants
                            if sg.recommended and not sg.disqualified][:3]
                if not selected:
                    selected = [sg for sg in scored_grants
                                if not sg.disqualified][:3]
                if not selected:
                    selected = scored_grants[:3]

                csv_bytes, docx_bytes = await run_pipeline_phase6_to_8(
                    ctx=ctx, selected_grants=selected,
                    progress_callback=print
                )
                result["csv"] = csv_bytes
                result["docx"] = docx_bytes
                result["n"] = len(selected)
                result["top"] = scored_grants[0].score

            except Exception as e:
                import traceback
                result["error"] = f"{str(e)}\n{traceback.format_exc()}"

        asyncio.run(_async())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=300)

    if "error" in result:
        return None, None, f"❌ {result['error']}"
    if "csv" not in result:
        return None, None, "❌ Pipeline timed out. Try Max Grants = 3 and Research Depth = quick."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(result["csv"])
        csv_path = f.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        f.write(result["docx"])
        docx_path = f.name

    return (
        csv_path,
        docx_path,
        f"✅ Done — {result['n']} grant(s) drafted, top score {result['top']}. Download below."
    )


# ── GRADIO UI ─────────────────────────────────────────────────────────────

with gr.Blocks(title="GrantRadar", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎯 GrantRadar
    **AI-powered grant discovery and proposal drafting for nonprofits.**
    Enter your organization's website URL. GrantRadar finds matching grants,
    scores them for fit, and drafts proposal sections — delivered as a
    ready-to-edit Word document.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            org_url = gr.Textbox(
                label="Organization Website URL",
                placeholder="https://www.yourorg.org"
            )
        with gr.Column(scale=1):
            org_type = gr.Dropdown(
                choices=["nonprofit", "for-profit", "government"],
                value="nonprofit",
                label="Organization Type"
            )

    user_context = gr.Textbox(
        label="Additional Context (optional)",
        placeholder="e.g. Youth programs, Dallas-Fort Worth, grants $25K–$100K",
        lines=2
    )

    with gr.Row():
        with gr.Column(scale=1):
            max_grants = gr.Slider(
                minimum=3, maximum=5, value=3, step=1,
                label="Max Grants to Analyze",
                info="Keep at 3 for fastest results"
            )
        with gr.Column(scale=1):
            research_depth = gr.Dropdown(
                choices=["quick", "standard", "deep"],
                value="quick",
                label="Research Depth"
            )

    run_btn = gr.Button("🔍 Run GrantRadar", variant="primary", size="lg")
    status_msg = gr.Markdown("")

    gr.Markdown("## Downloads")
    with gr.Row():
        csv_download = gr.File(label="Grant Opportunities (CSV)")
        docx_download = gr.File(label="Full Report with Proposals (DOCX)")

    run_btn.click(
        fn=run_grantradar,
        inputs=[org_url, org_type, user_context, max_grants, research_depth],
        outputs=[csv_download, docx_download, status_msg]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
