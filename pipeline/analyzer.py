"""
pipeline/analyzer.py

GrantAnalyzer agent — fetches a grant page and extracts structured data.
Called in parallel via asyncio.gather() in app.py, once per GrantCandidate.

Output feeds FitScorer and ComplianceChecker — quality of extraction here
directly determines scoring accuracy and compliance check usefulness.
"""

import os
import asyncio
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import GrantCandidate, GrantData
from pipeline.context import PipelineContext
from tools.fetch_tools import fetch_page
from tools.search_tools import search_grants
from tools.parse_utils import parse_json_output

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ── AGENT: GrantAnalyzer ──────────────────────────────────────────────────
# One instance handles one candidate. Called N times in parallel.
# Uses Flash — extraction work, not complex reasoning.

grant_analyzer = Agent(
    name="GrantAnalyzer",
    instructions="""You are a grant analyst. Your job is to extract detailed information
    from a grant program page.

    Steps:
    1. Use fetch_page() to fetch the grant URL provided.
    2. If the page is sparse or returns an error, use search_grants() to find
       supplemental information about the grant program.
    3. Extract all available information and return ONLY a JSON object with no markdown fences:
    {
        "award_min": 0,
        "award_max": 0,
        "deadline": "Rolling or date or Unknown",
        "eligibility_requirements": ["requirement1", "requirement2"],
        "hard_disqualifiers": ["disqualifier1"],
        "required_attachments": ["attachment1", "attachment2"],
        "funder_priorities": ["priority1", "priority2"],
        "application_complexity": "low or medium or high"
    }

    Use 0 for award amounts if not found. Use "Unknown" for deadline if not found.
    application_complexity: low = simple form, medium = narrative required,
    high = full proposal with budget and attachments.""",
    tools=[fetch_page, search_grants],      # two tools — fetch first, search as fallback
    model=LitellmModel(model="gemini/gemini-2.5-flash")
)


def parse_grant_data(raw_output: str, candidate: GrantCandidate) -> GrantData:
    """
    Parse GrantAnalyzer's JSON output into a GrantData object.
    Wraps the original GrantCandidate so downstream agents have full context.
    Falls back to a minimal GrantData if parsing fails — keeps pipeline moving.
    """
    try:
        data = parse_json_output(raw_output)
        return GrantData(candidate=candidate, **data)   # inject candidate into the parsed data
    except Exception:
        # return minimal object rather than crashing the whole parallel batch
        return GrantData(candidate=candidate)


async def run_analysis_parallel(
    candidates: list[GrantCandidate],
    ctx: PipelineContext,
    max_grants: int = 15
) -> list[GrantData]:
    """
    Run GrantAnalyzer in parallel across all candidates.
    Caps at max_grants to control cost — set lower during development.
    Called by app.py after Phase 3 completes.
    """
    # cap candidates to avoid runaway costs during testing
    candidates_to_analyze = candidates[:max_grants]

    tasks = [
        Runner.run(
            grant_analyzer,
            f"Analyze this grant: {c.name} by {c.funder}. URL: {c.url}",
            context=ctx
        )
        for c in candidates_to_analyze
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    grant_data_list = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Analysis failed for {candidates_to_analyze[i].name}: {result}")
            continue
        grant_data = parse_grant_data(result.final_output, candidates_to_analyze[i])
        grant_data_list.append(grant_data)

    return grant_data_list