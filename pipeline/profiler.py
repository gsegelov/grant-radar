"""
pipeline/profiler.py

OrgProfiler agent — synthesizes scraped site text into a structured OrgProfile.
Uses Pro — this requires reasoning and inference, not just extraction.

Output feeds: FitScorer, StrategicBriefWriter, ProposalDrafter, PipelineContext.
The quality of search_queries generated here determines what grants get found.
"""

import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (Agent, Runner, set_default_openai_client,
                    set_default_openai_api, set_tracing_disabled)
from models.schemas import OrgProfile, ScrapedSite
from pipeline.context import PipelineContext
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


# ── AGENT: OrgProfiler ────────────────────────────────────────────────────
# No tools — reasons over text passed directly in the prompt.
# Receives all scraped text combined into one input string.

org_profiler = Agent(
    name="OrgProfiler",
    instructions="""You are an expert nonprofit analyst. Your job is to read an
    organization's website text and produce a structured profile used for grant matching.

    Analyze the text carefully and infer what isn't explicitly stated where reasonable.
    For example: budget range can be inferred from staff size, program scale, and geography.

    Generate 6-8 targeted search queries in search_queries. These will be used to find
    grant opportunities. Make each query target a different funding angle:
    - sector + geography + org type
    - specific program area + funder type
    - eligibility tag combinations
    - deadline ranges if relevant

    Return ONLY a JSON object with no markdown fences:
    {
        "name": "organization name",
        "mission_summary": "2-3 sentence synthesis",
        "primary_sector": "single main sector",
        "secondary_sectors": ["sector1", "sector2"],
        "geographic_scope": "where they operate",
        "annual_budget_range": "estimated range or Unknown",
        "grant_size_target": "appropriate grant size or Unknown",
        "eligibility_tags": ["tag1", "tag2", "tag3"],
        "inferred_programs": ["program1", "program2"],
        "search_queries": ["query1", "query2", "query3", "query4", "query5", "query6"]
    }""",
    model="gemini-2.5-pro"      # Pro — requires real reasoning
)


def build_profiler_input(site: ScrapedSite, user_context: str) -> str:
    """
    Combine all scraped text into one input string for OrgProfiler.
    Includes user_context so any priorities or constraints the user specified
    are factored into the profile and search queries.
    """
    parts = ["HOMEPAGE:\n" + site.homepage_text]

    # add each subpage with its URL as a label
    for url, text in site.subpage_texts.items():
        parts.append(f"PAGE {url}:\n{text}")

    # append user context if provided — influences search query generation
    if user_context:
        parts.append(f"USER PRIORITIES:\n{user_context}")

    return "\n\n---\n\n".join(parts)    # join sections with a clear separator


def parse_org_profile(raw_output: str) -> OrgProfile:
    """Parse OrgProfiler's JSON text output into an OrgProfile object."""
    data = parse_json_output(raw_output)
    return OrgProfile(**data)