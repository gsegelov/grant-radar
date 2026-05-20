"""
pipeline/scraper.py

WebScraper agent — fetches and extracts text from an organization's website.
URLValidator — input guardrail that fires before WebScraper runs.

Gemini limitation: cannot use tools + structured JSON output simultaneously.
Since every agent in this pipeline uses at least one tool, all agents return
JSON text that we parse manually and validate with Pydantic.
"""

import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (Agent, Runner, input_guardrail, GuardrailFunctionOutput,
                    RunContextWrapper, set_default_openai_client,
                    set_default_openai_api, set_tracing_disabled)
from models.schemas import ScrapedSite, UrlValidationResult
from pipeline.context import PipelineContext
from tools.fetch_tools import fetch_page
from tools.parse_utils import parse_json_output

load_dotenv()

# ── GEMINI SETUP ──────────────────────────────────────────────────────────
# Routes SDK calls through Gemini's OpenAI-compatible endpoint.
# set_tracing_disabled silences OPENAI_API_KEY warnings.
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ── GUARDRAIL: URLValidator ───────────────────────────────────────────────
# Fires automatically before WebScraper runs on every call.
# Pure Python HTTP check — no LLM sub-agent, no credits spent.
# Only blocks social media, Wikipedia, and unreachable URLs.

@input_guardrail
async def url_validator_guardrail(
    ctx: RunContextWrapper[PipelineContext],
    agent: Agent,
    input: str
) -> GuardrailFunctionOutput:
    """Check the submitted URL before allowing WebScraper to run."""
    import requests

    # blocked domains — never legitimate org websites
    BLOCKED_DOMAINS = [
        "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
        "wikipedia.org", "youtube.com", "tiktok.com"
    ]

    # check for blocked domains — no network request needed
    for domain in BLOCKED_DOMAINS:
        if domain in input:
            return GuardrailFunctionOutput(
                output_info=UrlValidationResult(
                    is_valid=False,
                    reason=f"URL appears to be a {domain} page, not an organization website."
                ),
                tripwire_triggered=True
            )

    # for everything else — direct HTTP check, no LLM involved
    try:
        response = await asyncio.to_thread(
            requests.get, input, timeout=15, headers={"User-Agent": "Mozilla/5.0"}
        )
        is_valid = response.status_code < 500   # accept anything that isn't a server error
        return GuardrailFunctionOutput(
            output_info=UrlValidationResult(
                is_valid=is_valid,
                reason="OK" if is_valid else f"HTTP {response.status_code}"
            ),
            tripwire_triggered=not is_valid
        )
    except Exception as e:
        return GuardrailFunctionOutput(
            output_info=UrlValidationResult(is_valid=False, reason=str(e)),
            tripwire_triggered=True
        )


# ── AGENT: WebScraper ─────────────────────────────────────────────────────
# Attached guardrail fires before this agent runs on every call.
# Uses Flash — scraping is mechanical work, not complex reasoning.

web_scraper = Agent(
    name="WebScraper",
    instructions="""You are a web scraper. Your job is to extract readable text from an
    organization's website.

    Steps:
    1. Use fetch_page() to fetch the homepage URL provided.
    2. Look for internal navigation links — paths like /about, /programs, /impact, /services.
    3. Fetch up to 4 of the most relevant subpages using fetch_page().
    4. Return ONLY a JSON object with no markdown fences:
    {
        "homepage_text": "full text of homepage",
        "subpage_texts": {"/about": "text here", "/programs": "text here"},
        "nav_links": ["/about", "/programs", "/contact"]
    }

    If a page fetch fails, skip it and continue.""",
    tools=[fetch_page],
    model="gemini-2.5-flash",
    input_guardrails=[url_validator_guardrail]
)


# ── PARSER ────────────────────────────────────────────────────────────────
# Gemini can't use tools + output_type simultaneously, so the agent returns
# raw JSON text. This converts it into a validated ScrapedSite Pydantic object.
# Called by pipeline code in app.py after Runner.run(web_scraper, ...) completes.

def parse_scraped_site(raw_output: str) -> ScrapedSite:
    """Parse WebScraper's JSON text output into a ScrapedSite object."""
    data = parse_json_output(raw_output)
    return ScrapedSite(**data)