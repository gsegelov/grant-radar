"""
pipeline/scraper.py

WebScraper agent — fetches and extracts text from an organization's website.
URLValidator — input guardrail that fires before WebScraper runs.
"""

import os
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import ScrapedSite, UrlValidationResult
from pipeline.context import PipelineContext
from tools.fetch_tools import fetch_page, validate_url

load_dotenv()

# ── GEMINI SETUP ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ── GUARDRAIL: URLValidator ───────────────────────────────────────────────
# Runs before WebScraper on every call.
# Uses a fast Flash model — this is a quick judgment call, not heavy reasoning.

@input_guardrail
async def url_validator_guardrail(
    ctx: RunContextWrapper[PipelineContext],
    agent: Agent,
    input: str
) -> GuardrailFunctionOutput:
    """Check the submitted URL before allowing WebScraper to run."""

     # run a lightweight sub-agent to validate the URL
    validator_agent = Agent(
        name="URLValidator",
        instructions="""You are a URL validator. Use the validate_url tool on the URL provided.
        Return your result immediately. Do not fetch the page content.""",
        tools=[validate_url],
        output_type=UrlValidationResult,
        model=LitellmModel(model="gemini/gemini-2.5-flash")
     )

    result = await Runner.run(validator_agent, input, context=ctx.context)
    validation: UrlValidationResult = result.final_output

    return GuardrailFunctionOutput(
        output_info=validation,                         # pass the result through for logging
        tripwire_triggered=not validation.is_valid      # True = stop the pipeline
    )


# ── AGENT: WebScraper ─────────────────────────────────────────────────────

web_scraper = Agent(
    name="WebScraper",
    instructions="""You are a web scraper. Your job is to extract readable text from an
    organization's website.

    Steps:
    1. Use fetch_page() to fetch the homepage URL provided.
    2. Extract all navigation links from the text — look for paths like /about, /programs,
       /impact, /services, /mission.
    3. Fetch up to 4 of the most relevant subpages using fetch_page().
    4. Return a ScrapedSite object with homepage_text, subpage_texts, and nav_links.

    If a page fetch fails, skip it and continue — do not stop the pipeline.""",
    tools=[fetch_page],
    output_type=ScrapedSite,
    model=LitellmModel(model="gemini/gemini-2.5-flash"),
    input_guardrails=[url_validator_guardrail]      # fires before every run
)