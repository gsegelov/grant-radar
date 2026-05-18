"""
pipeline/scraper.py

WebScraper agent — fetches and extracts text from an organization's website.
URLValidator — input guardrail that fires before WebScraper runs.

Gemini limitation: cannot use tools + structured JSON output simultaneously.
Since every agent in this pipeline uses at least one tool, all agents return
JSON text that we parse manually and validate with Pydantic.
"""

import os
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import ScrapedSite, UrlValidationResult
from pipeline.context import PipelineContext
from tools.fetch_tools import fetch_page, validate_url
from tools.parse_utils import parse_json_output  # shared JSON parsing utility

load_dotenv()   # load .env so GOOGLE_API_KEY is available

# connect to Gemini using the API key from .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ── GUARDRAIL: URLValidator ───────────────────────────────────────────────
# Fires automatically before WebScraper runs on every call.
# If the URL is invalid, tripwire_triggered=True stops the pipeline here —
# no scraping, no profiling, no credits spent.

@input_guardrail  # SDK decorator — registers this as an input guardrail function
async def url_validator_guardrail(
    ctx: RunContextWrapper[PipelineContext],    # wrapper around our shared PipelineContext
    agent: Agent,                               # the agent this guardrail is attached to
    input: str                                  # the raw input string passed to the agent
) -> GuardrailFunctionOutput:                   # SDK-required return type for guardrails
    """Check the submitted URL before allowing WebScraper to run."""

    # lightweight sub-agent — Flash is fast and cheap for a simple judgment call
    validator_agent = Agent(
        name="URLValidator",
        instructions="""You are a URL validator. Use the validate_url tool on the URL provided.
        After calling the tool, return ONLY a JSON object with no markdown fences:
        {"is_valid": true or false, "reason": "explanation here"}""",
        tools=[validate_url],
        model=LitellmModel(model="gemini/gemini-2.5-flash")
    )

    # run the validator sub-agent inside the guardrail
    result = await Runner.run(validator_agent, input, context=ctx.context)

    # parse JSON output using shared utility — handles fences and escape issues
    data = parse_json_output(result.final_output)
    validation = UrlValidationResult(**data)    # unpack dict into Pydantic model

    return GuardrailFunctionOutput(
        output_info=validation,                     # passed through for logging
        tripwire_triggered=not validation.is_valid  # True = stop pipeline, False = continue
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
    tools=[fetch_page],                                 # only tool this agent can call
    model=LitellmModel(model="gemini/gemini-2.5-flash"),
    input_guardrails=[url_validator_guardrail]          # runs before every WebScraper call
)


# ── PARSER ────────────────────────────────────────────────────────────────
# Gemini can't use tools + output_type simultaneously, so the agent returns
# raw JSON text. This converts it into a validated ScrapedSite Pydantic object.
# Called by pipeline code in app.py after Runner.run(web_scraper, ...) completes.

def parse_scraped_site(raw_output: str) -> ScrapedSite:
    """Parse WebScraper's JSON text output into a ScrapedSite object."""
    data = parse_json_output(raw_output)    # shared utility handles fences and escape fixes
    return ScrapedSite(**data)              # Pydantic validates all fields on construction