"""
pipeline/scraper.py

WebScraper agent — fetches and extracts text from an organization's website.
URLValidator — input guardrail that fires before WebScraper runs.

Gemini limitation: cannot use tools + structured JSON output simultaneously.
All agents return JSON text; we parse it manually and validate with Pydantic.
"""

import os
import json
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner, input_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import ScrapedSite, UrlValidationResult
from pipeline.context import PipelineContext
from tools.fetch_tools import fetch_page, validate_url

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

    # run the validator sub-agent synchronously inside the guardrail
    result = await Runner.run(validator_agent, input, context=ctx.context)

    # Gemini often wraps JSON in ```json ... ``` fences despite instructions — strip them
    raw = result.final_output.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]       # take the part after the opening fence
        if raw.startswith("json"):
            raw = raw[4:]               # strip the "json" language tag if present
    data = json.loads(raw.strip())      # parse the clean JSON string into a dict
    validation = UrlValidationResult(**data)    # unpack dict into our Pydantic model

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
# Because Gemini can't use tools + output_type simultaneously, the agent
# returns raw JSON text. This function converts that text into a validated
# ScrapedSite Pydantic object. Called by pipeline code in app.py.

def parse_scraped_site(raw_output: str) -> ScrapedSite:
    """Parse WebScraper's JSON text output into a ScrapedSite object."""
    import re
    raw = raw_output.strip()
    # strip markdown fences Gemini adds
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    # fix invalid backslash escapes in scraped text content
    # valid JSON escapes are: \" \\ \/ \b \f \n \r \t \uXXXX
    # any other \X is invalid — replace with just the character
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    data = json.loads(raw)
    return ScrapedSite(**data)