"""
tools/parse_utils.py

Shared JSON parsing utility for all pipeline agents.

Gemini returns JSON wrapped in markdown fences despite instructions.
Every agent uses this function to clean and parse its output before
constructing Pydantic objects. Fix once here — all agents benefit.
"""

import re
import json


def parse_json_output(raw_output: str) -> dict:
    """
    Clean and parse a JSON string returned by a Gemini agent.

    Handles two common Gemini quirks:
    1. JSON wrapped in ```json ... ``` markdown fences
    2. Invalid backslash escape sequences in scraped text content

    Returns a plain dict — caller constructs the Pydantic object.
    Raises json.JSONDecodeError if the output cannot be parsed.
    """
    raw = raw_output.strip()

    # strip markdown fences if present
    if "```" in raw:
        raw = raw.split("```")[1]       # content between first and second fence
        if raw.startswith("json"):
            raw = raw[4:]               # strip the "json" language tag
    raw = raw.strip()

    # fix invalid backslash escapes — valid JSON escapes are: \" \\ \/ \b \f \n \r \t \uXXXX
    # scraped web content often contains bare backslashes that break json.loads
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

    return json.loads(raw)