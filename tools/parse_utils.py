"""
tools/parse_utils.py

Shared JSON parsing utility for all pipeline agents.

Gemini returns JSON wrapped in markdown fences despite instructions.
Every agent uses this function to clean and parse its output before
constructing Pydantic objects. Fix once here — all agents benefit.
"""

import re
import json


def parse_json_output(raw_output: str) -> dict | list:
    """
    Clean and parse a JSON string returned by a Gemini agent.

    Handles three common Gemini quirks:
    1. JSON wrapped in ```json ... ``` markdown fences
    2. Invalid backslash escape sequences in scraped text content
    3. Extra text after the closing brace or bracket
    """
    raw = raw_output.strip()

    # strip markdown fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # fix invalid backslash escapes
    raw = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)

    # extract just the JSON object or array — ignore any trailing text
    if raw.startswith("{"):
        end = _find_closing(raw, "{", "}")
    elif raw.startswith("["):
        end = _find_closing(raw, "[", "]")
    else:
        end = len(raw)

    return json.loads(raw[:end])


def _find_closing(s: str, open_char: str, close_char: str) -> int:
    """Find the index just after the matching closing bracket."""
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(s):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i + 1
    return len(s)