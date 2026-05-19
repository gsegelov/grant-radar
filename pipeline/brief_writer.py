"""
pipeline/brief_writer.py

StrategicBriefWriter agent — writes application angle briefs for top scored grants.
Runs sequentially (not parallel) — only processes top 5 grants after HITL selection.
"""

import os
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import ScoredGrant, GrantBrief, OrgProfile
from pipeline.context import PipelineContext
from tools.parse_utils import parse_json_output

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ── AGENT: StrategicBriefWriter ───────────────────────────────────────────
# No tools — reasons over scored grant and org profile passed in the prompt.
# Uses Pro — strategic framing requires real reasoning, not extraction.

strategic_brief_writer = Agent(
    name="StrategicBriefWriter",
    instructions="""You are a grant strategy consultant. Your job is to write a
    strategic application brief for one grant opportunity.

    You will receive: a ScoredGrant object and an OrgProfile.

    Write a brief that tells the grant writer exactly how to frame the application
    for this specific funder. Be concrete and specific — reference actual funder
    priorities and actual org programs by name.

    Return ONLY a JSON object with no markdown fences:
    {
        "grant_name": "exact grant name",
        "recommended_framing": "how to position the org for this funder",
        "priorities_to_emphasize": ["priority1", "priority2", "priority3"],
        "sections_to_strengthen": ["section1", "section2"],
        "narrative_hook": "suggested opening sentence for the application"
    }""",
    model=LitellmModel(model="gemini/gemini-2.5-pro")   # Pro — strategic reasoning
)


def build_brief_input(scored_grant: ScoredGrant, org_profile: OrgProfile) -> str:
    """Build input string combining scored grant and org profile for StrategicBriefWriter."""
    return f"""ORG PROFILE:
{org_profile.model_dump_json(indent=2)}

SCORED GRANT:
{scored_grant.model_dump_json(indent=2)}"""


def parse_grant_brief(raw_output: str) -> GrantBrief:
    """Parse StrategicBriefWriter's JSON output into a GrantBrief object."""
    try:
        data = parse_json_output(raw_output)
        return GrantBrief(**data)
    except Exception as e:
        # return minimal brief rather than crashing — drafter can still run
        return GrantBrief(
            grant_name="Unknown",
            recommended_framing="Brief generation failed — review grant manually.",
            priorities_to_emphasize=[],
            narrative_hook="Manual review required."
        )


async def run_brief_writing(
    selected_grants: list[ScoredGrant],
    org_profile: OrgProfile,
    ctx: PipelineContext
) -> list[GrantBrief]:
    """
    Run StrategicBriefWriter sequentially across selected grants.
    Sequential (not parallel) — brief quality benefits from focused attention
    per grant, and this phase only processes 2-5 grants after HITL selection.
    Called by app.py after the HITL gate.
    """
    briefs = []
    for sg in selected_grants:
        try:
            result = await Runner.run(
                strategic_brief_writer,
                build_brief_input(sg, org_profile),
                context=ctx
            )
            brief = parse_grant_brief(result.final_output)
            briefs.append(brief)
        except Exception as e:
            print(f"Brief writing failed for {sg.grant.candidate.name}: {e}")
    return briefs