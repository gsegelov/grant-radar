"""
pipeline/drafter.py

ProposalDrafter agent — drafts four standard proposal sections for one grant.
DraftQualityValidator — output guardrail that checks draft meets minimum standards.
Runs sequentially after StrategicBriefWriter completes.
"""

import os
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner, output_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import ScoredGrant, GrantBrief, ProposalDraft, OrgProfile
from pipeline.context import PipelineContext
from tools.parse_utils import parse_json_output

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# minimum word count per section — below this triggers the quality guardrail
MIN_SECTION_WORDS = 100


# ── GUARDRAIL: DraftQualityValidator ──────────────────────────────────────
# Fires after every ProposalDrafter run.
# Pure Python — checks word count and funder name presence.
# If it trips, app.py retries once before setting needs_manual_completion=True.

@output_guardrail
async def draft_quality_validator(
    ctx: RunContextWrapper[PipelineContext],
    agent: Agent,
    output: str                             # raw text output from ProposalDrafter
) -> GuardrailFunctionOutput:
    """Check that draft sections meet minimum quality standards."""
    try:
        data = parse_json_output(output)

        sections = [
            data.get("organizational_capacity", ""),
            data.get("problem_statement", ""),
            data.get("project_description", ""),
            data.get("evaluation_plan", "")
        ]

        # check every section meets minimum word count
        for section in sections:
            if len(section.split()) < MIN_SECTION_WORDS:
                return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    except Exception:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)


# ── AGENT: ProposalDrafter ────────────────────────────────────────────────
# No tools — drafts from org profile, brief, and scored grant in the prompt.
# Uses Pro — drafting requires contextual writing quality.

proposal_drafter = Agent(
    name="ProposalDrafter",
    instructions="""You are a professional grant writer. Draft four proposal sections
    for one grant application.

    You will receive: an OrgProfile, a GrantBrief, and a ScoredGrant.

    Use the GrantBrief's recommended_framing and narrative_hook to shape the writing.
    Reference the funder's name and stated priorities specifically in each section.
    Use actual program names from OrgProfile.inferred_programs where relevant.
    Each section must be at least 150 words — write substantively, not generically.

    Return ONLY a JSON object with no markdown fences:
    {
        "grant_name": "exact grant name",
        "organizational_capacity": "section text here",
        "problem_statement": "section text here",
        "project_description": "section text here",
        "evaluation_plan": "section text here"
    }""",
    model=LitellmModel(model="gemini/gemini-2.5-pro"),
    output_guardrails=[draft_quality_validator]     # fires after every draft
)


def build_drafter_input(
    scored_grant: ScoredGrant,
    brief: GrantBrief,
    org_profile: OrgProfile
) -> str:
    """Build input string combining all context needed for proposal drafting."""
    return f"""ORG PROFILE:
{org_profile.model_dump_json(indent=2)}

STRATEGIC BRIEF:
{brief.model_dump_json(indent=2)}

SCORED GRANT:
{scored_grant.model_dump_json(indent=2)}"""


def parse_proposal_draft(raw_output: str) -> ProposalDraft:
    """Parse ProposalDrafter's JSON output into a ProposalDraft object."""
    try:
        data = parse_json_output(raw_output)
        return ProposalDraft(**data)
    except Exception:
        return ProposalDraft(
            grant_name="Unknown",
            organizational_capacity="Draft generation failed — manual completion required.",
            problem_statement="Draft generation failed — manual completion required.",
            project_description="Draft generation failed — manual completion required.",
            evaluation_plan="Draft generation failed — manual completion required.",
            needs_manual_completion=True    # flag for DOCX warning banner
        )


async def run_drafting(
    selected_grants: list[ScoredGrant],
    briefs: list[GrantBrief],
    org_profile: OrgProfile,
    ctx: PipelineContext
) -> list[ProposalDraft]:
    """
    Run ProposalDrafter sequentially for each selected grant.
    Retries once if DraftQualityValidator trips — sets needs_manual_completion=True
    on second failure rather than crashing.
    Called by app.py after brief writing completes.
    """
    # build lookup so we can find the right brief for each grant
    brief_lookup = {b.grant_name: b for b in briefs}
    drafts = []

    for sg in selected_grants:
        grant_name = sg.grant.candidate.name
        brief = brief_lookup.get(grant_name)

        if not brief:
            print(f"No brief found for {grant_name} — skipping draft")
            continue

        try:
            result = await Runner.run(
                proposal_drafter,
                build_drafter_input(sg, brief, org_profile),
                context=ctx
            )
            draft = parse_proposal_draft(result.final_output)
            drafts.append(draft)

        except Exception as e:
            print(f"Draft failed for {grant_name}: {e} — flagging for manual completion")
            # create a flagged placeholder rather than crashing
            draft = parse_proposal_draft("{}")
            draft.grant_name = grant_name
            drafts.append(draft)

    return drafts