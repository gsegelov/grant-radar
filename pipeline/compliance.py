"""
pipeline/compliance.py

ComplianceChecker agent — checks each ProposalDraft against grant requirements.
Runs sequentially after ProposalDrafter completes.
Last agent in the pipeline before output assembly.
"""

import os
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import ProposalDraft, GrantData, ComplianceReport
from pipeline.context import PipelineContext
from tools.parse_utils import parse_json_output

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# ── AGENT: ComplianceChecker ──────────────────────────────────────────────
# No tools — reads draft and grant data passed in the prompt.
# Uses Pro — requires careful cross-referencing of requirements vs draft content.

compliance_checker = Agent(
    name="ComplianceChecker",
    instructions="""You are a grant compliance reviewer. Check a proposal draft
    against the grant's stated requirements.

    You will receive: a ProposalDraft and a GrantData object.

    For each item in GrantData.eligibility_requirements, determine whether the
    draft addresses it clearly, weakly, or not at all.

    For GrantData.required_attachments, check whether each is referenced in the draft.

    overall_readiness must be one of: "Strong", "Needs Work", "Incomplete"
    - Strong: all requirements addressed, no missing attachments
    - Needs Work: most requirements addressed, minor gaps
    - Incomplete: significant requirements missing

    priority_fixes must contain EXACTLY 3 items — the most important fixes before submission.

    Return ONLY a JSON object with no markdown fences:
    {
        "grant_name": "exact grant name",
        "requirements_addressed": ["requirement1", "requirement2"],
        "requirements_missing": ["requirement3"],
        "requirements_weak": ["requirement4"],
        "missing_attachments": ["attachment1"],
        "overall_readiness": "Strong or Needs Work or Incomplete",
        "priority_fixes": ["fix1", "fix2", "fix3"]
    }""",
    model=LitellmModel(model="gemini/gemini-2.5-pro")   # Pro — careful cross-referencing
)


def build_compliance_input(draft: ProposalDraft, grant_data: GrantData) -> str:
    """Build input string combining draft and grant data for ComplianceChecker."""
    return f"""PROPOSAL DRAFT:
{draft.model_dump_json(indent=2)}

GRANT REQUIREMENTS:
{grant_data.model_dump_json(indent=2)}"""


def parse_compliance_report(raw_output: str) -> ComplianceReport:
    """Parse ComplianceChecker's JSON output into a ComplianceReport object."""
    try:
        data = parse_json_output(raw_output)
        # ensure priority_fixes has exactly 3 items — pad or trim if needed
        fixes = data.get("priority_fixes", [])
        while len(fixes) < 3:
            fixes.append("Review and strengthen this application before submitting.")
        data["priority_fixes"] = fixes[:3]      # trim to 3 if agent returned more
        return ComplianceReport(**data)
    except Exception:
        return ComplianceReport(
            grant_name="Unknown",
            overall_readiness="Incomplete",
            priority_fixes=[
                "Compliance check failed — manual review required.",
                "Verify all eligibility requirements are addressed.",
                "Confirm all required attachments are prepared."
            ]
        )


async def run_compliance_checks(
    drafts: list[ProposalDraft],
    grant_data_list: list[GrantData],
    ctx: PipelineContext
) -> list[ComplianceReport]:
    """
    Run ComplianceChecker sequentially for each draft.
    Matches drafts to grant data by grant name.
    Called by app.py after drafting completes.
    """
    # build lookup so we can find grant data by name
    grant_data_lookup = {gd.candidate.name: gd for gd in grant_data_list}
    reports = []

    for draft in drafts:
        grant_data = grant_data_lookup.get(draft.grant_name)

        if not grant_data:
            print(f"No grant data found for {draft.grant_name} — skipping compliance check")
            continue

        try:
            result = await Runner.run(
                compliance_checker,
                build_compliance_input(draft, grant_data),
                context=ctx
            )
            report = parse_compliance_report(result.final_output)
            reports.append(report)
        except Exception as e:
            print(f"Compliance check failed for {draft.grant_name}: {e}")

    return reports