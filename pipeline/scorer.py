"""
pipeline/scorer.py

FitScorer agent — scores one GrantData object against the OrgProfile.
ScoringValidator — output guardrail that fires after every FitScorer run.
Called in parallel via asyncio.gather() in app.py, once per GrantData.
"""

import os
import asyncio
from dotenv import load_dotenv
from google import genai
from agents import Agent, Runner, output_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.extensions.models.litellm_model import LitellmModel
from models.schemas import GrantData, ScoredGrant, OrgProfile
from pipeline.context import PipelineContext
from tools.parse_utils import parse_json_output

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# minimum score threshold for a grant to be marked recommended
RECOMMENDATION_THRESHOLD = 60


# ── GUARDRAIL: ScoringValidator ───────────────────────────────────────────
# Fires after every FitScorer run.
# Pure Python — no LLM needed, just arithmetic checks.
# Trips if: no recommended grants exist, or all scores are within 5 points
# of each other (degenerate distribution — model isn't differentiating).

@output_guardrail  # SDK decorator — registers this as an output guardrail
async def scoring_validator(
    ctx: RunContextWrapper[PipelineContext],
    agent: Agent,
    output: str                             # raw text output from FitScorer
) -> GuardrailFunctionOutput:
    """Validate that scoring produced meaningful, differentiated results."""

    try:
        scored_grants = ctx.context.scored_grants   # read accumulated scores from context

        if not scored_grants:
            # no scores yet — too early to validate, let it pass
            return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

        non_disqualified = [sg for sg in scored_grants if not sg.disqualified]

        # check 1: at least one recommended grant exists
        has_recommendation = any(sg.recommended for sg in non_disqualified)

        # check 2: score range is meaningful — not all bunched together
        if len(non_disqualified) > 1:
            scores = [sg.score for sg in non_disqualified]
            score_range = max(scores) - min(scores)
            degenerate = score_range <= 5      # all scores within 5 points = not differentiating
        else:
            degenerate = False                  # only one grant — range check doesn't apply

        should_trip = not has_recommendation or degenerate

        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=should_trip)

    except Exception:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)


# ── AGENT: FitScorer ──────────────────────────────────────────────────────
# No tools — reasons over grant data and org profile passed in the prompt.
# Uses Pro — scoring requires nuanced judgment, not just extraction.

fit_scorer = Agent(
    name="FitScorer",
    instructions="""You are a grant fit analyst. Score how well a grant opportunity
    matches an organization's profile.

    You will receive: a GrantData object and an OrgProfile.

    Scoring criteria:
    - Mission alignment with funder priorities (40 points)
    - Eligibility match — org meets stated requirements (30 points)
    - Geographic and budget fit (20 points)
    - Application feasibility given org capacity (10 points)

    Immediately set disqualified=true and score=0 if any hard_disqualifier applies.

    Set recommended=true if score >= 60 AND disqualified=false.

    Return ONLY a JSON object with no markdown fences:
    {
        "score": 0-100,
        "rationale": "2-3 sentence explanation",
        "top_alignment_points": ["point1", "point2"],
        "gaps": ["gap1", "gap2"],
        "disqualified": false,
        "disqualifiers": [],
        "recommended": false
    }""",
    model=LitellmModel(model="gemini/gemini-2.5-pro"),  # Pro — nuanced judgment required
    output_guardrails=[scoring_validator]                # fires after every score
)


def build_scorer_input(grant_data: GrantData, org_profile: OrgProfile) -> str:
    """
    Build the input string for FitScorer combining grant data and org profile.
    Both objects serialized to JSON so the agent has all fields available.
    """
    return f"""ORG PROFILE:
{org_profile.model_dump_json(indent=2)}

GRANT TO SCORE:
{grant_data.model_dump_json(indent=2)}"""


def parse_scored_grant(raw_output: str, grant_data: GrantData) -> ScoredGrant:
    """
    Parse FitScorer's JSON output into a ScoredGrant object.
    Injects the original GrantData so the full chain is preserved.
    Falls back to a disqualified placeholder if parsing fails.
    """
    try:
        data = parse_json_output(raw_output)
        return ScoredGrant(grant=grant_data, **data)
    except Exception:
        return ScoredGrant(
            grant=grant_data,
            score=0,
            rationale="Scoring failed — manual review required.",
            top_alignment_points=[],
            disqualified=True,
            recommended=False
        )


async def run_scoring_parallel(
    grant_data_list: list[GrantData],
    org_profile: OrgProfile,
    ctx: PipelineContext
) -> list[ScoredGrant]:
    """
    Run FitScorer in parallel across all GrantData objects.
    Stores results in ctx.scored_grants as they complete so the guardrail
    can check accumulating scores across the batch.
    Called by app.py after Phase 4 completes.
    """
    tasks = [
        Runner.run(
            fit_scorer,
            build_scorer_input(gd, org_profile),
            context=ctx
        )
        for gd in grant_data_list
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    scored_grants = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Scoring failed for {grant_data_list[i].candidate.name}: {result}")
            continue
        scored = parse_scored_grant(result.final_output, grant_data_list[i])
        scored_grants.append(scored)
        ctx.scored_grants.append(scored)    # update context so guardrail sees accumulating scores

    # sort by score descending — rank 1 = best fit
    return sorted(scored_grants, key=lambda sg: sg.score, reverse=True)