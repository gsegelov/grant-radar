"""
pipeline/context.py

Shared state for the entire GrantRadar pipeline run.
One PipelineContext is created at the start of each run and passed as
context= to every Runner.run() call. Agents read from it via tools;
pipeline code in app.py reads and writes fields directly between phases.
"""

import uuid
from dataclasses import dataclass, field
from models.schemas import OrgProfile, GrantCandidate, GrantData, ScoredGrant


@dataclass
class PipelineContext:

    # ── USER INPUTS (set before pipeline starts) ──────────────────────────
    org_url: str                        # the URL the user submitted
    org_type: str                       # "nonprofit" | "for-profit" | "government"
    user_context: str = ""              # optional priorities, program names, requirements

    # ── PIPELINE CONFIG (set before pipeline starts) ──────────────────────
    max_grants_to_analyze: int = 15     # cap on how many grants Phase 4 analyzes
    research_depth: str = "standard"    # "quick" | "standard" | "deep"
    run_id: str = field(               # unique ID for this run — auto-generated
        default_factory=lambda: str(uuid.uuid4())[:8]   # first 8 chars is enough
    )

    # ── ACCUMULATED RESULTS (populated during pipeline run) ───────────────
    org_profile: OrgProfile | None = None               # set after Phase 2
    all_candidates: list[GrantCandidate] = field(default_factory=list)  # set after Phase 3
    all_grant_data: list[GrantData] = field(default_factory=list)       # set after Phase 4
    scored_grants: list[ScoredGrant] = field(default_factory=list)      # set after Phase 5
    selected_grants: list[ScoredGrant] = field(default_factory=list)    # set after HITL
    stage_log: list[dict] = field(default_factory=list)                 # running log of completed stages