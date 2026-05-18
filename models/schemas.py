"""
models/schemas.py

Single source of truth for all data models in GrantRadar.
Every agent imports its input/output types from here — never defined elsewhere.
"""

from pydantic import BaseModel, Field


# ── MODEL 1 of 9 ─────────────────────────────────────────────────────────
# Used by: URLValidator (the input guardrail that runs before anything else)
# Purpose: Did the URL pass inspection? If is_valid=False, the pipeline
#          stops immediately — no scraping, no API calls, no cost.

class UrlValidationResult(BaseModel):
    is_valid: bool          # True = URL looks like a real org site, proceed
    reason: str             # Plain-English explanation shown to the user if invalid


# ── MODEL 2 of 9 ─────────────────────────────────────────────────────────
# Produced by: WebScraper (Phase 1)
# Consumed by: OrgProfiler (Phase 2)
# Purpose: Everything we scraped from the org's website, organized by page.

class ScrapedSite(BaseModel):
    homepage_text: str                      # full text of the root URL — always required
    subpage_texts: dict[str, str] = {}      # key = URL, value = that page's text
                                            # e.g. {"/about": "We are a nonprofit..."}
                                            # default empty dict — some sites have no subpages
    nav_links: list[str] = []               # internal links found on the homepage
                                            # OrgProfiler uses this to understand site structure
                                            # without us having to re-fetch the page


# ── MODEL 3 of 9 ─────────────────────────────────────────────────────────
# Produced by: OrgProfiler (Phase 2)
# Consumed by: FitScorer, StrategicBriefWriter, ProposalDrafter + stored in PipelineContext
# Purpose: Synthesized profile of the org — this is what we score grants AGAINST.

class OrgProfile(BaseModel):
    name: str                               # org name from their website
    mission_summary: str                    # 2–3 sentence synthesis of what they do
    primary_sector: str                     # single main sector
    secondary_sectors: list[str] = []       # other sectors — optional
    geographic_scope: str = "Unknown"       # where they operate
    annual_budget_range: str = "Unknown"    # estimated from site signals
    grant_size_target: str = "Unknown"      # what award sizes make sense for them
    eligibility_tags: list[str]             # tags FitScorer matches against grant requirements
    inferred_programs: list[str] = []       # specific program names for use in drafting
    search_queries: list[str]               # 6–8 search strings GrantDiscoverer will execute


# ── MODEL 4 of 9 ─────────────────────────────────────────────────────────
# Produced by: GrantDiscoverer (Phase 3)
# Consumed by: GrantAnalyzer (Phase 4)
# Purpose: A grant opportunity found in search — lightweight, just enough to
#          know what to fetch next.

class GrantCandidate(BaseModel):
    name: str           # grant program name
    funder: str         # foundation or agency offering it
    url: str            # link to the full grant page
    description: str    # brief description from search results
    source_query: str   # which search query surfaces this result
                        # useful for debugging if discovery is thin


# ── MODEL 5 of 9 ─────────────────────────────────────────────────────────
# Produced by: GrantAnalyzer (Phase 4)
# Consumed by: FitScorer, ComplianceChecker
# Purpose: Deep analysis of one grant page — everything needed to score fit
#          and check compliance.

class GrantData(BaseModel):
    candidate: GrantCandidate                       # the original search result this is based on
    award_min: int = 0                              # minimum award in USD - 0 if not found
    award_max: int = 0                              # maximum award in USD - 0 if not found
    deadline: str = "unknown"                       # application deadline
    eligibility_requirements: list[str] = []        # stated eligibility criteria
    hard_disqualifiers: list[str] = []              # explicit exclusions from the funder
    required_attachments: list[str] = []            # documents required to apply
    funder_priorities: list[str] =[]                # what the funder cares about
    application_complexity: str = "medium"          # "low" | "medium" | "high"


# ── MODEL 6 of 9 ─────────────────────────────────────────────────────────
# Produced by: FitScorer (Phase 5)
# Consumed by: StrategicBriefWriter, ProposalDrafter, CSV export, HITL table
# Purpose: A grant scored against the org profile — includes the score,
#          reasoning, and whether it's worth applying to.

class ScoredGrant(BaseModel):
    grant: GrantData                    # full grant analysis this score is based on
    score: int = Field(ge=0, le=100)    # fit score - ge=0 le=100 means Pydantic rejects anything out of that range automatically
    rationale: str                      # 2-3 sentence explanation for the grant writer
    top_alignment_points: list[str]     # specific reasons this grant fits the org
    gaps: list[str] = []                # known weaknesses - empty if none
    disqualified: bool = False          # True if a hard disqualifier applies
    disqualifiers: list[str] =[]        # which disqualifiers triggered - empty if clean
    recommended: bool = False           # True if score is strong AND not disqualified


# ── MODEL 7 of 9 ─────────────────────────────────────────────────────────
# Produced by: StrategicBriefWriter (Phase 6)
# Consumed by: ProposalDrafter, DOCX export
# Purpose: Strategic advice for one grant — how to frame the application
#          before any drafting begins.

class GrantBrief(BaseModel):
    grant_name: str                             # matches GrantCandidate.name
    recommended_framing: str                    # how to position the org for this funder
    priorities_to_emphasize: list[str]          # funder priorities the application should lead with
    sections_to_strengthen: list[str] = []      # areas needing extra attention for this funder
    narrative_hook: str                         # suggested opening angle for the application


# ── MODEL 8 of 9 ─────────────────────────────────────────────────────────
# Produced by: ProposalDrafter (Phase 7)
# Consumed by: ComplianceChecker, DOCX export
# Purpose: Four standard proposal sections for one grant application.

class ProposalDraft(BaseModel):
    grant_name: str                         # matches GrantCandidate.name
    organizational_capacity: str            # org's track record and ability to execute
    problem_statement: str                  # the problem the org addresses and why it matters
    project_description: str                # what the org will do with the grant funds
    evaluation_plan: str                    # how outcomes will be measured and reported
    needs_manual_completion: bool = False   # NOT set by the agent - set by the pipeline
                                            # if DraftQualityValidator trips twice in a row- triggers a warning banner in the DOCX output


# ── MODEL 9 of 9 ─────────────────────────────────────────────────────────
# Produced by: ComplianceChecker (Phase 8)
# Consumed by: DOCX export
# Purpose: Checklist of what the draft covers, what it misses, and what to
#          fix before submitting.

class ComplianceReport(BaseModel):
    grant_name: str                         # matches GrantCandidate.name
    requirements_addressed: list[str] = []  # eligibility requirements the draft covers
    requirements_missing: list[str] = []    # requirements with no coverage in the draft
    requirements_weak: list[str] = []       # requierments mentioned but not convincingly
    missing_attachments: list[str] = []     # required documents not referenced in draft
    overall_readiness: str = "Needs Work"   # "Strong" | "Needs Work" | "Incomplete"
    priority_fixes: list[str]               # top 3 fixes before submission - required (no default because report is useless without it)

