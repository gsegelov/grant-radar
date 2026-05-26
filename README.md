# GrantRadar
## AI-powered grant discovery and proposal drafting for nonprofits

> **Tech stack:** OpenAI Agents SDK · Python · Gemini 2.5 Pro/Flash · Gradio · Hugging Face Spaces
> **Live demo:** [huggingface.co/spaces/geliach/grant-radar](https://huggingface.co/spaces/geliach/grant-radar)

---

## What This Is

Nonprofits spend 2–3 days per funding cycle on work that is almost entirely systematic — searching grant databases, reading eligibility criteria, scoring fit against their mission, and drafting boilerplate proposal sections. GrantRadar automates all of it. Given only an organization's website URL, an 8-agent AI pipeline discovers relevant grants, scores each one for fit, and delivers a ready-to-edit Word document containing a strategic application brief, a first-pass proposal draft, and a compliance checklist — in under 10 minutes.

The output is not a summary or a list of links. It is a complete working document a grant writer can open in Word and start editing immediately.

---

## Live Demo

**[→ Try GrantRadar on Hugging Face Spaces](https://huggingface.co/spaces/geliach/grant-radar)**

Enter any nonprofit's website URL. The pipeline scrapes the site, infers the organization's mission and eligibility profile, runs parallel grant searches across multiple angles, scores each result, and produces a downloadable CSV + DOCX.

> **Note:** Runs on HF Spaces free CPU tier. First run for a new org takes 5–8 minutes.
> Use Research Depth: **quick** and Max Grants: **3** for fastest results.

---

## Architecture

```
INPUT: org URL + org type + optional priorities
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 — Sequential                      │
│  WebScraper → fetches homepage + up to 4 subpages           │
│  [Input guardrail: blocks social media / unreachable URLs]  │
└────────────────────────┬────────────────────────────────────┘
                         │ ScrapedSite
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2 — Sequential                      │
│  OrgProfiler → synthesizes mission, eligibility tags,       │
│                generates 6–8 targeted search queries        │
└────────────────────────┬────────────────────────────────────┘
                         │ OrgProfile
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASES 3–5 — Parallel (asyncio.gather)         │
│  GrantDiscoverer × N  → executes each search query          │
│  GrantAnalyzer × N    → deep-reads each grant page          │
│  FitScorer × N        → scores each grant 0–100             │
└────────────────────────┬────────────────────────────────────┘
                         │ list[ScoredGrant], sorted by score
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  PHASES 6–8 — Sequential                     │
│  StrategicBriefWriter → application angle per top grant     │
│  ProposalDrafter      → 4 proposal sections per grant       │
│  ComplianceChecker    → flags missing requirements          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
OUTPUT: grant_opportunities.csv  +  full_report.docx
```

**Pattern:** Deterministic Pipeline — Python controls all phase transitions. No orchestrator agent decides what runs next. Parallel execution in Phases 3–5 reduces a 15-grant analysis from ~2 minutes sequential to ~12 seconds.

---

## Agent Roster

| Agent | Model | Role |
|---|---|---|
| WebScraper | Gemini 2.5 Flash | Fetches homepage + subpages via `fetch_page()` |
| OrgProfiler | Gemini 2.5 Pro | Synthesizes site text → structured org profile + search queries |
| GrantDiscoverer | Gemini 2.5 Flash | Executes one search query → list of GrantCandidate objects |
| GrantAnalyzer | Gemini 2.5 Flash | Fetches grant page → extracts eligibility, deadline, award range |
| FitScorer | Gemini 2.5 Pro | Scores one grant against org profile → 0–100 with written rationale |
| StrategicBriefWriter | Gemini 2.5 Pro | Writes application angle brief per top grant |
| ProposalDrafter | Gemini 2.5 Pro | Drafts 4 proposal sections tailored to each funder's priorities |
| ComplianceChecker | Gemini 2.5 Pro | Verifies draft against stated grant requirements |

Flash for extraction work. Pro for reasoning and writing. This split cuts API costs by ~60% versus using Pro throughout.

---

## Output Example

For `habitat.org` with context "grants 50k–500k":

**CSV:** 3 grants ranked by fit score with funder, award range, deadline, rationale, alignment points, and gaps.

**DOCX (42KB):** For each drafted grant:
- **Strategic Brief** — recommended framing, funder priorities to emphasize, narrative hook
- **Proposal Draft** — Organizational Capacity, Problem Statement, Project Description, Evaluation Plan — each section tailored to that funder's stated priorities
- **Compliance Report** — requirements addressed, requirements missing, top 3 priority fixes

---

## Shared State Pattern

All 8 agents share a single `PipelineContext` dataclass passed via `RunContextWrapper`. Every agent has access to the original org URL, the profiler's inferred eligibility tags, and the user's stated priorities without re-requesting that information.

```python
@dataclass
class PipelineContext:
    org_url: str
    org_type: str
    user_context: str = ""
    org_profile: OrgProfile | None = None        # set after Phase 2
    all_candidates: list[GrantCandidate] = ...   # set after Phase 3
    all_grant_data: list[GrantData] = ...        # set after Phase 4
    scored_grants: list[ScoredGrant] = ...       # set after Phase 5
```

---

## Pydantic Data Contracts

Every phase-to-phase handoff is typed. No dict passing between agents.

```
ScrapedSite → OrgProfile → GrantCandidate → GrantData → ScoredGrant → GrantBrief → ProposalDraft → ComplianceReport
```

Any agent can be tested in isolation with known input objects, and any parsing failure surfaces immediately rather than propagating silently.

---

## Setup

```bash
git clone https://github.com/gsegelov/grant-radar.git
cd grant-radar
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your keys to .env
python app.py
```

Open `http://127.0.0.1:7860`.

**Required `.env` variables:**
```bash
GOOGLE_API_KEY=your_gemini_key       # Gemini 2.5 Pro + Flash via Google AI Studio
TAVILY_API_KEY=your_tavily_key       # 1,000 free searches/month
SEARCH_PROVIDER=tavily               # tavily | duckduckgo (free fallback, no key needed)
```

---

## Technical Decisions

| Decision | Choice | Why |
|---|---|---|
| LLM | Gemini 2.5 Pro/Flash | OpenAI-compatible endpoint means any provider is swappable in one config line |
| Search | Tavily + DuckDuckGo fallback | Tavily returns LLM-ready summaries; DuckDuckGo as free zero-config fallback |
| Pipeline pattern | Deterministic Python sequencing | Phase order is always identical; easier to debug, replay, and audit than LLM routing |
| Parallelism | `asyncio.gather()` in Phases 3–5 | 15 grants sequential ≈ 2 min; parallel ≈ 12 sec |
| Output format | DOCX | Grant writers need to edit the draft directly; PDF is read-only |
| Folder name | `pipeline/` not `agents/` | `agents/` collides with the `openai-agents` package import at runtime |

---

## Stack

- **Agent framework:** OpenAI Agents SDK (`openai-agents`)
- **LLM:** Gemini 2.5 Pro (reasoning/writing) + Gemini 2.5 Flash (extraction/retrieval)
- **LLM routing:** OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`)
- **Search:** Tavily API (primary) · DuckDuckGo (fallback)
- **UI:** Gradio
- **Deployment:** Hugging Face Spaces (CPU Basic, free tier)
- **Data contracts:** Pydantic v2
- **Async:** Python asyncio + threading

---

> **LLM portability:** Built with Gemini via the OpenAI-compatible endpoint.
> Any OpenAI-compatible model can be swapped in by changing `base_url` and `api_key` in `app.py`.