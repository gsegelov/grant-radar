---
title: GrantRadar
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# GrantRadar

AI-powered grant discovery and proposal drafting for nonprofits.

Enter your organization's website URL and GrantRadar will find matching grants, score them for fit, and draft proposal sections for the best opportunities.

## Setup

Add the following secrets in Space Settings → Variables and Secrets:
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY`
- `SEARCH_PROVIDER` (set to `tavily`)
- `MAX_GRANTS_TO_ANALYZE` (set to `5`)
- `RESEARCH_DEPTH` (set to `standard`)

## Stack

Built with the OpenAI Agents SDK, Gemini API, Tavily search, and Gradio.