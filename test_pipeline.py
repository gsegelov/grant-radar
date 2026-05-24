"""test_pipeline.py — diagnose the hang without Gradio in the way"""
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled

load_dotenv()
gemini_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60.0
)
set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

from app import run_pipeline_phase1_to_5

async def main():
    print("Starting pipeline test — no Gradio...")
    ctx, scored_grants = await run_pipeline_phase1_to_5(
        org_url="https://www.habitat.org",  # known real nonprofit
        org_type="nonprofit",
        user_context="Texas housing, grants under $100K",
        max_grants=3,            # ← keep this small during testing
        research_depth="quick",
        progress_callback=print  # ← print to terminal so you see every step
    )
    print(f"\nDone — {len(scored_grants)} grants scored")
    for sg in scored_grants:
        print(f"  {sg.score} | {sg.grant.candidate.name}")

asyncio.run(main())