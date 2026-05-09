import argparse
import asyncio
import json
import traceback
from datetime import date, datetime
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # Must come before agent imports — they call init_chat_model at module level

from utils.logger import setup_logging, get_logger
from agents.orchestration import build_research_workflow
from agents.states import WorkflowState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from psycopg_pool import AsyncConnectionPool
from utils.config import secrets
from utils.db import get_valuation
from schemas.base import validate_tickers
from utils.checkpoint_helpers import _latest_checkpoint_thread_id, _checkpoint_model_updates

# Initialize logger
setup_logging()
logger = get_logger(__name__)

DB_URI = secrets.SUPABASE_URI


async def _stream_workflow(
    workflow: Any,
    graph_input: Any,
    config: RunnableConfig,
) -> None:
    async for event in workflow.astream_log(
        graph_input,
        config,
        include_types=["llm", "tool"],
    ):
        for op in event.ops:
            if op["path"] == "/streamed_output/-":
                logger.info(f"Output: {op['value']}")
            elif op.get("op") == "add" and "/logs/" in op.get("path", ""):
                value = op.get("value", {})
                if isinstance(value, dict) and "name" in value:
                    logger.debug(f"[Stream] {value['name']}")

async def main():
    parser = argparse.ArgumentParser(description="Value Brief: Automated Daily Digest for Portfolio Tracking")
    parser.add_argument("--tickers", nargs="+", help="Tickers to track (overrides list in portfolio.json)")
    parser.add_argument("--portfolio", default="portfolio.json", help="Path to portfolio JSON file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each ticker from its latest checkpoint instead of starting a fresh run",
    )

    args = parser.parse_args()

    # Load tickers from JSON or arguments
    raw_tickers: list[str] = []
    if args.tickers:
        raw_tickers = args.tickers
    else:
        try:
            with open(args.portfolio, "r") as f:
                data = json.load(f)
                raw_tickers = data.get("tickers", [])
        except FileNotFoundError:
            logger.error(f"{args.portfolio} not found. Please provide tickers via --tickers or create portfolio.json.")
            return

    if not isinstance(raw_tickers, list):
        logger.error("Portfolio 'tickers' must be a list.")
        return

    if not raw_tickers:
        logger.warning("No tickers to track.")
        return

    try:
        tickers = validate_tickers(raw_tickers)
    except ValueError as e:
        logger.error(str(e))
        return


    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": None,
    }
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs=connection_kwargs,
    ) as pool:
        pool: AsyncConnectionPool[Any]
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        workflow = build_research_workflow(checkpointer=checkpointer)

        for ticker in tickers:
            try:
                logger.info(f"Starting agentic analysis for {ticker}...")

                if args.resume:
                    thread_id = await _latest_checkpoint_thread_id(pool, ticker)
                    if not thread_id:
                        logger.error(
                            "No checkpoint found for %s. Cannot resume.", ticker
                        )
                        continue

                    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
                    saved = await workflow.aget_state(config)
                    if not saved or not saved.values:
                        logger.error(
                            "Checkpoint state empty for %s (thread_id=%s). Cannot resume.",
                            ticker, thread_id,
                        )
                        continue

                    if not saved.next:
                        logger.info(
                            "Latest checkpoint for %s is already complete (thread_id=%s).",
                            ticker, thread_id,
                        )
                        continue

                    logger.info(
                        "Resuming %s from checkpoint (thread_id=%s, step=%s)",
                        ticker, thread_id, saved.next or "(end)",
                    )
                    model_updates = _checkpoint_model_updates(saved.values)
                    resume_input = (
                        Command(update=model_updates) if model_updates else None
                    )
                    await _stream_workflow(workflow, resume_input, config)
                else:
                    # ── Fresh run ──
                    existing_valuation = await get_valuation(ticker)
                    if existing_valuation:
                        logger.info(
                            f"Found existing valuation for {ticker}. "
                            "Using existing valuation."
                        )

                    run_dt = datetime.now().isoformat()
                    config: RunnableConfig = {
                        "configurable": {"thread_id": f"{ticker}-{run_dt}"}
                    }
                    state: WorkflowState = {
                        "date": date.today().isoformat(),
                        "run_datetime": run_dt,
                        "company": "",
                        "ticker": ticker,
                        "price_data": None,
                        "bull_thesis": "",
                        "bear_thesis": "",
                        "sources": [],
                        "judge_decision": "",
                        "valuation": existing_valuation,
                        "final_report": "",
                        # Hybrid RAG fields
                        "citation_manifest": [],
                        "curator_log": "",
                        "active_memory_ids": [],
                        "vault_artifacts": [],
                        "rag_context": "",
                        "retrieved_memory_ids": [],
                        "research_topics": [],
                        "retrieved_memory_outcomes": {},
                        "thesis_pillars": [],
                        "pillar_outcomes": [],
                    }
                    await _stream_workflow(workflow, state, config)
            except Exception as e:
                logger.error(f"Error running analysis for {ticker}: {e}")
                logger.debug(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
