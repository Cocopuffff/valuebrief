import argparse
import asyncio
import json
import traceback
from datetime import date, datetime
from typing import Any
from dotenv import load_dotenv
load_dotenv()  # Must come before agent imports — they call init_chat_model at module level

from utils.logger import setup_logging, get_logger
from agents.orchestration import build_research_workflow
from agents.states import WorkflowState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
from psycopg_pool import AsyncConnectionPool
from utils.config import secrets
from utils.db import get_valuation
from schemas.base import validate_tickers

# Initialize logger
setup_logging()
logger = get_logger(__name__)

DB_URI = secrets.SUPABASE_URI


async def main():
    parser = argparse.ArgumentParser(description="Value Brief: Automated Daily Digest for Portfolio Tracking")
    parser.add_argument("--tickers", nargs="+", help="Tickers to track (overrides list in portfolio.json)")
    parser.add_argument("--portfolio", default="portfolio.json", help="Path to portfolio JSON file")

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
                run_dt = datetime.now().isoformat()  # captured once; shared by thread_id and artifact filename
                config: RunnableConfig = {
                    "configurable": {"thread_id": f"{ticker}-{run_dt}"}
                }
                logger.info(f"Starting agentic analysis for {ticker}...")
                # Check if valuation already exists
                existing_valuation = await get_valuation(ticker)
                if existing_valuation:
                    logger.info(f"Found existing valuation for {ticker}. Using existing valuation.")

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
                }

                async for event in workflow.astream_log(state, config, include_types=["llm", "tool"]):
                    for op in event.ops:
                        if op["path"] == "/streamed_output/-":
                            logger.info(f"Output: {op['value']}")
                        elif op.get("op") == "add" and "/logs/" in op.get("path", ""):
                            value = op.get("value", {})
                            if isinstance(value, dict) and "name" in value:
                                logger.debug(f"[Stream] {value['name']}")
            except Exception as e:
                logger.error(f"Error running analysis for {ticker}: {e}")
                logger.debug(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
