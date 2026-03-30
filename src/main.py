import argparse
import asyncio
import json
import traceback
from datetime import date
from dotenv import load_dotenv
load_dotenv()  # Must come before agent imports — they call init_chat_model at module level

from agents.orchestration import workflow
from agents.states import WorkflowState

async def main():
    parser = argparse.ArgumentParser(description="Value Brief: Automated Daily Digest for Portfolio Tracking")
    parser.add_argument("--tickers", nargs="+", help="Tickers to track (overrides list in portfolio.json)")
    parser.add_argument("--portfolio", default="portfolio.json", help="Path to portfolio JSON file")
    
    args = parser.parse_args()

    # Load tickers from JSON or arguments
    tickers = []
    if args.tickers:
        tickers = args.tickers
    else:
        try:
            with open(args.portfolio, "r") as f:
                data = json.load(f)
                tickers = data.get("tickers", [])
        except FileNotFoundError:
            print(f"Error: {args.portfolio} not found. Please provide tickers via --tickers or create portfolio.json.")
            return

    if not tickers:
        print("No tickers to track.")
        return
    
    for ticker in tickers:
        try:
            print(f"Starting agentic analysis for {ticker}...")
            state: WorkflowState = {
                "date": date.today().isoformat(),
                "company": "",
                "ticker": ticker,
                "price_data": None,
                "bull_thesis": "",
                "bear_thesis": "",
                'bull_sources': [],
                "bear_sources": [],
                "judge_decision": "",
                "final_report": "",
            }
            # Use astream_log to stream events as the workflow runs
            async for event in workflow.astream_log(state, include_types=["llm", "tool"]):
                for op in event.ops:
                    if op["path"] == "/streamed_output/-":
                        print("Output:", op["value"])
                    elif op.get("op") == "add" and "/logs/" in op.get("path", ""):
                        # Only log actual node starts, not random text
                        value = op.get("value", {})
                        if isinstance(value, dict) and "name" in value:
                            print(f"[Stream] {value['name']}")
        except Exception as e:
            print(f"Error running analysis for {ticker}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())