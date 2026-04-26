import sys
import os
import httpx
from dotenv import load_dotenv
import time

load_dotenv()

# Add src to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from provider import FinancialDataProvider

def test_mappings():
    test_cases = [
        ("MZH.SI", "MZH.SIN"),
        ("9988.HK", "9988.HKG"),
        ("RY.TO", "RY.TRT"),
        ("HSBA.L", "HSBA.LON"),
        ("MC.PA", "MC.PAR"),
        ("BMW.F", "BMW.FRK"),
        ("BHP.AX", "BHP.AUS"),
        ("AAPL", "AAPL"), # US stock, no dot
        ("BABA.MEX", "BABA.MEX"), # Unknown suffix, should remain unchanged
    ]

    all_passed = True
    print("Running exchange mapping tests...")
    print("-" * 40)
    for yahoo_ticker, expected_av_ticker in test_cases:
        actual = FinancialDataProvider.translate_ticker_for_av(yahoo_ticker)
        if actual == expected_av_ticker:
            print(f"✅ PASS: {yahoo_ticker} -> {actual}")
        else:
            print(f"❌ FAIL: {yahoo_ticker} -> {actual} (Expected: {expected_av_ticker})")
            all_passed = False

    print("-" * 40)
    if all_passed:
        print("All mapping tests passed successfully!")
    else:
        print("Some tests failed.")
        sys.exit(1)

def test_live_api():
    test_cases = ["MZH.SI", "9988.HK", "RY.TO"]
    av_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not av_api_key:
        print("\nALPHAVANTAGE_API_KEY not found in environment, skipping live API tests.")
        return

    print("\nRunning live AlphaVantage API tests...")
    print("-" * 40)
    all_passed = True
    for yahoo_ticker in test_cases:
        av_ticker = FinancialDataProvider.translate_ticker_for_av(yahoo_ticker)
        # Many international stocks don't have OVERVIEW data, but do have GLOBAL_QUOTE data.
        # We test GLOBAL_QUOTE to verify if the symbol is recognized at all.
        url_quote = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={av_ticker}&apikey={av_api_key}"
        try:
            resp = httpx.get(url_quote, timeout=10)
            data = resp.json()
            if "Global Quote" in data and "01. symbol" in data["Global Quote"]:
                print(f"✅ PASS API: {yahoo_ticker} -> {av_ticker} returned valid quote data (Price: {data['Global Quote'].get('05. price')})")
            elif "Information" in data and "standard API call frequency" in data.get("Information", ""):
                print(f"⚠️ RATE LIMIT: {data.get('Information')}")
                # We won't fail the whole test suite on rate limit, just break
                break
            elif not data:
                print(f"❌ FAIL API: {yahoo_ticker} -> {av_ticker} returned empty data {{}} (Ticker may be invalid on AlphaVantage)")
                all_passed = False
            else:
                print(f"❌ FAIL API: {yahoo_ticker} -> {av_ticker} returned {data}")
                all_passed = False
        except Exception as e:
            print(f"❌ FAIL API: {yahoo_ticker} -> {av_ticker} raised exception {e}")
            all_passed = False
            
        # AlphaVantage free tier limits to 1 req / second
        time.sleep(2)

    print("-" * 40)
    if not all_passed:
        print("Some live API tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    test_mappings()
    test_live_api()
