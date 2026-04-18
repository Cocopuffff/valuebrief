import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from typing import Optional
from config import secrets
from models import ValuationModel
from logger import get_logger

logger = get_logger(__name__)

_pool = None

async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(conninfo=secrets.SUPABASE_URI, max_size=10, kwargs={"prepare_threshold": None}, open=False)
        await _pool.open()
    return _pool

async def upsert_valuation(model: ValuationModel) -> None:
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("""
                    INSERT INTO valuations (ticker, company, current_price, currency, base_revenue, shares_outstanding, expected_value, expected_cagr, dispersion_ratio, recommendation, thesis_data, valuation_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET
                        company = EXCLUDED.company,
                        current_price = EXCLUDED.current_price,
                        currency = EXCLUDED.currency,
                        base_revenue = EXCLUDED.base_revenue,
                        shares_outstanding = EXCLUDED.shares_outstanding,
                        expected_value = EXCLUDED.expected_value,
                        expected_cagr = EXCLUDED.expected_cagr,
                        dispersion_ratio = EXCLUDED.dispersion_ratio,
                        recommendation = EXCLUDED.recommendation,
                        thesis_data = EXCLUDED.thesis_data,
                        valuation_data = EXCLUDED.valuation_data,
                        updated_at = NOW()
                """, (
                    model.ticker, model.company, model.current_price, model.currency, 
                    model.base_revenue, model.shares_outstanding, model.expected_value, 
                    model.expected_cagr, model.dispersion_ratio, model.recommendation, 
                    Jsonb(model.thesis_data) if model.thesis_data else Jsonb({}), 
                    Jsonb(model.valuation_data) if model.valuation_data else Jsonb({})
                ))
    except Exception as e:
        ticker_val = model.ticker if model else "Unknown"
        logger.error(f"Failed to upsert valuation for ticker {ticker_val}: {e}")


async def get_valuation(ticker: str) -> Optional[ValuationModel]:
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute("""
                SELECT * FROM valuations WHERE ticker = %s
            """, (ticker,))
            row = await cur.fetchone()
            if not row:
                logger.info(f"No valuation found for ticker {ticker}")
                return None
                
            # Unpack full structural elements from our JSONB payloads
            val_data = row.get("valuation_data", {})
            if "scenarios" in val_data:
                row["scenarios"] = val_data["scenarios"]
                
            return ValuationModel(**row)
            