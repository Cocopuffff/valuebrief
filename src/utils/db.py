import asyncio
from collections.abc import Awaitable, Callable
from typing import Optional, Any, TypeVar

import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from schemas import ValuationModel
from utils.config import secrets
from utils.logger import get_logger

logger = get_logger(__name__)

_T = TypeVar("_T")

_TRANSIENT_SQLSTATES = {
    "08000",  # connection_exception
    "08003",  # connection_does_not_exist
    "08006",  # connection_failure
    "08007",  # transaction_resolution_unknown
    "57P01",  # admin_shutdown
    "57P02",  # crash_shutdown
    "57P03",  # cannot_connect_now
}
_TRANSIENT_MESSAGE_FRAGMENTS = (
    "connection is lost",
    "terminating connection due to administrator command",
    "server closed the connection unexpectedly",
    "ssl syscall error: eof detected",
)


class _DbPoolManager:
    def __init__(self) -> None:
        self._pool: AsyncConnectionPool[Any] | None = None

    async def get_pool(self) -> AsyncConnectionPool[Any]:
        if self._pool is None or self._pool.closed:
            self._pool = AsyncConnectionPool(
                conninfo=secrets.SUPABASE_URI,
                min_size=0,
                max_size=10,
                max_idle=300,
                kwargs={"prepare_threshold": None},
                check=AsyncConnectionPool.check_connection,
                open=False,
            )
            await self._pool.open()
        return self._pool

    async def close_pool(self) -> None:
        if self._pool is not None and not self._pool.closed:
            await self._pool.close()
        self._pool = None


_pool_manager = _DbPoolManager()


async def get_pool() -> AsyncConnectionPool[Any]:
    return await _pool_manager.get_pool()


async def close_pool() -> None:
    """Close the shared application pool explicitly during shutdown."""
    await _pool_manager.close_pool()


def valuation_outbox_payload(model: ValuationModel) -> dict[str, Any]:
    """Return the database-column payload needed to replay a valuation upsert."""
    return {
        "ticker": model.ticker,
        "company": model.company,
        "current_price": model.current_price,
        "currency": model.currency,
        "base_revenue": model.base_revenue,
        "shares_outstanding": model.shares_outstanding,
        "expected_value": model.expected_value,
        "expected_cagr": model.expected_cagr,
        "dispersion_ratio": model.dispersion_ratio,
        "recommendation": model.recommendation,
        "thesis_data": model.thesis_data or {},
        "valuation_data": model.valuation_data or {},
    }


async def _execute_valuation_upsert(payload: dict[str, Any]) -> None:
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
                payload["ticker"],
                payload["company"],
                payload["current_price"],
                payload["currency"],
                payload["base_revenue"],
                payload["shares_outstanding"],
                payload["expected_value"],
                payload["expected_cagr"],
                payload["dispersion_ratio"],
                payload["recommendation"],
                Jsonb(payload.get("thesis_data") or {}),
                Jsonb(payload.get("valuation_data") or {}),
            ))


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_db_error(exc: BaseException) -> bool:
    """Return True for lost/admin-terminated PostgreSQL connections."""
    for current in _iter_exception_chain(exc):
        sqlstate = getattr(current, "sqlstate", None)
        if sqlstate in _TRANSIENT_SQLSTATES:
            return True
        if isinstance(current, (psycopg.OperationalError, psycopg.InterfaceError)):
            return True
        message = str(current).lower()
        if any(fragment in message for fragment in _TRANSIENT_MESSAGE_FRAGMENTS):
            return True
    return False


async def run_db_operation(
    operation: Callable[[], Awaitable[_T]],
    *,
    operation_name: str,
    attempts: int = 2,
) -> _T:
    """Run a DB operation, retrying once when a pooled connection was killed."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as e:
            last_error = e
            if attempt >= attempts or not is_transient_db_error(e):
                raise
            logger.warning(
                "[DB] %s failed on a transient connection error; retrying "
                "with a fresh pooled connection (%d/%d): %s",
                operation_name,
                attempt + 1,
                attempts,
                e,
            )
            await asyncio.sleep(0.2 * attempt)

    raise RuntimeError(f"{operation_name} failed") from last_error


async def upsert_valuation(model: ValuationModel) -> None:
    payload = valuation_outbox_payload(model)
    try:
        await run_db_operation(
            lambda: _execute_valuation_upsert(payload),
            operation_name="upsert valuation",
        )
    except Exception as e:
        ticker_val = model.ticker if model else "Unknown"
        if is_transient_db_error(e):
            from utils.db_outbox import append_db_outbox_entry

            await append_db_outbox_entry(
                operation="upsert",
                table="valuations",
                ticker=ticker_val,
                payload=payload,
                error=str(e),
            )
            logger.warning(
                "Queued valuation upsert for %s in local DB outbox after transient failure: %s",
                ticker_val,
                e,
            )
            return
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
