from typing import Any
from pydantic import BaseModel
from schemas import Asset, ValuationModel

def _model_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return dict(value.__dict__)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected model or dict, got {type(value).__name__}")


def _normalize_valuation(value: Any) -> ValuationModel | None:
    """Revalidate checkpoint valuation data after LangGraph msgpack reloads.

    Older checkpoints can deserialize Pydantic models with nested dictionaries
    instead of nested model instances. Revalidating here prevents downstream
    computed fields from seeing raw dicts for DCF scenarios.
    """
    if value is None:
        return None

    payload = _model_payload(value)
    for field_name, alias in (
        ("expected_value_snapshot", "expected_value"),
        ("expected_cagr_snapshot", "expected_cagr"),
        ("dispersion_ratio_snapshot", "dispersion_ratio"),
        ("recommendation_snapshot", "recommendation"),
    ):
        if field_name in payload:
            payload.setdefault(alias, payload.pop(field_name))

    # Computed fields appear in model dumps but are not accepted as inputs.
    payload.pop("scenario_margins", None)

    return ValuationModel.model_validate(payload)


def _normalize_asset(value: Any) -> Asset | None:
    if value is None:
        return None
    return Asset.model_validate(_model_payload(value))


async def _latest_checkpoint_thread_id(
    pool: AsyncConnectionPool[Any],
    ticker: str,
) -> str | None:
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT thread_id FROM checkpoints "
                "WHERE thread_id LIKE %s AND checkpoint_ns = '' "
                "ORDER BY checkpoint_id DESC LIMIT 1",
                (f"{ticker}-%",),
            )
            row = await cur.fetchone()
    return row[0] if row else None


def _checkpoint_model_updates(values: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if "valuation" in values:
        normalized_valuation = _normalize_valuation(values.get("valuation"))
        if normalized_valuation is not values.get("valuation"):
            updates["valuation"] = normalized_valuation

    if "price_data" in values:
        normalized_asset = _normalize_asset(values.get("price_data"))
        if normalized_asset is not values.get("price_data"):
            updates["price_data"] = normalized_asset

    return updates
