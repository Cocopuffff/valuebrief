import asyncio

import pytest

from utils.db import is_transient_db_error, run_db_operation


class AdminShutdownError(Exception):
    sqlstate = "57P01"


def test_is_transient_db_error_detects_admin_shutdown_sqlstate():
    assert is_transient_db_error(AdminShutdownError("terminating connection"))


def test_run_db_operation_retries_transient_connection_error_once():
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AdminShutdownError("terminating connection due to administrator command")
        return "ok"

    result = asyncio.run(run_db_operation(operation, operation_name="test operation"))

    assert result == "ok"
    assert calls == 2


def test_run_db_operation_does_not_retry_non_transient_error():
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("bad query shape")

    with pytest.raises(ValueError):
        asyncio.run(run_db_operation(operation, operation_name="test operation"))

    assert calls == 1
