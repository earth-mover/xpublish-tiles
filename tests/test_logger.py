import asyncio
import logging
import time

import pytest

from xpublish_tiles import lib, telemetry
from xpublish_tiles.lib import async_run
from xpublish_tiles.logger import (
    log_duration,
    record_wait,
    setup_logging,
    with_accumulated_logs,
)


@with_accumulated_logs(log_message_fn=lambda: "tile 4/4/10")
async def _endpoint():
    with log_duration("async_load data subsets"):
        pass
    await async_run(_render)


def _render():
    with log_duration("render quadmesh"):
        pass


@pytest.fixture
def restore_logging():
    yield
    setup_logging(logging.INFO)


async def test_info_level_emits_one_summary_line(capsys, restore_logging):
    setup_logging(logging.INFO)
    await _endpoint()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    (line,) = lines
    assert line.startswith("🔧 tile 4/4/10 (total: ")
    assert "ms async_load data subsets" in line
    assert "ms render quadmesh" in line


async def test_debug_level_keeps_detail_and_adds_summary(capsys, restore_logging):
    setup_logging(logging.DEBUG)
    await _endpoint()
    lines = capsys.readouterr().out.strip().splitlines()
    assert "ms render quadmesh" in lines[0]
    assert any("render quadmesh" in line for line in lines[1:])


async def test_warning_level_emits_nothing(capsys, restore_logging):
    setup_logging(logging.WARNING)
    await _endpoint()
    assert capsys.readouterr().out == ""


@with_accumulated_logs(log_message_fn=lambda: "keyed")
async def _keyed_endpoint():
    with log_duration("async_load data subsets", key="load"):
        time.sleep(0.01)
    await async_run(_keyed_render)
    await async_run(_keyed_render)
    return "tile"


def _keyed_render():
    with log_duration("render (3, 5, 5) quadmesh", key="render"):
        time.sleep(0.01)


async def test_root_span_gets_summed_stage_times(fake_span):
    assert await _keyed_endpoint() == "tile"
    m = fake_span.metrics
    assert m.keys() == {
        "tiles.stage_ms.load",
        "tiles.stage_ms.render",
        "tiles.wait_ms.thread_pool",
        "tiles.total_ms",
    }
    assert m["tiles.stage_ms.load"] >= 10
    # two renders on pool threads add up under one key
    assert m["tiles.stage_ms.render"] >= 20
    assert m["tiles.wait_ms.thread_pool"] >= 0
    assert m["tiles.total_ms"] >= m["tiles.stage_ms.load"] + m["tiles.stage_ms.render"]
    assert fake_span.tags == {"tiles.status": "ok"}


async def test_thread_pool_wait_is_measured(fake_span, monkeypatch):
    # one pool slot: the second task waits for the first one
    monkeypatch.setitem(lib._semaphores, asyncio.get_running_loop(), asyncio.Semaphore(1))

    @with_accumulated_logs(log_message_fn=lambda: "contended")
    async def endpoint():
        await asyncio.gather(async_run(time.sleep, 0.05), async_run(time.sleep, 0.05))

    await endpoint()
    assert fake_span.metrics["tiles.wait_ms.thread_pool"] >= 40


async def test_failed_request_is_tagged_error(fake_span):
    error = ValueError("boom")

    @with_accumulated_logs(log_message_fn=lambda: "boom")
    async def endpoint():
        with log_duration("render", key="render"):
            raise error

    with pytest.raises(ValueError) as excinfo:
        await endpoint()
    assert excinfo.value is error
    assert fake_span.tags == {"tiles.status": "error"}
    assert "tiles.stage_ms.render" in fake_span.metrics


async def test_cancelled_request_records_open_stage(fake_span):
    started = asyncio.Event()

    @with_accumulated_logs(log_message_fn=lambda: "slow")
    async def endpoint():
        with log_duration("async_load data subsets", key="load"):
            started.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(endpoint())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_span.tags == {"tiles.status": "cancelled"}
    assert fake_span.metrics["tiles.total_ms"] >= 0
    assert fake_span.metrics["tiles.stage_ms.load"] >= 0


@log_duration("select subsets", key="select")
async def _decorated_select(delay: float) -> str:
    await asyncio.sleep(delay)
    return "subsets"


@log_duration("coarsen", key="coarsen")
def _decorated_coarsen() -> str:
    time.sleep(0.02)
    return "coarse"


async def test_decorated_async_function_records_its_stage(fake_span):
    @with_accumulated_logs(log_message_fn=lambda: "decorated")
    async def endpoint():
        return await _decorated_select(0.02)

    assert await endpoint() == "subsets"
    assert fake_span.metrics["tiles.stage_ms.select"] >= 20


async def test_concurrent_decorated_calls_each_record(fake_span):
    async def late_select():
        await asyncio.sleep(0.05)
        return await _decorated_select(0.02)

    @with_accumulated_logs(log_message_fn=lambda: "concurrent")
    async def endpoint():
        return await asyncio.gather(_decorated_select(0.1), late_select())

    assert await endpoint() == ["subsets", "subsets"]
    # 100 + 20 ms; a start time shared by both calls gives about 50 + 20
    assert fake_span.metrics["tiles.stage_ms.select"] >= 110


async def test_decorated_sync_function_records_its_stage(fake_span):
    @with_accumulated_logs(log_message_fn=lambda: "sync")
    async def endpoint():
        return await async_run(_decorated_coarsen)

    assert await endpoint() == "coarse"
    assert fake_span.metrics["tiles.stage_ms.coarsen"] >= 20


async def test_decorated_function_records_and_reraises(fake_span):
    error = ValueError("boom")

    @log_duration("select subsets", key="select")
    async def failing():
        await asyncio.sleep(0.02)
        raise error

    @with_accumulated_logs(log_message_fn=lambda: "fails")
    async def endpoint():
        await failing()

    with pytest.raises(ValueError) as excinfo:
        await endpoint()
    assert excinfo.value is error
    assert fake_span.metrics["tiles.stage_ms.select"] >= 20


async def test_no_root_span_is_a_noop(monkeypatch):
    monkeypatch.setattr(telemetry, "root_span", lambda: None)
    assert await _keyed_endpoint() == "tile"


async def test_raising_set_metric_does_not_change_response(monkeypatch):
    class BrokenSpan:
        def set_metric(self, key: str, value: float) -> None:
            raise RuntimeError("telemetry bug")

        def set_tag(self, key: str, value: str) -> None:
            raise RuntimeError("telemetry bug")

    monkeypatch.setattr(telemetry, "root_span", BrokenSpan)
    assert await _keyed_endpoint() == "tile"


def _raise_name() -> str:
    raise RuntimeError("name bug")


async def test_raising_log_message_fn_is_ignored(fake_span, capsys, restore_logging):
    setup_logging(logging.INFO)

    @with_accumulated_logs(log_message_fn=_raise_name)
    async def named():
        with log_duration("render", key="render"):
            pass
        return "tile"

    assert await named() == "tile"
    assert capsys.readouterr().out.startswith("🔧 named (total: ")
    assert fake_span.tags == {"tiles.status": "ok"}


async def test_raising_log_message_fn_keeps_original_error(fake_span, restore_logging):
    setup_logging(logging.INFO)
    error = ValueError("boom")

    @with_accumulated_logs(log_message_fn=_raise_name)
    async def endpoint():
        with log_duration("render", key="render"):
            raise error

    with pytest.raises(ValueError) as excinfo:
        await endpoint()
    assert excinfo.value is error


async def test_log_message_fn_runs_once(fake_span, restore_logging):
    setup_logging(logging.INFO)
    calls: list[None] = []

    @with_accumulated_logs(log_message_fn=lambda: calls.append(None) or "once")
    async def endpoint():
        with log_duration("render", key="render"):
            pass

    await endpoint()
    assert len(calls) <= 1


def test_no_request_context_is_a_noop():
    with log_duration("outside", key="render"):
        pass
    record_wait("thread_pool", 1.0)
