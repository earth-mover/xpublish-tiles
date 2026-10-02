import logging

import pytest

from xpublish_tiles.lib import async_run
from xpublish_tiles.logger import log_duration, setup_logging, with_accumulated_logs


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
