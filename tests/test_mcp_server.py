"""Tests for halvoice.mcp_server on MCP SDK 2.x; no audio device is touched."""

import asyncio
import os
import sys

import pytest

from halvoice import mcp_server, tts

STARTUP_TIMEOUT_S = 60


def test_server_lists_its_tools():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    assert {t.name for t in tools} == {"speak", "listen", "set_voice_speed"}


def test_listen_does_not_touch_environment(monkeypatch):
    seen = {}

    def fake_listen_once(on_first_voice=None, max_utterance_s=None):
        seen["cap"] = max_utterance_s
        return "hello"

    monkeypatch.setattr(mcp_server, "listen_once", fake_listen_once)
    before = dict(os.environ)
    assert mcp_server.listen(seconds=7) == "hello"
    assert dict(os.environ) == before
    assert seen["cap"] == 7


def test_set_voice_speed_changes_the_client_not_the_environment(monkeypatch):
    class FakeClient:
        speed = 0.92

    client = FakeClient()
    monkeypatch.setattr(tts, "_shared_client", lambda: client)
    before = dict(os.environ)
    assert "1.20" in mcp_server.set_voice_speed(5.0)
    assert client.speed == 1.2
    assert dict(os.environ) == before


@pytest.mark.integration
def test_voice_mcp_lists_tools():
    """`hal-voice mcp` starts over stdio and lists its tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def list_names():
        params = StdioServerParameters(command=sys.executable, args=["-m", "halvoice.cli", "mcp"])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return {t.name for t in (await session.list_tools()).tools}

    names = asyncio.run(asyncio.wait_for(list_names(), STARTUP_TIMEOUT_S))
    assert {"speak", "listen", "set_voice_speed"} <= names
