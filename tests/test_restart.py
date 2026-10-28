"""The Discord `restart` (and the update's "Restart now" button) must come back.

Live on 2026-10-28 it didn't: the command awaited `bot.close()` and only then
called os.execv — but closing the bot can end the event loop before control
returns from that await, so the exec never ran. asyncio.run() returned, the
"restart" flag bot.py checks wasn't set, and the process hung at exit
waiting for the console's input thread. The console's own `restart` had
already been fixed for exactly this: set the flag first, let bot.py exec.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogs import maintenance


class ClosingEndsTheLoop:
    """A bot whose close() never hands control back, as on the live box."""
    def __init__(self):
        self.voice_clients = []
        self._restart_requested = False
        self.closed = False

    async def close(self):
        self.closed = True
        raise asyncio.CancelledError     # the loop is torn down mid-await


@pytest.mark.asyncio
async def test_restart_is_requested_before_closing(monkeypatch):
    execs = []
    monkeypatch.setattr("os.execv", lambda *a: execs.append(a))
    bot = ClosingEndsTheLoop()
    cog = maintenance.MaintenanceCog(bot)
    with pytest.raises(asyncio.CancelledError):
        await cog.do_restart()
    assert bot.closed
    assert bot._restart_requested is True     # bot.py re-execs once the loop is gone
    assert execs == []                        # not from inside the dying loop


@pytest.mark.asyncio
async def test_voice_is_left_before_closing(monkeypatch):
    monkeypatch.setattr("os.execv", lambda *a: None)
    bot = ClosingEndsTheLoop()
    vc = MagicMock()
    vc.disconnect = AsyncMock()
    bot.voice_clients = [vc]
    with pytest.raises(asyncio.CancelledError):
        await maintenance.MaintenanceCog(bot).do_restart()
    vc.disconnect.assert_awaited_once()
