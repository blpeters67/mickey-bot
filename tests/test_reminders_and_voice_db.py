"""Database behaviour for repeating reminders and voice-time tracking.

Both features live or die on their edge cases — a reminder that replays a
day's backlog after a restart, or a crash that banks a weekend of phantom
voice time — so those are what's tested here rather than the happy path alone.
"""
from __future__ import annotations

import asyncio
import functools
import time

import aiosqlite
import pytest

from core import db, recurrence


def run(coro):
    return asyncio.run(coro)


def asynctest(fn):
    """Run an async test body with asyncio.run, matching the house style in
    test_db_features.py — no pytest-asyncio dependency or mode config."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


@pytest.fixture(autouse=True)
def _clean_db():
    """Fresh tables per test — these features are all about accumulated state,
    so leakage between tests would make failures meaningless."""
    async def _reset():
        await db.init_db()
        async with aiosqlite.connect(db.settings.db_path) as conn:
            for table in ("reminders", "user_settings", "voice_time",
                          "voice_open_sessions", "voice_stats_optout"):
                await conn.execute(f"DELETE FROM {table}")
            await conn.commit()
    run(_reset())


# ── reminders ────────────────────────────────────────────────────────────────

async def _add(message="test", due_in=-1, schedule=None, user_id=7):
    return await db.add_reminder(
        guild_id=1, channel_id=2, user_id=user_id, message=message,
        due_at=time.time() + due_in,
        repeat_kind=schedule.kind if schedule else None,
        repeat_spec=schedule.spec if schedule else None,
    )


@asynctest
async def test_one_shot_reminder_is_deleted_after_firing():
    await _add("once")
    fired = await db.pop_due_reminders(time.time())
    assert [r["message"] for r in fired] == ["once"]
    assert fired[0]["next_due"] is None
    assert await db.list_reminders(7) == []


@asynctest
async def test_repeating_reminder_survives_and_is_rescheduled():
    sched = recurrence.Schedule("interval", "900")
    rid = await _add("repeat", schedule=sched)

    fired = await db.pop_due_reminders(time.time())
    assert [r["message"] for r in fired] == ["repeat"]
    assert fired[0]["next_due"] > time.time()

    rows = await db.list_reminders(7)
    assert len(rows) == 1
    assert rows[0]["id"] == rid
    assert rows[0]["fire_count"] == 1
    assert rows[0]["due_at"] > time.time()


@asynctest
async def test_repeating_reminder_does_not_refire_immediately():
    """The reschedule has to land in the future or the loop spins."""
    await _add("repeat", schedule=recurrence.Schedule("interval", "900"))
    await db.pop_due_reminders(time.time())
    assert await db.pop_due_reminders(time.time()) == []


@asynctest
async def test_fire_count_accumulates_across_deliveries():
    await _add("repeat", schedule=recurrence.Schedule("interval", "600"))
    for expected in (1, 2, 3):
        # Pretend enough time passed for the next occurrence.
        rows = await db.list_reminders(7)
        await db.pop_due_reminders(rows[0]["due_at"] + 1)
        assert (await db.list_reminders(7))[0]["fire_count"] == expected


@asynctest
async def test_a_long_outage_does_not_replay_the_backlog():
    """Down for a day on an hourly reminder must fire once, not 24 times."""
    sched = recurrence.Schedule("interval", "3600")
    await _add("hourly", due_in=-24 * 3600, schedule=sched)

    fired = await db.pop_due_reminders(time.time())
    assert len(fired) == 1
    assert fired[0]["next_due"] > time.time()
    assert fired[0]["next_due"] - time.time() <= 3600 + 5
    assert await db.pop_due_reminders(time.time()) == []


@asynctest
async def test_clock_schedules_use_the_users_stored_timezone():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    await db.set_user_timezone(7, "Asia/Tokyo")
    await _add("standup", schedule=recurrence.Schedule("clock", "*:09:00"))

    fired = await db.pop_due_reminders(time.time())
    nxt = datetime.fromtimestamp(fired[0]["next_due"], ZoneInfo("Asia/Tokyo"))
    assert (nxt.hour, nxt.minute) == (9, 0)


@asynctest
async def test_missing_timezone_falls_back_to_utc_without_erroring():
    from datetime import datetime, timezone as dtz

    await _add("standup", schedule=recurrence.Schedule("clock", "*:09:00"))
    fired = await db.pop_due_reminders(time.time())
    nxt = datetime.fromtimestamp(fired[0]["next_due"], dtz.utc)
    assert (nxt.hour, nxt.minute) == (9, 0)


@asynctest
async def test_a_corrupt_repeat_spec_is_delivered_once_then_retired():
    """A bad spec must not wedge the delivery loop retrying forever."""
    await _add("broken", schedule=recurrence.Schedule("clock", "not:a:spec"))
    fired = await db.pop_due_reminders(time.time())
    assert len(fired) == 1
    assert fired[0]["next_due"] is None
    assert await db.list_reminders(7) == []


@asynctest
async def test_mixed_batch_retires_one_shots_and_keeps_repeats():
    await _add("once")
    await _add("repeat", schedule=recurrence.Schedule("interval", "900"))
    fired = await db.pop_due_reminders(time.time())
    assert {r["message"] for r in fired} == {"once", "repeat"}
    assert [r["message"] for r in await db.list_reminders(7)] == ["repeat"]


@asynctest
async def test_not_yet_due_reminders_are_left_alone():
    await _add("later", due_in=3600)
    assert await db.pop_due_reminders(time.time()) == []
    assert len(await db.list_reminders(7)) == 1


@asynctest
async def test_cancelling_a_repeating_reminder_stops_it():
    rid = await _add("repeat", schedule=recurrence.Schedule("interval", "900"))
    assert await db.delete_reminder(rid, 7) is True
    assert await db.pop_due_reminders(time.time()) == []


@asynctest
async def test_you_cannot_cancel_someone_elses_reminder():
    rid = await _add("mine", due_in=3600, user_id=7)
    assert await db.delete_reminder(rid, 999) is False
    assert len(await db.list_reminders(7)) == 1


@asynctest
async def test_repeating_count_is_separate_from_the_total():
    await _add("once", due_in=3600)
    await _add("a", due_in=3600, schedule=recurrence.Schedule("interval", "900"))
    await _add("b", due_in=3600, schedule=recurrence.Schedule("interval", "900"))
    assert await db.count_reminders(7) == 3
    assert await db.count_reminders_for(7, repeating=True) == 2
    assert await db.count_reminders_for(7, repeating=False) == 1


# ── user timezone ────────────────────────────────────────────────────────────

@asynctest
async def test_timezone_round_trip():
    assert await db.get_user_timezone(7) is None
    await db.set_user_timezone(7, "Europe/London")
    assert await db.get_user_timezone(7) == "Europe/London"
    await db.set_user_timezone(7, "America/Denver")
    assert await db.get_user_timezone(7) == "America/Denver"
    await db.set_user_timezone(7, None)
    assert await db.get_user_timezone(7) is None


# ── voice time ───────────────────────────────────────────────────────────────

@asynctest
async def test_a_session_banks_its_elapsed_time():
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 1800)
    banked = await db.voice_session_close(1, 7, now)
    assert banked == pytest.approx(1800, abs=2)

    board = await db.voice_leaderboard(1)
    assert board[0]["user_id"] == 7
    assert board[0]["seconds"] == pytest.approx(1800, abs=2)
    assert board[0]["sessions"] == 1


@asynctest
async def test_sessions_accumulate_across_joins():
    now = time.time()
    for _ in range(3):
        await db.voice_session_open(1, 7, 100, now - 600)
        await db.voice_session_close(1, 7, now)
    board = await db.voice_leaderboard(1)
    assert board[0]["seconds"] == pytest.approx(1800, abs=5)
    assert board[0]["sessions"] == 3


@asynctest
async def test_time_is_attributed_per_channel():
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 600)
    await db.voice_session_close(1, 7, now)
    await db.voice_session_open(1, 7, 200, now - 1200)
    await db.voice_session_close(1, 7, now)

    detail = await db.voice_user_detail(1, 7)
    assert {r["channel_id"]: round(r["seconds"]) for r in detail} == \
        {100: 600, 200: 1200}
    channels = await db.voice_channel_totals(1)
    assert channels[0]["channel_id"] == 200      # busiest first


@asynctest
async def test_closing_without_an_open_session_is_a_no_op():
    assert await db.voice_session_close(1, 7, time.time()) == 0.0
    assert await db.voice_leaderboard(1) == []


@asynctest
async def test_reopening_replaces_rather_than_duplicates():
    """Two joins with no leave in between must not leave two open rows."""
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 600)
    await db.voice_session_open(1, 7, 200, now - 300)
    assert len(await db.voice_open_sessions()) == 1
    await db.voice_session_close(1, 7, now)
    detail = await db.voice_user_detail(1, 7)
    assert [r["channel_id"] for r in detail] == [200]


@asynctest
async def test_max_seconds_caps_a_session_left_open_by_a_crash():
    """A restart can't know when they left — cap rather than bank a weekend."""
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 72 * 3600)
    banked = await db.voice_session_close(1, 7, now, max_seconds=12 * 3600)
    assert banked == pytest.approx(12 * 3600, abs=2)


@asynctest
async def test_max_seconds_does_not_inflate_a_short_session():
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 300)
    banked = await db.voice_session_close(1, 7, now, max_seconds=12 * 3600)
    assert banked == pytest.approx(300, abs=2)


@asynctest
async def test_guilds_are_kept_separate():
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 600)
    await db.voice_session_close(1, 7, now)
    await db.voice_session_open(2, 7, 300, now - 1200)
    await db.voice_session_close(2, 7, now)

    assert round((await db.voice_guild_total(1))[0]) == 600
    assert round((await db.voice_guild_total(2))[0]) == 1200


@asynctest
async def test_opting_out_erases_existing_rows():
    """An opt-out that leaves the data behind isn't one."""
    now = time.time()
    await db.voice_session_open(1, 7, 100, now - 600)
    await db.voice_session_close(1, 7, now)
    await db.voice_session_open(1, 7, 100, now)      # and one in progress
    assert await db.voice_leaderboard(1)

    await db.set_voice_optout(7, True)
    assert await db.voice_leaderboard(1) == []
    assert await db.voice_open_sessions() == []
    assert await db.voice_optout_ids() == {7}


@asynctest
async def test_opting_back_in_clears_the_flag():
    await db.set_voice_optout(7, True)
    await db.set_voice_optout(7, False)
    assert await db.voice_optout_ids() == set()


@asynctest
async def test_opting_out_does_not_touch_other_users():
    now = time.time()
    for uid in (7, 8):
        await db.voice_session_open(1, uid, 100, now - 600)
        await db.voice_session_close(1, uid, now)
    await db.set_voice_optout(7, True)
    board = await db.voice_leaderboard(1)
    assert [r["user_id"] for r in board] == [8]


@asynctest
async def test_guild_total_counts_distinct_people():
    now = time.time()
    for uid in (7, 8, 9):
        await db.voice_session_open(1, uid, 100, now - 600)
        await db.voice_session_close(1, uid, now)
    total, users = await db.voice_guild_total(1)
    assert users == 3
    assert total == pytest.approx(1800, abs=10)


@asynctest
async def test_empty_guild_totals_are_zero_not_none():
    assert await db.voice_guild_total(999) == (0.0, 0)
    assert await db.voice_leaderboard(999) == []
    assert await db.voice_user_detail(999, 7) == []
