"""Tests for `-remindme every …` schedule parsing and next-fire maths."""
from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import pytest

from core import recurrence as r
from core.recurrence import Schedule, ScheduleError

_DEFAULT = dtime(14, 30)


def _parse(text):
    return r.parse_schedule(text, default_time=_DEFAULT)


# ── one-shots pass straight through ──────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "2h30m take the pizza out", "10m stretch", "90 seconds", "",
    "everything is fine",       # starts with "every"-ish but isn't a keyword
])
def test_non_recurring_text_is_untouched(text):
    sched, rest = _parse(text)
    assert sched is None
    assert rest == text


# ── intervals ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,seconds,msg", [
    ("every 2h stretch",        7200,   "stretch"),
    ("every 90m water",         5400,   "water"),
    ("every 3 days backup",     259200, "backup"),
    ("every 1h30m check in",    5400,   "check in"),
    ("each 45m posture",        2700,   "posture"),
    ("every 12h",               43200,  ""),
])
def test_interval_parsing(text, seconds, msg):
    sched, rest = _parse(text)
    assert sched.kind == "interval"
    assert sched.seconds == seconds
    assert rest == msg


def test_interval_below_the_floor_is_rejected():
    """A 9-second repeat is a rate-limit incident, not a reminder."""
    with pytest.raises(ScheduleError, match="10m"):
        _parse("every 9 seconds spam")


def test_bare_number_is_treated_as_an_interval_not_a_clock_time():
    """`every 9` must not silently become "every day at 09:00"."""
    with pytest.raises(ScheduleError):
        _parse("every 9 do a thing")


def test_interval_above_a_year_is_rejected():
    with pytest.raises(ScheduleError, match="365"):
        _parse("every 400 days renew")


# ── clock schedules ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,spec,msg", [
    ("every day 9am standup",          "*:09:00",         "standup"),
    ("every day at 9am standup",       "*:09:00",         "standup"),
    ("every monday 9am sync",          "0:09:00",         "sync"),
    ("every monday at 9 am sync",      "0:09:00",         "sync"),
    ("every weekday 8:30 scrum",       "0,1,2,3,4:08:30", "scrum"),
    ("every weekend 10am chores",      "5,6:10:00",       "chores"),
    ("every mon,wed,fri 6pm gym",      "0,2,4:18:00",     "gym"),
    ("every tuesday 21:00 raid",       "1:21:00",         "raid"),
    ("every sunday midnight backup",   "6:00:00",         "backup"),
    ("every day noon lunch",           "*:12:00",         "lunch"),
    ("every fri 5pm deploy freeze",    "4:17:00",         "deploy freeze"),
    ("every sat & sun 11am brunch",    "5,6:11:00",       "brunch"),
])
def test_clock_parsing(text, spec, msg):
    sched, rest = _parse(text)
    assert sched.kind == "clock"
    assert sched.spec == spec
    assert rest == msg


def test_meridiem_is_not_left_in_the_message():
    """`9 am` splits into two tokens; both belong to the time."""
    _sched, rest = _parse("every monday at 9 am send the report")
    assert rest == "send the report"


def test_day_without_a_time_uses_the_default():
    sched, rest = _parse("every day water the plants")
    assert sched.spec == "*:14:30"
    assert rest == "water the plants"


def test_everyday_is_a_recurrence_word():
    sched, rest = _parse("everyday drink water")
    assert sched.kind == "clock" and sched.spec == "*:14:30"
    assert rest == "drink water"


def test_day_without_a_time_and_no_default_is_an_error():
    with pytest.raises(ScheduleError, match="time of day"):
        r.parse_schedule("every monday standup", default_time=None)


def test_unparseable_recurrence_explains_the_options():
    with pytest.raises(ScheduleError, match="every day 9am"):
        _parse("every blue moon howl")


@pytest.mark.parametrize("text", ["every day 25:00 x", "every day 13pm x"])
def test_impossible_clock_times_are_rejected(text):
    with pytest.raises(ScheduleError):
        _parse(text)


# ── describe() ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,spec,expect", [
    ("interval", "7200",            "every 2h"),
    ("interval", "5400",            "every 1h 30m"),
    ("clock",    "*:09:00",         "every day at 09:00"),
    ("clock",    "0,1,2,3,4:08:30", "every weekday at 08:30"),
    ("clock",    "5,6:10:00",       "every weekend at 10:00"),
    ("clock",    "0:09:00",         "every Monday at 09:00"),
    ("clock",    "0,2,4:18:00",     "every Mon, Wed & Fri at 18:00"),
])
def test_describe(kind, spec, expect):
    assert Schedule(kind, spec).describe() == expect


def test_every_parse_round_trips_through_describe():
    """Whatever we store must be renderable — no spec without a phrasing."""
    for text in ("every 2h x", "every day 9am x", "every weekday 8:30 x",
                 "every weekend 10am x", "every monday 9am x",
                 "every mon,wed,fri 6pm x"):
        sched, _ = _parse(text)
        assert sched.describe().startswith("every ")


# ── persistence ──────────────────────────────────────────────────────────────

def test_from_db_round_trip():
    sched, _ = _parse("every weekday 8:30 scrum")
    assert Schedule.from_db(sched.kind, sched.spec) == sched


@pytest.mark.parametrize("kind,spec", [
    (None, None), ("interval", None), (None, "7200"), ("cron", "* * * * *"),
])
def test_from_db_rejects_junk(kind, spec):
    """A one-shot reminder's NULL columns must not become a broken schedule."""
    assert Schedule.from_db(kind, spec) is None


# ── next_after ───────────────────────────────────────────────────────────────

def _ts(y, mo, d, h, mi, tz="UTC"):
    return datetime(y, mo, d, h, mi, tzinfo=ZoneInfo(tz)).timestamp()


def _at(ts, tz="UTC"):
    return datetime.fromtimestamp(ts, ZoneInfo(tz))


def test_clock_next_is_later_today_when_the_time_has_not_passed():
    sched, _ = _parse("every day 18:00 dinner")
    nxt = _at(sched.next_after(_ts(2026, 7, 27, 9, 0)))
    assert (nxt.month, nxt.day, nxt.hour) == (7, 27, 18)


def test_clock_next_rolls_to_tomorrow_once_the_time_has_passed():
    sched, _ = _parse("every day 09:00 standup")
    nxt = _at(sched.next_after(_ts(2026, 7, 27, 9, 0)))
    assert (nxt.day, nxt.hour) == (28, 9)


def test_weekly_skips_to_the_right_weekday():
    sched, _ = _parse("every monday 09:00 sync")
    # 2026-07-28 is a Tuesday, so the next Monday is 2026-08-03.
    nxt = _at(sched.next_after(_ts(2026, 7, 28, 12, 0)))
    assert nxt.weekday() == 0
    assert (nxt.month, nxt.day, nxt.hour) == (8, 3, 9)


def test_weekday_schedule_skips_the_weekend():
    sched, _ = _parse("every weekday 08:30 scrum")
    # Friday 2026-07-31 at 12:00 → next is Monday 2026-08-03.
    nxt = _at(sched.next_after(_ts(2026, 7, 31, 12, 0)))
    assert nxt.weekday() == 0 and nxt.day == 3


def test_weekend_schedule_skips_the_week():
    sched, _ = _parse("every weekend 10:00 chores")
    nxt = _at(sched.next_after(_ts(2026, 7, 27, 12, 0)))   # Monday
    assert nxt.weekday() == 5 and nxt.day == 1             # Saturday 1 Aug


def test_clock_time_is_interpreted_in_the_users_timezone():
    """09:00 means 09:00 where they are, not 09:00 UTC."""
    sched, _ = _parse("every day 09:00 standup")
    base = _ts(2026, 7, 27, 3, 0)      # 03:00 UTC = 23:00 (26th) in New York
    ny = _at(sched.next_after(base, "America/New_York"), "America/New_York")
    assert (ny.day, ny.hour) == (27, 9)
    utc = _at(sched.next_after(base, "UTC"))
    assert (utc.day, utc.hour) == (27, 9)
    # Same wall time, different instants — that's the whole point.
    assert sched.next_after(base, "America/New_York") != sched.next_after(base, "UTC")


def test_clock_survives_a_dst_transition():
    """London springs forward on 2026-03-29; 09:00 stays 09:00 either side.

    The two base times straddle the change, so the fire times land in GMT and
    BST respectively — same wall clock, an hour apart in real terms. A naive
    "add 86400 seconds" scheduler drifts to 08:00 or 10:00 here.
    """
    sched, _ = _parse("every day 09:00 standup")
    before = _at(sched.next_after(_ts(2026, 3, 27, 12, 0), "Europe/London"),
                 "Europe/London")
    after  = _at(sched.next_after(_ts(2026, 3, 29, 12, 0), "Europe/London"),
                 "Europe/London")
    assert before.hour == 9 and after.hour == 9
    assert before.utcoffset() != after.utcoffset()   # the clocks did change


def test_next_after_is_always_in_the_future():
    for text in ("every day 09:00 x", "every monday 09:00 x",
                 "every weekday 08:30 x", "every 2h x"):
        sched, _ = _parse(text)
        now = datetime.now(timezone.utc).timestamp()
        assert sched.next_after(now) > now, text


def test_intervals_skip_missed_occurrences_instead_of_replaying_them():
    """A bot that was down for a day must not deliver a day of backlog."""
    sched, _ = _parse("every 1h x")
    now  = datetime.now(timezone.utc).timestamp()
    long_ago = now - 24 * 3600
    nxt = sched.next_after(long_ago)
    assert nxt > now
    assert nxt - now <= 3600 + 1


# ── timezones ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("utc", "UTC"), ("UTC", "UTC"), ("gmt", "UTC"),
    ("pst", "America/Los_Angeles"), ("PT", "America/Los_Angeles"),
    ("est", "America/New_York"), ("eastern", "America/New_York"),
    ("uk", "Europe/London"), ("london", "Europe/London"),
    ("Europe/London", "Europe/London"), ("europe/london", "Europe/London"),
    ("America/New_York", "America/New_York"),
    ("asia/tokyo", "Asia/Tokyo"), ("jst", "Asia/Tokyo"),
    ("New York", "America/New_York"),
])
def test_normalise_timezone(raw, expect):
    assert r.normalise_timezone(raw) == expect


@pytest.mark.parametrize("raw", ["", None, "banana", "Mars/Olympus_Mons", "??"])
def test_normalise_timezone_rejects_junk(raw):
    assert r.normalise_timezone(raw) is None


def test_us_abbreviations_map_to_real_zones_so_dst_still_works():
    """'EST' as a fixed -05:00 would be an hour wrong for half the year."""
    tz = r.get_timezone(r.normalise_timezone("est"))
    summer = datetime(2026, 7, 1, 12, tzinfo=tz)
    winter = datetime(2026, 1, 1, 12, tzinfo=tz)
    assert summer.utcoffset() != winter.utcoffset()


def test_get_timezone_never_raises():
    for bad in (None, "", "not/a/zone", "banana"):
        assert r.get_timezone(bad).key == "UTC"


# ── parse_when: one-off dates and times ──────────────────────────────────────

_MON = datetime(2026, 7, 27, 14, 30, tzinfo=ZoneInfo("Europe/London"))  # a Monday


def _when(text, tz="Europe/London", now=_MON):
    return r.parse_when(text, tz_name=tz, now=now)


@pytest.mark.parametrize("text,expect", [
    ("3pm",              (2026, 7, 27, 15, 0)),    # later today
    ("15:30",            (2026, 7, 27, 15, 30)),
    ("9am",              (2026, 7, 28, 9, 0)),     # already gone → tomorrow
    ("noon",             (2026, 7, 28, 12, 0)),    # gone → tomorrow
    ("midnight",         (2026, 7, 28, 0, 0)),
    ("today 5pm",        (2026, 7, 27, 17, 0)),
    ("tonight",          (2026, 7, 27, 20, 0)),    # evening, not "now o'clock"
    ("tomorrow",         (2026, 7, 28, 14, 30)),
    ("tomorrow 9am",     (2026, 7, 28, 9, 0)),
    ("friday",           (2026, 7, 31, 14, 30)),
    ("friday 3pm",       (2026, 7, 31, 15, 0)),
    ("next monday",      (2026, 8, 3, 14, 30)),
    ("monday 4pm",       (2026, 7, 27, 16, 0)),    # later today
    ("monday 10am",      (2026, 8, 3, 10, 0)),     # gone → next Monday
    ("aug 5",            (2026, 8, 5, 14, 30)),
    ("5 aug",            (2026, 8, 5, 14, 30)),
    ("5th august",       (2026, 8, 5, 14, 30)),
    ("august 5 3pm",     (2026, 8, 5, 15, 0)),
    ("2026-08-01",       (2026, 8, 1, 14, 30)),
    ("2026-08-01 14:00", (2026, 8, 1, 14, 0)),
    ("in 2 hours",       (2026, 7, 27, 16, 30)),
    ("in 30m",           (2026, 7, 27, 15, 0)),
    ("in 3 days",        (2026, 7, 30, 14, 30)),
    ("at 6pm",           (2026, 7, 27, 18, 0)),    # joining words ignored
    ("on friday at 9am", (2026, 7, 31, 9, 0)),
])
def test_parse_when(text, expect):
    got = _when(text)
    assert (got.year, got.month, got.day, got.hour, got.minute) == expect


def test_a_month_day_already_past_rolls_to_next_year():
    got = _when("jan 1")
    assert (got.year, got.month, got.day) == (2027, 1, 1)


def test_an_explicit_past_date_is_left_alone():
    """If you spell out a date, you meant it — don't silently move it."""
    got = _when("2026-01-01")
    assert got.year == 2026 and got < _MON


def test_relative_results_are_always_in_the_future():
    for text in ("3pm", "9am", "noon", "friday", "monday 10am", "tomorrow"):
        assert _when(text) > _MON, text


def test_parse_when_uses_the_given_timezone():
    """3pm means 3pm where the user is."""
    london = _when("3pm", tz="Europe/London")
    tokyo  = _when("3pm", tz="Asia/Tokyo")
    assert london.hour == tokyo.hour == 15
    assert london.timestamp() != tokyo.timestamp()


def test_parse_when_defaults_to_utc():
    got = r.parse_when("3pm", now=_MON)
    assert got.tzinfo is not None
    assert got.utcoffset().total_seconds() == 0


@pytest.mark.parametrize("text", [
    "", "   ", "banana", "next banana", "the day after tomorrow",
    "in a bit", "in ages", "feb 30", "2026-13-45", "sometime",
])
def test_parse_when_rejects_what_it_cannot_read(text):
    with pytest.raises(ScheduleError):
        _when(text)


def test_parse_when_error_suggests_real_formats():
    with pytest.raises(ScheduleError, match="tomorrow 9am"):
        _when("banana")


def test_weekday_group_words_are_not_dates():
    """'weekday' and 'weekend' are schedule words — they're not one day."""
    for word in ("weekday", "weekend", "weekdays"):
        with pytest.raises(ScheduleError):
            _when(word)


def test_parse_when_survives_a_dst_boundary():
    """26 Oct 2026 London falls back; 9am the day after is still 9am."""
    before = datetime(2026, 10, 24, 12, 0, tzinfo=ZoneInfo("Europe/London"))
    got = r.parse_when("monday 9am", tz_name="Europe/London", now=before)
    assert (got.month, got.day, got.hour) == (10, 26, 9)
    assert got.utcoffset() != before.utcoffset()   # the clocks went back
