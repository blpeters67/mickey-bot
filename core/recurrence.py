"""
Repeating-schedule parsing for `-remindme every …`.

Pure Python, no Discord and no database, so the whole grammar is unit-testable.
Two shapes cover everything the command accepts:

  * **interval** — "every 2h", "every 90m", "every 3 days". Fires a fixed
    number of seconds apart. No timezone involved.
  * **clock** — "every day 9am", "every monday 9am", "every weekday 8:30",
    "every mon,wed,fri 18:00". Fires at a wall-clock time on chosen weekdays,
    which means it needs the user's timezone to mean anything.

Clock schedules are stored as ``"<days>:<HH>:<MM>"`` where *days* is either
``*`` or a comma-separated list of weekday numbers (Monday = 0, matching
``datetime.weekday()``). Intervals are stored as a plain second count. Both
live in one TEXT column, tagged by :attr:`Schedule.kind`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from .formatters import fmt_duration_long, parse_duration

# A repeating ping needs a floor the one-shot minimum (10s) doesn't: something
# that fires every few seconds is indistinguishable from an attack on the
# channel, and Discord's rate limiter would agree.
MIN_INTERVAL = 10 * 60          # 10 minutes
MAX_INTERVAL = 365 * 24 * 3600  # a year

_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday")
_DAY_SHORT = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Single weekdays, plus the three group words.
_WEEKDAY_WORDS: dict[str, tuple[int, ...]] = {
    "monday": (0,), "mon": (0,), "mo": (0,),
    "tuesday": (1,), "tue": (1,), "tues": (1,), "tu": (1,),
    "wednesday": (2,), "wed": (2,), "weds": (2,), "we": (2,),
    "thursday": (3,), "thu": (3,), "thur": (3,), "thurs": (3,), "th": (3,),
    "friday": (4,), "fri": (4,), "fr": (4,),
    "saturday": (5,), "sat": (5,), "sa": (5,),
    "sunday": (6,), "sun": (6,), "su": (6,),
    "weekday": (0, 1, 2, 3, 4), "weekdays": (0, 1, 2, 3, 4),
    "weekend": (5, 6), "weekends": (5, 6),
    "day": (0, 1, 2, 3, 4, 5, 6), "days": (0, 1, 2, 3, 4, 5, 6),
    "daily": (0, 1, 2, 3, 4, 5, 6),
}
_ALL_DAYS = (0, 1, 2, 3, 4, 5, 6)

_EVERY_WORDS = ("every", "each")
# "everyday drink water" / "daily standup" — recurrence with the "every" fused
# into the period word.
_BARE_DAILY = ("everyday", "daily", "nightly")

_TIME_RE = re.compile(
    r"^(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<mer>am|pm|a\.m\.|p\.m\.)?$"
)
_NAMED_TIMES = {"noon": dtime(12, 0), "midday": dtime(12, 0),
                "midnight": dtime(0, 0)}
_MERIDIEM_RE = re.compile(r"^(am|pm|a\.m\.|p\.m\.)[.,]?$", re.I)


class ScheduleError(ValueError):
    """Raised with a message meant to be shown to the user verbatim."""


@dataclass(frozen=True)
class Schedule:
    """A parsed repeat rule. ``kind`` is ``"interval"`` or ``"clock"``."""

    kind: str
    spec: str

    # ── persistence ──────────────────────────────────────────────────────────

    @classmethod
    def from_db(cls, kind: str | None, spec: str | None) -> "Schedule | None":
        """Rebuild from the two columns, tolerating a one-shot reminder's NULLs."""
        if not kind or not spec:
            return None
        if kind not in ("interval", "clock"):
            return None
        return cls(kind, spec)

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def seconds(self) -> int:
        """Interval length. Meaningless (and 0) for clock schedules."""
        return int(self.spec) if self.kind == "interval" else 0

    def _clock_parts(self) -> tuple[tuple[int, ...], int, int]:
        days_s, hh, mm = self.spec.split(":")
        days = _ALL_DAYS if days_s == "*" else tuple(
            sorted(int(d) for d in days_s.split(",")))
        return days, int(hh), int(mm)

    def describe(self) -> str:
        """Human phrasing for confirmations and the `reminders` list."""
        if self.kind == "interval":
            return f"every {fmt_duration_long(self.seconds)}"
        days, hh, mm = self._clock_parts()
        at = f"{hh:02d}:{mm:02d}"
        if days == _ALL_DAYS:
            return f"every day at {at}"
        if days == (0, 1, 2, 3, 4):
            return f"every weekday at {at}"
        if days == (5, 6):
            return f"every weekend at {at}"
        if len(days) == 1:
            return f"every {_DAY_NAMES[days[0]]} at {at}"
        names = [_DAY_SHORT[d] for d in days]
        return f"every {', '.join(names[:-1])} & {names[-1]} at {at}"

    # ── scheduling ───────────────────────────────────────────────────────────

    def next_after(self, after: float, tz_name: str = "UTC") -> float:
        """First fire time strictly after the `after` unix timestamp.

        Intervals step forward from `after` until they clear it, so a bot that
        was down for a day resumes on the next slot instead of delivering a
        day's worth of backlog at once.
        """
        if self.kind == "interval":
            step = self.seconds
            nxt  = after + step
            now  = datetime.now(timezone.utc).timestamp()
            if nxt <= now:
                # Skip the missed occurrences rather than replaying them.
                missed = int((now - nxt) // step) + 1
                nxt += missed * step
            return nxt

        days, hh, mm = self._clock_parts()
        tz    = get_timezone(tz_name)
        start = datetime.fromtimestamp(after, tz)
        # 8 candidate days covers "today, later" through "this weekday next week".
        for offset in range(0, 8):
            day = (start + timedelta(days=offset)).date()
            if day.weekday() not in days:
                continue
            when = datetime.combine(day, dtime(hh, mm), tzinfo=tz)
            # A DST spring-forward can push the wall time past the target; a
            # fall-back makes it ambiguous and fold=0 takes the earlier one.
            if when.timestamp() > after:
                return when.timestamp()
        # Unreachable for a non-empty day set, but never return the past.
        return after + 86400


# ──────────────────────────────────────────────────────────────────────────────
# Timezones
# ──────────────────────────────────────────────────────────────────────────────

# Everyday shorthands people actually type. Deliberately mapped to IANA zones
# rather than fixed offsets so DST keeps working.
_TZ_ALIASES = {
    "utc": "UTC", "gmt": "UTC", "z": "UTC",
    "est": "America/New_York", "edt": "America/New_York",
    "et": "America/New_York", "eastern": "America/New_York",
    "cst": "America/Chicago", "cdt": "America/Chicago",
    "ct": "America/Chicago", "central": "America/Chicago",
    "mst": "America/Denver", "mdt": "America/Denver",
    "mt": "America/Denver", "mountain": "America/Denver",
    "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
    "pt": "America/Los_Angeles", "pacific": "America/Los_Angeles",
    "uk": "Europe/London", "bst": "Europe/London", "london": "Europe/London",
    "ist": "Asia/Kolkata", "india": "Asia/Kolkata",
    "cet": "Europe/Paris", "cest": "Europe/Paris",
    "aest": "Australia/Sydney", "aedt": "Australia/Sydney",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "jst": "Asia/Tokyo", "tokyo": "Asia/Tokyo",
    "nyc": "America/New_York", "la": "America/Los_Angeles",
    "chicago": "America/Chicago", "denver": "America/Denver",
    "berlin": "Europe/Berlin", "paris": "Europe/Paris",
    "madrid": "Europe/Madrid", "rome": "Europe/Rome",
    "dublin": "Europe/Dublin", "lisbon": "Europe/Lisbon",
    "amsterdam": "Europe/Amsterdam", "toronto": "America/Toronto",
    "vancouver": "America/Vancouver", "auckland": "Pacific/Auckland",
}


def normalise_timezone(name: str | None) -> str | None:
    """'pst' / 'Europe/london' / 'UK' → a canonical IANA name, or None.

    US abbreviations resolve to the matching IANA zone, so "EST" keeps
    observing daylight saving rather than freezing at -05:00 all year.
    """
    if not name:
        return None
    raw = name.strip().replace(" ", "_")
    alias = _TZ_ALIASES.get(raw.lower())
    if alias:
        return alias
    # Try as-is, then title-cased per path segment ("europe/london").
    for candidate in (raw, "/".join(p.title() for p in raw.split("/"))):
        try:
            ZoneInfo(candidate)
            return candidate
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            continue

    # Bare city name — "New York", "Tokyo", "Berlin". Match the last segment
    # of every known zone so people don't have to know the continent prefix.
    if "/" not in raw:
        want = raw.lower()
        matches = sorted(
            z for z in available_timezones()
            if z.rsplit("/", 1)[-1].lower() == want
        )
        if len(matches) == 1:
            return matches[0]
        if matches:
            # Ambiguous (e.g. "Georgia"): prefer the common continents over
            # the legacy US/ and Etc/ trees rather than guessing at random.
            for prefix in ("America/", "Europe/", "Asia/", "Australia/",
                           "Africa/", "Pacific/"):
                for z in matches:
                    if z.startswith(prefix):
                        return z
            return matches[0]
    return None


def get_timezone(name: str | None) -> ZoneInfo:
    """Never raises — an unknown or missing zone falls back to UTC."""
    try:
        return ZoneInfo(name) if name else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return ZoneInfo("UTC")


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_days(token: str) -> tuple[int, ...] | None:
    """'monday' / 'weekday' / 'mon,wed,fri' → weekday numbers, or None."""
    out: list[int] = []
    for part in token.lower().replace("&", ",").replace("+", ",").split(","):
        part = part.strip().rstrip(".")
        if not part:
            continue
        found = _WEEKDAY_WORDS.get(part)
        if found is None:
            return None
        out.extend(found)
    return tuple(sorted(set(out))) if out else None


def _parse_clock(token: str) -> dtime | None:
    """'9am' / '9:30' / '21:00' / 'noon' → a time of day, or None."""
    t = token.lower().strip().rstrip(".,")
    if t in _NAMED_TIMES:
        return _NAMED_TIMES[t]
    m = _TIME_RE.match(t)
    if not m:
        return None
    hour = int(m.group("h"))
    minute = int(m.group("m") or 0)
    mer = (m.group("mer") or "").replace(".", "")
    if mer:
        if not 1 <= hour <= 12 or minute > 59:
            return None
        if mer == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = 12 if hour == 12 else hour + 12
    elif hour > 23 or minute > 59:
        return None
    return dtime(hour, minute)


def _looks_like_a_time(token: str) -> bool:
    """True for tokens only a clock time could be — has ':' or am/pm.

    Keeps "every 9" (9 seconds, rejected as too short) from silently becoming
    "every day at 09:00", which is a very different reminder.
    """
    t = token.lower().rstrip(".,")
    return t in _NAMED_TIMES or ":" in t or t.endswith(("am", "pm", "a.m.", "p.m."))


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ORDINAL_RE = re.compile(r"^(\d{1,2})(st|nd|rd|th)$")


def parse_when(
    text: str, *, tz_name: str = "UTC", now: datetime | None = None,
) -> datetime:
    """Parse a one-off date/time into a timezone-aware datetime.

    Deliberately narrower than a general natural-language date parser — it
    covers what people actually type in chat and rejects the rest with a
    usable error, rather than guessing:

        3pm · 15:30 · noon · midnight
        tomorrow · tomorrow 9am · today 5pm · tonight
        friday · friday 3pm · next friday 3pm
        aug 5 · 5 august · august 5 3pm · 5th aug
        2026-08-01 · 2026-08-01 14:00
        in 2 hours · in 30m · in 3 days

    A bare time that's already passed rolls to tomorrow; a bare weekday is
    the *next* one. Without a time, the current time of day is used, so
    "friday" means "this time on Friday".

    Raises :class:`ScheduleError` with a message meant for the user.
    """
    tz  = get_timezone(tz_name)
    now = now.astimezone(tz) if now else datetime.now(tz)
    raw = (text or "").strip().lower()
    if not raw:
        raise ScheduleError("Give me a time — `3pm`, `tomorrow 9am`, "
                            "`friday`, `aug 5`, or `in 2 hours`.")

    # "in 2 hours" / "in 90m" — purely relative, no calendar involved.
    if raw.startswith("in "):
        secs = parse_duration(raw[3:].replace(" ", ""))
        if not secs:
            raise ScheduleError(
                f"Couldn't read `{text[3:].strip()}` as a length of time. "
                "Try `in 2 hours`, `in 30m`, `in 3 days`.")
        return now + timedelta(seconds=secs)

    tokens = raw.replace(",", " ").split()
    if not tokens:
        raise ScheduleError("Give me a time.")

    # Pull a time off the end if there is one, so the rest is purely a date.
    when: dtime | None = None
    if len(tokens) >= 2 and _MERIDIEM_RE.match(tokens[-1]):
        when = _parse_clock(tokens[-2] + tokens[-1])
        if when is not None:
            tokens = tokens[:-2]
    if when is None and tokens:
        candidate = _parse_clock(tokens[-1])
        # Only treat a trailing token as a time if it *looks* like one, or if
        # it's all that's left — "aug 5" must not read 5 as 05:00.
        if candidate is not None and (_looks_like_a_time(tokens[-1])
                                      or len(tokens) == 1):
            when = candidate
            tokens = tokens[:-1]

    # Strip the joining words a person naturally types.
    tokens = [t for t in tokens if t not in ("at", "on", "the", "of", "this")]

    evening = tokens == ["tonight"]
    day, roll_days = _parse_when_date(tokens, now)
    if when is None:
        # "tonight" without a time means the evening, not this time of day.
        when = (dtime(20, 0) if evening
                else now.timetz().replace(second=0, microsecond=0))

    result = datetime.combine(day, dtime(when.hour, when.minute), tzinfo=tz)

    # A relative date that lands in the past means the *next* one: "3pm" after
    # 3pm is tomorrow, "monday 10am" on Monday afternoon is next Monday. An
    # explicit calendar date (roll_days == 0) is left alone — if you asked for
    # 2026-08-01 you meant it, even if it's gone.
    if roll_days and result <= now:
        result += timedelta(days=roll_days)
    return result


def _parse_when_date(tokens: list[str], now: datetime) -> tuple[date, int]:
    """Resolve the date part of `parse_when`.

    Returns ``(date, roll_days)``, where `roll_days` is how far to jump if the
    combined result turns out to be in the past — 1 for "no date given", 7 for
    a named weekday, and 0 for a date the user spelled out.
    """
    today = now.date()
    if not tokens:
        return today, 1

    joined = " ".join(tokens)

    if joined in ("today", "tonight"):
        return today, 0
    if joined == "tomorrow":
        return today + timedelta(days=1), 0
    if joined == "yesterday":
        return today - timedelta(days=1), 0

    # ISO date, on its own.
    m = _ISO_RE.match(tokens[0]) if len(tokens) == 1 else None
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))), 0
        except ValueError:
            raise ScheduleError(f"`{tokens[0]}` isn't a real date.") from None

    # Weekday, optionally prefixed with "next". Both "friday" and "next
    # friday" mean the coming Friday — English disagrees with itself about
    # which one skips a week, so neither does, and the rendered date makes
    # the result obvious anyway.
    want_next = tokens[0] in ("next", "coming")
    body = tokens[1:] if want_next else tokens
    if len(body) == 1:
        days = _WEEKDAY_WORDS.get(body[0])
        # Only single named weekdays are dates — "weekday"/"weekend" are
        # schedule words, not one day.
        if days and len(days) == 1:
            ahead = (days[0] - today.weekday()) % 7
            if ahead == 0 and want_next:
                ahead = 7
            return today + timedelta(days=ahead), 7

    # "aug 5" / "5 aug" / "5th august" / "august 5th"
    if len(body) == 2:
        a, b = body
        for month_tok, day_tok in ((a, b), (b, a)):
            month = _MONTHS.get(month_tok)
            if month is None:
                continue
            om = _ORDINAL_RE.match(day_tok)
            num = om.group(1) if om else day_tok
            if not num.isdigit():
                continue
            try:
                candidate = date(now.year, month, int(num))
            except ValueError:
                raise ScheduleError(f"There's no {month_tok} {num}.") from None
            # A date that's already gone this year means next year.
            if candidate < today:
                try:
                    candidate = candidate.replace(year=now.year + 1)
                except ValueError:      # 29 Feb into a non-leap year
                    raise ScheduleError(
                        f"There's no {month_tok} {num} next year.") from None
            return candidate, 0

    raise ScheduleError(
        f"Couldn't read `{' '.join(tokens)}` as a date.\n"
        "Try `3pm`, `tomorrow 9am`, `friday 3pm`, `aug 5`, `2026-08-01 14:00`, "
        "or `in 2 hours`.")


def parse_schedule(
    text: str, *, default_time: dtime | None = None,
) -> tuple[Schedule | None, str]:
    """Pull a leading repeat rule off `text`.

    Returns ``(schedule, remaining_text)``. `schedule` is None when the text
    doesn't start with a recurrence word, in which case `text` comes back
    untouched and the caller should treat it as a one-shot duration.

    `default_time` is used when a day is given without a time ("every day
    water the plants") — pass the user's current local time so it reads as
    "every day at about now".

    Raises :class:`ScheduleError` when the text clearly *means* to repeat but
    can't be understood, so the caller can show the reason instead of falling
    through to the one-shot parser and emitting a confusing error.
    """
    tokens = text.split()
    if not tokens:
        return None, text

    first = tokens[0].lower()
    if first in _BARE_DAILY:
        days: tuple[int, ...] | None = _ALL_DAYS
        i = 1
    elif first in _EVERY_WORDS:
        days = None
        i = 1
        # Day names can arrive glued ("mon,wed,fri") or spread across tokens
        # ("sat & sun", "mon and wed"), so keep consuming while they parse.
        collected: list[int] = []
        while i < len(tokens):
            token = tokens[i].strip(",")
            if token.lower() in ("&", "and", "+", "/") and collected:
                i += 1
                continue
            found = _parse_days(token)
            if not found:
                break
            collected.extend(found)
            i += 1
        if collected:
            days = tuple(sorted(set(collected)))
    else:
        return None, text

    # "every monday at 9am" — the joining word carries no information.
    if i < len(tokens) and tokens[i].lower() == "at":
        i += 1

    when: dtime | None = None
    if i < len(tokens) and (days is not None or _looks_like_a_time(tokens[i])):
        # "9 am" arrives as two tokens. Join them *before* trying the single
        # token, or bare "9" parses as 09:00 and strands "am" at the front of
        # the reminder text.
        if i + 1 < len(tokens) and _MERIDIEM_RE.match(tokens[i + 1]):
            when = _parse_clock(tokens[i] + tokens[i + 1])
            if when is not None:
                i += 2
        if when is None:
            when = _parse_clock(tokens[i])
            if when is not None:
                i += 1
            elif days is not None and tokens[i][:1].isdigit():
                # A day plus something numeric that isn't a valid time ("every
                # day 25:00") is a typo, not a reminder that starts with a
                # number — say so instead of silently using the default time.
                raise ScheduleError(
                    f"`{tokens[i]}` isn't a time of day. Try `9am`, `9:30`, "
                    "`18:00`, `noon` or `midnight`."
                )

    if days is not None or when is not None:
        if when is None:
            if default_time is None:
                raise ScheduleError(
                    "I need a time of day for that — try "
                    "`every day 9am` or `every monday 18:30`."
                )
            when = default_time
        spec_days = "*" if (days or _ALL_DAYS) == _ALL_DAYS else \
                    ",".join(str(d) for d in (days or _ALL_DAYS))
        return (Schedule("clock", f"{spec_days}:{when.hour:02d}:{when.minute:02d}"),
                " ".join(tokens[i:]))

    # No day and no time — the remainder should open with a duration. Try the
    # longest match first so "every 2 hours" beats "every 2".
    for span in (2, 1):
        if i + span > len(tokens):
            continue
        secs = parse_duration("".join(tokens[i:i + span]))
        if secs:
            if secs < MIN_INTERVAL:
                raise ScheduleError(
                    f"Repeating reminders can't fire more often than every "
                    f"**{fmt_duration_long(MIN_INTERVAL)}** — that one was "
                    f"{fmt_duration_long(secs)}."
                )
            if secs > MAX_INTERVAL:
                raise ScheduleError("Maximum repeat interval is 365 days.")
            return Schedule("interval", str(secs)), " ".join(tokens[i + span:])

    raise ScheduleError(
        "I couldn't work out how often to repeat that.\n"
        "Try `every 2h`, `every day 9am`, `every monday 18:30`, "
        "`every weekday 8:30`, or `every mon,wed,fri 12pm`."
    )
