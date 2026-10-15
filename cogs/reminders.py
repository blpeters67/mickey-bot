"""
Reminders cog — `remindme 2h30m take the pizza out`, and repeating ones like
`remindme every weekday 8:30 stand-up`.

Reminders persist in SQLite, so they survive restarts. A 15-second loop
delivers anything due; delivery is one transaction that deletes one-shots and
advances repeating ones, so a crash mid-batch can't replay reminders forever.

Repeating reminders come in two shapes (see `core.recurrence`): fixed
intervals ("every 2h") and wall-clock schedules ("every weekday 8:30"). The
second kind needs to know what "8:30" means to the person who set it, which is
what `-timezone` is for — until someone sets one they're treated as UTC.

Delivery goes to the channel the reminder was set in, with a fallback to DM
if the channel is gone (deleted, or the bot was kicked from the server).
"""
from __future__ import annotations

import time
from datetime import datetime

import discord
from discord.ext import commands, tasks

from core import db, recurrence
from core.formatters import fmt_duration_long, parse_duration
from core.logging_setup import get_logger
from core.recurrence import ScheduleError, get_timezone, normalise_timezone

log = get_logger(__name__)

MAX_PER_USER     = 25
MAX_REPEATING    = 10        # repeats never expire, so they need a tighter cap
MAX_DELAY        = 90 * 24 * 3600   # 90 days
MAX_MESSAGE_LEN  = 500


class RemindersCog(commands.Cog, name="Reminders"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.deliver.start()

    def cog_unload(self) -> None:
        if self.deliver.is_running():
            self.deliver.cancel()

    # ── commands ─────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="remindme", aliases=["remind"],
        description="Set a reminder — one-off or repeating",
        help=(
            "Pings you in this channel after the given time.\n\n"
            "**One-off**\n"
            "Durations: `30s`, `10m`, `2h`, `1h30m`, `2h15m30s`, or plain "
            "seconds. Maximum 90 days.\n\n"
            "**Repeating** — start with `every`:\n"
            "`every 2h` · `every 90m` · `every 3 days` — fixed intervals\n"
            "`every day 9am` · `every monday 18:30` · `every weekday 8:30`\n"
            "`every weekend 10am` · `every mon,wed,fri 6pm`\n\n"
            "Clock times use **your** timezone — set it once with "
            "`timezone Europe/London` (UTC until you do). Minimum repeat "
            "interval is 10 minutes.\n\n"
            "Reminders survive bot restarts. `reminders` lists yours, "
            "`reminders cancel <id>` stops one.\n\n"
            "**Examples**\n"
            "`remindme 10m stretch`\n"
            "`remindme 2h30m take the pizza out`\n"
            "`remindme every weekday 8:30 stand-up`\n"
            "`remindme every monday 9am send the weekly report`"
        ),
    )
    async def remindme(self, ctx: commands.Context, when: str, *,
                       message: str | None = None):
        # A repeat rule can run over several tokens ("every monday 9am"), and
        # discord.py has already split them across `when` and `message`.
        # Re-join, parse from the front, and whatever's left is the text.
        combined = f"{when} {message}".strip() if message else when.strip()

        tz_name = await db.get_user_timezone(ctx.author.id)
        now_local = datetime.now(get_timezone(tz_name))
        try:
            schedule, rest = recurrence.parse_schedule(
                combined, default_time=now_local.time().replace(second=0,
                                                                microsecond=0),
            )
        except ScheduleError as e:
            return await ctx.send(f"❌ {e}")

        if schedule is None:
            seconds = parse_duration(when)
            if seconds is None or seconds <= 0:
                return await ctx.send(
                    "❌ Couldn't parse that time. Try `10m`, `2h`, `1h30m`, "
                    "`90` (seconds), or a repeat like `every weekday 8:30`."
                )
            if seconds < 10:
                return await ctx.send("❌ Minimum is 10 seconds.")
            if seconds > MAX_DELAY:
                return await ctx.send("❌ Maximum is 90 days.")
            due_at = time.time() + seconds
            rest   = message
        else:
            if await db.count_reminders_for(ctx.author.id, repeating=True) \
                    >= MAX_REPEATING:
                return await ctx.send(
                    f"❌ You already have {MAX_REPEATING} repeating reminders — "
                    "they never expire on their own. "
                    f"`{ctx.prefix}reminders` to see them."
                )
            due_at = schedule.next_after(time.time(), tz_name or "UTC")

        if await db.count_reminders(ctx.author.id) >= MAX_PER_USER:
            return await ctx.send(
                f"❌ You already have {MAX_PER_USER} pending reminders. "
                f"`{ctx.prefix}reminders` to see them, "
                f"`{ctx.prefix}reminders cancel <id>` to drop one."
            )

        text = (rest or "").strip()[:MAX_MESSAGE_LEN] or "⏰ Reminder!"
        rid = await db.add_reminder(
            guild_id=ctx.guild.id if ctx.guild else None,
            channel_id=ctx.channel.id,
            user_id=ctx.author.id,
            message=text,
            due_at=due_at,
            repeat_kind=schedule.kind if schedule else None,
            repeat_spec=schedule.spec if schedule else None,
        )

        cancel_hint = f"*(id {rid} — `{ctx.prefix}reminders cancel {rid}` to cancel)*"
        if schedule is None:
            return await ctx.send(
                f"⏰ Got it — I'll remind you in "
                f"**{fmt_duration_long(int(due_at - time.time()))}**. {cancel_hint}"
            )

        # <t:…:R> renders in each reader's own local time, which sidesteps any
        # confusion about whose timezone the confirmation is quoting.
        note = ""
        if schedule.kind == "clock" and not tz_name:
            note = (f"\n-# Times are UTC — `{ctx.prefix}timezone Europe/London` "
                    "(or your zone) to use your own.")
        await ctx.send(
            f"🔁 Got it — **{schedule.describe()}**, starting "
            f"<t:{int(due_at)}:R>. {cancel_hint}{note}"
        )

    @commands.hybrid_command(
        name="timezone", aliases=["tz"],
        description="Set your timezone, used by repeating reminders",
        help=(
            "Repeating reminders with a clock time (`every weekday 8:30`) need "
            "to know which 8:30 you mean. Set it once and it applies to all of "
            "yours, in every server. Until you do, they run on UTC.\n\n"
            "Accepts IANA names (`Europe/London`, `America/New_York`, "
            "`Asia/Tokyo`) and common shorthands (`PST`, `EST`, `UK`, `CET`, "
            "`IST`). Abbreviations map to the real zone, so daylight saving "
            "keeps working.\n\n"
            "`timezone` on its own shows yours and the current time in it.\n"
            "`timezone clear` goes back to UTC.\n\n"
            "**Examples**\n"
            "`timezone Europe/London`\n"
            "`timezone PST`\n"
            "`timezone Australia/Sydney`"
        ),
    )
    async def timezone_cmd(self, ctx: commands.Context, *, zone: str | None = None):
        current = await db.get_user_timezone(ctx.author.id)

        if zone is None:
            tz  = get_timezone(current)
            now = datetime.now(tz)
            if current:
                return await ctx.send(
                    f"🕒 Your timezone is **{current}** — it's "
                    f"**{now:%H:%M}** there ({now:%Z}).\n"
                    f"-# `{ctx.prefix}timezone <zone>` to change, "
                    f"`{ctx.prefix}timezone clear` to reset to UTC."
                )
            return await ctx.send(
                f"🕒 You haven't set a timezone, so repeating reminders use "
                f"**UTC** (currently **{now:%H:%M}**).\n"
                f"-# `{ctx.prefix}timezone Europe/London` — IANA names or "
                "shorthands like `PST`, `EST`, `UK`."
            )

        if zone.strip().lower() in ("clear", "reset", "none", "off", "utc"):
            await db.set_user_timezone(ctx.author.id, None)
            return await ctx.send("🕒 Timezone cleared — back to **UTC**.")

        canonical = normalise_timezone(zone)
        if canonical is None:
            return await ctx.send(
                f"❌ Don't recognise the timezone `{zone}`.\n"
                "Use an IANA name like `Europe/London`, `America/New_York` or "
                "`Asia/Tokyo` — the full list is at "
                "<https://en.wikipedia.org/wiki/List_of_tz_database_time_zones>. "
                "Shorthands like `PST`, `EST`, `UK` and `CET` work too."
            )
        await db.set_user_timezone(ctx.author.id, canonical)
        now = datetime.now(get_timezone(canonical))
        await ctx.send(
            f"🕒 Timezone set to **{canonical}** — it's **{now:%H:%M}** there "
            f"({now:%Z}). Repeating reminders will use it from now on."
        )

    @commands.hybrid_command(
        name="when", aliases=["ts", "timestamp"],
        description="Turn a time into a timestamp everyone sees in their own zone",
        help=(
            "Converts a time you type into a **Discord timestamp** — one that "
            "renders in each reader's own local time, so nobody has to do the "
            "maths and nobody turns up an hour late.\n\n"
            "It's read in *your* timezone (`timezone` to set it), and you get "
            "every format back with the code to copy.\n\n"
            "**Understands**\n"
            "`3pm` · `15:30` · `noon` · `midnight`\n"
            "`tonight` · `tomorrow` · `tomorrow 9am` · `today 5pm`\n"
            "`friday` · `friday 3pm` · `next monday`\n"
            "`aug 5` · `5th august` · `august 5 3pm`\n"
            "`2026-08-01` · `2026-08-01 14:00`\n"
            "`in 2 hours` · `in 30m` · `in 3 days`\n\n"
            "A bare time that's already gone rolls to tomorrow, and a bare "
            "weekday means the next one.\n\n"
            "**Examples**\n"
            "`when friday 8pm`\n"
            "`when in 3 hours`\n"
            "`when 2026-12-25 09:00`"
        ),
    )
    async def when(self, ctx: commands.Context, *, moment: str | None = None):
        if not moment or not moment.strip():
            return await ctx.send(
                f"❌ When? Try `{ctx.prefix}when friday 8pm`, "
                f"`{ctx.prefix}when tomorrow 9am`, or `{ctx.prefix}when in 2 hours`."
            )

        tz_name = await db.get_user_timezone(ctx.author.id)
        try:
            moment_dt = recurrence.parse_when(moment, tz_name=tz_name or "UTC")
        except ScheduleError as e:
            return await ctx.send(f"❌ {e}")

        unix = int(moment_dt.timestamp())
        # Every style Discord renders, so you can pick the one that fits the
        # sentence you're writing rather than memorising the letters.
        styles = [
            ("F", "Full"), ("f", "Short"), ("D", "Date"), ("d", "Short date"),
            ("t", "Time"), ("T", "Time + seconds"), ("R", "Relative"),
        ]
        embed = discord.Embed(
            title=f"🕒 {moment_dt:%A %d %B %Y, %H:%M}",
            description="\n".join(
                f"<t:{unix}:{code}>  ·  `<t:{unix}:{code}>`" for code, _ in styles
            ),
            color=discord.Color.blurple(),
        )
        embed.set_footer(
            text=(f"Read as {tz_name or 'UTC'}"
                  + ("" if tz_name else f" — {ctx.prefix}timezone to change that")
                  + " · everyone sees these in their own timezone")
        )
        await ctx.send(embed=embed)

    @commands.hybrid_command(
        name="time", aliases=["localtime", "clock"],
        description="What time is it for you, someone else, or somewhere",
        help=(
            "`time` — your local time\n"
            "`time @member` — theirs, if they've set a timezone (worth a "
            "check before you ping someone at 3am)\n"
            "`time Tokyo` — the current time in any zone; IANA names, "
            "shorthands like `PST`, and bare city names all work\n\n"
            "**Examples**\n"
            "`time`\n"
            "`time @alex`\n"
            "`time Europe/Berlin`"
        ),
    )
    async def time_cmd(self, ctx: commands.Context, *, target: str | None = None):
        if not target or not target.strip():
            tz_name = await db.get_user_timezone(ctx.author.id)
            now = datetime.now(get_timezone(tz_name))
            if tz_name:
                return await ctx.send(
                    f"🕒 It's **{now:%H:%M}** for you — {now:%A %d %B}, "
                    f"{tz_name} ({now:%Z})."
                )
            return await ctx.send(
                f"🕒 **{now:%H:%M} UTC** — you haven't set a timezone.\n"
                f"-# `{ctx.prefix}timezone Europe/London` (or yours) and this "
                "becomes your actual local time."
            )

        # A member first: "time @dave" is the common case, and a display name
        # could otherwise collide with a city.
        member = None
        try:
            member = await commands.MemberConverter().convert(ctx, target.strip())
        except commands.BadArgument:
            pass

        if member is not None:
            tz_name = await db.get_user_timezone(member.id)
            if not tz_name:
                return await ctx.send(
                    f"🕒 **{member.display_name}** hasn't set a timezone, so I "
                    f"don't know. They can with `{ctx.prefix}timezone <zone>`.",
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            now = datetime.now(get_timezone(tz_name))
            hint = ""
            if now.hour < 8 or now.hour >= 23:
                hint = " 🌙 *probably asleep*"
            return await ctx.send(
                f"🕒 It's **{now:%H:%M}** for **{member.display_name}** — "
                f"{now:%A %d %B}, {tz_name}.{hint}",
                allowed_mentions=discord.AllowedMentions.none(),
            )

        canonical = normalise_timezone(target)
        if canonical is None:
            return await ctx.send(
                f"❌ `{target}` isn't a member I can find or a timezone I know.\n"
                "-# Try a mention, or a zone like `Europe/London`, `PST`, `Tokyo`."
            )
        now = datetime.now(get_timezone(canonical))
        await ctx.send(
            f"🕒 It's **{now:%H:%M}** in **{canonical}** — "
            f"{now:%A %d %B} ({now:%Z})."
        )

    @commands.hybrid_group(
        name="reminders", invoke_without_command=True,
        description="List or cancel your pending reminders",
        help=(
            "`reminders` — list your pending reminders with ids\n"
            "`reminders cancel <id>` — cancel one of yours (repeating included)"
        ),
    )
    async def reminders(self, ctx: commands.Context):
        rows = await db.list_reminders(ctx.author.id)
        if not rows:
            return await ctx.send(
                f"⏰ No pending reminders. Set one with "
                f"`{ctx.prefix}remindme 10m <text>` or "
                f"`{ctx.prefix}remindme every weekday 9am <text>`."
            )
        repeating = [r for r in rows if r.get("repeat_kind")]
        one_shot  = [r for r in rows if not r.get("repeat_kind")]

        def _line(r: dict) -> str:
            where = f"<#{r['channel_id']}>" if r["guild_id"] else "DM"
            sched = recurrence.Schedule.from_db(r.get("repeat_kind"),
                                                r.get("repeat_spec"))
            when  = f"<t:{int(r['due_at'])}:R>"
            if sched is not None:
                fired = (f" · fired {r['fire_count']}×"
                         if r.get("fire_count") else "")
                return (f"`#{r['id']}` 🔁 **{sched.describe()}** — next {when} "
                        f"({where}){fired}\n⤷ {r['message']}")
            return f"`#{r['id']}` {when} ({where}) — {r['message']}"

        sections: list[str] = []
        if repeating:
            sections.append("**Repeating**\n"
                            + "\n".join(_line(r) for r in repeating[:10]))
        if one_shot:
            sections.append("**One-off**\n"
                            + "\n".join(_line(r) for r in one_shot[:15]))
        shown = min(len(repeating), 10) + min(len(one_shot), 15)
        more  = f"\n\n…and {len(rows) - shown} more" if len(rows) > shown else ""

        embed = discord.Embed(
            title="⏰ Your reminders",
            description="\n\n".join(sections) + more,
            color=discord.Color.blurple(),
        )
        embed.set_footer(text=f"{ctx.prefix}reminders cancel <id> to cancel one")
        await ctx.send(embed=embed)

    @reminders.command(name="cancel", aliases=["delete", "remove", "stop"],
                       description="Cancel one of your reminders by id")
    async def reminders_cancel(self, ctx: commands.Context, reminder_id: int):
        if await db.delete_reminder(reminder_id, ctx.author.id):
            await ctx.send(f"🗑️ Reminder #{reminder_id} cancelled.")
        else:
            await ctx.send(f"❌ No reminder #{reminder_id} that belongs to you. "
                           f"`{ctx.prefix}reminders` shows yours.")

    # ── delivery loop ────────────────────────────────────────────────────────

    @tasks.loop(seconds=15)
    async def deliver(self) -> None:
        due = await db.pop_due_reminders(time.time())
        for r in due:
            sched = recurrence.Schedule.from_db(r.get("repeat_kind"),
                                                r.get("repeat_spec"))
            if sched is not None and r.get("next_due"):
                footer = (f"-# 🔁 {sched.describe()} — next "
                          f"<t:{int(r['next_due'])}:R> · "
                          f"`reminders cancel {r['id']}` to stop")
            else:
                footer = ("-# set " + fmt_duration_long(
                    max(1, int(time.time() - r["created_at"]))) + " ago")
            text = f"⏰ <@{r['user_id']}> — {r['message']}\n{footer}"

            channel = self.bot.get_channel(r["channel_id"])
            try:
                if channel is not None:
                    await channel.send(text)
                    continue
            except discord.HTTPException as e:
                log.warning("Reminder #%s channel send failed: %s", r["id"], e)
            # Channel gone or send failed — fall back to a DM.
            try:
                user = self.bot.get_user(r["user_id"]) \
                       or await self.bot.fetch_user(r["user_id"])
                await user.send(f"⏰ {r['message']}\n-# (the channel you set this "
                                "reminder in is no longer reachable)")
            except discord.HTTPException as e:
                log.warning("Reminder #%s undeliverable: %s", r["id"], e)

    @deliver.before_loop
    async def _before_deliver(self) -> None:
        await self.bot.wait_until_ready()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(RemindersCog(bot))
