"""
Server message statistics — who's talking, and where.

Commands (all under the `stats` group):
  -stats                    server overview: top senders + busiest channels
  -stats server             same as bare `stats` (so /stats server works too)
  -stats user [@member]     one member's totals, rank, and top channels
  -stats channel [#chan]    one channel's totals + its top senders
  -stats scan               backfill counts from message history (admin only)

Tracking is best-effort, the same way the emoji cog works: Discord exposes no
"messages per user" API, so the bot counts messages live from the moment it's
running. `stats scan` walks channel history to backfill (and is authoritative
for the channels it can read, so it's safe to re-run).

Counts are stored per (guild, channel, user) in the `message_stats` table, so
the same data answers both "across the whole server" and "in this channel".
"""
from __future__ import annotations

import time
from datetime import datetime, timezone as dtimezone
from typing import Optional

import discord
from discord.ext import commands

from core import db
from core.logging_setup import get_logger
# Emoji parsing lives with the emoji cog; the stats system reuses it for the
# unified scan + `stats emoji`. (emoji.py imports nothing from here, so no cycle.)
from cogs.emoji import CUSTOM_EMOJI_RE

log = get_logger(__name__)

# Only count "real" messages — not system notices (joins, pins, boosts, …),
# which are authored by a user but aren't something they typed.
_COUNTED_TYPES = {discord.MessageType.default, discord.MessageType.reply}

_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

# Ceiling on a single banked voice session. Only used when the real end time
# is unknowable — a restart or crash while someone was still in a call. A
# genuine 14-hour session still records in full, because its close event
# arrives normally; this only stops an offline weekend becoming a leaderboard.
MAX_SESSION = 12 * 3600


def _fmt_hours(seconds: float) -> str:
    """Voice durations read better in h/m than the h:m:s of a track length."""
    total_m = int(seconds // 60)
    h, m = divmod(total_m, 60)
    if h and m:
        return f"{h:,}h {m}m"
    if h:
        return f"{h:,}h"
    if m:
        return f"{m}m"
    return f"{int(seconds)}s"


def _rank_badge(position: int) -> str:
    """🥇/🥈/🥉 for the top three, `#4`, `#5`, … after that."""
    return _MEDALS.get(position, f"`#{position}`")


# ──────────────────────────────────────────────────────────────────────────────
# Emoji/sticker list helpers — moved here from the emoji cog so all the usage
# stats live in one place.
# ──────────────────────────────────────────────────────────────────────────────

def _parse_list_opts(opts: str) -> tuple[str, str]:
    """Parse '--native --sort=alpha' style options. Returns (filter, sort)."""
    s = opts.lower()
    filter_mode = "all"
    if "--native"   in s: filter_mode = "native"
    elif "--external" in s: filter_mode = "external"

    sort_mode = "used"
    if   "--sort=alpha"  in s or "--alpha"  in s: sort_mode = "alpha"
    elif "--sort=recent" in s or "--recent" in s: sort_mode = "recent"
    return filter_mode, sort_mode


def _render_emoji(bot, emoji_id: int, name: str, is_animated, is_native) -> str:
    """Render an emoji inline: the real emoji if it's ours, else a `<:name:id>`
    ref (which Discord shows if it can resolve it) with a code fallback."""
    if is_native:
        e = bot.get_emoji(emoji_id)
        return str(e) if e else f"`:{name}:`"
    pre = "a" if is_animated else ""
    return f"<{pre}:{name}:{emoji_id}>"


def _render_emoji_row(bot, r: dict) -> str:
    total = r["in_messages"] + r["in_reactions"]
    rendered = _render_emoji(bot, r["emoji_id"], r["name"], r["is_animated"], r["is_native"])
    tag = "" if r["is_native"] else "  *(external)*"
    if total == 0:
        return f"{rendered} `{r['name']}` — *unused*{tag}"
    if r["in_messages"] > 0 and r["in_reactions"] > 0:
        return (f"{rendered} `{r['name']}` — **{total}** uses "
                f"({r['in_messages']} msg / {r['in_reactions']} rxn){tag}")
    return f"{rendered} `{r['name']}` — **{total}** uses{tag}"


def _render_sticker_row(r: dict) -> str:
    tag = "" if r["is_native"] else "  *(external)*"
    uses = f"**{r['used_count']}** uses" if r["used_count"] else "*unused*"
    return f"`{r['name']}` — {uses}{tag}"


class _PageView(discord.ui.View):
    """Generic prev/next paginator: each row rendered via `render_row(row)`.
    Only the original requester can flip pages; ✖ deletes the message."""
    PER_PAGE        = 10
    DEFAULT_TIMEOUT = 300

    def __init__(self, *, author_id: int, title: str,
                 rows: list, render_row, color=discord.Color.blurple()):
        super().__init__(timeout=self.DEFAULT_TIMEOUT)
        self.author_id   = author_id
        self.title       = title
        self.rows        = rows
        self.render_row  = render_row
        self.color       = color
        self.idx         = 0
        self.total_pages = max(1, (len(rows) + self.PER_PAGE - 1) // self.PER_PAGE)
        self._refresh_buttons()

    def _refresh_buttons(self):
        at_start = self.idx == 0
        at_end   = self.idx >= self.total_pages - 1
        self.first_btn.disabled = at_start
        self.prev_btn.disabled  = at_start
        self.next_btn.disabled  = at_end
        self.last_btn.disabled  = at_end

    def build_embed(self) -> discord.Embed:
        start = self.idx * self.PER_PAGE
        page  = self.rows[start:start + self.PER_PAGE]
        lines = [self.render_row(r) for r in page] if page else ["*(empty)*"]
        embed = discord.Embed(title=self.title, description="\n".join(lines),
                              color=self.color)
        embed.set_footer(text=f"Page {self.idx + 1} of {self.total_pages}  ·  "
                              f"{len(self.rows)} total")
        return embed

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "❌ This menu belongs to someone else.", ephemeral=True)
            return False
        return True

    async def _go_to(self, interaction: discord.Interaction, new_idx: int):
        self.idx = max(0, min(self.total_pages - 1, new_idx))
        self._refresh_buttons()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    @discord.ui.button(label="⏮", style=discord.ButtonStyle.secondary)
    async def first_btn(self, interaction, _b):
        await self._go_to(interaction, 0)

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary)
    async def prev_btn(self, interaction, _b):
        await self._go_to(interaction, self.idx - 1)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary)
    async def next_btn(self, interaction, _b):
        await self._go_to(interaction, self.idx + 1)

    @discord.ui.button(label="⏭", style=discord.ButtonStyle.secondary)
    async def last_btn(self, interaction, _b):
        await self._go_to(interaction, self.total_pages - 1)

    @discord.ui.button(label="✖", style=discord.ButtonStyle.danger)
    async def close_btn(self, interaction: discord.Interaction, _b):
        try:
            await interaction.message.delete()
        except Exception:
            await interaction.response.defer()
        self.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Backfill scan UI — a Cancel button next to the live progress message.
# ──────────────────────────────────────────────────────────────────────────────

class _Canceller:
    def __init__(self) -> None:
        self.cancelled = False


class _CancelView(discord.ui.View):
    def __init__(self, author_id: int, canceller: _Canceller):
        super().__init__(timeout=None)   # scans can run for a while
        self.author_id = author_id
        self.canceller = canceller

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, emoji="✖")
    async def cancel_btn(self, interaction: discord.Interaction, button):
        if interaction.user.id != self.author_id:
            return await interaction.response.send_message(
                "❌ Only the person who started the scan can cancel.",
                ephemeral=True,
            )
        self.canceller.cancelled = True
        button.disabled = True
        button.label    = "Cancelling…"
        await interaction.response.edit_message(view=self)


class ServerStats(commands.Cog, name="Server Stats"):
    """Live message counting + the `stats` command group."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # Cached so the voice listener doesn't hit SQLite on every state change
        # (which fires for mute, deafen, stream and camera toggles too).
        self._voice_optouts: set[int] = set()

    async def cog_load(self) -> None:
        try:
            self._voice_optouts = await db.voice_optout_ids()
        except Exception:
            log.exception("couldn't load voice opt-outs — assuming none")

    async def cog_check(self, ctx: commands.Context) -> bool:
        # Every stat here is per-guild; there's nothing to show in a DM.
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    # ── Live tracking ────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Reconcile voice sessions left open by a restart.

        Anyone the DB thinks is still in a call but isn't gets their session
        closed with a capped credit; anyone who *is* in a call but has no open
        row (joined while the bot was down) gets one opened now. Without this,
        a nightly restart would quietly drop every in-progress session.
        """
        try:
            await self._reconcile_voice_sessions()
        except Exception:
            log.exception("voice session reconciliation failed")

    async def _reconcile_voice_sessions(self) -> None:
        now = time.time()
        open_rows = {(r["guild_id"], r["user_id"]): r
                     for r in await db.voice_open_sessions()}

        present: set[tuple[int, int]] = set()
        for guild in self.bot.guilds:
            for channel in guild.voice_channels:
                for member in channel.members:
                    if member.bot or member.id in self._voice_optouts:
                        continue
                    present.add((guild.id, member.id))
                    row = open_rows.get((guild.id, member.id))
                    if row is not None and row["channel_id"] == channel.id:
                        continue    # still going — leave the clock running
                    if row is not None:
                        # Moved channels while we were down: bank what we can.
                        await db.voice_session_close(
                            guild.id, member.id, now, max_seconds=MAX_SESSION)
                    await db.voice_session_open(guild.id, member.id, channel.id, now)

        stale = 0
        for (gid, uid), _row in open_rows.items():
            if (gid, uid) in present:
                continue
            # They left while the bot was down. We can't know when, so cap the
            # credit rather than banking however long the bot was offline.
            await db.voice_session_close(gid, uid, now, max_seconds=MAX_SESSION)
            stale += 1
        if stale:
            log.info("Closed %d voice session(s) left open by a restart", stale)

    @commands.Cog.listener()
    async def on_voice_state_update(self, member: discord.Member,
                                    before: discord.VoiceState,
                                    after: discord.VoiceState) -> None:
        """Bank voice time on leave, start the clock on join.

        Only channel changes matter — this event also fires for mute, deafen,
        self-stream and camera toggles, where before.channel == after.channel
        and there's nothing to record.
        """
        if member.bot or before.channel == after.channel:
            return
        if member.id in self._voice_optouts:
            return
        guild_id = (after.channel or before.channel).guild.id
        now = time.time()
        try:
            if before.channel is not None:
                await db.voice_session_close(guild_id, member.id, now,
                                             max_seconds=MAX_SESSION)
            if after.channel is not None:
                await db.voice_session_open(guild_id, member.id,
                                            after.channel.id, now)
        except Exception:
            log.exception("voice time record failed for %s", member.id)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or message.guild is None:
            return
        if message.type not in _COUNTED_TYPES:
            return
        try:
            await db.record_message(
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                user_id=message.author.id,
                now=message.created_at.timestamp(),
            )
        except Exception:
            log.exception("message stat record failed")

    # ── stats group ──────────────────────────────────────────────────────────

    @commands.hybrid_group(
        name="stats", invoke_without_command=True,
        description="Message stats: top senders, busiest channels, per-user/-channel",
        help=(
            "Who's talking, where, and which emoji/stickers get used. Counting is "
            "live; run `stats scan` once to backfill from history.\n\n"
            "**Subcommands**\n"
            "`stats` / `stats server` — top senders + busiest channels\n"
            "`stats user [@member]` — a member's messages, rank, top channels & emoji\n"
            "`stats channel [#channel]` — a channel's total + its top senders\n"
            "`stats emoji [@member]` — emoji usage: server leaderboard, or one "
            "member's most-used\n"
            "`stats sticker` — sticker usage leaderboard\n"
            "`stats voice [@member]` — time spent in voice channels "
            "(`optout` to be excluded)\n"
            "`stats scan` — backfill messages + emoji + stickers (needs *Manage Server*)"
        ),
    )
    async def stats(self, ctx: commands.Context):
        await self._send_server_overview(ctx)

    @stats.command(
        name="server", aliases=["overview", "all"],
        description="Server-wide top senders and busiest channels",
        help="Top message senders across the whole server, plus the busiest "
             "channels and overall totals.",
    )
    async def stats_server(self, ctx: commands.Context):
        await self._send_server_overview(ctx)

    async def _send_server_overview(self, ctx: commands.Context):
        gid = ctx.guild.id
        totals = await db.guild_message_totals(gid)
        if totals["total"] == 0:
            return await ctx.send(
                "📊 No message stats yet. Counting starts while I'm running — "
                f"come back after some chatter, or run `{ctx.prefix}stats scan` "
                "to backfill from history."
            )

        senders  = await db.top_senders(gid, limit=10)
        channels = await db.top_channels(gid, limit=5)

        embed = discord.Embed(
            title=f"📊 Message stats — {ctx.guild.name}",
            color=discord.Color.blurple(),
        )
        embed.description = (
            f"**{totals['total']:,}** messages from **{totals['users']:,}** "
            f"people across **{totals['channels']:,}** channels."
        )
        if senders:
            embed.add_field(
                name="🏆 Top senders",
                value="\n".join(
                    f"{_rank_badge(i)} <@{r['user_id']}> — **{r['total']:,}**"
                    for i, r in enumerate(senders, 1)
                ),
                inline=False,
            )
        if channels:
            embed.add_field(
                name="💬 Busiest channels",
                value="\n".join(
                    f"{_rank_badge(i)} <#{r['channel_id']}> — **{r['total']:,}**"
                    for i, r in enumerate(channels, 1)
                ),
                inline=False,
            )
        embed.set_footer(text=f"{ctx.prefix}stats user · {ctx.prefix}stats channel")
        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    @stats.command(
        name="user", aliases=["member", "me"],
        description="A member's message total, server rank, and top channels",
        help="Shows how many messages a member has sent, where they rank on the "
             "server, and the channels they post in most. Defaults to you; pass "
             "a mention or name to look someone else up.",
    )
    async def stats_user(self, ctx: commands.Context,
                         member: Optional[discord.Member] = None):
        member = member or ctx.author
        data = await db.user_message_stats(ctx.guild.id, member.id)
        if data["total"] == 0:
            who = "You have" if member == ctx.author else f"{member.display_name} has"
            return await ctx.send(
                f"📊 {who} no tracked messages yet. "
                f"(Run `{ctx.prefix}stats scan` to backfill from history.)"
            )

        embed = discord.Embed(
            title=f"📊 {member.display_name}'s message stats",
            color=member.color if member.color.value else discord.Color.blurple(),
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        rank_line = f" · rank **{_rank_badge(data['rank']).strip('`')}** on the server"
        embed.description = f"**{data['total']:,}** messages{rank_line}"
        if data["last_message_at"]:
            embed.description += (
                f"\nLast seen talking "
                f"<t:{int(data['last_message_at'])}:R>."
            )

        top = data["channels"][:5]
        if top:
            embed.add_field(
                name="Top channels",
                value="\n".join(
                    f"{_rank_badge(i)} <#{c['channel_id']}> — **{c['count']:,}**"
                    for i, c in enumerate(top, 1)
                ),
                inline=False,
            )

        # Emoji usage (per-user table).
        em = await db.user_emoji_stats(ctx.guild.id, member.id)
        if em["total"]:
            shown = " ".join(self._emoji_str(ctx.guild, e) for e in em["emojis"][:8])
            embed.add_field(
                name="😀 Emoji",
                value=(f"**{em['total']:,}** used · **{em['distinct']}** different\n"
                       f"Most-used: {shown}  (`{em['emojis'][0]['name']}` ×"
                       f"{em['emojis'][0]['count']:,})"),
                inline=False,
            )
        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    def _emoji_str(self, guild, e: dict) -> str:
        """Render a per-user emoji row entry (computes is_native from the guild)."""
        is_native = any(g.id == e["emoji_id"] for g in guild.emojis)
        return _render_emoji(self.bot, e["emoji_id"], e["name"],
                             e["is_animated"], is_native)

    @stats.command(
        name="channel", aliases=["chan"],
        description="A channel's message total and its top senders",
        help="Total messages tracked in a channel plus who posts there most. "
             "Defaults to the current channel; pass a #mention for another.",
    )
    async def stats_channel(self, ctx: commands.Context,
                            channel: Optional[discord.TextChannel] = None):
        channel = channel or ctx.channel
        total = await db.channel_message_total(ctx.guild.id, channel.id)
        if total == 0:
            return await ctx.send(
                f"📊 No tracked messages in {channel.mention} yet. "
                f"(Run `{ctx.prefix}stats scan` to backfill from history.)"
            )

        senders = await db.top_senders(ctx.guild.id, channel_id=channel.id, limit=10)
        embed = discord.Embed(
            title=f"📊 #{channel.name} — message stats",
            description=f"**{total:,}** messages tracked here.",
            color=discord.Color.blurple(),
        )
        if senders:
            embed.add_field(
                name="🏆 Top senders here",
                value="\n".join(
                    f"{_rank_badge(i)} <@{r['user_id']}> — **{r['total']:,}**"
                    for i, r in enumerate(senders, 1)
                ),
                inline=False,
            )
        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    # ── Emoji / sticker usage ────────────────────────────────────────────────

    @stats.command(
        name="emoji", aliases=["emojis"],
        description="Emoji usage — server leaderboard, or one member's most-used",
        help=(
            "With no argument, shows the server's emoji usage leaderboard "
            "(paginated). Pass a member to see *their* most-used emoji instead.\n\n"
            "**Filters/sorts** (leaderboard only): `--native` / `--external`, "
            "`--sort=alpha` / `--sort=recent` (default: by uses).\n\n"
            "**Examples**\n"
            "`stats emoji` · `stats emoji @ben` · `stats emoji --native`"
        ),
    )
    async def stats_emoji(self, ctx: commands.Context,
                          member: Optional[discord.Member] = None, *, opts: str = ""):
        if member is not None:
            return await self._send_user_emoji(ctx, member)

        filter_mode, sort_mode = _parse_list_opts(opts)
        rows = await db.list_emoji_usage(
            ctx.guild.id, filter_mode=filter_mode, sort_mode=sort_mode)
        # Include this server's unused emoji too (an inventory, not just clicks).
        if filter_mode in ("all", "native"):
            tracked = {r["emoji_id"] for r in rows}
            for e in ctx.guild.emojis:
                if e.id in tracked:
                    continue
                rows.append({
                    "emoji_id": e.id, "name": e.name, "is_native": 1,
                    "is_animated": int(e.animated), "in_messages": 0,
                    "in_reactions": 0, "last_used": 0.0,
                })
            if sort_mode == "alpha":
                rows.sort(key=lambda r: r["name"].lower())
            elif sort_mode == "recent":
                rows.sort(key=lambda r: -r["last_used"])
            else:
                rows.sort(key=lambda r: -(r["in_messages"] + r["in_reactions"]))

        if not rows:
            return await ctx.send(
                f"📊 No emoji usage tracked yet. Run `{ctx.prefix}stats scan` to "
                "backfill from history.")
        title = ("🙂 Emoji usage"
                 + (f" — *{filter_mode}*" if filter_mode != "all" else "")
                 + f" ({sort_mode})")
        view = _PageView(author_id=ctx.author.id, title=title, rows=rows,
                         render_row=lambda r: _render_emoji_row(self.bot, r))
        await ctx.send(embed=view.build_embed(), view=view)

    async def _send_user_emoji(self, ctx: commands.Context, member: discord.Member):
        data = await db.user_emoji_stats(ctx.guild.id, member.id)
        if data["total"] == 0:
            who = "You haven't" if member == ctx.author \
                else f"{member.display_name} hasn't"
            return await ctx.send(
                f"😶 {who} used any custom emoji I've tracked. "
                f"(Run `{ctx.prefix}stats scan` to backfill.)")
        embed = discord.Embed(
            title=f"😀 {member.display_name}'s emoji",
            color=member.color if member.color.value else discord.Color.blurple(),
            description=(f"**{data['total']:,}** emoji used · "
                         f"**{data['distinct']}** different"),
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        top = data["emojis"][:12]
        embed.add_field(
            name="Most-used",
            value="\n".join(
                f"{_rank_badge(i)} {self._emoji_str(ctx.guild, e)} "
                f"`:{e['name']}:` — **{e['count']:,}**"
                for i, e in enumerate(top, 1)
            ),
            inline=False,
        )
        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    @stats.command(
        name="sticker", aliases=["stickers"],
        description="Sticker usage leaderboard",
        help=(
            "Sticker usage leaderboard (paginated). Filters/sorts: `--native` / "
            "`--external`, `--sort=alpha` / `--sort=recent` (default: by uses)."
        ),
    )
    async def stats_sticker(self, ctx: commands.Context, *, opts: str = ""):
        filter_mode, sort_mode = _parse_list_opts(opts)
        rows = await db.list_sticker_usage(
            ctx.guild.id, filter_mode=filter_mode, sort_mode=sort_mode)
        if filter_mode in ("all", "native"):
            tracked = {r["sticker_id"] for r in rows}
            for s in ctx.guild.stickers:
                if s.id in tracked:
                    continue
                rows.append({
                    "sticker_id": s.id, "name": s.name, "is_native": 1,
                    "used_count": 0, "last_used": 0.0,
                })
            if sort_mode == "alpha":
                rows.sort(key=lambda r: r["name"].lower())
            elif sort_mode == "recent":
                rows.sort(key=lambda r: -r["last_used"])
            else:
                rows.sort(key=lambda r: -r["used_count"])

        if not rows:
            return await ctx.send(
                f"📊 No sticker usage tracked yet. Run `{ctx.prefix}stats scan` to "
                "backfill from history.")
        title = ("🖼️ Sticker usage"
                 + (f" — *{filter_mode}*" if filter_mode != "all" else "")
                 + f" ({sort_mode})")
        view = _PageView(author_id=ctx.author.id, title=title, rows=rows,
                         render_row=_render_sticker_row)
        await ctx.send(embed=view.build_embed(), view=view)

    # ── Voice time ───────────────────────────────────────────────────────────

    @stats.command(
        name="voice", aliases=["vc"],
        description="Who spends the most time in voice channels",
        help=(
            "Voice-channel time — the leaderboard, the busiest channels, and "
            "anyone currently in a call.\n\n"
            "Only **durations** are recorded: no audio, no transcript, no log "
            "of who was in a call with whom. Counting is live from whenever "
            "the bot is running, so it can't be backfilled the way "
            "`stats scan` backfills messages.\n\n"
            "**Usage**\n"
            "`stats voice` — server leaderboard + busiest channels\n"
            "`stats voice @member` — one person's time, by channel\n"
            "`stats voice optout` — stop counting you, and delete your rows\n"
            "`stats voice optin` — start counting again\n\n"
            "Sessions in progress are shown separately and get banked when "
            "the person leaves."
        ),
    )
    async def stats_voice(self, ctx: commands.Context, *, target: str = ""):
        arg = target.strip().lower()

        if arg in ("optout", "opt-out", "off", "exclude"):
            await db.set_voice_optout(ctx.author.id, True)
            self._voice_optouts.add(ctx.author.id)
            return await ctx.send(
                "🔇 You're out of voice-time tracking, and the rows I already "
                f"had for you are deleted. `{ctx.prefix}stats voice optin` to "
                "start counting again."
            )
        if arg in ("optin", "opt-in", "on", "include"):
            await db.set_voice_optout(ctx.author.id, False)
            self._voice_optouts.discard(ctx.author.id)
            # Start their clock now if they're already sitting in a call.
            voice = ctx.author.voice
            if voice and voice.channel:
                await db.voice_session_open(ctx.guild.id, ctx.author.id,
                                            voice.channel.id, time.time())
            return await ctx.send("🔊 Counting your voice time again.")

        if arg:
            try:
                member = await commands.MemberConverter().convert(ctx, target)
            except commands.BadArgument:
                return await ctx.send(
                    f"❌ No member matching `{target}`. Use a mention, or "
                    f"`{ctx.prefix}stats voice` for the leaderboard "
                    f"(`optout` / `optin` also work here)."
                )
            return await self._send_voice_user(ctx, member)

        await self._send_voice_overview(ctx)

    async def _live_sessions(self, guild_id: int) -> dict[int, float]:
        """{user_id: seconds so far} for calls in progress in this guild."""
        now = time.time()
        return {
            r["user_id"]: max(0.0, now - r["joined_at"])
            for r in await db.voice_open_sessions()
            if r["guild_id"] == guild_id
        }

    async def _send_voice_overview(self, ctx: commands.Context):
        gid = ctx.guild.id
        total, users = await db.voice_guild_total(gid)
        live = await self._live_sessions(gid)

        if total <= 0 and not live:
            return await ctx.send(
                "🔊 No voice time recorded yet. I count it live while I'm "
                "running — unlike messages, there's no history to backfill "
                "from, so this fills up from here on."
            )

        board    = await db.voice_leaderboard(gid, limit=10)
        channels = await db.voice_channel_totals(gid, limit=5)

        embed = discord.Embed(
            title=f"🔊 Voice time — {ctx.guild.name}",
            color=discord.Color.blurple(),
        )
        embed.description = (
            f"**{_fmt_hours(total)}** across **{users:,}** "
            f"{'person' if users == 1 else 'people'}."
        )
        if board:
            embed.add_field(
                name="🏆 Most time in voice",
                value="\n".join(
                    f"{_rank_badge(i)} <@{r['user_id']}> — "
                    f"**{_fmt_hours(r['seconds'])}** "
                    f"*({r['sessions']:,} session{'s' if r['sessions'] != 1 else ''})*"
                    for i, r in enumerate(board, 1)
                ),
                inline=False,
            )
        if channels:
            embed.add_field(
                name="📢 Busiest channels",
                value="\n".join(
                    f"{_rank_badge(i)} <#{r['channel_id']}> — "
                    f"**{_fmt_hours(r['seconds'])}**"
                    for i, r in enumerate(channels, 1)
                ),
                inline=False,
            )
        if live:
            top_live = sorted(live.items(), key=lambda kv: -kv[1])[:5]
            embed.add_field(
                name="🟢 In a call now",
                value="\n".join(
                    f"<@{uid}> — **{_fmt_hours(secs)}** so far"
                    for uid, secs in top_live
                ),
                inline=False,
            )
        embed.set_footer(
            text=f"{ctx.prefix}stats voice @member · "
                 f"{ctx.prefix}stats voice optout to be excluded"
        )
        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    async def _send_voice_user(self, ctx: commands.Context, member: discord.Member):
        gid  = ctx.guild.id
        rows = await db.voice_user_detail(gid, member.id)
        live = (await self._live_sessions(gid)).get(member.id, 0.0)

        if not rows and not live:
            if member.id in self._voice_optouts:
                return await ctx.send(
                    f"🔇 **{member.display_name}** has opted out of voice-time "
                    "tracking."
                )
            return await ctx.send(
                f"🔊 No voice time recorded for **{member.display_name}** yet."
            )

        banked = sum(r["seconds"] for r in rows)
        board  = await db.voice_leaderboard(gid, limit=1000)
        rank   = next((i for i, r in enumerate(board, 1)
                       if r["user_id"] == member.id), None)
        sessions = sum(r["sessions"] for r in rows)

        embed = discord.Embed(
            title=f"🔊 Voice time — {member.display_name}",
            color=member.color if member.color.value else discord.Color.blurple(),
        )
        embed.set_thumbnail(url=member.display_avatar.url)
        parts = [f"**{_fmt_hours(banked)}** over "
                 f"**{sessions:,}** session{'s' if sessions != 1 else ''}"]
        if rank:
            parts.append(f"ranked **{_rank_badge(rank).strip('`')}** on the server")
        embed.description = " · ".join(parts)
        if live:
            embed.description += f"\n🟢 In a call right now — **{_fmt_hours(live)}** so far."

        if rows:
            embed.add_field(
                name="📢 By channel",
                value="\n".join(
                    f"<#{r['channel_id']}> — **{_fmt_hours(r['seconds'])}**"
                    for r in rows[:10]
                ),
                inline=False,
            )
        last = max((r["last_seen"] for r in rows), default=0)
        if last:
            embed.set_footer(text="Last in voice")
            embed.timestamp = datetime.fromtimestamp(last, tz=dtimezone.utc)
        await ctx.send(embed=embed, allowed_mentions=discord.AllowedMentions.none())

    # ── Backfill scan ────────────────────────────────────────────────────────

    @stats.command(
        name="scan",
        description="Backfill messages + emoji + stickers from channel history",
        help=(
            "Walks every text channel the bot can read and, in one pass, counts "
            "each person's messages, the emoji they used, and stickers + "
            "reactions — then writes it all to the database.\n\n"
            "Run this once to fill in history; the live listeners handle new "
            "activity from then on.\n\n"
            "**Message counts replace** the per-channel totals (safe to re-run). "
            "**Emoji & sticker counts are added** (reactions can't be cleanly "
            "rebuilt), so re-running double-counts those — run it once.\n\n"
            "Threads aren't walked — main text channels only. Requires *Manage "
            "Server*. Big servers can take a few minutes; live progress + Cancel."
        ),
    )
    @commands.has_guild_permissions(manage_guild=True)
    async def stats_scan(self, ctx: commands.Context):
        canceller = _Canceller()
        view      = _CancelView(ctx.author.id, canceller)
        me        = ctx.guild.me
        channels  = list(ctx.guild.text_channels)

        status = await ctx.send(
            f"🔍 Scanning **{len(channels)} text channels** for messages, emoji "
            f"& stickers…\n⏱️ This can take a while on busy servers.",
            view=view,
        )

        msg_counts:  dict[tuple[int, int], dict] = {}   # (channel,user) -> {count,last}
        emojis:      dict[int, dict] = {}               # emoji_id -> aggregate
        user_emojis: dict[tuple[int, int], dict] = {}   # (user,emoji) -> {count,...}
        stickers:    dict[int, dict] = {}               # sticker_id -> aggregate
        scanned_ids: set[int] = set()
        skipped     = 0
        total_msgs  = 0
        start       = time.time()
        last_update = 0.0
        native_emoji_ids   = {e.id for e in ctx.guild.emojis}
        native_sticker_ids = {s.id for s in ctx.guild.stickers}

        for channel in channels:
            if canceller.cancelled:
                break

            perms = channel.permissions_for(me)
            if not (perms.read_messages and perms.read_message_history):
                skipped += 1
                continue

            try:
                async for message in channel.history(limit=None):
                    if canceller.cancelled:
                        break
                    if message.author.bot or message.type not in _COUNTED_TYPES:
                        continue

                    total_msgs += 1
                    ts  = message.created_at.timestamp()
                    uid = message.author.id

                    key = (channel.id, uid)
                    mc = msg_counts.get(key)
                    if mc is None:
                        msg_counts[key] = {"count": 1, "last": ts}
                    else:
                        mc["count"] += 1
                        if ts > mc["last"]:
                            mc["last"] = ts

                    # Custom emoji in the body, de-duped per message → aggregate
                    # (msg) + per-user.
                    seen: set[int] = set()
                    for m in CUSTOM_EMOJI_RE.finditer(message.content):
                        animated = m.group(1) == "a"
                        name     = m.group(2)
                        eid      = int(m.group(3))
                        if eid in seen:
                            continue
                        seen.add(eid)
                        is_native = eid in native_emoji_ids
                        e = emojis.setdefault(eid, {
                            "name": name, "is_native": is_native,
                            "is_animated": animated, "msg_count": 0,
                            "rxn_count": 0, "last_used": 0.0})
                        e["msg_count"] += 1
                        e["last_used"]  = max(e["last_used"], ts)
                        e["name"], e["is_native"] = name, is_native
                        ue = user_emojis.setdefault((uid, eid), {
                            "name": name, "is_animated": animated,
                            "count": 0, "last_used": 0.0})
                        ue["count"] += 1
                        ue["last_used"] = max(ue["last_used"], ts)
                        ue["name"], ue["is_animated"] = name, animated

                    for st in message.stickers:
                        is_native = st.id in native_sticker_ids
                        s = stickers.setdefault(st.id, {
                            "name": st.name, "is_native": is_native,
                            "used_count": 0, "last_used": 0.0})
                        s["used_count"] += 1
                        s["last_used"]   = max(s["last_used"], ts)
                        s["name"], s["is_native"] = st.name, is_native

                    # Reactions → aggregate only (can't map to users cheaply).
                    for reaction in message.reactions:
                        re_ = reaction.emoji
                        if isinstance(re_, str) or getattr(re_, "id", None) is None:
                            continue
                        cnt = reaction.count - (1 if reaction.me else 0)
                        if cnt <= 0:
                            continue
                        is_native = re_.id in native_emoji_ids
                        e = emojis.setdefault(re_.id, {
                            "name": re_.name, "is_native": is_native,
                            "is_animated": bool(getattr(re_, "animated", False)),
                            "msg_count": 0, "rxn_count": 0, "last_used": 0.0})
                        e["rxn_count"] += cnt
                        e["last_used"]  = max(e["last_used"], ts)
                        e["name"], e["is_native"] = re_.name, is_native

                    now = time.time()
                    if now - last_update > 3.0:
                        last_update = now
                        elapsed = int(now - start)
                        try:
                            await status.edit(
                                content=(
                                    f"🔍 Scanning **#{channel.name}**\n"
                                    f"  📝 {total_msgs:,} messages\n"
                                    f"  🙂 {len(emojis):,} emoji · 🖼️ "
                                    f"{len(stickers):,} stickers\n"
                                    f"  ⏱️ {elapsed}s elapsed"
                                ),
                                view=view,
                            )
                        except discord.HTTPException:
                            pass
            except discord.Forbidden:
                skipped += 1
                continue
            except Exception:
                log.exception("stats scan failed in #%s", channel.name)
                continue

            # Only mark a channel authoritative if we finished it — a cancelled
            # channel has partial counts and must keep its live-tracked data.
            if canceller.cancelled:
                break
            scanned_ids.add(channel.id)

        view.stop()   # scan finished — release the timeout=None cancel view
        try:
            await status.edit(content="💾 Saving to database…", view=None)
            await db.replace_message_stats(
                ctx.guild.id, scanned_channel_ids=scanned_ids, counts=msg_counts)
            await db.bulk_record_scan(
                guild_id=ctx.guild.id, emojis=emojis, stickers=stickers)
            await db.bulk_record_user_emoji(ctx.guild.id, user_emojis)
        except Exception as e:
            log.exception("stats scan DB write failed")
            return await status.edit(content=f"❌ DB write failed: {e}", view=None)

        elapsed = int(time.time() - start)
        verdict = "✖ Scan cancelled (saved what was scanned)." \
            if canceller.cancelled else "✅ Scan complete!"
        await status.edit(
            content=(
                f"{verdict}\n"
                f"  📝 {total_msgs:,} messages ({len(scanned_ids)} channels)\n"
                f"  🙂 {len(emojis):,} emoji · 🖼️ {len(stickers):,} stickers\n"
                + (f"  🔒 {skipped} channels skipped (no read permission)\n"
                   if skipped else "")
                + f"  ⏱️ {elapsed}s\n"
                f"Try `{ctx.prefix}stats` · `{ctx.prefix}stats emoji`."
            ),
            view=None,
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ServerStats(bot))
