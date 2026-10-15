"""
Music "wrapped" — leaderboards built from fully-played tracks.

Every time a track plays to its *natural* end (not a skip — see
MusicCog._on_track_finished), one row lands in the `track_plays` table. This cog
just reads that log back as leaderboards:

  -topsongs [window]      most-played tracks in the server
  -topartists [window]    most-played artists
  -toplisteners [window]  who queued the most fully-played tracks
  -wrapped [@user] [window]  one member's personal recap

`window` is day / week / month / year / all (default: week for the boards,
all-time for `wrapped`). Because only completed plays are logged, dumping an
8,000-song playlist into the queue and skipping through it doesn't pad anyone's
stats.
"""
from __future__ import annotations

import time
from typing import Optional

import discord
from discord.ext import commands

from core import db
from core.formatters import fmt_duration_long
from core.logging_setup import get_logger

log = get_logger(__name__)

_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

# Window keyword → (label, seconds back). None seconds = all time.
_WINDOWS: dict[str, tuple[str, Optional[int]]] = {
    "day":     ("today", 86_400),
    "today":   ("today", 86_400),
    "24h":     ("the last 24h", 86_400),
    "week":    ("this week", 7 * 86_400),
    "7d":      ("the last 7 days", 7 * 86_400),
    "month":   ("this month", 30 * 86_400),
    "30d":     ("the last 30 days", 30 * 86_400),
    "year":    ("this year", 365 * 86_400),
    "all":     ("all time", None),
    "alltime": ("all time", None),
    "forever": ("all time", None),
}


def _rank(position: int) -> str:
    return _MEDALS.get(position, f"`#{position}`")


def _parse_window(token: str | None, *, default: str) -> tuple[str, Optional[float]]:
    """Resolve a window keyword to (label, since_timestamp|None)."""
    key = (token or default).lower().strip()
    label, secs = _WINDOWS.get(key, _WINDOWS[default])
    since = None if secs is None else time.time() - secs
    return label, since


class MusicStatsCog(commands.Cog, name="Music Stats"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_check(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    def _name_for(self, guild: discord.Guild, user_id: int) -> str:
        m = guild.get_member(user_id)
        return m.display_name if m else f"User {user_id}"

    # ── leaderboards ──────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="topsongs", aliases=["topsong"],
        description="Most-played songs (window: day/week/month/year/all)",
    )
    async def topsongs(self, ctx: commands.Context, window: str | None = None):
        label, since = _parse_window(window, default="week")
        rows = await db.top_tracks(ctx.guild.id, since=since, limit=10)
        if not rows:
            return await ctx.send(f"📭 No songs have finished playing {label} yet.")
        lines = []
        for i, r in enumerate(rows, 1):
            artist = f" — {r['artist']}" if r["artist"] else ""
            title = f"[{r['title']}]({r['url']})" if r["url"] else r["title"]
            lines.append(f"{_rank(i)} {title}{artist} · **{r['plays']}** play(s)")
        embed = discord.Embed(
            title=f"🎵 Top songs · {label}",
            description="\n".join(lines),
            color=discord.Color.blurple(),
        )
        await ctx.send(embed=embed)

    @commands.hybrid_command(
        name="topartists", aliases=["topartist"],
        description="Most-played artists (window: day/week/month/year/all)",
    )
    async def topartists(self, ctx: commands.Context, window: str | None = None):
        label, since = _parse_window(window, default="week")
        rows = await db.top_artists(ctx.guild.id, since=since, limit=10)
        if not rows:
            return await ctx.send(
                f"📭 No artist data {label} yet — most plays are tagged once "
                "they come from recognised music sources."
            )
        lines = [f"{_rank(i)} **{r['artist']}** · {r['plays']} play(s)"
                 for i, r in enumerate(rows, 1)]
        embed = discord.Embed(
            title=f"🎤 Top artists · {label}",
            description="\n".join(lines),
            color=discord.Color.purple(),
        )
        await ctx.send(embed=embed)

    @commands.hybrid_command(
        name="toplisteners", aliases=["topdj", "topdjs"],
        description="Who queued the most fully-played songs",
    )
    async def toplisteners(self, ctx: commands.Context, window: str | None = None):
        label, since = _parse_window(window, default="week")
        rows = await db.top_listeners(ctx.guild.id, since=since, limit=10)
        if not rows:
            return await ctx.send(f"📭 Nobody's finished a song {label} yet.")
        lines = []
        for i, r in enumerate(rows, 1):
            listened = fmt_duration_long(int(r["seconds"] or 0))
            lines.append(
                f"{_rank(i)} {self._name_for(ctx.guild, r['user_id'])} · "
                f"**{r['plays']}** play(s) · {listened}"
            )
        embed = discord.Embed(
            title=f"🎧 Top listeners · {label}",
            description="\n".join(lines),
            color=discord.Color.green(),
        )
        await ctx.send(embed=embed)

    # ── personal recap ────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="wrapped", aliases=["mywrapped"],
        description="Your music recap — top songs, artists, and listening time",
        help=(
            "Your personal listening recap from fully-played songs you queued.\n\n"
            "`wrapped` — your all-time recap\n"
            "`wrapped week` — just this week (day/week/month/year/all)\n"
            "`wrapped @user` — someone else's recap"
        ),
    )
    async def wrapped(self, ctx: commands.Context, *, args: str | None = None):
        # Manual parse so "@user", "week", or both work in any order without
        # tripping the member converter on a bare window keyword.
        target = ctx.message.mentions[0] if ctx.message.mentions else ctx.author
        window_token = None
        for tok in (args or "").split():
            if tok.startswith("<@"):
                continue                       # the mention, already handled
            window_token = tok
            break
        label, since = _parse_window(window_token, default="all")

        gid, uid = ctx.guild.id, target.id
        totals  = await db.user_play_totals(gid, uid, since=since)
        if not totals["plays"]:
            who = "You haven't" if target == ctx.author else f"{target.display_name} hasn't"
            return await ctx.send(
                f"📭 {who} finished any songs {label}. Play something all the "
                "way through and it'll show up here."
            )
        songs   = await db.user_top_tracks(gid, uid, since=since, limit=5)
        artists = await db.user_top_artists(gid, uid, since=since, limit=5)

        embed = discord.Embed(
            title=f"🎁 {target.display_name}'s wrapped · {label}",
            color=discord.Color.gold(),
        )
        embed.add_field(
            name="📊 Totals",
            value=(f"**{totals['plays']}** songs played · "
                   f"{fmt_duration_long(int(totals['seconds'] or 0))} listened"),
            inline=False,
        )
        if songs:
            embed.add_field(
                name="🎵 Top songs",
                value="\n".join(
                    f"{_rank(i)} {s['title']}"
                    + (f" — {s['artist']}" if s['artist'] else "")
                    + f" · {s['plays']}×"
                    for i, s in enumerate(songs, 1)
                ),
                inline=False,
            )
        if artists:
            embed.add_field(
                name="🎤 Top artists",
                value="\n".join(
                    f"{_rank(i)} **{a['artist']}** · {a['plays']}×"
                    for i, a in enumerate(artists, 1)
                ),
                inline=False,
            )
        embed.set_thumbnail(url=target.display_avatar.url)
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusicStatsCog(bot))
