"""
Liked songs — a personal, global "favorites" list.

`-like` while a song is playing saves it to the caller's own liked-songs list
(stored in core.db, keyed by user id only — your likes follow you across every
server). `-likes play` queues them all back up. This is the music equivalent of
a Spotify "Liked Songs" playlist.

Persistence lives entirely in the DB (liked_songs table), so likes survive
restarts and re-execs like everything else.
"""
from __future__ import annotations

import random

import discord
from discord.ext import commands

from core import audio, db
from core.formatters import fmt_duration
from core.logging_setup import get_logger
from core.state import get_state

log = get_logger(__name__)

_LIST_PAGE = 15   # liked songs shown per `-likes` embed


def _artist_of(track: dict) -> str | None:
    """Best-effort artist from a resolved track dict."""
    return track.get("artist") or track.get("uploader") or None


def _likeable_url(track: dict) -> str | None:
    """The stable URL we can re-resolve later, or None if the track can't be
    liked (e.g. a Discord upload whose CDN link expires)."""
    url = track.get("webpage_url")
    if not url or not str(url).startswith("http"):
        return None
    if audio.is_discord_cdn(url):
        return None
    return url


async def _connect_to_author(ctx: commands.Context):
    """Join/move to the caller's voice channel via the music cog's connector.

    Delegating matters: the music cog connects with `VoiceRecvClient` and
    attaches the rolling `-clip` recorder. A plain `.connect()` here used to
    leave the session with a receive-less voice client — `-clip` and voice
    control were silently dead whenever the bot entered voice via
    `likes play`. Imported lazily so `reload music` is honoured."""
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("❌ You must be in a voice channel.")
        return None
    from cogs.music import _connect_to_author as _music_connect
    return await _music_connect(ctx)


class FavoritesCog(commands.Cog, name="Favorites"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ── like / unlike ─────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="like", aliases=["fav"],
        description="Save the current song (or a search/URL) to your liked songs",
        help=(
            "Add a song to your personal liked-songs list — it follows you "
            "across every server.\n\n"
            "`like` — like whatever's playing now (run again to unlike it)\n"
            "`like <search or URL>` — like a specific song without playing it\n\n"
            "See them with `likes`, queue them with `likes play`."
        ),
    )
    async def like(self, ctx: commands.Context, *, query: str | None = None):
        if query:
            return await self._like_query(ctx, query)

        # No argument → like (or toggle off) the current track.
        track = get_state(ctx.guild.id).current if ctx.guild else None
        if not track:
            return await ctx.send(
                "❌ Nothing is playing. Use `like <song>` to like something "
                "specific, or start a song first."
            )
        url = _likeable_url(track)
        if url is None:
            return await ctx.send(
                "❌ This track can't be liked — it's a direct upload with no "
                "stable link to save."
            )
        title = track.get("title") or url
        if await db.is_liked(ctx.author.id, url):
            await db.remove_like(ctx.author.id, url)
            return await ctx.send(f"💔 Unliked **{title}**.")
        await db.add_like(
            ctx.author.id, url=url, title=title,
            artist=_artist_of(track), duration=int(track.get("duration") or 0),
        )
        await ctx.send(f"❤️ Liked **{title}** — find it in `{ctx.prefix}likes`.")

    async def _like_query(self, ctx: commands.Context, query: str) -> None:
        query = query.strip().strip("<>")
        async with ctx.typing():
            try:
                track = await audio.fetch_track(query)
            except Exception as e:
                log.exception("like: fetch_track failed")
                return await ctx.send(f"❌ {audio.friendly_error_msg(e)}")
        if not track:
            return await ctx.send("❌ Couldn't find anything for that.")
        url = _likeable_url(track)
        if url is None:
            return await ctx.send("❌ That result has no stable link to save.")
        title = track.get("title") or url
        added = await db.add_like(
            ctx.author.id, url=url, title=title,
            artist=_artist_of(track), duration=int(track.get("duration") or 0),
        )
        if not added:
            return await ctx.send(f"ℹ️ **{title}** is already in your liked songs.")
        await ctx.send(f"❤️ Liked **{title}**.")

    @commands.hybrid_command(
        name="unlike", description="Remove the current song from your liked songs",
        help="Removes whatever's playing now from your liked songs.",
    )
    async def unlike(self, ctx: commands.Context):
        track = get_state(ctx.guild.id).current if ctx.guild else None
        if not track:
            return await ctx.send("❌ Nothing is playing.")
        url = _likeable_url(track)
        if url is None or not await db.is_liked(ctx.author.id, url):
            return await ctx.send("❌ That song isn't in your liked songs.")
        await db.remove_like(ctx.author.id, url)
        await ctx.send(f"💔 Unliked **{track.get('title') or url}**.")

    # ── likes (list / play) ───────────────────────────────────────────────────

    @commands.hybrid_group(
        name="likes", aliases=["liked", "favorites"], invoke_without_command=True,
        description="Show liked songs (yours or another member's)",
        help=(
            "`likes` — list your liked songs\n"
            "`likes @user` — list someone else's\n"
            "`likes play [shuffle]` — queue all of your liked songs"
        ),
    )
    async def likes(self, ctx: commands.Context, member: discord.Member | None = None):
        target = member or ctx.author
        total = await db.count_likes(target.id)
        if total == 0:
            who = "You have" if target == ctx.author else f"{target.display_name} has"
            return await ctx.send(
                f"📭 {who} no liked songs yet. Play something and `{ctx.prefix}like` it!"
            )
        rows = await db.list_likes(target.id, limit=_LIST_PAGE)
        lines = []
        for i, r in enumerate(rows, 1):
            dur = f" `[{fmt_duration(r['duration'])}]`" if r["duration"] else ""
            artist = f" — {r['artist']}" if r["artist"] else ""
            lines.append(f"`{i}.` [{r['title']}]({r['url']}){artist}{dur}")
        embed = discord.Embed(
            title=f"❤️ {target.display_name}'s liked songs",
            description="\n".join(lines),
            color=discord.Color.red(),
        )
        shown = min(_LIST_PAGE, total)
        footer = f"Showing {shown} of {total}"
        if target == ctx.author:
            footer += f" · {ctx.prefix}likes play to queue them"
        embed.set_footer(text=footer)
        await ctx.send(embed=embed)

    @likes.command(
        name="play", aliases=["queue"],
        description="Queue all of your liked songs (add 'shuffle' to randomize)",
    )
    async def likes_play(self, ctx: commands.Context, *, flags: str | None = None):
        shuffle = bool(flags and "shuffle" in flags.lower())
        rows = await db.list_likes(ctx.author.id, limit=1000)
        if not rows:
            return await ctx.send(
                f"📭 You have no liked songs yet — `{ctx.prefix}like` one first."
            )
        vc = await _connect_to_author(ctx)
        if vc is None:
            return

        tracks = [{
            "title":          r["title"],
            "stream_url":     None,
            "webpage_url":    r["url"],
            "duration":       r["duration"],
            "thumbnail":      None,
            "requester":      ctx.author.display_name,
            "requester_id":   ctx.author.id,
            "_needs_resolve": True,
            "_search_query":  r["url"],
        } for r in rows]
        if shuffle:
            random.shuffle(tracks)

        state = get_state(ctx.guild.id)
        # Let the playback engine report skips/stalls back to this channel.
        state.text_channel_id = ctx.channel.id
        state.queue.extend(tracks)
        note = " 🎲 *shuffled*" if shuffle else ""
        await ctx.send(f"❤️ Queued **{len(tracks)}** liked song(s).{note}")

        if not vc.is_playing() and not vc.is_paused():
            music = self.bot.get_cog("Music")
            if music is not None:
                music._kickstart(ctx.guild, vc)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(FavoritesCog(bot))
