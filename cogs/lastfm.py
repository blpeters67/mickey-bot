"""
Last.fm scrobbling — opt-in, per user.

Each person links their own Last.fm account once (`-lastfm link` → authorise in
the browser → `-lastfm confirm`). After that, whenever a song finishes playing,
it's scrobbled to every linked + enabled listener who's currently in the voice
channel — a shared "listening party" scrobble.

The whole feature is gated on the bot owner configuring LASTFM_API_KEY /
LASTFM_API_SECRET (settings.lastfm_enabled). With no key, the commands explain
that it's unconfigured and the music engine's scrobble hook is a no-op.

The Music cog calls `now_playing()` when a track starts and `scrobble_track()`
when it finishes (see MusicCog._scrobble_finished / _announce_now_playing).
"""
from __future__ import annotations

import asyncio

import discord
from discord.ext import commands

from core import db, lastfm
from core.config import settings
from core.logging_setup import get_logger

log = get_logger(__name__)


def _artist_title(track: dict) -> tuple[str | None, str]:
    """Best-effort (artist, title) for scrobbling. Prefers the extractor's
    artist/track tags; falls back to splitting a 'Artist - Title' video name."""
    artist = track.get("artist")
    title = track.get("song_title") or track.get("title") or ""
    if artist:
        return artist, title
    raw = track.get("title") or ""
    if " - " in raw:
        a, _, t = raw.partition(" - ")
        return a.strip(), t.strip()
    return None, title


class LastFMCog(commands.Cog, name="LastFM"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ── hooks called by the Music cog ─────────────────────────────────────────

    def _listeners_in_voice(self, guild: discord.Guild) -> list[int]:
        """Human members currently in the bot's voice channel."""
        vc = guild.voice_client
        if not vc or not vc.channel:
            return []
        return [m.id for m in vc.channel.members if not m.bot]

    async def now_playing(self, guild: discord.Guild, track: dict) -> None:
        if not settings.lastfm_enabled:
            return
        artist, title = _artist_title(track)
        if not artist or not title:
            return
        links = await db.lastfm_active_links(self._listeners_in_voice(guild))
        if not links:
            return
        duration = int(track.get("duration") or 0) or None
        await asyncio.gather(*(
            lastfm.update_now_playing(l["session_key"], artist=artist,
                                      track=title, duration=duration)
            for l in links
        ), return_exceptions=True)

    async def scrobble_track(self, guild: discord.Guild, track: dict) -> None:
        if not settings.lastfm_enabled:
            return
        artist, title = _artist_title(track)
        if not artist or not title:
            return
        links = await db.lastfm_active_links(self._listeners_in_voice(guild))
        if not links:
            return
        duration = int(track.get("duration") or 0) or None
        results = await asyncio.gather(*(
            lastfm.scrobble(l["session_key"], artist=artist, track=title,
                            duration=duration)
            for l in links
        ), return_exceptions=True)
        ok = sum(1 for r in results if r is True)
        if ok:
            log.info("Scrobbled %r by %r for %d listener(s)", title, artist, ok)

    # ── user commands ─────────────────────────────────────────────────────────

    @commands.hybrid_group(
        name="lastfm", aliases=["fm", "scrobble"], invoke_without_command=True,
        description="Link your Last.fm account to scrobble what plays",
        help=(
            "Scrobble songs to your own Last.fm while you listen.\n\n"
            "`lastfm link` — start linking your account\n"
            "`lastfm confirm` — finish after you authorise in the browser\n"
            "`lastfm on` / `lastfm off` — toggle your scrobbling\n"
            "`lastfm status` — show your link + state\n"
            "`lastfm unlink` — disconnect your account"
        ),
    )
    async def lastfm_group(self, ctx: commands.Context):
        await self._status(ctx)

    def _require_enabled(self, ctx) -> str | None:
        if not settings.lastfm_enabled:
            return ("❌ Last.fm scrobbling isn't set up on this bot — the owner "
                    "needs to configure `LASTFM_API_KEY` and `LASTFM_API_SECRET`.")
        return None

    @lastfm_group.command(name="link", description="Start linking your Last.fm account")
    async def lastfm_link(self, ctx: commands.Context):
        if (err := self._require_enabled(ctx)):
            return await ctx.send(err)
        token = await lastfm.get_token()
        if not token:
            return await ctx.send("❌ Couldn't reach Last.fm right now. Try again later.")
        await db.lastfm_set_pending(ctx.author.id, token)
        url = lastfm.auth_url(token)
        dm = (
            "🔗 **Link your Last.fm account**\n"
            f"1. Open this and click *Allow access*:\n{url}\n"
            f"2. Then come back and run `{ctx.prefix}lastfm confirm`.\n"
            "-# Only you can authorise this; the link is tied to your account."
        )
        try:
            await ctx.author.send(dm)
            where = "" if isinstance(ctx.channel, discord.DMChannel) else \
                " — check your DMs"
            await ctx.send(f"📨 Sent you the link{where}.")
        except discord.Forbidden:
            # DMs closed — fall back to posting in-channel.
            await ctx.send(dm)

    @lastfm_group.command(name="confirm",
                          description="Finish linking after authorising in the browser")
    async def lastfm_confirm(self, ctx: commands.Context):
        if (err := self._require_enabled(ctx)):
            return await ctx.send(err)
        row = await db.lastfm_get(ctx.author.id)
        if not row or not row["pending_tok"]:
            return await ctx.send(
                f"❌ Nothing to confirm. Start with `{ctx.prefix}lastfm link` first."
            )
        sess = await lastfm.get_session(row["pending_tok"])
        if not sess:
            return await ctx.send(
                "❌ That isn't authorised yet. Open the link from `lastfm link`, "
                "click *Allow access*, then run this again."
            )
        await db.lastfm_complete(ctx.author.id, session_key=sess["key"],
                                 username=sess["name"])
        await ctx.send(
            f"✅ Linked to **{sess['name']}**! Songs you listen to will scrobble. "
            f"Toggle anytime with `{ctx.prefix}lastfm off`."
        )

    @lastfm_group.command(name="on", description="Enable your scrobbling")
    async def lastfm_on(self, ctx: commands.Context):
        row = await db.lastfm_get(ctx.author.id)
        if not row or not row["session_key"]:
            return await ctx.send(f"❌ Link first with `{ctx.prefix}lastfm link`.")
        await db.lastfm_set_enabled(ctx.author.id, True)
        await ctx.send("✅ Scrobbling **on**.")

    @lastfm_group.command(name="off", description="Pause your scrobbling")
    async def lastfm_off(self, ctx: commands.Context):
        row = await db.lastfm_get(ctx.author.id)
        if not row or not row["session_key"]:
            return await ctx.send("ℹ️ You're not linked, so nothing's scrobbling.")
        await db.lastfm_set_enabled(ctx.author.id, False)
        await ctx.send("⏸️ Scrobbling **off**. Re-enable with `lastfm on`.")

    @lastfm_group.command(name="unlink", description="Disconnect your Last.fm account")
    async def lastfm_unlink(self, ctx: commands.Context):
        await db.lastfm_unlink(ctx.author.id)
        await ctx.send("✅ Unlinked your Last.fm account.")

    @lastfm_group.command(name="status", description="Show your Last.fm link status")
    async def lastfm_status(self, ctx: commands.Context):
        await self._status(ctx)

    async def _status(self, ctx: commands.Context) -> None:
        if not settings.lastfm_enabled:
            return await ctx.send(
                "ℹ️ Last.fm scrobbling isn't configured on this bot."
            )
        row = await db.lastfm_get(ctx.author.id)
        if not row or not row["session_key"]:
            return await ctx.send(
                f"🎵 You haven't linked Last.fm. Run `{ctx.prefix}lastfm link` to start."
            )
        state = "🟢 on" if row["enabled"] else "⏸️ off"
        await ctx.send(
            f"🎵 Linked to **{row['username']}** · scrobbling {state}."
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(LastFMCog(bot))
