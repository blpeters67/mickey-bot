"""
Karaoke — AI-separated instrumental, delivered as a file.

Why a file and not playback
---------------------------
The first version swapped the instrumental into the live player. That cannot
work on a CPU-only host, and the reason is arithmetic rather than a bug:
separation runs at roughly realtime or slower, so by the time a 4-minute track
is separated the 4-minute track has already finished playing. The swap was a
race that the song always won, and the user just heard the original with its
vocals — exactly as reported.

Handing back a file removes the race entirely. Nothing has to be ready before
a deadline: you ask, it works for a few minutes, you get an MP3. It's also
more useful — the file can be reused anywhere, not just in this voice channel.

The output is 192k rather than 320k so a typical song lands under Discord's
10 MB attachment limit and arrives as an actual attachment instead of a link
to a file host. Anything bigger falls back through the same catbox/litterbox
tiers the `download` command already uses.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import tempfile

import discord
from discord.ext import commands

from core import audio, db, separation
from core.config import settings
from core.formatters import fmt_duration, fmt_size
from core.logging_setup import get_logger
from core.separation import SeparationError
from core.state import get_state

log = get_logger(__name__)

#: Characters that upset one filesystem or another, plus anything Discord
#: renders oddly in an attachment name.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _safe_filename(title: str, suffix: str = " (Instrumental)",
                   ext: str = ".mp3", limit: int = 90) -> str:
    """A human-readable, filesystem-safe attachment name.

    The cache stores files under a hash, which is right for a cache and wrong
    for something a person downloads — nobody wants `a3f9c1….mp3` in their
    Downloads folder.
    """
    # Substitute a space rather than deleting: tabs and newlines are control
    # characters, so removing them outright welds words together
    # ("many\t\tspaces" -> "manyspaces"). A space also reads better for
    # separators that happen to be reserved — "AC/DC" becomes "AC DC", not
    # "ACDC". The collapse below tidies up the runs this creates.
    name = _UNSAFE.sub(" ", title or "track")
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > limit:
        name = name[:limit].rstrip()
    return f"{name or 'track'}{suffix}{ext}"


class KaraokeCog(commands.Cog, name="Karaoke"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        #: Guilds with a request in flight, so one person can't queue five
        #: separations by mashing the command.
        self._busy: set[int] = set()

    async def cog_check(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    # ── main command ─────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="karaoke", aliases=["kb", "instrumental"],
        description="Strip the vocals out of a song and send it back as a file",
        help=("Runs an AI separation model over a track and sends you the "
              "instrumental as an MP3 — a real model, not the `filter karaoke` "
              "channel trick, which just cancels whatever's centre-panned and "
              "takes the drums and bass with it.\n\n"
              "`karaoke <song or url>` — separate that\n"
              "`karaoke` — separate whatever's playing\n"
              "`karaoke cache` — how many instrumentals are stored\n"
              "`karaoke model` — which model is in use\n\n"
              "It does not touch playback. Separation takes a few minutes per "
              "track on a CPU host, so the bot keeps playing while it works "
              "and posts the file when it's done. Each track is only ever "
              "separated once — after that it's instant."),
    )
    async def karaoke(self, ctx: commands.Context, *, query: str | None = None):
        arg = (query or "").strip()
        word = arg.split()[0].lower() if arg else ""

        if word == "cache":
            return await self._cache(ctx, arg.split()[1:])
        if word == "model":
            return await self._model(ctx)

        if not separation.available():
            return await ctx.send(
                "❌ AI vocal separation isn't installed on my host.\n"
                "-# Owner: `./install.sh --karaoke` in the bot directory, then "
                "restart me. (Or `venv/bin/pip install \"audio-separator[cpu]\"` "
                "— it must go in the venv, not system Python.)"
            )

        if ctx.guild.id in self._busy:
            return await ctx.send(
                "⏳ I'm already separating something for this server. "
                "One at a time — it's CPU-heavy.")

        # Work out what to separate, without touching the player.
        if arg:
            status = await ctx.send(f"🔎 Looking up **{arg}**…")
            try:
                track = await audio.fetch_track(arg)
            except Exception as e:
                log.exception("karaoke lookup failed")
                return await status.edit(content=f"❌ Couldn't find that: {e}")
            if not track:
                return await status.edit(content=f"❌ No results for **{arg}**.")
        else:
            track = get_state(ctx.guild.id).current
            if not track:
                return await ctx.send(
                    "❌ Nothing playing — give me a song: `karaoke <name or url>`.")
            # Copy it: separation resolves stream URLs and we must not mutate
            # the dict the player is actively using.
            track = dict(track)
            status = await ctx.send("🎤 Setting up…")

        self._busy.add(ctx.guild.id)
        try:
            await self._run(ctx, status, track)
        finally:
            self._busy.discard(ctx.guild.id)

    async def _run(self, ctx: commands.Context, status: discord.Message,
                   track: dict) -> None:
        title = track.get("title") or "track"
        source_id = (track.get("webpage_url") or "").strip()
        if not source_id:
            return await status.edit(
                content="❌ That track has no stable URL, so I can't cache an "
                        "instrumental for it.")

        model_file, _desc = separation.model_filename()
        duration = float(track.get("duration") or 0.0)

        path = await db.get_stem(source_id, model_file)
        if path:
            await status.edit(content=f"⚡ **{title}** — already separated, sending…")
        else:
            try:
                await audio.resolve_stream_url(track)
            except Exception as e:
                log.exception("karaoke resolve failed")
                return await status.edit(content=f"❌ Couldn't fetch that track: {e}")

            eta = separation.estimate_seconds(duration or 240.0)
            queued = " (queued behind another job)" if separation.busy() else ""
            await status.edit(content=(
                f"🎛️ Separating **{title}**{queued}\n"
                f"-# About {fmt_duration(int(eta))} at ~"
                f"{separation.throughput():.2f}× realtime. Runs at low priority, "
                f"so playback isn't affected — I'll post the file when it's done."
            ))
            try:
                result = await separation.separate_to_instrumental(
                    source_id=source_id, stream_url=track["stream_url"],
                    duration=duration,
                )
            except SeparationError as e:
                return await status.edit(content=f"❌ {e}")
            except Exception:
                log.exception("karaoke separation failed")
                return await status.edit(
                    content="❌ Separation failed — details are in my logs.")

            path = result.path
            await db.put_stem(source_id, model_file, path,
                              duration=duration, title=title)
            await db.evict_stems(max(0, settings.stem_cache_mb) * 1_000_000)

        await self._deliver(ctx, status, path, title=title, duration=duration)

    # ── delivery ─────────────────────────────────────────────────────────────

    async def _deliver(self, ctx: commands.Context, status: discord.Message,
                       path: str, *, title: str, duration: float) -> None:
        """Send the instrumental, using the download cog's upload tiers.

        Reusing `_route_upload` rather than reimplementing it keeps the size
        thresholds, the catbox/litterbox fallbacks and the message formatting
        in exactly one place — this is the same problem `download` already
        solved, and two copies of it would drift within a release.
        """
        if not (path and os.path.exists(path)):
            return await status.edit(
                content="❌ The instrumental vanished before I could send it.")

        # Upload under a readable name rather than the cache's hash.
        tmpdir = tempfile.mkdtemp(prefix="musicbot-kb-", dir=settings.download_tmp)
        nice = os.path.join(tmpdir, _safe_filename(title))
        try:
            os.link(path, nice)          # cheap; same filesystem
        except OSError:
            shutil.copy2(path, nice)

        try:
            dl = self.bot.get_cog("Downloads")
            if dl is not None and hasattr(dl, "_route_upload"):
                from cogs.downloads import JobCanceller
                await dl._route_upload(
                    ctx, status, nice, title=f"{title} (Instrumental)",
                    duration=duration, has_video=False,
                    canceller=JobCanceller(), emoji="🎤",
                )
            else:
                await self._deliver_plain(ctx, status, nice, title, duration)
        except Exception:
            log.exception("karaoke upload failed")
            with contextlib.suppress(discord.HTTPException):
                await status.edit(content="❌ Couldn't upload the instrumental.")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def _deliver_plain(self, ctx: commands.Context,
                             status: discord.Message, path: str,
                             title: str, duration: float) -> None:
        """Fallback when the Downloads cog isn't loaded: Discord only."""
        size = os.path.getsize(path)
        limit = getattr(ctx.guild, "filesize_limit", 10 * 1024 * 1024)
        if size > limit:
            return await status.edit(
                content=f"❌ The instrumental is {fmt_size(size)}, over this "
                        f"server's {fmt_size(limit)} upload limit, and the "
                        f"Downloads cog isn't loaded to host it elsewhere.")
        await ctx.send(
            content=f"🎤 **{title} (Instrumental)** "
                    f"[{fmt_duration(int(duration))}] `{fmt_size(size)}` — "
                    f"requested by *{ctx.author.display_name}*",
            file=discord.File(path, filename=os.path.basename(path)),
        )
        with contextlib.suppress(discord.HTTPException):
            await status.delete()

    # ── subcommands ──────────────────────────────────────────────────────────

    async def _cache(self, ctx: commands.Context, rest: list[str]) -> None:
        if rest and rest[0].lower() in ("clear", "wipe", "purge"):
            if not ctx.author.guild_permissions.manage_guild:
                return await ctx.send("❌ That needs Manage Server.")
            n, freed = await db.clear_stem_cache()
            return await ctx.send(
                f"🗑️ Cleared {n} cached instrumental(s), freed {fmt_size(freed)}.")

        stats = await db.stem_cache_stats()
        cap = max(0, settings.stem_cache_mb) * 1_000_000
        pct = (stats["bytes"] / cap * 100) if cap else 0
        model_file, desc = separation.model_filename()
        await ctx.send(
            f"🎛️ **Karaoke cache**\n"
            f"{stats['count']} instrumental(s), {fmt_size(stats['bytes'])} of "
            f"{fmt_size(cap)} ({pct:.0f}%)\n"
            f"{fmt_duration(int(stats['seconds']))} of separated music\n"
            f"-# Model `{model_file}` — {desc}. Measured "
            f"{separation.throughput():.2f}× realtime here."
        )

    async def _model(self, ctx: commands.Context) -> None:
        current, desc = separation.model_filename()
        lines = "\n".join(f"`{k}` — {d}" for k, (_f, d) in separation.MODELS.items())
        await ctx.send(
            f"🎛️ Current model: `{current}`\n-# {desc}\n\n{lines}\n"
            f"-# Set with `SEPARATION_MODEL` in the environment and restart. "
            f"The cache is keyed by model, so switching re-separates everything."
        )


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(KaraokeCog(bot))
