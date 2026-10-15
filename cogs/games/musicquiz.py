"""
Music quiz — "name that tune" using the server's own listening history.

The bot plays a ~25-second snippet of a song and the first person to type the
title (or artist) in chat wins the round. The song pool is the server's play
history filtered to *real music* (see db.quiz_pool / audio._apply_music_meta),
so the random meme video someone played last week won't show up. A `-quiz
exclude` command lets you blocklist anything that slips through.

The quiz plays audio through the same voice connection the music player uses,
so it refuses to start while music is playing and bypasses the music engine
entirely (its own vc.play, no playback_gen) to avoid tangling with it.
"""
from __future__ import annotations

import asyncio
import random
import time

import discord
from discord.ext import commands

from core import audio, db
from core.logging_setup import get_logger
from core.state import get_state
from core.voice_intent import _normalize, _ratio

log = get_logger(__name__)

ROUNDS_DEFAULT = 5
ROUNDS_MAX     = 15
ROUND_SECONDS  = 30      # guessing window per round
SNIPPET_SECONDS = 25     # how long the clip plays
MIN_POOL       = 4       # not worth starting with fewer eligible songs

# Noise words stripped from titles before matching, so "Song (Official Video)"
# is guessable as just "song".
_TITLE_NOISE = {
    "official", "video", "audio", "lyrics", "lyric", "hd", "hq", "4k",
    "remaster", "remastered", "explicit", "mv", "m v", "visualizer",
    "music", "feat", "ft", "prod",
}


def _candidates(track: dict) -> list[str]:
    """Acceptable normalized answers for a track: the song title, the artist,
    and the pieces of an 'Artist - Title' video name."""
    out: list[str] = []

    def add(s: str | None):
        if not s:
            return
        n = _strip_noise(_normalize(s))
        if n and n not in out:
            out.append(n)

    add(track.get("song_title"))
    add(track.get("artist"))
    raw = track.get("title") or ""
    add(raw)
    if " - " in raw:
        left, _, right = raw.partition(" - ")
        add(left)
        add(right)
    return out


def _strip_noise(norm: str) -> str:
    """Drop trailing junk words from an already-normalized string."""
    words = [w for w in norm.split() if w not in _TITLE_NOISE]
    return " ".join(words).strip()


def is_correct_guess(guess: str, track: dict) -> bool:
    """Lenient match: exact, high edit-ratio, or a solid substring either way."""
    g = _strip_noise(_normalize(guess))
    if len(g) < 2:
        return False
    for cand in _candidates(track):
        if len(cand) < 2:
            continue
        if g == cand:
            return True
        if _ratio(g, cand) >= 0.8:
            return True
        if len(cand) >= 5 and cand in g:
            return True
        if len(g) >= 5 and g in cand:
            return True
    return False


class MusicQuizCog(commands.Cog, name="Music Quiz"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # guild_id -> {"cancel": bool} for any quiz currently running.
        self._games: dict[int, dict] = {}

    async def cog_check(self, ctx: commands.Context) -> bool:
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    @commands.hybrid_group(
        name="quiz", aliases=["musicquiz", "nametune"], invoke_without_command=True,
        description="Name-that-tune using this server's listening history",
        help=(
            "Guess the song from a short clip — first correct answer in chat wins.\n\n"
            "`quiz start [rounds]` — start a game (default 5 rounds)\n"
            "`quiz stop` — stop the current game\n"
            "`quiz leaderboard` — all-time scores\n"
            "`quiz exclude [url]` — blocklist a song (or the one playing now)\n"
            "`quiz include <url>` — un-blocklist a song"
        ),
    )
    async def quiz(self, ctx: commands.Context):
        await ctx.send_help(ctx.command)

    # ── start / stop ──────────────────────────────────────────────────────────

    @quiz.command(name="start", description="Start a music quiz (optionally N rounds)")
    async def quiz_start(self, ctx: commands.Context, rounds: int = ROUNDS_DEFAULT):
        if ctx.guild.id in self._games:
            return await ctx.send("❌ A quiz is already running here. `quiz stop` first.")

        vc, i_connected = await self._acquire_voice(ctx)
        if vc is None:
            return

        pool = await db.quiz_pool(ctx.guild.id)
        if len(pool) < MIN_POOL:
            if i_connected:
                await vc.disconnect()
            return await ctx.send(
                "❌ Not enough recognised songs in this server's history yet "
                f"(need {MIN_POOL}). Play more music — only real songs count, "
                "not clips/memes."
            )

        rounds = max(1, min(rounds, ROUNDS_MAX, len(pool)))
        random.shuffle(pool)
        picks = pool[:rounds]
        game = {"cancel": False}
        self._games[ctx.guild.id] = game
        scores: dict[int, int] = {}

        await ctx.send(
            f"🎶 **Music Quiz!** {rounds} round(s). I'll play a clip — type the "
            f"**song title** (or artist) in chat. {ROUND_SECONDS}s per round. Go!"
        )
        try:
            for i, track in enumerate(picks, 1):
                if game["cancel"]:
                    break
                await self._run_round(ctx, vc, track, i, rounds, scores, game)
                if game["cancel"]:
                    break
                await asyncio.sleep(2)
        except Exception:
            log.exception("music quiz round failed")
        finally:
            if vc.is_playing():
                vc.stop()
            self._games.pop(ctx.guild.id, None)
            await self._show_results(ctx, scores)
            if i_connected and vc.is_connected():
                await vc.disconnect()

    async def _run_round(self, ctx, vc, track, idx, total, scores, game) -> None:
        stub = {
            "title":          track["title"],
            "artist":         track.get("artist"),
            "song_title":     None,
            "webpage_url":    track["url"],
            "duration":       track.get("duration") or 0,
            "stream_url":     None,
            "_needs_resolve": True,
            "_search_query":  track["url"],
        }
        try:
            await audio.resolve_stream_url(stub)
        except Exception:
            log.warning("quiz: couldn't resolve %r — skipping", track.get("title"))
            await ctx.send(f"⏭️ *(round {idx} skipped — couldn't load a song)*")
            return

        dur = stub.get("duration") or 0
        latest = max(0, dur - SNIPPET_SECONDS)
        offset = random.uniform(min(20, latest), latest) if latest > 0 else 0.0
        source = audio.create_audio_source(stub["stream_url"], volume=0.5, seek=offset,
                                           http_headers=stub.get("_http_headers"))
        try:
            vc.play(source)
        except discord.ClientException:
            await ctx.send(f"⏭️ *(round {idx} skipped — playback error)*")
            return

        await ctx.send(f"🎧 **Round {idx}/{total}** — name that tune!")

        winner = await self._collect_guess(ctx, track, game)
        if vc.is_playing():
            vc.stop()

        artist = f" by **{track['artist']}**" if track.get("artist") else ""
        answer = f"**{track['title']}**{artist}"
        if winner is not None:
            scores[winner.id] = scores.get(winner.id, 0) + 1
            await db.quiz_add_points(ctx.guild.id, winner.id, 1)
            await ctx.send(f"✅ {winner.mention} got it! It was {answer}.")
        elif not game["cancel"]:
            await ctx.send(f"⏰ Time! It was {answer}.")

    async def _collect_guess(self, ctx, track, game):
        """Wait up to ROUND_SECONDS for the first correct chat guess."""
        deadline = time.monotonic() + ROUND_SECONDS

        def check(m: discord.Message) -> bool:
            return (m.channel.id == ctx.channel.id and not m.author.bot
                    and m.guild is not None and m.guild.id == ctx.guild.id)

        while not game["cancel"]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                msg = await self.bot.wait_for("message", timeout=remaining, check=check)
            except asyncio.TimeoutError:
                return None
            if is_correct_guess(msg.content, track):
                return msg.author
        return None

    async def _show_results(self, ctx, scores: dict[int, int]) -> None:
        if not scores:
            return await ctx.send("🏁 Quiz over — nobody scored. Next time!")
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        lines = []
        for i, (uid, pts) in enumerate(ranked):
            member = ctx.guild.get_member(uid)
            name = member.display_name if member else f"User {uid}"
            lines.append(f"{medals.get(i, f'#{i + 1}')} {name} — **{pts}**")
        embed = discord.Embed(
            title="🏁 Quiz results",
            description="\n".join(lines),
            color=discord.Color.blurple(),
        )
        await ctx.send(embed=embed)

    @quiz.command(name="stop", description="Stop the running music quiz")
    async def quiz_stop(self, ctx: commands.Context):
        game = self._games.get(ctx.guild.id)
        if not game:
            return await ctx.send("❌ No quiz is running.")
        game["cancel"] = True
        vc = ctx.voice_client
        if vc and vc.is_playing():
            vc.stop()
        await ctx.send("🛑 Stopping the quiz…")

    # ── leaderboard + excludes ────────────────────────────────────────────────

    @quiz.command(name="leaderboard", aliases=["scores", "lb"],
                  description="All-time music-quiz scores")
    async def quiz_leaderboard(self, ctx: commands.Context):
        rows = await db.quiz_leaderboard(ctx.guild.id)
        if not rows:
            return await ctx.send("📭 No quiz scores yet. Play `quiz start`!")
        medals = {0: "🥇", 1: "🥈", 2: "🥉"}
        lines = []
        for i, r in enumerate(rows):
            member = ctx.guild.get_member(r["user_id"])
            name = member.display_name if member else f"User {r['user_id']}"
            lines.append(f"{medals.get(i, f'#{i + 1}')} {name} — **{r['points']}**")
        embed = discord.Embed(
            title="🏆 Music quiz — all-time",
            description="\n".join(lines),
            color=discord.Color.gold(),
        )
        await ctx.send(embed=embed)

    @quiz.command(name="exclude", aliases=["block"],
                  description="Blocklist a song from the quiz (URL, or the one playing)")
    async def quiz_exclude(self, ctx: commands.Context, url: str | None = None):
        if not url:
            current = get_state(ctx.guild.id).current
            url = current.get("webpage_url") if current else None
        if not url or not str(url).startswith("http"):
            return await ctx.send(
                "❌ Give me a song URL, or run this while a song is playing."
            )
        await db.quiz_exclude(ctx.guild.id, url)
        await ctx.send("🚫 That song won't appear in the quiz anymore.")

    @quiz.command(name="include", aliases=["unblock"],
                  description="Remove a song from the quiz blocklist")
    async def quiz_include(self, ctx: commands.Context, url: str):
        removed = await db.quiz_include(ctx.guild.id, url)
        if removed:
            await ctx.send("✅ That song can appear in the quiz again.")
        else:
            await ctx.send("ℹ️ That URL wasn't on the blocklist.")

    # ── voice acquisition ─────────────────────────────────────────────────────

    async def _acquire_voice(self, ctx: commands.Context):
        """Get a voice client for the quiz, refusing if music is playing.
        Returns (vc, i_connected) — i_connected True if WE opened the connection
        (so we disconnect at the end). (None, False) on failure."""
        state = get_state(ctx.guild.id)
        vc = ctx.voice_client
        if (vc and (vc.is_playing() or vc.is_paused())) or state.current:
            await ctx.send(
                "❌ Music is playing right now — `stop` it first, then start the quiz."
            )
            return None, False
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("❌ Join a voice channel first.")
            return None, False
        if vc is None:
            # Connect through the music cog so the client is a VoiceRecvClient
            # with the clip recorder attached — a plain .connect() here used
            # to leave -clip / voice control dead for the whole session.
            from cogs.music import _connect_to_author as _music_connect
            new_vc = await _music_connect(ctx)
            return (new_vc, True) if new_vc is not None else (None, False)
        if vc.channel != ctx.author.voice.channel:
            await vc.move_to(ctx.author.voice.channel)
        return vc, False


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusicQuizCog(bot))
