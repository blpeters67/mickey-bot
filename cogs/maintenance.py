"""
Maintenance — update dependencies and restart, from Discord.

yt-dlp breaks whenever YouTube changes something, which is often. The fix is
always the same upgrade, and needing SSH for it is friction at exactly the
wrong moment.

The trap this cog is built around: **a package upgrade doesn't affect the
running bot.** Python imported yt-dlp at startup and keeps those objects until
the process exits, so `update` alone leaves playback just as broken and looks
like it did nothing. Hence: every result reports before → after versions, and
a successful update that changed something offers a Restart button right
there. Restarting is the half people forget.

Owner-only, without exception. This runs pip against the bot's own
environment; Manage Server is nowhere near a high enough bar.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys

import discord
from discord.ext import commands

from core import updater
from core.logging_setup import get_logger

log = get_logger(__name__)

#: Discord message ceiling, minus room for our own framing.
_MAX_FIELD = 1500


def _fmt_changes(result: updater.UpdateResult) -> str:
    if not result.changed:
        return "Everything was already up to date."
    lines = []
    for dist, (old, new) in sorted(result.changed.items()):
        if old is None:
            lines.append(f"+ {dist}  (new) → {new}")
        elif new is None:
            lines.append(f"- {dist}  {old} → removed")
        else:
            lines.append(f"↑ {dist}  {old} → {new}")
    return "```diff\n" + "\n".join(lines)[:_MAX_FIELD] + "\n```"


class RestartView(discord.ui.View):
    """A Restart button, usable only by the person who ran the update."""

    def __init__(self, cog: "MaintenanceCog", author_id: int):
        super().__init__(timeout=300)
        self.cog = cog
        self.author_id = author_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message(
                "Only whoever ran the update can restart me.", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Restart now", style=discord.ButtonStyle.danger,
                       emoji="🔄")
    async def restart(self, interaction: discord.Interaction,
                      button: discord.ui.Button):
        button.disabled = True
        await interaction.response.edit_message(view=self)
        await interaction.followup.send("🔄 Restarting…")
        await self.cog.do_restart()


class MaintenanceCog(commands.Cog, name="Maintenance"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._running = False

    # ── update ───────────────────────────────────────────────────────────────

    @commands.command(name="update", aliases=["upgrade"], hidden=True)
    @commands.is_owner()
    async def update(self, ctx: commands.Context, *, what: str | None = None):
        """Upgrade the bot's Python dependencies.

        `update`                 — everything in requirements.txt (incl. yt-dlp)
        `update ytdlp`           — just yt-dlp, the usual culprit; quickest
        `update ytdlp nightly`   — yt-dlp nightly: YouTube fixes land here
                                   days to weeks before a stable release
        `update ytdlp stable`    — come back off nightly
        `update all`             — the above plus the installed AI extras
        `update <package>`       — one specific package
        """
        if self._running:
            return await ctx.send("⏳ An update is already running.")

        arg = (what or "").strip().lower()
        parts = arg.split()
        head = parts[0] if parts else ""
        channel = "stable"
        if len(parts) > 1 and parts[1] in ("nightly", "pre", "master", "dev"):
            channel = "nightly"

        self._running = True
        try:
            async with ctx.typing():
                if head in ("ytdlp", "yt-dlp", "ytdl", "yt_dlp"):
                    label = "nightly" if channel == "nightly" else "stable"
                    status = await ctx.send(f"⏳ Upgrading **yt-dlp** ({label})…")
                    result = await updater.upgrade_packages(
                        [updater.YTDLP_SPEC], channel=channel)   # + its YouTube solver
                elif head in ("", "core", "requirements"):
                    status = await ctx.send(
                        "⏳ Upgrading core dependencies (yt-dlp included)…")
                    result = await updater.upgrade_requirements(extras=False)
                elif head == "all":
                    status = await ctx.send(
                        "⏳ Upgrading everything, including the AI extras. "
                        "This can take several minutes…")
                    result = await updater.upgrade_requirements(extras=True)
                else:
                    # A specific package. Keep it to one token so this can't be
                    # turned into arbitrary pip flags.
                    pkg = head
                    if not all(c.isalnum() or c in "-_.[]" for c in pkg):
                        return await ctx.send("❌ That doesn't look like a package name.")
                    status = await ctx.send(f"⏳ Upgrading **{pkg}**…")
                    result = await updater.upgrade_packages([pkg])
        finally:
            self._running = False

        if not result.ok:
            log.error("update failed:\n%s", result.output[-4000:])
            return await status.edit(
                content=f"❌ Update failed.\n```\n{result.error[:600]}\n```\n"
                        f"-# Full output is in my logs.")

        body = _fmt_changes(result)
        if result.anything_changed:
            await status.edit(
                content=(
                    f"✅ **Update complete.**\n{body}\n"
                    f"⚠️ **I'm still running the old code.** Python loaded these "
                    f"at startup and keeps them until I restart — so yt-dlp "
                    f"won't actually be fixed until then."
                ),
                view=RestartView(self, ctx.author.id),
            )
        else:
            await status.edit(content=f"✅ **Already up to date.**\n{body}")

    # ── yt-dlp diagnostics ───────────────────────────────────────────────────

    @commands.command(name="ytdiag", aliases=["diagnose"], hidden=True)
    @commands.is_owner()
    async def ytdiag(self, ctx: commands.Context, *, url: str = ""):
        """Work out why a specific URL fails.

        Uses the bot's own yt-dlp options and cookies, then actually fetches a
        kilobyte of the media — because extraction succeeding proves nothing
        when the 403 happens at fetch time.
        """
        url = url.strip().strip("<>")
        if not url:
            return await ctx.send(
                f"Usage: `{ctx.prefix}ytdiag <url>` — tells you why that link fails.")

        from core import ytdiag as ytd

        status = await ctx.send("🔬 Diagnosing… (probing formats, ~15s)")
        try:
            rep = await asyncio.wait_for(ytd.diagnose(url), timeout=180)
        except asyncio.TimeoutError:
            return await status.edit(content="❌ Diagnosis timed out.")
        except Exception as e:
            log.exception("ytdiag failed")
            return await status.edit(content=f"❌ Diagnosis blew up: `{e}`")

        text = rep.text()
        if len(text) <= 1850:
            await status.edit(content=f"```\n{text}\n```")
        else:
            await status.edit(content="🔬 Diagnosis attached.")
            await ctx.send(file=discord.File(
                io.BytesIO(text.encode()), filename="ytdiag.txt"))

    # ── versions ─────────────────────────────────────────────────────────────

    @commands.command(name="versions", aliases=["deps"], hidden=True)
    @commands.is_owner()
    async def versions(self, ctx: commands.Context):
        """Show what's installed — the quick way to check whether an upgrade
        landed and whether the restart actually happened."""
        core = updater.versions_of(updater.CORE_WATCH)
        extras = updater.versions_of(updater.installed_optional_dists())
        lines = [f"{d:<18} {v or '—'}" for d, v in core.items()]
        if extras:
            lines.append("")
            lines += [f"{d:<18} {v or '—'}" for d, v in extras.items()]
        await ctx.send(
            f"📦 **Installed**\n```\n" + "\n".join(lines)[:_MAX_FIELD] +
            f"\n```\n-# Python {sys.version.split()[0]} at `{sys.executable}`")

    # ── restart ──────────────────────────────────────────────────────────────

    @commands.command(name="restart", aliases=["reboot"], hidden=True)
    @commands.is_owner()
    async def restart(self, ctx: commands.Context):
        """Restart the bot in place, picking up any upgraded packages."""
        await ctx.send("🔄 Restarting…")
        await self.do_restart()

    async def do_restart(self) -> None:
        """Replace this process with a fresh one.

        `os.execv` keeps the same PID, which matters under systemd and any
        other supervisor — it sees a continuously running service rather than
        an exit it has to react to. Voice is disconnected first so the bot
        doesn't leave a ghost connection behind in a channel.
        """
        for vc in list(self.bot.voice_clients):
            try:
                await vc.disconnect(force=True)
            except Exception:
                pass
        try:
            await self.bot.close()
        except Exception:
            log.debug("close() raised during restart", exc_info=True)

        log.info("Restarting: execv %s %s", sys.executable, sys.argv)
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            # If exec fails we've already closed the gateway, so the only sane
            # move is to exit and let the supervisor start us again.
            log.exception("execv failed — exiting for the supervisor to restart")
            os._exit(1)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MaintenanceCog(bot))
