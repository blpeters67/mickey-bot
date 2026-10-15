"""
Admin / housekeeping cog:
  * Per-guild prefix (slash + classic) — actually persists per guild.
  * Auto-generated `help` from the bot's commands.
  * Global on_command_error so users get clean messages, not silence.
  * on_voice_state_update — disconnect when alone in a channel; resync state
    if the bot itself gets disconnected by an admin; autojoin (opt-in per
    server) when someone enters voice in a server where nobody else is.
  * Idle-disconnect timer started in setup_hook.
"""
from __future__ import annotations

import asyncio
import io
import time
from collections import defaultdict

import discord
from discord.ext import commands, tasks

from core import console, db
from core.config import settings
from core.formatters import fmt_duration_long, parse_duration
from core.logging_setup import get_logger
from core.state import reset_state

log = get_logger(__name__)


class AdminCog(commands.Cog, name="Admin"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self._last_active: dict[int, float] = defaultdict(time.time)
        self._autojoin_locks: dict[int, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()
        # Always run the loop. The global `idle_seconds=0` no longer means
        # "disable the loop entirely" — per-guild settings can still enable
        # idle disconnect even when the global default is off. The loop
        # body short-circuits per-guild instead.
        self.idle_check.start()

    def cog_unload(self) -> None:
        if self.idle_check.is_running():
            self.idle_check.cancel()

    # ── owner console bridge ──────────────────────────────────────────────────

    @commands.command(name="console", aliases=["con"], hidden=True)
    @commands.is_owner()
    async def console_cmd(self, ctx: commands.Context, *, line: str = ""):
        """Run the bot's admin-console commands from Discord (owner only).

        e.g. `-console stats`, `-console guilds`, `-console reload music`,
        `-console threads`, `-console restart`.
        """
        line = line.strip()
        names = sorted(console.COMMANDS) + ["threads"]
        if not line:
            return await ctx.send(
                "🖥️ **Console** *(owner only)* — run the bot's terminal commands "
                f"here.\nUsage: `{ctx.prefix}console <command> [args]`\n"
                "Available: " + ", ".join(f"`{n}`" for n in names)
            )

        parts = line.split()
        name, args = parts[0].lower(), parts[1:]

        # restart/quit tear the process down — fire them, don't await output.
        if name in ("restart", "quit"):
            verb = "Restarting" if name == "restart" else "Shutting down"
            sent = await ctx.send(
                f"🔄 {verb}… *(watch the bot reconnect)*"
            )
            if name == "restart":
                # The restart re-execs the process, wiping all memory, so we
                # stash where to announce "back online" in the DB. on_ready in
                # bot.py reads it after reconnecting and edits this message.
                import json
                await db.set_app_setting("restart_notify", json.dumps({
                    "channel_id": ctx.channel.id,
                    "message_id": sent.id,
                    "at": time.time(),
                }))
            asyncio.create_task(console.COMMANDS[name][0](self.bot, args))
            return

        try:
            async with ctx.typing():
                out = await console.run_console_command(self.bot, name, args)
        except Exception as e:
            log.exception("console bridge command %r failed", name)
            return await ctx.send(f"❌ `{name}` failed: {e}")
        await self._send_console_output(ctx, out)

    async def _send_console_output(self, ctx: commands.Context, text: str) -> None:
        text = (text or "").rstrip() or "(no output)"
        if len(text) <= 1900:
            await ctx.send(f"```\n{text}\n```")
        else:
            # Too long for a message — attach the full output as a file.
            await ctx.send(file=discord.File(
                io.BytesIO(text.encode()), filename="console.txt"))

    # NB: no local @console_cmd.error handler. The old one re-raised anything
    # that wasn't NotOwner, which logged an "Ignoring exception in command
    # error handler" traceback on every non-owner attempt. The global
    # on_command_error below handles NotOwner (and everything else) once.

    # ── prefix ───────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="prefix", description="Change this server's prefix",
                              help=(
                                  "Sets a new command prefix for this server. Up to 5 characters.\n"
                                  "Each server has its own prefix; this doesn't affect other servers.\n\n"
                                  "**Examples**\n"
                                  "`prefix !` — use `!` instead of `-`\n"
                                  "`prefix !!` — use `!!`\n\n"
                                  "Requires *Manage Server* permission."
                              ))
    @commands.has_guild_permissions(manage_guild=True)
    async def prefix(self, ctx: commands.Context, new_prefix: str):
        if len(new_prefix) > 5:
            return await ctx.send("❌ Prefix must be 5 characters or fewer.")
        await db.set_prefix(ctx.guild.id, new_prefix)
        await ctx.send(f"✅ Prefix for this server is now `{new_prefix}`.")

    # ── autoleave ────────────────────────────────────────────────────────────

    @commands.hybrid_group(
        name="autoleave", invoke_without_command=True,
        description="Configure when the bot leaves voice automatically",
        help=(
            "Per-server config for auto-leave behaviour. Two knobs:\n\n"
            "**Idle disconnect** — how long the bot waits when nothing is "
            "playing before leaving the channel.\n"
            "**Leave when alone** — whether the bot disconnects the moment "
            "the last human user leaves its voice channel.\n\n"
            "**Subcommands**\n"
            "`autoleave show` — current settings for this server\n"
            "`autoleave idle <time>` — set idle timeout. Accepts `10m`, `1h`, "
            "`600s`, plain seconds, or `off`/`0` to disable\n"
            "`autoleave alone <on|off>` — toggle leave-when-alone\n"
            "`autoleave reset` — clear all overrides; use bot defaults again\n\n"
            "Requires *Manage Server*. Settings persist across restarts."
        ),
    )
    async def autoleave(self, ctx: commands.Context):
        # No subcommand → show.
        await self._autoleave_show(ctx)

    async def _autoleave_show(self, ctx: commands.Context) -> None:
        gs = await db.get_guild_settings(ctx.guild.id)

        # Resolve effective values for display.
        idle_override = gs["idle_seconds"]
        idle_effective = idle_override if idle_override is not None else settings.idle_seconds
        if idle_override is None:
            idle_line = f"`{fmt_duration_long(idle_effective)}` *(bot default)*"
        elif idle_override == 0:
            idle_line = "`off` *(this server has disabled idle disconnect)*"
        else:
            idle_line = f"`{fmt_duration_long(idle_override)}` *(server override)*"

        alone_override = gs["leave_when_alone"]
        # Default is on — that's the only behaviour the bot originally had.
        if alone_override is None:
            alone_line = "`on` *(default)*"
        else:
            alone_line = (
                f"`{'on' if alone_override else 'off'}` "
                "*(server override)*"
            )

        embed = discord.Embed(
            title="🔧 Auto-leave settings",
            color=discord.Color.blurple(),
            description=(
                f"**Idle disconnect:** {idle_line}\n"
                f"**Leave when alone:** {alone_line}\n\n"
                f"Change with `{ctx.prefix}autoleave idle <time>` or "
                f"`{ctx.prefix}autoleave alone <on|off>`."
            ),
        )
        await ctx.send(embed=embed)

    @autoleave.command(
        name="show",
        description="Show this server's auto-leave settings",
        help="Display the current idle timeout and leave-when-alone state.",
    )
    async def autoleave_show(self, ctx: commands.Context):
        await self._autoleave_show(ctx)

    @autoleave.command(
        name="idle",
        description="Set idle-disconnect time (e.g. `10m`, `1h`, `off`)",
        help=(
            "Sets how long the bot waits with nothing playing before "
            "leaving voice.\n\n"
            "**Accepts**\n"
            "`10m`, `1h`, `30s`, plain seconds (`600`), or "
            "compound (`1h30m`)\n"
            "`0`, `off`, `never`, `disabled` — never auto-leave on idle\n\n"
            "**Examples**\n"
            "`autoleave idle 10m`\n"
            "`autoleave idle 1h30m`\n"
            "`autoleave idle off`"
        ),
    )
    @commands.has_guild_permissions(manage_guild=True)
    async def autoleave_idle(self, ctx: commands.Context, *, value: str):
        seconds = parse_duration(value)
        if seconds is None:
            return await ctx.send(
                "❌ Couldn't parse that time. Try `10m`, `1h`, `600s`, or `off`."
            )
        # Sanity floor — sub-5s values are almost always a mistake and would
        # turn auto-leave into a race with the connect handshake.
        if 0 < seconds < 5:
            return await ctx.send("❌ Minimum is 5 seconds (or `off` to disable).")
        await db.set_idle_seconds(ctx.guild.id, seconds)
        if seconds == 0:
            return await ctx.send("✅ Idle disconnect is now **off** for this server.")
        await ctx.send(
            f"✅ Idle disconnect set to **{fmt_duration_long(seconds)}** for this server."
        )

    @autoleave.command(
        name="alone",
        description="Toggle leave-when-alone (`on` or `off`)",
        help=(
            "Controls whether the bot leaves voice the moment the last "
            "human user disconnects from its channel.\n\n"
            "**Examples**\n"
            "`autoleave alone on`  — leave immediately when alone (default)\n"
            "`autoleave alone off` — stay even when no humans are in the channel"
        ),
    )
    @commands.has_guild_permissions(manage_guild=True)
    async def autoleave_alone(self, ctx: commands.Context, *, value: str):
        v = value.strip().lower()
        if   v in ("on", "yes", "true", "1", "enable", "enabled"):   enabled = True
        elif v in ("off", "no", "false", "0", "disable", "disabled"): enabled = False
        else:
            return await ctx.send("❌ Use `on` or `off`.")
        await db.set_leave_when_alone(ctx.guild.id, enabled)
        state_word = "on" if enabled else "off"
        await ctx.send(f"✅ Leave-when-alone is now **{state_word}** for this server.")

    @autoleave.command(
        name="reset",
        description="Clear server overrides and use bot defaults",
        help="Removes both overrides for this server. The bot will use its "
             "global defaults again (idle = whatever the operator configured "
             "in .env; leave-when-alone = on).",
    )
    @commands.has_guild_permissions(manage_guild=True)
    async def autoleave_reset(self, ctx: commands.Context):
        await db.set_idle_seconds(ctx.guild.id, None)
        await db.set_leave_when_alone(ctx.guild.id, None)
        await ctx.send("✅ Auto-leave settings reset to defaults for this server.")

    # ── autojoin ─────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="autojoin",
        description="Join people automatically when they enter voice in an empty server (on/off)",
        help=(
            "When someone joins a voice channel and **nobody else is in voice "
            "anywhere in the server**, I join them straight away — no need to "
            "summon me. I don't start playing anything; just `play` (or talk to "
            "me with voice control).\n\n"
            "Off by default, per server; persists across restarts. Changing it "
            "needs *Manage Server*.\n\n"
            "Things to know:\n"
            "• Like being summoned, joining starts the rolling voice buffer used by "
            "`clip` (and voice control, for people who opted in).\n"
            "• With `autoleave alone` on (the default) I leave again when they "
            "do; `autoleave idle` still applies if nothing gets played.\n"
            "• I never autojoin the AFK channel or a stage, a channel I can't "
            "speak in, or a full one.\n\n"
            "**Examples**\n`autojoin on` · `autojoin off` · `autojoin` *(show)*"
        ),
    )
    async def autojoin(self, ctx: commands.Context, value: str | None = None):
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        if value is None:
            gs = await db.get_guild_settings(ctx.guild.id)
            on = gs.get("autojoin") == 1
            return await ctx.send(
                f"🚪 Autojoin is **{'on' if on else 'off'}**. "
                f"`{ctx.prefix}autojoin on|off` to change *(Manage Server)*."
            )
        if not ctx.author.guild_permissions.manage_guild:
            raise commands.MissingPermissions(["manage_guild"])
        v = value.strip().lower()
        if   v in ("on", "yes", "true", "1", "enable", "enabled"):    enabled = True
        elif v in ("off", "no", "false", "0", "disable", "disabled"): enabled = False
        else:
            return await ctx.send("❌ Use `on` or `off`.")
        await db.set_autojoin(ctx.guild.id, enabled)
        if enabled:
            await ctx.send(
                "🚪 Autojoin **on** — when someone joins a voice channel while nobody "
                "else in the server is in voice, I'll join them. (That starts the "
                "rolling `clip` buffer, same as summoning me.)")
        else:
            await ctx.send("🚪 Autojoin **off**.")

    def _autojoin_target(self, member: discord.Member, before, after):
        """The channel to autojoin for this voice-state change, or None."""
        if member.bot or after.channel is None:
            return None
        guild = member.guild
        afk = guild.afk_channel
        # A fresh join (or coming back from AFK) — not a move between rooms,
        # not a mute/deafen toggle.
        if before.channel is not None and before.channel != afk:
            return None
        channel = after.channel
        if not isinstance(channel, discord.VoiceChannel) or channel == afk:
            return None
        if guild.voice_client is not None:
            return None
        # "Nobody else in voice" is judged at the moment they joined: if two
        # people arrive together, the first one still gets the bot.
        if not self._voice_is_empty_besides(guild, member):
            return None
        return channel

    @staticmethod
    def _voice_is_empty_besides(guild: discord.Guild, member: discord.Member) -> bool:
        """Nobody but `member` is in voice (people idling in AFK don't count;
        bots don't count)."""
        afk = guild.afk_channel
        for ch in [*guild.voice_channels, *guild.stage_channels]:
            if ch == afk:
                continue
            if any(not m.bot and m.id != member.id for m in ch.members):
                return False
        return True

    async def _maybe_autojoin(self, member: discord.Member, channel) -> None:
        guild = member.guild
        lock = self._autojoin_locks.setdefault(guild.id, asyncio.Lock())
        if lock.locked():
            return                       # someone else's join is being handled
        async with lock:
            try:
                gs = await db.get_guild_settings(guild.id)
            except Exception:
                log.exception("autojoin: couldn't read settings for guild %s", guild.id)
                return
            if gs.get("autojoin") != 1:
                return
            # Re-check after the await: things move fast in voice.
            if guild.voice_client is not None:
                return
            if member.voice is None or member.voice.channel != channel:
                return
            perms = channel.permissions_for(guild.me)
            if not (perms.view_channel and perms.connect and perms.speak):
                return
            if (channel.user_limit and len(channel.members) >= channel.user_limit
                    and not perms.move_members):
                return
            music = self.bot.get_cog("Music")
            if music is None:
                return
            if await music.autojoin(member, channel):
                self._last_active[guild.id] = time.time()   # idle timer starts now

    # ── invite ───────────────────────────────────────────────────────────────

    # The bitmask is computed from the permissions the bot actually needs to
    # function. If you add features that need new permissions, regenerate this
    # at https://discordapi.com/permissions.html.
    #
    # Currently includes: View Channels, Send Messages, Embed Links,
    # Attach Files, Read Message History, Add Reactions, Use External Emojis,
    # Connect, Speak, Use Voice Activity, Use Slash Commands,
    # Manage Messages (share deletes the poster's message),
    # Manage Webhooks (share reposts with the sharer's name/avatar).
    _INVITE_PERMS = 2184563776 | (1 << 13) | (1 << 29)   # = 2721442880

    @commands.hybrid_command(name="invite", description="Get a link to add the bot to another server",
                              help=(
                                  "Returns an OAuth invite link with the right permissions pre-checked. "
                                  "To use it, the person clicking the link needs **Manage Server** on "
                                  "the destination server."
                              ))
    async def invite(self, ctx: commands.Context):
        url = discord.utils.oauth_url(
            client_id=self.bot.user.id,
            permissions=discord.Permissions(self._INVITE_PERMS),
            scopes=("bot", "applications.commands"),
        )
        embed = discord.Embed(
            title="🔗 Invite me to a server",
            description=(
                f"[Click here to add me to another server]({url})\n\n"
                "You need **Manage Server** permission on the destination "
                "server to add bots."
            ),
            color=discord.Color.blurple(),
        )
        embed.set_footer(text="The pre-checked permissions are the minimum needed for music + games.")
        await ctx.send(embed=embed)

    # ── help ─────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name="help", description="Show help — try `help <command>` or `help <section>`")
    async def help_cmd(self, ctx: commands.Context, *, query: str | None = None):
        prefix = await db.get_prefix(ctx.guild.id) if ctx.guild else settings.default_prefix
        cogs   = self.bot.cogs
        query  = (query or "").strip()

        if query:
            q = query.lower()

            # 1. Exact command or subcommand, aliases included
            #    (`help play`, `help pl`, `help playlist add`).
            cmd = self.bot.get_command(q)
            if cmd is not None and not cmd.hidden:
                return await ctx.send(embed=self._command_embed(cmd, prefix))

            # 2. Section name — exact first, then unambiguous prefix
            #    (`help music`, `help mus`).
            cog_match = next(
                (c for c in cogs.values() if c.qualified_name.lower() == q), None,
            )
            if cog_match is None:
                starts = [c for c in cogs.values()
                          if c.qualified_name.lower().startswith(q)]
                if len(starts) == 1:
                    cog_match = starts[0]
            if cog_match:
                for embed in self._cog_embeds(cog_match, prefix):
                    await ctx.send(embed=embed)
                return

            # 3. Substring match across every command and alias
            #    (`help sponsor` → sponsorblock). One hit shows it directly;
            #    several show a disambiguation list.
            all_cmds = [c for c in self.bot.walk_commands() if not c.hidden]
            matches  = [c for c in all_cmds
                        if q in c.qualified_name.lower()
                        or any(q in a.lower() for a in c.aliases)]
            if len(matches) == 1:
                return await ctx.send(embed=self._command_embed(matches[0], prefix))
            if matches:
                lines = "\n".join(
                    f"`{prefix}{c.qualified_name}` — {(c.description or c.help or '—').splitlines()[0][:80]}"
                    for c in matches[:15]
                )
                embed = discord.Embed(
                    title=f'🔎 Commands matching "{query}"',
                    description=lines, color=discord.Color.blurple(),
                )
                embed.set_footer(text=f"{prefix}help <command> for details")
                return await ctx.send(embed=embed)

            # 4. Nothing — suggest the closest names.
            import difflib
            pool = ({c.qualified_name.lower() for c in all_cmds}
                    | {a.lower() for c in all_cmds for a in c.aliases}
                    | {c.qualified_name.lower() for c in cogs.values()})
            close = difflib.get_close_matches(q, pool, n=3, cutoff=0.5)
            hint = (" Did you mean " + ", ".join(f"`{m}`" for m in close) + "?"
                    if close else "")
            return await ctx.send(
                f"❌ No command or section called `{query}`.{hint} "
                f"Try `{prefix}help` for the list."
            )

        embed = discord.Embed(
            title="🤖 Bot Help",
            description=(
                f"Prefix: `{prefix}`\n"
                f"`{prefix}help <command>` for command details (e.g. `{prefix}help resize`)\n"
                f"`{prefix}help <section>` for a whole section (e.g. `{prefix}help Music`)"
            ),
            color=discord.Color.blurple(),
        )
        for cog in cogs.values():
            cmds = [c for c in cog.get_commands() if not c.hidden]
            if not cmds:
                continue
            preview = "  ".join(f"`{prefix}{c.name}`" for c in cmds[:8])
            if len(cmds) > 8:
                preview += f"  …+{len(cmds) - 8}"
            embed.add_field(name=cog.qualified_name, value=preview, inline=False)
        await ctx.send(embed=embed)

    def _cog_embeds(self, cog: commands.Cog, prefix: str) -> list[discord.Embed]:
        """One-line-per-command section listing, including group subcommands.

        Built in the embed *description* (4096 chars), not fields — embeds cap
        at 25 fields and the Music cog alone has more commands than that.
        Splits into multiple embeds if a section ever outgrows one.
        """
        lines: list[str] = []
        for cmd in cog.get_commands():
            if cmd.hidden:
                continue
            entries = [cmd]
            if isinstance(cmd, commands.Group):
                entries += [s for s in cmd.commands if not s.hidden]
            for c in entries:
                sig  = c.signature
                desc = (c.description or (c.help or "").strip() or "—").splitlines()[0]
                if len(desc) > 80:
                    desc = desc[:77] + "…"
                lines.append(f"`{prefix}{c.qualified_name}{(' ' + sig) if sig else ''}` — {desc}")

        embeds: list[discord.Embed] = []
        chunk: list[str] = []
        size = 0
        for line in lines:
            if size + len(line) + 1 > 3900:
                embeds.append(discord.Embed(description="\n".join(chunk),
                                            color=discord.Color.blurple()))
                chunk, size = [], 0
            chunk.append(line)
            size += len(line) + 1
        embeds.append(discord.Embed(description="\n".join(chunk) or "—",
                                    color=discord.Color.blurple()))
        embeds[0].title = f"{cog.qualified_name} commands"
        embeds[-1].set_footer(text=f"{prefix}help <command> for details and examples")
        return embeds

    def _command_embed(self, cmd: commands.Command, prefix: str) -> discord.Embed:
        """Detailed help for one command — signature, description, aliases,
        long-form `help=` text (often containing examples)."""
        sig   = cmd.signature
        title = f"{prefix}{cmd.qualified_name}" + (f" {sig}" if sig else "")

        # Long-form help (the `help=` kwarg) goes in the description — embed
        # fields cap at 1024 chars and several commands' help text is longer,
        # which made Discord 400 the whole message.
        long_help = (cmd.help or "").strip()
        short     = (cmd.description or "").strip()
        body = short or long_help or "*(no description)*"
        if long_help and long_help != short and short:
            body = f"{short}\n\n{long_help}"
        if len(body) > 4000:
            body = body[:3997] + "…"
        embed = discord.Embed(
            title=title, description=body, color=discord.Color.blurple(),
        )

        if cmd.aliases:
            embed.add_field(
                name="Aliases",
                value=", ".join(f"`{prefix}{a}`" for a in cmd.aliases),
                inline=False,
            )

        # If it's a group, list its subcommands too.
        if isinstance(cmd, commands.Group):
            subs = [s for s in cmd.commands if not s.hidden]
            if subs:
                lines = []
                for s in subs:
                    s_sig = s.signature
                    line  = f"`{prefix}{cmd.name} {s.name}{(' ' + s_sig) if s_sig else ''}`"
                    desc  = s.description or s.help or ""
                    lines.append(f"{line}  —  {desc}" if desc else line)
                embed.add_field(name="Subcommands", value="\n".join(lines), inline=False)

        return embed

    # ── voice state — idle / orphan handling ─────────────────────────────────

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member:  discord.Member,
        before:  discord.VoiceState,
        after:   discord.VoiceState,
    ) -> None:
        guild = member.guild
        vc    = guild.voice_client

        # Bot itself appears to have left voice. This fires both on permanent
        # disconnects (admin kicked us, /stop) AND on transient WebSocket
        # blips (Discord drops the voice gateway and discord.py auto-reconnects).
        # We can't tell which it is from the event alone — but if it was a
        # blip, the bot will be back in voice within a few seconds. So we
        # wait and re-check before nuking the queue.
        if member.id == self.bot.user.id and before.channel and not after.channel:
            asyncio.create_task(self._maybe_reset_after_disconnect(guild))
            return

        target = self._autojoin_target(member, before, after)
        if target is not None:
            task = asyncio.create_task(self._maybe_autojoin(member, target))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return

        if not vc:
            return

        if vc.channel and (before.channel == vc.channel or after.channel == vc.channel):
            humans = [m for m in vc.channel.members if not m.bot]
            if humans:
                self._last_active[guild.id] = time.time()
            else:
                # Per-guild override: leave_when_alone can be set to 0 (off)
                # to keep the bot in voice when everyone leaves. Default is
                # on — that's the only behaviour the original bot had.
                gs = await db.get_guild_settings(guild.id)
                enabled = gs["leave_when_alone"]
                if enabled == 0:
                    return
                log.info("Alone in %s — disconnecting.", vc.channel.name)
                reset_state(guild.id)
                await vc.disconnect()

    async def _maybe_reset_after_disconnect(self, guild: discord.Guild) -> None:
        """Wait a few seconds, then either clear queue (real disconnect) or
        leave it alone (discord.py reconnected and resumed)."""
        await asyncio.sleep(8)   # discord.py's voice reconnect typically lands in 2-5s
        if guild.voice_client is not None:
            # Reconnected — discord.py is back in voice on its own. The
            # `vc.play()` that was in progress kept running through the blip
            # (ffmpeg doesn't care that the gateway dropped), so the queue
            # state is still valid.
            log.info("Voice reconnected in %s — keeping queue.", guild.name)
            return
        log.info("Bot disconnected from %s — resetting state.", guild.name)
        reset_state(guild.id)

    @tasks.loop(seconds=30)
    async def idle_check(self) -> None:
        now = time.time()
        for vc in list(self.bot.voice_clients):
            guild = vc.guild
            playing = vc.is_playing() or vc.is_paused()
            if playing:
                self._last_active[guild.id] = now
                continue
            # Resolve effective idle timeout: per-guild override (None=use global,
            # 0=disabled, >0=this many seconds) wins over the bot-wide default.
            gs = await db.get_guild_settings(guild.id)
            effective = gs["idle_seconds"] if gs["idle_seconds"] is not None \
                                            else settings.idle_seconds
            if effective <= 0:
                # Disabled for this guild — never auto-leave on idle.
                continue
            last = self._last_active.get(guild.id, now)
            if now - last >= effective:
                log.info("Idle disconnect from %s.", guild.name)
                reset_state(guild.id)
                try:
                    await vc.disconnect()
                except Exception:
                    log.exception("idle disconnect failed")

    @idle_check.before_loop
    async def _before_idle(self) -> None:
        await self.bot.wait_until_ready()

    # ── error handler ────────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_command_error(self, ctx: commands.Context, error: Exception) -> None:
        # Slash invocations of hybrid commands wrap the real error in
        # HybridCommandError(CommandInvokeError(...)) — unwrap all layers so
        # both invocation styles hit the same friendly branches below.
        while isinstance(error, (commands.HybridCommandError,
                                 commands.CommandInvokeError,
                                 discord.app_commands.CommandInvokeError)):
            error = error.original

        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.MissingRequiredArgument):
            return await ctx.send(f"❌ Missing argument: `{error.param.name}`.")
        if isinstance(error, commands.MissingRequiredAttachment):
            return await ctx.send(
                f"❌ This command needs an attached file (`{error.param.name}`)."
            )
        if isinstance(error, commands.BadArgument):
            return await ctx.send(f"❌ Bad argument: {error}")
        if isinstance(error, commands.MissingPermissions):
            return await ctx.send("❌ You don't have permission to do that.")
        if isinstance(error, commands.NoPrivateMessage):
            return await ctx.send("❌ This command only works in a server.")
        if isinstance(error, commands.NotOwner):
            return await ctx.send("❌ That command is owner-only.")
        if isinstance(error, commands.CommandOnCooldown):
            return await ctx.send(
                f"❌ Slow down — try again in {error.retry_after:.0f}s.")
        # Catch-all for the remaining CheckFailure family — must come after
        # the specific subclasses above. Without this branch they fell through
        # to "Something went wrong" plus a logged stack trace.
        if isinstance(error, commands.CheckFailure):
            return await ctx.send("❌ You can't use that command here.")

        log.exception("Unhandled error in %s: %s",
                      getattr(ctx.command, "qualified_name", "?"), error)
        try:
            await ctx.send("❌ Something went wrong. The error has been logged.")
        except Exception:
            pass


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(AdminCog(bot))
