"""
Starboard — when a message collects enough ⭐ reactions, repost it to a
dedicated channel so the good stuff doesn't scroll away.

Per-guild config (all persisted in core.db, so it survives restarts):
  * channel   — where starred messages get reposted (None = disabled)
  * threshold — how many of the star emoji are needed (default 1)
  * emoji     — which reaction counts (default ⭐)

Reactions are handled raw (`on_raw_reaction_add` / `_remove`) so old, uncached
messages still work. Each change recounts the message's stars from scratch — the
authoritative source — then posts, updates the count on, or removes the board
entry. A per-message lock serialises rapid reactions so we never double-post.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

import discord
from discord.ext import commands

from core import db
from core.logging_setup import get_logger

log = get_logger(__name__)


class StarboardCog(commands.Cog, name="Starboard"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # One lock per source message so concurrent reactions don't race into
        # two board posts. Created on demand; fine to let the map grow.
        self._locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── config commands ───────────────────────────────────────────────────────

    @commands.hybrid_group(
        name="starboard", aliases=["stars"], invoke_without_command=True,
        description="Configure the starboard (Manage Server)",
        help=(
            "Repost messages that get enough ⭐ reactions to a chosen channel.\n\n"
            "`starboard show` — current settings\n"
            "`starboard channel #chan` — set the board channel *(Manage Server)*\n"
            "`starboard threshold <n>` — stars needed *(Manage Server)*\n"
            "`starboard emoji <e>` — which reaction counts *(Manage Server)*\n"
            "`starboard disable` — turn it off *(Manage Server)*"
        ),
    )
    async def starboard(self, ctx: commands.Context):
        await self._show(ctx)

    @starboard.command(name="show", description="Show the current starboard settings")
    async def starboard_show(self, ctx: commands.Context):
        await self._show(ctx)

    async def _show(self, ctx: commands.Context) -> None:
        cfg = await db.get_starboard(ctx.guild.id)
        if not cfg["channel_id"]:
            return await ctx.send(
                f"⭐ Starboard is **off**. Turn it on with "
                f"`{ctx.prefix}starboard channel #channel`."
            )
        chan = ctx.guild.get_channel(cfg["channel_id"])
        chan_label = chan.mention if chan else f"`{cfg['channel_id']}` *(missing)*"
        plural = "" if cfg["threshold"] == 1 else "s"
        await ctx.send(
            f"⭐ **Starboard**\n"
            f"• Channel: {chan_label}\n"
            f"• Threshold: **{cfg['threshold']}** {cfg['emoji']} reaction{plural}\n"
            f"• Emoji: {cfg['emoji']}"
        )

    @starboard.command(name="channel",
                       description="Set the starboard channel (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def starboard_channel(self, ctx: commands.Context,
                                channel: discord.TextChannel):
        me = channel.permissions_for(ctx.guild.me)
        if not (me.send_messages and me.embed_links):
            return await ctx.send(
                f"❌ I need **Send Messages** and **Embed Links** in "
                f"{channel.mention}."
            )
        await db.set_starboard_channel(ctx.guild.id, channel.id)
        cfg = await db.get_starboard(ctx.guild.id)
        await ctx.send(
            f"✅ Starboard set to {channel.mention} — messages with "
            f"**{cfg['threshold']}** {cfg['emoji']} get posted there."
        )

    @starboard.command(name="threshold", aliases=["count"],
                       description="Set how many stars are needed (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def starboard_threshold(self, ctx: commands.Context, count: int):
        if count < 1:
            return await ctx.send("❌ Threshold must be at least 1.")
        await db.set_starboard_threshold(ctx.guild.id, count)
        plural = "" if count == 1 else "s"
        await ctx.send(f"✅ Threshold set to **{count}** star{plural}.")

    @starboard.command(name="emoji",
                       description="Set which reaction emoji counts (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def starboard_emoji(self, ctx: commands.Context, emoji: str):
        emoji = emoji.strip()
        if not emoji:
            return await ctx.send("❌ Give me an emoji to use.")
        await db.set_starboard_emoji(ctx.guild.id, emoji)
        await ctx.send(f"✅ Starboard now counts {emoji} reactions.")

    @starboard.command(name="disable", aliases=["off"],
                       description="Turn the starboard off (Manage Server)")
    @commands.has_guild_permissions(manage_guild=True)
    async def starboard_disable(self, ctx: commands.Context):
        await db.set_starboard_channel(ctx.guild.id, None)
        await ctx.send("✅ Starboard disabled.")

    # ── reaction handling ─────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        await self._sync(payload)

    @commands.Cog.listener()
    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent):
        await self._sync(payload)

    async def _sync(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id is None:
            return
        cfg = await db.get_starboard(payload.guild_id)
        board_id = cfg["channel_id"]
        if not board_id:
            return                                   # disabled
        if str(payload.emoji) != cfg["emoji"]:
            return                                   # not the star emoji
        if payload.channel_id == board_id:
            return                                   # ignore stars on the board

        guild = self.bot.get_guild(payload.guild_id)
        if guild is None:
            return
        board = guild.get_channel(board_id)
        if board is None:
            return                                   # board channel deleted

        src_channel = guild.get_channel(payload.channel_id)
        if src_channel is None:
            return
        async with self._locks[payload.message_id]:
            try:
                message = await src_channel.fetch_message(payload.message_id)
            except discord.NotFound:
                # Source deleted — clean up any board post we made.
                await self._remove_post(guild, board, payload.message_id)
                return
            except discord.HTTPException:
                return
            count = self._count_stars(message, cfg["emoji"])
            await self._apply(guild, board, message, count, cfg)

    @staticmethod
    def _count_stars(message: discord.Message, emoji: str) -> int:
        for reaction in message.reactions:
            if str(reaction.emoji) == emoji:
                return reaction.count
        return 0

    async def _apply(self, guild, board: discord.TextChannel,
                     message: discord.Message, count: int, cfg: dict) -> None:
        existing = await db.starboard_get_post(guild.id, message.id)

        if count < cfg["threshold"]:
            # Dropped below the bar — remove the board post if there is one.
            if existing:
                await self._remove_post(guild, board, message.id,
                                        board_msg_id=existing["board_msg_id"])
            return

        embed = self._build_embed(message, count, cfg["emoji"])
        label = f"{cfg['emoji']} **{count}**"

        if existing:
            # Already on the board — just refresh the count.
            try:
                board_msg = await board.fetch_message(existing["board_msg_id"])
                await board_msg.edit(content=label, embed=embed)
                await db.starboard_upsert_post(
                    guild.id, message.id, board_msg.id, count)
            except discord.NotFound:
                # Board post was deleted by a human — repost it.
                await self._create_post(guild, board, message, count, label, embed)
            except discord.HTTPException:
                log.exception("Starboard: failed to update post")
        else:
            await self._create_post(guild, board, message, count, label, embed)

    async def _create_post(self, guild, board, message, count, label, embed) -> None:
        try:
            board_msg = await board.send(content=label, embed=embed)
        except discord.HTTPException:
            log.exception("Starboard: failed to post message")
            return
        await db.starboard_upsert_post(guild.id, message.id, board_msg.id, count)

    async def _remove_post(self, guild, board, source_msg_id: int,
                           *, board_msg_id: int | None = None) -> None:
        if board_msg_id is None:
            existing = await db.starboard_get_post(guild.id, source_msg_id)
            if not existing:
                return
            board_msg_id = existing["board_msg_id"]
        try:
            board_msg = await board.fetch_message(board_msg_id)
            await board_msg.delete()
        except discord.HTTPException:
            pass                                     # already gone — fine
        await db.starboard_delete_post(guild.id, source_msg_id)

    @staticmethod
    def _build_embed(message: discord.Message, count: int,
                     emoji: str) -> discord.Embed:
        embed = discord.Embed(
            description=message.content or "",
            color=discord.Color.gold(),
            timestamp=message.created_at,
        )
        embed.set_author(
            name=message.author.display_name,
            icon_url=message.author.display_avatar.url,
        )
        # First image attachment (or embed image) as the preview.
        image_url = None
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                image_url = att.url
                break
        if image_url is None:
            for e in message.embeds:
                if e.image:
                    image_url = e.image.url
                    break
                if e.thumbnail:
                    image_url = e.thumbnail.url
                    break
        if image_url:
            embed.set_image(url=image_url)

        # Non-image attachments get listed as links so they're not lost.
        files = [a for a in message.attachments if a.url != image_url]
        if files:
            embed.add_field(
                name="Attachments",
                value="\n".join(f"[{a.filename}]({a.url})" for a in files[:5]),
                inline=False,
            )
        embed.add_field(
            name="Source",
            value=f"[Jump to message]({message.jump_url}) · {message.channel.mention}",
            inline=False,
        )
        embed.set_footer(text=f"{emoji} {count}")
        return embed


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(StarboardCog(bot))
