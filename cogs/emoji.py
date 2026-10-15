"""
Emoji & sticker management.

Commands:
  -emoji add <name> [source]      add an emoji from an attachment, URL,
                                  or another emoji ref like <:laugh:123>
  -emoji remove <name>            delete one of this server's emojis
  -sticker add <name> <keyword>   add a sticker (from attachment, URL,
                                  or by replying to a message with one)
  -sticker remove <name>

This cog still records emoji/sticker *usage* live (it counts emoji refs in
messages + reactions and stickers in messages), but the stats VIEWS live with
the rest of the stats now — `stats emoji`, `stats sticker`, and the unified
`stats scan` (see cogs/serverstats.py).
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

import aiohttp
import discord
from discord.ext import commands

from core import db, imageops
from core.imageops import ImageOpError
from core.logging_setup import get_logger

log = get_logger(__name__)

EMOJI_MAX_SIZE   = 256 * 1024
STICKER_MAX_SIZE = 512 * 1024
FETCH_TIMEOUT    = 10
LIST_PAGE_LIMIT  = 50  # max rows per list command response

# Oversized sources are fitted rather than refused, so the fetch ceiling is
# much higher than Discord's own limit — we just need enough of the original
# to downscale from. Anything past this is a mistake, not an emoji.
SOURCE_FETCH_MAX = 25 * 1024 * 1024


def _fit_or_explain(data: bytes, *, kind: str) -> tuple[bytes, str] | tuple[None, str]:
    """Fit bytes to Discord's emoji/sticker rules. Returns (data, note) or
    (None, error) — the error is already user-facing.

    Without Pillow this can only pass the original through and let Discord
    reject it, which is what the cog did before.
    """
    fit = imageops.fit_emoji if kind == "emoji" else imageops.fit_sticker
    limit = EMOJI_MAX_SIZE if kind == "emoji" else STICKER_MAX_SIZE
    if not imageops.pillow_available():
        if len(data) > limit:
            return None, (f"❌ That image is {len(data)//1024} KB; Discord's "
                          f"{kind} limit is {limit//1024} KB, and Pillow isn't "
                          "installed on the host so I can't resize it.")
        return data, ""
    try:
        return fit(data)
    except ImageOpError as e:
        return None, f"❌ {e}"
    except Exception:
        log.exception("%s fitting failed", kind)
        # Fitting is a convenience — if it breaks, let Discord have its say.
        return data, ""

# <a:name:id> or <:name:id>
CUSTOM_EMOJI_RE = re.compile(r"<(a?):(\w+):(\d+)>")


async def _fetch_bytes(url: str, *, max_size: int) -> bytes:
    """GET a URL, return bytes. Raises ValueError if the file is over max_size."""
    timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            # Honour content-length if the server sent one — saves us reading
            # a 50 MB blob just to reject it.
            cl = resp.headers.get("content-length")
            if cl and int(cl) > max_size:
                raise ValueError(
                    f"file too large: {int(cl)//1024} KB (max {max_size//1024} KB)"
                )
            data = await resp.read()
    if len(data) > max_size:
        raise ValueError(
            f"file too large: {len(data)//1024} KB (max {max_size//1024} KB)"
        )
    return data


class EmojiCog(commands.Cog, name="Emoji"):
    async def cog_check(self, ctx: commands.Context) -> bool:
        # Emojis, stickers, and their usage stats are all per-guild.
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ── Usage listeners ──────────────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Track emoji refs and stickers in every message. Skip bots so we
        # don't count our own usage embeds.
        if message.author.bot or not message.guild:
            return

        now = time.time()

        # Custom emoji refs in the message body. De-dupe within one message:
        # spamming "<:laugh:123><:laugh:123>" still counts as one use.
        seen: set[int] = set()
        for match in CUSTOM_EMOJI_RE.finditer(message.content):
            animated = match.group(1) == "a"
            name     = match.group(2)
            eid      = int(match.group(3))
            if eid in seen:
                continue
            seen.add(eid)
            is_native = any(e.id == eid for e in message.guild.emojis)
            try:
                await db.record_emoji_use(
                    guild_id=message.guild.id, emoji_id=eid, name=name,
                    is_native=is_native, is_animated=animated,
                    kind="message", now=now,
                )
                await db.record_user_emoji(
                    guild_id=message.guild.id, user_id=message.author.id,
                    emoji_id=eid, name=name, is_animated=animated, now=now,
                )
            except Exception:
                log.exception("emoji usage record failed")

        # Stickers attached to the message.
        for st in message.stickers:
            is_native = any(s.id == st.id for s in message.guild.stickers)
            try:
                await db.record_sticker_use(
                    guild_id=message.guild.id, sticker_id=st.id,
                    name=st.name, is_native=is_native, now=now,
                )
            except Exception:
                log.exception("sticker usage record failed")

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        if user.bot or not reaction.message.guild:
            return
        e = reaction.emoji
        # Unicode reactions are strings; we don't track those (would need a
        # full unicode-emoji DB and the user mostly cares about custom ones).
        if isinstance(e, str) or e.id is None:
            return
        is_native = any(x.id == e.id for x in reaction.message.guild.emojis)
        try:
            now = time.time()
            await db.record_emoji_use(
                guild_id=reaction.message.guild.id, emoji_id=e.id, name=e.name,
                is_native=is_native, is_animated=bool(e.animated),
                kind="reaction", now=now,
            )
            await db.record_user_emoji(
                guild_id=reaction.message.guild.id, user_id=user.id,
                emoji_id=e.id, name=e.name, is_animated=bool(e.animated), now=now,
            )
        except Exception:
            log.exception("reaction usage record failed")

    # ── Emoji command group ─────────────────────────────────────────────────

    @commands.hybrid_group(
        name="emoji", invoke_without_command=True,
        description="Manage emojis (add/remove). Usage stats live in `stats emoji`.",
        help=(
            "**Subcommands**\n"
            "`emoji add <name> [source]` — add from attachment, URL, or `<:name:id>`\n"
            "`emoji remove <name>` — delete one\n\n"
            "📊 Emoji **usage stats** moved to `stats emoji` (and `stats emoji "
            "@member` for one person); backfill with `stats scan`."
        ),
    )
    async def emoji_group(self, ctx: commands.Context):
        prefix = ctx.prefix
        embed = discord.Embed(
            title="🙂 Emoji commands",
            color=discord.Color.blurple(),
            description=(
                f"`{prefix}emoji add <name>` — attach image, or pass `<:other:id>`, or a URL\n"
                f"`{prefix}emoji remove <name>`\n"
                f"-# Usage stats are now `{prefix}stats emoji` · `{prefix}stats sticker`"
            ),
        )
        await ctx.send(embed=embed)

    @emoji_group.command(
        name="add",
        description="Add an emoji from an image, another emoji, or a URL",
        help=(
            "**Sources** (one of):\n"
            "• Attach an image to the message\n"
            "• Pass another emoji ref: `emoji add laugh <:laugh:123456789>`\n"
            "• Pass a URL: `emoji add fire https://example.com/fire.png`\n\n"
            "Name must be 2–32 characters, letters/digits/underscore only.\n\n"
            "**Any size works.** Discord caps emojis at 128px and 256 KB; "
            "anything bigger is resized to fit automatically, and animated "
            "GIFs stay animated (frames are dropped before the animation is, "
            "and the bot says what it had to give up)."
        ),
    )
    @commands.has_guild_permissions(manage_emojis=True)
    async def emoji_add(
        self, ctx: commands.Context, name: str, source: Optional[str] = None,
        image: Optional[discord.Attachment] = None,
    ):
        if not re.fullmatch(r"\w{2,32}", name):
            return await ctx.send(
                "❌ Name must be 2–32 characters, letters/digits/underscore only."
            )

        # Pull bytes from one of three places, in priority order:
        # (1) slash command `image:` attachment, (2) prefix message attachment,
        # (3) `source` arg (emoji ref or URL).
        attachment = image or (
            ctx.message.attachments[0] if ctx.message.attachments else None
        )

        try:
            if attachment is not None:
                if attachment.size > SOURCE_FETCH_MAX:
                    return await ctx.send(
                        f"❌ That's {attachment.size//1024//1024} MB — I'll "
                        "resize anything reasonable, but that's not an emoji."
                    )
                data = await attachment.read()
            elif source:
                # PartialEmoji.from_str handles <:name:id>, <a:name:id>, and
                # raw unicode. For unicode there's no image to steal, bail.
                pe = discord.PartialEmoji.from_str(source)
                if pe.id is not None:
                    url = str(pe.url)
                elif source.startswith(("http://", "https://")):
                    url = source
                else:
                    return await ctx.send(
                        "❌ Source must be an emoji like `<:name:id>` or a URL."
                    )
                data = await _fetch_bytes(url, max_size=SOURCE_FETCH_MAX)
            else:
                return await ctx.send(
                    "❌ Provide an image attachment, an emoji reference, or a URL."
                )
        except ValueError as e:
            return await ctx.send(f"❌ {e}")
        except Exception as e:
            log.exception("emoji source fetch failed")
            return await ctx.send(f"❌ Couldn't fetch source: {e}")

        # Pillow is CPU-bound; a large animated GIF takes long enough to
        # notice, so keep it off the event loop.
        before = len(data)
        data, note = await asyncio.to_thread(_fit_or_explain, data, kind="emoji")
        if data is None:
            return await ctx.send(note)

        try:
            new = await ctx.guild.create_custom_emoji(name=name, image=data)
        except discord.HTTPException as e:
            # Slot-full, name conflicts, and anything else Discord objects to.
            msg = getattr(e, "text", str(e)) or str(e)
            return await ctx.send(f"❌ Discord rejected it: {msg}")

        tail = ""
        if note:
            tail = f"\n-# {note} ({before//1024} KB → {len(data)//1024} KB)"
        await ctx.send(f"✅ Added {new} `:{new.name}:`{tail}")

    @emoji_group.command(
        name="remove", aliases=["delete"],
        description="Delete an emoji from this server",
        help="Pass the emoji name (no colons), the `:name:`, or the full `<:name:id>` ref.",
    )
    @commands.has_guild_permissions(manage_emojis=True)
    async def emoji_remove(self, ctx: commands.Context, *, name: str):
        target = None
        try:
            pe = discord.PartialEmoji.from_str(name)
            if pe.id is not None:
                target = discord.utils.get(ctx.guild.emojis, id=pe.id)
        except Exception:
            pass
        if target is None:
            stripped = name.strip(":")
            target = discord.utils.get(ctx.guild.emojis, name=stripped)
        if target is None:
            return await ctx.send(f"❌ No emoji `{name}` on this server.")

        try:
            await target.delete()
            await ctx.send(f"🗑️ Removed `:{target.name}:`")
        except discord.HTTPException as e:
            await ctx.send(f"❌ {getattr(e, 'text', e)}")

    # ── Sticker command group ───────────────────────────────────────────────

    @commands.hybrid_group(
        name="sticker", invoke_without_command=True,
        description="Manage and inspect stickers (use a subcommand)",
        help=(
            "**Subcommands**\n"
            "`sticker add <name> <keyword>` — `keyword` is the unicode emoji "
            "the sticker shows up under in the picker (e.g. `joy`, `fire`)\n"
            "`sticker remove <name>`\n\n"
            "📊 Sticker **usage stats** moved to `stats sticker`.\n"
            "Sticker source: attach a PNG, reply to a message with a sticker, "
            "or pass a URL. Discord requires 320×320, PNG, ≤ 512 KB."
        ),
    )
    async def sticker_group(self, ctx: commands.Context):
        prefix = ctx.prefix
        embed = discord.Embed(
            title="🖼️ Sticker commands",
            color=discord.Color.blurple(),
            description=(
                f"`{prefix}sticker add <name> <keyword>` "
                "*(attach image, reply to sticker, or pass URL)*\n"
                f"`{prefix}sticker remove <name>`\n"
                f"-# Usage stats are now `{prefix}stats sticker`"
            ),
        )
        await ctx.send(embed=embed)

    @sticker_group.command(
        name="add",
        description="Add a sticker. Attach an image, reply to a sticker, or pass a URL",
        help=(
            "**Sources** (one of):\n"
            "• Attach an image — any size or format\n"
            "• Reply to a message that contains a sticker — first sticker is grabbed\n"
            "• Pass a URL as the third arg: `sticker add silly joy https://...`\n\n"
            "`keyword` is the unicode emoji name (no colons) under which the "
            "sticker shows up in Discord's picker — `joy`, `fire`, `eyes`, etc.\n\n"
            "**Any size works.** Discord requires exactly 320×320 PNG under "
            "512 KB; the bot scales your image to fit and centres it on a "
            "transparent canvas, so a non-square source isn't stretched or "
            "cropped."
        ),
    )
    @commands.has_guild_permissions(manage_emojis_and_stickers=True)
    async def sticker_add(
        self, ctx: commands.Context, name: str, keyword: str,
        url: Optional[str] = None,
        image: Optional[discord.Attachment] = None,
    ):
        if not 2 <= len(name) <= 30:
            return await ctx.send("❌ Sticker name must be 2–30 characters.")
        if not re.fullmatch(r"[a-z0-9_]+", keyword.lower()):
            return await ctx.send(
                "❌ `keyword` must be a unicode emoji name (e.g. `joy`, `fire`)."
            )

        # Source priority: slash attachment, prefix attachment, reply, URL arg.
        attachment = image or (
            ctx.message.attachments[0] if ctx.message.attachments else None
        )

        ref_sticker = None
        if attachment is None and url is None:
            # Maybe the user is replying to a stickered message?
            ref = getattr(ctx.message, "reference", None)
            if ref and ref.message_id:
                try:
                    ref_msg = await ctx.channel.fetch_message(ref.message_id)
                    if ref_msg.stickers:
                        ref_sticker = ref_msg.stickers[0]
                except Exception:
                    pass

        try:
            if attachment is not None:
                if attachment.size > SOURCE_FETCH_MAX:
                    return await ctx.send(
                        f"❌ That's {attachment.size//1024//1024} MB — I'll "
                        "resize anything reasonable, but that's not a sticker."
                    )
                data = await attachment.read()
            elif ref_sticker is not None:
                data = await _fetch_bytes(str(ref_sticker.url),
                                          max_size=SOURCE_FETCH_MAX)
            elif url:
                data = await _fetch_bytes(url, max_size=SOURCE_FETCH_MAX)
            else:
                return await ctx.send(
                    "❌ Attach an image, reply to a message with a sticker, "
                    "or pass a URL."
                )
        except ValueError as e:
            return await ctx.send(f"❌ {e}")
        except Exception as e:
            log.exception("sticker source fetch failed")
            return await ctx.send(f"❌ Couldn't fetch source: {e}")

        before = len(data)
        data, note = await asyncio.to_thread(_fit_or_explain, data, kind="sticker")
        if data is None:
            return await ctx.send(note)

        # discord.py wants a File-like object for the sticker upload. The name
        # must end .png — fit_sticker always produces one.
        from io import BytesIO
        file_obj = discord.File(BytesIO(data), filename=f"{name}.png")
        try:
            new = await ctx.guild.create_sticker(
                name=name,
                description=f"Added by {ctx.author.display_name}",
                emoji=keyword.lower(),
                file=file_obj,
                reason=f"Added via /sticker add by {ctx.author}",
            )
        except discord.HTTPException as e:
            msg = getattr(e, "text", str(e)) or str(e)
            return await ctx.send(f"❌ Discord rejected it: {msg}")

        tail = ""
        if note:
            tail = f"\n-# {note} ({before//1024} KB → {len(data)//1024} KB)"
        await ctx.send(
            f"✅ Added sticker `{new.name}` (keyword: `:{keyword}:`){tail}")

    @sticker_group.command(
        name="remove", aliases=["delete"],
        description="Delete a sticker from this server",
        help="Pass the sticker's name (case-insensitive).",
    )
    @commands.has_guild_permissions(manage_emojis_and_stickers=True)
    async def sticker_remove(self, ctx: commands.Context, *, name: str):
        name_l = name.lower().strip()
        target = next((s for s in ctx.guild.stickers if s.name.lower() == name_l), None)
        if target is None:
            return await ctx.send(f"❌ No sticker named `{name}` on this server.")
        try:
            await target.delete()
            await ctx.send(f"🗑️ Removed sticker `{target.name}`")
        except discord.HTTPException as e:
            await ctx.send(f"❌ {getattr(e, 'text', e)}")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(EmojiCog(bot))
