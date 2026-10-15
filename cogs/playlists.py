"""
Saved-playlists cog. All persistence goes through core.db (SQLite),
replacing the original playlists.json.

Bug fixes vs the original:
  * The doubled `await ctx.send(await ctx.send(...))` in `addmany` is gone.
  * The N+1 yt-dlp probe in `play` is replaced with extract_flat for
    YouTube URLs and the lazy resolver kicks in at play-time.
  * Concurrent writes don't corrupt the data (SQLite WAL).
"""
from __future__ import annotations

import asyncio
import random

import aiohttp
import discord
from discord.ext import commands

from core import audio, db
from core.logging_setup import get_logger
from core.state import get_state

log = get_logger(__name__)


async def _connect_to_author(ctx: commands.Context):
    """Join/move to the caller's voice channel via the music cog's connector.

    Delegating (rather than calling `.connect()` here) matters: the music
    cog connects with `VoiceRecvClient` and attaches the rolling `-clip`
    recorder. This cog's old plain `.connect()` left the whole session with
    a voice client that can't receive audio — `-clip` and voice control were
    silently dead whenever the bot entered voice via `playlist play`, and
    stayed dead until a full disconnect. Imported lazily so `reload music`
    is honoured."""
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.send("❌ You must be in a voice channel.")
        return None
    from cogs.music import _connect_to_author as _music_connect
    return await _music_connect(ctx)


async def _kickstart_music(bot: commands.Bot, guild: discord.Guild,
                            vc: discord.VoiceClient) -> None:
    music = bot.get_cog("Music")
    if music is None:
        log.warning("Music cog not loaded — can't kickstart playback.")
        return
    music._kickstart(guild, vc)     # supervised: a failure reaches the log


class PlaylistsCog(commands.Cog, name="Playlists"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def cog_check(self, ctx: commands.Context) -> bool:
        # Playlists are per-guild; every command here reads ctx.guild.id.
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    @commands.group(name="playlist", aliases=["pl"], invoke_without_command=True,
                     description="Saved playlists — create, add, play, share",
                     help=(
                         "Manage saved playlists. Each server has its own.\n\n"
                         "**Subcommands**\n"
                         "`playlist new <name>` — create empty\n"
                         "`playlist add <name> <url>` — add tracks\n"
                         "`playlist addmany <name>` — paste many URLs interactively\n"
                         "`playlist play <name>` — queue the whole thing\n"
                         "`playlist list` / `info` / `rename` / `delete` / `remove` / `import`"
                     ))
    async def playlist(self, ctx: commands.Context):
        prefix = ctx.prefix
        embed = discord.Embed(title="📂 Playlist commands", color=discord.Color.blurple())
        embed.add_field(name="Create / edit", value=(
            f"`{prefix}playlist new <name>`\n"
            f"`{prefix}playlist add <name> <url> [url ...]`\n"
            f"`{prefix}playlist addmany <name>` *(interactive)*\n"
            f"`{prefix}playlist remove <name> <indices>`\n"
            f"`{prefix}playlist rename <old> <new>`\n"
            f"`{prefix}playlist delete <name>`"
        ), inline=False)
        embed.add_field(name="Use", value=(
            f"`{prefix}playlist play <name>`\n"
            f"`{prefix}playlist list`\n"
            f"`{prefix}playlist info <name>`\n"
            f"`{prefix}playlist import <name> <txt-url>`"
        ), inline=False)
        await ctx.send(embed=embed)

    @playlist.command(name="new", description="Create a new empty playlist",
                       help="Creates a playlist with the given name. Add tracks afterward.")
    async def new(self, ctx: commands.Context, *, name: str):
        ok = await db.create_playlist(ctx.guild.id, name, creator=ctx.author.display_name)
        if not ok:
            return await ctx.send(f"❌ A playlist called **{name}** already exists.")
        await ctx.send(f"✅ Created playlist **{name}**. Add tracks with "
                       f"`{ctx.prefix}playlist add {name} <url>`.")

    @playlist.command(name="add")
    async def add(self, ctx: commands.Context, name: str, *, urls: str):
        pl = await db.get_playlist(ctx.guild.id, name)
        if not pl:
            return await ctx.send(f"❌ Playlist `{name}` not found.")
        good = [u.strip("<>") for u in urls.split() if u.strip("<>").startswith("http")]
        if not good:
            return await ctx.send("❌ No valid URLs found.")
        added, dupes = await db.add_playlist_links(ctx.guild.id, name, good)
        msg = f"✅ Added **{added}** track(s) to **{pl['name']}**."
        if dupes:
            msg += f"\n⚠️ Skipped {len(dupes)} duplicate(s)."
        await ctx.send(msg)

    @playlist.command(name="addmany")
    async def addmany(self, ctx: commands.Context, *, name: str):
        pl = await db.get_playlist(ctx.guild.id, name)
        if not pl:
            return await ctx.send(f"❌ Playlist `{name}` not found.")
        await ctx.send(
            f"📝 Adding to **{pl['name']}** — paste your links (one per line, "
            "or many at once). Type `done` when finished, `cancel` to abort. *(60s timeout)*"
        )

        added_total = 0
        dupes_total = 0

        def check(m: discord.Message) -> bool:
            return m.author == ctx.author and m.channel == ctx.channel

        while True:
            try:
                m = await self.bot.wait_for("message", timeout=60, check=check)
            except asyncio.TimeoutError:
                break
            text = m.content.strip()
            if text.lower() == "done":
                break
            if text.lower() == "cancel":
                return await ctx.send("❌ Cancelled — no changes saved.")
            urls = [tok.strip("<>") for line in text.splitlines() for tok in line.split()
                    if tok.strip("<>").startswith("http")]
            if urls:
                count, dupes = await db.add_playlist_links(ctx.guild.id, name, urls)
                added_total += count
                dupes_total += len(dupes)
                await m.add_reaction("✅")

        if not added_total:
            return await ctx.send("❌ Nothing added.")
        msg = f"✅ Added **{added_total}** track(s) to **{pl['name']}**."
        if dupes_total:
            msg += f"\n⚠️ Skipped {dupes_total} duplicate(s)."
        await ctx.send(msg)

    @playlist.command(name="remove")
    async def remove(self, ctx: commands.Context, name: str, *, args: str):
        from core.formatters import parse_indices
        pl = await db.get_playlist(ctx.guild.id, name)
        if not pl:
            return await ctx.send(f"❌ Playlist `{name}` not found.")
        idxs = parse_indices(args)
        if not idxs:
            return await ctx.send("❌ Invalid format. Examples: `3` · `1,3,5` · `2-5`.")
        bad = [i for i in idxs if not 1 <= i <= len(pl["links"])]
        if bad:
            return await ctx.send(f"❌ Out of range: {sorted(bad)}. "
                                  f"Playlist has {len(pl['links'])}.")
        removed = await db.remove_playlist_links(ctx.guild.id, name, idxs)
        await ctx.send(f"🗑️ Removed **{len(removed)}** from **{pl['name']}**.")

    @playlist.command(name="rename")
    async def rename(self, ctx: commands.Context, old: str, *, new: str):
        result = await db.rename_playlist(ctx.guild.id, old, new)
        if result == "missing":
            return await ctx.send(f"❌ Playlist `{old}` not found.")
        if result == "exists":
            return await ctx.send(f"❌ A playlist called `{new}` already exists.")
        await ctx.send(f"✅ Renamed **{old}** → **{new}**.")

    @playlist.command(name="delete")
    async def delete(self, ctx: commands.Context, *, name: str):
        if await db.delete_playlist(ctx.guild.id, name):
            await ctx.send(f"🗑️ Deleted playlist **{name}**.")
        else:
            await ctx.send(f"❌ Playlist `{name}` not found.")

    @playlist.command(name="list")
    async def list_cmd(self, ctx: commands.Context):
        rows = await db.list_playlists(ctx.guild.id)
        if not rows:
            return await ctx.send(f"📂 No saved playlists. Try `{ctx.prefix}playlist new <name>`.")
        embed = discord.Embed(title="📂 Saved playlists", color=discord.Color.blurple())
        for r in rows:
            embed.add_field(
                name=r["name"],
                value=f"{r['link_count']} track(s) · by {r['creator']}",
                inline=False,
            )
        await ctx.send(embed=embed)

    @playlist.command(name="info")
    async def info(self, ctx: commands.Context, *, name: str):
        pl = await db.get_playlist(ctx.guild.id, name)
        if not pl:
            return await ctx.send(f"❌ Playlist `{name}` not found.")
        if not pl["links"]:
            return await ctx.send(f"📂 **{pl['name']}** is empty.")
        page_size = 15
        first = "\n".join(f"`{i}.` <{u}>" for i, u in enumerate(pl["links"][:page_size], 1))
        more  = f"\n…and {len(pl['links']) - page_size} more" if len(pl['links']) > page_size else ""
        embed = discord.Embed(
            title=f"📂 {pl['name']}",
            description=first + more,
            color=discord.Color.blurple(),
        )
        embed.set_footer(text=f"{len(pl['links'])} tracks · created by {pl['creator']}")
        await ctx.send(embed=embed)

    @playlist.command(
        name="play",
        description="Queue a saved playlist (`--shuffle` to randomize order)",
        help=(
            "Adds every track in the saved playlist to the queue.\n\n"
            "Pass `--shuffle` (or `--random`) anywhere in the name to "
            "randomize the new batch — existing queue items stay in place.\n\n"
            "**Examples**\n"
            "`playlist play roadtrip`\n"
            "`playlist play roadtrip --shuffle`"
        ),
    )
    async def play(self, ctx: commands.Context, *, name: str):
        # Pull --shuffle / --random out of the name. We use token-level
        # matching rather than a substring check so a playlist literally
        # named "no--shuffle-please" wouldn't get false-positived.
        shuffle = False
        cleaned: list[str] = []
        for tok in name.split():
            if tok.lower() in ("--shuffle", "--random"):
                shuffle = True
            else:
                cleaned.append(tok)
        name = " ".join(cleaned).strip()
        if not name:
            return await ctx.send("❌ Provide a playlist name.")

        pl = await db.get_playlist(ctx.guild.id, name)
        if not pl:
            return await ctx.send(f"❌ Playlist `{name}` not found.")
        if not pl["links"]:
            return await ctx.send(f"📂 **{pl['name']}** is empty.")
        vc = await _connect_to_author(ctx)
        if vc is None:
            return

        state  = get_state(ctx.guild.id)
        epoch  = state.queue_epoch
        # Let the playback engine report skips/stalls back to this channel.
        state.text_channel_id = ctx.channel.id
        status = await ctx.send(f"⏳ Loading **{pl['name']}** ({len(pl['links'])} tracks)…")

        # Collect locally first so we can shuffle the batch atomically and
        # not leave a half-built queue if shuffle is requested mid-load.
        new_tracks: list[dict] = []
        failed = 0
        for url in pl["links"]:
            try:
                if audio.is_spotify_url(url):
                    if not audio.spotify_ready():
                        failed += 1
                        continue
                    for t in await audio.fetch_spotify_tracks(url):
                        t["requester"] = ctx.author.display_name
                        t["requester_id"] = ctx.author.id
                        new_tracks.append(t)
                else:
                    new_tracks.append({
                        "title":          url,
                        "stream_url":     None,
                        "webpage_url":    url,
                        "duration":       0,
                        "thumbnail":      None,
                        "requester":      ctx.author.display_name,
                        "requester_id":   ctx.author.id,
                        "_needs_resolve": True,
                        "_search_query":  url,
                    })
            except Exception:
                log.exception("queueing %s failed", url)
                failed += 1

        if shuffle:
            random.shuffle(new_tracks)

        # Loading awaited the network: a stop/disconnect meanwhile wins.
        live = ctx.guild.voice_client
        if state.queue_epoch != epoch or live is None or not live.is_connected():
            return await status.edit(content=f"⏹️ Playback was stopped while I was loading "
                                             f"**{pl['name']}**, so nothing was queued.")
        vc = live
        for t in new_tracks:
            t["request_channel_id"] = ctx.channel.id
        state.queue.extend(new_tracks)
        added = len(new_tracks)

        suffix = f" ({failed} failed)" if failed else ""
        shuffle_note = " 🎲 *shuffled*" if shuffle else ""
        await status.edit(
            content=f"📂 Queued **{added}** track(s) from **{pl['name']}**.{shuffle_note}{suffix}"
        )
        if not vc.is_playing() and not vc.is_paused():
            await _kickstart_music(self.bot, ctx.guild, vc)

    @playlist.command(
        name="savequeue", aliases=["save-queue", "snapshot"],
        description="Save the current queue as a new playlist",
        help=(
            "Snapshots the now-playing track plus everything queued into a "
            "new saved playlist, so you can `playlist play` the same set later.\n\n"
            "**Example**\n`playlist savequeue friday-night`"
        ),
    )
    async def savequeue(self, ctx: commands.Context, *, name: str):
        state  = get_state(ctx.guild.id)
        tracks = ([state.current] if state.current else []) + list(state.queue)
        urls: list[str] = []
        skipped = 0
        for t in tracks:
            url = t.get("webpage_url")
            # Attachment/CDN links expire — a playlist of those would be dead
            # within a day, so they're skipped rather than saved.
            if url and url.startswith("http") and not audio.is_discord_cdn(url):
                if url not in urls:
                    urls.append(url)
            else:
                skipped += 1
        if not urls:
            return await ctx.send("❌ Nothing in the queue that can be saved "
                                  "(attachment tracks can't be).")
        ok = await db.create_playlist(ctx.guild.id, name,
                                      creator=ctx.author.display_name)
        if not ok:
            return await ctx.send(
                f"❌ A playlist called **{name}** already exists. Pick another "
                f"name, or `{ctx.prefix}playlist delete {name}` first."
            )
        added, _dupes = await db.add_playlist_links(ctx.guild.id, name, urls)
        note = f" ({skipped} attachment track(s) couldn't be saved)" if skipped else ""
        await ctx.send(
            f"💾 Saved **{added}** track(s) to playlist **{name}**.{note}\n"
            f"Play it back anytime with `{ctx.prefix}playlist play {name}`."
        )

    @playlist.command(name="import")
    async def import_cmd(self, ctx: commands.Context, name: str, txt_url: str):
        txt_url = txt_url.strip("<>")
        if not txt_url.startswith("http"):
            return await ctx.send("❌ URL must start with http(s).")
        if "pastebin.com" in txt_url and "/raw/" not in txt_url:
            txt_url = txt_url.replace("pastebin.com/", "pastebin.com/raw/")
        elif "gist.github.com" in txt_url and "/raw" not in txt_url:
            txt_url = txt_url.rstrip("/") + "/raw"

        status = await ctx.send(f"⏳ Fetching `{txt_url}` …")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    txt_url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={"User-Agent": "DiscordMusicBot/2.0"},
                ) as r:
                    r.raise_for_status()
                    raw = await r.text()
        except Exception as e:
            return await status.edit(content=f"❌ Couldn't fetch: {e}")

        urls: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("http"):
                urls.append(line)
        if not urls:
            return await status.edit(content="❌ No URLs found in that file.")
        if len(urls) > 500:
            return await status.edit(content=f"❌ Too many URLs ({len(urls)}). Maximum is 500.")

        await db.overwrite_playlist(
            ctx.guild.id, name,
            creator=ctx.author.display_name, source_url=txt_url,
        )
        await db.add_playlist_links(ctx.guild.id, name, urls)
        await status.edit(content=f"✅ Imported **{name}** with **{len(urls)}** tracks.")


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(PlaylistsCog(bot))
