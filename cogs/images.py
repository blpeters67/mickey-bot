"""
Image editing cog — the things you need in a pinch and don't want to open
Photoshop for.

    -flip h            -rotate 90        -crop 1:1         -scale 50%
    -shrink 500kb      -thumb            -grayscale        -invert
    -blur 6            -sharpen          -brightness 1.4   -contrast 1.2
    -caption top | bottom                -watermark @me
    -nobg              (AI background removal)

Every command takes an attached image or a direct image URL, and replies with
the result as an attachment. The transforms themselves live in
`core.imageops`, which is pure and unit-tested; this file is the Discord layer
plus the download/upload plumbing.

Two optional dependencies, both degrading to a clear message rather than a
crash: **Pillow** (everything here) and **rembg** (`-nobg` only).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import urllib.parse

import aiohttp
import discord
from discord.ext import commands

from core import imageops
from core.config import settings
from core.formatters import fmt_size, parse_size, unwrap_url
from core.imageops import ImageOpError
from core.logging_setup import get_logger

log = get_logger(__name__)

# Discord's attachment ceiling on an unboosted server. Results over this get
# shrunk to fit rather than failing — a slightly smaller image is a much
# better outcome than an error.
DISCORD_LIMIT   = 10 * 1024 * 1024
UPLOAD_TARGET   = int(DISCORD_LIMIT * 0.95)
MAX_SOURCE      = 40 * 1024 * 1024

IMAGE_EXTS = ("png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff")

_PILLOW_MISSING = (
    "❌ Image editing isn't available — the host doesn't have **Pillow** "
    "installed.\n-# The bot owner can enable it with `pip install Pillow` "
    "and a restart."
)
_REMBG_MISSING = (
    "❌ Background removal isn't enabled on this bot.\n"
    "-# Owner: `pip install rembg onnxruntime` on the host, then restart. "
    "The first run downloads a ~180 MB model."
)


class ImagesCog(commands.Cog, name="Images"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        # rembg's session is expensive to build (it loads the ONNX model), so
        # it's created once on first use and kept.
        self._rembg_session = None
        self._rembg_lock = asyncio.Lock()

    # ── shared plumbing ──────────────────────────────────────────────────────

    async def _fetch_source(
        self, ctx: commands.Context,
        attachment: discord.Attachment | None, url: str | None, workdir: str,
    ) -> tuple[str, str] | None:
        """Get the input image onto disk. Returns (path, filename) or None."""
        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]

        if attachment is not None:
            if attachment.size > MAX_SOURCE:
                await ctx.send(f"❌ That image is {fmt_size(attachment.size)} — "
                               f"the limit is {fmt_size(MAX_SOURCE)}.")
                return None
            name = os.path.basename(attachment.filename or "") or "image.png"
            path = os.path.join(workdir, name)
            try:
                await attachment.save(path)
            except discord.HTTPException as e:
                await ctx.send(f"❌ Couldn't fetch the attachment: {e}")
                return None
            return path, name

        if not url:
            await ctx.send(
                "❌ Attach an image or give me a direct image URL.\n"
                "-# Right-clicking a message → Apps also works for some of these."
            )
            return None

        url = unwrap_url(url)
        if not url.startswith(("http://", "https://")):
            await ctx.send(f"❌ `{url}` isn't a URL I can fetch.")
            return None
        try:
            timeout = aiohttp.ClientTimeout(total=120, sock_connect=20)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(
                    url, headers={"User-Agent": "DiscordMusicBot/2.0"},
                ) as resp:
                    resp.raise_for_status()
                    ctype = (resp.content_type or "").lower()
                    if ctype.startswith("text/"):
                        await ctx.send(
                            "❌ That link is a web page, not an image file. "
                            "Right-click the image itself and copy *its* address."
                        )
                        return None
                    if (resp.content_length or 0) > MAX_SOURCE:
                        await ctx.send(
                            f"❌ That image is "
                            f"{fmt_size(resp.content_length)} — the limit is "
                            f"{fmt_size(MAX_SOURCE)}.")
                        return None
                    name = os.path.basename(
                        urllib.parse.urlparse(url).path.rstrip("/")) or "image.png"
                    path = os.path.join(workdir, name)
                    got = 0
                    with open(path, "wb") as fh:
                        async for chunk in resp.content.iter_chunked(1 << 16):
                            got += len(chunk)
                            if got > MAX_SOURCE:
                                await ctx.send(
                                    f"❌ That image went over the "
                                    f"{fmt_size(MAX_SOURCE)} limit mid-download.")
                                return None
                            fh.write(chunk)
            return path, name
        except aiohttp.ClientError as e:
            await ctx.send(f"❌ Couldn't download that image: {e}")
            return None

    async def _run(
        self, ctx: commands.Context, op, *,
        attachment: discord.Attachment | None, url: str | None,
        label: str, suffix: str, need_alpha: bool = False,
        note: str | None = None,
    ) -> None:
        """Fetch → open → `op(image)` → send. The spine of every command here.

        `op` is a plain function taking a Pillow image and returning one; it
        runs in a worker thread because Pillow is CPU-bound and would
        otherwise stall the bot's event loop mid-resize.
        """
        if not imageops.pillow_available():
            return await ctx.send(_PILLOW_MISSING)

        workdir = tempfile.mkdtemp(prefix="musicbot-img-", dir=settings.download_tmp)
        try:
            got = await self._fetch_source(ctx, attachment, url, workdir)
            if got is None:
                return
            src, src_name = got
            stem, src_ext = os.path.splitext(src_name)

            async with ctx.typing():
                try:
                    result = await asyncio.to_thread(self._apply, src, op)
                except ImageOpError as e:
                    return await ctx.send(f"❌ {e}")
                except Exception:
                    log.exception("%s failed on %s", label, src_name)
                    return await ctx.send(
                        f"❌ `{label}` failed on that image. It may be corrupt "
                        "or in a format I can't read.")

                im, extra_note = result
                fmt, ext = imageops.output_format(im, src_ext,
                                                  need_alpha=need_alpha)
                out = os.path.join(workdir, f"{stem}_{suffix}.{ext}")
                await asyncio.to_thread(imageops.save_image, im, out, fmt)

                size = os.path.getsize(out)
                shrunk = ""
                if size > UPLOAD_TARGET:
                    # Auto-fit rather than refusing: a 12 MB PNG the user can
                    # actually receive beats a clean error about 10 MB.
                    try:
                        data, _q, dims = await asyncio.to_thread(
                            imageops.shrink_to, im, UPLOAD_TARGET, fmt)
                        ext = "jpg"
                        out = os.path.join(workdir, f"{stem}_{suffix}.jpg")
                        with open(out, "wb") as fh:
                            fh.write(data)
                        shrunk = (f"\n-# Result was {fmt_size(size)} — "
                                  f"compressed to {dims[0]}×{dims[1]} jpg to fit "
                                  "Discord's 10 MB limit.")
                        size = os.path.getsize(out)
                    except ImageOpError:
                        return await ctx.send(
                            f"❌ The result came out at {fmt_size(size)}, over "
                            "Discord's 10 MB limit, and wouldn't compress down.")

            parts = [f"🖼️ **{stem}_{suffix}.{ext}** `{im.width}×{im.height}` "
                     f"`{fmt_size(size)}`"]
            if note:
                parts.append(note)
            if extra_note:
                parts.append(extra_note)
            await ctx.send(
                " · ".join(parts) + shrunk,
                file=discord.File(out, filename=os.path.basename(out)),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _apply(src: str, op) -> tuple:
        """Open, transform, and normalise the op's return value."""
        im = imageops.open_image(src)
        result = op(im)
        if isinstance(result, tuple):
            return result
        return result, None

    # ── geometry ─────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="flip", aliases=["mirror"],
        description="Flip an image horizontally or vertically",
        help=(
            "Mirrors an attached image (or one at a URL).\n\n"
            "`flip h` — left-right, the mirror-selfie fix\n"
            "`flip v` — top-bottom\n\n"
            "**Examples**\n"
            "`flip h` *(with an image attached)*\n"
            "`flip v https://cdn.example.com/photo.jpg`"
        ),
    )
    async def flip(self, ctx: commands.Context, direction: str = "h",
                   attachment: discord.Attachment | None = None,
                   *, url: str | None = None):
        direction, url, err = _split_arg(
            direction, url, imageops.parse_flip, "h",
            "❌ Flip which way? `h` (horizontal) or `v` (vertical).")
        if err:
            return await ctx.send(err)
        await self._run(ctx, lambda im: imageops.flip(im, direction),
                        attachment=attachment, url=url,
                        label="flip", suffix="flip")

    @commands.hybrid_command(
        name="rotate", aliases=["turn"],
        description="Rotate an image (90/180/270, any angle, or auto)",
        help=(
            "Rotates an image clockwise.\n\n"
            "`rotate 90` · `rotate 180` · `rotate 270` — exact quarter turns\n"
            "`rotate left` / `rotate right` — the same, in words\n"
            "`rotate 7` — any angle; the canvas grows and the corners go "
            "transparent\n"
            "`rotate auto` — straighten a phone photo using its EXIF "
            "orientation tag, which is why some pictures look sideways "
            "everywhere except the app they were taken in\n\n"
            "**Examples**\n"
            "`rotate 90` *(with an image attached)*\n"
            "`rotate auto`\n"
            "`rotate 180 https://cdn.example.com/photo.jpg`"
        ),
    )
    async def rotate(self, ctx: commands.Context, angle: str = "90",
                     attachment: discord.Attachment | None = None,
                     *, url: str | None = None):
        angle, url, err = _split_arg(
            angle, url, imageops.parse_rotation, "90",
            "❌ Rotate by how much? `90`, `180`, `270`, `left`, `right`, "
            "any number of degrees, or `auto` for the EXIF orientation.")
        if err:
            return await ctx.send(err)

        if angle == "auto":
            # open_image already applies exif_transpose, so "auto" is a no-op
            # transform that re-saves with the rotation baked into the pixels.
            return await self._run(
                ctx, lambda im: (im, "straightened via EXIF"),
                attachment=attachment, url=url,
                label="rotate", suffix="rotated")
        await self._run(ctx, lambda im: imageops.rotate(im, angle),
                        attachment=attachment, url=url,
                        label="rotate", suffix="rotated",
                        need_alpha=angle % 90 != 0)

    @commands.hybrid_command(
        name="crop",
        description="Crop an image to a ratio, size, or explicit box",
        help=(
            "Centre-crops an image, or cuts an exact box out of it.\n\n"
            "`crop 1:1` — square, for an avatar\n"
            "`crop 16:9` — widescreen\n"
            "`crop 800x600` — centred at that pixel size\n"
            "`crop 800x600+50+20` — exact box, offset 50px right and 20px down\n"
            "`crop 80%` — trim 10% off each edge\n\n"
            "**Examples**\n"
            "`crop 1:1` *(with an image attached)*\n"
            "`crop 16:9 https://cdn.example.com/photo.jpg`"
        ),
    )
    async def crop(self, ctx: commands.Context, spec: str | None = None,
                   attachment: discord.Attachment | None = None,
                   *, url: str | None = None):
        if spec and _is_url(spec) and not url:
            url, spec = unwrap_url(spec), None
        if not spec:
            return await ctx.send(
                "❌ Crop to what? `1:1`, `16:9`, `800x600`, `800x600+50+20`, "
                "or `80%`.")

        def _op(im):
            box = imageops.parse_crop(spec, im.width, im.height)
            out = imageops.crop(im, box)
            return out, f"cropped from {im.width}×{im.height}"

        await self._run(ctx, _op, attachment=attachment, url=url,
                        label="crop", suffix="crop")

    @commands.hybrid_command(
        name="scale", aliases=["imageresize", "imgresize"],
        description="Resize an image by width, percentage, or exact size",
        help=(
            "Resizes an image with high-quality Lanczos resampling.\n\n"
            "`scale 800` — 800px wide, aspect ratio kept\n"
            "`scale x600` — 600px tall, aspect ratio kept\n"
            "`scale 50%` — half size\n"
            "`scale 2x` — double\n"
            "`scale 800x600` — exact, ignoring aspect ratio\n\n"
            "*(For video, that's `resize` — this one is images.)*\n\n"
            "**Examples**\n"
            "`scale 50%` *(with an image attached)*\n"
            "`scale 1920 https://cdn.example.com/photo.jpg`"
        ),
    )
    async def scale(self, ctx: commands.Context, spec: str | None = None,
                    attachment: discord.Attachment | None = None,
                    *, url: str | None = None):
        if spec and _is_url(spec) and not url:
            url, spec = unwrap_url(spec), None
        if not spec:
            return await ctx.send(
                "❌ Resize to what? `800`, `50%`, `2x`, `x600`, or `800x600`.")

        def _op(im):
            size = imageops.parse_scale(spec, im.width, im.height)
            out  = imageops.scale(im, size)
            return out, f"was {im.width}×{im.height}"

        await self._run(ctx, _op, attachment=attachment, url=url,
                        label="scale", suffix="scaled")

    @commands.hybrid_command(
        name="shrink", aliases=["compress", "imgcompress"],
        description="Compress an image under a target file size",
        help=(
            "Squeezes an image below a size you name — for an upload limit, a "
            "form that won't take anything over 2 MB, or an email attachment.\n\n"
            "Quality is lowered first, and only when that isn't enough does it "
            "start reducing resolution, so you lose as little as possible. The "
            "reply says what it ended up at.\n\n"
            "Sizes: `500kb` · `2mb` · `0.5mb` · `500` *(bare number = KB)*\n\n"
            "**Examples**\n"
            "`shrink 500kb` *(with an image attached)*\n"
            "`shrink 2mb https://cdn.example.com/photo.png`"
        ),
    )
    async def shrink(self, ctx: commands.Context, target: str | None = None,
                     attachment: discord.Attachment | None = None,
                     *, url: str | None = None):
        if target and _is_url(target) and not url:
            url, target = unwrap_url(target), None
        if not target:
            return await ctx.send("❌ Shrink to what size? `500kb`, `2mb`, `0.5mb`.")

        # parse_size treats a bare number as MB (right for video); for an image
        # a bare "500" almost always means kilobytes.
        raw = target.strip().lower()
        target_bytes = (int(float(raw) * 1024) if raw.replace(".", "", 1).isdigit()
                        else parse_size(raw))
        if not target_bytes or target_bytes < 5 * 1024:
            return await ctx.send(
                "❌ Give me a target like `500kb`, `2mb` or `0.5mb` "
                "*(minimum 5 KB)*.")

        if not imageops.pillow_available():
            return await ctx.send(_PILLOW_MISSING)

        workdir = tempfile.mkdtemp(prefix="musicbot-img-", dir=settings.download_tmp)
        try:
            got = await self._fetch_source(ctx, attachment, url, workdir)
            if got is None:
                return
            src, src_name = got
            stem, src_ext = os.path.splitext(src_name)
            before = os.path.getsize(src)

            async with ctx.typing():
                try:
                    im = await asyncio.to_thread(imageops.open_image, src)
                    fmt, _ext = imageops.output_format(im, src_ext)
                    data, quality, dims = await asyncio.to_thread(
                        imageops.shrink_to, im, target_bytes, fmt)
                except ImageOpError as e:
                    return await ctx.send(f"❌ {e}")
                except Exception:
                    log.exception("shrink failed on %s", src_name)
                    return await ctx.send("❌ Couldn't compress that image.")

            out = os.path.join(workdir, f"{stem}_small.jpg")
            with open(out, "wb") as fh:
                fh.write(data)
            saved = (1 - len(data) / before) * 100 if before else 0
            resized = (f", resized to {dims[0]}×{dims[1]}"
                       if dims != (im.width, im.height) else "")
            await ctx.send(
                f"🗜️ **{stem}_small.jpg** — {fmt_size(before)} → "
                f"**{fmt_size(len(data))}** *({saved:.0f}% smaller, "
                f"quality {quality}{resized})*",
                file=discord.File(out, filename=os.path.basename(out)),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @commands.hybrid_command(
        name="thumb", aliases=["thumbnail"],
        description="Make a small preview of an image",
        help=(
            "Shrinks an image to fit a small box (320px by default) with its "
            "aspect ratio intact — a quick preview or a forum avatar.\n\n"
            "`thumb` — 320px\n"
            "`thumb 128` — any edge size up to 1024\n\n"
            "**Examples**\n"
            "`thumb` *(with an image attached)*\n"
            "`thumb 64 https://cdn.example.com/photo.jpg`"
        ),
    )
    async def thumb(self, ctx: commands.Context, edge: str = "320",
                    attachment: discord.Attachment | None = None,
                    *, url: str | None = None):
        if edge and _is_url(edge) and not url:
            url, edge = unwrap_url(edge), "320"
        try:
            px = int(edge)
        except ValueError:
            return await ctx.send("❌ Thumbnail size has to be a number, e.g. `128`.")
        if not 16 <= px <= 1024:
            return await ctx.send("❌ Thumbnail size has to be between 16 and 1024.")
        await self._run(ctx, lambda im: imageops.thumbnail(im, px),
                        attachment=attachment, url=url,
                        label="thumb", suffix="thumb")

    # ── colour ───────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="grayscale", aliases=["greyscale", "bw", "blackandwhite"],
        description="Convert an image to black and white",
        help="Desaturates an image to greyscale. Transparency is preserved.\n\n"
             "**Examples**\n`grayscale` *(with an image attached)*\n"
             "`grayscale https://cdn.example.com/photo.jpg`",
    )
    async def grayscale(self, ctx: commands.Context,
                        attachment: discord.Attachment | None = None,
                        *, url: str | None = None):
        await self._run(ctx, imageops.grayscale, attachment=attachment, url=url,
                        label="grayscale", suffix="bw")

    @commands.hybrid_command(
        name="invert", aliases=["negative"],
        description="Invert an image's colours",
        help="Flips every colour to its opposite — a photo negative. "
             "Transparency is preserved.\n\n"
             "**Examples**\n`invert` *(with an image attached)*",
    )
    async def invert(self, ctx: commands.Context,
                     attachment: discord.Attachment | None = None,
                     *, url: str | None = None):
        await self._run(ctx, imageops.invert, attachment=attachment, url=url,
                        label="invert", suffix="inverted")

    @commands.hybrid_command(
        name="blur",
        description="Blur an image (Gaussian)",
        help=(
            "Gaussian blur. The radius controls how strong it is — 2 is "
            "subtle, 4 is the default, 20 obliterates detail (useful for "
            "hiding something in a screenshot).\n\n"
            "**Examples**\n"
            "`blur` *(with an image attached)*\n"
            "`blur 12`\n"
            "`blur 6 https://cdn.example.com/photo.jpg`"
        ),
    )
    async def blur(self, ctx: commands.Context, radius: str | None = None,
                   attachment: discord.Attachment | None = None,
                   *, url: str | None = None):
        if radius and _is_url(radius) and not url:
            url, radius = unwrap_url(radius), None
        try:
            r = imageops.parse_factor(radius, default=4.0, lo=0.1, hi=100,
                                      label="Blur radius")
        except ImageOpError as e:
            return await ctx.send(f"❌ {e}")
        await self._run(ctx, lambda im: imageops.blur(im, r),
                        attachment=attachment, url=url,
                        label="blur", suffix="blur")

    @commands.hybrid_command(
        name="sharpen",
        description="Sharpen a soft image",
        help=(
            "Increases edge contrast. 2.0 is the default; above ~4 it starts "
            "looking crunchy and amplifies compression artefacts.\n\n"
            "**Examples**\n`sharpen` *(with an image attached)*\n`sharpen 3`"
        ),
    )
    async def sharpen(self, ctx: commands.Context, factor: str | None = None,
                      attachment: discord.Attachment | None = None,
                      *, url: str | None = None):
        if factor and _is_url(factor) and not url:
            url, factor = unwrap_url(factor), None
        try:
            f = imageops.parse_factor(factor, default=2.0, lo=0.0, hi=10,
                                      label="Sharpen factor")
        except ImageOpError as e:
            return await ctx.send(f"❌ {e}")
        await self._run(ctx, lambda im: imageops.sharpen(im, f),
                        attachment=attachment, url=url,
                        label="sharpen", suffix="sharp")

    @commands.hybrid_command(
        name="brightness", aliases=["brighten"],
        description="Brighten or darken an image",
        help=(
            "Above 1 brightens, below 1 darkens. `1.4` lifts a dim photo; "
            "`0.6` pulls back an overexposed one. Percentages work too.\n\n"
            "**Examples**\n"
            "`brightness 1.4` *(with an image attached)*\n"
            "`brightness 60%`"
        ),
    )
    async def brightness(self, ctx: commands.Context, factor: str | None = None,
                         attachment: discord.Attachment | None = None,
                         *, url: str | None = None):
        if factor and _is_url(factor) and not url:
            url, factor = unwrap_url(factor), None
        try:
            f = imageops.parse_factor(factor, default=1.3, lo=0.0, hi=5,
                                      label="Brightness")
        except ImageOpError as e:
            return await ctx.send(f"❌ {e}")
        await self._run(ctx, lambda im: imageops.brightness(im, f),
                        attachment=attachment, url=url,
                        label="brightness", suffix="bright")

    @commands.hybrid_command(
        name="contrast",
        description="Increase or reduce an image's contrast",
        help=(
            "Above 1 increases contrast, below 1 flattens it. Pairs well with "
            "`brightness` for rescuing a washed-out screenshot.\n\n"
            "**Examples**\n`contrast 1.5` *(with an image attached)*\n"
            "`contrast 0.8`"
        ),
    )
    async def contrast(self, ctx: commands.Context, factor: str | None = None,
                       attachment: discord.Attachment | None = None,
                       *, url: str | None = None):
        if factor and _is_url(factor) and not url:
            url, factor = unwrap_url(factor), None
        try:
            f = imageops.parse_factor(factor, default=1.3, lo=0.0, hi=5,
                                      label="Contrast")
        except ImageOpError as e:
            return await ctx.send(f"❌ {e}")
        await self._run(ctx, lambda im: imageops.contrast(im, f),
                        attachment=attachment, url=url,
                        label="contrast", suffix="contrast")

    # ── annotation ───────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="caption", aliases=["meme"],
        description="Add meme-style text to the top and/or bottom",
        help=(
            "Adds big outlined white text across the image, sized to fit and "
            "wrapped automatically.\n\n"
            "Separate top and bottom text with `|`:\n"
            "`caption top text | bottom text`\n"
            "`caption just the top`\n"
            "`caption | just the bottom`\n\n"
            "**Examples**\n"
            "`caption when the build passes | on the first try` "
            "*(with an image attached)*\n"
            "`caption me explaining this bug`"
        ),
    )
    async def caption(self, ctx: commands.Context, *, text: str | None = None):
        attachment = (ctx.message.attachments[0]
                      if ctx.message and ctx.message.attachments else None)
        url = None
        if not text or not text.strip():
            return await ctx.send(
                "❌ What should it say? `caption top text | bottom text` "
                "*(either side can be left empty)*.")

        # A trailing URL means the image came from a link, not an attachment.
        parts = text.rsplit(None, 1)
        if attachment is None and len(parts) == 2 and _is_url(parts[1]):
            text, url = parts[0], unwrap_url(parts[1])

        top, _, bottom = text.partition("|")
        top, bottom = top.strip(), bottom.strip()
        if not top and not bottom:
            return await ctx.send("❌ Both halves of that caption are empty.")
        if len(top) + len(bottom) > 400:
            return await ctx.send("❌ That caption is too long (400 characters max).")

        await self._run(ctx, lambda im: imageops.caption(im, top, bottom),
                        attachment=attachment, url=url,
                        label="caption", suffix="caption")

    @commands.hybrid_command(
        name="watermark", aliases=["stamp"],
        description="Stamp small text in a corner of an image",
        help=(
            "Adds small semi-transparent text in a corner — a credit, a "
            "handle, a 'DRAFT'.\n\n"
            "Corner is optional and defaults to bottom-right: `br`, `bl`, "
            "`tr`, `tl`.\n\n"
            "**Examples**\n"
            "`watermark @alex` *(with an image attached)*\n"
            "`watermark tl CONFIDENTIAL`"
        ),
    )
    async def watermark(self, ctx: commands.Context, *, text: str | None = None):
        attachment = (ctx.message.attachments[0]
                      if ctx.message and ctx.message.attachments else None)
        url = None
        if not text or not text.strip():
            return await ctx.send("❌ What should the watermark say?")

        parts = text.rsplit(None, 1)
        if attachment is None and len(parts) == 2 and _is_url(parts[1]):
            text, url = parts[0], unwrap_url(parts[1])

        corner = "br"
        first, _, rest = text.strip().partition(" ")
        if first.lower() in ("br", "bl", "tr", "tl") and rest.strip():
            corner, text = first.lower(), rest

        text = text.strip()[:100]
        if not text:
            return await ctx.send("❌ What should the watermark say?")
        await self._run(ctx, lambda im: imageops.watermark(im, text, corner=corner),
                        attachment=attachment, url=url,
                        label="watermark", suffix="wm", need_alpha=True)

    # ── QR codes ─────────────────────────────────────────────────────────────

    @commands.hybrid_command(
        name="qr", aliases=["qrcode"],
        description="Make a QR code, or read one out of an image",
        help=(
            "**Make one** — pass the text or link:\n"
            "`qr https://example.com` · `qr wifi password: hunter2`\n"
            "Handy for getting something onto a phone without typing it.\n\n"
            "**Read one** — attach a screenshot or photo containing a code, "
            "or use `qr read <image url>`. It decodes barcodes too, and "
            "upscales small or blurry images before giving up.\n\n"
            "⚠️ A decoded link is shown as **plain text, not a clickable "
            "link** — QR codes are a common phishing vector precisely because "
            "you can't see where they go. Read it before you follow it.\n\n"
            "**Examples**\n"
            "`qr https://example.com`\n"
            "`qr` *(with a screenshot of a QR code attached)*\n"
            "`qr read https://cdn.example.com/poster.jpg`"
        ),
    )
    async def qr(self, ctx: commands.Context, *, text: str | None = None):
        if not imageops.pillow_available():
            return await ctx.send(_PILLOW_MISSING)

        attachment = (ctx.message.attachments[0]
                      if ctx.message and ctx.message.attachments else None)
        body = (text or "").strip()

        # "read"/"scan" forces the decode path; an attachment implies it.
        first, _, rest = body.partition(" ")
        wants_read = first.lower() in ("read", "scan", "decode")
        if wants_read:
            body = rest.strip()

        if attachment is not None or wants_read:
            url = body if body and _is_url(body) else None
            if attachment is None and url is None:
                return await ctx.send(
                    "❌ Attach an image with a QR code in it, or give me a "
                    "direct image URL to read.")
            return await self._read_qr(ctx, attachment=attachment, url=url)

        if not body:
            return await ctx.send(
                "❌ Give me something to encode — `qr https://example.com` — "
                "or attach an image with a QR code in it to read one.")
        await self._make_qr(ctx, body)

    async def _make_qr(self, ctx: commands.Context, text: str) -> None:
        if not imageops.qr_available():
            return await ctx.send(
                "❌ QR generation isn't enabled on this bot.\n"
                "-# Owner: `pip install qrcode` on the host, then restart.")

        workdir = tempfile.mkdtemp(prefix="musicbot-img-", dir=settings.download_tmp)
        try:
            async with ctx.typing():
                try:
                    im = await asyncio.to_thread(imageops.make_qr, text)
                except ImageOpError as e:
                    return await ctx.send(f"❌ {e}")
                except Exception:
                    log.exception("qr generation failed")
                    return await ctx.send("❌ Couldn't build a QR code from that.")
                out = os.path.join(workdir, "qr.png")
                await asyncio.to_thread(imageops.save_image, im, out, "PNG")

            preview = text if len(text) <= 100 else text[:97] + "…"
            await ctx.send(
                f"🔳 QR code for `{preview}` `{im.width}×{im.height}`",
                file=discord.File(out, filename="qr.png"),
                allowed_mentions=discord.AllowedMentions.none(),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _read_qr(self, ctx: commands.Context, *,
                       attachment: discord.Attachment | None,
                       url: str | None) -> None:
        if not imageops.qr_reader_available():
            return await ctx.send(
                "❌ QR reading isn't enabled on this bot.\n"
                "-# Owner: `pip install zxing-cpp` on the host, then restart.")

        workdir = tempfile.mkdtemp(prefix="musicbot-img-", dir=settings.download_tmp)
        try:
            got = await self._fetch_source(ctx, attachment, url, workdir)
            if got is None:
                return
            src, src_name = got

            async with ctx.typing():
                try:
                    found = await asyncio.to_thread(imageops.read_qr, src)
                except ImageOpError as e:
                    return await ctx.send(f"❌ {e}")
                except Exception:
                    log.exception("qr decode failed on %s", src_name)
                    return await ctx.send("❌ Couldn't read that image.")

            if not found:
                return await ctx.send(
                    f"🔳 No QR code or barcode found in **{src_name}**.\n"
                    "-# A sharper or larger crop of just the code usually helps.")

            lines = []
            for fmt, value in found[:5]:
                # Deliberately not a link. A QR code is exactly the situation
                # where the destination is invisible until you decode it, and
                # a one-click link would undo the point of showing it.
                lines.append(f"**{fmt}**\n```\n{value[:800]}\n```")
            head = (f"🔳 Found **{len(found)}** code"
                    f"{'s' if len(found) != 1 else ''} in **{src_name}**")
            warn = ""
            if any(v.lower().startswith(("http://", "https://"))
                   for _f, v in found):
                warn = ("\n-# ⚠️ Shown as plain text on purpose — check where "
                        "it goes before you visit it.")
            await ctx.send(head + "\n" + "\n".join(lines) + warn,
                           allowed_mentions=discord.AllowedMentions.none())
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ── AI background removal ────────────────────────────────────────────────

    async def _ensure_rembg(self):
        """Build the rembg session once, off the event loop."""
        if self._rembg_session is not None:
            return self._rembg_session
        async with self._rembg_lock:
            if self._rembg_session is None:
                from rembg import new_session
                self._rembg_session = await asyncio.to_thread(
                    new_session, "u2net")
                log.info("Loaded rembg u2net session")
        return self._rembg_session

    @commands.hybrid_command(
        name="nobg", aliases=["removebg", "rmbg", "cutout"],
        description="Remove an image's background with AI",
        help=(
            "Cuts the subject out of an image and returns a transparent PNG — "
            "for a product shot, a profile picture, or anything you want to "
            "paste onto a different background.\n\n"
            "Runs locally on the host via "
            "[rembg](https://github.com/danielgatis/rembg) (U²-Net), so there's "
            "no API key and no per-image cost. It's best on a clear "
            "foreground subject; fine detail like loose hair or fur is where "
            "it struggles.\n\n"
            "The **first** run downloads a ~180 MB model, so it can take a "
            "minute — after that it's a few seconds.\n\n"
            "Owner setup: `pip install rembg onnxruntime`.\n\n"
            "**Examples**\n"
            "`nobg` *(with an image attached)*\n"
            "`nobg https://cdn.example.com/product.jpg`"
        ),
    )
    async def nobg(self, ctx: commands.Context,
                   attachment: discord.Attachment | None = None,
                   *, url: str | None = None):
        try:
            import rembg  # noqa: F401  (presence check only)
        except ImportError:
            return await ctx.send(_REMBG_MISSING)
        if not imageops.pillow_available():
            return await ctx.send(_PILLOW_MISSING)

        if attachment is None and ctx.message and ctx.message.attachments:
            attachment = ctx.message.attachments[0]
        if attachment is None and not url:
            return await ctx.send("❌ Attach an image or give me a direct image URL.")

        first_run = self._rembg_session is None
        status = await ctx.send(
            "🪄 Removing background…"
            + ("\n-# First run on this host — downloading the model, which can "
               "take a minute." if first_run else "")
        )
        workdir = tempfile.mkdtemp(prefix="musicbot-img-", dir=settings.download_tmp)
        try:
            got = await self._fetch_source(ctx, attachment, url, workdir)
            if got is None:
                return await status.delete()
            src, src_name = got
            stem = os.path.splitext(src_name)[0]

            try:
                session = await self._ensure_rembg()
            except Exception as e:
                log.exception("rembg session failed to load")
                return await status.edit(
                    content=f"❌ Couldn't load the background-removal model: {e}")

            def _cut() -> bytes:
                from rembg import remove
                with open(src, "rb") as fh:
                    return remove(fh.read(), session=session)

            try:
                data = await asyncio.to_thread(_cut)
            except Exception as e:
                log.exception("rembg failed on %s", src_name)
                return await status.edit(content=f"❌ Background removal failed: {e}")

            out = os.path.join(workdir, f"{stem}_nobg.png")
            with open(out, "wb") as fh:
                fh.write(data)

            size = os.path.getsize(out)
            if size > UPLOAD_TARGET:
                # A transparent cutout has to stay PNG, so scale it down
                # rather than falling back to a JPEG that loses the alpha.
                try:
                    im = await asyncio.to_thread(imageops.open_image, out)
                    small = await asyncio.to_thread(
                        imageops.scale, im,
                        imageops.parse_scale("70%", im.width, im.height))
                    await asyncio.to_thread(imageops.save_image, small, out, "PNG")
                    size = os.path.getsize(out)
                except Exception:
                    log.exception("nobg downscale failed")
                if size > DISCORD_LIMIT:
                    return await status.edit(
                        content=(f"❌ The cutout came out at {fmt_size(size)}, "
                                 "over Discord's 10 MB limit. Try `scale 50%` "
                                 "on the source first."))

            await status.delete()
            await ctx.send(
                f"🪄 **{stem}_nobg.png** `{fmt_size(size)}` — transparent PNG",
                file=discord.File(out, filename=os.path.basename(out)),
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _is_url(text: str | None) -> bool:
    return bool(text) and unwrap_url(text).startswith(("http://", "https://"))


def _split_arg(value: str | None, url: str | None, parser, default: str,
               error: str):
    """Resolve a `<option> [url]` pair where the URL may be in either slot.

    Returns (parsed_value, url, error_message_or_None) — the same shape the
    media commands use, so `flip https://…` works as well as `flip h`.
    """
    if value and _is_url(value) and not url:
        return parser(default), unwrap_url(value), None
    parsed = parser(value)
    if parsed is None:
        return None, url, error
    return parsed, url, None


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(ImagesCog(bot))
