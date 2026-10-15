"""
Image transforms and the little parsers that feed them.

Everything here is pure: an image in, an image out, no Discord and no I/O.
That keeps the fiddly parts — aspect-ratio maths, the binary search that hits
a target file size, meme text wrapping — unit-testable without a bot.

Pillow is imported lazily inside the functions that need it so the module (and
its parsers) can be imported and tested on a host that hasn't installed it.
"""
from __future__ import annotations

import io
import os
import re

# Fonts to try for caption/watermark text, best first. DejaVu ships with
# essentially every Linux distro and with Pillow itself; Impact is the classic
# meme face but is rarely installed outside Windows.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

# Ceilings that keep a malicious or careless upload from exhausting memory.
# A decompression bomb is a 10 KB PNG that expands to 40 000 × 40 000 pixels.
MAX_PIXELS      = 50_000_000
MAX_OUTPUT_EDGE = 8000


class ImageOpError(ValueError):
    """Raised with a message intended to be shown to the user verbatim."""


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsers
# ──────────────────────────────────────────────────────────────────────────────

_FLIP_WORDS = {
    "h": "h", "horizontal": "h", "horizontally": "h", "x": "h",
    "lr": "h", "mirror": "h", "left-right": "h",
    "v": "v", "vertical": "v", "vertically": "v", "y": "v",
    "ud": "v", "flop": "v", "up-down": "v", "upside-down": "v",
}


def parse_flip(text: str | None) -> str | None:
    """'h' / 'vertical' / 'mirror' → 'h' or 'v'. None if unrecognised."""
    if not text:
        return None
    return _FLIP_WORDS.get(text.strip().lower())


def parse_rotation(text: str | None) -> int | str | None:
    """Degrees clockwise, or the string 'auto' for EXIF orientation.

    Any multiple of 90 is lossless-ish; other angles are allowed but will
    need a background fill, so they're normalised to 0–359 and handled by
    :func:`rotate`.
    """
    if not text:
        return None
    t = text.strip().lower().rstrip("°").replace("deg", "").strip()
    if t in ("auto", "exif", "fix", "straighten"):
        return "auto"
    if t in ("left", "ccw", "anticlockwise", "counterclockwise"):
        return 270
    if t in ("right", "cw", "clockwise"):
        return 90
    if t in ("flip", "half", "upsidedown", "upside-down"):
        return 180
    try:
        deg = int(round(float(t)))
    except ValueError:
        return None
    return deg % 360


_RATIO_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)$")
_BOX_RE   = re.compile(r"^(\d+)\s*[x×]\s*(\d+)(?:\s*\+\s*(\d+)\s*\+\s*(\d+))?$")


def parse_crop(text: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Turn a crop spec into a (left, top, right, bottom) box.

    Accepted:
      ``1:1`` / ``16:9``    centre-crop to an aspect ratio
      ``800x600``           centre-crop to a pixel size
      ``800x600+50+20``     explicit box at an offset
      ``80%``               centre-crop to a percentage of each side

    Raises :class:`ImageOpError` with an explanation on anything else.
    """
    t = (text or "").strip().lower()
    if not t:
        raise ImageOpError("Tell me how to crop — `1:1`, `16:9`, `800x600`, "
                           "`800x600+50+20`, or `80%`.")

    if t.endswith("%"):
        try:
            pct = float(t[:-1])
        except ValueError:
            raise ImageOpError(f"`{text}` isn't a percentage.") from None
        if not 1 <= pct <= 100:
            raise ImageOpError("Crop percentage has to be between 1 and 100.")
        w = max(1, int(width * pct / 100))
        h = max(1, int(height * pct / 100))
        return _centre_box(width, height, w, h)

    m = _BOX_RE.match(t)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if w <= 0 or h <= 0:
            raise ImageOpError("Crop width and height must be positive.")
        if m.group(3) is not None:
            left, top = int(m.group(3)), int(m.group(4))
            if left >= width or top >= height:
                raise ImageOpError(
                    f"That offset starts outside the image "
                    f"({width}×{height}).")
            return (left, top, min(width, left + w), min(height, top + h))
        return _centre_box(width, height, min(w, width), min(h, height))

    m = _RATIO_RE.match(t)
    if m:
        rw, rh = float(m.group(1)), float(m.group(2))
        if rw <= 0 or rh <= 0:
            raise ImageOpError("Aspect ratio parts must be positive.")
        target = rw / rh
        if width / height > target:      # too wide — trim the sides
            w = int(round(height * target))
            h = height
        else:                            # too tall — trim top and bottom
            w = width
            h = int(round(width / target))
        return _centre_box(width, height, max(1, w), max(1, h))

    raise ImageOpError(
        f"Don't understand the crop `{text}`.\n"
        "Try `1:1`, `16:9`, `800x600`, `800x600+50+20`, or `80%`.")


def _centre_box(width: int, height: int, w: int, h: int) -> tuple[int, int, int, int]:
    left = max(0, (width - w) // 2)
    top  = max(0, (height - h) // 2)
    return (left, top, min(width, left + w), min(height, top + h))


def parse_scale(text: str, width: int, height: int) -> tuple[int, int]:
    """Turn a resize spec into a target (width, height).

    Accepted: ``800`` (width, aspect kept), ``50%``, ``800x600`` (exact),
    ``x600`` (height, aspect kept), ``2x`` (multiplier).
    """
    t = (text or "").strip().lower().replace(" ", "")
    if not t:
        raise ImageOpError("Give me a size — `800`, `50%`, `800x600`, or `x600`.")

    def _clamp(w: int, h: int) -> tuple[int, int]:
        w, h = max(1, w), max(1, h)
        if max(w, h) > MAX_OUTPUT_EDGE:
            raise ImageOpError(
                f"That would be {w}×{h} — the limit is "
                f"{MAX_OUTPUT_EDGE}px on the longest side.")
        return w, h

    if t.endswith("%"):
        try:
            pct = float(t[:-1])
        except ValueError:
            raise ImageOpError(f"`{text}` isn't a percentage.") from None
        if pct <= 0:
            raise ImageOpError("Scale percentage has to be positive.")
        return _clamp(int(round(width * pct / 100)), int(round(height * pct / 100)))

    if t.endswith("x") and t[:-1].replace(".", "", 1).isdigit():
        mult = float(t[:-1])
        if mult <= 0:
            raise ImageOpError("Scale multiplier has to be positive.")
        return _clamp(int(round(width * mult)), int(round(height * mult)))

    if t.startswith("x") and t[1:].isdigit():
        h = int(t[1:])
        return _clamp(int(round(width * h / height)), h)

    m = re.match(r"^(\d+)[x×](\d+)$", t)
    if m:
        return _clamp(int(m.group(1)), int(m.group(2)))

    if t.isdigit():
        w = int(t)
        return _clamp(w, int(round(height * w / width)))

    raise ImageOpError(
        f"Don't understand the size `{text}`.\n"
        "Try `800` (width), `50%`, `800x600`, `x600` (height), or `2x`.")


def parse_factor(text: str | None, *, default: float, lo: float, hi: float,
                 label: str) -> float:
    """A multiplier like `1.5`, `150%` or `+50%`, clamped to a sane range."""
    if not text:
        return default
    t = text.strip().lower().lstrip("+")
    try:
        value = float(t[:-1]) / 100 if t.endswith("%") else float(t)
    except ValueError:
        raise ImageOpError(f"`{text}` isn't a number. {label} takes e.g. "
                           f"`1.5`, `0.7`, or `150%`.") from None
    if not lo <= value <= hi:
        raise ImageOpError(f"{label} has to be between {lo:g} and {hi:g} "
                           f"— got {value:g}.")
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Loading / saving
# ──────────────────────────────────────────────────────────────────────────────

def pillow_available() -> bool:
    try:
        import PIL  # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False


def open_image(path: str):
    """Open an image, apply its EXIF rotation, and reject bombs.

    Phone photos carry their orientation in EXIF rather than in the pixels, so
    without `exif_transpose` every op below would silently work on a sideways
    image and then write it out with the tag stripped.
    """
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        im = Image.open(path)
        im.load()
    except Image.DecompressionBombError:
        raise ImageOpError("That image is too large to process safely.") from None
    except Exception:
        raise ImageOpError(
            "That doesn't look like an image I can open. Supported: png, jpg, "
            "webp, gif, bmp, tiff.") from None
    if im.width * im.height > MAX_PIXELS:
        raise ImageOpError(
            f"That image is {im.width}×{im.height} — too big to process.")
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def output_format(im, src_ext: str, *, need_alpha: bool = False) -> tuple[str, str]:
    """Pick (Pillow format, file extension) for the result.

    Transparency decides it: a JPEG can't carry an alpha channel, so anything
    that has one (or is about to gain one) has to land as PNG regardless of
    what came in.
    """
    ext = (src_ext or "").lower().lstrip(".")
    has_alpha = need_alpha or im.mode in ("RGBA", "LA", "PA") or \
        (im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        return "PNG", "png"
    if ext in ("jpg", "jpeg"):
        return "JPEG", "jpg"
    if ext == "webp":
        return "WEBP", "webp"
    if ext in ("bmp", "tif", "tiff"):
        return "PNG", "png"
    return "PNG", "png"


def save_image(im, path: str, fmt: str, *, quality: int = 92) -> None:
    """Write an image, converting the mode if the format demands it."""
    if fmt == "JPEG" and im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    kwargs: dict = {}
    if fmt == "JPEG":
        kwargs = {"quality": quality, "optimize": True, "progressive": True}
    elif fmt == "WEBP":
        kwargs = {"quality": quality, "method": 6}
    elif fmt == "PNG":
        kwargs = {"optimize": True}
    im.save(path, fmt, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Operations
# ──────────────────────────────────────────────────────────────────────────────

def flip(im, direction: str):
    from PIL import Image
    return im.transpose(Image.FLIP_LEFT_RIGHT if direction == "h"
                        else Image.FLIP_TOP_BOTTOM)


def rotate(im, degrees: int):
    """Rotate clockwise. Right angles are exact; anything else expands the
    canvas and fills the corners with transparency."""
    from PIL import Image
    deg = degrees % 360
    if deg == 0:
        return im
    if deg == 90:
        return im.transpose(Image.ROTATE_270)     # PIL rotates anticlockwise
    if deg == 180:
        return im.transpose(Image.ROTATE_180)
    if deg == 270:
        return im.transpose(Image.ROTATE_90)
    if im.mode not in ("RGBA", "LA"):
        im = im.convert("RGBA")
    return im.rotate(-deg, expand=True, resample=Image.BICUBIC)


def crop(im, box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    if right - left < 1 or bottom - top < 1:
        raise ImageOpError("That crop leaves nothing behind.")
    return im.crop(box)


def scale(im, size: tuple[int, int]):
    from PIL import Image
    return im.resize(size, Image.LANCZOS)


def thumbnail(im, edge: int = 320):
    from PIL import Image
    out = im.copy()
    out.thumbnail((edge, edge), Image.LANCZOS)
    return out


def grayscale(im):
    from PIL import ImageOps
    # Preserve alpha: converting straight to "L" would drop it.
    if im.mode in ("RGBA", "LA"):
        alpha = im.getchannel("A")
        out = ImageOps.grayscale(im.convert("RGB")).convert("RGBA")
        out.putalpha(alpha)
        return out
    return ImageOps.grayscale(im.convert("RGB"))


def invert(im):
    from PIL import ImageOps
    if im.mode in ("RGBA", "LA"):
        alpha = im.getchannel("A")
        out = ImageOps.invert(im.convert("RGB")).convert("RGBA")
        out.putalpha(alpha)
        return out
    return ImageOps.invert(im.convert("RGB"))


def blur(im, radius: float = 4.0):
    from PIL import ImageFilter
    return im.filter(ImageFilter.GaussianBlur(radius))


def sharpen(im, factor: float = 2.0):
    from PIL import ImageEnhance
    return ImageEnhance.Sharpness(im).enhance(factor)


def brightness(im, factor: float = 1.3):
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(im).enhance(factor)


def contrast(im, factor: float = 1.3):
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(im).enhance(factor)


def shrink_to(im, target_bytes: int, fmt: str) -> tuple[bytes, int, tuple[int, int]]:
    """Squeeze an image under `target_bytes`. Returns (data, quality, size).

    Quality is searched first because it's free; only when even the lowest
    usable quality overshoots does it start scaling the image down, which is
    the more destructive lever. Raises :class:`ImageOpError` if the target is
    unreachable.
    """
    if fmt == "PNG":
        # PNG is lossless, so there's no quality dial to turn — resolution is
        # the only lever, and JPEG will beat it every time at a size target.
        fmt = "JPEG"
    work = im.convert("RGB") if fmt == "JPEG" and im.mode != "RGB" else im

    for scale_factor in (1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12):
        if scale_factor < 1.0:
            from PIL import Image
            w = max(1, int(work.width * scale_factor))
            h = max(1, int(work.height * scale_factor))
            candidate = work.resize((w, h), Image.LANCZOS)
        else:
            candidate = work
        lo, hi, best = 20, 95, None
        while lo <= hi:                      # binary search on quality
            q = (lo + hi) // 2
            buf = io.BytesIO()
            save_image(candidate, buf, fmt, quality=q)
            data = buf.getvalue()
            if len(data) <= target_bytes:
                best = (data, q, candidate.size)
                lo = q + 1                   # try to spend the headroom
            else:
                hi = q - 1
        if best is not None:
            return best
    raise ImageOpError(
        "Couldn't get it under that size without destroying it. Try a bigger "
        "target, or `scale` it down first.")


# ──────────────────────────────────────────────────────────────────────────────
# Fitting images to Discord's emoji / sticker limits
#
# Discord's rules are strict and its error message is not helpful ("File cannot
# be larger than 256.0 kb"), so rather than bouncing the upload back at the
# user these fit the image for them.
# ──────────────────────────────────────────────────────────────────────────────

EMOJI_MAX_BYTES   = 256 * 1024
EMOJI_EDGE        = 128      # Discord renders emoji at 128px; larger is wasted
STICKER_MAX_BYTES = 512 * 1024
STICKER_SIZE      = 320      # required exactly, not a maximum


def is_animated(data: bytes) -> bool:
    """True for a multi-frame GIF/WEBP/PNG."""
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            return getattr(im, "n_frames", 1) > 1
    except Exception:
        return False


def _save_gif_frames(frames, durations, loop: int) -> bytes:
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:],
                   duration=durations, loop=loop, optimize=True,
                   disposal=2)
    return buf.getvalue()


def _fit_animated(data: bytes, *, max_bytes: int, edge: int) -> tuple[bytes, str]:
    """Shrink an animated GIF until it fits, keeping it animated.

    Three levers, applied in that order because that's the order of how much
    you notice them: resolution, then frame rate, then colour depth. Falls
    back to a single still frame only if the animation can't be made to fit
    at all — a static emoji beats a rejected upload.
    """
    from PIL import Image, ImageSequence

    src = Image.open(io.BytesIO(data))
    loop = src.info.get("loop", 0)

    def _frames_at(scale: float, step: int):
        src.seek(0)
        out, durs = [], []
        for i, frame in enumerate(ImageSequence.Iterator(src)):
            if i % step:
                continue
            durs.append(frame.info.get("duration", 100) * step)
            f = frame.convert("RGBA")
            w = max(1, int(f.width * scale))
            h = max(1, int(f.height * scale))
            out.append(f.resize((w, h), Image.LANCZOS).convert(
                "P", palette=Image.ADAPTIVE, colors=256))
        return out, durs

    base = min(1.0, edge / max(src.width, src.height))
    for scale, step in ((base, 1), (base, 2), (base * 0.75, 2),
                        (base * 0.6, 3), (base * 0.5, 3), (base * 0.4, 4)):
        frames, durs = _frames_at(scale, step)
        if not frames:
            break
        out = _save_gif_frames(frames, durs, loop)
        if len(out) <= max_bytes:
            note = ""
            if step > 1:
                note = f"kept every {step}{'nd' if step == 2 else 'rd' if step == 3 else 'th'} frame"
            return out, note

    # Couldn't keep it moving — fall back to the first frame.
    src.seek(0)
    still = src.convert("RGBA")
    still.thumbnail((edge, edge), Image.LANCZOS)
    buf = io.BytesIO()
    still.save(buf, "PNG", optimize=True)
    return buf.getvalue(), "too big to stay animated — used the first frame"


def fit_emoji(data: bytes, *, max_bytes: int = EMOJI_MAX_BYTES,
              edge: int = EMOJI_EDGE) -> tuple[bytes, str]:
    """Make `data` uploadable as a custom emoji. Returns (bytes, note).

    `note` is empty when nothing had to be given up, and otherwise says what
    changed, so the user isn't surprised by a quietly degraded emoji.
    """
    from PIL import Image

    if len(data) <= max_bytes and not _oversized(data, edge):
        return data, ""

    if is_animated(data):
        return _fit_animated(data, max_bytes=max_bytes, edge=edge)

    im = Image.open(io.BytesIO(data))
    im = im.convert("RGBA")
    original = im.size
    im.thumbnail((edge, edge), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    out = buf.getvalue()

    if len(out) > max_bytes:
        # A 128px PNG almost always fits; if not, quantise the palette.
        for colors in (256, 128, 64, 32):
            q = im.convert("P", palette=Image.ADAPTIVE, colors=colors)
            buf = io.BytesIO()
            q.save(buf, "PNG", optimize=True)
            if len(buf.getvalue()) <= max_bytes:
                out = buf.getvalue()
                break
        else:
            raise ImageOpError(
                "Couldn't get that under Discord's 256 KB emoji limit even at "
                "128px. Try a simpler image.")

    note = ""
    if original != im.size:
        note = f"resized {original[0]}×{original[1]} → {im.width}×{im.height}"
    return out, note


def _oversized(data: bytes, edge: int) -> bool:
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            return max(im.size) > edge
    except Exception:
        return False


def fit_sticker(data: bytes, *, max_bytes: int = STICKER_MAX_BYTES,
                size: int = STICKER_SIZE) -> tuple[bytes, str]:
    """Make `data` uploadable as a sticker. Returns (bytes, note).

    Discord requires exactly 320×320 PNG, so a non-square source is scaled to
    fit and centred on a transparent canvas rather than being stretched or
    cropped — neither of which the user asked for.
    """
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)          # APNG/GIF: sticker uploads take a single frame
    original = im.size
    im = im.convert("RGBA")

    changed = original != (size, size)
    if original != (size, size):
        fitted = im.copy()
        fitted.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(fitted, ((size - fitted.width) // 2,
                              (size - fitted.height) // 2))
        im = canvas

    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    out = buf.getvalue()
    if len(out) > max_bytes:
        for colors in (256, 128, 64, 32):
            q = im.convert("P", palette=Image.ADAPTIVE, colors=colors)
            buf = io.BytesIO()
            q.save(buf, "PNG", optimize=True)
            if len(buf.getvalue()) <= max_bytes:
                out = buf.getvalue()
                break
        else:
            raise ImageOpError(
                "Couldn't get that under Discord's 512 KB sticker limit at "
                "320×320. Try a simpler image.")

    note = (f"fitted {original[0]}×{original[1]} → {size}×{size}"
            if changed else "")
    return out, note


# ──────────────────────────────────────────────────────────────────────────────
# QR codes
# ──────────────────────────────────────────────────────────────────────────────

QR_MAX_CHARS = 1500          # well inside version-40's ~2 900 byte ceiling


def qr_available() -> bool:
    try:
        import qrcode  # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False


def qr_reader_available() -> bool:
    try:
        import zxingcpp  # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False


def make_qr(text: str, *, scale: int = 10, border: int = 4):
    """Render `text` as a QR code image.

    Error correction is set to Q (~25% recoverable) rather than the default M,
    because these get screenshotted, resized and photographed off screens —
    the extra redundancy costs a slightly denser code and buys a lot of
    real-world scannability.
    """
    import qrcode

    if not text or not text.strip():
        raise ImageOpError("Nothing to encode.")
    if len(text) > QR_MAX_CHARS:
        raise ImageOpError(
            f"That's {len(text)} characters — a QR code tops out around "
            f"{QR_MAX_CHARS}. Shorten it, or link to it instead.")
    qr = qrcode.QRCode(
        version=None,                      # auto-size to the content
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=scale, border=border,
    )
    qr.add_data(text)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def read_qr(path: str) -> list[tuple[str, str]]:
    """Decode every barcode in an image. Returns [(format, text), …].

    Tries the image as-is first, then upscaled — a QR photographed or
    screenshotted small often decodes only after enlargement.
    """
    import zxingcpp
    from PIL import Image

    im = open_image(path)
    attempts = [im]
    if max(im.size) < 800:
        scale = 800 / max(im.size)
        attempts.append(im.resize((int(im.width * scale),
                                   int(im.height * scale)), Image.LANCZOS))
    # Some codes only read once colour is out of the way.
    attempts.append(im.convert("L"))

    for candidate in attempts:
        try:
            results = zxingcpp.read_barcodes(candidate)
        except Exception:
            continue
        found = [(str(r.format).rsplit(".", 1)[-1], r.text)
                 for r in results if r.text]
        if found:
            return found
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Text: captions and watermarks
# ──────────────────────────────────────────────────────────────────────────────

def _load_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        # Pillow bundles DejaVuSans; findable by name on most installs.
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        # Last resort: the bitmap default, which ignores `size` entirely.
        return ImageFont.load_default()


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy word wrap. Words longer than a line are broken mid-word."""
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    out: list[str] = []
    for line in lines:
        while draw.textlength(line, font=font) > max_width and len(line) > 1:
            cut = len(line) - 1
            while cut > 1 and draw.textlength(line[:cut], font=font) > max_width:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
        out.append(line)
    return out


def caption(im, top: str = "", bottom: str = ""):
    """Meme-style outlined text across the top and/or bottom of the image."""
    from PIL import ImageDraw

    out  = im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im.copy()
    draw = ImageDraw.Draw(out)
    # Font size tracks the image so a caption looks the same on a 400px
    # thumbnail and a 4000px photo.
    size   = max(14, int(out.height * 0.1))
    margin = max(6, int(out.height * 0.02))
    maxw   = out.width - margin * 2

    for text, at_top in ((top, True), (bottom, False)):
        if not text.strip():
            continue
        font  = _load_font(size)
        lines = _wrap(draw, text.upper(), font, maxw)
        # Shrink until the block uses at most 40% of the height.
        while len(lines) * size > out.height * 0.4 and size > 14:
            size = int(size * 0.85)
            font = _load_font(size)
            lines = _wrap(draw, text.upper(), font, maxw)

        line_h = size * 1.1
        block  = line_h * len(lines)
        y = margin if at_top else out.height - block - margin
        stroke = max(1, size // 12)
        for line in lines:
            w = draw.textlength(line, font=font)
            draw.text(((out.width - w) / 2, y), line, font=font,
                      fill=(255, 255, 255, 255),
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            y += line_h
    return out


def watermark(im, text: str, *, corner: str = "br", opacity: float = 0.55):
    """Stamp small semi-transparent text in a corner."""
    from PIL import Image, ImageDraw

    base  = im.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)
    size  = max(12, int(base.height * 0.035))
    font  = _load_font(size)
    pad   = max(6, int(base.height * 0.015))

    tw = draw.textlength(text, font=font)
    th = size * 1.2
    positions = {
        "br": (base.width - tw - pad, base.height - th - pad),
        "bl": (pad, base.height - th - pad),
        "tr": (base.width - tw - pad, pad),
        "tl": (pad, pad),
    }
    x, y = positions.get(corner, positions["br"])
    alpha = int(max(0.0, min(1.0, opacity)) * 255)
    # A dark outline keeps it legible over both light and dark backgrounds.
    draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha),
              stroke_width=max(1, size // 14), stroke_fill=(0, 0, 0, alpha))
    return Image.alpha_composite(base, layer)
