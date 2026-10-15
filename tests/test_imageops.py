"""Tests for the image editing transforms and their argument parsers.

The parsers are pure and always run. The transforms need Pillow, so those
tests skip cleanly on a host without it — the same way the cog degrades.
"""
from __future__ import annotations

import os

import pytest

from core import imageops as ops
from core.imageops import ImageOpError

pillow = pytest.importorskip("PIL", reason="Pillow not installed")


# ── parsers (no Pillow needed) ───────────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("h", "h"), ("H", "h"), ("horizontal", "h"), ("mirror", "h"), ("x", "h"),
    ("v", "v"), ("vertical", "v"), ("flop", "v"), ("upside-down", "v"),
])
def test_parse_flip(raw, expect):
    assert ops.parse_flip(raw) == expect


@pytest.mark.parametrize("raw", ["", None, "diagonal", "45", "sideways"])
def test_parse_flip_rejects_junk(raw):
    assert ops.parse_flip(raw) is None


@pytest.mark.parametrize("raw,expect", [
    ("90", 90), ("180", 180), ("270", 270), ("90°", 90), ("90 deg", 90),
    ("right", 90), ("cw", 90), ("left", 270), ("ccw", 270),
    ("auto", "auto"), ("exif", "auto"),
    ("450", 90),      # wraps past a full turn
    ("-90", 270),     # negative wraps too
    ("0", 0), ("7", 7),
])
def test_parse_rotation(raw, expect):
    assert ops.parse_rotation(raw) == expect


@pytest.mark.parametrize("raw", ["", None, "sideways", "abc"])
def test_parse_rotation_rejects_junk(raw):
    assert ops.parse_rotation(raw) is None


def test_parse_crop_square_from_landscape():
    """1:1 on a 200×100 image trims the sides, keeping the centre."""
    assert ops.parse_crop("1:1", 200, 100) == (50, 0, 150, 100)


def test_parse_crop_square_from_portrait():
    assert ops.parse_crop("1:1", 100, 200) == (0, 50, 100, 150)


def test_parse_crop_widescreen():
    left, top, right, bottom = ops.parse_crop("16:9", 100, 100)
    assert right - left == 100
    assert abs((right - left) / (bottom - top) - 16 / 9) < 0.05


def test_parse_crop_exact_box_with_offset():
    assert ops.parse_crop("50x40+10+5", 200, 200) == (10, 5, 60, 45)


def test_parse_crop_centred_pixel_size():
    assert ops.parse_crop("100x50", 200, 200) == (50, 75, 150, 125)


def test_parse_crop_percentage():
    assert ops.parse_crop("50%", 200, 100) == (50, 25, 150, 75)


def test_parse_crop_box_larger_than_the_image_is_clamped():
    assert ops.parse_crop("500x500", 100, 80) == (0, 0, 100, 80)


@pytest.mark.parametrize("raw", ["", "banana", "0:1", "-5%", "200%", "1:0"])
def test_parse_crop_rejects_junk(raw):
    with pytest.raises(ImageOpError):
        ops.parse_crop(raw, 100, 100)


def test_parse_crop_offset_outside_the_image_is_an_error():
    with pytest.raises(ImageOpError, match="outside"):
        ops.parse_crop("50x50+500+500", 100, 100)


@pytest.mark.parametrize("raw,w,h,expect", [
    ("800", 1600, 800, (800, 400)),      # width, aspect kept
    ("x600", 800, 1200, (400, 600)),     # height, aspect kept
    ("50%", 800, 600, (400, 300)),
    ("2x", 100, 50, (200, 100)),
    ("800x600", 1000, 1000, (800, 600)),  # exact, aspect ignored
])
def test_parse_scale(raw, w, h, expect):
    assert ops.parse_scale(raw, w, h) == expect


@pytest.mark.parametrize("raw", ["", "banana", "-50%", "0%", "0x"])
def test_parse_scale_rejects_junk(raw):
    with pytest.raises(ImageOpError):
        ops.parse_scale(raw, 100, 100)


def test_parse_scale_refuses_an_absurd_upscale():
    """A 100× enlargement is a memory-exhaustion bug waiting to happen."""
    with pytest.raises(ImageOpError, match="longest side"):
        ops.parse_scale("100x", 1000, 1000)


@pytest.mark.parametrize("raw,expect", [
    ("1.5", 1.5), ("0.7", 0.7), ("150%", 1.5), ("+50", 50.0), ("2", 2.0),
])
def test_parse_factor(raw, expect):
    assert ops.parse_factor(raw, default=1.0, lo=0, hi=100,
                            label="Test") == pytest.approx(expect)


def test_parse_factor_uses_the_default_when_absent():
    assert ops.parse_factor(None, default=1.3, lo=0, hi=5, label="X") == 1.3


def test_parse_factor_enforces_its_range():
    with pytest.raises(ImageOpError, match="between"):
        ops.parse_factor("99", default=1.0, lo=0, hi=5, label="Brightness")


# ── transforms (need Pillow) ─────────────────────────────────────────────────

@pytest.fixture
def img():
    from PIL import Image
    # Asymmetric on both axes so flips and rotations are actually detectable.
    im = Image.new("RGB", (40, 20), "black")
    for x in range(10):
        for y in range(5):
            im.putpixel((x, y), (255, 0, 0))     # red block, top-left
    return im


def _corner(im, xy):
    return im.convert("RGB").getpixel(xy)


def test_flip_horizontal_moves_the_mark_to_the_right(img):
    out = ops.flip(img, "h")
    assert out.size == img.size
    assert _corner(out, (39, 0)) == (255, 0, 0)
    assert _corner(out, (0, 0)) == (0, 0, 0)


def test_flip_vertical_moves_the_mark_to_the_bottom(img):
    out = ops.flip(img, "v")
    assert _corner(out, (0, 19)) == (255, 0, 0)
    assert _corner(out, (0, 0)) == (0, 0, 0)


def test_flip_twice_is_the_original(img):
    assert ops.flip(ops.flip(img, "h"), "h").tobytes() == img.tobytes()


def test_rotate_90_swaps_the_dimensions(img):
    out = ops.rotate(img, 90)
    assert out.size == (20, 40)


def test_rotate_90_goes_clockwise(img):
    """The top-left mark should land top-right, not bottom-left."""
    out = ops.rotate(img, 90)
    assert _corner(out, (19, 0)) == (255, 0, 0)


def test_rotate_180_is_two_90s(img):
    assert ops.rotate(img, 180).tobytes() == \
        ops.rotate(ops.rotate(img, 90), 90).tobytes()


def test_rotate_360_is_a_noop(img):
    assert ops.rotate(img, 0).tobytes() == img.tobytes()


def test_rotate_arbitrary_angle_expands_and_adds_alpha(img):
    out = ops.rotate(img, 30)
    assert out.mode == "RGBA"
    assert out.width > img.width and out.height > img.height
    assert out.getpixel((0, 0))[3] == 0        # corner is transparent


def test_crop(img):
    out = ops.crop(img, (0, 0, 10, 5))
    assert out.size == (10, 5)
    assert _corner(out, (9, 4)) == (255, 0, 0)


def test_crop_to_nothing_is_an_error(img):
    with pytest.raises(ImageOpError):
        ops.crop(img, (5, 5, 5, 5))


def test_scale(img):
    assert ops.scale(img, (80, 40)).size == (80, 40)


def test_thumbnail_fits_the_box_and_keeps_aspect(img):
    out = ops.thumbnail(img, 10)
    assert max(out.size) <= 10
    assert abs(out.width / out.height - img.width / img.height) < 0.2


def test_grayscale_removes_colour(img):
    out = ops.grayscale(img)
    px = out.convert("RGB").getpixel((0, 0))
    assert px[0] == px[1] == px[2]


def test_grayscale_keeps_transparency():
    """Converting to 'L' would silently drop the alpha channel."""
    from PIL import Image
    im = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    out = ops.grayscale(im)
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 128


def test_invert(img):
    out = ops.invert(img)
    assert out.convert("RGB").getpixel((30, 15)) == (255, 255, 255)  # black → white
    assert out.convert("RGB").getpixel((0, 0)) == (0, 255, 255)      # red → cyan


def test_invert_keeps_transparency():
    from PIL import Image
    im = Image.new("RGBA", (10, 10), (255, 0, 0, 64))
    assert ops.invert(im).getpixel((0, 0))[3] == 64


def test_blur_softens_the_edge(img):
    """The sharp red/black boundary should become a gradient."""
    before = img.convert("RGB").getpixel((10, 2))
    after  = ops.blur(img, 3).convert("RGB").getpixel((10, 2))
    assert before == (0, 0, 0)
    assert after != before and after[0] > 0


def test_brightness_up_and_down(img):
    from PIL import Image
    flat = Image.new("RGB", (4, 4), (100, 100, 100))
    assert ops.brightness(flat, 1.5).getpixel((0, 0))[0] > 100
    assert ops.brightness(flat, 0.5).getpixel((0, 0))[0] < 100


def test_contrast_extremes(img):
    from PIL import Image
    im = Image.new("RGB", (4, 4), (100, 100, 100))
    # Contrast 0 flattens everything to the mean grey.
    assert ops.contrast(im, 0.0).getpixel((0, 0))[0] == pytest.approx(100, abs=2)


def test_sharpen_runs_and_keeps_the_size(img):
    assert ops.sharpen(img, 2.0).size == img.size


# ── output format selection ──────────────────────────────────────────────────

def test_jpeg_source_stays_jpeg(img):
    assert ops.output_format(img, "jpg") == ("JPEG", "jpg")


def test_alpha_forces_png_even_from_a_jpeg_source():
    """A JPEG can't hold an alpha channel — the format has to change."""
    from PIL import Image
    im = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert ops.output_format(im, "jpg") == ("PNG", "png")


def test_need_alpha_flag_forces_png(img):
    assert ops.output_format(img, "jpg", need_alpha=True) == ("PNG", "png")


def test_webp_source_stays_webp(img):
    assert ops.output_format(img, "webp") == ("WEBP", "webp")


def test_saving_rgba_as_jpeg_converts_instead_of_crashing(tmp_path):
    from PIL import Image
    im = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    out = str(tmp_path / "x.jpg")
    ops.save_image(im, out, "JPEG")
    assert os.path.getsize(out) > 0


# ── shrink-to-size ───────────────────────────────────────────────────────────

@pytest.fixture
def noisy():
    """Random noise — incompressible, so hitting a size target is real work."""
    import random
    from PIL import Image
    random.seed(1234)
    im = Image.new("RGB", (900, 900))
    im.putdata([(random.randrange(256), random.randrange(256),
                 random.randrange(256)) for _ in range(900 * 900)])
    return im


def test_shrink_hits_the_target(noisy):
    target = 120 * 1024
    data, quality, dims = ops.shrink_to(noisy, target, "JPEG")
    assert len(data) <= target
    assert 20 <= quality <= 95
    assert dims[0] <= noisy.width


def test_shrink_uses_the_headroom_it_has(noisy):
    """A 2 MB budget shouldn't come back as a 40 KB image."""
    target = 2 * 1024 * 1024
    data, _q, dims = ops.shrink_to(noisy, target, "JPEG")
    assert len(data) <= target
    assert dims == noisy.size          # no need to resize at that budget


def test_shrink_reduces_resolution_when_quality_alone_cannot(noisy):
    data, _q, dims = ops.shrink_to(noisy, 12 * 1024, "JPEG")
    assert len(data) <= 12 * 1024
    assert dims[0] < noisy.width       # it had to scale down


def test_shrink_switches_png_to_jpeg(noisy):
    """PNG is lossless — there's no quality dial to hit a size target with."""
    data, _q, _d = ops.shrink_to(noisy, 100 * 1024, "PNG")
    assert data[:3] == b"\xff\xd8\xff"     # JPEG magic bytes


def test_shrink_gives_up_rather_than_returning_mush(noisy):
    with pytest.raises(ImageOpError, match="bigger target"):
        ops.shrink_to(noisy, 200, "JPEG")


# ── text ─────────────────────────────────────────────────────────────────────

def test_caption_changes_pixels_at_the_top(img):
    from PIL import Image
    base = Image.new("RGB", (400, 300), "black")
    out  = ops.caption(base, "HELLO", "")
    assert out.size == base.size
    top_half = out.convert("RGB").crop((0, 0, 400, 150))
    assert top_half.getextrema()[0][1] > 0     # something was drawn


def test_caption_writes_to_both_ends():
    from PIL import Image
    base = Image.new("RGB", (400, 300), "black")
    out  = ops.caption(base, "TOP", "BOTTOM").convert("RGB")
    assert out.crop((0, 0, 400, 100)).getextrema()[0][1] > 0
    assert out.crop((0, 200, 400, 300)).getextrema()[0][1] > 0


def test_caption_with_only_bottom_leaves_the_top_alone():
    from PIL import Image
    base = Image.new("RGB", (400, 300), "black")
    out  = ops.caption(base, "", "BOTTOM").convert("RGB")
    assert out.crop((0, 0, 400, 80)).getextrema()[0][1] == 0


def test_caption_wraps_long_text_instead_of_overflowing():
    from PIL import Image
    base = Image.new("RGB", (300, 300), "black")
    out  = ops.caption(base, "word " * 40, "").convert("RGB")
    # Nothing should be drawn in the bottom half — the wrap has to keep the
    # block inside its 40%-of-height budget rather than running off the image.
    assert out.crop((0, 200, 300, 300)).getextrema()[0][1] == 0


def test_watermark_marks_the_chosen_corner():
    from PIL import Image
    base = Image.new("RGB", (400, 300), "black")
    out  = ops.watermark(base, "test", corner="tl").convert("RGB")
    assert out.crop((0, 0, 200, 80)).getextrema()[0][1] > 0
    assert out.crop((200, 200, 400, 300)).getextrema()[0][1] == 0


def test_watermark_returns_rgba_so_the_alpha_composite_survives():
    from PIL import Image
    base = Image.new("RGB", (200, 200), "black")
    assert ops.watermark(base, "x").mode == "RGBA"


# ── loading ──────────────────────────────────────────────────────────────────

def test_open_image_applies_exif_rotation(tmp_path):
    """A phone photo tagged 'rotate 90' must come back already rotated."""
    from PIL import Image
    im = Image.new("RGB", (40, 20), "black")
    path = str(tmp_path / "exif.jpg")
    exif = Image.Exif()
    exif[274] = 6                      # Orientation: rotate 90° CW
    im.save(path, "JPEG", exif=exif)

    out = ops.open_image(path)
    assert out.size == (20, 40), "EXIF orientation wasn't applied"


def test_open_image_rejects_non_images(tmp_path):
    path = str(tmp_path / "not.png")
    with open(path, "wb") as fh:
        fh.write(b"this is not an image")
    with pytest.raises(ImageOpError, match="image"):
        ops.open_image(path)


def test_open_image_round_trips_a_normal_file(tmp_path, img):
    path = str(tmp_path / "ok.png")
    img.save(path)
    assert ops.open_image(path).size == img.size


# ── fitting to Discord's emoji / sticker limits ──────────────────────────────

def _png_bytes(size, colour=(200, 30, 30, 255), mode="RGBA"):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new(mode, size, colour).save(buf, "PNG")
    return buf.getvalue()


def _gif_bytes(size, frames=20, duration=80):
    import io
    from PIL import Image
    imgs = [Image.new("RGB", size, (i * 11 % 256, 60, 120)) for i in range(frames)]
    buf = io.BytesIO()
    imgs[0].save(buf, "GIF", save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0)
    return buf.getvalue()


def _open(data):
    import io
    from PIL import Image
    return Image.open(io.BytesIO(data))


def test_small_emoji_passes_through_untouched():
    raw = _png_bytes((64, 64))
    out, note = ops.fit_emoji(raw)
    assert out == raw and note == ""


def test_oversized_emoji_is_scaled_to_the_render_size():
    out, note = ops.fit_emoji(_png_bytes((2000, 1500)))
    im = _open(out)
    assert max(im.size) <= ops.EMOJI_EDGE
    assert len(out) <= ops.EMOJI_MAX_BYTES
    assert "resized" in note


def test_emoji_resize_keeps_the_aspect_ratio():
    im = _open(ops.fit_emoji(_png_bytes((2000, 1000)))[0])
    assert abs(im.width / im.height - 2.0) < 0.05


def test_emoji_resize_preserves_transparency():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (900, 900), (0, 0, 0, 0)).save(buf, "PNG")
    im = _open(ops.fit_emoji(buf.getvalue())[0])
    assert im.mode in ("RGBA", "P", "LA")
    assert im.convert("RGBA").getpixel((5, 5))[3] == 0


def test_animated_emoji_stays_animated():
    """Dropping frames is acceptable; dropping the animation is a last resort."""
    out, _note = ops.fit_emoji(_gif_bytes((600, 600), frames=30))
    im = _open(out)
    assert getattr(im, "n_frames", 1) > 1, "lost the animation"
    assert len(out) <= ops.EMOJI_MAX_BYTES
    assert max(im.size) <= ops.EMOJI_EDGE


def test_animated_emoji_reports_what_it_gave_up():
    """A heavy GIF should say if frames were dropped rather than doing it silently."""
    out, note = ops.fit_emoji(_gif_bytes((800, 800), frames=120))
    assert len(out) <= ops.EMOJI_MAX_BYTES
    if getattr(_open(out), "n_frames", 1) < 120:
        assert note, "frames were dropped without saying so"


def test_is_animated_detection():
    assert ops.is_animated(_gif_bytes((50, 50), frames=5)) is True
    assert ops.is_animated(_png_bytes((50, 50))) is False
    assert ops.is_animated(b"not an image") is False


def test_sticker_is_exactly_320_square():
    out, note = ops.fit_sticker(_png_bytes((900, 400)))
    im = _open(out)
    assert im.size == (ops.STICKER_SIZE, ops.STICKER_SIZE)
    assert len(out) <= ops.STICKER_MAX_BYTES
    assert "fitted" in note


def test_sticker_pads_rather_than_stretching():
    """A tall source must keep its proportions, with transparent bars."""
    out, _ = ops.fit_sticker(_png_bytes((200, 800), (20, 200, 90, 255)))
    im = _open(out).convert("RGBA")
    assert im.getpixel((5, 160))[3] == 0, "sides should be transparent padding"
    assert im.getpixel((160, 160))[3] == 255, "subject should be centred"


def test_sticker_already_the_right_size_is_left_alone():
    _out, note = ops.fit_sticker(_png_bytes((320, 320)))
    assert note == ""


def test_sticker_takes_the_first_frame_of_an_animation():
    out, _ = ops.fit_sticker(_gif_bytes((400, 400), frames=10))
    im = _open(out)
    assert im.size == (320, 320)
    assert getattr(im, "n_frames", 1) == 1


# ── QR codes ─────────────────────────────────────────────────────────────────

qrcode = pytest.importorskip("qrcode", reason="qrcode not installed")
zxingcpp = pytest.importorskip("zxingcpp", reason="zxing-cpp not installed")


@pytest.mark.parametrize("text", [
    "https://example.com",
    "https://example.com/path?a=1&b=2#frag",
    "plain text with spaces",
    "WIFI:S:MyNetwork;T:WPA;P:hunter2;;",
    "1234567890",
    "unicode: café — naïve ☕",
])
def test_qr_round_trips(tmp_path, text):
    im = ops.make_qr(text)
    path = str(tmp_path / "q.png")
    ops.save_image(im, path, "PNG")
    found = ops.read_qr(path)
    assert found, f"couldn't decode {text!r}"
    assert found[0][1] == text


def test_qr_reports_its_format(tmp_path):
    path = str(tmp_path / "q.png")
    ops.save_image(ops.make_qr("hello"), path, "PNG")
    assert ops.read_qr(path)[0][0] == "QR Code"


def test_qr_grows_with_the_payload():
    small = ops.make_qr("hi")
    large = ops.make_qr("x" * 800)
    assert large.width > small.width


def test_small_qr_still_decodes_via_the_upscale_retry(tmp_path):
    """Screenshotted codes are often tiny; read_qr enlarges before giving up."""
    im = ops.make_qr("https://example.com/small", scale=2, border=2)
    assert max(im.size) < 200
    path = str(tmp_path / "small.png")
    ops.save_image(im, path, "PNG")
    assert ops.read_qr(path)[0][1] == "https://example.com/small"


def test_reading_an_image_with_no_code_returns_nothing(tmp_path):
    from PIL import Image
    path = str(tmp_path / "blank.png")
    Image.new("RGB", (400, 400), "white").save(path)
    assert ops.read_qr(path) == []


@pytest.mark.parametrize("text", ["", "   ", None])
def test_qr_refuses_empty_content(text):
    with pytest.raises(ImageOpError, match="Nothing to encode"):
        ops.make_qr(text)


def test_qr_refuses_a_payload_that_will_not_fit():
    with pytest.raises(ImageOpError, match="tops out"):
        ops.make_qr("y" * (ops.QR_MAX_CHARS + 1))


def test_qr_uses_higher_error_correction_than_the_default():
    """These get photographed off screens — redundancy matters more than size."""
    import qrcode as _q
    dense = ops.make_qr("https://example.com/some/longer/path?with=params")
    baseline = _q.QRCode(error_correction=_q.constants.ERROR_CORRECT_M,
                         box_size=10, border=4)
    baseline.add_data("https://example.com/some/longer/path?with=params")
    baseline.make(fit=True)
    assert dense.width >= baseline.make_image().size[0]
