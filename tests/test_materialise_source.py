"""Tests for the source-fetching step shared by -convert and -transcode.

This logic used to live inline in `_do_convert`; it was lifted out so
`-transcode` could reuse it. These tests pin the behaviour `-convert` had
before the move: attachments save straight to the workdir, direct file links
stream to disk, HTML pages fall through to yt-dlp, and every failure path
leaves the user with an explanation rather than a traceback.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from cogs.downloads import DownloadsCog, JobCanceller


@pytest.fixture
def cog():
    return DownloadsCog(MagicMock())


@pytest.fixture
def status():
    s = MagicMock()
    s.edit = AsyncMock()
    return s


def _attachment(filename: str, *, saves=True):
    att = MagicMock(spec=discord.Attachment)
    att.filename = filename
    if saves:
        async def _save(path):
            with open(path, "wb") as fh:
                fh.write(b"data")
        att.save = AsyncMock(side_effect=_save)
    else:
        att.save = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(status=404), "gone"))
    return att


@pytest.mark.asyncio
async def test_attachment_lands_in_the_workdir(cog, status, tmp_path):
    got = await cog._materialise_source(
        attachment=_attachment("clip.mkv"), url=None, workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="convert")

    assert got is not None
    path, name = got
    assert name == "clip.mkv"
    assert os.path.dirname(path) == str(tmp_path)
    assert os.path.exists(path)
    status.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_attachment_without_a_filename_still_gets_one(cog, status, tmp_path):
    got = await cog._materialise_source(
        attachment=_attachment(""), url=None, workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="convert")
    assert got is not None and got[1] == "file"


@pytest.mark.asyncio
async def test_attachment_filename_cannot_escape_the_workdir(cog, status, tmp_path):
    """A crafted filename must not write outside the temp directory."""
    got = await cog._materialise_source(
        attachment=_attachment("../../etc/passwd"), url=None,
        workdir=str(tmp_path), status=status,
        canceller=JobCanceller(), label="convert")

    assert got is not None
    path, name = got
    assert name == "passwd"
    assert os.path.realpath(path).startswith(os.path.realpath(str(tmp_path)))


@pytest.mark.asyncio
async def test_failed_attachment_download_explains_itself(cog, status, tmp_path):
    got = await cog._materialise_source(
        attachment=_attachment("clip.mkv", saves=False), url=None,
        workdir=str(tmp_path), status=status,
        canceller=JobCanceller(), label="convert")

    assert got is None
    msg = status.edit.await_args.kwargs["content"]
    assert msg.startswith("❌") and "attachment" in msg


@pytest.mark.asyncio
async def test_direct_url_returns_the_downloaded_path(cog, status, tmp_path):
    src = tmp_path / "remote.mp4"
    src.write_bytes(b"data")
    cog._download_direct = AsyncMock(return_value=str(src))

    got = await cog._materialise_source(
        attachment=None, url="https://x/remote.mp4", workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="convert")

    assert got == (str(src), "remote.mp4")


@pytest.mark.asyncio
async def test_oversized_url_reports_the_limit(cog, status, tmp_path):
    cog._download_direct = AsyncMock(side_effect=ValueError("File is 5 GB — too big"))

    got = await cog._materialise_source(
        attachment=None, url="https://x/huge.mp4", workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="convert")

    assert got is None
    assert "too big" in status.edit.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_network_failure_is_caught(cog, status, tmp_path):
    cog._download_direct = AsyncMock(side_effect=OSError("connection reset"))

    got = await cog._materialise_source(
        attachment=None, url="https://x/a.mp4", workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="transcode")

    assert got is None
    assert "Download failed" in status.edit.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_html_page_falls_through_to_ytdlp(cog, status, tmp_path):
    """A YouTube link isn't a file — _download_direct returns None for it."""
    fetched = tmp_path / "Some Video.webm"
    fetched.write_bytes(b"data")
    cog._download_direct = AsyncMock(return_value=None)
    cog._fetch_via_ytdlp = AsyncMock(
        return_value=(str(fetched), "Some Video", 12.0, True, 720))

    got = await cog._materialise_source(
        attachment=None, url="https://youtu.be/abc", workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="transcode")

    assert got == (str(fetched), "Some Video.webm")
    # raw_media keeps yt-dlp from applying an mp3/mp4 postprocessor that
    # breaks on sources without the stream it assumes.
    assert cog._fetch_via_ytdlp.await_args.kwargs["raw_media"] is True


@pytest.mark.asyncio
async def test_ytdlp_failure_stays_quiet(cog, status, tmp_path):
    """_fetch_via_ytdlp already wrote the error — don't overwrite it."""
    cog._download_direct = AsyncMock(return_value=None)
    cog._fetch_via_ytdlp = AsyncMock(return_value=None)

    got = await cog._materialise_source(
        attachment=None, url="https://youtu.be/private", workdir=str(tmp_path),
        status=status, canceller=JobCanceller(), label="convert")

    assert got is None
    status.edit.assert_not_awaited()
