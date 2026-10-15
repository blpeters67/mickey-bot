"""
The last stretch of the wake cue, measured with discord.py's real audio
player thread: from the moment the cog places the cue (TTSCog.cue) to the
first audible frame of it handed to the voice transport — with nothing
playing, over music, over a line of speech being spoken, and over paused
music.

    venv/bin/python scripts/cue_mixer_timing.py [--n 200]

Only the transport is fake (tests/_playback_rig.py: frames are counted, not
Opus-encoded and sent — encoding a frame takes well under a millisecond).
After this the frame still has to cross Discord and the listener's jitter
buffer, which nothing here measures.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DISCORD_TOKEN", "timing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="cues per condition")
    a = ap.parse_args()

    import discord
    from discord.player import AudioPlayer

    import cogs.tts as tts_mod
    from core.config import settings
    from core.voice_mix import CueTiming, DuckingSource, SpeechClip, SpeechSource
    from tests._playback_rig import PlayerVC, TONE

    class Music(discord.AudioSource):
        def read(self):
            return TONE

        def is_opus(self):
            return False

    async def go():
        loop = asyncio.get_running_loop()
        out = {}
        for cond in ("idle", "music", "speech", "paused"):
            got = []
            for _ in range(a.n):
                vc = PlayerVC(loop, types.SimpleNamespace(id=1), player_cls=AudioPlayer)
                guild = types.SimpleNamespace(id=7, voice_client=vc)
                bot = types.SimpleNamespace(get_guild=lambda gid: guild, loop=loop,
                                            get_cog=lambda n: None)
                cog = tts_mod.TTSCog(bot)
                cog._bot_in_voice = lambda g: True
                if cond == "music":
                    vc.play(DuckingSource(Music()))
                elif cond == "speech":
                    vc.play(SpeechSource(SpeechClip(TONE * 500, lead_in=0.0)))
                elif cond == "paused":
                    vc.play(DuckingSource(Music()))
                    await asyncio.sleep(0.05)
                    vc.pause()
                if cond != "idle":
                    await asyncio.sleep(0.05 + random.random() * 0.02)   # any phase of a frame
                timing = CueTiming(time.monotonic())
                assert cog.cue(guild, settings.voice_cue_file, timing=timing)
                deadline = time.monotonic() + 1.0
                while timing.first_frame is None and time.monotonic() < deadline:
                    await asyncio.sleep(0.002)
                if timing.first_frame is not None:
                    got.append(timing.ms("scheduled", "first_frame"))
                vc.stop()
                for t in list(cog._fx_tasks):
                    t.cancel()
            got.sort()
            out[cond] = {"n": len(got), "median_ms": round(statistics.median(got), 1),
                         "p95_ms": round(got[int(0.95 * (len(got) - 1))], 1),
                         "max_ms": round(got[-1], 1)}
            print(cond, out[cond], flush=True)
        return out
    asyncio.run(go())


if __name__ == "__main__":
    main()
