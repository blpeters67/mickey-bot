"""Interleavings missed by the initial Voice DJ implementation."""
import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

import cogs.music as music
import cogs.voicedj as vd
from core.state import clear_queue, get_state, reset_state
from core.voice_frontend import VoiceFrontEnd, VAD_WINDOW_S
from core.voice_intent import Intent
from core.voice_pipeline import Job
from tests._playback_rig import make_track
from tests.test_playback_engine import Engine, ok
from tests.test_voice_followup import _Dec, _Vad, gated_lookup
from tests.test_voicedj_flow import Rig, GID, UID, OTHER
from tests.test_voice_smart import posted, smart
from core.voice_llm import LLMReply, parse_reply


def test_failed_old_resolve_cannot_consume_a_new_request(monkeypatch):
    async def go():
        e = Engine(monkeypatch, 990123)
        entered, release = asyncio.Event(), asyncio.Event()
        a, b = make_track('Old request'), make_track('New request')
        e.script[b['stream_url']] = ok(5)

        async def resolve(track, **kw):
            if track is a:
                entered.set()
                await release.wait()
                raise ValueError('old lookup failed')
            return track

        monkeypatch.setattr(music.audio, 'resolve_stream_url', resolve)
        e.state.queue.append(a)
        old = asyncio.create_task(e.cog._play_next(e.guild, e.vc, e.state.playback_gen))
        await entered.wait()
        reset_state(e.guild.id)
        e.state.queue.append(b)
        new = asyncio.create_task(e.cog._play_next(e.guild, e.vc, e.state.playback_gen))
        release.set()
        await asyncio.gather(old, new)
        await e.idle()
        assert e.finished == ['New request']
        assert not e.state.failures and e.vc.play_calls == 1
    asyncio.run(go())


@pytest.mark.parametrize('invalidate', [reset_state, clear_queue])
def test_stop_or_clear_during_autoplay_lookup_does_not_repopulate_queue(monkeypatch, invalidate):
    async def go():
        e = Engine(monkeypatch, 990124)
        entered, release = asyncio.Event(), asyncio.Event()
        async def settings(gid): return {'autoplay': 1}
        async def fetch(*args, **kw):
            entered.set()
            await release.wait()
            return make_track('Related song')
        monkeypatch.setattr(music.db, 'get_guild_settings', settings)
        monkeypatch.setattr(music.audio, 'fetch_autoplay_track', fetch)
        task = asyncio.create_task(e.cog._try_autoplay(e.guild, e.state, make_track('Seed')))
        await entered.wait()
        invalidate(e.guild.id)
        release.set()
        assert await task is False
        assert not e.state.queue
    asyncio.run(go())


@pytest.mark.parametrize('kind', ['armed', 'confirmation'])
def test_context_survives_while_timely_speech_is_still_being_captured(monkeypatch, kind):
    async def go():
        rig = Rig(monkeypatch)
        try:
            now = time.monotonic()
            front = VoiceFrontEnd(lambda s: None, decoder_factory=_Dec, vad_factory=_Vad)
            rig.cog._front = front
            stream = front._stream_for((GID, UID), tag=(0, 0))
            stream.segmenter.push(np.ones(512, dtype=np.float32), 1.0, now - 3 + VAD_WINDOW_S)
            if kind == 'armed':
                context = vd._Armed(now - 6, now - 2, 0)
                contexts = rig.cog._armed
                text, action = 'pause', 'pause'
            else:
                context = vd._Pending(Intent('stop'), now - 15, now - 2, 0)
                contexts = rig.cog._pending
                text, action = 'yes', 'stop'
            contexts[(GID, UID)] = context
            rig.cog._last_refresh = now
            await rig.cog._maintenance.coro(rig.cog)
            assert contexts.get((GID, UID)) is context
            job = Job(GID, UID, None, now - 3.32, now, 3000, session=0,
                      epoch=0, speech_start=now - 3)
            await rig.cog._handle_transcript(job, text)
            await rig.settle()
            assert [e[2] for e in rig.executed] == [action]
        finally:
            await rig.close()
    asyncio.run(go())


def test_processing_grace_is_enforced_even_between_maintenance_ticks(monkeypatch):
    async def go():
        rig = Rig(monkeypatch)
        try:
            now = time.monotonic()
            p = vd._Pending(Intent('stop'), now - 50, now - 30, 0)
            rig.cog._pending[(GID, UID)] = p
            job = Job(GID, UID, None, now - 40, now - 39, 500, session=0, epoch=0)
            assert p.accepts(job)
            await rig.cog._handle_transcript(job, 'yes')
            await rig.settle()
            assert not rig.executed
        finally:
            await rig.close()
    asyncio.run(go())


def test_typed_clear_invalidates_running_and_waiting_voice_lookups(monkeypatch):
    async def go():
        rig = Rig(monkeypatch)
        look = gated_lookup(monkeypatch)
        try:
            rig.say('hey dj play first song')
            await asyncio.wait_for(look.started.wait(), 3)
            rig.say('hey dj play second song')
            for _ in range(100):
                if len(rig.cog._plays.get(GID, ())) == 2:
                    break
                await asyncio.sleep(.01)
            assert len(rig.cog._plays[GID]) == 2
            guild = rig.guilds[GID]
            current = {'title': 'Already playing'}
            get_state(GID).current = current
            sent = []
            async def send(text): sent.append(text)
            ctx = SimpleNamespace(guild=guild, voice_client=guild.voice_client,
                                  author=guild.get_member(UID), send=send)
            await music.MusicCog.clear.callback(music.MusicCog(rig.bot), ctx)
            look.release.set()
            await rig.settle()
            assert get_state(GID).current is current
            assert not get_state(GID).queue
            assert look.calls == ['first song']
            assert not any('✅ Queued' in (f[2] or '') for f in rig.feedback)
        finally:
            look.release.set()
            await rig.close()
    asyncio.run(go())


def test_voice_clear_invalidates_typed_admission(monkeypatch):
    async def go():
        rig = Rig(monkeypatch)
        try:
            guild = rig.guilds[GID]
            epoch = get_state(GID).queue_epoch
            await rig.cog._execute_intent(guild, guild.get_member(UID), guild.voice_client, Intent('clear'))
            sent = []
            async def send(text): sent.append(text)
            ctx = SimpleNamespace(guild=guild, send=send)
            assert await music.MusicCog(rig.bot)._admit_after_lookup(ctx, epoch, 'Late song') is None
            assert 'queue was cleared' in sent[0]
        finally:
            await rig.close()
    asyncio.run(go())


@smart
async def test_typing_cannot_resurrect_after_reply_during_channel_lookup(rig):
    entered, release = asyncio.Event(), asyncio.Event()
    async def echo(guild):
        entered.set()
        await release.wait()
        return rig.channel_txt
    rig.cog._echo_channel = echo
    guild = rig.guilds[GID]
    work = asyncio.create_task(rig.cog._typing_start(guild, guild.get_member(UID), reason='heard'))
    await entered.wait()
    rig.cog._typing_stop((GID, UID))
    release.set()
    await work
    assert not rig.cog._typing and not rig.channel_txt.typing_calls


@smart
async def test_typing_rechecks_consent_after_channel_lookup(rig):
    entered, release = asyncio.Event(), asyncio.Event()
    async def echo(guild):
        entered.set()
        await release.wait()
        return rig.channel_txt
    rig.cog._echo_channel = echo
    guild = rig.guilds[GID]
    work = asyncio.create_task(rig.cog._typing_start(guild, guild.get_member(UID), reason='heard'))
    await entered.wait()
    rig.cog._auth[GID] = frozenset()
    release.set()
    await work
    assert not rig.cog._typing and not rig.channel_txt.typing_calls


@smart
async def test_slow_ai_answer_does_not_block_another_speakers_skip(rig):
    entered, release = asyncio.Event(), asyncio.Event()
    class SlowLLM:
        async def ask(self, *args, **kw):
            entered.set()
            await release.wait()
            return LLMReply('question', reply='Paris.')

        async def plan(self, said, context, **kw):
            from core import voice_plan as vp
            return vp.from_reply(await self.ask(said))
    rig.cog.llm = SlowLLM()
    rig.say('hey dj what is the capital of france')
    await asyncio.wait_for(entered.wait(), 3)
    rig.say('hey dj skip', uid=OTHER)
    try:
        async def skipped():
            while not any(e[1] == OTHER and e[2] == 'skip' for e in rig.executed):
                await asyncio.sleep(.01)
        await asyncio.wait_for(skipped(), 1.5)
        assert not release.is_set()
    finally:
        release.set()
    await rig.settle()
    assert rig.tts.said == ['Paris.']


@smart
async def test_disabling_ai_while_it_is_thinking_prevents_a_late_answer(rig):
    entered, release = asyncio.Event(), asyncio.Event()
    class SlowLLM:
        async def ask(self, *args, **kw):
            entered.set()
            await release.wait()
            return LLMReply('question', reply='Paris.')

        async def plan(self, said, context, **kw):
            from core import voice_plan as vp
            return vp.from_reply(await self.ask(said))
    rig.cog.llm = SlowLLM()
    rig.say('hey dj what is the capital of france')
    await asyncio.wait_for(entered.wait(), 3)
    rig.cog._cfg[GID]['ai'] = False
    release.set()
    await rig.settle()
    assert not rig.tts.said and not posted(rig)


@pytest.mark.parametrize('content', [42, ['bad'], '{"kind":"commands","actions":42}'])
def test_malformed_ai_response_is_rejected_without_raising(content):
    assert parse_reply(content) is None
