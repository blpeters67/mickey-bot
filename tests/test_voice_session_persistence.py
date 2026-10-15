"""Voice control must survive the bot going down.

The report: "every time the bot goes down, it forgets that I have voice
enabled in the server and that I am already opted in for using voice
controls." Two separate things can make it *look* that way, and both are pinned
here:

  * the settings (guild enabled + who opted in) live in the database and have
    to be re-read into memory at startup — a fresh process starts with nothing;
  * the bot's *voice connection* cannot survive `os.execv`/a crash, and
    autojoin only fires when somebody *enters* voice, so the people who were
    already sitting in the channel got no bot back — which reads as "it forgot
    voice control is on".
"""
from __future__ import annotations

import asyncio
import functools
import types

import aiosqlite
import pytest

import cogs.voicedj as vd
from core import db

GID = 990001          # a guild id no other test uses
UID = 42


def run(coro):
    return asyncio.run(coro)


def asynctest(fn):
    """House style (see test_db_features.py): no pytest-asyncio dependency."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


@pytest.fixture(autouse=True)
def _clean_db():
    async def _reset():
        await db.init_db()
        async with aiosqlite.connect(db.settings.db_path) as conn:
            await conn.execute("DELETE FROM voice_settings WHERE guild_id = ?", (GID,))
            await conn.execute("DELETE FROM voice_users WHERE guild_id = ?", (GID,))
            await conn.execute("DELETE FROM app_settings WHERE key LIKE ?",
                               (f"{vd._VOICE_CHANNEL_KEY}%",))
            await conn.commit()
    run(_reset())


# ── fakes ────────────────────────────────────────────────────────────────────

class _Perms:
    def __init__(self, connect=True, speak=True):
        self.connect, self.speak = connect, speak


class _Channel:
    """Just enough of a voice channel for the rejoin decision."""

    def __init__(self, cid, *, members=(), name="big-room", perms=None):
        self.id, self.name, self.members = cid, name, list(members)
        self.guild = None
        self._perms = perms or _Perms()

    def permissions_for(self, member):
        return self._perms


class _Guild:
    def __init__(self, gid=GID, *, voice_channels=(), afk=None, voice_client=None):
        self.id, self.name = gid, "Test Server"
        self.me = types.SimpleNamespace(id=1)
        self.afk_channel = afk
        self.voice_client = voice_client
        self.voice_channels = list(voice_channels)
        self._channels = {c.id: c for c in voice_channels}
        if afk is not None:
            self._channels[afk.id] = afk
        for c in [*voice_channels, *([afk] if afk else [])]:
            c.guild = self

    def get_channel(self, cid):
        return self._channels.get(cid)


class _Music:
    """Stands in for the Music cog: records the joins it is asked for."""

    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    async def autojoin(self, member, channel):
        self.calls.append((member.id, channel.id))
        return self.ok


def human(uid=UID, name="Alex"):
    return types.SimpleNamespace(id=uid, bot=False, display_name=name)


def _bot(guilds=(), music=None):
    cogs = {"Music": music} if music is not None else {}
    gmap = {g.id: g for g in guilds}
    return types.SimpleNamespace(
        user=types.SimpleNamespace(id=1),
        get_guild=gmap.get,
        get_cog=lambda name: cogs.get(name),
        loop=None,
    )


def _cog(bot):
    """A cog in a brand-new process: no memory of anything yet."""
    cog = vd.VoiceDJ(bot)
    cog._whisper_ok = True
    return cog


def channel_key(cid):
    return f"{vd._VOICE_CHANNEL_KEY}{cid}"


# ── the settings come back ───────────────────────────────────────────────────

@asynctest
async def test_enabled_guild_and_optins_are_restored_at_startup():
    await db.set_voice_enabled(GID, True)
    await db.set_voice_optin(GID, UID, True)

    cog = _cog(_bot())                       # nothing in memory
    assert cog._auth == {} and cog._cfg == {}

    await cog._refresh_all()

    assert cog._cfg[GID]["enabled"] is True
    assert cog._auth[GID] == frozenset({UID})


@asynctest
async def test_opt_out_and_disabled_guild_stay_that_way_across_a_restart():
    await db.set_voice_enabled(GID, True)
    await db.set_voice_optin(GID, UID, False)

    cog = _cog(_bot())
    await cog._refresh_all()
    assert cog._auth[GID] == frozenset()     # opted out, so not listened to

    await db.set_voice_enabled(GID, False)
    await db.set_voice_optin(GID, UID, True)
    restarted = _cog(_bot())
    await restarted._refresh_all()
    assert not restarted._auth.get(GID)                   # nobody gets listened to
    assert not restarted._cfg.get(GID, {}).get("enabled")  # and it stays off


@asynctest
async def test_wake_words_and_reply_channel_are_restored_too():
    await db.set_voice_enabled(GID, True)
    await db.set_voice_wakes(GID, ["hey dj", "yo dj"])
    await db.set_voice_feedback_channel(GID, 4242)

    cog = _cog(_bot())
    await cog._refresh_all()
    assert cog._cfg[GID]["wake"] == ["hey dj", "yo dj"]
    assert cog._cfg[GID]["channel_id"] == 4242


# ── the bot remembers where it was ───────────────────────────────────────────

@asynctest
async def test_voice_channel_is_remembered_and_forgotten_on_leave():
    cog = _cog(_bot())
    await cog._remember_voice_channel(GID, types.SimpleNamespace(id=777))
    assert await db.get_app_setting(channel_key(GID)) == "777"

    await cog._remember_voice_channel(GID, None)       # the bot left
    assert not await db.get_app_setting(channel_key(GID))


@asynctest
async def test_the_bot_own_voice_state_change_writes_the_memory():
    chan = _Channel(777)
    guild = _Guild(voice_channels=[chan])
    cog = _cog(_bot([guild]))
    bot_member = types.SimpleNamespace(id=1, guild=guild, bot=True)

    await cog.on_voice_state_update(bot_member,
                                    types.SimpleNamespace(channel=None),
                                    types.SimpleNamespace(channel=chan))
    await asyncio.gather(*cog._bg)          # the write is spawned, not awaited
    assert await db.get_app_setting(channel_key(GID)) == "777"

    await cog.on_voice_state_update(bot_member,
                                    types.SimpleNamespace(channel=chan),
                                    types.SimpleNamespace(channel=None))
    await asyncio.gather(*cog._bg)
    assert not await db.get_app_setting(channel_key(GID))


# ── the bot comes back to the channel it was in ──────────────────────────────

@asynctest
async def test_rejoin_puts_the_bot_back_while_somebody_is_still_there():
    await db.set_voice_enabled(GID, True)
    chan = _Channel(777, members=[human()])
    guild = _Guild(voice_channels=[chan])
    music = _Music()
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()
    await cog._remember_voice_channel(GID, chan)

    await cog._rejoin_voice()
    assert music.calls == [(UID, 777)]


@asynctest
async def test_rejoin_does_nothing_when_the_room_is_empty():
    await db.set_voice_enabled(GID, True)
    chan = _Channel(777, members=[])
    guild = _Guild(voice_channels=[chan])
    music = _Music()
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()
    await cog._remember_voice_channel(GID, chan)

    await cog._rejoin_voice()
    assert music.calls == []          # nobody to come back for


@asynctest
async def test_rejoin_does_nothing_when_voice_control_is_off_there():
    chan = _Channel(777, members=[human()])
    guild = _Guild(voice_channels=[chan])
    music = _Music()
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()          # guild never enabled voice
    await cog._remember_voice_channel(GID, chan)

    await cog._rejoin_voice()
    assert music.calls == []


@asynctest
async def test_rejoin_does_nothing_when_the_bot_is_already_connected():
    await db.set_voice_enabled(GID, True)
    chan = _Channel(777, members=[human()])
    guild = _Guild(voice_channels=[chan], voice_client=types.SimpleNamespace())
    music = _Music()
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()
    await cog._remember_voice_channel(GID, chan)

    await cog._rejoin_voice()
    assert music.calls == []


@asynctest
async def test_rejoin_skips_a_channel_we_cannot_speak_in_or_that_is_gone():
    await db.set_voice_enabled(GID, True)
    quiet = _Channel(777, members=[human()], perms=_Perms(speak=False))
    gone = _Channel(888, members=[human()])
    guild = _Guild(voice_channels=[quiet, gone])
    music = _Music()
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()

    await cog._remember_voice_channel(GID, quiet)
    await cog._rejoin_voice()
    assert music.calls == []          # no speak permission → don't barge in

    guild.voice_channels = []         # channel was deleted meanwhile
    await cog._remember_voice_channel(GID, gone)
    await cog._rejoin_voice()
    assert music.calls == []


@asynctest
async def test_rejoin_never_joins_the_afk_channel():
    await db.set_voice_enabled(GID, True)
    afk = _Channel(777, members=[human()], name="AFK")
    guild = _Guild(voice_channels=[afk], afk=afk)
    music = _Music()
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()
    await cog._remember_voice_channel(GID, afk)

    await cog._rejoin_voice()
    assert music.calls == []


@asynctest
async def test_rejoin_gives_up_quietly_when_the_join_fails():
    await db.set_voice_enabled(GID, True)
    chan = _Channel(777, members=[human()])
    guild = _Guild(voice_channels=[chan])
    music = _Music(ok=False)               # Discord refused the handshake
    cog = _cog(_bot([guild], music))
    await cog._refresh_all()
    await cog._remember_voice_channel(GID, chan)

    await cog._rejoin_voice()              # must not raise
    assert music.calls == [(UID, 777)]
    assert guild.voice_client is None      # nothing was claimed as connected


@asynctest
async def test_rejoin_is_survivable_without_a_music_cog():
    await db.set_voice_enabled(GID, True)
    chan = _Channel(777, members=[human()])
    cog = _cog(_bot([_Guild(voice_channels=[chan])], music=None))
    await cog._refresh_all()
    await cog._remember_voice_channel(GID, chan)
    await cog._rejoin_voice()              # no Music cog loaded: just a no-op
