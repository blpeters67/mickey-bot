"""Autojoin: join someone who enters voice when nobody else in the server is
in voice (cogs/admin.py listener + cogs/music.py connect). Mocked Discord
objects; the settings go through the real database layer (test DB)."""
from __future__ import annotations

import asyncio
import types

import discord
import pytest

import cogs.admin as admin
import cogs.music as music
from core import db

GID = 4242


class Perms:
    def __init__(self, **kw):
        self.view_channel = kw.get("view_channel", True)
        self.connect = kw.get("connect", True)
        self.speak = kw.get("speak", True)
        self.move_members = kw.get("move_members", False)
        self.send_messages = kw.get("send_messages", True)


class _FakeChannel:
    """Real VoiceChannel/StageChannel subclasses (so isinstance checks are
    honest) without discord.py's HTTP-backed state."""

    def __init__(self, cid, name, members=(), *, perms=None, user_limit=0):
        self.id = cid
        self.name = name
        self.user_limit = user_limit
        self.guild = None
        self._fake_members = list(members)
        self._perms = perms or Perms()

    @property
    def members(self):
        return self._fake_members

    def permissions_for(self, _obj):
        return self._perms

    def __repr__(self):
        return f"<{self.name}>"


class FakeVoice(_FakeChannel, discord.VoiceChannel):
    pass


class FakeStage(_FakeChannel, discord.StageChannel):
    pass


def voice_channel(cid, name, members=(), *, perms=None, user_limit=0, stage=False):
    return (FakeStage if stage else FakeVoice)(cid, name, members, perms=perms,
                                               user_limit=user_limit)


def person(uid, *, bot=False):
    return types.SimpleNamespace(id=uid, bot=bot, display_name=f"user{uid}", voice=None)


class World:
    """A guild with a few voice channels, the admin cog, and a fake music cog."""

    def __init__(self, *, setting=1):
        self.lounge = voice_channel(1, "Lounge")
        self.gaming = voice_channel(2, "Gaming")
        self.afk = voice_channel(3, "AFK")
        self.stage = voice_channel(4, "Stage", stage=True)
        self.guild = types.SimpleNamespace(
            id=GID, name="g", voice_client=None, afk_channel=self.afk,
            voice_channels=[self.lounge, self.gaming, self.afk], stage_channels=[self.stage],
            me=person(1, bot=True), system_channel=None)
        for ch in (self.lounge, self.gaming, self.afk, self.stage):
            ch.guild = self.guild
        self.joined = []
        world = self

        class FakeMusic:
            async def autojoin(self, member, channel):
                world.joined.append((member.id, channel.name))
                world.guild.voice_client = types.SimpleNamespace(channel=channel)
                return True
        self.bot = types.SimpleNamespace(user=types.SimpleNamespace(id=1),
                                         get_cog=lambda name: FakeMusic() if name == "Music" else None)
        self.cog = admin.AdminCog.__new__(admin.AdminCog)
        self.cog.bot = self.bot
        self.cog._last_active = {}
        self.cog._autojoin_locks = {}
        self.cog._tasks = set()
        self.setting = setting

    def enter(self, member, channel, *, before=None):
        """`member` moves from `before` to `channel`; returns the listener coroutine."""
        member.guild = self.guild
        if before is not None and member in before._fake_members:
            before._fake_members.remove(member)
        if channel is not None:
            channel._fake_members.append(member)
        member.voice = types.SimpleNamespace(channel=channel) if channel else None
        b = types.SimpleNamespace(channel=before)
        a = types.SimpleNamespace(channel=channel)
        return self.cog.on_voice_state_update(member, b, a)

    async def settle(self):
        for _ in range(5):
            await asyncio.sleep(0)
        if self.cog._tasks:
            await asyncio.gather(*self.cog._tasks)


@pytest.fixture
def world(monkeypatch):
    w = World()

    async def settings(gid):
        return {"idle_seconds": None, "leave_when_alone": None, "autoplay": None,
                "sponsorblock": None, "autojoin": w.setting}
    monkeypatch.setattr(admin.db, "get_guild_settings", settings)
    return w


def run(coro):
    return asyncio.run(coro)


def test_joins_the_first_person_into_an_empty_server(world):
    async def go():
        await world.enter(person(10), world.lounge)
        await world.settle()
    run(go())
    assert world.joined == [(10, "Lounge")]


@pytest.mark.parametrize("setting", [None, 0])
def test_off_by_default_and_when_disabled(world, setting):
    world.setting = setting

    async def go():
        await world.enter(person(10), world.lounge)
        await world.settle()
    run(go())
    assert world.joined == []


def test_not_when_someone_is_already_in_another_channel(world):
    world.gaming._fake_members.append(person(20))

    async def go():
        await world.enter(person(10), world.lounge)
        await world.settle()
    run(go())
    assert world.joined == []


def test_afk_idlers_and_bots_dont_count_as_people(world):
    world.afk._fake_members.append(person(20))
    world.gaming._fake_members.append(person(30, bot=True))

    async def go():
        await world.enter(person(10), world.lounge)
        await world.settle()
    run(go())
    assert world.joined == [(10, "Lounge")]


@pytest.mark.parametrize("case", ["bot", "afk", "stage", "move", "already_connected",
                                  "no_speak", "full"])
def test_cases_that_must_not_autojoin(world, case):
    member = person(10, bot=(case == "bot"))
    target, before = world.lounge, None
    if case == "afk":
        target = world.afk
    elif case == "stage":
        target = world.stage
    elif case == "move":
        before = world.gaming
        world.gaming._fake_members.append(member)
    elif case == "already_connected":
        world.guild.voice_client = types.SimpleNamespace(channel=world.gaming)
    elif case == "no_speak":
        world.lounge._perms = Perms(speak=False)
    elif case == "full":
        world.lounge.user_limit = 1

    async def go():
        await world.enter(member, target, before=before)
        await world.settle()
    run(go())
    assert world.joined == []


def test_coming_back_from_afk_counts_as_joining(world):
    member = person(10)
    world.afk._fake_members.append(member)

    async def go():
        await world.enter(member, world.lounge, before=world.afk)
        await world.settle()
    run(go())
    assert world.joined == [(10, "Lounge")]


def test_two_people_joining_at_once_join_once(world):
    async def go():
        a, b = person(10), person(11)
        await world.enter(a, world.lounge)
        await world.enter(b, world.gaming)          # same instant, other room
        await world.settle()
    run(go())
    assert len(world.joined) == 1


def test_member_who_left_before_settings_loaded_is_not_followed(world, monkeypatch):
    gate = asyncio.Event

    async def go():
        ev = gate()

        async def slow_settings(gid):
            await ev.wait()
            return {"autojoin": 1}
        monkeypatch.setattr(admin.db, "get_guild_settings", slow_settings)
        m = person(10)
        await world.enter(m, world.lounge)
        await asyncio.sleep(0)
        await world.enter(m, None, before=world.lounge)    # gone again
        ev.set()
        await world.settle()
    run(go())
    assert world.joined == []


# ── the toggle, through the real settings table ──────────────────────────────

class Ctx:
    def __init__(self, *, manage=True):
        self.guild = types.SimpleNamespace(id=GID + 1)
        self.prefix = "-"
        self.author = types.SimpleNamespace(guild_permissions=types.SimpleNamespace(manage_guild=manage))
        self.sent = []

    async def send(self, content=None, **kw):
        self.sent.append(content)


def test_toggle_command_persists_and_requires_manage_server():
    async def go():
        await db.init_db()
        cog = admin.AdminCog.__new__(admin.AdminCog)
        ctx = Ctx()
        await admin.AdminCog.autojoin.callback(cog, ctx)
        assert "**off**" in ctx.sent[-1]
        await admin.AdminCog.autojoin.callback(cog, ctx, "on")
        assert (await db.get_guild_settings(GID + 1))["autojoin"] == 1
        await admin.AdminCog.autojoin.callback(cog, ctx)
        assert "**on**" in ctx.sent[-1]
        await admin.AdminCog.autojoin.callback(cog, ctx, "off")
        assert (await db.get_guild_settings(GID + 1))["autojoin"] == 0
        with pytest.raises(discord.ext.commands.MissingPermissions):
            await admin.AdminCog.autojoin.callback(cog, Ctx(manage=False), "on")
        await admin.AdminCog.autojoin.callback(cog, ctx, "maybe")
        assert ctx.sent[-1].startswith("❌")
    run(go())


# ── the connect itself (cogs/music.py) ───────────────────────────────────────

def test_music_autojoin_connects_and_starts_the_recorder_quietly():
    async def go():
        attached = []

        class Rec:
            async def attach(self, vc, channel, **kw):
                attached.append((vc, channel, kw))
        text = types.SimpleNamespace(id=55, permissions_for=lambda me: Perms())
        vchat = voice_channel(56, "vc-chat")                 # a voice channel's own chat
        guild = types.SimpleNamespace(id=GID + 2, name="g", voice_client=None,
                                      me=person(1, bot=True), system_channel=text)
        chan = voice_channel(57, "Lounge")
        chan.guild = guild
        connected = []

        async def connect(*, cls):
            connected.append(cls)
            vc = types.SimpleNamespace(channel=chan)
            guild.voice_client = vc
            return vc
        chan.connect = connect
        bot = types.SimpleNamespace(get_cog=lambda n: Rec() if n == "Recording" else None,
                                    get_channel=lambda cid: vchat if cid == 56 else None, loop=None)
        cog = music.MusicCog(bot)
        music.get_state(guild.id).text_channel_id = 56     # last command came from a VC chat
        assert await cog.autojoin(person(10), chan) is True
        assert connected and attached and attached[0][1] is text   # system channel, not the VC chat
        # An autojoin must not decide where voice-control replies go.
        assert attached[0][2] == {"reply_here": False}
        assert not music.get_state(guild.id).queue and guild.id not in cog._joining
        # Already connected → nothing to do.
        assert await cog.autojoin(person(11), chan) is False
    run(go())


def test_a_summon_during_an_autojoin_waits_for_it():
    async def go():
        cog = music.MusicCog(types.SimpleNamespace(get_cog=lambda n: None))
        ev = asyncio.Event()
        cog._joining[GID + 3] = ev
        chan = types.SimpleNamespace(name="Lounge")
        vc = types.SimpleNamespace(channel=chan, is_connected=lambda: True,
                                   is_playing=lambda: False, is_paused=lambda: False)
        guild = types.SimpleNamespace(id=GID + 3, voice_client=None)
        sent = []

        async def send(content=None, **kw):
            sent.append(content)
        class SummonCtx:
            bot = types.SimpleNamespace(get_cog=lambda n: cog if n == "Music" else None)
            channel = None
            author = types.SimpleNamespace(voice=types.SimpleNamespace(channel=chan))

            @property
            def voice_client(self):
                return self.guild.voice_client
        ctx = SummonCtx()
        ctx.guild = guild
        ctx.send = send

        async def finish_join():
            await asyncio.sleep(0.05)
            guild.voice_client = vc
            ev.set()
        asyncio.create_task(finish_join())
        got = await music._connect_to_author(ctx)
        assert got is vc and sent == []
    run(go())


# ── where voice-control replies go (the #general regression) ─────────────────

def test_music_channel_is_remembered_across_a_restart():
    async def go():
        await db.init_db()
        gid = GID + 10
        music.get_state(gid).text_channel_id = None
        ctx = types.SimpleNamespace(guild=types.SimpleNamespace(id=gid),
                                    channel=types.SimpleNamespace(id=777))
        music._remember_text_channel(ctx)                  # someone typed -play in #spam
        await asyncio.gather(*music._bg_tasks)
        assert await db.get_app_setting(f"music_channel:{gid}") == "777"
        music.get_state(gid).text_channel_id = None        # …the bot restarts…
        await music.MusicCog(types.SimpleNamespace(get_cog=lambda n: None)).cog_load()
        assert music.get_state(gid).text_channel_id == 777
    run(go())


def _dj_with_channels(gid, *, summoned=None, remembered=None, configured=None):
    import cogs.voicedj as vd
    spam = types.SimpleNamespace(id=777, name="spam", permissions_for=lambda m: Perms())
    general = types.SimpleNamespace(id=1, name="general", permissions_for=lambda m: Perms())
    vchat = voice_channel(56, "vc-chat")
    chans = {777: spam, 1: general, 56: vchat}
    rec = types.SimpleNamespace(text_channels={gid: summoned} if summoned else {})
    bot = types.SimpleNamespace(loop=None, get_channel=chans.get,
                                get_cog=lambda n: rec if n == "Recording" else None)
    cog = vd.VoiceDJ(bot)
    cog._cfg[gid] = {"enabled": True, "wake": ["hey dj"], "channel_id": configured}
    guild = types.SimpleNamespace(id=gid, system_channel=general, me=person(1, bot=True),
                                  get_channel=chans.get)
    music.get_state(gid).text_channel_id = remembered
    return cog, guild, chans


@pytest.mark.parametrize("summoned,remembered,configured,expected", [
    (None, 777, None, 777),      # autojoined after a restart: where music was last used
    (None, 56, None, 1),         # never a voice channel's own chat → system channel
    (None, None, None, 1),       # nothing known at all → system channel
    ("summon", 777, None, 99),   # a typed summon still wins over the memory
    (None, 777, 1, 1),           # `voice channel #x` always wins
])
def test_voice_replies_go_where_music_is_used(summoned, remembered, configured, expected):
    async def go():
        gid = GID + 20
        summon_ch = types.SimpleNamespace(id=99) if summoned else None
        cog, guild, _chans = _dj_with_channels(gid, summoned=summon_ch, remembered=remembered,
                                               configured=configured)
        try:
            ch = await cog._echo_channel(guild)
        finally:
            cog._asr.shutdown()
            cog._models.shutdown()
        assert ch is not None and ch.id == expected
    run(go())
