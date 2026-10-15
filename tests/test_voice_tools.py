"""The voice DJ's capability registry (core/voice_tools.py): every command
the bot has is either run by voice — through the same code as typed — or
excused with a reason; every bridge matches its command's real parameters;
and a bridged command keeps its own permission checks."""
from __future__ import annotations

import asyncio
import importlib
import inspect
import re
import types

import discord
import pytest
from discord.ext import commands

import bot as botmod
from core import voice_plan as vp
from core import voice_tools as vt
from core.voice_tools import NOT_BY_VOICE, OPS


def _all_commands() -> dict[str, commands.Command]:
    """Every prefix/hybrid command the cogs declare, by qualified name."""
    out: dict[str, commands.Command] = {}
    for mod_name in botmod.COG_MODULES:
        mod = importlib.import_module(mod_name)
        for _n, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, commands.Cog) and cls.__module__ == mod.__name__:
                stack = list(cls.__cog_commands__)
                while stack:
                    c = stack.pop()
                    out[c.qualified_name] = c
                    if isinstance(c, commands.Group):
                        stack.extend(c.commands)
    return out


def _context_menus() -> set[str]:
    src = open("cogs/downloads.py", encoding="utf-8").read()
    return {f"menu:{m}" for m in re.findall(r'ContextMenu\(name="([^"]+)"', src)}


def test_every_command_is_covered_by_voice_or_excused():
    names = set(_all_commands()) | _context_menus()
    rows = vt.coverage_table(names)
    unclassified = [r[0] for r in rows if r[1] == "UNCLASSIFIED"]
    assert not unclassified, f"say how these work by voice (core/voice_tools.py): {unclassified}"
    cov = vt.coverage()
    assert not (set(cov) & set(NOT_BY_VOICE)), "a command is both covered and excused"
    assert not [c for c in cov if c not in names], "an op covers a command that doesn't exist"
    assert not [c for c in NOT_BY_VOICE if c not in names], "an excuse for a missing command"
    voice = sum(1 for r in rows if r[1].startswith("voice"))
    assert voice >= 70                                    # (72 at round 8)


def test_every_bridge_matches_its_commands_real_parameters():
    cmds = _all_commands()
    step_fields = set(vp.Step.__dataclass_fields__)
    for spec in OPS.values():
        targets = set(spec.choices.values()) | ({spec.command} if spec.command else set())
        for name in targets:
            assert name in cmds, f"{spec.name}: no command {name!r}"
            params = cmds[name].clean_params
            for p, field in spec.params.items():
                if p in params:
                    assert field in step_fields, f"{spec.name}: no step field {field!r}"
            missing = [p for p, prm in params.items() if prm.required and p not in spec.params]
            assert not missing, f"{spec.name} → {name}: required {missing} not given by voice"
        for p in spec.params:
            assert any(p in cmds[n].clean_params for n in targets), \
                f"{spec.name}: {p!r} isn't a parameter of {sorted(targets)}"


def test_the_planner_is_told_every_operation_and_the_typed_only_ones():
    text = vt.planner_ops_text()
    for name in OPS:
        assert f"  {name}:" in text
    assert "Typed-only commands" in text and "blur" in text
    assert vt.typed_only_reason("-blur") and vt.typed_only_reason("voice enable")
    assert vt.typed_only_reason("skip") is None


# ── a bridged command, run as if typed ───────────────────────────────────────

class _Ch:
    def __init__(self):
        self.sent = []
        self.id = 77

    async def send(self, content=None, **kw):
        self.sent.append(content)

    def permissions_for(self, _m):
        return types.SimpleNamespace(send_messages=True)


class _Fake(commands.Cog):
    def __init__(self):
        self.calls = []

    async def cog_check(self, ctx):
        if ctx.guild is None:
            raise commands.NoPrivateMessage()
        return True

    @commands.command(name="fakeadmin")
    @commands.has_guild_permissions(manage_guild=True)
    async def fakeadmin(self, ctx, rounds: int = 1):
        self.calls.append(("admin", rounds))
        await ctx.send(f"admin ran {rounds}")

    @commands.command(name="fakeplain")
    async def fakeplain(self, ctx, *, query: str | None = None):
        self.calls.append(("plain", query))
        await ctx.send(f"plain {query}")


def test_a_bridged_command_keeps_its_permission_checks_and_typed_arguments(monkeypatch):
    import cogs.voicedj as vd

    async def go():
        bot = commands.Bot(command_prefix="-", intents=discord.Intents.none(), help_command=None)
        fake = _Fake()
        await bot.add_cog(fake)
        ch = _Ch()
        guild = types.SimpleNamespace(id=9, me=None, voice_client=None, get_member=lambda i: None)

        def member(admin):
            return types.SimpleNamespace(id=5, display_name="Alex", bot=False, guild=guild,
                                         guild_permissions=discord.Permissions(manage_guild=admin))
        cog = vd.VoiceDJ(bot)

        async def echo(g):
            return ch
        cog._echo_channel = echo

        async def prefix(g):
            return "-"
        cog._prefix = prefix
        spec_admin = vt.OpSpec("x", "do x", "", "command", command="fakeadmin",
                               params={"rounds": "count"})
        spec_plain = vt.OpSpec("y", "do y", "", "command", command="fakeplain",
                               params={"query": "value"})
        st = vp.Step(1, "x", count=3)
        status, msg = await cog._run_bridged(guild, member(False), spec_admin, st, "hey dj x")
        assert status == "failed" and "permission" in msg and fake.calls == []
        status, msg = await cog._run_bridged(guild, member(True), spec_admin, st, "hey dj x")
        assert status == "done" and fake.calls == [("admin", 3)] and ch.sent == ["admin ran 3"]
        status, _ = await cog._run_bridged(guild, member(False), spec_plain,
                                           vp.Step(1, "y", value="mr brightside"), "")
        assert status == "done" and fake.calls[-1] == ("plain", "mr brightside")
        bad = vp.Step(1, "x", count=1)
        spec_text = vt.OpSpec("z", "do z", "", "command", command="fakeadmin",
                              params={"rounds": "value"})
        status, msg = await cog._run_bridged(guild, member(True), spec_text,
                                             vp.Step(1, "z", value="lots"), "")
        assert status == "failed" and "isn't a number" in msg
        await bot.close()
    asyncio.run(go())


def test_a_command_missing_an_input_asks_for_it_instead_of_failing_quietly():
    """ "Hey DJ, set a reminder" with no time: the step asks "When should I
    remind you?" (and the question window opens), the command isn't run."""
    import cogs.voicedj as vd
    from core import voice_exec as vx

    class _Bot:
        def get_command(self, name):
            kw = inspect.Parameter.KEYWORD_ONLY
            return types.SimpleNamespace(clean_params={
                "when": commands.Parameter("when", kw, annotation=str),
                "message": commands.Parameter("message", kw, annotation=str, default=None)})

    cog = vd.VoiceDJ.__new__(vd.VoiceDJ)
    cog.bot = _Bot()
    ran = []

    async def no_run(*a, **k):
        ran.append(a)
        return "done", ""
    cog._run_bridged = no_run
    spec = OPS["remind"]
    assert cog._bridged_missing(spec, vp.Step(1, "remind", title="take the bins out")) == "when"
    assert cog._bridged_missing(spec, vp.Step(1, "remind", value="in an hour")) == ""
    run = vx.PlanRun.__new__(vx.PlanRun)
    run.cog, run.act = cog, types.SimpleNamespace(text="hey dj remind me to take the bins out")
    res = asyncio.run(run._run_step(vp.Step(1, "remind", title="take the bins out"), None, None))
    assert res.status == "failed" and res.ask == "When should I remind you?" and not ran
    assert res.message.startswith("❓")
