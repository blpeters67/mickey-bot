"""
Yahtzee cog. 1–6 players, up to 3 rolls per turn, classic scorecard.
"""
from __future__ import annotations

import random
from collections import Counter

import discord
from discord.ext import commands

from core.logging_setup import get_logger

log = get_logger(__name__)

CATEGORIES_UPPER = [
    ("ones",   "Ones"),  ("twos",   "Twos"),  ("threes", "Threes"),
    ("fours",  "Fours"), ("fives",  "Fives"), ("sixes",  "Sixes"),
]
CATEGORIES_LOWER = [
    ("three_kind",     "Three of a Kind"),
    ("four_kind",      "Four of a Kind"),
    ("full_house",     "Full House"),
    ("small_straight", "Small Straight"),
    ("large_straight", "Large Straight"),
    ("yahtzee",        "Yahtzee"),
    ("chance",         "Chance"),
]
ALL_CATEGORIES = CATEGORIES_UPPER + CATEGORIES_LOWER
CAT_KEYS = [k for k, _ in ALL_CATEGORIES]
CAT_LABELS = dict(ALL_CATEGORIES)

# Keycap emojis render large and readable in Discord — the unicode die
# glyphs (⚀⚁⚂…) are nearly illegible at message font size.
DICE_FACES = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣"}

yz_games: dict[int, dict] = {}


def yz_score(category: str, dice: list[int]) -> int:
    c = Counter(dice)
    total = sum(dice)
    if category == "ones":    return c[1] * 1
    if category == "twos":    return c[2] * 2
    if category == "threes":  return c[3] * 3
    if category == "fours":   return c[4] * 4
    if category == "fives":   return c[5] * 5
    if category == "sixes":   return c[6] * 6
    if category == "three_kind":
        return total if any(v >= 3 for v in c.values()) else 0
    if category == "four_kind":
        return total if any(v >= 4 for v in c.values()) else 0
    if category == "full_house":
        return 25 if sorted(c.values()) == [2, 3] else 0
    if category == "small_straight":
        u = set(dice)
        return 30 if any(set(s).issubset(u) for s in (
            {1,2,3,4}, {2,3,4,5}, {3,4,5,6})) else 0
    if category == "large_straight":
        return 40 if set(dice) in ({1,2,3,4,5}, {2,3,4,5,6}) else 0
    if category == "yahtzee":
        return 50 if any(v == 5 for v in c.values()) else 0
    if category == "chance":
        return total
    return 0


def yz_new_game(guild_id: int) -> dict:
    return {
        "guild_id":   guild_id, "phase": "lobby", "players": [],
        "channel_id": None,    "host_id": None,
        "turn_idx":   0,       "dice": [0]*5, "keep": [False]*5,
        "rolls_left": 3,
    }

def yz_get(gid): return yz_games.get(gid)


def yz_dice_display(dice, keep):
    if all(d == 0 for d in dice):
        return "🎲 *(not rolled yet)*"
    out = []
    for d, k in zip(dice, keep):
        face = DICE_FACES.get(d, "❔")
        out.append(f"{face}🔒" if k else face)
    return "  ".join(out) + "\n*🔒 = held — toggle with the dice buttons below*"


def yz_scorecard_field(p):
    parts = []
    used = p["scores"]
    for k, label in CATEGORIES_UPPER:
        v = used.get(k)
        parts.append(f"{label}: {v if v is not None else '—'}")
    upper_sum = sum(v for k, v in used.items() if k in {x for x,_ in CATEGORIES_UPPER} and v is not None)
    bonus = 35 if upper_sum >= 63 else 0
    parts.append(f"*Upper: {upper_sum}{' (+35 bonus!)' if bonus else ''}*")
    for k, label in CATEGORIES_LOWER:
        v = used.get(k)
        parts.append(f"{label}: {v if v is not None else '—'}")
    total = sum(v for v in used.values() if v is not None) + bonus
    parts.append(f"**Total: {total}**")
    return "\n".join(parts)


def yz_build_embed(g, *, title="🎲 Yahtzee"):
    embed = discord.Embed(title=title, color=discord.Color.gold())
    if g["phase"] == "lobby":
        names = "\n".join(f"• {p['name']}" for p in g["players"]) or "*(no one yet)*"
        embed.add_field(name="Players", value=names, inline=False)
        return embed
    cur = g["players"][g["turn_idx"]]
    embed.add_field(
        name=f"{cur['name']}'s turn (rolls left: {g['rolls_left']})",
        value=yz_dice_display(g["dice"], g["keep"]),
        inline=False,
    )
    for p in g["players"]:
        marker = " 👈" if p is cur else ""
        embed.add_field(
            name=f"{p['name']}{marker}",
            value=yz_scorecard_field(p),
            inline=True,
        )
    return embed


class YzLobbyView(discord.ui.View):
    def __init__(self, cog): super().__init__(timeout=None); self.cog = cog

    @discord.ui.button(label="Join", style=discord.ButtonStyle.success, emoji="✋")
    async def join(self, i, _b):
        g = yz_get(i.guild_id)
        if not g or g["phase"] != "lobby":
            return await i.response.send_message("❌ No lobby.", ephemeral=True)
        if any(p["id"] == i.user.id for p in g["players"]):
            return await i.response.send_message("❌ Already joined.", ephemeral=True)
        if len(g["players"]) >= 6:
            return await i.response.send_message("❌ Max 6 players.", ephemeral=True)
        g["players"].append({
            "id": i.user.id, "name": i.user.display_name, "scores": {},
        })
        await i.response.edit_message(embed=yz_build_embed(g), view=self)

    @discord.ui.button(label="Start", style=discord.ButtonStyle.primary, emoji="🎲")
    async def start(self, i, _b):
        g = yz_get(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        if g["host_id"] != i.user.id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        if not g["players"]:
            return await i.response.send_message("❌ Need at least 1 player.", ephemeral=True)
        g["phase"] = "playing"
        await i.response.edit_message(embed=yz_build_embed(g), view=None)
        await self.cog.start_turn(g, i.channel)


class YzKeepView(discord.ui.View):
    def __init__(self, cog):
        super().__init__(timeout=None)
        self.cog = cog

    def sync_buttons(self, g) -> None:
        """Mirror the game state onto the buttons: each die button shows its
        rolled value (green when held), and everything except Score greys out
        once the rolls are spent."""
        no_rolls = g["rolls_left"] <= 0
        for idx, btn in enumerate((self.d1, self.d2, self.d3, self.d4, self.d5)):
            val  = g["dice"][idx]
            held = g["keep"][idx]
            btn.label    = f"{'🔒 ' if held else ''}{val}" if val else f"D{idx + 1}"
            btn.style    = (discord.ButtonStyle.success if held
                            else discord.ButtonStyle.secondary)
            btn.disabled = val == 0 or no_rolls
        self.roll.disabled = no_rolls

    async def _toggle(self, i, idx):
        g = yz_get(i.guild_id)
        if not g or g["phase"] != "playing":
            return await i.response.send_message("❌ No game.", ephemeral=True)
        cur = g["players"][g["turn_idx"]]
        if cur["id"] != i.user.id:
            return await i.response.send_message("❌ Not your turn.", ephemeral=True)
        if g["dice"][idx] == 0:
            return await i.response.send_message("❌ Haven't rolled yet.", ephemeral=True)
        if g["rolls_left"] <= 0:
            return await i.response.send_message("❌ No rolls left — pick a category.", ephemeral=True)
        g["keep"][idx] = not g["keep"][idx]
        self.sync_buttons(g)
        await i.response.edit_message(embed=yz_build_embed(g), view=self)

    @discord.ui.button(label="D1", style=discord.ButtonStyle.secondary)
    async def d1(self, i, _b): await self._toggle(i, 0)
    @discord.ui.button(label="D2", style=discord.ButtonStyle.secondary)
    async def d2(self, i, _b): await self._toggle(i, 1)
    @discord.ui.button(label="D3", style=discord.ButtonStyle.secondary)
    async def d3(self, i, _b): await self._toggle(i, 2)
    @discord.ui.button(label="D4", style=discord.ButtonStyle.secondary)
    async def d4(self, i, _b): await self._toggle(i, 3)
    @discord.ui.button(label="D5", style=discord.ButtonStyle.secondary)
    async def d5(self, i, _b): await self._toggle(i, 4)

    @discord.ui.button(label="🎲 Roll", style=discord.ButtonStyle.primary, row=1)
    async def roll(self, i, _b):
        g = yz_get(i.guild_id)
        if not g or g["phase"] != "playing":
            return await i.response.send_message("❌ No game.", ephemeral=True)
        cur = g["players"][g["turn_idx"]]
        if cur["id"] != i.user.id:
            return await i.response.send_message("❌ Not your turn.", ephemeral=True)
        if g["rolls_left"] <= 0:
            return await i.response.send_message("❌ No rolls left.", ephemeral=True)
        for idx in range(5):
            if not g["keep"][idx]:
                g["dice"][idx] = random.randint(1, 6)
        g["rolls_left"] -= 1
        # Keep this view even at 0 rolls — removing it took the 📋 Score
        # button away too, softlocking the turn after a third roll. The
        # die/roll buttons grey out instead.
        self.sync_buttons(g)
        await i.response.edit_message(embed=yz_build_embed(g), view=self)
        await self.cog.maybe_show_score_view(g, i.channel, cur)

    @discord.ui.button(label="📋 Score", style=discord.ButtonStyle.success, row=1)
    async def score(self, i, _b):
        g = yz_get(i.guild_id)
        if not g or g["phase"] != "playing":
            return await i.response.send_message("❌ No game.", ephemeral=True)
        cur = g["players"][g["turn_idx"]]
        if cur["id"] != i.user.id:
            return await i.response.send_message("❌ Not your turn.", ephemeral=True)
        if all(d == 0 for d in g["dice"]):
            return await i.response.send_message("❌ Roll first.", ephemeral=True)
        await i.response.send_message(
            f"📋 Pick a category to score, **{cur['name']}**:",
            view=YzScoreView(self.cog, cur, list(g["dice"])),
            ephemeral=True,
        )


class YzScoreView(discord.ui.View):
    def __init__(self, cog, player, dice):
        super().__init__(timeout=120)
        self.cog = cog
        options = []
        for k, label in ALL_CATEGORIES:
            if k in player["scores"]:
                continue
            options.append(discord.SelectOption(
                label=f"{label} ({yz_score(k, dice)} pts)", value=k,
            ))
        if options:
            self.select.options = options[:25]
        else:
            self.select.disabled = True

    @discord.ui.select(placeholder="Choose a category…", min_values=1, max_values=1, options=[
        discord.SelectOption(label="placeholder", value="placeholder")
    ])
    async def select(self, i, sel):
        g = yz_get(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        cur = g["players"][g["turn_idx"]]
        if cur["id"] != i.user.id:
            return await i.response.send_message("❌ Not your turn.", ephemeral=True)
        cat = sel.values[0]
        if cat in cur["scores"]:
            return await i.response.send_message("❌ Already scored.", ephemeral=True)
        score = yz_score(cat, g["dice"])
        cur["scores"][cat] = score
        await i.response.edit_message(
            content=f"✅ {CAT_LABELS[cat]}: **{score} pts**", view=None,
        )
        await self.cog.advance(g, i.channel)


class YzPlayAgainView(discord.ui.View):
    def __init__(self, cog, players, host_id, channel_id):
        super().__init__(timeout=60)
        self.cog        = cog
        self.players    = players
        self.host_id    = host_id
        self.channel_id = channel_id

    @discord.ui.button(label="Play Again", style=discord.ButtonStyle.success, emoji="🔄")
    async def again(self, i, _b):
        if i.user.id != self.host_id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        g = yz_new_game(i.guild_id)
        g["channel_id"] = self.channel_id
        g["host_id"]    = self.host_id
        g["players"] = [{"id": p["id"], "name": p["name"], "scores": {}} for p in self.players]
        g["phase"]   = "playing"
        yz_games[i.guild_id] = g
        await i.response.edit_message(content="🔄 New game!", view=None)
        await self.cog.start_turn(g, self.cog.bot.get_channel(self.channel_id))
        self.stop()

    @discord.ui.button(label="End", style=discord.ButtonStyle.danger, emoji="🛑")
    async def end(self, i, _b):
        if i.user.id != self.host_id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        await i.response.edit_message(content="🎲 Thanks for playing!", view=None)
        self.stop()


class YahtzeeCog(commands.Cog, name="Yahtzee"):
    def __init__(self, bot): self.bot = bot

    @staticmethod
    def _track_view(g, view):
        """Stop the previous live view before registering a new one — views
        with timeout=None are never released by discord.py until stopped, so
        every turn's YzKeepView used to linger for the process's lifetime.
        Pass None to just stop the current one."""
        old = g.get("_live_view")
        if old is not None:
            old.stop()
        g["_live_view"] = view
        return view

    @commands.group(name="yahtzee", aliases=["yz"], invoke_without_command=True)
    async def yahtzee(self, ctx):
        await ctx.send(f"Use `{ctx.prefix}yz new` to start a Yahtzee game.")

    @yahtzee.command(name="new")
    async def new(self, ctx):
        if ctx.guild.id in yz_games:
            return await ctx.send(f"❌ Already running. Use `{ctx.prefix}yz end` first.")
        g = yz_new_game(ctx.guild.id)
        g["channel_id"] = ctx.channel.id
        g["host_id"]    = ctx.author.id
        g["players"].append({"id": ctx.author.id, "name": ctx.author.display_name, "scores": {}})
        yz_games[ctx.guild.id] = g
        await ctx.send(
            f"🎲 **{ctx.author.display_name}** opened a Yahtzee game!",
            embed=yz_build_embed(g),
            view=self._track_view(g, YzLobbyView(self)),
        )

    @yahtzee.command(name="end")
    async def end(self, ctx):
        g = yz_get(ctx.guild.id)
        if not g: return await ctx.send("❌ No game.")
        if g["host_id"] != ctx.author.id: return await ctx.send("❌ Host only.")
        self._track_view(g, None)
        yz_games.pop(ctx.guild.id, None)
        await ctx.send("🎲 Game ended.")

    async def start_turn(self, g, channel):
        g["dice"]       = [0] * 5
        g["keep"]       = [False] * 5
        g["rolls_left"] = 3
        cur  = g["players"][g["turn_idx"]]
        view = self._track_view(g, YzKeepView(self))
        view.sync_buttons(g)
        await channel.send(
            f"**{cur['name']}**'s turn — click 🎲 Roll to begin.",
            embed=yz_build_embed(g),
            view=view,
        )

    async def maybe_show_score_view(self, g, channel, cur):
        if g["rolls_left"] == 0:
            await channel.send(
                f"📋 No rolls left, **{cur['name']}** — pick a category.",
            )

    async def advance(self, g, channel):
        if all(k in p["scores"] for p in g["players"] for k in CAT_KEYS):
            await self.end_game(g, channel)
            return
        n = len(g["players"])
        for _ in range(n):
            g["turn_idx"] = (g["turn_idx"] + 1) % n
            p = g["players"][g["turn_idx"]]
            if not all(k in p["scores"] for k in CAT_KEYS):
                break
        await self.start_turn(g, channel)

    async def end_game(self, g, channel):
        totals = []
        for p in g["players"]:
            upper = sum(v for k, v in p["scores"].items()
                        if k in {x for x, _ in CATEGORIES_UPPER} and v is not None)
            bonus = 35 if upper >= 63 else 0
            total = sum(v for v in p["scores"].values() if v is not None) + bonus
            totals.append((p, total))
        totals.sort(key=lambda x: x[1], reverse=True)
        winner, top_score = totals[0]
        lines = [f"🏆 **{winner['name']}** wins with **{top_score} points**!\n"]
        lines.append("**Final scores:**")
        for p, t in totals:
            lines.append(f"  {p['name']}: **{t}**")
        players = list(g["players"])
        host_id = g["host_id"]
        chan_id = g["channel_id"]
        self._track_view(g, None)
        yz_games.pop(g["guild_id"], None)
        await channel.send(
            "\n".join(lines),
            view=YzPlayAgainView(self, players, host_id, chan_id),
        )


async def setup(bot):
    await bot.add_cog(YahtzeeCog(bot))
