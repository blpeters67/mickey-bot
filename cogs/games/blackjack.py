"""
Blackjack cog. Same game as the original, with a subtle fix: the
original had `g["turn_idx"] -= 1` after a hit because the dispatch
function unconditionally incremented turn_idx after — a tangled hack.
Here we only increment on stand/double/bust, so the turn naturally
stays put on a hit.
"""
from __future__ import annotations

import random

import discord
from discord.ext import commands

from core.logging_setup import get_logger

log = get_logger(__name__)

BJ_STARTING_CHIPS = 500
BJ_RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
BJ_SUITS = ["♠","♥","♦","♣"]

bj_games: dict[int, dict] = {}


def bj_make_deck():
    deck = [{"rank": r, "suit": s} for r in BJ_RANKS for s in BJ_SUITS] * 6
    random.shuffle(deck)
    return deck

def bj_card_str(c): return f"{c['rank']}{c['suit']}"

def bj_hand_str(cards, hide_second=False):
    if hide_second and len(cards) >= 2:
        return f"{bj_card_str(cards[0])}  🂠"
    return "  ".join(bj_card_str(c) for c in cards)

def bj_hand_value(cards):
    val = aces = 0
    for c in cards:
        if c["rank"] in ("J","Q","K"): val += 10
        elif c["rank"] == "A":         val += 11; aces += 1
        else:                          val += int(c["rank"])
    while val > 21 and aces:
        val -= 10
        aces -= 1
    return val


def bj_new_game(guild_id):
    return {
        "guild_id": guild_id, "phase": "lobby", "players": [],
        "dealer_hand": [], "deck": [], "channel_id": None, "host_id": None,
        "turn_idx": 0, "msg_id": None,
    }

def bj_get(gid): return bj_games.get(gid)


def bj_build_embed(g, *, reveal_dealer=False):
    embed = discord.Embed(title="🃏 Blackjack", color=discord.Color.dark_green())
    dval = bj_hand_value(g["dealer_hand"])
    dstr = bj_hand_str(g["dealer_hand"], hide_second=not reveal_dealer)
    embed.add_field(
        name="Dealer",
        value=(f"{dstr}  ({dval})" if reveal_dealer else dstr) or "—",
        inline=False,
    )
    for i, p in enumerate(g["players"]):
        val    = bj_hand_value(p["hand"])
        marker = " 👈" if (g["phase"] == "playing" and i == g["turn_idx"]) else ""
        if   p.get("busted"):   status = " 💥 BUST"
        elif p.get("stood"):    status = " ✋ Stand"
        elif p.get("bj"):       status = " ⭐ Blackjack!"
        elif p.get("doubled"):  status = " ×2"
        else:                   status = ""
        embed.add_field(
            name=f"{p['name']}{marker}  (bet: ${p['bet']}  chips: ${p['chips']}){status}",
            value=f"{bj_hand_str(p['hand'])}  ({val})",
            inline=False,
        )
    return embed


class BjBetModal(discord.ui.Modal, title="Place Your Bet"):
    amount = discord.ui.TextInput(label="Bet amount", placeholder="e.g. 50",
                                   min_length=1, max_length=6)

    def __init__(self, cog):
        super().__init__()
        self.cog = cog

    async def on_submit(self, i):
        try:
            amt = int(self.amount.value)
        except ValueError:
            return await i.response.send_message("❌ Enter a number.", ephemeral=True)
        await self.cog.place_bet(i, amt)


class BjLobbyView(discord.ui.View):
    def __init__(self, cog): super().__init__(timeout=None); self.cog = cog

    @discord.ui.button(label="Join", style=discord.ButtonStyle.success, emoji="✋")
    async def join(self, i, _b):
        g = bj_get(i.guild_id)
        if not g or g["phase"] != "lobby":
            return await i.response.send_message("❌ No lobby.", ephemeral=True)
        if any(p["id"] == i.user.id for p in g["players"]):
            return await i.response.send_message("❌ Already joined.", ephemeral=True)
        if len(g["players"]) >= 7:
            return await i.response.send_message("❌ Table full.", ephemeral=True)
        g["players"].append({
            "id": i.user.id, "name": i.user.display_name,
            "chips": BJ_STARTING_CHIPS, "bet": 0, "hand": [],
            "stood": False, "doubled": False, "busted": False, "bj": False,
        })
        await i.response.edit_message(embed=bj_build_embed(g), view=self)
        await i.followup.send(f"✅ **{i.user.display_name}** joined with ${BJ_STARTING_CHIPS}!")

    @discord.ui.button(label="Start / Deal", style=discord.ButtonStyle.primary, emoji="🃏")
    async def start(self, i, _b):
        g = bj_get(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        if g["host_id"] != i.user.id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        if not g["players"]:
            return await i.response.send_message("❌ Need at least 1 player.", ephemeral=True)
        if g["phase"] != "lobby":
            return await i.response.send_message("❌ Already started.", ephemeral=True)
        g["phase"] = "betting"
        g["deck"]  = bj_make_deck()
        await i.response.edit_message(embed=bj_build_embed(g), view=None)
        msg = await i.channel.send(
            "💰 **Place your bets!**",
            view=self.cog._track_view(g, BjBettingView(self.cog)),
        )
        g["msg_id"] = msg.id


class BjBettingView(discord.ui.View):
    def __init__(self, cog): super().__init__(timeout=None); self.cog = cog

    @discord.ui.button(label="Place Bet", style=discord.ButtonStyle.primary, emoji="💰")
    async def bet(self, i, _b):
        g = bj_get(i.guild_id)
        if not g or g["phase"] != "betting":
            return await i.response.send_message("❌ Not betting phase.", ephemeral=True)
        p = next((p for p in g["players"] if p["id"] == i.user.id), None)
        if not p: return await i.response.send_message("❌ Not in game.", ephemeral=True)
        if p["bet"] > 0: return await i.response.send_message("❌ Already bet.", ephemeral=True)
        await i.response.send_modal(BjBetModal(self.cog))


class BjReplayView(discord.ui.View):
    def __init__(self, cog): super().__init__(timeout=60); self.cog = cog

    @discord.ui.button(label="Play Again", style=discord.ButtonStyle.success, emoji="🔄")
    async def again(self, i, _b):
        g = bj_get(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        if g["host_id"] != i.user.id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        broke = [p for p in g["players"] if p["chips"] <= 0]
        if broke:
            await i.channel.send(f"💸 {', '.join(p['name'] for p in broke)} ran out and left.")
        g["players"] = [p for p in g["players"] if p["chips"] > 0]
        if not g["players"]:
            bj_games.pop(i.guild_id, None)
            return await i.response.send_message("Everyone's broke — game over!")
        g["phase"] = "betting"
        g["deck"]  = bj_make_deck()
        for p in g["players"]:
            p["bet"] = 0
        await i.response.send_message(
            "💰 New round! Place your bets.",
            view=self.cog._track_view(g, BjBettingView(self.cog)),
        )
        self.stop()

    @discord.ui.button(label="End", style=discord.ButtonStyle.danger, emoji="🛑")
    async def end(self, i, _b):
        g = bj_get(i.guild_id)
        if not g: return
        if g["host_id"] != i.user.id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        bj_games.pop(i.guild_id, None)
        await i.response.send_message("🃏 Game ended.")
        self.stop()


class BjActionView(discord.ui.View):
    def __init__(self, cog): super().__init__(timeout=None); self.cog = cog

    @discord.ui.button(label="Hit",         style=discord.ButtonStyle.success,   emoji="👊")
    async def hit(self, i, _b):    await self.cog.action(i, "hit")
    @discord.ui.button(label="Stand",       style=discord.ButtonStyle.primary,   emoji="✋")
    async def stand(self, i, _b):  await self.cog.action(i, "stand")
    @discord.ui.button(label="Double Down", style=discord.ButtonStyle.secondary, emoji="💰")
    async def double(self, i, _b): await self.cog.action(i, "double")


class BlackjackCog(commands.Cog, name="Blackjack"):
    def __init__(self, bot): self.bot = bot

    @staticmethod
    def _track_view(g, view):
        """Stop the previous live view before registering a new one — views
        with timeout=None are never released by discord.py until stopped, so
        each turn's action view used to linger for the process's lifetime.
        Pass None to just stop the current one."""
        old = g.get("_live_view")
        if old is not None:
            old.stop()
        g["_live_view"] = view
        return view

    @commands.group(name="blackjack", aliases=["bj"], invoke_without_command=True)
    async def blackjack(self, ctx):
        await ctx.send(f"Use `{ctx.prefix}bj new` to start a table.")

    @blackjack.command(name="new")
    async def new(self, ctx):
        if ctx.guild.id in bj_games:
            return await ctx.send(f"❌ Already running. Use `{ctx.prefix}bj end` first.")
        g = bj_new_game(ctx.guild.id)
        g["channel_id"] = ctx.channel.id
        g["host_id"]    = ctx.author.id
        g["players"].append({
            "id": ctx.author.id, "name": ctx.author.display_name,
            "chips": BJ_STARTING_CHIPS, "bet": 0, "hand": [],
            "stood": False, "doubled": False, "busted": False, "bj": False,
        })
        bj_games[ctx.guild.id] = g
        await ctx.send(
            f"🃏 **{ctx.author.display_name}** opened a Blackjack table!",
            embed=bj_build_embed(g),
            view=self._track_view(g, BjLobbyView(self)),
        )

    @blackjack.command(name="end")
    async def end(self, ctx):
        g = bj_get(ctx.guild.id)
        if not g: return await ctx.send("❌ No game.")
        if g["host_id"] != ctx.author.id: return await ctx.send("❌ Host only.")
        self._track_view(g, None)
        bj_games.pop(ctx.guild.id, None)
        await ctx.send("🃏 Game ended.")

    async def place_bet(self, i, amt):
        g = bj_get(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        p = next((p for p in g["players"] if p["id"] == i.user.id), None)
        if not p: return await i.response.send_message("❌ Not in game.", ephemeral=True)
        if amt < 1 or amt > p["chips"]:
            return await i.response.send_message(
                f"❌ Bet must be $1–${p['chips']}.", ephemeral=True
            )
        p["bet"]    = amt
        p["chips"] -= amt
        await i.response.send_message(f"✅ **{p['name']}** bet ${amt}.")
        if all(p["bet"] > 0 for p in g["players"]):
            await self.deal_round(g, self.bot.get_channel(g["channel_id"]))

    async def deal_round(self, g, channel):
        for p in g["players"]:
            p["hand"]  = [g["deck"].pop(), g["deck"].pop()]
            p["stood"] = p["doubled"] = p["busted"] = p["bj"] = False
        g["dealer_hand"] = [g["deck"].pop(), g["deck"].pop()]
        g["turn_idx"]    = 0
        g["phase"]       = "playing"
        for p in g["players"]:
            if bj_hand_value(p["hand"]) == 21:
                p["bj"] = p["stood"] = True
        await self.next_turn(g, channel)

    async def next_turn(self, g, channel):
        while g["turn_idx"] < len(g["players"]):
            p = g["players"][g["turn_idx"]]
            if not p["stood"] and not p["busted"]:
                break
            g["turn_idx"] += 1
        if g["turn_idx"] >= len(g["players"]):
            await self.dealer_turn(g, channel)
            return
        p   = g["players"][g["turn_idx"]]
        val = bj_hand_value(p["hand"])
        view = self._track_view(g, BjActionView(self))
        if p["chips"] < p["bet"]:
            view.double.disabled = True
        msg = await channel.send(
            f"**{p['name']}**'s turn — {bj_hand_str(p['hand'])} ({val})",
            embed=bj_build_embed(g), view=view,
        )
        g["msg_id"] = msg.id

    async def dealer_turn(self, g, channel):
        g["phase"] = "dealer"
        while bj_hand_value(g["dealer_hand"]) < 17:
            g["dealer_hand"].append(g["deck"].pop())
        await self.resolve(g, channel, bj_hand_value(g["dealer_hand"]))

    async def resolve(self, g, channel, dealer_val):
        g["phase"] = "done"
        # A natural (2-card) blackjack beats any dealer 21 made with 3+ cards;
        # it only pushes against the dealer's own natural.
        dealer_natural = dealer_val == 21 and len(g["dealer_hand"]) == 2
        lines = [f"**Dealer:** {bj_hand_str(g['dealer_hand'])} ({dealer_val})\n"]
        for p in g["players"]:
            pval = bj_hand_value(p["hand"])
            if p["busted"]:
                lines.append(f"❌ **{p['name']}** busted — lost ${p['bet']}")
            elif p.get("bj") and not dealer_natural:
                win = int(p["bet"] * 1.5)
                p["chips"] += p["bet"] + win
                lines.append(f"⭐ **{p['name']}** Blackjack! +${win}")
            elif dealer_val > 21 or pval > dealer_val:
                p["chips"] += p["bet"] * 2
                lines.append(f"✅ **{p['name']}** wins! +${p['bet']}")
            elif pval == dealer_val:
                p["chips"] += p["bet"]
                lines.append(f"🤝 **{p['name']}** push.")
            else:
                lines.append(f"❌ **{p['name']}** lost ${p['bet']}.")
            p["bet"] = 0
        lines.append("\n**Chip counts:**")
        for p in sorted(g["players"], key=lambda x: x["chips"], reverse=True):
            lines.append(f"  {p['name']}: ${p['chips']}")
        self._track_view(g, None)   # round over — release the action view
        await channel.send("\n".join(lines), embed=bj_build_embed(g, reveal_dealer=True))
        await channel.send("Play again?", view=BjReplayView(self))

    async def action(self, i, action):
        g = bj_get(i.guild_id)
        if not g or g["phase"] != "playing":
            return await i.response.send_message("❌ Not playing.", ephemeral=True)
        p = g["players"][g["turn_idx"]] if g["turn_idx"] < len(g["players"]) else None
        if not p or p["id"] != i.user.id:
            return await i.response.send_message("❌ Not your turn.", ephemeral=True)

        channel = self.bot.get_channel(g["channel_id"])
        if g.get("msg_id"):
            try:
                old = await channel.fetch_message(g["msg_id"])
                await old.edit(view=None)
            except Exception:
                pass

        advance = True

        if action == "hit":
            p["hand"].append(g["deck"].pop())
            val = bj_hand_value(p["hand"])
            if val > 21:
                p["busted"] = True
                await i.response.send_message(f"💥 **{p['name']}** busted ({val})")
            elif val == 21:
                p["stood"] = True
                await i.response.send_message(f"✅ **{p['name']}** hit 21!")
            else:
                await i.response.send_message(
                    f"👊 **{p['name']}** hits — {bj_hand_str(p['hand'])} ({val})"
                )
                advance = False
        elif action == "stand":
            p["stood"] = True
            await i.response.send_message(
                f"✋ **{p['name']}** stands at {bj_hand_value(p['hand'])}."
            )
        elif action == "double":
            if p["chips"] < p["bet"]:
                return await i.response.send_message("❌ Not enough chips.", ephemeral=True)
            p["chips"] -= p["bet"]
            p["bet"]   *= 2
            p["hand"].append(g["deck"].pop())
            p["doubled"] = p["stood"] = True
            val = bj_hand_value(p["hand"])
            if val > 21: p["busted"] = True
            await i.response.send_message(
                f"×2 **{p['name']}** doubles — {bj_hand_str(p['hand'])} ({val})"
                + (" 💥 BUST" if p["busted"] else "")
            )

        if advance:
            g["turn_idx"] += 1
        await self.next_turn(g, channel)


async def setup(bot):
    await bot.add_cog(BlackjackCog(bot))
