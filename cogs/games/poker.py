"""
Texas Hold'em — same gameplay as the original bot.py, lifted into a cog
so the file can be loaded/reloaded independently.
"""
from __future__ import annotations

import itertools
import random

import discord
from discord.ext import commands

from core.logging_setup import get_logger

log = get_logger(__name__)

POKER_STARTING_CHIPS = 1000
POKER_SMALL_BLIND    = 10
POKER_BIG_BLIND      = 20

SUITS  = ["♠", "♥", "♦", "♣"]
RANKS  = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
RANK_V = {r: i for i, r in enumerate(RANKS)}

HAND_NAMES = {
    8: "Straight Flush", 7: "Four of a Kind", 6: "Full House",
    5: "Flush",          4: "Straight",       3: "Three of a Kind",
    2: "Two Pair",       1: "One Pair",       0: "High Card",
}

poker_games: dict[int, dict] = {}


def make_deck():
    return [{"rank": r, "suit": s} for r in RANKS for s in SUITS]

def card_str(c):  return f"{c['rank']}{c['suit']}"
def hand_str(cs): return "  ".join(card_str(c) for c in cs)
def card_val(c):  return RANK_V[c["rank"]]


def eval_five(cards):
    vals     = sorted([card_val(c) for c in cards], reverse=True)
    suits    = [c["suit"] for c in cards]
    flush    = len(set(suits)) == 1
    sv       = vals[:]
    straight = sv == list(range(sv[0], sv[0] - 5, -1))
    if set(sv) == {12, 0, 1, 2, 3}:
        straight = True
        sv = [3, 2, 1, 0, -1]
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freqs  = [g[1] for g in groups]
    gvals  = [g[0] for g in groups]
    if straight and flush: return (8, sv)
    if freqs[0] == 4:      return (7, gvals)
    if freqs[0] == 3 and freqs[1] == 2: return (6, gvals)
    if flush:              return (5, vals)
    if straight:           return (4, sv)
    if freqs[0] == 3:      return (3, gvals)
    if freqs[0] == 2 and freqs[1] == 2: return (2, gvals)
    if freqs[0] == 2:      return (1, gvals)
    return (0, vals)


def best_hand(seven):
    best = None
    for combo in itertools.combinations(seven, 5):
        score = eval_five(list(combo))
        if best is None or score > best:
            best = score
    return best


def new_game(guild_id):
    return {
        "guild_id":   guild_id,
        "phase":      "lobby",
        "players":    [],
        "deck":       [],
        "community":  [],
        "pot":        0,
        "current_bet":0,
        "turn_idx":   0,
        "dealer_idx": 0,
        "channel_id": None,
        "host_id":    None,
        "acted":      set(),
        "view_msg_id":None,
    }

def get_poker(gid):    return poker_games.get(gid)
def active_players(g): return [p for p in g["players"] if not p["folded"] and not p["allin"]]
def alive_players(g):  return [p for p in g["players"] if not p["folded"]]


def next_turn(g):
    n = len(g["players"])
    for _ in range(n):
        g["turn_idx"] = (g["turn_idx"] + 1) % n
        p = g["players"][g["turn_idx"]]
        if not p["folded"] and not p["allin"]:
            return


def all_acted(g):
    ap = active_players(g)
    if not ap:
        return True
    for p in ap:
        if p["id"] not in g["acted"]:
            return False
        if p["bet"] < g["current_bet"] and p["chips"] > 0:
            return False
    return True


def compute_pots(players):
    """Split the hand's chips into main/side pots from per-player totals.

    Returns ``[(amount, eligible_ids), ...]`` ordered main pot first. Each
    distinct all-in contribution level caps a tier: everyone pays in up to
    that level, and only non-folded players who reached the level can win it.
    Folded players' chips stay in the tiers they contributed to (dead money).
    A tier whose only reachers all folded is merged into the previous tier
    rather than orphaned. Pure — unit-tested in tests/test_poker.py."""
    contributors = [p for p in players if p.get("total_bet", 0) > 0]
    # Tier boundaries come from *non-folded* players only — a folded player's
    # contribution level caps nothing, it's just dead money poured into the
    # tiers below (splitting on it would print a pointless "side pot").
    levels = sorted({p["total_bet"] for p in contributors if not p["folded"]})
    pots: list[tuple[int, set]] = []
    prev = 0
    for lv in levels:
        amount = sum(max(0, min(p["total_bet"], lv) - prev) for p in contributors)
        eligible = {p["id"] for p in contributors
                    if p["total_bet"] >= lv and not p["folded"]}
        if amount > 0 and eligible:
            pots.append((amount, eligible))
        prev = lv
    # Folded players may have bet beyond the top surviving level — that excess
    # is dead money too; merge it into the last pot rather than orphaning it.
    leftover = sum(max(0, p["total_bet"] - prev) for p in contributors)
    if leftover > 0 and pots:
        pots[-1] = (pots[-1][0] + leftover, pots[-1][1])
    return pots


def build_game_embed(g, title="♠ Texas Hold'em"):
    phase_labels = {
        "lobby": "Lobby", "preflop": "Pre-Flop", "flop": "Flop",
        "turn": "Turn", "river": "River", "showdown": "Showdown",
    }
    embed = discord.Embed(title=title, color=discord.Color.green())
    embed.add_field(name="Community Cards",
                    value=hand_str(g["community"]) if g["community"] else "—",
                    inline=False)
    embed.add_field(name="Pot",   value=f"${g['pot']}",                          inline=True)
    embed.add_field(name="Phase", value=phase_labels.get(g["phase"], g["phase"]), inline=True)

    lines = []
    for i, p in enumerate(g["players"]):
        marker = " 👈" if (g["phase"] not in ("lobby", "showdown")
                          and i == g["turn_idx"]
                          and not p["folded"] and not p["allin"]) else ""
        status  = " *(folded)*" if p["folded"] else (" *(all-in)*" if p["allin"] else "")
        bet_str = f"  bet: ${p['bet']}" if p["bet"] > 0 else ""
        lines.append(f"**{p['name']}**{marker} — ${p['chips']}{bet_str}{status}")
    embed.add_field(name="Players", value="\n".join(lines) or "None", inline=False)

    if g["phase"] not in ("lobby", "showdown"):
        cp = g["players"][g["turn_idx"]] if 0 <= g["turn_idx"] < len(g["players"]) else None
        if cp and not cp["folded"] and not cp["allin"]:
            to_call = g["current_bet"] - cp["bet"]
            embed.set_footer(text=f"👆 {cp['name']}'s turn  •  to call: ${to_call}  •  pot: ${g['pot']}")
    return embed


class RaiseModal(discord.ui.Modal, title="Raise Amount"):
    amount = discord.ui.TextInput(
        label="Raise by how much?",
        placeholder=f"Minimum {POKER_BIG_BLIND}",
        min_length=1, max_length=6,
    )

    def __init__(self, cog, game):
        super().__init__()
        self.cog  = cog
        self.game = game

    async def on_submit(self, interaction):
        try:
            amt = int(self.amount.value)
        except ValueError:
            return await interaction.response.send_message("❌ Enter a valid number.", ephemeral=True)
        await self.cog.handle_action(interaction, "raise", amt)


class PokerView(discord.ui.View):
    def __init__(self, cog, game):
        super().__init__(timeout=None)
        self.cog  = cog
        self.game = game
        cp = game["players"][game["turn_idx"]] if 0 <= game["turn_idx"] < len(game["players"]) else None
        to_call = (game["current_bet"] - cp["bet"]) if cp else 0
        self.check_btn.disabled = to_call > 0
        self.call_btn.label     = f"Call ${to_call}" if to_call > 0 else "Call"
        self.call_btn.disabled  = to_call <= 0

    @discord.ui.button(label="Check", style=discord.ButtonStyle.secondary, emoji="✅")
    async def check_btn(self, i, _b):  await self.cog.handle_action(i, "check")

    @discord.ui.button(label="Call", style=discord.ButtonStyle.primary, emoji="📞")
    async def call_btn(self, i, _b):   await self.cog.handle_action(i, "call")

    @discord.ui.button(label="Raise", style=discord.ButtonStyle.success, emoji="📈")
    async def raise_btn(self, i, _b):
        g = get_poker(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        cp = g["players"][g["turn_idx"]] if 0 <= g["turn_idx"] < len(g["players"]) else None
        if not cp or cp["id"] != i.user.id:
            return await i.response.send_message("❌ Not your turn.", ephemeral=True)
        await i.response.send_modal(RaiseModal(self.cog, g))

    @discord.ui.button(label="All-In", style=discord.ButtonStyle.danger, emoji="🚀")
    async def allin_btn(self, i, _b):  await self.cog.handle_action(i, "allin")

    @discord.ui.button(label="Fold",  style=discord.ButtonStyle.danger, emoji="🃏")
    async def fold_btn(self, i, _b):   await self.cog.handle_action(i, "fold")

    @discord.ui.button(label="My Hand", style=discord.ButtonStyle.secondary, emoji="👁", row=1)
    async def hand_btn(self, i, _b):
        g = get_poker(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        p = next((p for p in g["players"] if p["id"] == i.user.id), None)
        if not p: return await i.response.send_message("❌ You're not in this game.", ephemeral=True)
        if not p["hand"]: return await i.response.send_message("❌ No cards yet.", ephemeral=True)
        comm   = hand_str(g["community"]) if g["community"] else "*(none yet)*"
        status = " *(folded)*" if p["folded"] else (" *(all-in)*" if p["allin"] else "")
        await i.response.send_message(
            f"**Your hand{status}:** {hand_str(p['hand'])}\n"
            f"**Community:** {comm}\n"
            f"**Your chips:** ${p['chips']}  •  **Pot:** ${g['pot']}",
            ephemeral=True,
        )


class LobbyView(discord.ui.View):
    def __init__(self, cog, game):
        super().__init__(timeout=None)
        self.cog  = cog
        self.game = game

    @discord.ui.button(label="Join Game", style=discord.ButtonStyle.success, emoji="✋")
    async def join(self, i, _b):
        g = get_poker(i.guild_id)
        if not g or g["phase"] != "lobby":
            return await i.response.send_message("❌ No lobby.", ephemeral=True)
        if any(p["id"] == i.user.id for p in g["players"]):
            return await i.response.send_message("❌ You're already in.", ephemeral=True)
        if len(g["players"]) >= 9:
            return await i.response.send_message("❌ Game is full.", ephemeral=True)
        g["players"].append({
            "id": i.user.id, "name": i.user.display_name,
            "chips": POKER_STARTING_CHIPS, "hand": [],
            "bet": 0, "folded": False, "allin": False,
        })
        await i.response.edit_message(embed=build_game_embed(g, "♠ Texas Hold'em — Lobby"), view=self)
        await i.followup.send(f"✅ **{i.user.display_name}** joined! (${POKER_STARTING_CHIPS} chips)")

    @discord.ui.button(label="Deal Cards", style=discord.ButtonStyle.primary, emoji="🃏")
    async def start(self, i, _b):
        g = get_poker(i.guild_id)
        if not g: return await i.response.send_message("❌ No game.", ephemeral=True)
        if g.get("host_id") != i.user.id:
            return await i.response.send_message("❌ Only the host can start.", ephemeral=True)
        if g["phase"] != "lobby":
            return await i.response.send_message("❌ Already started.", ephemeral=True)
        if len(g["players"]) < 2:
            return await i.response.send_message("❌ Need at least 2 players.", ephemeral=True)
        await i.response.defer()
        await self.cog.deal(g, i.channel)


class PlayAgainView(discord.ui.View):
    def __init__(self, cog, players, channel_id, host_id):
        super().__init__(timeout=60)
        self.cog        = cog
        self.players    = players
        self.channel_id = channel_id
        self.host_id    = host_id

    @discord.ui.button(label="Play Again (keep balances)", style=discord.ButtonStyle.success, emoji="♻️")
    async def again(self, i, _b):
        if i.user.id != self.host_id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        cont = [p for p in self.players if p["chips"] > 0]
        if len(cont) < 2:
            return await i.response.send_message("❌ Not enough players with chips.", ephemeral=True)
        g = new_game(i.guild_id)
        g["channel_id"] = self.channel_id
        g["host_id"]    = self.host_id
        g["players"] = [
            {"id": p["id"], "name": p["name"], "chips": p["chips"],
             "hand": [], "bet": 0, "folded": False, "allin": False} for p in cont
        ]
        poker_games[i.guild_id] = g
        self.stop()
        await i.response.edit_message(view=None)
        await self.cog.deal(g, self.cog.bot.get_channel(self.channel_id))

    @discord.ui.button(label="End Game", style=discord.ButtonStyle.danger, emoji="🛑")
    async def end(self, i, _b):
        if i.user.id != self.host_id:
            return await i.response.send_message("❌ Host only.", ephemeral=True)
        self.stop()
        await i.response.edit_message(content="🃏 Thanks for playing!", view=None)


class PokerCog(commands.Cog, name="Poker"):
    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def _track_view(g, view):
        """Stop the previous live view before registering a new one. Views with
        timeout=None stay registered in discord.py's view store until stopped,
        so every action prompt a hand posted used to linger for the process's
        lifetime. Pass None to just stop the current one (game over)."""
        old = g.get("_live_view")
        if old is not None:
            old.stop()
        g["_live_view"] = view
        return view

    @commands.group(name="poker", invoke_without_command=True)
    async def poker(self, ctx):
        await ctx.send(f"Use `{ctx.prefix}poker new` to create a lobby.")

    @poker.command(name="new")
    async def new(self, ctx):
        if ctx.guild.id in poker_games:
            return await ctx.send(f"❌ A game is running. Use `{ctx.prefix}poker end` first.")
        g = new_game(ctx.guild.id)
        g["channel_id"] = ctx.channel.id
        g["host_id"]    = ctx.author.id
        poker_games[ctx.guild.id] = g
        await ctx.send(
            f"♠ **{ctx.author.display_name}** opened a poker game!",
            embed=build_game_embed(g, "♠ Texas Hold'em — Lobby"),
            view=self._track_view(g, LobbyView(self, g)),
        )

    @poker.command(name="hand")
    async def hand(self, ctx):
        g = get_poker(ctx.guild.id)
        if not g or g["phase"] == "lobby":
            return await ctx.send("❌ No active game.")
        p = next((p for p in g["players"] if p["id"] == ctx.author.id), None)
        if not p: return await ctx.send("❌ You're not in this game.")
        try:
            await ctx.author.send(f"♠ **Your hand:** {hand_str(p['hand'])}")
            await ctx.send("📨 Sent to your DMs.", delete_after=5)
        except discord.Forbidden:
            await ctx.send(f"❌ Couldn't DM you. (Others can see!) {hand_str(p['hand'])}")

    @poker.command(name="end")
    async def end(self, ctx):
        g = get_poker(ctx.guild.id)
        if not g: return await ctx.send("❌ No game running.")
        if g.get("host_id") != ctx.author.id:
            return await ctx.send("❌ Only the host can end the game.")
        self._track_view(g, None)
        poker_games.pop(ctx.guild.id, None)
        await ctx.send("🃏 Game ended.")

    async def deal(self, g, channel):
        g["deck"] = make_deck()
        random.shuffle(g["deck"])
        n = len(g["players"])
        for p in g["players"]:
            p["hand"]      = [g["deck"].pop(), g["deck"].pop()]
            p["bet"]       = 0
            p["total_bet"] = 0          # whole-hand contribution — drives side pots
            p["folded"] = p["allin"] = False
        sb = (g["dealer_idx"] + 1) % n
        bb = (g["dealer_idx"] + 2) % n
        # Blinds are capped at the player's stack (a short stack posting a
        # blind used to go chip-negative) — a capped blind is simply all-in.
        sb_p, bb_p = g["players"][sb], g["players"][bb]
        sb_amt = min(POKER_SMALL_BLIND, sb_p["chips"])
        bb_amt = min(POKER_BIG_BLIND, bb_p["chips"])
        sb_p["chips"] -= sb_amt
        sb_p["bet"] = sb_p["total_bet"] = sb_amt
        bb_p["chips"] -= bb_amt
        bb_p["bet"] = bb_p["total_bet"] = bb_amt
        if sb_p["chips"] == 0: sb_p["allin"] = True
        if bb_p["chips"] == 0: bb_p["allin"] = True
        g["pot"]         = sb_amt + bb_amt
        g["current_bet"] = POKER_BIG_BLIND
        g["turn_idx"]    = (bb + 1) % n
        g["phase"]       = "preflop"
        g["acted"]       = set()
        g["community"]   = []

        for p in g["players"]:
            user = self.bot.get_user(p["id"])
            if user:
                try:
                    await user.send(
                        f"♠ **Your hand:**\n```{hand_str(p['hand'])}```\n"
                        "*Community cards will be posted in the channel.*"
                    )
                except discord.Forbidden:
                    pass

        cp = g["players"][g["turn_idx"]]
        msg = await channel.send(
            f"🃏 Cards dealt! Blinds: SB ${POKER_SMALL_BLIND} / BB ${POKER_BIG_BLIND}\n"
            f"**{cp['name']}**'s turn.",
            embed=build_game_embed(g, "♠ Texas Hold'em — Pre-Flop"),
            view=self._track_view(g, PokerView(self, g)),
        )
        g["view_msg_id"] = msg.id

    async def advance_street(self, g, channel):
        for p in g["players"]:
            p["bet"] = 0
        g["current_bet"] = 0
        g["acted"]       = set()
        n   = len(g["players"])
        idx = (g["dealer_idx"] + 1) % n
        for _ in range(n):
            p = g["players"][idx]
            if not p["folded"] and not p["allin"]:
                g["turn_idx"] = idx
                break
            idx = (idx + 1) % n

        remaining = alive_players(g)
        if len(remaining) == 1:
            winner = remaining[0]
            winner["chips"] += g["pot"]
            lines = [f"🏆 **{winner['name']}** wins **${g['pot']}** — everyone else folded!"]
            if g["community"]:
                lines.append(f"**Community:** {hand_str(g['community'])}")
            lines.append("**All hands:**")
            for p in g["players"]:
                if p["hand"]:
                    seven = p["hand"] + g["community"]
                    score = best_hand(seven) if len(seven) >= 5 else None
                    label = f"*{HAND_NAMES[score[0]]}*" if score else ""
                    fld   = " *(folded)*" if p["folded"] else ""
                    lines.append(f"  **{p['name']}**{fld}: {hand_str(p['hand'])} {label}")
            await channel.send("\n".join(lines))
            await self.finish(g, channel)
            return

        # Betting is impossible when fewer than two players can still act
        # (everyone else is all-in) — run out the rest of the board and go
        # straight to showdown. The old behaviour dealt one street at a time
        # and posted action buttons nobody could press, soft-locking the hand
        # until the all-in player spammed a $0 call.
        if len(active_players(g)) <= 1:
            dealt = False
            if len(g["community"]) == 0:
                g["community"] += [g["deck"].pop() for _ in range(3)]
                dealt = True
            if len(g["community"]) == 3:
                g["community"].append(g["deck"].pop())
                dealt = True
            if len(g["community"]) == 4:
                g["community"].append(g["deck"].pop())
                dealt = True
            g["phase"] = "river"
            if dealt:
                await channel.send(
                    f"🚀 All-in — running out the board: {hand_str(g['community'])}"
                )
            await self.showdown(g, channel)
            return

        if g["phase"] == "preflop":
            g["community"] += [g["deck"].pop() for _ in range(3)]
            g["phase"] = "flop";  label = "Flop"
        elif g["phase"] == "flop":
            g["community"].append(g["deck"].pop())
            g["phase"] = "turn";  label = "Turn"
        elif g["phase"] == "turn":
            g["community"].append(g["deck"].pop())
            g["phase"] = "river"; label = "River"
        else:
            await self.showdown(g, channel)
            return

        cp  = g["players"][g["turn_idx"]]
        msg = await channel.send(
            f"**{label}:** {hand_str(g['community'])}\n**{cp['name']}**'s turn.",
            embed=build_game_embed(g, f"♠ Texas Hold'em — {label}"),
            view=self._track_view(g, PokerView(self, g)),
        )
        g["view_msg_id"] = msg.id

    async def showdown(self, g, channel):
        contenders = alive_players(g)
        for p in contenders:
            score = best_hand(p["hand"] + g["community"])
            p["_score"]     = score
            p["_hand_name"] = HAND_NAMES[score[0]]

        lines = [f"**Community:** {hand_str(g['community'])}\n"]
        for p in sorted(contenders, key=lambda x: x["_score"], reverse=True):
            lines.append(f"**{p['name']}**: {hand_str(p['hand'])} — *{p['_hand_name']}*")
        lines.append("")

        # Pay out per pot tier. An all-in player can only win as much from
        # each opponent as they put in themselves; anything above that forms
        # side pots contested only by the deeper stacks. (The old code paid
        # the entire pot to the best hand, letting a $50 all-in scoop chips
        # they never covered.)
        pots = compute_pots(g["players"])
        if not pots:
            # Defensive: total_bet data missing (e.g. cog reloaded mid-hand)
            # — fall back to the old single-pot behaviour so chips never vanish.
            pots = [(g["pot"], {p["id"] for p in contenders})]
        by_id = {p["id"]: p for p in contenders}
        for idx, (amount, eligible_ids) in enumerate(pots):
            elig = [by_id[i] for i in eligible_ids if i in by_id]
            if not elig:
                elig = contenders          # defensive — shouldn't happen
            best    = max(p["_score"] for p in elig)
            winners = [p for p in elig if p["_score"] == best]
            share, odd = divmod(amount, len(winners))
            for i, w in enumerate(winners):
                w["chips"] += share + (odd if i == 0 else 0)
            label = ("the pot" if len(pots) == 1
                     else "the main pot" if idx == 0
                     else f"side pot {idx}")
            if len(winners) == 1:
                lines.append(f"🏆 **{winners[0]['name']}** wins **${amount}** "
                             f"({label}) with *{winners[0]['_hand_name']}*!")
            else:
                names = ", ".join(w["name"] for w in winners)
                lines.append(f"🤝 {names} split {label} — **${share}** each.")

        await channel.send("\n".join(lines))
        await self.finish(g, channel)

    async def finish(self, g, channel):
        lines = ["**📊 Chip counts:**"]
        for p in sorted(g["players"], key=lambda x: x["chips"], reverse=True):
            lines.append(f"  {p['name']}: **${p['chips']}**" + (" *(broke)*" if p["chips"] == 0 else ""))
        broke = [p["name"] for p in g["players"] if p["chips"] == 0]
        if broke:
            lines.append(f"\n💸 {', '.join(broke)} {'is' if len(broke)==1 else 'are'} out of chips.")
        self._track_view(g, None)
        poker_games.pop(g["guild_id"], None)
        view = PlayAgainView(self, g["players"], g["channel_id"], g["host_id"])
        await channel.send("\n".join(lines) + "\n\nHost: play again or end?", view=view)

    async def handle_action(self, interaction, action, amount=0):
        g = get_poker(interaction.guild_id)
        if not g or g["phase"] in ("lobby", "showdown"):
            return await interaction.response.send_message("❌ No active round.", ephemeral=True)
        cp = g["players"][g["turn_idx"]] if 0 <= g["turn_idx"] < len(g["players"]) else None
        if not cp:
            return await interaction.response.send_message("❌ No active player.", ephemeral=True)
        if cp["id"] != interaction.user.id:
            return await interaction.response.send_message(
                f"❌ It's **{cp['name']}'s** turn.", ephemeral=True
            )

        to_call = g["current_bet"] - cp["bet"]
        msg_out = ""

        if action == "fold":
            cp["folded"] = True
            g["acted"].add(cp["id"])
            msg_out = f"🃏 **{cp['name']}** folded."
        elif action == "check":
            if to_call > 0:
                return await interaction.response.send_message(
                    f"❌ Can't check — bet is ${g['current_bet']}.", ephemeral=True
                )
            g["acted"].add(cp["id"])
            msg_out = f"✅ **{cp['name']}** checked."
        elif action == "call":
            actual = min(to_call, cp["chips"])
            cp["chips"]     -= actual
            cp["bet"]       += actual
            cp["total_bet"]  = cp.get("total_bet", 0) + actual
            g["pot"]        += actual
            if cp["chips"] == 0:
                cp["allin"] = True
                msg_out = f"💰 **{cp['name']}** called ${actual} — **all-in**!"
            else:
                msg_out = f"✅ **{cp['name']}** called ${actual}."
            g["acted"].add(cp["id"])
        elif action == "raise":
            total_bet = g["current_bet"] + amount
            extra     = total_bet - cp["bet"]
            if amount < POKER_BIG_BLIND:
                return await interaction.response.send_message(
                    f"❌ Minimum raise is ${POKER_BIG_BLIND}.", ephemeral=True
                )
            if extra > cp["chips"]:
                return await interaction.response.send_message(
                    f"❌ Not enough chips (have ${cp['chips']}).", ephemeral=True
                )
            cp["chips"]      -= extra
            cp["bet"]         = total_bet
            cp["total_bet"]   = cp.get("total_bet", 0) + extra
            g["pot"]         += extra
            g["current_bet"]  = total_bet
            g["acted"]        = {cp["id"]}
            msg_out = f"📈 **{cp['name']}** raised to **${total_bet}**."
        elif action == "allin":
            amt = cp["chips"]
            cp["bet"]       += amt
            cp["total_bet"]  = cp.get("total_bet", 0) + amt
            g["pot"]        += amt
            cp["chips"]      = 0
            cp["allin"]      = True
            if cp["bet"] > g["current_bet"]:
                g["current_bet"] = cp["bet"]
                g["acted"]       = {cp["id"]}
            else:
                g["acted"].add(cp["id"])
            msg_out = f"🚀 **{cp['name']}** is **ALL-IN** for ${cp['bet']}!"

        channel = self.bot.get_channel(g["channel_id"])

        if g.get("view_msg_id"):
            try:
                old = await channel.fetch_message(g["view_msg_id"])
                await old.edit(view=None)
            except Exception:
                pass

        if all_acted(g) or len(active_players(g)) == 0:
            if not interaction.response.is_done():
                await interaction.response.send_message(msg_out)
            else:
                await channel.send(msg_out)
            await self.advance_street(g, channel)
            return

        next_turn(g)
        np = g["players"][g["turn_idx"]]
        to_next = g["current_bet"] - np["bet"]
        if not interaction.response.is_done():
            await interaction.response.send_message(msg_out)
        else:
            await channel.send(msg_out)
        msg = await channel.send(
            f"**{np['name']}**'s turn.  (to call: ${to_next}  •  pot: ${g['pot']})",
            embed=build_game_embed(g),
            view=self._track_view(g, PokerView(self, g)),
        )
        g["view_msg_id"] = msg.id


async def setup(bot):
    await bot.add_cog(PokerCog(bot))
