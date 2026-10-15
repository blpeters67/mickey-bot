"""
What the voice DJ can do: the capability registry.

Every operation a spoken request can turn into is declared here once — its
arguments, what it means, whether it adds songs or needs a lookup first,
whether it needs a spoken "yes" — and the planner's instructions
(core/voice_llm.py) are written from these declarations, so the model is told
exactly what exists. The executor (core/voice_exec.py) runs each one through
the same code the typed commands use:

  * queue and player operations → core/queue_ops.py (shared with
    cogs/music.py) and the voice cog's control path;
  * songs → the voice cog's checked lookups (catalogue, then a verified
    upload);
  * everything else → the typed command itself ("bridged"): looked up by its
    registered name, its own checks run (`can_run`: cog checks, permission
    decorators, global checks — the same as typing it), called with typed,
    validated arguments through a voice context whose replies go to the
    voice reply channel. Never a command string, never a shell.

NOT_BY_VOICE gives, for every remaining command, the reason it isn't run by
voice (it needs a file, buttons, typed text, admin rights…). Asked for one of
those, the planner uses `typed_only`, and the bot says how to do it instead of
pretending it ran. tests/test_voice_tools.py loads every cog and fails when a
command is neither covered nor excused — and checks each bridge against the
command's real parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpSpec:
    name: str
    say: str                          # what it does, in a few words (for messages)
    doc: str                          # for the planner: arguments and meaning
    kind: str                         # song | queue | player | info | command
    required: tuple[str, ...] = ()    # step fields that must be filled in
    refs: dict = field(default_factory=dict)   # field → "ref" | "dest"
    adds: bool = False                # puts new entries in the queue
    lookup: bool = False              # needs a network lookup before it can act
    confirm: bool = False             # always needs a spoken yes first
    command: str = ""                 # bridged: the qualified command name it invokes
    #: bridged: command parameter ← step field ("value", "title", "count"…)
    params: dict = field(default_factory=dict)
    #: bridged with a choice: step.value → qualified command name (the one
    #: run); `command` is then only the default.
    choices: dict = field(default_factory=dict)
    covers: tuple[str, ...] = ()      # typed commands this is the voice version of


_SONG = ('title = the song\'s real title, artist = who performs it; heard_title / '
         'heard_artist = the title and the words after "by" exactly as the speech-to-text '
         'wrote them ("" if not said)')

OPS: dict[str, OpSpec] = {s.name: s for s in [
    # ── adding songs ─────────────────────────────────────────────────────────
    OpSpec("add_song", "queue a song",
           f"one specific song. {_SONG}. count = how many separate copies (\"five times\" "
           "is 5). when: \"\" end of the queue, \"next\" right after the current song, "
           "\"now\" instead of the current song (no separate skip needed).",
           "song", ("title",), adds=True, lookup=True, covers=("play", "playnext")),
    OpSpec("find_song", "find a described song",
           "ONE particular song they describe but don't name, when you aren't sure which it is "
           "(\"that Yung Gravy song that samples Panic! At The Disco\"): value = what to look "
           "up, artist if known. when as add_song. Never for \"some songs\" or \"random "
           "songs\" — you pick those yourself as add_song steps.",
           "song", ("value",), adds=True, lookup=True),
    OpSpec("add_album", "queue songs from an album",
           "songs from one album: album = its title, artist, tracks = which ones: \"all\", "
           "\"1-3\", \"7, 9\", \"first 3\", \"last 2\", \"random 3\", \"first half\", "
           "\"second half\", \"all except last\", \"all except 2, 5\", \"first 7 except 6\", "
           "\"first half except last\" (numbers are album track numbers; first/last/halves "
           "after except count within what's being trimmed). count = how many times; when "
           "as add_song.",
           "song", ("album",), adds=True, lookup=True),
    OpSpec("add_top", "queue an artist's most popular songs",
           "artist, count = how many (default 5).", "song", ("artist",), adds=True, lookup=True),
    OpSpec("add_latest", "queue an artist's newest songs",
           "artist, count = how many (default 1). For \"new\"/\"latest\" songs you may not "
           "know.", "song", ("artist",), adds=True, lookup=True),
    OpSpec("search", "search YouTube and let them pick",
           "value = what to search for: posts the top results with buttons to pick one "
           "(for \"search for …\", \"show me results for …\").", "song", ("value",),
           lookup=True, command="search", params={"query": "value"}, covers=("search",)),
    OpSpec("play_likes", "play their liked songs", "queues the songs they liked (value = "
           "\"shuffle\" to shuffle them).", "song", lookup=True, command="likes play",
           params={"flags": "value"}, covers=("likes play",)),
    OpSpec("play_playlist", "play a saved playlist", "value = the playlist's name.",
           "song", ("value",), lookup=True, command="playlist play", params={"name": "value"},
           covers=("playlist play",)),
    # ── the queue ────────────────────────────────────────────────────────────
    OpSpec("remove", "remove songs from the queue",
           "target = which queued song(s). \"Skip/remove the second song in the queue\" is "
           "remove pos:2 (not a skip of the current song).",
           "queue", ("target",), refs={"target": "ref"}, covers=("remove",)),
    OpSpec("move", "move a song in the queue",
           "target = which queued song, to = where (front, end, pos:N, after:REF, "
           "before:REF). \"Play X next\" for a song already queued is move to front.",
           "queue", ("target", "to"), refs={"target": "ref", "to": "dest"}, covers=("move",)),
    OpSpec("swap", "swap two queued songs", "target and to = the two queued songs.",
           "queue", ("target", "to"), refs={"target": "ref", "to": "ref"}, covers=("swap",)),
    OpSpec("skipto", "jump to a queued song",
           "target = a queued song: everything before it is dropped and it plays now.",
           "queue", ("target",), refs={"target": "ref"}, covers=("skipto",)),
    OpSpec("shuffle", "shuffle the queue", "shuffles the upcoming queue (not the current song).",
           "queue", covers=("shuffle",)),
    OpSpec("reverse", "reverse the queue", "reverses the upcoming queue.", "queue",
           covers=("reverse",)),
    OpSpec("balance", "alternate the queue between requesters",
           "round-robins the queue between the people who requested songs.", "queue",
           covers=("balance",)),
    OpSpec("dedupe", "remove duplicate songs",
           "removes repeats of songs already queued or playing.", "queue", covers=("dedupe",)),
    OpSpec("clear", "clear the queue", "empties the upcoming queue (the current song keeps "
           "playing).", "queue", confirm=True, covers=("clear",)),
    OpSpec("save_queue", "save the queue as a playlist", "value = the playlist's name.",
           "queue", ("value",), command="playlist savequeue", params={"name": "value"},
           covers=("playlist savequeue",)),
    # ── the player ───────────────────────────────────────────────────────────
    OpSpec("skip", "skip the current song",
           "skips the song playing now (\"skip this\", \"next song\", \"skip to the next "
           "one\"); count = how many songs (\"skip two songs\": this one and the next). "
           "\"Skip the next song\" / \"the next three songs\" are upcoming songs: remove "
           "pos:1 / range:1-3, and the current one keeps playing.", "player", covers=("skip",)),
    OpSpec("pause", "pause", "pauses the music.", "player", covers=("pause",)),
    OpSpec("resume", "resume", "resumes paused music.", "player", covers=("resume",)),
    OpSpec("stop", "stop the music and clear the queue",
           "ends the music and empties the queue (for a short break use pause).",
           "player", confirm=True, covers=("stop",)),
    OpSpec("restart", "restart the song", "this song from the start.", "player"),
    OpSpec("seek", "jump within the song",
           "value = where: \"1:30\", \"90\" (seconds), \"+30\" forward, \"-15\" back.",
           "player", ("value",), lookup=True, command="seek", params={"position": "value"},
           covers=("seek",)),
    OpSpec("volume", "change the volume",
           "value = a level 0-200 (\"set it to 50\" is \"50\"), a change (\"turn it up by "
           "10%\" is \"+10\", \"down by 20\" is \"-20\"), or \"up\" / \"down\" for a "
           "step.", "player", ("value",), covers=("volume",)),
    OpSpec("loop", "loop", "value = \"song\" (the current song), \"queue\" (everything, "
           "e.g. \"loop them all\", \"loop the rest\") or \"off\".", "player", covers=("loop",)),
    OpSpec("back", "replay the previous song", "plays the previous song again now.",
           "player", lookup=True, command="back", covers=("back",)),
    OpSpec("filter", "set an audio filter",
           "value = bassboost, nightcore, slowed, 8d, tremolo, vibrato, karaoke, "
           "\"speed 1.5\" or off.", "player", ("value",), lookup=True, command="filter",
           params={"name": "value"}, covers=("filter",)),
    OpSpec("autoplay", "turn autoplay on or off",
           "value = on or off: keep playing related songs when the queue runs out.",
           "player", ("value",), command="autoplay", params={"value": "value"},
           covers=("autoplay",)),
    OpSpec("sponsorblock", "turn SponsorBlock on or off",
           "value = on or off: skip sponsor segments in videos.", "player", ("value",),
           command="sponsorblock", params={"value": "value"}, covers=("sponsorblock",)),
    OpSpec("stop_talking", "stop talking", "stops the bot's text-to-speech mid-sentence.",
           "player", command="ttsstop", covers=("ttsstop",)),
    # ── questions about what's going on ──────────────────────────────────────
    OpSpec("now_playing", "say what's playing", "tells them what's playing now.", "info",
           covers=("nowplaying",)),
    OpSpec("queue_info", "say what's in the queue",
           "tells them what's queued (count = how many upcoming songs to name).", "info",
           covers=("queue",)),
    OpSpec("history", "say what played recently", "the songs played recently.", "info",
           covers=("history",)),
    OpSpec("answer", "answer a question",
           "value = your spoken answer to anything else they asked (one to three short "
           "sentences). Facts about the queue and the music come from the state shown "
           "below — or use now_playing / queue_info, which read it live.", "info", ("value",)),
    # ── other cogs' commands, run as they are ────────────────────────────────
    OpSpec("lyrics", "post the lyrics", "posts the lyrics of the current song (value = "
           "another song to look up, or \"\").", "command", lookup=True, command="lyrics",
           params={"query": "value"}, covers=("lyrics",)),
    OpSpec("like", "like the song", "saves the current song (or value = a song) to their "
           "liked songs.", "command", lookup=True, command="like", params={"query": "value"},
           covers=("like",)),
    OpSpec("unlike", "unlike the song", "removes the current song from their likes.",
           "command", command="unlike", covers=("unlike",)),
    OpSpec("likes", "list their liked songs", "posts the songs they liked.", "command",
           command="likes", covers=("likes",)),
    OpSpec("playlists", "list the saved playlists", "posts the saved playlists.", "command",
           command="playlist list", covers=("playlist list", "playlist")),
    OpSpec("playlist_info", "show a saved playlist", "value = the playlist's name: posts its "
           "songs.", "command", ("value",), command="playlist info", params={"name": "value"},
           covers=("playlist info",)),
    OpSpec("clip", "clip the last moments", "posts a clip of the last few seconds of "
           "voice chat (value = how many seconds, \"\" for the default).", "command",
           lookup=True, command="clip", params={"options": "value"}, covers=("clip",)),
    OpSpec("remind", "set a reminder",
           "value = when (\"in 10 minutes\", \"at 5pm\", \"every day at 9\"), title = what "
           "to remind them of.", "command", ("value",), command="remindme",
           params={"when": "value", "message": "title"}, covers=("remindme",)),
    OpSpec("reminders", "list their reminders", "posts their reminders.", "command",
           command="reminders", covers=("reminders",)),
    OpSpec("cancel_reminder", "cancel a reminder", "count = the reminder's number (from "
           "their list).", "command", ("count",), command="reminders cancel",
           params={"reminder_id": "count"}, covers=("reminders cancel",)),
    OpSpec("time", "tell the time", "value = a place or person (\"\" for theirs).", "command",
           command="time", params={"target": "value"}, covers=("time",)),
    OpSpec("timestamp", "make a time everyone can read", "value = a time (\"5pm tomorrow\"):"
           " posts it as a timestamp everyone sees in their own zone.", "command", ("value",),
           command="when", params={"moment": "value"}, covers=("when",)),
    OpSpec("set_timezone", "set their timezone", "value = a timezone or city (\"Eastern\", "
           "\"London\").", "command", ("value",), command="timezone", params={"zone": "value"},
           covers=("timezone",)),
    OpSpec("top_songs", "show the server's top songs", "value = a time window "
           "(\"day\", \"week\", \"month\", \"year\", \"all\" or \"\").", "command",
           command="topsongs", params={"window": "value"}, covers=("topsongs",)),
    OpSpec("top_artists", "show the server's top artists", "value = a time window.",
           "command", command="topartists", params={"window": "value"}, covers=("topartists",)),
    OpSpec("top_listeners", "show the server's top listeners", "value = a time window.",
           "command", command="toplisteners", params={"window": "value"},
           covers=("toplisteners",)),
    OpSpec("wrapped", "show their music wrapped", "value = options (\"\" for theirs).",
           "command", command="wrapped", params={"args": "value"}, covers=("wrapped",)),
    OpSpec("stats", "show statistics",
           "value = \"me\" (their message stats), \"server\" (top senders and channels) or "
           "\"voice\" (time in voice channels).", "command", command="stats user",
           choices={"me": "stats user", "": "stats user", "server": "stats server",
                    "voice": "stats voice"},
           covers=("stats user", "stats server", "stats voice", "stats")),
    OpSpec("quiz_start", "start a music quiz", "count = rounds.", "command",
           command="quiz start", params={"rounds": "count"}, covers=("quiz start", "quiz")),
    OpSpec("quiz_stop", "stop the music quiz", "ends the quiz.", "command",
           command="quiz stop", covers=("quiz stop",)),
    OpSpec("quiz_scores", "show the quiz leaderboard", "posts the all-time quiz scores.",
           "command", command="quiz leaderboard", covers=("quiz leaderboard",)),
    OpSpec("game", "start a game",
           "value = blackjack, poker or yahtzee: starts a new game in the reply channel "
           "(played with its buttons).", "command", ("value",), command="blackjack new",
           choices={"blackjack": "blackjack new", "poker": "poker new",
                    "yahtzee": "yahtzee new"},
           covers=("blackjack new", "poker new", "yahtzee new", "blackjack", "poker",
                   "yahtzee")),
    OpSpec("karaoke", "karaoke mode", "posts the current song (or value = a song) with the "
           "vocals taken out.", "command", lookup=True, command="karaoke",
           params={"query": "value"}, covers=("karaoke",)),
    OpSpec("download", "post the song as a file", "value = the song or link to post as a "
           "downloadable file (the current song if they say \"this\").", "command", ("value",),
           lookup=True, command="download", params={"query": "value"}, covers=("download",)),
    OpSpec("set_tts_voice", "change their TTS voice", "value = a voice name (\"ava\", "
           "\"reset\"), or \"\" to list them.", "command", lookup=True, command="ttsvoice",
           params={"voice": "value"}, covers=("ttsvoice",)),
    OpSpec("voice_status", "show the voice-control settings", "posts the voice-control "
           "status.", "command", command="voice status", covers=("voice status", "voice")),
    OpSpec("lastfm_status", "show their Last.fm link", "posts their Last.fm status.",
           "command", command="lastfm status", covers=("lastfm status", "lastfm")),
    OpSpec("help", "show the help", "value = a command or section (\"\" for the overview).",
           "command", command="help", params={"query": "value"}, covers=("help",)),
    OpSpec("invite", "post the invite link", "posts the link to add the bot to a server.",
           "command", command="invite", covers=("invite",)),
    OpSpec("stop_listening", "stop listening to them",
           "opts them out of voice control (they can opt back in with the voice optin "
           "command).", "command", command="voice optout", covers=("voice optout",)),
    # ── the rest: said, not pretended ────────────────────────────────────────
    OpSpec("typed_only", "explain how to type it",
           "they asked for something that has to be typed (see the typed-only list below): "
           "value = the command's name. I tell them how instead of doing nothing.",
           "info", ("value",)),
]}


#: Operations that take effect immediately and touch nothing but the player
#: (no lookups, no queue positions): a request made only of these doesn't wait
#: behind other people's song searches.
INSTANT = frozenset({"skip", "pause", "resume", "volume", "loop", "restart",
                     "now_playing", "queue_info", "history", "answer", "typed_only"})


# ── coverage: every command, by voice or why not ──────────────────────────────

_ATTACH = "needs an attached file or a link"
_ADMIN = "server configuration — typed by someone with Manage Server, not by voice"
_OWNER = "the bot owner's maintenance tool — never by voice"
_TEXT = "a typed-text tool (needs the text to work on)"
_SETUP = "per-person account setup — typed once"
_PLAYLIST_EDIT = "editing saved playlists (names and links) — typed"
_HERE = "you can only talk to me when I'm already in your channel"
_GAME_MOVE = "a game's own moves are its buttons, not commands"

#: command qualified name → reason it isn't run by voice.
NOT_BY_VOICE: dict[str, str] = {
    # music
    "join": _HERE,
    "leave": "disconnecting would end voice control mid-sentence — say stop, or type leave",
    "rejoin": "rebuilds the voice connection voice control runs on — type it",
    # downloads / media tools
    "chapters": _ATTACH, "convert": _ATTACH, "desilence": _ATTACH, "normalize": _ATTACH,
    "ocr": _ATTACH, "resize": _ATTACH, "share": _ATTACH, "snip": _ATTACH, "speed": _ATTACH,
    "transcode": _ATTACH, "transcribe": _ATTACH, "transcribemodel": _ADMIN,
    "translate": _TEXT,
    # images
    **{c: _ATTACH for c in ("blur", "brightness", "caption", "contrast", "crop", "flip",
                            "grayscale", "invert", "nobg", "rotate", "scale", "sharpen",
                            "shrink", "thumb", "watermark", "qr")},
    # emoji / stickers
    "emoji": _ADMIN, "emoji add": _ADMIN, "emoji remove": _ADMIN,
    "sticker": _ADMIN, "sticker add": _ADMIN, "sticker remove": _ADMIN,
    # games
    **{c: _GAME_MOVE for c in ("blackjack end", "poker end", "poker hand", "yahtzee end")},
    "quiz exclude": "marks a song for the quiz by link — typed",
    "quiz include": "marks a song for the quiz by link — typed",
    # playlists (editing)
    **{f"playlist {c}": _PLAYLIST_EDIT for c in ("add", "addmany", "delete", "import",
                                                 "new", "remove", "rename")},
    # last.fm
    **{c: _SETUP for c in ("lastfm confirm", "lastfm link", "lastfm off", "lastfm on",
                           "lastfm unlink")},
    # stats
    "stats channel": "a channel's statistics — typed with the channel",
    "stats emoji": "emoji statistics — typed", "stats sticker": "sticker statistics — typed",
    "stats scan": _ADMIN,
    # starboard / admin / maintenance
    **{c: _ADMIN for c in ("starboard", "starboard channel", "starboard disable",
                           "starboard emoji", "starboard show", "starboard threshold",
                           "autojoin", "autoleave", "autoleave alone", "autoleave idle",
                           "autoleave reset", "autoleave show", "prefix")},
    **{c: _OWNER for c in ("console", "restart", "update", "versions", "ytdiag")},
    # message context menus (right-click a message → Apps): they act on that message
    **{f"menu:{c}": "a right-click menu on a message — it needs the message"
       for c in ("Transcribe", "Extract text", "Convert format", "Re-encode codec")},
    # tts
    "tts": "you're already talking — this is for people without a microphone",
    # voice control's own settings
    **{f"voice {s}": _ADMIN for s in ("ai", "channel", "cue", "debug", "default", "diag",
                                      "disable", "enable", "model", "wake", "wake add",
                                      "wake clear", "wake remove", "wake set", "doctor")},
    **{f"voice {s}": _SETUP for s in ("optin", "feedback", "clip")},
}


def coverage() -> dict[str, str]:
    """command qualified name → the voice operation covering it (the op
    name), for every command some operation covers."""
    out: dict[str, str] = {}
    for spec in OPS.values():
        for c in spec.covers:
            out.setdefault(c, spec.name)
    return out


def coverage_table(commands) -> list[tuple[str, str, str]]:
    """(command, "voice: <op>" | "not by voice", reason/meaning) for every
    command in `commands` (qualified names), sorted. A command neither
    covered nor excused shows "UNCLASSIFIED" (the test fails on that)."""
    cov = coverage()
    rows = []
    for name in sorted(set(commands)):
        if name in cov:
            spec = OPS[cov[name]]
            rows.append((name, f"voice: {spec.name}", spec.say))
        elif name in NOT_BY_VOICE:
            rows.append((name, "not by voice", NOT_BY_VOICE[name]))
        else:
            rows.append((name, "UNCLASSIFIED", ""))
    return rows


def typed_only_reason(name: str) -> str | None:
    """Why `name` (a command) is typed rather than said, or None."""
    return NOT_BY_VOICE.get(" ".join((name or "").lower().lstrip("-/!").split()))


def planner_ops_text() -> str:
    """The operation list for the planner's instructions."""
    lines = []
    for kind, title in (("song", "Adding songs"), ("queue", "The queue"),
                        ("player", "The player"), ("info", "Questions"),
                        ("command", "Other things")):
        lines.append(f"{title}:")
        for spec in OPS.values():
            if spec.kind == kind:
                extra = " (they must say yes first)" if spec.confirm else ""
                lines.append(f"  {spec.name}: {spec.doc}{extra}")
    groups: dict[str, list[str]] = {}
    for name, why in NOT_BY_VOICE.items():
        groups.setdefault(why, []).append(name)
    lines.append("Typed-only commands (use typed_only): " + "; ".join(
        f"{', '.join(sorted(names))} ({why})" for why, names in groups.items()))
    return "\n".join(lines)
