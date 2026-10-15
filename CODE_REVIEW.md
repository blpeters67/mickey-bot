# Code review — July 17, 2026

Full pass over bot.py, core/, all 18 cogs, and the test suite. Baseline health is good: all 149 tests pass, every file compiles, and the architecture (lazy stream resolution, playback generations, per-guild state) is sound. The issues below are ordered by how likely they are to be causing the problems you've been seeing.

**Update (July 18): items 1–6 are now FIXED in the code, per your go-ahead.** Files touched: playlists.py, favorites.py, musicquiz.py (voice connection), voicedj.py (wake rendering), poker.py (all-in runout, side pots + 8 new unit tests in tests/test_poker.py, blind capping), blackjack.py (natural-vs-21, view cleanup), yahtzee.py (view cleanup), downloads.py (progress-race aclose calls, view cleanup), serverstats.py (view cleanup), admin.py (error-handler consolidation), core/state.py (unused import). Everything compiles, pyflakes is clean apart from the intentional `faster_whisper` probes, and the full test suite passes (149 original + 8 new). Items 7–8 remain as notes.

## 1. HIGH — FIXED — Bot joins voice "broken" when summoned by `playlist play`, `likes play`, or `quiz start`

`cogs/music.py` connects with `voice_recv.VoiceRecvClient` and attaches the rolling recorder, but three other cogs have their own copy-pasted `_connect_to_author` that connects with the plain default client and never attaches the recorder:

- `cogs/playlists.py:33` — `ctx.author.voice.channel.connect()`
- `cogs/favorites.py:53` — same
- `cogs/games/musicquiz.py:320` — same

Consequence: whenever the bot's first entry into voice happens via `-playlist play` or `-likes play`, that entire session has no voice buffer. `-clip` replies "I'm not recording any voice chat right now" and "hey dj" voice control is silently dead. Worse, it's sticky: a later `-play` reuses the existing (wrong-class) voice client, and `Recording.attach` just logs a warning and gives up. It only heals after a full disconnect and a reconnect via `-play`/`-join`. This exactly matches "features randomly stop working depending on the day."

Fix: delete the duplicate helpers and have playlists/favorites call the music cog's `_connect_to_author` (or move it into a shared module), so every path connects with `VoiceRecvClient` and calls `Recording.attach`. The quiz's temporary connection should use the same class too.

## 2. MEDIUM — FIXED — Voice-command confirmations show a Python list

`cogs/voicedj.py:636` interpolates the wake-word *list* into the user-facing confirmation:

```
⚠️ Say "['hey dj', 'computer'] confirm" to stop.
```

(verified — that is literally what renders). Should be something like `self._fmt_wakes(wake)` or `wake[0]`. Every destructive voice command ("stop", "clear") shows this, so voice users see it constantly.

## 3. MEDIUM — FIXED — Poker: all-in hands stall, and there are no side pots

`cogs/games/poker.py`:

- When every remaining player is all-in, `advance_street` deals the next street and posts an action view addressed to a player who can't act. The board doesn't run out on its own — someone has to keep pressing Call (which calls $0) to zombie-walk the game to showdown. Needs "fewer than 2 players can still act → auto-deal remaining streets and go to showdown."
- No side-pot logic: a short-stacked all-in winner takes the entire pot, including chips they never covered. Classic simplification, but with real chip balances carried between hands via Play Again, it distorts the game.

Also a rules nuance in blackjack (`resolve`): a player's natural blackjack against a dealer's 3-card 21 pays as a push; by standard rules the natural wins.

## 4. LOW — FIXED — Progress-message race exists outside `-transcribe`

You already fixed the "stuck at 100%" race in `_do_transcribe` with `_ThrottledStatusUpdater.aclose()`, but the same updater is used without `aclose()` in `_do_share`, `_do_convert`, `_do_download`, and `_do_resize`. Most of those end by deleting the status message (which kills the race), but the error paths edit it, so a late in-flight progress edit can still overwrite an error/final text. Two small improvements: call `await updater.aclose()` before any final edit in those paths, and flush the last `_pending` content in `aclose()` (currently an update that lands during the throttle window is dropped, so bars can freeze at 97%).

## 5. LOW — FIXED — Error-handling polish in admin.py

- `_console_error` re-raises after replying, which produces an "Ignoring exception in command error handler" traceback in the logs every time a non-owner tries `-console`. Just `return` for `NotOwner` and only re-raise the rest.
- The global `on_command_error` listener fires for *every* command error (discord.py dispatches it in a `finally`), and it doesn't recognize `commands.CheckFailure`, `commands.NotOwner`, or `commands.MissingRequiredAttachment` — those fall through to "Something went wrong" plus a logged stack. Worth adding branches.

## 6. LOW — FIXED — Views with `timeout=None` are never stopped

`JobView` (every download/convert/resize job), the poker/blackjack/yahtzee lobby+action views, and `_CancelView` all use `timeout=None` and never call `.stop()`. discord.py keeps such views registered until stopped, so listener state accumulates for the life of the process. Cosmetic at your scale, but a one-line `view.stop()` at the end of each job would tidy it.

## 7. Performance note — one DB connection per event

`record_message` (every guild message) and the emoji listeners (up to 2 writes per emoji per message) each open a fresh SQLite connection and commit individually. WAL keeps it safe, but on a busy server this is the biggest source of avoidable I/O. An easy win, if it ever shows up in profiling, is a small in-memory accumulator flushed every few seconds, or one shared long-lived connection.

## 8. Observations, no action needed

- `token.txt` and `spotify.txt` contain real credentials. Both are gitignored and there's no `.git` here, so no leak — just don't zip/share the folder as-is.
- `-analyzeduration 0 -probesize 32` in `create_audio_source` is aggressive; it's fine for typical YouTube/SoundCloud streams but can misdetect odd containers. If you ever see instant playback failures on unusual sources, that's the first knob to turn.
- `vc.stop()` immediately followed by `vc.play()` in `_begin_playback` can very occasionally raise "Already playing audio" (discord.py player-thread race). It's caught and logged, but the symptom is a track that silently doesn't restart after a seek/filter. A tiny retry would mask it.
- Pyflakes: unused imports in `cogs/admin.py` (`get_state`), `core/state.py` (`Any`); the `faster_whisper` ones are intentional probes.
- A playlist whose links all fail to resolve drains silently with no channel message (`_play_next` skips each and returns).

## Round 2 (July 18) — music playback deep dive, ALL FIXED

Prompted by two reported symptoms: "queued a song, said ✅, never played" and "says it's playing but no audio". Six root causes were confirmed in the code and fixed.

**Queued-but-never-plays:**

**R2.1 — Stale stream URLs.** Despite the "lazy resolution" design note, `-play <song>` fully resolves at queue time and `resolve_stream_url` trusted any existing URL forever. yt-dlp URLs expire (~6h on YouTube), so tracks deep in a long queue "played" into a 403 and got skipped. Fix: every track now carries `_resolved_at`, and play-time resolution re-fetches anything older than 45 minutes (`STREAM_URL_TTL` in core/audio.py). Verified with behavioral tests: fresh tracks short-circuit, stale and legacy dicts re-resolve.

**R2.2 — Silent skip black hole.** Every play-time failure (resolve error, dead link, region lock) dropped the track with only a log line — the user saw ✅ and then nothing. Fix: `GuildState.text_channel_id` remembers where music was last controlled from, and the engine now posts "⚠️ Skipping **track** — reason" (batched for playlists) plus a note when the queue ends because of it. Autoplay failure also announces itself now.

**R2.3 — Concurrent kickstart lost tracks.** Two users queueing into an idle bot raced `_play_next`; both popped a track and the loser's vanished. Fix: a per-guild `advance_lock` serialises advancement; the loser re-checks the generation inside the lock and bails with its track still queued. Verified by simulation: two concurrent kickstarts → one playback, zero lost tracks.

**R2.4 — Zombie voice clients.** A voice client whose session died (failed handshake, unrecovered gateway drop) was reused forever; every `vc.play` raised "Not connected", was swallowed, and *all* future plays silently did nothing until `-leave`. Fix: `_connect_to_author` detects `not vc.is_connected()`, force-disconnects, and reconnects fresh; connect timeouts now produce a clear message instead of a traceback.

**Playing-but-silent:**

**R2.5 — Swallowed play failures with a live progress bar.** `current` and `track_started_at` are set before `vc.play()`, and only `ClientException` was caught — anything else (notably `OpusNotLoaded` when libopus is missing) escaped into the task void while `nowplaying` rendered a moving wall-clock progress bar over dead air. Fix: broadened handling with a specific libopus hint, generic fallback, and a channel notice in each case.

**R2.6 — ffmpeg stalls forever.** The `-reconnect` flags had no read timeout, so a connected-but-silent CDN blocked the player thread indefinitely — `is_playing()` true, zero audio, until a manual skip. Fix: `-rw_timeout 15000000` (15s) aborts stalled reads; the after-callback then fires and the queue advances.

**R2.7 — New `rejoin` command** (aliases `reconnect`, `fixaudio`). For wedged voice sessions that no bot-side code can prevent (Discord voice-server failover, DAVE re-key hiccups): tears the connection down, rebuilds it fresh, and resumes the current track at its previous position with the queue intact. When "playing but silent" ever happens again, this is the one-command fix — and `filter` changes now refresh stale URLs before restarting ffmpeg too.

Verification: full shadow-tree compile clean, pyflakes clean, 149 + 8 tests pass, TTL/locking/notification behavior confirmed by direct simulation.

## What's in good shape

Tests (149 passing, fast), the playback-generation design, the DB layer (WAL, explicit cascade handling, careful upserts), cancellation plumbing in downloads, the transcribe rate-limit fix, error message hygiene for yt-dlp, and the wake-word/intent parser (well-tested). The refactor from the original single file has clearly paid off.
