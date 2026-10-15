# Music Bot

A multi-server Discord music bot — refactored from a 4,000-line single file
into a proper cog-based project. Plays from YouTube, SoundCloud, Spotify,
Bandcamp, and direct file links; supports saved playlists; downloads tracks;
shows synced karaoke lyrics; and includes Poker, Blackjack, and Yahtzee.

It has also grown a decent media toolbox along the way — converting,
transcoding, snipping and splitting video, cleaning up audio, editing images,
reading text out of screenshots, transcribing speech, and repeating reminders
that respect your timezone.

## Requirements

- **Python 3.10+**
- **ffmpeg** on your PATH
- **Deno** on your PATH, for YouTube (`winget install --id=DenoLand.Deno` on Windows, `curl -fsSL https://deno.land/install.sh | sh` on Linux — the bot also finds it in `~/.deno/bin` when that isn't on its PATH, so `-restart` is enough after installing). yt-dlp runs YouTube's JavaScript challenge with it; without Deno (or Node 22+) and the `yt-dlp-ejs` solver that `yt-dlp[default]` in requirements.txt installs, it can't, and works with fewer formats — a likely cause of on-and-off 403s. `python bot.py --preflight` and `-ytdiag <link>` check it.
- **libopus** — the Opus audio codec (`sudo apt install libopus0` on Debian/Ubuntu; usually preinstalled on most desktop distros but missing from minimal server installs). Without it the bot crashes on `play` with "Opus library needed in order to use voice."
- A Discord bot token (free, from <https://discord.com/developers/applications>)
- *Optional:* Spotify API credentials, free from <https://developer.spotify.com/dashboard>

## Setup

1. Install Python deps. On Linux, `install.sh` builds a venv, installs
   everything, and offers to apt-get any missing system packages:

   ```bash
   ./install.sh              # core + the tiny extras (qr codes, langdetect)
   ./install.sh --all        # ...plus every AI extra — needs ~5 GB
   ./install.sh --karaoke    # ...just AI vocal separation
   ./install.sh --help       # every flag
   ```

   The AI extras (karaoke, `-transcribe`, `-translate`, `-nobg`) are opt-in
   because they pull torch and friends — several GB and several minutes. Add
   `--gpu` for CUDA builds, `--yes` for an unattended run, `--no-apt` to keep
   the script away from your package manager.

   Or do it by hand:
   ```bash
   pip install -r requirements.txt
   ```

2. Pick **one** way to give the bot its Discord token:

   **Option A — `.env` (recommended)**
   ```bash
   cp .env.example .env
   # edit .env, paste your token after DISCORD_TOKEN=
   ```

   **Option B — `token.txt` (simpler)**

   Open `token.txt` and replace `PASTE_YOUR_TOKEN_HERE` with your real token,
   on its own line. Anything after a `#` is treated as a comment.

   *(If both are present, `.env` wins.)*

3. *(Optional)* Add Spotify credentials to `.env`:
   ```
   SPOTIFY_CLIENT_ID=...
   SPOTIFY_CLIENT_SECRET=...
   ```
   Without these, the bot still works — it just won't accept Spotify links.

4. Run it:
   ```bash
   python bot.py
   ```

The first run creates `bot.db` (SQLite) in the project directory and syncs
slash commands. Slash commands can take a few minutes to appear in Discord
the first time.

## Inviting the bot

In the Discord Developer Portal → your app → **OAuth2 → URL Generator**,
tick the scopes **`bot`** and **`applications.commands`**. Under bot
permissions you need at least: Send Messages, Embed Links, Attach Files,
Read Message History, Connect, Speak, Use Voice Activity, Use Slash Commands.

For lyrics-without-DM and games to work cleanly, also enable the
**Message Content Intent** and **Server Members Intent** under
*Privileged Gateway Intents* on the Bot tab.

## Commands

The default prefix is `-`. Change it per-server with `-prefix <new>`
(or `/prefix`). Every command also has a slash version.

| | |
| --- | --- |
| `play <query>` | Search or URL — YouTube, SoundCloud, Spotify, Bandcamp, Discord CDN |
| `playnext <query>` | Add to the front of the queue |
| `search <query>` | Top 5 YouTube results in a picker — choose instead of taking the first match |
| `queue`, `nowplaying`, `skip`, `pause`, `resume`, `stop`, `leave` | The usual |
| `seek 1:30` · `seek 1:23:45` · `seek +30` · `seek -15` | Includes h:m:s — long tracks work |
| `volume 0–1000` | 100 = normal; >200 boosts and may distort |
| `loop [song\|queue\|off]`, `shuffle`, `balance`, `clear`, `remove 1,3,5-8` | Queue management. `loop` alone toggles looping the current song; `loop queue` loops the whole queue — finished and skipped songs go round again, last. |
| `move 5 1` · `swap 1 4` · `skipto 4` · `reverse` · `dedupe` | More queue management |
| `history` / `back` | Recently played tracks · replay the previous one |
| `filter bassboost` · `filter nightcore` · `filter speed 1.5` · `filter off` | Audio filters (also: slowed, 8d, tremolo, vibrato, karaoke) |
| `autoplay on/off` | Radio mode — keep playing related tracks when the queue runs out *(off by default)* |
| `sponsorblock on/off` | Auto-skip sponsor reads in YouTube tracks via SponsorBlock *(on by default)* |
| `autojoin on/off` | When someone joins a voice channel and **nobody else in the server is in voice**, the bot joins them automatically — no summoning needed (it doesn't start playing anything). *Off by default, per server, Manage Server to change.* Never joins the AFK channel, a stage, a full channel or one it can't speak in; `autoleave` rules still apply afterwards. Like being summoned, it starts the rolling `clip` buffer. |
| `playlist new/add/remove/play/list/info/import/rename/delete` | Saved per-server playlists |
| `playlist savequeue <name>` | Snapshot the current queue into a saved playlist |
| `download <query>` / `dl mp3 <query>` / `dl mp4 <query>` | Pulls audio/video — Discord (≤10MB), catbox.moe (≤200MB), or litterbox.catbox.moe (≤1GB, 72h expiry). Sources up to 3GB get re-encoded to fit; bigger than that is refused. |
| `resize <link> <size>` | Download and re-encode to a target size. Smart parser: `500mb` / `1g` / `5GB` / `0.5g` / `500` (bare number = MB). |
| `convert [format]` | Convert an attached file or URL — images (png/jpg/webp/bmp/gif), audio (mp3/wav/flac/ogg/opus/m4a), video (mp4/mkv/webm/mov). Omit the format and the bot detects the file type and offers target buttons. Video→mp3 extracts audio, video→gif animates, mp4↔mkv↔mov remuxes losslessly. |
| `transcode [codec] [quality]` | Re-encode a video's **codec** — `h264` (plays everywhere), `h265` (smaller, Apple-friendly), `vp9` (browser-native), `av1` (smallest). Fixes AV1/`av01` files that stutter or won't play. Omit the codec and the bot reports what it is and offers buttons. Quality is `high`/`balanced`/`small`. Container follows the codec, audio rides along untouched, GPU encoding used when the host has it. *(Container-only changes are `convert` — that remuxes instantly.)* |
| `snip <url> 1:00 1:30` | Cut a section out of a video/audio. End can be relative: `snip <url> 12:05 +20`. `mp3` at the end for audio-only. YouTube-style links download only the clipped part. |
| `chapters <url>` | Lists a video's own chapter markers with timestamps and lengths, then a dropdown to grab one or **Split all** to get every chapter as its own file. `chapters mp3 <url>` for audio. Falls back to pointing at `snip` when a video publishes no chapters. |
| `normalize [target]` | Evens out loudness with two-pass EBU R128 — the fix for a recording that's too quiet or wildly uneven. `speech` (−16 LUFS, default) · `music` (−14, Spotify/YouTube level) · `broadcast` (−23). Video keeps its video track stream-copied. |
| `desilence [strength]` | Cuts the silent gaps out of a recording: start, end, and everything between. `gentle` / `normal` / `aggressive`. Always leaves a little breathing room, reports how much it removed, and always outputs audio (trimming would desync video). |
| `speed 1.5` | Speeds up or slows down audio/video **without changing pitch**, so a 1.5× podcast still sounds like the people in it. Video is re-encoded in step to stay in sync. `0.25`–`4`, or `150%` / `2x` / `half`. Completes the `normalize` + `desilence` trio. |
| `ocr [lang]` | Pulls the text out of an attached screenshot or image so it's selectable and searchable — error messages, receipts, slides. Upscales and contrast-corrects first; reads the first frame of a video. `ocr langs` lists installed languages. Needs `apt install tesseract-ocr` on the host *(optional)*. |
| `translate [lang] <text>` | Translates into English by default, or any language you name. **Attach a screenshot** and it OCRs the text first, so a foreign menu or error becomes English in one step. Runs locally — no API key, nothing sent anywhere. `translate langs` shows what's installed; `translate install <from> <to>` adds a pair *(owner)*. Needs `pip install argostranslate langdetect` *(optional)*. |
| `qr <text>` · `qr` *(with image)* | Makes a QR code from text or a link — the quick way to get something onto a phone. Attach an image instead and it reads codes (and barcodes) out of it, upscaling blurry ones first. Decoded links are shown as **plain text, not clickable**, since a QR code is exactly where you can't see the destination. Needs `pip install qrcode zxing-cpp` *(optional)*. |
| `clip [duration] [split]` | Clip recent **voice chat** to audio: `clip 5m`, `clip 2m 23s`, `clip 90s`. Add `split` for one `.mkv` with a separate track per person. The bot keeps a rolling 5-min voice buffer while it's in a channel. |
| `transcribe` | Speech-to-text for an attached file or URL, with timestamps. Needs `pip install faster-whisper` on the host (optional); model defaults to `WHISPER_MODEL` in `.env` (`base`), and `transcribemodel <name>` switches it live (separate from the voice-command model). |
| `flip h/v` · `rotate 90/auto` · `crop 1:1` · `scale 50%` | **Image editing** — the things you'd otherwise open Photoshop for. `rotate auto` straightens phone photos from their EXIF tag; `crop` takes `1:1`, `16:9`, `800x600`, `800x600+50+20` or `80%`; `scale` takes a width, `50%`, `2x`, `x600` or `800x600`. |
| `shrink 500kb` · `thumb [px]` | Squeeze an image under a size limit (quality first, resolution only if it must), or make a small preview. |
| `grayscale` · `invert` · `blur [r]` · `sharpen` · `brightness 1.4` · `contrast 1.2` | One-shot adjustments. `blur 20` is handy for hiding something in a screenshot. |
| `caption top \| bottom` · `watermark [corner] <text>` | Meme-style outlined text that wraps and auto-sizes, or a small semi-transparent corner stamp. |
| `nobg` | **AI background removal** — returns a transparent PNG cutout. Runs locally via [rembg](https://github.com/danielgatis/rembg) (U²-Net): no API key, no per-image cost. Needs `pip install rembg onnxruntime` *(optional)*; the first run downloads a ~180 MB model. |
| `remindme 2h30m <text>` | Pings you in the channel when time's up. `reminders` lists yours, `reminders cancel <id>` cancels. Survives restarts. |
| `remindme every weekday 8:30 <text>` | **Repeating reminders.** Intervals (`every 2h`, `every 3 days`) or clock schedules (`every day 9am`, `every monday 18:30`, `every weekend 10am`, `every mon,wed,fri 6pm`). Minimum repeat is 10 minutes. |
| `when friday 8pm` | Turns a time into a **Discord timestamp** that renders in each reader's own local time — nobody does the maths, nobody turns up an hour late. Understands `3pm`, `tonight`, `tomorrow 9am`, `friday 3pm`, `aug 5`, `2026-08-01 14:00`, `in 2 hours`. Read in your timezone; you get every format with the code to copy. |
| `time [@member \| zone]` | Your local time, someone else's *(worth checking before you ping them at 3am)*, or the time in any zone. |
| `timezone <zone>` | Sets *your* timezone so clock reminders mean what you think — `timezone Europe/London`, `timezone PST`, `timezone Tokyo`. UTC until you set one. Bare city names and US abbreviations work; abbreviations map to real zones so DST keeps working. |
| `stats voice [@member]` | Who spends the most time in voice channels, the busiest channels, and anyone in a call right now. Only **durations** are stored — no audio, no per-call logs. `stats voice optout` excludes you and deletes your existing rows. Counted live; can't be backfilled. |
| `stats` · `stats user [@member]` · `stats channel [#channel]` · `stats emoji [@member]` · `stats sticker` | Server stats in one place — top senders, busiest channels, per-user/-channel rankings, **and** emoji/sticker usage (including *who* uses which emoji: `stats user` and `stats emoji @member` show a person's most-used). One `stats scan` backfills messages + emoji + stickers from history *(needs Manage Server)*. Live counting runs whenever the bot's up. *(Emoji/sticker `add`/`remove` stay under `emoji`/`sticker`.)* |
| `voice enable` · `voice optin` · `voice wake <phrase>` | **Hands-free voice control** — say e.g. *"hey DJ, play mr brightside"* / *"…skip"* / *"…hold up"* / *"…start over"* / *"…loop off"* / *"…volume fifty"* in the voice channel and the bot acts. You can pause after the wake word (*"hey DJ" … "skip"*). **Start with the wake word**: one in the middle of a sentence (*"I said hey DJ skip"*, *"don't say hey DJ stop"*) is ignored, because that's quoting, not asking. Admin enables it, each person opts in — only opted-in speakers are processed, and only from the moment they opt in. `stop`, `clear` and spoken volumes over 200% ask *"…? Say yes or no"* (refusals like *"yeah no"* cancel). Replies: `voice feedback text/voice/chime/silent` (voice = spoken back). `voice status` shows whether it's actually working; `voice diag` / `voice debug` / `voice clip` for troubleshooting; `voice doctor` checks the host. Also *"…shuffle"*, *"…loop the queue"*. With the **smart tier** (`voice ai`, needs an OpenRouter key on the host) just say what you want: *"…play the first three songs off Panic! At The Disco's Vices and Virtues, turn it down to 50, shuffle, skip the first one and loop the rest"* — the album's tracks are looked up in a music catalogue and every step runs in the order you said it; *"…play the song from Top Gun"*, *"…play Owl City's most popular songs"*, or ask a question (*"…who sings this?"*) and it answers out loud. Needs `faster-whisper` on the host; `base.en` by default — `voice model <name>` (bot owner) switches it. |
| `tts <text>` · `ttsvoice [name]` · `ttsstop` | **Talk without a mic** — the bot says your message in your voice channel in a realistic neural voice. Over music, the music ducks under the speech and swells back after; paused music stays paused (and `resume` mid-sentence brings it back in underneath). Mentions are read as names, links as *"a YouTube link"*, chat shorthand in full (*idk* → *I don't know*), and the speaker's name is announced whenever it changes. `ttsvoice` lists voices and each person picks their own so people can tell who's talking (`ttsvoice ava`, or any Microsoft voice id such as `fr-FR-DeniseNeural`); `ttsstop` cuts it off. Alias: `say`. |
| `lyrics` / `lyrics <song>` | **Karaoke lyrics that stay in time.** Picks the lyrics by how long the recording *actually* is rather than by title alone, so a live take or an extended mix doesn't get the album timings; measures any silence before the music starts and corrects for uploads re-encoded at a different speed. Follows pause, seek and speed filters. If a track is still off, `lyrics offset +1.5` (later) / `-1.5` (earlier) fixes it and **saves the correction for that upload**, so it's right for everyone next time. `lyrics offset` shows the current correction, `lyrics offset reset` forgets it. Falls back to static lyrics — and says why — rather than scrolling timings it doesn't trust. |
| `karaoke <song>` / `karaoke` | **Strips the vocals with an AI separation model** and sends you the instrumental as an MP3 — a real model (UVR/MDX-Net via `audio-separator`), not the old `filter karaoke` channel trick that took the drums and bass out along with the vocal. Doesn't touch playback: separation is slower than realtime on CPU, so the bot keeps playing and posts the file when it's ready (a few minutes). Each track is separated **once** and cached, so the second request is instant. Files land as a Discord attachment, or a catbox/litterbox link if oversized. `karaoke cache` shows what's stored. *Optional: `./install.sh --karaoke`.* |
| `poker new`, `bj new`, `yz new` | Lobby-style games |
| `update` · `update ytdlp` · `update all` | **Upgrade the bot's Python packages from Discord** *(owner only)* — the fix for yt-dlp breaking, which happens whenever YouTube changes something. Bare `update` does everything in requirements.txt, `update ytdlp` is the quick targeted fix, `update all` also covers the AI extras. Reports before → after versions so you can see what actually moved, and offers a **Restart** button: a package upgrade doesn't affect the running process, so without the restart nothing changes. `versions` lists what's installed. *(Terminal equivalent: `./update.sh`.)* |
| `restart` | Restart the bot in place *(owner only)*. Keeps the same PID, so systemd sees a continuously running service. |
| `prefix <new>` | Change this server's prefix *(needs Manage Server)* |
| `invite` | Get an OAuth link to add the bot to another server |
| `help` | Auto-generated from loaded cogs |

### Right-click menus

Four of these are also **message context menus** — right-click any message →
*Apps*: **Transcribe**, **Extract text**, **Convert format**, **Re-encode
codec**. The bot finds the file on that message (attachment first, then the
first link in the text) and runs the same job, so you don't have to
copy the URL out first. Useful mostly for voice messages, which are otherwise
unsearchable.

## Admin console

When the bot is running interactively (i.e. inside a `screen` or `tmux`
session, or just a regular terminal), it gives you a prompt for server-side
admin commands. Type `help` at the `>` prompt to see them all:

```
> help
Available console commands:
  guilds        List every server the bot is in
  guild         Detailed info for one guild (by id or name)
  voice         List active voice connections + what's playing
  stats         Uptime, memory, latency, totals
  invites       Get an invite URL for each guild
  leave         Leave a guild by id
  reload        Reload a cog without restarting
  quit          Shut the bot down cleanly
  help          Show this list
```

The most useful one is `guilds`, which lists every server the bot is in
with IDs, owner, and member count — useful when you've forgotten which
private servers it's joined. `invites` will fetch (or create) a working
invite URL for each so you can actually go look at them.

**From Discord too:** the bot owner can run any of these without a terminal via
`-console <command>` (alias `-con`) — e.g. `-console stats`, `-console reload
music`, `-console restart`. Output comes back in the channel (as a file if it's
long). It's owner-only and hidden from `help`.

To use it: attach to the bot's screen (`screen -r musicbot`), and you'll
see the `>` prompt. Detach again with `Ctrl+A D` when you're done; the
bot keeps running.

If the bot is launched non-interactively (systemd unit, nohup, redirected
stdin), the console is automatically disabled and the bot runs silently.

## Multi-server

A single running instance handles every server it's invited to. All state
(queues, playlists, prefixes, uploads) is keyed by `guild_id`, so servers
can't see or affect each other. No special config needed.

You'll only need sharding at thousands of servers — change `commands.Bot`
to `commands.AutoShardedBot` in `bot.py` if you ever get there.

## Notes / caveats

- **Spotify private and Blend playlists** are not supported — they require
  per-user OAuth, which the bot doesn't do. Public playlists, albums, and
  tracks work fine.
- **Login-gated sources** (many Instagram posts, private/age-restricted
  YouTube, members-only) can't be fetched anonymously — the bot replies that
  the post is private/login-gated. To enable them, give yt-dlp a cookies file:
  export `cookies.txt` from a logged-in browser (e.g. the "Get cookies.txt
  LOCALLY" extension) and either drop it in the project root or point
  `YTDLP_COOKIES=/path/to/cookies.txt` at it in `.env`. On a desktop host you
  can instead set `YTDLP_COOKIES_FROM_BROWSER=firefox` (or `chrome`,
  `firefox:profilename`) to read cookies straight from the browser. Cookies are
  account credentials — keep the file private; `cookies.txt` is in `.gitignore`,
  but don't commit it or share it.
- **Stream URLs are resolved at play-time**, not when you queue a track, so
  tracks that sit in a long queue don't silently expire.
- **Idle disconnect**: 5 minutes by default (`IDLE_DISCONNECT_SECONDS` in
  `.env`). Set to `0` to disable.
- **Upload routing**: files ≤ 10 MB go to Discord (native inline preview).
  10 MB – 200 MB → [catbox.moe](https://catbox.moe), permanent link.
  200 MB – 1 GB → [litterbox.catbox.moe](https://litterbox.catbox.moe), link
  expires after 72 hours. Sources between 1 GB and 3 GB are re-encoded down
  to ~950 MB before uploading. Sources larger than 3 GB are refused, since
  any compression to fit would be too aggressive to look good — use the
  `resize` command if you really want to push past that.
- **Resize**: `-resize <link> <target>` downloads then re-encodes to the
  target size. Anything `parse_size` accepts works for the target: `500mb`,
  `1g`, `5GB`, `0.5g`, or just `500` (treated as MB). Maximum target is
  1 GB (the litterbox ceiling).
- **Optional extras.** Several features degrade to a clear message instead of
  breaking the bot when their dependency is missing:
  `-transcribe` and voice control need `pip install faster-whisper`;
  `-ocr` needs the system package `tesseract-ocr` (plus
  `tesseract-ocr-<lang>` for anything but English);
  `-nobg` needs `pip install rembg onnxruntime`;
  `-qr` needs `pip install qrcode zxing-cpp` (tiny, no system libraries);
  `-translate` needs `pip install argostranslate langdetect` plus a
  per-pair model download;
  the image cog needs `Pillow`, which is now a normal requirement.
- **Emoji and sticker uploads are fitted, not refused.** Discord caps emojis
  at 128px/256 KB and requires stickers to be exactly 320×320 PNG under
  512 KB. Anything bigger is resized on the way in — animated GIFs stay
  animated (frames are dropped before the animation is), non-square stickers
  are centred on a transparent canvas rather than stretched, and the bot says
  what it had to give up.
- **Repeating reminders** are rescheduled inside the same transaction that
  delivers them, and missed occurrences are skipped rather than replayed — a
  bot that was down overnight fires an hourly reminder once on restart, not
  eight times. Clock schedules recompute against the user's timezone each
  time, so `every weekday 8:30` stays at 08:30 across a DST change instead of
  drifting an hour like a fixed 86 400-second interval would.
- **Voice time** is recorded as durations only, per (guild, user, channel).
  Sessions are persisted on join so a restart mid-call doesn't lose them, and
  a session left open by a crash is credited at most 12 hours — the bot can't
  know when someone actually left, and an offline weekend shouldn't win the
  leaderboard. `-stats voice optout` stops the counting *and* deletes what's
  already there.
- **TTS** (`-tts`) uses Microsoft's neural voices through
  [edge-tts](https://github.com/rany2/edge-tts) — the service behind Edge's
  "Read aloud": free, no key, but online, so the text is sent to Microsoft to
  be spoken. Speech is mixed *into* the music rather than replacing it
  (`core/voice_mix.py`), so the queue, the playback clock and synced lyrics
  are untouched by it. `.env` knobs: `TTS_VOICE` (default voice — a name from
  `-ttsvoice` or a full voice id; `andrew`), `TTS_MAX_CHARS` (longest message
  read out; `400`), `TTS_DUCK_PERCENT` (music level under speech; `25` ≈ −12 dB).
  Microsoft occasionally changes the service; `./update.sh` pulls the
  edge-tts release that follows it.
- **Voice control** (`voice …`) — how it hears you, since that's where it
  used to go wrong. The recorder hands each decrypted Opus packet (with its RTP
  sequence number and timestamp) to a per-speaker front end: a persistent Opus
  decoder that conceals lost packets instead of skipping them, a proper
  anti-aliased 48→16 kHz resampler, **Silero VAD** (the model ships inside
  faster-whisper — no extra install), and a segmenter with pre-roll and an
  end-of-speech pause. An open mic, music bleed or chatter right before a
  command no longer swallows it. Recognition runs one clip at a time on its own
  thread. Playback controls (skip, pause, volume…) act immediately even while a
  song search is running; song requests are looked up one after another so
  they queue in the order asked; *"stop"* / *"clear the queue"* call off song
  requests heard before them. Every command is re-checked before it acts — and
  a song request again after its search: opt-outs, `voice disable`, a typed
  `stop`, the bot moving, or the speaker leaving cancel anything still in
  flight. Each clip keeps the session and consent it was captured under, and
  yes/no answers are timed from when you actually started speaking (not from
  the buffered audio before it) — a quick "yes" counts, and an answer captured
  in time still counts if the CPU is slow to transcribe it. Nothing older than
  ~20 s acts, and a command understood too late says so. What people say is only
  logged during `voice debug` (time-limited); `voice diag` shows counts and
  reasons instead. `.env` knobs, all optional: `VOICE_WHISPER_MODEL`
  (`base.en`), `VOICE_BEAM_SIZE` (`1`), `VOICE_CPU_THREADS` (`0` = auto),
  `VOICE_DEVICE` / `VOICE_COMPUTE_TYPE` (`cpu` / `int8`), `VOICE_LANGUAGE`
  (`en`), `VOICE_VAD_THRESHOLD` (`0.5`), `VOICE_END_SILENCE_MS` (`600`),
  `VOICE_ASR_HINTS` (off — prompt text can be echoed back on noise).
- **Smart Voice DJ** (`voice ai`) — when it's on, every *song request* and
  anything the built-in voice commands don't cover goes, as text, to an LLM
  through [OpenRouter](https://openrouter.ai) that works out what you actually
  want instead of searching YouTube for your sentence: *"the song from Top
  Gun"* → Kenny Loggins – Danger Zone; *"the first three songs off Panic! At
  The Disco's Vices and Virtues"* → an **album lookup** (Deezer, falling back
  to iTunes — keyless public catalogues, `core/music_catalog.py`) that queues
  tracks 1-3 in album order; an artist's **most popular** or **newest** songs
  the same way; several commands in one sentence.
  **Plans, not just commands** (round 8): the AI is shown what is playing,
  the upcoming queue (each entry with its own id and position — `pos:1` is the
  next song, the one playing is separate) and this person's last few
  requests, and answers with a plan: numbered steps over every operation the
  bot has by voice — add/find songs and albums, remove, move, swap, skip to,
  shuffle, reverse, balance, dedupe, clear, volume, loop, seek, filters,
  lyrics, likes, playlists, reminders, stats, games… (72 commands; the other
  101 are typed-only, with the reason — see `VOICE_COVERAGE.md`) — with
  repeat counts (*"five separate times"*) and *"…but only after X"* even when
  X is said last. It can look up an album's track list first. The steps run
  one at a time against the **live** queue: every queued copy of a song has
  its own identity, positions are counted when the step runs, *"move the
  Phoenix to the front"* means the copy this request just added, and if
  someone else (or playback) changed the songs a step points at, it says so
  instead of moving or removing the wrong one. Typed commands run by voice go
  through the same code and the same permission checks as typed. The reply
  lists what was done, what failed and what wasn't done because of it.
  Albums: *all*, *tracks 2 to 5*, *the first/last three*, *the first half*,
  *the latter half* (odd lengths: the first half gets the extra track — 11 →
  1-6 and 7-11), *everything except the last song*, *the first half except
  its last track*. Limits are said, never cut quietly: up to 2,000 characters
  of speech (about two minutes) go to the planner whole, at most 40
  steps, 25 copies of one song and 120 single songs a request. A request
  longer than ~10 s arrives in parts; they're held until it ends, then the
  whole audio is recognised once more (no seam in a name said across the
  cut) and planned as one. Catalogue songs are queued
  like a Spotify playlist and found on YouTube as *"Artist - Title"* when their
  turn comes; deluxe/anniversary/live editions are avoided unless asked for.
  Songs go to the end of the queue unless you say otherwise: *"…play X
  next"* puts them right after the current song; *"…skip this and play the
  seventh and ninth songs off Pray for the Wicked instead"* (or *"…play X
  now"*) puts them in front and skips once — and if they can't be found,
  nothing is skipped. Which tracks can be said in words (*"the seventh and
  ninth"*, *"the first two and the last one"*); anything unclear is asked back,
  never guessed. A brief voice-receive dropout (DAVE recovers it within a
  second or two) no longer calls off a request that was already heard; the bot
  actually moving or leaving does, and says so.
  **Songs the AI doesn't know** — newer than the model, or misheard (*"I Write
  Him's Not Travis D's by Young Gravy"*) — are matched against the artist's
  real songs in the catalogue by spelling *and* sound, so that's Yung Gravy's
  "I Write Hymns Not Travesties"; the words you actually said beat the AI's
  guess (it once swapped an unknown "Cordial" for the Set It Off song that was
  playing), and when two titles are close the AI picks from the real list. A
  song you **describe** instead of naming (*"that Yung Gravy song that samples
  Panic! At The Disco"*) is picked from that artist's songs, or **looked up
  online** (OpenRouter's web search, about a cent a lookup; `VOICE_WEB_LOOKUP`,
  at most `VOICE_WEB_LOOKUPS_PER_HOUR` = 30). Then the YouTube upload has to be
  that song — its title words, the artist in the title or channel, its length;
  covers, live takes, remixes and slowed/sped-up edits lose unless you asked
  for one — and if no result is, the bot says it couldn't find it instead of
  playing whatever YouTube ranked first. Misheard command words are commands:
  *"tip this"* is *"skip this"*.
  **The artist you said is the artist you get**, or it asks: *"Call My Name
  by Priest"* is never Prince's song because Prince is better known — the
  words as heard are kept apart from the AI's spelling fixes, a correction has
  to sound like what was said, and an artist the catalogue doesn't know is
  searched as said and must show up in the upload. *"Kaimara by…"* with no
  artist is a question (*"Kaimara by who?"*), not a random video.
  Plain controls (*"skip"*, *"louder"*, *"shuffle"*) stay instant and
  offline, and without the AI (or when it fails) the built-in grammar still
  handles songs as before. Plans pass the same checks as spoken commands —
  consent, staleness, a yes/no for stop/clear/very loud (once per request) —
  plus a safety net for *"don't …"*. Questions get a short spoken answer
  (TTS, music ducked) that's also posted; a question and actions can be mixed
  in one request. Speech recognition stays local; sent to the AI are the text
  after the wake word, the player's state, the queue's song titles with who
  added each, and this person's last few requests and results — never audio.
  The planner is `google/gemini-2.5-flash` (falling back to
  `openai/gpt-4.1-mini`): on the live bench — the brief's requests and
  11 real recordings, as said and as the bot heard them — it planned 72 of 72
  right, median 2.4 s, about $0.0015 a request; flash-lite was faster
  (1.3 s) but got 32 of 36 (it dropped steps from long requests).
  `VOICE_PLAN_MODEL` / `VOICE_PLAN_FALLBACKS` / `VOICE_PLAN_TIMEOUT` (`20` s)
  change it. Everything else the AI does (picking a song from a list,
  looking one up online) uses the model below. Model:
  `google/gemini-2.5-flash-lite`, falling back to `openai/gpt-4.1-nano` if
  Google is unavailable; measured from the bot host (`core/voice_llm.py`),
  about 0.6 s and well under a cent per 100 requests. `voice ai` shows the
  model, calls, median time and spend since restart; `voice ai on|off`,
  `voice ai chat on|off` *(Manage Server)*. `.env`: `OPENROUTER_API_KEY`,
  `VOICE_LLM` (on), `VOICE_LLM_MODEL`, `VOICE_LLM_FALLBACKS`, `VOICE_LLM_URL`
  (any OpenAI-compatible endpoint, e.g. a local `llama-server`),
  `VOICE_LLM_TIMEOUT` (`8` s). AI replies run per speaker with bounded queues,
  so another speaker's controls never wait for an answer. `VOICE_TYPING`
  controls the reply-channel typing indicator; `VOICE_PROBE_MS=0` disables the
  early wake check. Indicators stop on a reply/drop and expire on their own.
- **Wake-word cue** — the bot's version of a smart speaker's light: the moment
  it hears the wake word it plays a short sound effect in the voice channel
  (`assets/wake_cue.wav`) and posts `:ear:*Now listening...*` in the
  voice-reply channel, at the same time. It comes from the **instant wake
  word**: a tiny model that only knows "hey DJ" (`core/wake_spotter.py` —
  openWakeWord's audio features plus our classifier, `assets/wakeword/`)
  listens to each opted-in speaker and fires right as "DJ" ends; the speech
  recogniser then checks just those words on its own thread, on a short
  window first (~0.15 s instead of the 30-second pass's ~0.9 s;
  `VOICE_FAST_WAKE_VERIFY_WINDOW`, `0` = only the old pass) and the old pass
  as the last look for the hard ones (noise, an unusual voice;
  `VOICE_FAST_WAKE_VERIFY_FULL`), so most "hey BJ"s and "hey, did you…"s cue
  nothing — and the sound goes straight onto the mixer,
  never behind a line of `-tts` being spoken or a full speech queue.
  Replaying 11 real microphone recordings through the bot's own
  Opus/VAD/detector path: every wake caught and checked, the cue ~70 ms
  (median) after the end of "DJ", ~125 ms with the recogniser busy — a
  server-side replay estimate; Discord's transport and the listener's client
  add their own delay (`assets/wakeword/README.md` has the details). If the
  full transcript then disagrees, the message is deleted. The older early
  check (speech-to-text on the first second or
  two) and the finished transcript remain as backstops — once per utterance,
  never twice. It only knows "hey DJ": servers with other wake words get the
  speech-to-text cue. `VOICE_FAST_WAKE=false` turns it off,
  `VOICE_FAST_WAKE_THRESHOLD` overrides the model's own threshold,
  `VOICE_FAST_WAKE_VERIFY=false` skips the check (faster, but expect false
  alarms) and `VOICE_FAST_WAKE_VERIFY_MODEL` picks the checker's model (e.g.
  `tiny.en`: faster, misses more). `voice diag` counts fires, cues,
  turned-down fires, and what the full transcript said. `voice cue`
  shows the setting, `voice cue on|off` switches the **sound** off per server
  *(Manage Server)*; the message always posts. Host knobs: `VOICE_CUE=false`
  turns the whole cue off, `VOICE_CUE_SOUND=false` mutes just the sound,
  `VOICE_CUE_FILE` points at a different WAV (48 kHz/16-bit/stereo is played
  as-is; anything else is converted), `VOICE_CUE_VOLUME` scales it. The sound
  plays over the music like `-tts` does, and is deliberately *not* counted as
  the bot speaking, so saying a command right after the cue still works.
  Both typed and spoken clear invalidate older pending song admissions.
  Wake/reply windows preserve timely speech while it is still being captured,
  within a bounded processing grace period.
- **Measuring voice recognition**: don't choose a model by its name — run
  `python scripts/voice_eval.py <folder>` over your own labelled clips
  (`voice clip` DMs you exactly what the model heard, which is how you build
  the set). It compares models, beam sizes and thread counts on false actions,
  command accuracy and p95 delay. Song requests are scored by title *and*
  artist (*"the pretender by foo fighters"* is wrong if you asked for Infected
  Mushroom), with exact-query accuracy reported separately; long clips are
  replayed through the bot's own wake-window rules with their timestamps. Its
  timings are recognition only — `voice diag` shows end-of-speech → first
  audio for songs requested by voice. `python bot.py --preflight` checks the
  interpreter that actually runs the bot: packages, Opus, DAVE, voice receive,
  ffmpeg, VAD, and a timed model load.
- **When a song won't play.** "Added to the queue" only means the search
  worked; `nowplaying`/`queue` say **Starting…** until the first frame of
  music actually goes out, and only then does it count as played (history,
  stats, Last.fm). FFmpeg failures are no longer mistaken for the end of the
  song: the bot keeps FFmpeg's exit status and the tail of its error output,
  retries once with a freshly resolved link (resuming where it cut out), and
  otherwise posts why in the channel the song was requested from — e.g.
  *"the site refused the audio stream (HTTP 403)"* — then moves on. `queue`
  also shows the last failure, so an empty queue right after a request
  explains itself. For a YouTube **403**, run `-ytdiag <link>` *(owner)*: it
  fetches a kilobyte of the real audio link with the bot's own settings and
  points at the likely cause — yt-dlp unable to solve YouTube's player
  challenge (Deno or the `yt-dlp-ejs` solver missing), an outdated yt-dlp
  (`-update ytdlp`, which now also updates the solver, then restart), or an
  IPv4/IPv6 address mismatch. yt-dlp's own warnings now appear in the log as
  `yt-dlp: …` lines instead of being discarded. If FFmpeg or Opus is missing, or the voice connection is
  gone, the queue is left alone and the message says what to fix. The log
  line `Playback failed (…)` carries FFmpeg's own last words with stream URLs,
  signatures and cookies redacted. Headers the extractor requires (Referer,
  User-Agent…) are passed to FFmpeg.
- **Voice clipping** (`-clip`): while the bot is in a voice channel it keeps a
  rolling 5-minute buffer of everyone's voice **in memory** (nothing hits disk
  until someone runs `-clip`), and it announces this when it starts. Calls are
  end-to-end encrypted (DAVE — mandatory on Discord since 2026-03-02); the bot
  decrypts as a normal member of the call's encryption group, so E2EE stays on.
  Tune the window with `CLIP_BUFFER_SECONDS`. Needs `discord-ext-voice-recv`
  (already in requirements.txt) and Python ≤ 3.12 (the extension uses the stdlib
  `audioop`, dropped in 3.13).

## Tests

```bash
pytest tests/
```

~1700 tests. The pure-Python helpers (formatters, date/recurrence parsing, image
transforms, codec/filter command building, poker and blackjack hand
evaluation) are unit-tested, and the DB layer is exercised against a
throwaway SQLite file pinned in `conftest.py`.

Many tests shell out to the real tools and skip themselves when those aren't
installed: ffmpeg round-trips that transcode a real AV1 clip and verify the
result with ffprobe, a loudness test confirming a −30 dB file comes back at
−16 LUFS, a speed test checking the audio and video streams stay the same
length, OCR against rendered text, and QR codes generated and decoded back.
`argostranslate` is too heavy to install just for tests, so the translation
backend is stubbed — which still exercises the pivot routing and error paths,
the parts that are actually ours. The Discord-touching code is
integration-only, with two exceptions that drive discord.py's real
audio-player thread against a fake voice socket: the `-tts` mixer's tests
(a skip, pause or resume mid-sentence is checked frame by frame), and the
playback-engine tests (`tests/test_playback_engine.py`) — scripted FFmpeg
stand-ins for every failure path, plus real FFmpeg processes reading from a
local HTTP server that answers 403 or demands a header. The voice-control
tests use a scripted transcriber: they check decisions and timing, not how
well Whisper recognises speech — measure that with `scripts/voice_eval.py`.
Likewise the smart tier's tests use a scripted model, a local stand-in for
OpenRouter and for the Deezer/iTunes catalogues: they check the request, how
answers are vetted, and what the bot does with them and in which order — not
how well a real model understands people (that was measured separately; see
`core/voice_llm.py`).

## Project layout

```
musicbot/
├── bot.py                  entry point
├── core/                   non-Discord building blocks
│   ├── config.py           env / token.txt loader
│   ├── logging_setup.py
│   ├── db.py               aiosqlite — replaces uploads.json + playlists.json
│   ├── state.py            per-guild queue / playback state
│   ├── playback_clock.py   true playback position (frame-accurate, pause-safe)
│   ├── lyrics_sync.py      lyrics matching + time alignment
│   ├── updater.py          pip upgrades + version diffing
│   ├── separation.py       AI stem separation (subprocess, CPU-throttled)
│   ├── audio.py            yt-dlp + Spotify + ffmpeg glue
│   ├── recurrence.py       "every weekday 8:30" + "friday 3pm" + timezones
│   ├── imageops.py         Pillow transforms, QR, emoji/sticker fitting
│   ├── translate.py        local argostranslate wrapper + language detection
│   ├── tts.py              edge-tts synthesis, voices, chat-text → speech cleanup
│   ├── voice_frontend.py   voice control audio: Opus → PLC → 16 kHz → Silero VAD → segments
│   ├── voice_pipeline.py   voice control job queue, ASR runner, model loader, metrics
│   ├── voice_intent.py     voice command grammar (wake words, commands, yes/no)
│   ├── voice_llm.py        voice smart tier: the LLM request, and its answers vetted
│   ├── voice_tools.py      what voice can do: the typed operation registry + coverage
│   ├── voice_plan.py       a plan's steps, checks, run order, and queue references
│   ├── voice_exec.py       running a plan step by step against the live queue
│   ├── queue_ops.py        queue operations shared by typed commands and voice
│   ├── music_catalog.py    album track lists / popular / newest songs, which song a title means (Deezer, iTunes)
│   ├── wake_spotter.py     instant "hey DJ": a tiny wake-word model on each speaker's audio
│   ├── whisper.py          faster-whisper loading + the voice-command transcribe call
│   ├── preflight.py        host self-check (`bot.py --preflight`, `voice doctor`)
│   ├── voice_mix.py        speech over music (ducking source, pause-safe)
│   ├── ffmpeg_source.py    FFmpeg music source: exit status, stderr tail, failure → words
│   └── formatters.py       pure helpers, unit-tested
├── cogs/                   Discord command groups
│   ├── music.py
│   ├── admin.py            prefix, help, autoleave/autojoin, errors
│   ├── playlists.py
│   ├── downloads.py        dl/convert/transcode/snip/chapters/ocr/normalize/…
│   ├── images.py           flip/rotate/crop/scale/caption/nobg/…
│   ├── recording.py        rolling voice buffer + -clip (voice chat)
│   ├── serverstats.py      message + voice-time counting, -stats
│   ├── voicedj.py          hands-free voice control (consent, jobs, dispatch, replies)
│   ├── tts.py              -tts / -ttsvoice / -ttsstop — talking for people without a mic
│   ├── lyrics.py           karaoke display, offset nudging
│   ├── karaoke.py          AI vocal removal, delivered as a file
│   └── games/
│       ├── poker.py
│       ├── blackjack.py
│       └── yahtzee.py
├── scripts/
│   ├── voice_eval.py       measure voice recognition on your own recordings
│   ├── voice_replay.py     replay recordings through the real front end: wake timing, what's heard
│   ├── wake_eval_synth.py  the wake word on other (synthetic) voices, speeds, noise, look-alikes
│   ├── cue_mixer_timing.py cue placed → first audible frame, with discord.py's player thread
│   ├── voice_plan_bench.py the live planner model on the test requests (needs the API key)
│   └── voice_coverage.py   regenerate VOICE_COVERAGE.md
└── tests/
```

## Security

`.env` and `token.txt` are both in `.gitignore`. Don't commit either.
If you ever do leak a token, regenerate it at
<https://discord.com/developers/applications> → your app → Bot → Reset Token.
The old token stops working immediately. The same goes for
`OPENROUTER_API_KEY`: give the key a spending limit on OpenRouter, and if it
ever leaks, delete it there and put a new one in `.env`.
