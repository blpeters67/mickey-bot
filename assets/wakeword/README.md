# Instant wake word ("hey DJ")

Used by `core/wake_spotter.py`: a tiny model listens to each opted-in
speaker's audio and fires within about a tenth of a second after "hey DJ" is
said. The speech recogniser then checks those few words before the cue plays.

| File | What | Origin / licence |
|---|---|---|
| `melspectrogram.onnx` | 16 kHz audio → 32 mel bands every 10 ms | openWakeWord v0.5.1 release (github.com/dscripka/openWakeWord), Apache-2.0. sha256 `ba2b0e0f…b176f` |
| `embedding_model.onnx` | 76 mel frames → one 96-number embedding (every 80 ms) | openWakeWord v0.5.1 release, Apache-2.0; based on Google's speech_embedding model (Apache-2.0). sha256 `70d16429…75c1f` |
| `hey_dj.npz` | our classifier: the last 16 embeddings (1.28 s) → probability of "hey DJ"; `threshold` stored with it | trained for this bot, 2026-10-13 |

The two openWakeWord files are byte-identical to the official GitHub release
(sha256 checked against the release download).

## How `hey_dj.npz` was made

- **Network**: Flatten(16×96) → Linear 64 → LayerNorm → ReLU → Linear 64 →
  LayerNorm → ReLU → Linear 1 (same shape as openWakeWord's own heads), run
  with numpy — no torch in the bot.
- **Positives**: 4,500 synthetic utterances ×2 augmentations — "hey DJ" /
  "hey, DJ" / "hey D.J." / "hey deejay" / "hi DJ" / "yo DJ", sometimes after a
  filler ("um, …") and usually followed by a command — from Piper TTS voices
  (libritts_r's 904 speakers, VCTK, L2-ARCTIC, ARCTIC and 17 single-speaker
  voices; the end of the wake phrase from Piper's phoneme alignments).
  Labels: windows ending from 60 ms before to 360 ms after the end of "DJ".
- **Negatives**: 4,000 synthetic confusables and chat lines from the same
  voices ("hey Jay", "hey BJ", "hey, did you…", "the DJ", "hey Siri", …), the
  "hey D…" windows before the end of each positive and the command after it,
  LibriSpeech test-other (real read speech), and 800,000 windows (≈284 h) of
  openWakeWord's precomputed ACAV100M features (speech, music, noise).
- **Augmentation** (everything): level, room reverb, mic band-limiting,
  MUSAN noise / coloured noise / hum at 0–30 dB SNR, and a round trip through
  Opus at 24–96 kbps with occasional packet loss, decoded and resampled by the
  bot's own front end (`core/voice_frontend.py`'s decimator).
- **Training**: 15,000 steps of AdamW, negatives weighted up to 20× as
  training goes on, hard-negative mining every 3,000 steps.

## Measured (held-out voices and audio, 2026-10-13)

At the bundled threshold 0.3: hears 96 % of 500 held-out "hey DJ" clips
(voices never used in training), firing a median 60 ms *before* the end of
"DJ" (p90 80 ms after). It also fires on similar openings — 18 % of
held-out confusable clips, ≈13 per hour of general audio (openWakeWord's
10.7 h validation set) — which is why nothing is shown until the speech
recogniser agrees (`WakeVerifier`): on 60 held-out confusables it accepted
none.

Through the bot's real front end (Opus packets → decoder → decimator →
Silero VAD → spotter): 8/8 held-out "hey DJ" sentences fired between 175 ms
before and 80 ms after the end of "DJ"; 5 of 6 other sentences didn't fire
at all (the sixth, "hey, did you finish the build?", is for the verifier).
Extra CPU: about 21 ms per second of a speaker's audio on a 2.1 GHz Xeon core.

Not measured: real people's voices and microphones. `voice diag` counts what
it does live — fired, cued, turned down by the recogniser, confirmed or
contradicted by the full transcript, missed.

## Measured again (2026-10-14): real recordings, the faster check, other voices

All on a 2-core Xeon (2.1 GHz) sandbox — not the bot's own host, and not live
Discord. Scripts: `scripts/voice_replay.py`, `scripts/wake_eval_synth.py`,
`scripts/cue_mixer_timing.py`.

**11 real microphone recordings** (3–18 s requests), replayed as Discord
delivers them — Opus at 64 kbps in 20 ms packets → the bot's decoder,
decimator, Silero VAD and spotter. The end of "DJ" is annotated from
`large-v3` word timings, refined to where the voiced sound stops.

| | idle | recogniser busy on another clip | two speakers 50 ms apart |
|---|---|---|---|
| detector fired | 11/11 | 11/11 | 22/22 |
| first fire vs end of "DJ" | median −70 ms (p90 −20, max +30) | same | same |
| check (short window) | median 136 ms, p90 163 | median 181 ms | queued behind each other |
| check heard "Hey DJ" | 11/11 first fires | 11/11 | 22/22 |
| **cue: end of "DJ" → first cue frame** | **median 72 ms, p90 172, p95 271** | **median 125, p90 220, p95 385** | **median 150, p90 276, p95 290, max 604** |

Before this round the cue waited for the 30-second check: ~0.85 s here
(0.85–0.89 s measured on the i7 by Codex) on top of the fire. The old and new
checks agreed on 17/17 fires of these recordings. The cue estimate is fire +
queue wait + check + up to one 20 ms frame; it isn't microphone-to-ear —
the speaker's client, Discord's servers and the listener's jitter buffer add
their own delay, which nothing here measures.

**Mixer** (discord.py's real player thread, fake transport; 200 cues each):
from the cue being placed to its first audible frame going to the transport —
nothing playing: median 0.4 ms (max 1.3); over music: 10.7 ms (p95 19.5 — the
next 20 ms frame); over a `-tts` line being spoken: 10.9 ms (p95 19.5); over
paused music: 0.4 ms (max 1.7). Before this round the cue was a line in the
speech queue: behind a line being spoken it waited for that line to end, and
a full queue refused it.

**Other voices** — flite's 5 synthetic voices (US/Scottish, male/female;
robotic, unlike a real microphone), 8 "hey DJ …" sentences and 20
look-alikes ("hey Jay", "hey BJ", "hey TJ", "a DJ set", "DJ Khaled", "hey
Siri", "heyday", …), at 0.85/1.0/1.2× speed, clean and with a 10 dB
pink-noise-plus-chords bed:

- The detector fired on 237/240 wake sentences and on 270/600 look-alikes
  (45 %). Its scores don't separate them: at ≥ 0.99, 165/240 wakes but also
  93/600 look-alikes. So cueing on the score alone (`VOICE_FAST_WAKE_INSTANT`)
  would be wrong about one look-alike in seven; it stays off, and the check
  stays.
- The short looks alone (5 s, then 10 s), all three speeds, clean: 96/120
  wake sentences accepted, 16/300 look-alikes let through — "A DJ set at the
  club" (the "a DJ" spelling, which the command grammar also accepts as a
  wake word because the recogniser writes "hey" as "a"), "hey BJ", "hey
  Deja", "hey PJ". Dropping the "a" spelling would turn away 14 real wakes to
  stop 7 look-alikes, so it stays.
- With 10 dB noise the short looks accept few (23/120) — the recogniser
  hears "Thank you, Jay" — which is what the last look is for. A wake no look
  accepts still gets its cue from the full transcript, the old slow way; what
  the command does isn't affected either way.
- Which check: the same fires, every way (flite voices at normal speed; 40
  wake sentences and 100 look-alikes per row; accepted wakes / look-alikes
  let through / time to accept, median and p90):

  | check | clean | 20 dB | 10 dB |
  |---|---|---|---|
  | 30-second pass (before this round) | 39 / 8 · 944, 1007 ms | 26 / 3 · 947, 964 ms | 14 / 1 · 1025 ms |
  | 5 s window, then 10 s | 36 / 6 · 136, 179 ms | 23 / 4 · 154, 370 ms | 7 / 1 · 174 ms |
  | **5 s, then 10 s, then the 30-second pass (now)** | **39 / 9 · 136, 351 ms** | **29 / 5 · 343, 1319 ms** | **15 / 1 · 1277 ms** |

  So the short windows alone hear fewer of the hard wakes; with the
  30-second pass as the last look, nothing is lost against the old check and
  the clear ones — all 11 real recordings — are still accepted in ~0.15 s. The cost
  is checker time on a "no" (≈1.3 s here instead of ≈0.35 s), which only
  matters if several people's fires queue up at once (each waits at most
  1.5 s, then the transcript cues instead). `VOICE_FAST_WAKE_VERIFY_FULL=false`
  drops the last look.

Not measured: real other people, accents and microphones (the synthetic
voices are a stand-in, and a poor one for accents); the bot host's own
numbers; the delay Discord itself adds before a listener hears the cue.
