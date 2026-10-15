"""Generate "hey DJ" wake-word training/eval features.

Positives: Piper TTS (many speakers) saying a wake variant, optionally followed by a
command; the sample where the wake phrase ends comes from phoneme alignments.
Negatives: confusable phrases and general chat (same voices), LibriSpeech (real speech).
Every clip goes through: level, optional reverb / band-limit, background noise, Opus
round trip at 48 kHz, and the bot's own 48 -> 16 kHz decimator - then openWakeWord
features (batch mode == the streaming extractor, verified).

Usage: make_data.py <split: train|eval> <kind: pos|adv> <n> <seed> <out.npz>
       make_data.py libri <parquet> <stride> <seed> <out.npz>
"""
import os, sys, random, glob, io, time
import numpy as np

import onnxruntime as _ort
_Orig = _ort.SessionOptions
class _OneThread(_Orig):
    def __init__(self):
        super().__init__()
        self.intra_op_num_threads = 1
        self.inter_op_num_threads = 1
_ort.SessionOptions = _OneThread

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import scipy.signal as ss
import soundfile as sf
import gen
import owwfeat as F
from core.voice_frontend import Decimator, pcm48_stereo_to_mono_float

NOISE = sorted(glob.glob(os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "data", "noise", "*.wav")))
SR = 16000

MULTI = {  # voice: (train speaker range, eval speaker range)
    "en_US-libritts_r-medium": (range(0, 800), range(800, 904)),
    "en_GB-vctk-medium": (range(0, 90), range(90, 109)),
    "en_US-l2arctic-medium": (range(0, 20), range(20, 24)),
    "en_US-arctic-medium": (range(0, 15), range(15, 18)),
    "en_GB-aru-medium": (range(0, 10), range(10, 12)),
    "en_GB-semaine-medium": (range(0, 3), range(3, 4)),
}
SINGLE_TRAIN = ["en_US-hfc_female-medium", "en_US-hfc_male-medium", "en_US-joe-medium",
                "en_US-john-medium", "en_US-kusal-medium", "en_US-lessac-medium",
                "en_US-ljspeech-medium", "en_US-norman-medium", "en_US-reza_ibrahim-medium",
                "en_US-sam-medium", "en_US-bryce-medium", "en_US-danny-low", "en_US-kathleen-low",
                "en_GB-southern_english_female-low", "en_GB-northern_english_male-medium",
                "en_GB-jenny_dioco-medium", "en_GB-alba-medium"]
SINGLE_EVAL = ["en_US-ryan-medium", "en_US-amy-medium", "en_GB-alan-medium",
               "en_US-kristin-medium", "en_GB-cori-medium"]


def speakers(split):
    out = []
    for v, (tr, ev) in MULTI.items():
        rng = tr if split == "train" else ev
        out += [(v, s) for s in rng]
    out += [(v, None) for v in (SINGLE_TRAIN if split == "train" else SINGLE_EVAL)]
    return out


# (text, words in the wake phrase, weight)
WAKES = [("Hey DJ", 2, 40), ("Hey, DJ", 2, 12), ("Hey D.J.", 2, 10), ("Hey deejay", 2, 8),
         ("Hi DJ", 2, 12), ("Hey dee jay", 3, 6), ("Yo DJ", 2, 5), ("Hey DJ DJ", 2, 0)]
WAKES = [w for w in WAKES if w[2]]
FILLERS = ["Um,", "Uh,", "Okay,", "So,", "Hey,", "Yo,", "Oh,", "Alright,"]
COMMANDS = """skip this song|play Fireflies by Owl City|turn it up|turn the volume down|pause|pause the music
what's playing right now|who sings this|play something chill|skip|next song|can you play some Queen
stop|resume|loop this song|shuffle the queue|play the next one|play Mr. Brightside
put on some Taylor Swift|what song is this|louder|play it again|restart the song
add Bohemian Rhapsody to the queue|play the first three songs off Vices and Virtues|can you skip this
I want to hear two songs|clear the queue|play my playlist|turn it down a little|play some lo-fi beats
skip this one please|can you play I Write Sins Not Tragedies|play No Disrespect by Set It Off
what's the capital of Australia|play Cordial by Set It Off|play something by Yung Gravy|volume fifty
play the latest Drake song|go back to the start|play that again|can you put on some rock music
I need some workout music|play the song from Top Gun|skip two songs|stop the music
play The Pretender by the Foo Fighters|can you turn it up a bit|play some country music|unpause
what's the weather like|play Never Gonna Give You Up|keep going|play the album Abbey Road
play Danger Zone|loop the queue|play some jazz|how loud is it|play the next song please
can you play something upbeat|play Sweater Weather|play Blinding Lights|play Hotel California
put this on repeat|turn the music off|play some music|play anything|surprise me""".replace("\n", "|").split("|")
NEAR = """Hey|Hey!|Hey Jay|Hey G|Hey Gee|Hey D|Hey day|Hey, TJ|Hey BJ|Hey PJ|Hey CJ|Hey AJ|Hey EJ|Hey KJ
Hey RJ|Hey MJ|Hey DeeDee|Hey Deja|Hey Jamie|Hey Jake|Hey Jade|Hey Dee|Hey Jeff|Hey dude|Hey guys
Hey dear|Hey there|Hey you|Hey Dave|Hey Danny|Hey Siri|Hey Google|Okay Google|Alexa|Hey Jarvis
Hey baby|Hey, today|Hey, did you|Hey, did you see that?|Hey did ya hear that|Hey, didn't you say that
Hey, they did it|Hey, Daisy|Hey, Debbie|A DJ|The DJ|DJ|DJs|My DJ|Our DJ|That DJ|DJ Khaled
the DJ is playing|a DJ set|he's a DJ|Hey, he's a DJ now|Hey, Jaden|Hey Deacon|Hey Dejan|Hey Diego
Hey Duke|Hey James|Hey Jenny|Hey Gina|Hey Jay Z|Hey, did Jay|Hey, deja vu|Hey, wait|Hey, what's up
Hey guys, can you hear me|Hey, I'm back|Hey, did you get that|Hey, can you|Hey, hey, hey|Hey, it's me
Hey, the DJ|Hey, the deejay said|They DJ'd|Hey Jude|Hey Ya|Hey, Joe|Hey, Jo|Hey, dummy|Hey, Jeremy
Hey, DJ's not here|Okay|Okay, Jay|Hey, Gigi|Hey, Teddy|Hey, Eddie|Hey, Edgy|Hey, edgy|Hey, Reggie
Hey, Jenny|Hey, Kenji|Hey, Sergei|Hey Freddie|Hey, ready|Hey, Shady""".replace("\n", "|").split("|")
GENERAL = """Did you see that clip yesterday?|I'm going to grab some food real quick|Can you hear me now?
That was the best game we've played all week|Wait, who just joined the call?|Let me check my inventory
I think the server is lagging again|Okay, I'm ready when you are|Turn your mic down, it's echoing
We should do another raid tonight|My internet is so slow today|Hold on, my dog is barking
That song is actually really good|Who picked this music?|I have work in the morning
Did anybody else get the update?|Let's go to the nether|I died again, this is ridiculous
Can somebody send me the link?|I'll be right back|That's hilarious|No way, that's crazy
What are we doing this weekend?|I can't find the diamonds anywhere|He's been playing all day
Do you want to play another round?|The DJ at the wedding was terrible|Turn it up, I love this part
Skip it, I hate this one|Okay, today was a long day|Jay, did you finish the build?
They played this song at the party|I need a new keyboard|Did you eat yet?|What time is it there?
I'm so tired right now|Let me share my screen|Can you see my screen?|That's not fair
Hey, can somebody help me with this?|We need more wood|Let's build a house over there
Yo, what's up everybody|Hello? Is anyone there?|Good morning|Good night guys
Give me a second|Sorry, I was muted|You're breaking up|Say that again?
I'm playing on my laptop today|My headphones are dying|This map is huge|Where are you?
Follow me, I know a shortcut|Watch out behind you|Nice shot|Get over here
I'm out of ammo|Does anyone have extra food?|Let's trade|I'll give you two diamonds for that
What's the plan?|We should probably sleep|It's getting dark|The creepers are coming
I just got a new job|My mom is calling me|Brb, bathroom|I'm back
That's what she said|Bro, what are you doing?|Dude, that was insane|Okay, okay, okay
What did you say?|Huh?|Wait, what?|Yeah, I know|No, I don't think so|Maybe later
Let's do it|Sounds good|For real?|I'm dead|Stop it|Leave me alone|Come on
Play the game|Play it cool|Is this the new DJ?|Who's the DJ tonight?|I want to be a DJ
The deejay played my favorite song|Jade and Jake are coming over|Gee, I don't know
Today is Tuesday|Did you hear about the deal?|Daisy is my cat's name|Deja vu, man
Hey, it's me again|Hey, how are you doing?|Hey, guess what|Hey, look at this""".replace("\n", "|").split("|")


def pick_speaker(rng, spk_list):
    # libritts_r dominates the list already (800 of ~1050); singles are added with extra weight
    return spk_list[rng.integers(len(spk_list))]


def make_text(rng, kind):
    """-> (text, wake_words or 0)"""
    if kind == "pos":
        w = WAKES[rng.choice(len(WAKES), p=np.array([x[2] for x in WAKES]) / sum(x[2] for x in WAKES))]
        text, ww = w[0], w[1]
        if rng.random() < 0.12:
            f = FILLERS[rng.integers(len(FILLERS))]
            text, ww = f"{f} {text}", ww + 1
        if rng.random() < 0.35:
            text += [".", "!", "?"][rng.integers(3)]
        else:
            cmd = COMMANDS[rng.integers(len(COMMANDS))]
            text += (", " if rng.random() < 0.5 else " ") + cmd + "."
        return text, ww
    if rng.random() < 0.65:
        t = NEAR[rng.integers(len(NEAR))]
        if rng.random() < 0.45 and not t.endswith(("?", "!")):
            t += (", " if rng.random() < 0.5 else " ") + (COMMANDS if rng.random() < 0.5 else GENERAL)[rng.integers(40)].lower().rstrip("?.!") + "."
        return t, 0
    return GENERAL[rng.integers(len(GENERAL))], 0


# ── augmentation ────────────────────────────────────────────────────────────

def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


_noise_cache = {}


def noise_segment(rng, n):
    kind = rng.random()
    if kind < 0.6 and NOISE:
        p = NOISE[rng.integers(len(NOISE))]
        if p not in _noise_cache:
            a, sr = sf.read(p, dtype="float32", always_2d=True)
            a = a.mean(1)
            if sr != SR:
                a = ss.resample_poly(a, SR, sr).astype(np.float32)
            if len(_noise_cache) > 40:
                _noise_cache.pop(next(iter(_noise_cache)))
            _noise_cache[p] = a
        a = _noise_cache[p]
        if len(a) < n:
            a = np.tile(a, int(np.ceil(n / max(1, len(a)))) + 1)
        s = rng.integers(0, len(a) - n + 1)
        return a[s:s + n].copy()
    if kind < 0.85:   # coloured noise
        w = rng.standard_normal(n).astype(np.float32)
        beta = rng.choice([0.0, 1.0, 2.0])
        if beta:
            f = np.fft.rfft(w)
            k = np.arange(len(f)); k[0] = 1
            w = np.fft.irfft(f / k ** (beta / 2), n).astype(np.float32)
        return w
    # tones / hum / simple "music"
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for _ in range(rng.integers(1, 5)):
        f0 = float(rng.choice([50, 60, 110, 220, 330, 440, 523, 659, 784]) * rng.uniform(0.97, 1.03))
        for h in range(1, 4):
            out += (0.5 ** h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)).astype(np.float32)
    return out


def rir(rng):
    rt60 = rng.uniform(0.12, 0.7)
    n = int(SR * rt60 * 1.2)
    t = np.arange(n) / SR
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60).astype(np.float32)
    h[:int(0.002 * SR)] = 0
    h[0] = 1.0 / rng.uniform(0.3, 1.0)       # direct path
    return h / np.max(np.abs(h))


_ENC = {}


def opus_roundtrip(x16, rng):
    """16 kHz float -> 48 kHz stereo s16 -> Opus -> decode (+loss/PLC) -> the bot's decimator."""
    import discord.opus as dop
    x48 = ss.resample_poly(x16, 3, 1).astype(np.float32)
    pcm = (np.clip(x48, -1, 1) * 32767).astype("<i2")
    st = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    fr = 960 * 2
    n = len(st) // fr
    br = int(rng.choice([24, 32, 48, 64, 96]))
    enc = dop.Encoder()
    enc.set_bitrate(br)
    dec = dop.Decoder()
    loss = float(rng.choice([0.0, 0.0, 0.0, 0.0, 0.02, 0.05]))
    out = []
    for i in range(n):
        pkt = enc.encode(st[i * fr:(i + 1) * fr].tobytes(), 960)
        if loss and rng.random() < loss:
            out.append(pcm48_stereo_to_mono_float(dec.decode(None, fec=False)))
        else:
            out.append(pcm48_stereo_to_mono_float(dec.decode(pkt, fec=False)))
    mono48 = np.concatenate(out)
    return Decimator().process(mono48)


def augment(x, rng, wake_end):
    """x: float32 16 kHz speech. Returns (int16-scale float32 audio, wake_end shifted)."""
    pre = int(SR * rng.uniform(1.9, 2.8))
    post = int(SR * rng.uniform(0.7, 1.3))
    x = x / (_rms(x) + 1e-9) * 10 ** (rng.uniform(-32, -14) / 20)
    if rng.random() < 0.4:
        h = rir(rng)
        wet = ss.fftconvolve(x, h)[:len(x) + len(h) // 3].astype(np.float32)
        wet = wet / (_rms(wet) + 1e-9) * _rms(x)
        x = np.concatenate([x, np.zeros(len(wet) - len(x), np.float32)])
        mix = rng.uniform(0.3, 1.0)
        x = (1 - mix) * x + mix * wet
    y = np.concatenate([np.zeros(pre, np.float32), x, np.zeros(post, np.float32)])
    if rng.random() < 0.5:
        b, a = ss.butter(2, rng.uniform(90, 300) / (SR / 2), "high")
        y = ss.lfilter(b, a, y).astype(np.float32)
        b, a = ss.butter(4, rng.uniform(3400, 7600) / (SR / 2), "low")
        y = ss.lfilter(b, a, y).astype(np.float32)
    if rng.random() < 0.8:
        nz = noise_segment(rng, len(y))
        snr = rng.uniform(0, 30)
        nz = nz / (_rms(nz) + 1e-9) * _rms(x) * 10 ** (-snr / 20)
        if rng.random() < 0.25:     # noise gate: background only while talking (Krisp-like)
            env = np.zeros(len(y), np.float32)
            env[pre:pre + len(x)] = 1
            env = np.convolve(env, np.ones(1600) / 1600, mode="same").astype(np.float32)
            nz = nz * env
        y = y + nz
    peak = np.max(np.abs(y))
    if peak > 0.99:
        y = y * (0.99 / peak)
    delay = 0
    if rng.random() < 0.85:
        y2 = opus_roundtrip(y, rng)
        delay = OPUS_DELAY
        y = y2
    return (y * 32767).astype(np.float32), (None if wake_end is None else wake_end + pre + delay)


def _measure_opus_delay():
    rng = np.random.default_rng(0)
    x = np.zeros(SR * 2, np.float32)
    x[SR] = 0.9
    x = np.convolve(x, ss.windows.hann(64), mode="same").astype(np.float32)
    import discord.opus as dop
    x48 = ss.resample_poly(x, 3, 1).astype(np.float32)
    pcm = (np.clip(x48, -1, 1) * 32767).astype("<i2")
    st = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    enc, dec = dop.Encoder(), dop.Decoder()
    enc.set_bitrate(64)
    out = [pcm48_stereo_to_mono_float(dec.decode(enc.encode(st[i * 1920:(i + 1) * 1920].tobytes(), 960)))
           for i in range(len(st) // 1920)]
    y = Decimator().process(np.concatenate(out))
    return int(np.argmax(np.abs(y)) - SR)


OPUS_DELAY = 0


# ── features & labels ──────────────────────────────────────────────────────

POS_LO, POS_HI = -0.06, 0.36        # window end relative to the wake end, seconds
NEG_BEFORE = -0.30                  # windows ending earlier than this: negative ("hey d…")
NEG_AFTER = 2.0                     # windows ending this long after: the wake is out of view


def window_ends(J):
    return np.array([1280 * j + 12512 for j in range(J)])


def features_and_labels(y, wake_end, keep_bg=2, rng=None):
    e = F.clip_features(y)
    J = len(e)
    ends = window_ends(J)
    X, L = [], []
    for j in range(15, J):
        end = ends[j]
        if wake_end is None:
            lab = 0
        else:
            d = (end - wake_end) / SR
            if POS_LO <= d <= POS_HI:
                lab = 1
            elif d < NEG_BEFORE or d > NEG_AFTER:
                lab = 0
            else:
                continue
        X.append(e[j - 15:j + 1])
        L.append(lab)
    return np.array(X, np.float16).reshape(-1, 16, 96), np.array(L, np.int8), e.astype(np.float16)


def run_synth(split, kind, n, seed, out):
    global OPUS_DELAY
    OPUS_DELAY = _measure_opus_delay()
    rng = np.random.default_rng(seed)
    spk = speakers(split)
    # group by voice to keep few models loaded
    jobs = []
    for i in range(n):
        if rng.random() < 0.22:     # extra weight on single-speaker voices / small sets
            pool = [s for s in spk if s[0] != "en_US-libritts_r-medium"]
        else:
            pool = spk
        jobs.append(pool[rng.integers(len(pool))])
    jobs.sort(key=lambda s: s[0])
    Xs, Ls, seqs, meta = [], [], [], []
    t0 = time.time()
    for i, (v, s) in enumerate(jobs):
        text, ww = make_text(rng, kind)
        try:
            x, we = gen.synth(v, text, s, length_scale=rng.uniform(0.78, 1.3),
                              noise_scale=rng.uniform(0.4, 0.9), noise_w=rng.uniform(0.5, 1.1),
                              wake_words=ww if kind == "pos" else 0)
        except Exception as e:
            print("synth fail", v, s, text, e, flush=True)
            continue
        if kind == "pos" and (we is None or we < 1000):
            continue
        reps = 2 if (kind == "pos" and split == "train") else 1
        for r in range(reps):
            y, wend = augment(x, rng, we if kind == "pos" else None)
            X, L, seq = features_and_labels(y, wend)
            Xs.append(X); Ls.append(L)
            if split == "eval":
                seqs.append(seq)
                meta.append((v, -1 if s is None else s, text, -1 if wend is None else int(wend)))
        if i % 200 == 0:
            print(f"{split} {kind} {i}/{n} {time.time() - t0:.0f}s", flush=True)
        if gen.voice.cache_info().currsize > 3:
            gen.voice.cache_clear()
    X = np.concatenate(Xs); L = np.concatenate(Ls)
    kw = {}
    if split == "eval":
        kw = dict(seqs=np.array(seqs, dtype=object), meta=np.array(meta, dtype=object))
    np.savez(out, X=X, L=L, opus_delay=OPUS_DELAY, **kw)
    print("saved", out, X.shape, int(L.sum()), f"{time.time() - t0:.0f}s", flush=True)


def run_libri(parquet, stride, seed, out, max_utts=None):
    global OPUS_DELAY
    OPUS_DELAY = _measure_opus_delay()
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    t = pq.read_table(parquet, columns=["audio"])
    col = t.column("audio").to_pylist()
    Xs, seqs = [], []
    t0 = time.time()
    for i, a in enumerate(col[:max_utts] if max_utts else col):
        x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        if sr != SR:
            x = ss.resample_poly(x, SR, sr).astype(np.float32)
        y, _ = augment(x, rng, None)
        X, L, seq = features_and_labels(y, None)
        Xs.append(X[::stride])
        if stride == 1:
            seqs.append(seq)
        if i % 200 == 0:
            print(f"libri {i}/{len(col)} {time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(Xs)
    kw = dict(seqs=np.array(seqs, dtype=object)) if seqs else {}
    np.savez(out, X=X, L=np.zeros(len(X), np.int8), **kw)
    print("saved", out, X.shape, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "libri":
        run_libri(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
                  int(sys.argv[6]) if len(sys.argv) > 6 else None)
    else:
        run_synth(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
