"""Synthesis helpers: Piper multi-speaker TTS with phoneme alignments -> 16 kHz audio
and the sample index where the wake phrase ends."""
import numpy as np, json, os
from functools import lru_cache
from piper import PiperVoice, SynthesisConfig
import scipy.signal as ss

VDIR = os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "valign")   # Piper voices, patched for alignments
BOUND = set(" ,.!?;:—-")

@lru_cache(maxsize=None)
def voice(name):
    return PiperVoice.load(os.path.join(VDIR, name + ".onnx"))

def n_speakers(name):
    return voice(name).config.num_speakers

def synth(name, text, spk=None, length_scale=1.0, noise_scale=0.667, noise_w=0.8, wake_words=0):
    """Returns (audio float32 16 kHz in [-1,1], wake_end_sample or None)."""
    v = voice(name)
    cfg = SynthesisConfig(speaker_id=spk if v.config.num_speakers > 1 else None,
                          length_scale=length_scale, noise_scale=noise_scale, noise_w_scale=noise_w)
    chunks = list(v.synthesize(text, cfg, include_alignments=True))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sr = v.config.sample_rate
    end = None
    if wake_words:
        c = chunks[0]
        al = c.phoneme_alignments
        if al is None:
            raise RuntimeError("no alignments")
        pos = 0; words = 0; in_word = False; last_word_end = None
        for a in al:
            ph = a.phoneme
            is_b = ph in BOUND or ph in ("^", "$")
            if not is_b:
                in_word = True
                pos += a.num_samples
                last_word_end = pos
            else:
                if in_word:
                    words += 1; in_word = False
                    if words == wake_words:
                        end = last_word_end; break
                pos += a.num_samples
        if end is None and in_word and words + 1 == wake_words:
            end = last_word_end
    if sr != 16000:
        g = np.gcd(16000, sr)
        audio = ss.resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        if end is not None:
            end = int(round(end * 16000 / sr))
    return audio.astype(np.float32), end
