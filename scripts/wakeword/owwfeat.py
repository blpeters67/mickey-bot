"""Streaming openWakeWord-compatible features (melspectrogram -> 96-d embeddings
every 80 ms), numpy + onnxruntime only. Mirrors openwakeword.utils.AudioFeatures."""
import numpy as np, onnxruntime as ort, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "wakeword")

def _sess(path):
    o = ort.SessionOptions(); o.inter_op_num_threads = 1; o.intra_op_num_threads = 1
    o.log_severity_level = 4
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=o)

MEL = _sess(os.path.join(D, "melspectrogram.onnx"))
EMB = _sess(os.path.join(D, "embedding_model.onnx"))

def melspec(x):
    """x: float32 samples at int16 scale, shape (n,). Returns (frames, 32)."""
    out = MEL.run(None, {"input": x[None, :].astype(np.float32)})[0]
    return np.squeeze(out) / 10.0 + 2.0

def embed(mel_windows):
    """(B, 76, 32) -> (B, 96)"""
    return EMB.run(None, {"input_1": mel_windows[..., None].astype(np.float32)})[0].reshape(len(mel_windows), 96)

def clip_features(x16):
    """Whole clip (float32 int16-scale, 16 kHz) -> embeddings (T, 96), one per 80 ms,
    the t-th one covering mel frames up to 8*(t+1) + ... (batch, as oww _get_embeddings)."""
    spec = melspec(x16)
    wins = [spec[i:i+76] for i in range(0, spec.shape[0], 8) if spec[i:i+76].shape[0] == 76]
    if not wins: return np.zeros((0, 96), np.float32)
    return embed(np.stack(wins))

class Stream:
    """Streaming per-speaker feature state."""
    CH = 1280
    def __init__(self):
        rng = np.random.default_rng(0)
        self.raw = np.zeros(480, np.float32)          # history for the mel hop context
        self.mel = np.ones((76, 32), np.float32)      # as oww
        init = clip_features((rng.integers(-1000, 1000, 16000 * 4)).astype(np.float32))
        self.feat = init[-16:].copy()
        self.pending = np.zeros(0, np.float32)
    def push(self, x16):
        """Feed int16-scale float samples. Returns the number of new 80 ms embedding frames."""
        buf = np.concatenate([self.pending, x16]) if len(self.pending) else x16
        n = len(buf) // self.CH
        new = 0
        for i in range(n):
            chunk = buf[i*self.CH:(i+1)*self.CH]
            spec = melspec(np.concatenate([self.raw[-480:], chunk]))
            self.raw = chunk
            self.mel = np.vstack([self.mel, spec])[-200:]
            e = embed(self.mel[-76:][None])[0]
            self.feat = np.vstack([self.feat[1:], e[None]])
            new += 1
        self.pending = buf[n*self.CH:]
        return new
    def window(self):
        return self.feat[None].astype(np.float32)     # (1, 16, 96)
