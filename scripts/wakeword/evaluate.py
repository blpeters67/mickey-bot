"""Evaluate a trained head the way the bot will use it (numpy forward pass over
streaming windows): recall + latency on held-out voices, false alarms on held-out
confusables, LibriSpeech and the openWakeWord validation stream (10.7 h)."""
import os, sys, json
import numpy as np

FD = os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "feats")
DD = os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "data")
W = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "model", "head.npz")
P = dict(np.load(W))
names = sorted({k.rsplit(".", 1)[0] for k in P})


def forward(x):
    """x: (N, 16, 96) -> sigmoid scores. Mirrors nn.Sequential(Flatten, Linear, LN, ReLU, Linear, LN, ReLU, Linear)."""
    h = x.reshape(len(x), -1).astype(np.float32)
    def lin(h, i):
        return h @ P[f"net.{i}.weight"].T + P[f"net.{i}.bias"]
    def ln(h, i, eps=1e-5):
        m = h.mean(-1, keepdims=True); v = h.var(-1, keepdims=True)
        return (h - m) / np.sqrt(v + eps) * P[f"net.{i}.weight"] + P[f"net.{i}.bias"]
    h = np.maximum(ln(lin(h, 1), 2), 0)
    h = np.maximum(ln(lin(h, 4), 5), 0)
    z = lin(h, 7)[:, 0]
    return 1 / (1 + np.exp(-z))


def windows(seq):
    seq = np.asarray(seq, np.float32)
    J = len(seq)
    if J < 16:
        return np.zeros((0, 16, 96), np.float32), np.zeros(0)
    idx = np.arange(15, J)
    X = np.stack([seq[j - 15:j + 1] for j in idx])
    ends = 1280 * idx + 12512
    return X, ends


def seq_scores(seqs):
    out = []
    for s in seqs:
        X, ends = windows(s)
        out.append((forward(X) if len(X) else np.zeros(0), ends))
    return out


pos = np.load(os.path.join(FD, "eval_pos.npz"), allow_pickle=True)
adv = np.load(os.path.join(FD, "eval_adv.npz"), allow_pickle=True)
lib = np.load(os.path.join(FD, "eval_libri.npz"), allow_pickle=True) if os.path.exists(
    os.path.join(FD, "eval_libri.npz")) else None
P_SC = seq_scores(pos["seqs"])
P_WE = [int(m[3]) for m in pos["meta"]]
P_TXT = [m[2] for m in pos["meta"]]
A_SC = seq_scores(adv["seqs"])
A_TXT = [m[2] for m in adv["meta"]]
L_SC = seq_scores(lib["seqs"]) if lib is not None else []
val = np.load(os.path.join(DD, "validation_set_features.npy"), mmap_mode="r")
# sliding 16-frame windows over the validation stream, in chunks
V_MAX = []
step = 50000
for i in range(0, len(val) - 16, step):
    blk = np.asarray(val[i:i + step + 15], np.float32)
    X = np.lib.stride_tricks.sliding_window_view(blk, (16, 96))[:, 0]
    V_MAX.append(forward(X))
V = np.concatenate(V_MAX)
VAL_HOURS = len(val) * 0.08 / 3600
LIB_HOURS = sum(len(s[1]) for s in L_SC) * 0.08 / 3600 if L_SC else 0


def fires(sc, thr, k=1):
    """indices where k consecutive windows are >= thr (detection at the k-th)."""
    above = sc >= thr
    if k == 1:
        return np.flatnonzero(above)
    run = np.convolve(above.astype(int), np.ones(k, int), "valid") >= k
    return np.flatnonzero(run) + k - 1


def count_events(sc, thr, k=1, refractory=25):
    """Separate detections (a detection silences the next `refractory` windows ~2 s)."""
    idx = fires(sc, thr, k)
    n, last = 0, -10**9
    for i in idx:
        if i - last > refractory:
            n += 1
            last = i
    return n


def report(thr, k=1):
    hits, lat, early = 0, [], 0
    for (sc, ends), we in zip(P_SC, P_WE):
        idx = fires(sc, thr, k)
        good = [i for i in idx if we - 0.2 * 16000 <= ends[i] <= we + 1.2 * 16000]
        if any(ends[i] < we - 0.2 * 16000 for i in idx):
            early += 1
        if good:
            hits += 1
            lat.append((ends[good[0]] - we) / 16000)
    a_fa = sum(1 for sc, _ in A_SC if len(fires(sc, thr, k)))
    l_fa = sum(count_events(sc, thr, k) for sc, _ in L_SC)
    # gated: only the first ~3 s of each utterance (the wake word must open it)
    l_fa_gated = sum(1 for sc, ends in L_SC if any(1.9 * 16000 <= ends[i] <= 5.3 * 16000
                                                   for i in fires(sc, thr, k)))
    v_fa = count_events(V, thr, k)
    lat = np.array(lat) if lat else np.array([np.nan])
    return dict(thr=thr, k=k, recall=hits / len(P_SC), early=early, lat_med=float(np.median(lat)),
                lat_p90=float(np.percentile(lat, 90)), adv_fa=a_fa / max(1, len(A_SC)),
                libri_fa_per_h=l_fa / max(LIB_HOURS, 1e-9), libri_fa_gated=l_fa_gated / max(1, len(L_SC)),
                val_fa_per_h=v_fa / VAL_HOURS)


if __name__ == "__main__":
    print(f"eval: {len(P_SC)} positive clips (held-out voices), {len(A_SC)} confusable, "
          f"{len(L_SC)} LibriSpeech utts ({LIB_HOURS:.2f} h), validation stream {VAL_HOURS:.1f} h")
    for k in (1, 2):
        for thr in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
            r = report(thr, k)
            print(f"k={k} thr={thr:.2f}  recall {r['recall']:.3f} (early {r['early']})  latency med "
                  f"{r['lat_med']*1000:.0f} ms p90 {r['lat_p90']*1000:.0f} ms | FA: confusable "
                  f"{r['adv_fa']:.3f}/utt, libri {r['libri_fa_per_h']:.2f}/h (gated {r['libri_fa_gated']:.4f}/utt), "
                  f"val {r['val_fa_per_h']:.2f}/h", flush=True)
    # which confusables fire at 0.9
    bad = {}
    for (sc, _), t in zip(A_SC, A_TXT):
        if len(fires(sc, 0.9)):
            bad[t] = bad.get(t, 0) + 1
    print("confusables firing at 0.9:", sorted(bad.items(), key=lambda x: -x[1])[:25])
    missed = [t for (sc, ends), we, t in zip(P_SC, P_WE, P_TXT)
              if not any(we - 3200 <= ends[i] <= we + 19200 for i in fires(sc, 0.9))]
    print(f"missed at 0.9 ({len(missed)}):", missed[:25])
