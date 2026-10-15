"""Train the "hey DJ" head on openWakeWord embeddings (16 x 96 windows).

Positives: windows ending at the end of the wake phrase (synthetic voices, augmented).
Negatives: confusable phrases, the "hey D…" prefix windows, commands after the wake
phrase, LibriSpeech (real speech) and ACAV100M features (speech, music, noise).

Writes weights as .npz for a numpy forward pass (no torch in the bot).
"""
import glob, os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
FD = os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "feats")
DD = os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("KWS_DIR", "/tmp/kws"), "model")
STEPS = int(os.environ.get("STEPS", 12000))
HID = int(os.environ.get("HID", 64))
SEED = int(os.environ.get("SEED", 0))
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def load(name):
    d = np.load(os.path.join(FD, name), allow_pickle=True)
    return d["X"], d["L"]


Xp, Lp = load(os.environ.get("POS_FILE", "train_pos.npz"))
POS = Xp[Lp == 1]
NEG_POS = Xp[Lp == 0]                       # "hey d…" prefixes and the commands after
ADV, _ = load("train_adv.npz")
_lib = [load(n)[0] for n in ("train_libri_clean.npz", "train_libri_other.npz")
        if os.path.exists(os.path.join(FD, n))]
LIB = np.concatenate(_lib) if _lib else ADV[:1000]
class Parts:
    """Several memory-mapped arrays indexed as one (the ACAV features are 2.4 GB)."""

    def __init__(self, arrays):
        self.arrays = arrays
        self.starts = np.cumsum([0] + [len(a) for a in arrays])

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), 16, 96), np.float16)
        part = np.searchsorted(self.starts, idx, side="right") - 1
        for k in np.unique(part):
            sel = part == k
            out[sel] = self.arrays[k][np.sort(idx[sel] - self.starts[k])][np.argsort(np.argsort(idx[sel] - self.starts[k]))]
        return out


ACAV = Parts([np.memmap(p, dtype=np.float16, mode="r").reshape(-1, 16, 96)
              for p in sorted(glob.glob(os.path.join(DD, "acav_part*.bin")))
              if os.path.getsize(p) == 100000 * 3072])     # complete downloads only
print(f"pos {len(POS)}  neg-in-pos {len(NEG_POS)}  adv {len(ADV)}  libri {len(LIB)}  acav {len(ACAV)}",
      flush=True)

POOLS = [("pos", POS, 1), ("negpos", NEG_POS, 0), ("adv", ADV, 0), ("libri", LIB, 0), ("acav", ACAV, 0)]
MIX = {"pos": float(os.environ.get("MIX_POS", 0.2)), "negpos": 0.1, "adv": 0.22, "libri": 0.2, "acav": 0.28}
NEG_W_MAX = float(os.environ.get("NEG_W_MAX", 20))     # negatives weigh more and more (precision)
MINE_EVERY = int(os.environ.get("MINE_EVERY", 0)) or None


class Head(nn.Module):
    def __init__(self, hid=HID):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(16 * 96, hid), nn.LayerNorm(hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def batch(n, hard=None):
    xs, ys = [], []
    for name, pool, lab in POOLS:
        k = int(round(n * MIX[name]))
        if hard is not None and name in hard and len(hard[name]) and lab == 0:
            # half of a negative pool's share from its hardest examples
            kh = k // 2
            idx = hard[name][rng.integers(0, len(hard[name]), kh)]
            xs.append(np.asarray(pool[idx])); ys.append(np.full(kh, lab))
            k -= kh
        idx = rng.integers(0, len(pool), k)
        xs.append(np.asarray(pool[idx])); ys.append(np.full(k, lab))
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32))
    y = torch.from_numpy(np.concatenate(ys).astype(np.float32))
    return x, y


def scores(model, X, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            blk = X[np.arange(i, min(i + bs, len(X)))] if isinstance(X, Parts) else X[i:i + bs]
            out.append(torch.sigmoid(model(torch.from_numpy(np.asarray(blk, np.float32)))).numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0)


model = Head()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss(reduction="none")
hard = None
t0 = time.time()
mine_at = set(range(MINE_EVERY, STEPS, MINE_EVERY)) if MINE_EVERY else {STEPS // 3, 2 * STEPS // 3}
for step in range(1, STEPS + 1):
    if step in mine_at:
        # hard negative mining: the negatives the model currently likes most
        hard = {}
        for name, pool, lab in POOLS:
            if lab == 0:
                s = scores(model, pool)
                hard[name] = np.argsort(-s)[: max(200, len(pool) // 50)]
        for g in opt.param_groups:
            g["lr"] *= 0.6
        print(f"step {step}: mined hard negatives " +
              ", ".join(f"{k} top {s:.3f}" for k, s in
                        ((n, float(scores(model, np.asarray(p[hard[n][:1]]))[0])) for n, p, l in POOLS if l == 0)),
              flush=True)
    x, y = batch(1024, hard)
    # small embedding noise: robustness to mics/codecs the training audio lacks
    x = x + 0.05 * torch.randn_like(x)
    neg_w = 1.0 + (NEG_W_MAX - 1.0) * min(1.0, step / (0.7 * STEPS))
    w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, neg_w))
    loss = (lossf(model(x), y * 0.98 + 0.01) * w).sum() / w.sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

# export: plain arrays for a numpy forward pass
sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
np.savez(os.path.join(OUT, "head.npz"), **sd)
torch.save(model.state_dict(), os.path.join(OUT, "head.pt"))
print("saved", OUT, list(sd), flush=True)
