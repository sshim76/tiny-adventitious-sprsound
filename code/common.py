"""Shared data handling, models and training utilities for lightweight adventitious respiratory sound detection."""
import os, glob, json, math, time, pickle
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "data/src/sprsound")
torch.set_num_threads(2)
HOP_S = 0.01        # 10 ms frames
CROP = 400          # 4-s analysis window
GRID = 100          # windows start on a 1-s grid
N_MELS = 64
CAS_T = {"Wheeze", "Rhonchi", "Stridor", "Wheeze+Crackle"}       # continuous adventitious sounds
DAS_T = {"Coarse Crackle", "Fine Crackle", "Wheeze+Crackle"}      # discontinuous adventitious sounds
MIN_OVERLAP = 0.3   # seconds of an event that must fall inside a window for the window to be positive


# ---------------------------------------------------------------- data
def load_events():
    p = os.path.join(RES, "events.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    ev = {}
    for js in glob.glob(os.path.join(SRC, "BioCAS20*", "*_json", "**", "*.json"), recursive=True):
        a = json.load(open(js))
        ev[os.path.basename(js)[:-5]] = [(int(e["start"]) / 1000, int(e["end"]) / 1000, e["type"])
                                          for e in a.get("event_annotation", [])]
    pickle.dump(ev, open(p, "wb"))
    return ev


def load_index():
    """Recordings with event annotations (poor-quality recordings have none and are excluded)."""
    d = pd.read_csv(os.path.join(RES, "index.csv"), dtype={"patient": str})
    d = d[d.n_events > 0].copy()
    # some files carry no patient number; group them by release, age and sex (a conservative proxy cluster)
    miss = d.patient.isna()
    d.loc[miss, "patient"] = "anon" + d.loc[miss, "release"].astype(str) + "_" + d.loc[miss, "age"].astype(str) + "_" + d.loc[miss, "sex"].astype(str)
    ev = d.event_types.fillna("")
    d["cas"] = ev.apply(lambda s: int(any(t in CAS_T for t in s.split(";"))))
    d["das"] = ev.apply(lambda s: int(any(t in DAS_T for t in s.split(";"))))
    d["cohort"] = d.release.map({2022: "dev", 2023: "dev", 2024: "ext2024", 2025: "ext2025"})
    d["pneu"] = d.diagnosis.fillna("").str.startswith("Pneumonia").astype(int)
    return d.reset_index(drop=True)


_MELS = None
def mels():
    global _MELS
    if _MELS is None:
        z = np.load(os.path.join(ROOT, "data/cache/logmel.npz"))
        _MELS = {k: z[k] for k in z.files}
    return _MELS


def norm_stats(stems):
    m = mels(); a = np.concatenate([m[s].astype(np.float32).ravel()[::7] for s in stems])
    return float(a.mean()), float(a.std())


def pad_to(x, n):
    if x.shape[1] >= n:
        return x
    rep = int(math.ceil(n / x.shape[1]))
    return np.tile(x, (1, rep))[:, :n]


def starts_for(n_frames):
    n = max(n_frames, CROP)
    s = list(range(0, n - CROP + 1, GRID))
    if s[-1] != n - CROP:
        s.append(n - CROP)
    return s


def window_labels(events, t0_frames, n_frames_orig):
    """CAS/DAS labels of a 4-s window starting at frame t0 (events beyond a padded recording wrap around)."""
    a, b = t0_frames * HOP_S, (t0_frames + CROP) * HOP_S
    dur = n_frames_orig * HOP_S; lab = [0, 0]
    for s, e, t in events:
        for k in range(int(math.ceil(CROP / max(n_frames_orig, 1))) + 1):   # repeats when the recording is tiled
            ov = min(b, e + k * dur) - max(a, s + k * dur)
            if ov >= min(MIN_OVERLAP, 0.5 * (e - s)):
                if t in CAS_T: lab[0] = 1
                if t in DAS_T: lab[1] = 1
    return lab


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, df, mu, sd, events, soft=None, aug=True, noise_aug=False):
        self.s = df.stem.values; self.mu, self.sd, self.aug, self.ev, self.soft = mu, sd, aug, events, soft
        self.noise_aug = noise_aug
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        raw = mels()[self.s[i]].astype(np.float32); n0 = raw.shape[1]
        x = pad_to(raw, CROP); st = starts_for(x.shape[1]); j = np.random.randint(len(st)); t0 = st[j]
        y = window_labels(self.ev[self.s[i]], t0, n0)
        x = x[:, t0:t0 + CROP]
        if self.noise_aug and np.random.rand() < 0.5:     # white noise at a random SNR between 0 and 30 dB
            x = add_noise_logmel(x, np.random.uniform(0, 30), np.random)
        x = (x - self.mu) / self.sd
        if self.aug:
            x = x + np.random.uniform(-0.3, 0.3)
            for _ in range(2):
                f = np.random.randint(0, 8); f0 = np.random.randint(0, N_MELS - f); x[f0:f0 + f] = 0
                t = np.random.randint(0, 40); tt = np.random.randint(0, CROP - t); x[:, tt:tt + t] = 0
        soft = self.soft[self.s[i]][j] if self.soft is not None else np.zeros(2, np.float32)
        return torch.from_numpy(x[None].copy()), torch.tensor(y, dtype=torch.float32), torch.from_numpy(np.asarray(soft, np.float32))


def windows(x, mu, sd):
    x = pad_to(x.astype(np.float32), CROP)
    return np.stack([(x[:, s:s + CROP] - mu) / sd for s in starts_for(x.shape[1])])[:, None]


# ---------------------------------------------------------------- models
class DSBlock(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False); self.b1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False); self.b2 = nn.BatchNorm2d(cout); self.pool = pool
    def forward(self, x):
        x = F.relu(self.b1(self.dw(x))); x = F.relu(self.b2(self.pw(x)))
        return F.max_pool2d(x, self.pool) if self.pool else x


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity(); self.pool = pool
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); x = F.relu(h + self.sc(x))
        return F.max_pool2d(x, self.pool) if self.pool else x


ARCH = {"tiny": ([24, 32, 48, 64], DSBlock, 16), "small": ([32, 64, 96, 128], DSBlock, 24),
        "medium": ([32, 64, 128, 256], ConvBlock, 32)}


class Net(nn.Module):
    """Strided stem, four blocks, pooling over frequency then mean+max over time; two logits (CAS, DAS)."""
    def __init__(self, kind="tiny"):
        super().__init__()
        ch, B, stem = ARCH[kind]
        self.stem = nn.Sequential(nn.Conv2d(1, stem, 3, stride=(2, 2), padding=1, bias=False), nn.BatchNorm2d(stem), nn.ReLU())
        layers, cin = [], stem
        for i, c in enumerate(ch):
            layers.append(B(cin, c, (2, 2) if i < 3 else None)); cin = c
        self.blocks = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.2); self.head = nn.Linear(cin, 2)
    def forward(self, x):
        h = self.blocks(self.stem(x)).mean(dim=2)
        h = 0.5 * h.amax(dim=2) + 0.5 * h.mean(dim=2)
        return self.head(self.drop(h))


def n_params(m): return sum(p.numel() for p in m.parameters())


def mflops(m):
    """2 x multiply-accumulates of conv and linear layers for one 4-s window, in millions."""
    tot = [0]
    def conv_hook(mod, inp, out):
        tot[0] += out.numel() * mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
    def lin_hook(mod, inp, out): tot[0] += mod.in_features * mod.out_features
    hs = [mm.register_forward_hook(conv_hook) for mm in m.modules() if isinstance(mm, nn.Conv2d)]
    hs += [mm.register_forward_hook(lin_hook) for mm in m.modules() if isinstance(mm, nn.Linear)]
    m.eval()
    with torch.no_grad():
        m(torch.zeros(1, 1, N_MELS, CROP))
    for h in hs: h.remove()
    return 2 * tot[0] / 1e6


# ---------------------------------------------------------------- training / inference
def train_model(df, events, kind="tiny", seed=0, epochs=20, mu=None, sd=None, soft=None, kd_w=0.0, T=2.0,
                bs=64, lr=3e-3, log=None, noise_aug=False):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(kind)
    dl = torch.utils.data.DataLoader(TrainSet(df, mu, sd, events, soft=soft, noise_aug=noise_aug), batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.1)
    t0 = time.time()
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y, z in dl:
            out = m(x)
            loss = F.binary_cross_entropy_with_logits(out, y)
            if soft is not None and kd_w > 0:
                kd = F.binary_cross_entropy_with_logits(out / T, torch.sigmoid(z / T)) * T * T
                loss = (1 - kd_w) * loss + kd_w * kd
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        if log and (ep % 5 == 4 or ep == epochs - 1):
            log(f"  {kind}{'-kd' if soft is not None else ''} seed{seed} ep{ep + 1} loss {tot / len(dl):.4f} {time.time() - t0:.0f}s")
    m.eval()
    return m


@torch.no_grad()
def window_logits(m, stems, mu, sd, noise_snr=None, rng=None):
    """Per-recording array of window logits (n_windows x 2)."""
    out = {}
    for s in stems:
        x = mels()[s].astype(np.float32)
        if noise_snr is not None:
            x = add_noise_logmel(x, noise_snr, rng)
        out[s] = m(torch.from_numpy(windows(x, mu, sd))).numpy()
    return out


def recording_scores(wl):
    """Presence score of a recording = maximum window logit (per class)."""
    return {s: v.max(0) for s, v in wl.items()}


def add_noise_logmel(x, snr_db, rng):
    """White noise added in the power-mel domain at a recording-level signal-to-noise ratio."""
    p = np.exp(x.astype(np.float64)); sig = p.mean()
    noise = rng.chisquare(2, size=p.shape) / 2 * sig / (10 ** (snr_db / 10))
    return np.log(p + noise + 1e-6).astype(np.float32)
