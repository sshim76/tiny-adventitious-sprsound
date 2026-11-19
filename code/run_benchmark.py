"""Official BioCAS 2022 protocol, Task 1-1 (event level, Normal vs Adventitious).
Training: official SPRSound training set (1,949 recordings, 6,656 events); 20% of its patients are held out to set
the decision threshold. Test: official SPRSound testing set (intra- and inter-patient parts) and Grand Challenge'23.
The networks are the Tiny and Reference architectures of the main analysis with their two outputs (CAS, DAS);
the adventitious score of an event is the larger of the two logits.
Usage: python3 -I code/run_benchmark.py   -> results/bench_preds.pkl, results/bench_events.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

L, EPOCHS, SEEDS = 300, 15, [0, 1, 2]
LOG = open(os.path.join(RES, "bench.log"), "a")
def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


full = pd.read_csv(os.path.join(RES, "index.csv"), dtype={"patient": str})
full = full[full.release <= 2023]
full["grp"] = np.where(full.release == 2022, np.where(full.part == "train2022_wav", "train", full.subset.str.replace("_test_json", "")), "gc23")
ev = load_events()
rows = []
for _, r in full.iterrows():
    for k, (s, e, t) in enumerate(ev.get(r.stem, [])):
        rows.append(dict(stem=r.stem, k=k, patient=r.patient, grp=r.grp, start=s, end=e, type=t,
                         cas=int(t in CAS_T), das=int(t in DAS_T)))
E = pd.DataFrame(rows); E["adv"] = (E.type != "Normal").astype(int)
# hold out 20% of the official training patients for threshold selection
tp = np.array(sorted(E[E.grp == "train"].patient.unique().tolist()), dtype=object)
np.random.default_rng(2022).shuffle(tp)
hold = set(tp[:int(round(0.2 * len(tp)))])
E["part"] = np.where(E.grp != "train", E.grp, np.where(E.patient.isin(hold), "hold", "fit"))
E.to_csv(os.path.join(RES, "bench_events.csv"), index=False)
log(str(E.groupby("part").agg(events=("adv", "size"), adv=("adv", "sum"), patients=("patient", "nunique"))))


def crop(i):
    r = E.iloc[i]; x = mels()[r.stem].astype(np.float32)
    a = max(0, int(round(r.start / HOP_S))); b = min(x.shape[1], max(a + 1, int(round(r.end / HOP_S))))
    x = x[:, a:b]
    if x.shape[1] < L:
        x = np.tile(x, (1, int(np.ceil(L / x.shape[1]))))[:, :L]
    elif x.shape[1] > L:
        c = (x.shape[1] - L) // 2; x = x[:, c:c + L]
    return x


X = np.empty((len(E), N_MELS, L), np.float16)          # kept in half precision to limit memory
for i in range(len(E)):
    X[i] = crop(i)
import common; common._MELS = None                      # the recording-level cache is no longer needed
fit = np.where(E.part == "fit")[0]
sub = X[fit[::4]].astype(np.float32); mu, sd = float(sub.mean()), float(sub.std()); del sub
for i in range(0, len(X), 512):
    X[i:i + 512] = ((X[i:i + 512].astype(np.float32) - mu) / sd).astype(np.float16)
Y = E[["cas", "das"]].values.astype(np.float32)
json.dump({"mu": mu, "sd": sd, "L": L}, open(os.path.join(RES, "bench_norm.json"), "w"))


class EvSet(torch.utils.data.Dataset):
    def __init__(self, idx, soft=None): self.idx, self.soft = idx, soft
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = self.idx[j]; x = X[i].astype(np.float32) + np.random.uniform(-0.3, 0.3)
        for _ in range(2):
            f = np.random.randint(0, 8); f0 = np.random.randint(0, N_MELS - f); x[f0:f0 + f] = 0
            t = np.random.randint(0, 30); t0 = np.random.randint(0, L - t); x[:, t0:t0 + t] = 0
        z = self.soft[j] if self.soft is not None else np.zeros(2, np.float32)
        return torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(Y[i]), torch.from_numpy(np.asarray(z, np.float32))


def train(kind, seed, soft=None, kd_w=0.5, T=2.0, bs=64, lr=3e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(kind); dl = torch.utils.data.DataLoader(EvSet(fit, soft), batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=EPOCHS * len(dl), pct_start=0.1)
    for ep in range(EPOCHS):
        m.train()
        for x, y, z in dl:
            out = m(x); loss = F.binary_cross_entropy_with_logits(out, y)
            if soft is not None:
                loss = (1 - kd_w) * loss + kd_w * T * T * F.binary_cross_entropy_with_logits(out / T, torch.sigmoid(z / T))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    m.eval(); return m


@torch.no_grad()
def infer(m, idx):
    return np.concatenate([m(torch.from_numpy(X[idx[i:i + 256]][:, None].astype(np.float32))).numpy() for i in range(0, len(idx), 256)])


if os.environ.get("DRY"):          # smoke test: one short epoch on a subset, nothing saved
    EPOCHS = 1; fit = fit[:256]
    m = train("tiny", 0, soft=np.zeros((len(fit), 2), np.float32)); out = infer(m, np.where(E.part != "fit")[0][:300])
    print("dry run ok", X.shape, out.shape, mu, sd); sys.exit(0)
pp = os.path.join(RES, "bench_preds.pkl")
P = pickle.load(open(pp, "rb")) if os.path.exists(pp) else {}
ev_idx = np.where(E.part != "fit")[0]
for name, kind in [("tiny", "tiny"), ("medium", "medium")]:
    for s in SEEDS:
        if (name, s) in P: continue
        t0 = time.time(); m = train(kind, s)
        P[(name, s)] = infer(m, ev_idx)
        if kind == "medium": P[("teacher_fit", s)] = infer(m, fit)
        torch.save(m.state_dict(), os.path.join(RES, "models", f"bench_{name}_s{s}.pt"))
        pickle.dump(P, open(pp, "wb")); log(f"bench {name} seed {s} done in {time.time() - t0:.0f}s")
soft = np.mean([P[("teacher_fit", s)] for s in SEEDS], axis=0)
for s in SEEDS:
    if ("tiny_kd", s) in P: continue
    t0 = time.time(); m = train("tiny", s, soft=soft)
    P[("tiny_kd", s)] = infer(m, ev_idx)
    torch.save(m.state_dict(), os.path.join(RES, "models", f"bench_tiny_kd_s{s}.pt"))
    pickle.dump(P, open(pp, "wb")); log(f"bench tiny_kd seed {s} done in {time.time() - t0:.0f}s")
P["ev_idx"] = ev_idx; pickle.dump(P, open(pp, "wb"))
log("benchmark done")
