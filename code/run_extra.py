"""Additional experiments for the revised manuscript.
  cv2025  within-release 5-fold patient-level cross-validation on the 2025 release (Tiny, no distillation)
  roll    rolling-origin training: Tiny (no distillation) trained on 2022-2024, tested on 2025
  adapt   adaptation of the Tiny + KD models to the 2025 release with k labelled patients
  kdabl   distillation ablation (temperature and soft-label weight)
Usage: python3 -I code/run_extra.py cv2025 roll adapt kdabl
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import roc_auc_score, roc_curve

SEEDS = [0, 1, 2]
LOG = open(os.path.join(RES, "extra.log"), "a")
def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


d = load_index(); events = load_events()
sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
d = d.drop(columns=["part"]).merge(sp[["stem", "part"]], on="stem")
nrm = json.load(open(os.path.join(RES, "norm.json"))); MU, SD = nrm["mu"], nrm["sd"]


def load(p):
    p = os.path.join(RES, p)
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}


def dump(o, p):
    pickle.dump(o, open(os.path.join(RES, p), "wb"))


# ------------------------------------------------------------------ within-release CV on 2025
def stage_cv2025():
    out = load("cv2025_preds.pkl")
    t = d[d.part == "ext2025"].reset_index(drop=True)
    pts = np.array(sorted(t.patient.unique().tolist()), dtype=object)
    np.random.default_rng(2025).shuffle(pts)
    folds = np.array_split(pts, 5)
    pd.DataFrame([(p, f) for f, ps in enumerate(folds) for p in ps], columns=["patient", "fold"]).to_csv(
        os.path.join(RES, "cv2025_folds.csv"), index=False)
    for seed in SEEDS:
        for f in range(5):
            if (seed, f) in out:
                continue
            te = t[t.patient.isin(set(folds[f]))]; tr = t[~t.patient.isin(set(folds[f]))].reset_index(drop=True)
            mu, sd = norm_stats(tr.stem.values)
            m = train_model(tr, events, kind="tiny", seed=seed, epochs=20, mu=mu, sd=sd, log=None)
            out[(seed, f)] = window_logits(m, te.stem.values, mu, sd); dump(out, "cv2025_preds.pkl")
            log(f"cv2025 seed {seed} fold {f} done ({len(tr)} train / {len(te)} test recordings)")


# ------------------------------------------------------------------ rolling origin
def stage_roll():
    out = load("roll_preds.pkl")
    tr = d[d.part.isin(["train", "ext2024"])].reset_index(drop=True)
    mu, sd = norm_stats(tr.stem.values)
    json.dump({"mu": mu, "sd": sd, "n_train": len(tr), "n_patients": int(tr.patient.nunique())},
              open(os.path.join(RES, "norm_roll.json"), "w"))
    ev_stems = d[d.part.isin(["val", "ext2025"])].stem.values
    for seed in SEEDS:
        if ("tiny_roll", seed) in out:
            continue
        m = train_model(tr, events, kind="tiny", seed=seed, epochs=20, mu=mu, sd=sd, log=log)
        torch.save(m.state_dict(), os.path.join(RES, "models", f"tiny_roll_s{seed}.pt"))
        out[("tiny_roll", seed)] = window_logits(m, ev_stems, mu, sd); dump(out, "roll_preds.pkl")
        log(f"tiny_roll seed {seed} done")


# ------------------------------------------------------------------ adaptation to the 2025 release
@torch.no_grad()
def embed(m, x):
    h = m.blocks(m.stem(x)).mean(dim=2)
    return 0.5 * h.amax(dim=2) + 0.5 * h.mean(dim=2)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def metrics(y, s, th):
    pr = s >= th
    return dict(auroc=roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
                sens=pr[y == 1].mean() if y.sum() else np.nan, spec=1 - pr[y == 0].mean() if (y == 0).sum() else np.nan)


def stage_adapt(ks=(5, 10, 20, 40), draws=10):
    preds = load("dl_preds.pkl")
    t = d[d.part == "ext2025"].reset_index(drop=True); v = d[d.part == "val"].reset_index(drop=True)
    pts = np.array(sorted(t.patient.unique().tolist()), dtype=object)
    Y = t[["cas", "das"]].values
    rows = []
    done = os.path.join(RES, "adaptation.csv")
    complete = []
    if os.path.exists(done):                 # resume: keep seeds for which every k was finished
        ex = pd.read_csv(done)
        complete = [sd_ for sd_ in SEEDS if set(ex[ex.seed == sd_].k.unique()) == set(ks)]
        rows = ex[ex.seed.isin(complete)].to_dict("records")
    for seed in SEEDS:
        if seed in complete:
            continue
        m0 = Net("tiny"); m0.load_state_dict(torch.load(os.path.join(RES, "models", f"tiny_kd_s{seed}.pt"))); m0.eval()
        # window embeddings, window labels and source scores for every 2025 recording
        emb, wlab = {}, {}
        for s in t.stem:
            raw = mels()[s]; n0 = raw.shape[1]
            emb[s] = embed(m0, torch.from_numpy(windows(raw, MU, SD))).numpy()
            wlab[s] = np.array([window_labels(events[s], t0, n0) for t0 in starts_for(n0)], dtype=np.float32)
        src = np.stack([preds[("tiny_kd", seed)][s].max(0) for s in t.stem])
        vs = np.stack([preds[("tiny_kd", seed)][s].max(0) for s in v.stem])
        th_src = [youden(v[c].values, vs[:, ci]) for ci, c in enumerate(["cas", "das"])]
        W0 = m0.head.weight.detach().clone(); b0 = m0.head.bias.detach().clone()
        for k in ks:
            for dr in range(draws):
                rng = np.random.default_rng(1000 * k + dr)
                A = set(rng.choice(pts, k, replace=False).tolist())
                ia = t.index[t.patient.isin(A)].values; it = t.index[~t.patient.isin(A)].values
                stA = t.stem.values[ia]

                def record(method, scores_all, n_ok=None):
                    for ci, c in enumerate(["cas", "das"]):
                        ya = Y[ia, ci]; ok = 0 < ya.sum() < len(ya)
                        th = youden(ya, scores_all[ia, ci]) if (ok and method != "source") else th_src[ci]
                        if method == "source":
                            th = th_src[ci]
                        mm = metrics(Y[it, ci], scores_all[it, ci], th)
                        rows.append(dict(method=method, seed=seed, k=k, draw=dr, cls=c, n_adapt_rec=len(ia),
                                         n_adapt_pos=int(ya.sum()), recal=int(ok and method != "source"), **mm))

                record("source", src)                      # no adaptation, validation threshold
                record("threshold", src)                   # threshold re-selected on the adaptation patients
                # head refit: 130 parameters, L2 penalty towards the source head
                XA = torch.from_numpy(np.concatenate([emb[s] for s in stA])); YA = torch.from_numpy(np.concatenate([wlab[s] for s in stA]))
                W = W0.clone().requires_grad_(True); b = b0.clone().requires_grad_(True)
                opt = torch.optim.Adam([W, b], lr=0.01)
                for _ in range(200):
                    loss = F.binary_cross_entropy_with_logits(XA @ W.T + b, YA) + 0.01 * ((W - W0) ** 2).sum()
                    opt.zero_grad(); loss.backward(); opt.step()
                Wn, bn = W.detach().numpy(), b.detach().numpy()
                head = np.stack([(emb[s] @ Wn.T + bn).max(0) for s in t.stem])
                record("head", head)
                # full fine-tuning of all parameters, batch-norm statistics frozen
                m = copy.deepcopy(m0); torch.manual_seed(seed * 100 + dr); np.random.seed(seed * 100 + dr)
                dl = torch.utils.data.DataLoader(TrainSet(t.iloc[ia], MU, SD, events), batch_size=32, shuffle=True)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                for ep in range(10):
                    m.train()
                    for mod in m.modules():
                        if isinstance(mod, nn.BatchNorm2d): mod.eval()
                    for x, y, _ in dl:
                        loss = F.binary_cross_entropy_with_logits(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
                m.eval()
                wl = window_logits(m, t.stem.values, MU, SD)
                record("finetune", np.stack([wl[s].max(0) for s in t.stem]))
            pd.DataFrame(rows).to_csv(done, index=False)
            log(f"adapt seed {seed} k {k} done")


# ------------------------------------------------------------------ distillation ablation
def stage_kdabl():
    out = load("kdabl_preds.pkl"); preds = load("dl_preds.pkl")
    tr = d[d.part == "train"].reset_index(drop=True)
    ev_stems = d[d.part != "train"].stem.values
    for name, T, w in [("kd_T1", 1.0, 0.5), ("kd_T4", 4.0, 0.5), ("kd_w1", 2.0, 1.0)]:
        for seed in SEEDS:
            if (name, seed) in out:
                continue
            m = train_model(tr, events, kind="tiny", seed=seed, epochs=20, mu=MU, sd=SD, soft=preds["teacher_train"], kd_w=w, T=T, log=None)
            out[(name, seed)] = window_logits(m, ev_stems, MU, SD); dump(out, "kdabl_preds.pkl")
            log(f"{name} seed {seed} done")


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time(); log(f"=== stage {st}")
        {"cv2025": stage_cv2025, "roll": stage_roll, "adapt": stage_adapt, "kdabl": stage_kdabl}[st]()
        log(f"=== stage {st} finished in {(time.time() - t0) / 60:.1f} min")
