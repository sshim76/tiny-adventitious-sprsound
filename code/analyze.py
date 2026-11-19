"""Evaluation of all models: discrimination, thresholds, paired comparisons, event-type sensitivity, INT8,
noise robustness and the exploratory diagnosis analysis. Usage: python3 -I code/analyze.py -> results/*.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

NB = 2000
SEEDS = [0, 1, 2]
d = load_index()
sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
d = d.drop(columns=["part"]).merge(sp[["stem", "part"]], on="stem")
ev = load_events()
preds = pickle.load(open(os.path.join(RES, "dl_preds.pkl"), "rb"))
nrm = json.load(open(os.path.join(RES, "norm.json"))); mu, sd = nrm["mu"], nrm["sd"]
PARTS = ["val", "ext2024", "ext2025"]
CLS = ["cas", "das"]
E = d[d.part.isin(PARTS)].reset_index(drop=True)

# ---------------- recording-level scores per model and seed ----------------
S = {}
for key, wl in preds.items():
    if not isinstance(key, tuple):
        continue
    name, seed = key
    if seed in SEEDS:
        rs = recording_scores(wl); S[(name, seed)] = np.stack([rs[s] for s in E.stem])
# acoustic-feature baseline (logistic regression per class, recording level)
af = pd.read_csv(os.path.join(RES, "acoustic_features.csv")); AC = [c for c in af.columns if c != "stem"]
tr = d[d.part == "train"].merge(af, on="stem"); Ea = E.merge(af, on="stem", how="left")
for seed in SEEDS:   # deterministic; repeated so that every model has three entries
    cols = []
    for c in CLS:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(tr[AC], tr[c])
        cols.append(m.decision_function(Ea[AC]))
    S[("acoustic_lr", seed)] = np.stack(cols, 1)
int8 = pickle.load(open(os.path.join(RES, "int8_preds.pkl"), "rb")) if os.path.exists(os.path.join(RES, "int8_preds.pkl")) else {}
for (name, seed), wl in int8.items():
    rs = recording_scores(wl); S[(name + "_int8", seed)] = np.stack([rs[s] for s in E.stem])
MODELS = [m for m in ["acoustic_lr", "tiny", "tiny_kd", "small", "medium", "tiny_int8", "tiny_kd_int8", "small_int8", "tiny_kd_na"]
          if all((m, s) in S for s in SEEDS)]
print("models:", MODELS)

groups = {p: E.index[E.part == p].values for p in PARTS}
pat = E.patient.values


def boot_idx(idx, nb=NB, seed=0):
    rng = np.random.default_rng(seed); g = pat[idx]; ug = np.unique(g)
    by = {k: idx[g == k] for k in ug}
    for _ in range(nb):
        yield np.concatenate([by[k] for k in rng.choice(ug, len(ug))])


def mean_auc(model, idx, ci, f=roc_auc_score):
    y = E[CLS[ci]].values[idx]
    return np.mean([f(y, S[(model, s)][idx, ci]) for s in SEEDS])


# ---------------- discrimination table ----------------
rows = []
for p in PARTS:
    idx = groups[p]
    for ci, c in enumerate(CLS):
        y = E[c].values[idx]
        bs = list(boot_idx(idx, seed=ci))
        for mname in MODELS:
            aucs = [roc_auc_score(y, S[(mname, s)][idx, ci]) for s in SEEDS]
            aps = [average_precision_score(y, S[(mname, s)][idx, ci]) for s in SEEDS]
            bb = [np.mean([roc_auc_score(E[c].values[b], S[(mname, s)][b, ci]) for s in SEEDS]) for b in bs
                  if 0 < E[c].values[b].sum() < len(b)]
            # threshold: Youden index on the validation split, per seed
            sens, spec = [], []
            for s in SEEDS:
                vi = groups["val"]; fpr, tpr, thr = roc_curve(E[c].values[vi], S[(mname, s)][vi, ci])
                th = thr[np.argmax(tpr - fpr)]; pr = S[(mname, s)][idx, ci] >= th
                sens.append(pr[y == 1].mean()); spec.append(1 - pr[y == 0].mean())
            rows.append(dict(part=p, cls=c, model=mname, n=len(idx), n_pos=int(y.sum()), prev=y.mean(),
                             auroc=np.mean(aucs), auroc_sd=np.std(aucs, ddof=1), ci_lo=np.percentile(bb, 2.5), ci_hi=np.percentile(bb, 97.5),
                             auprc=np.mean(aps), auprc_sd=np.std(aps, ddof=1), sens=np.mean(sens), spec=np.mean(spec)))
        print(p, c, "done", flush=True)
disc = pd.DataFrame(rows); disc.to_csv(os.path.join(RES, "discrimination.csv"), index=False)

# ---------------- paired comparisons (difference of seed-mean AUROC) ----------------
PAIRS = [("tiny_kd", "tiny"), ("tiny_kd", "medium"), ("tiny_kd", "small"), ("tiny", "acoustic_lr"), ("tiny_kd", "tiny_kd_int8"), ("tiny_kd_na", "tiny_kd")]
rows = []
for p in PARTS:
    idx = groups[p]
    for ci, c in enumerate(CLS):
        y = E[c].values[idx]
        for a, b in PAIRS:
            if a not in MODELS or b not in MODELS: continue
            obs = mean_auc(a, idx, ci) - mean_auc(b, idx, ci)
            dd = [mean_auc(a, bi, ci) - mean_auc(b, bi, ci) for bi in boot_idx(idx, nb=NB, seed=10 + ci)
                  if 0 < E[c].values[bi].sum() < len(bi)]
            dd = np.array(dd)
            rows.append(dict(part=p, cls=c, a=a, b=b, diff=obs, ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5),
                             p=min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))))
pd.DataFrame(rows).to_csv(os.path.join(RES, "paired.csv"), index=False)

# ---------------- sensitivity by event type (external sets pooled; proposed model) ----------------
rows = []
ext = np.concatenate([groups["ext2024"], groups["ext2025"]])
types = ["Fine Crackle", "Coarse Crackle", "Wheeze", "Rhonchi", "Stridor", "Wheeze+Crackle"]
for mname in ["tiny_kd", "medium"]:
    for t in types:
        has = E.event_types.fillna("").apply(lambda s: t in s.split(";")).values
        ii = ext[has[ext]]
        ci = 0 if t in ("Wheeze", "Rhonchi", "Stridor") else 1
        se = []
        for s in SEEDS:
            vi = groups["val"]; fpr, tpr, thr = roc_curve(E[CLS[ci]].values[vi], S[(mname, s)][vi, ci])
            th = thr[np.argmax(tpr - fpr)]; se.append((S[(mname, s)][ii, ci] >= th).mean())
        rows.append(dict(model=mname, event_type=t, target=CLS[ci], n_recordings=len(ii), sensitivity=np.mean(se), sd=np.std(se, ddof=1)))
pd.DataFrame(rows).to_csv(os.path.join(RES, "event_type_sensitivity.csv"), index=False)

# ---------------- noise robustness (white noise in the mel-power domain) ----------------
noise_p = os.path.join(RES, "noise.csv")
if not os.path.exists(noise_p):
    rows = []
    ext_stems = E.stem.values[ext]
    for mname, kind in [("tiny", "tiny"), ("tiny_kd", "tiny"), ("small", "small")]:
        for s in SEEDS:
            m = Net(kind); m.load_state_dict(torch.load(os.path.join(RES, "models", f"{mname}_s{s}.pt"))); m.eval()
            for snr in [20, 10, 5, 0]:
                rs = recording_scores(window_logits(m, ext_stems, mu, sd, noise_snr=snr, rng=np.random.default_rng(100 + s)))
                sc = np.stack([rs[x] for x in ext_stems])
                for p in ["ext2024", "ext2025"]:
                    sel = (E.part.values[ext] == p)
                    for ci, c in enumerate(CLS):
                        rows.append(dict(model=mname, seed=s, snr=snr, part=p, cls=c,
                                         auroc=roc_auc_score(E[c].values[ext][sel], sc[sel, ci])))
            print("noise", mname, s, flush=True)
    nz = pd.DataFrame(rows)
    clean = disc[disc.model.isin(["tiny", "tiny_kd", "small"]) & disc.part.isin(["ext2024", "ext2025"])]
    for _, r in clean.iterrows():
        for s in SEEDS:
            nz.loc[len(nz)] = dict(model=r.model, seed=s, snr=np.inf, part=r.part, cls=r.cls,
                                   auroc=roc_auc_score(E[r.cls].values[groups[r.part]], S[(r.model, s)][groups[r.part], CLS.index(r.cls)]))
    nz.to_csv(noise_p, index=False)

# ---------------- exploratory: adventitious-sound burden and diagnosis (patient level) ----------------
rows = []
P = E.copy()
P["p_das"] = 1 / (1 + np.exp(-np.mean([S[("tiny_kd", s)][:, 1] for s in SEEDS], 0)))
P["p_cas"] = 1 / (1 + np.exp(-np.mean([S[("tiny_kd", s)][:, 0] for s in SEEDS], 0)))
P["dx"] = P.diagnosis.fillna("")
for part in ["val", "ext2024"]:
    q = P[P.part == part].groupby("patient").agg(dx=("dx", "first"), das=("das", "mean"), cas=("cas", "mean"),
                                                 p_das=("p_das", "mean"), p_cas=("p_cas", "mean"))
    q = q[q.dx != ""]
    pn = q.dx.str.startswith("Pneumonia")
    for comp, sel, lab in [("pneumonia vs control", pn | (q.dx.str.lower() == "control group"), pn),
                           ("pneumonia vs all other diagnoses", q.index == q.index, pn),
                           ("severe vs non-severe pneumonia", pn, q.dx == "Pneumonia (severe)")]:
        qq = q[sel]; yy = lab[sel]
        for f in ["das", "p_das", "cas", "p_cas"]:
            if yy.sum() < 5:
                continue
            rows.append(dict(part=part, comparison=comp, feature=f, n_pos=int(yy.sum()), n_neg=int((~yy).sum()),
                             auroc=roc_auc_score(yy, qq[f]) if 0 < yy.sum() < len(yy) else np.nan))
pd.DataFrame(rows).to_csv(os.path.join(RES, "diagnosis_exploratory.csv"), index=False)
print("analysis done")
