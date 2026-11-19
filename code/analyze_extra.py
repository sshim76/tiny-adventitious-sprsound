"""Additional analyses for the revised manuscript (no training). Usage: python3 -I code/analyze_extra.py
Writes results/x_*.csv. Sections that depend on run_extra.py outputs are skipped until those files exist.
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression

NB, SEEDS, CLS = 2000, [0, 1, 2], ["cas", "das"]
d = load_index(); ev = load_events()
sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
d = d.drop(columns=["part"]).merge(sp[["stem", "part"]], on="stem")
preds = pickle.load(open(os.path.join(RES, "dl_preds.pkl"), "rb"))
PARTS = ["val", "ext2024", "ext2025"]
E = d[d.part.isin(PARTS)].reset_index(drop=True)
pat = E.patient.values
G = {p: E.index[E.part == p].values for p in PARTS}
Y = E[CLS].values


def rec(wl_by_stem, agg="max"):
    f = {"max": lambda v: v.max(0), "mean": lambda v: v.mean(0),
         "top2": lambda v: np.sort(v, 0)[-2:].mean(0), "lse": lambda v: np.log(np.exp(v).mean(0))}[agg]
    return np.stack([f(wl_by_stem[s]) for s in E.stem])


def boots(idx, nb=NB, seed=0):
    rng = np.random.default_rng(seed); g = pat[idx]; ug = np.unique(g); by = {k: idx[g == k] for k in ug}
    for _ in range(nb):
        yield np.concatenate([by[k] for k in rng.choice(ug, len(ug))])


def seed_auc(S, idx, ci):
    return float(np.mean([roc_auc_score(Y[idx, ci], s[idx, ci]) for s in S]))


def ci_auc(S, idx, ci, seed=0):
    bb = [seed_auc(S, b, ci) for b in boots(idx, seed=seed) if 0 < Y[b, ci].sum() < len(b)]
    return np.percentile(bb, 2.5), np.percentile(bb, 97.5)


def ci_diff(SA, SB, idx, ci, seed=1):
    dd = np.array([seed_auc(SA, b, ci) - seed_auc(SB, b, ci) for b in boots(idx, seed=seed) if 0 < Y[b, ci].sum() < len(b)])
    return np.percentile(dd, 2.5), np.percentile(dd, 97.5), min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))


S = {m: [rec(preds[(m, s)]) for s in SEEDS] for m in ["tiny", "tiny_kd", "small", "medium"]}

# ---------------- 1. aggregation of window scores ----------------
rows = []
for agg in ["max", "top2", "lse", "mean"]:
    Sa = [rec(preds[("tiny_kd", s)], agg) for s in SEEDS]
    for p in PARTS:
        for ci, c in enumerate(CLS):
            a = [roc_auc_score(Y[G[p], ci], s[G[p], ci]) for s in Sa]
            rows.append(dict(agg=agg, part=p, cls=c, auroc=np.mean(a), sd=np.std(a, ddof=1)))
pd.DataFrame(rows).to_csv(os.path.join(RES, "x_aggregation.csv"), index=False)

# ---------------- 2. window-level discrimination ----------------
wl_lab = {}
for s in E.stem:
    n0 = mels()[s].shape[1]
    wl_lab[s] = np.array([window_labels(ev[s], t0, n0) for t0 in starts_for(n0)])
rows = []
for m in ["tiny", "tiny_kd", "small", "medium"]:
    for p in PARTS:
        st = E.stem.values[G[p]]; yw = np.concatenate([wl_lab[s] for s in st])
        for ci, c in enumerate(CLS):
            a = [roc_auc_score(yw[:, ci], np.concatenate([preds[(m, sd_)][s] for s in st])[:, ci]) for sd_ in SEEDS]
            rows.append(dict(model=m, part=p, cls=c, n_windows=len(yw), n_pos=int(yw[:, ci].sum()), auroc=np.mean(a), sd=np.std(a, ddof=1)))
pd.DataFrame(rows).to_csv(os.path.join(RES, "x_window_level.csv"), index=False)

# ---------------- 3. subgroups (external releases pooled; Tiny + KD) ----------------
ext = np.concatenate([G["ext2024"], G["ext2025"]])
E["age_band"] = pd.cut(E.age, [0, 3, 6, 100], right=False, labels=["<3 y", "3 to <6 y", ">=6 y"]).astype(str)
E["sex_l"] = E.sex.map({0: "Male", 1: "Female"})
E["site"] = E["loc"].map({"p1": "Left posterior", "p2": "Left lateral", "p3": "Right posterior", "p4": "Right lateral"}).fillna("Additional sites")
rows = []
for var in ["age_band", "sex_l", "site", "part"]:
    for lev in sorted(E.loc[ext, var].unique()):
        idx = ext[(E.loc[ext, var] == lev).values]
        for ci, c in enumerate(CLS):
            if Y[idx, ci].sum() < 10:
                continue
            lo, hi = ci_auc(S["tiny_kd"], idx, ci)
            rows.append(dict(variable=var, level=lev, cls=c, n=len(idx), n_pos=int(Y[idx, ci].sum()),
                             auroc=seed_auc(S["tiny_kd"], idx, ci), ci_lo=lo, ci_hi=hi))
pd.DataFrame(rows).to_csv(os.path.join(RES, "x_subgroups.csv"), index=False)

# ---------------- 4. calibration of the recording score (logistic recalibration on each set) ----------------
rows = []
for m in ["tiny_kd", "medium"]:
    for p in PARTS:
        for ci, c in enumerate(CLS):
            sl, ic, br, mp = [], [], [], []
            for s in S[m]:
                z = s[G[p], ci]; y = Y[G[p], ci]
                lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
                sl.append(lr.coef_[0, 0]); ic.append(lr.intercept_[0]); pr = 1 / (1 + np.exp(-z))
                br.append(np.mean((pr - y) ** 2)); mp.append(pr.mean())
            rows.append(dict(model=m, part=p, cls=c, prevalence=Y[G[p], ci].mean(), mean_pred=np.mean(mp),
                             slope=np.mean(sl), intercept=np.mean(ic), brier=np.mean(br)))
pd.DataFrame(rows).to_csv(os.path.join(RES, "x_calibration.csv"), index=False)

# ---------------- 5. activation memory and front-end cost ----------------
p5 = os.path.join(RES, "x_memory_frontend.csv")
if not os.path.exists(p5):
    import librosa
    rows = []
    for kind in ["tiny", "small", "medium"]:
        m = Net(kind).eval(); peak = [0]; first = [0]
        def hook(mod, inp, out):
            tot = inp[0].numel() + out.numel(); peak[0] = max(peak[0], tot)
        hs = [mm.register_forward_hook(hook) for mm in m.modules() if isinstance(mm, (nn.Conv2d, nn.Linear))]
        with torch.no_grad(): m(torch.zeros(1, 1, N_MELS, CROP))
        for h in hs: h.remove()
        rows.append(dict(item=kind, peak_act_elements=peak[0], peak_act_int8_kb=peak[0] / 1024, peak_act_fp32_kb=4 * peak[0] / 1024))
    x = np.random.default_rng(0).standard_normal(4 * 8000).astype(np.float32)
    def fe():
        return np.log(librosa.feature.melspectrogram(y=x, sr=8000, n_fft=256, win_length=200, hop_length=80, n_mels=64, fmin=50, fmax=4000, power=2.0) + 1e-6)
    for _ in range(10): fe()
    ts = []
    for _ in range(200):
        t = time.perf_counter(); fe(); ts.append((time.perf_counter() - t) * 1000)
    # arithmetic of the front end: 400 frames x (256-point real FFT + 64 mel filters over 129 bins + log)
    # per frame: radix-2 FFT (N/2 log2 N butterflies x 10 flops) + power (3 x 129) + sparse mel (2 x 258) + 64 logs
    per_frame = 10 * 128 * 8 + 3 * 129 + 2 * 258 + 64
    rows.append(dict(item="frontend", frontend_ms_per_window=statistics.median(ts), frontend_mflops_per_window=400 * per_frame / 1e6,
                     frontend_mflops_per_second=100 * per_frame / 1e6))
    pd.DataFrame(rows).to_csv(p5, index=False)

# ---------------- 6. duplicate audio ----------------
p6 = os.path.join(RES, "audio_md5.csv")
if not os.path.exists(p6):  # MD5 checksum of every audio file in results/index.csv
    import hashlib
    ix = pd.read_csv(os.path.join(RES, "index.csv"))
    pd.DataFrame({"stem": ix.stem, "md5": [hashlib.md5(open(os.path.join(SRC, f), "rb").read()).hexdigest() for f in ix.file]}).to_csv(p6, index=False)
md = pd.read_csv(p6).merge(sp[["stem", "part"]], on="stem", how="left")
dup = md[md.duplicated("md5", keep=False)]
dup.to_csv(os.path.join(RES, "x_duplicates.csv"), index=False)

# ---------------- 7. rolling origin (Tiny without distillation trained on 2022-2024) ----------------
pr_ = os.path.join(RES, "roll_preds.pkl")
if os.path.exists(pr_):
    roll = pickle.load(open(pr_, "rb"))
    if all(("tiny_roll", s) in roll for s in SEEDS):
        rows = []
        for p in ["val", "ext2025"]:
            idx = G[p]; st = E.stem.values[idx]
            SR = [np.zeros((len(E), 2)) for _ in SEEDS]
            for k, s in enumerate(SEEDS):
                SR[k][idx] = np.stack([roll[("tiny_roll", s)][x].max(0) for x in st])
            for ci, c in enumerate(CLS):
                lo, hi = ci_auc(SR, idx, ci); dlo, dhi, pv = ci_diff(SR, S["tiny"], idx, ci)
                a = [roc_auc_score(Y[idx, ci], s_[idx, ci]) for s_ in SR]
                rows.append(dict(part=p, cls=c, auroc_roll=np.mean(a), sd=np.std(a, ddof=1), ci_lo=lo, ci_hi=hi,
                                 auroc_base=seed_auc(S["tiny"], idx, ci), diff=np.mean(a) - seed_auc(S["tiny"], idx, ci),
                                 d_lo=dlo, d_hi=dhi, p=pv))
        pd.DataFrame(rows).to_csv(os.path.join(RES, "x_rolling.csv"), index=False)

# ---------------- 8. within-release cross-validation on 2025 ----------------
pc = os.path.join(RES, "cv2025_preds.pkl")
if os.path.exists(pc):
    cv = pickle.load(open(pc, "rb"))
    if all((s, f) in cv for s in SEEDS for f in range(5)):
        idx = G["ext2025"]; st = E.stem.values[idx]
        SC = [np.zeros((len(E), 2)) for _ in SEEDS]
        for k, s in enumerate(SEEDS):
            allw = {}
            for f in range(5): allw.update(cv[(s, f)])
            SC[k][idx] = np.stack([allw[x].max(0) for x in st])
        rows = []
        for ci, c in enumerate(CLS):
            a = [roc_auc_score(Y[idx, ci], s_[idx, ci]) for s_ in SC]; lo, hi = ci_auc(SC, idx, ci)
            for base in ["tiny", "tiny_kd", "medium"]:
                dlo, dhi, pv = ci_diff(SC, S[base], idx, ci)
                rows.append(dict(cls=c, auroc_cv=np.mean(a), sd=np.std(a, ddof=1), ci_lo=lo, ci_hi=hi, base=base,
                                 auroc_base=seed_auc(S[base], idx, ci), diff=np.mean(a) - seed_auc(S[base], idx, ci), d_lo=dlo, d_hi=dhi, p=pv))
        pd.DataFrame(rows).to_csv(os.path.join(RES, "x_cv2025.csv"), index=False)
        pickle.dump(SC, open(os.path.join(RES, "x_cv2025_scores.pkl"), "wb"))

# ---------------- 9. adaptation summary ----------------
pa = os.path.join(RES, "adaptation.csv")
if os.path.exists(pa):
    a = pd.read_csv(pa)
    a["bacc"] = (a.sens + a.spec) / 2
    per_draw = a.groupby(["method", "k", "cls", "draw"])[["auroc", "sens", "spec", "bacc", "recal", "n_adapt_rec", "n_adapt_pos"]].mean().reset_index()
    summ = per_draw.groupby(["method", "k", "cls"]).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"), sens=("sens", "mean"), sens_sd=("sens", "std"),
                                                        spec=("spec", "mean"), spec_sd=("spec", "std"), bacc=("bacc", "mean"), bacc_sd=("bacc", "std"),
                                                        recal=("recal", "mean"), n_rec=("n_adapt_rec", "mean"), n_pos=("n_adapt_pos", "mean"), draws=("draw", "nunique")).reset_index()
    summ.to_csv(os.path.join(RES, "x_adaptation_summary.csv"), index=False)

# ---------------- 10. distillation ablation ----------------
pk = os.path.join(RES, "kdabl_preds.pkl")
if os.path.exists(pk):
    kd = pickle.load(open(pk, "rb"))
    rows = []
    cfgs = {"tiny": ("none", "-", "-"), "kd_T1": ("KD", 1, 0.5), "tiny_kd": ("KD", 2, 0.5), "kd_T4": ("KD", 4, 0.5), "kd_w1": ("KD, soft labels only", 2, 1.0)}
    for name, (lab, T, w) in cfgs.items():
        src = preds if name in ("tiny", "tiny_kd") else kd
        if not all((name, s) in src for s in SEEDS):
            continue
        Sk = [rec(src[(name, s)]) for s in SEEDS]
        for p in PARTS:
            for ci, c in enumerate(CLS):
                a = [roc_auc_score(Y[G[p], ci], s_[G[p], ci]) for s_ in Sk]
                rows.append(dict(config=name, label=lab, T=T, w=w, part=p, cls=c, auroc=np.mean(a), sd=np.std(a, ddof=1)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "x_kd_ablation.csv"), index=False)
# ---------------- 11. official BioCAS 2022 protocol, Task 1-1 (event level) ----------------
pb = os.path.join(RES, "bench_preds.pkl")
if os.path.exists(pb):
    B = pickle.load(open(pb, "rb"))
    if all((m, s) in B for m in ["tiny", "tiny_kd", "medium"] for s in SEEDS):
        Ev = pd.read_csv(os.path.join(RES, "bench_events.csv"), dtype={"patient": str})
        Ev = Ev[Ev.part != "fit"].reset_index(drop=True)
        ya = Ev.adv.values

        def score(y, pr):
            se = pr[y == 1].mean(); sp_ = 1 - pr[y == 0].mean(); a_s = (se + sp_) / 2
            hs = 2 * se * sp_ / (se + sp_) if se + sp_ > 0 else 0.0
            return se, sp_, a_s, hs, (a_s + hs) / 2

        sets = {"hold": Ev.part == "hold", "intra": Ev.part == "intra", "inter": Ev.part == "inter",
                "test2022": Ev.part.isin(["intra", "inter"]), "gc23": Ev.part == "gc23"}
        rows = []
        for m in ["tiny", "tiny_kd", "medium"]:
            for s in SEEDS:
                z = B[(m, s)].max(1)                      # adventitious score = larger of the CAS and DAS logits
                h = sets["hold"].values; cand = np.unique(z[h])
                th = cand[int(np.argmax([score(ya[h], z[h] >= c)[4] for c in cand]))]
                for name, sel in sets.items():
                    sel = sel.values; se, sp_, a_s, hs, sc = score(ya[sel], z[sel] >= th)
                    rows.append(dict(model=m, seed=s, set=name, n=int(sel.sum()), n_adv=int(ya[sel].sum()), SE=se, SP=sp_, AS=a_s, HS=hs,
                                     Score=sc, auroc=roc_auc_score(ya[sel], z[sel]), threshold=th))
        bd = pd.DataFrame(rows); bd.to_csv(os.path.join(RES, "x_benchmark_seeds.csv"), index=False)
        g = bd.groupby(["model", "set"])
        out = g[["SE", "SP", "AS", "HS", "Score", "auroc"]].mean().join(g[["Score", "auroc"]].std().add_suffix("_sd")).join(g[["n", "n_adv"]].first()).reset_index()
        out.to_csv(os.path.join(RES, "x_benchmark.csv"), index=False)
# ---------------- 12. how adventitious labels are distributed within recordings, by cohort ----------------
D = d.copy(); D["cohort3"] = D.part.replace({"train": "dev", "val": "dev"})
rows = []
for _, r in D.iterrows():
    Ev_ = ev[r.stem]; n = len(Ev_)
    rows.append(dict(cohort=r.cohort3, cas=r.cas, das=r.das, dur=r.dur,
                     f_das=sum(t in DAS_T for _, _, t in Ev_) / n, f_cas=sum(t in CAS_T for _, _, t in Ev_) / n,
                     t_das=sum(e - s_ for s_, e, t in Ev_ if t in DAS_T), t_cas=sum(e - s_ for s_, e, t in Ev_ if t in CAS_T),
                     t_ann=sum(e - s_ for s_, e, t in Ev_)))
A = pd.DataFrame(rows); out = []
for c in ["dev", "ext2024", "ext2025"]:
    q = A[A.cohort == c]; row = dict(cohort=c, recordings=len(q), annotated_share_of_time=q.t_ann.sum() / q.dur.sum())
    for k in ["das", "cas"]:
        pos = q[q[k] == 1]
        row.update({f"{k}_pos": len(pos), f"{k}_event_share": pos[f"f_{k}"].mean(), f"{k}_all_events": (pos[f"f_{k}"] == 1).mean(),
                    f"{k}_median_time_s": pos[f"t_{k}"].median()})
    out.append(row)
pd.DataFrame(out).to_csv(os.path.join(RES, "x_annotation_structure.csv"), index=False)
print("extra analysis done")
