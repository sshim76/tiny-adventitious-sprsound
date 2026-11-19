"""Tables of the manuscript (markdown) from results/. Usage: python3 -I code/make_tables.py
Tables whose result files are not yet present are skipped."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = os.path.join(ROOT, "tables"); os.makedirs(OUT, exist_ok=True)
R = lambda f: os.path.join(RES, f)
have = lambda f: os.path.exists(R(f))


def w(name, text):
    text = text.replace("−0.000", "0.000").replace("-0.000", "0.000").replace("+0.000", "0.000")
    import re
    text = re.sub(r"(?<![\w.:|-])-(?=\d)", "−", text)
    open(os.path.join(OUT, name + ".md"), "w").write(text.strip() + "\n")


def sep(*ws):
    return "|" + "|".join(":" + "-" * x for x in ws) + "|"


pv = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
NAME = {"tiny_kd_na": "Tiny + KD, noise-augmented", "acoustic_lr": "Acoustic features + LR", "tiny": "Tiny", "tiny_kd": "Tiny + KD",
        "small": "Small", "medium": "Reference", "tiny_kd_int8": "Tiny + KD, INT8"}
PART = {"train": "Training", "val": "Validation", "ext2024": "External 2024", "ext2025": "External 2025"}
d = load_index(); sp = pd.read_csv(R("split.csv"), dtype={"patient": str})
d = d.drop(columns=["part"]).merge(sp[["stem", "part"]], on="stem")
ev = load_events()

# ---------------- Table 1: cohorts ----------------
rows = {}
for p in ["train", "val", "ext2024", "ext2025"]:
    g = d[d.part == p]; pt = g.drop_duplicates("patient"); q = pt.age.quantile([.25, .5, .75]).values
    evs = [(t, e - s) for st in g.stem for (s, e, t) in ev[st]]
    types = pd.Series([t for t, _ in evs]).value_counts(); dur = np.median([x for _, x in evs]); dx = pt.diagnosis
    rows[p] = {
        "Recordings, n": f"{len(g):,}", "Patients, n": f"{g.patient.nunique()}",
        "Recordings per patient, median (IQR)": f"{g.groupby('patient').size().median():.0f} ({g.groupby('patient').size().quantile(.25):.0f}–{g.groupby('patient').size().quantile(.75):.0f})",
        "Age, years, median (IQR)": f"{q[1]:.1f} ({q[0]:.1f}–{q[2]:.1f})",
        "Female patients, %": f"{100 * (pt.sex == 1).mean():.0f}",
        "Audio, h": f"{g.dur.sum() / 3600:.1f}",
        "Recordings with CAS, n (%)": f"{g.cas.sum()} ({100 * g.cas.mean():.1f})",
        "Recordings with DAS, n (%)": f"{g.das.sum()} ({100 * g.das.mean():.1f})",
        "Annotated events, n": f"{types.sum():,}",
        "Events per recording, median": f"{g.n_events.median():.0f}",
        "Event duration, s, median": f"{dur:.2f}",
        "Fine crackle events, %": f"{100 * types.get('Fine Crackle', 0) / types.sum():.1f}",
        "Coarse crackle events, %": f"{100 * types.get('Coarse Crackle', 0) / types.sum():.1f}",
        "Wheeze events, %": f"{100 * types.get('Wheeze', 0) / types.sum():.1f}",
        "Rhonchi / stridor events, %": f"{100 * types.get('Rhonchi', 0) / types.sum():.1f} / {100 * types.get('Stridor', 0) / types.sum():.1f}",
        "Patients with a recorded diagnosis, %": f"{100 * dx.notna().mean():.0f}",
        "Pneumonia, % of those diagnosed": f"{100 * dx.dropna().str.startswith('Pneumonia').mean():.0f}",
    }
keys = list(rows["train"].keys())
t = "| Characteristic | " + " | ".join(PART[p] for p in rows) + " |\n" + sep(36, 15, 15, 15, 15) + "\n"
t += "\n".join(f"| {k} | " + " | ".join(rows[p][k] for p in rows) + " |" for k in keys)
w("table1", t)
pd.DataFrame(rows).to_csv(R("x_table1_values.csv"))

# ---------------- Table 2: size, memory and latency ----------------
eff_f = "efficiency_v2.csv" if have("efficiency_v2.csv") else "efficiency.csv"
eff = pd.read_csv(R(eff_f)); eff["arch"] = eff.model.replace({"tiny_kd": "tiny"})
ea = eff.groupby("arch").mean(numeric_only=True)
mem = pd.read_csv(R("x_memory_frontend.csv")).set_index("item")
lines = ["| Model | Parameters | MFLOPs per window | Weights, FP32 / INT8 (kB) | ONNX file, INT8 (kB) | Peak activations, INT8 (kB) | Latency, FP32 / INT8 (ms) |",
         sep(18, 13, 10, 15, 11, 13, 14)]
lines.append("| Acoustic features + LR | 198 | – | 0.8 / – | – | – | – |")
for a, lab in [("tiny", "Tiny (with or without KD)"), ("small", "Small"), ("medium", "Reference")]:
    r = ea.loc[a]
    lines.append(f"| {lab} | {int(r.params):,} | {r.mflops:.1f} | {r.wmem_fp32_kb:,.1f} / {r.wmem_int8_kb:,.1f} | {r.int8_kb:,.1f} | "
                 f"{mem.loc[a, 'peak_act_int8_kb']:.0f} | {r.lat_fp32_ms:.2f} / {r.lat_int8_ms:.2f} |")
fe = mem.loc["frontend"]
lines.append(f"| Log-mel front end | – | {fe.frontend_mflops_per_window:.1f} | – | – | – | {fe.frontend_ms_per_window:.2f} / – |")
w("table2", "\n".join(lines))

# ---------------- Table 3: AUROC ----------------
disc = pd.read_csv(R("discrimination.csv"))
lines = ["| Model | Validation, CAS | Validation, DAS | External 2024, CAS | External 2024, DAS | External 2025, CAS | External 2025, DAS |",
         sep(16, 12, 12, 12, 12, 12, 12)]
for m in ["acoustic_lr", "tiny", "tiny_kd", "tiny_kd_int8", "small", "medium"]:
    cells = []
    for p in ["val", "ext2024", "ext2025"]:
        for c in ["cas", "das"]:
            r = disc[(disc.part == p) & (disc.cls == c) & (disc.model == m)].iloc[0]
            cells.append(f"{r.auroc:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})")
    lines.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
pos = []
for p in ["val", "ext2024", "ext2025"]:
    for c in ["cas", "das"]:
        r = disc[(disc.part == p) & (disc.cls == c)].iloc[0]; pos.append(f"{r.n_pos} / {r.n:,}")
lines.append("| Positive / all recordings | " + " | ".join(pos) + " |")
w("table3", "\n".join(lines))

# ---------------- Table 4: paired differences ----------------
pr = pd.read_csv(R("paired.csv"))
CMP = {("tiny_kd", "tiny"): "Tiny + KD − Tiny", ("tiny_kd", "small"): "Tiny + KD − Small", ("tiny_kd", "medium"): "Tiny + KD − Reference",
       ("tiny", "acoustic_lr"): "Tiny − acoustic LR", ("tiny_kd", "tiny_kd_int8"): "Tiny + KD: FP32 − INT8"}
lines = ["| Comparison | Class | Validation | External 2024 | External 2025 |", sep(22, 6, 22, 22, 22)]
for (a, b), lab in CMP.items():
    for c in ["cas", "das"]:
        cells = []
        for p in ["val", "ext2024", "ext2025"]:
            r = pr[(pr.a == a) & (pr.b == b) & (pr.cls == c) & (pr.part == p)].iloc[0]
            cells.append(f"{r['diff']:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f}); {pv(r.p)}")
        lines.append(f"| {lab if c == 'cas' else ''} | {c.upper()} | " + " | ".join(cells) + " |")
w("table4", "\n".join(lines))

# ---------------- Table 5: operating point and AUPRC ----------------
lines = ["| Cohort | Class | Prevalence | Model | AUPRC | Sensitivity | Specificity |", sep(14, 6, 10, 18, 8, 10, 10)]
for p in ["val", "ext2024", "ext2025"]:
    for c in ["cas", "das"]:
        first = True
        for m in ["tiny_kd_int8", "medium", "acoustic_lr"]:
            r = disc[(disc.part == p) & (disc.cls == c) & (disc.model == m)].iloc[0]
            lines.append(f"| {PART[p] if first and c == 'cas' else ''} | {c.upper() if first else ''} | {f'{r.prev:.3f}' if first else ''} | {NAME[m]} | {r.auprc:.3f} | {r.sens:.3f} | {r.spec:.3f} |")
            first = False
w("table5", "\n".join(lines))

# ---------------- Table 6: official challenge protocol ----------------
if have("x_benchmark.csv"):
    b = pd.read_csv(R("x_benchmark.csv")).set_index(["model", "set"])
    lines = ["| Entry | Parameters | 2022 test set: SE | 2022 test set: SP | 2022 test set: Score | GC'23 test set: SE | GC'23 test set: SP | GC'23 test set: Score |", sep(26, 13, 8, 8, 12, 8, 8, 12)]
    for m, lab, npar in [("tiny", "Tiny (this work)", "7,722"), ("tiny_kd", "Tiny + KD (this work)", "7,722"), ("medium", "Reference (this work)", "1,225,442")]:
        a, c = b.loc[(m, "test2022")], b.loc[(m, "gc23")]
        lines.append(f"| {lab} | {npar} | {a.SE:.3f} | {a.SP:.3f} | {a.Score:.3f} ± {a.Score_sd:.3f} | {c.SE:.3f} | {c.SP:.3f} | {c.Score:.3f} ± {c.Score_sd:.3f} |")
    lines += ["| 2022 challenge, 1st of 45 teams on this task (ECE UofT) | – | 0.889 | 0.896 | 0.893 | – | – | 0.748 |",
              "| 2022 challenge, overall winner (NIIT) | – | 0.932 | 0.847 | 0.889 | – | – | 0.785 |",
              "| 2022 challenge, 3rd on this task (IESOC A) | – | 0.817 | 0.883 | 0.849 | – | – | 0.733 |",
              "| 2022 challenge, 5th on this task (WELMO-Team) | – | 0.804 | 0.876 | 0.839 | – | – | – |",
              "| 2022 challenge, 6th on this task (Naseem Babu) | – | 0.836 | 0.836 | 0.836 | – | – | 0.720 |",
              "| 2022 challenge, 8th on this task (The Superego) | – | 0.841 | 0.799 | 0.820 | – | – | 0.668 |",
              "| 2022 challenge, 10th on this task (INTJ) | – | 0.797 | 0.763 | 0.780 | – | – | – |",
              "| 2023 challenge, best on this task (UoE AIT) | – | – | – | – | – | – | 0.810 |",
              "| 2023 challenge, overall winner (Bio ICDS) | – | – | – | – | – | – | 0.769 |"]
    w("table6", "\n".join(lines))

# ---------------- Table 7: annotation practice by cohort ----------------
if have("x_annotation_structure.csv"):
    an = pd.read_csv(R("x_annotation_structure.csv")).set_index("cohort")
    CN = {"dev": "Development (2022–2023)", "ext2024": "External 2024", "ext2025": "External 2025"}
    lines = ["| Measure | " + " | ".join(CN.values()) + " |", sep(54, 18, 14, 14)]
    def row(lab, f):
        lines.append(f"| {lab} | " + " | ".join(f(an.loc[c]) for c in CN) + " |")
    row("Share of recording time covered by annotated events, %", lambda r: f"{100 * r.annotated_share_of_time:.0f}")
    row("DAS-positive recordings, n", lambda r: f"{int(r.das_pos)}")
    row("– crackle events among their events, mean %", lambda r: f"{100 * r.das_event_share:.0f}")
    row("– recordings in which every event is a crackle, %", lambda r: f"{100 * r.das_all_events:.0f}")
    row("– time labelled as crackles per recording, median, s", lambda r: f"{r.das_median_time_s:.1f}")
    row("CAS-positive recordings, n", lambda r: f"{int(r.cas_pos)}")
    row("– continuous-sound events among their events, mean %", lambda r: f"{100 * r.cas_event_share:.0f}")
    row("– recordings in which every event is a continuous sound, %", lambda r: f"{100 * r.cas_all_events:.0f}")
    row("– time labelled as continuous sounds per recording, median, s", lambda r: f"{r.cas_median_time_s:.1f}")
    w("table7", "\n".join(lines))

# ---------------- Table 8: three training regimes for the 2025 release ----------------
if have("x_rolling.csv") and have("x_cv2025.csv"):
    ro = pd.read_csv(R("x_rolling.csv")); cv = pd.read_csv(R("x_cv2025.csv"))
    lines = ["| Training data for the Tiny network | Training recordings | CAS: AUROC (95% CI) | CAS: difference from row 1 | DAS: AUROC (95% CI) | DAS: difference from row 1 |",
             sep(34, 12, 16, 22, 16, 22)]
    cells = []
    for c in ["cas", "das"]:
        r = disc[(disc.part == "ext2025") & (disc.cls == c) & (disc.model == "tiny")].iloc[0]
        cells += [f"{r.auroc:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})", "reference"]
    lines.append(f"| 2022–2023 releases (main analysis) | {(d.part == 'train').sum():,} | " + " | ".join(cells) + " |")
    cells = []
    for c in ["cas", "das"]:
        r = ro[(ro.part == "ext2025") & (ro.cls == c)].iloc[0]
        cells += [f"{r.auroc_roll:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})", f"{r['diff']:+.3f} ({r.d_lo:+.3f} to {r.d_hi:+.3f}); {pv(r.p)}"]
    lines.append(f"| 2022–2024 releases (rolling origin) | {((d.part == 'train') | (d.part == 'ext2024')).sum():,} | " + " | ".join(cells) + " |")
    cells = []
    for c in ["cas", "das"]:
        r = cv[(cv.cls == c) & (cv.base == "tiny")].iloc[0]
        cells += [f"{r.auroc_cv:.3f} ({r.ci_lo:.3f}–{r.ci_hi:.3f})", f"{r['diff']:+.3f} ({r.d_lo:+.3f} to {r.d_hi:+.3f}); {pv(r.p)}"]
    folds = pd.read_csv(R("cv2025_folds.csv")); t25 = d[d.part == "ext2025"]
    ntr = [int((~t25.patient.isin(set(folds[folds.fold == f].patient))).sum()) for f in range(5)]
    lines.append(f"| 2025 release only (five-fold cross-validation) | {min(ntr):,}–{max(ntr):,} per fold | " + " | ".join(cells) + " |")
    w("table8", "\n".join(lines))

# ---------------- Table A9: adaptation to the 2025 release ----------------
if have("x_adaptation_summary.csv"):
    a = pd.read_csv(R("x_adaptation_summary.csv"))
    ML = {"source": "No adaptation", "threshold": "Threshold only", "head": "Linear head (130 parameters)", "finetune": "All parameters"}
    lines = ["| Class | Adapted | k = 5 | k = 10 | k = 20 | k = 40 |", sep(9, 26, 22, 22, 22, 22)]
    for c in ["das", "cas"]:
        for m in ["source", "threshold", "head", "finetune"]:
            cells = []
            for k in [5, 10, 20, 40]:
                r = a[(a.method == m) & (a.cls == c) & (a.k == k)].iloc[0]
                cells.append(f"{r.auroc:.3f}; {r.sens:.2f} / {r.spec:.2f}; {r.bacc:.3f}")
            lines.append(f"| {c.upper() if m == 'source' else ''} | {ML[m]} | " + " | ".join(cells) + " |")
        r = a[(a.method == "source") & (a.cls == c)]
        lines.append(f"|  | Adaptation recordings, mean (positive) | " + " | ".join(
            f"{r[r.k == k].n_rec.iloc[0]:.0f} ({r[r.k == k].n_pos.iloc[0]:.0f})" for k in [5, 10, 20, 40]) + " |")
    w("tableA9", "\n".join(lines))

# ---------------- Appendix tables ----------------
et = pd.read_csv(R("event_type_sensitivity.csv"))
lines = ["| Event type present | Target | Recordings, n | Tiny + KD | Reference |", sep(20, 8, 10, 16, 16)]
for t in et.event_type.unique():
    a = et[(et.event_type == t) & (et.model == "tiny_kd")].iloc[0]; b = et[(et.event_type == t) & (et.model == "medium")].iloc[0]
    lines.append(f"| {t.replace('+', ' + ').lower().capitalize()} | {a.target.upper()} | {a.n_recordings} | {a.sensitivity:.3f} ± {a.sd:.3f} | {b.sensitivity:.3f} ± {b.sd:.3f} |")
w("tableA1", "\n".join(lines))

nz = pd.read_csv(R("noise.csv")); g = nz.groupby(["model", "part", "cls", "snr"]).auroc.mean().reset_index()
lines = ["| Model | Cohort | Class | Clean | 20 dB | 10 dB | 5 dB | 0 dB |", sep(22, 12, 6, 8, 8, 8, 8, 8)]
for m in ["tiny", "tiny_kd", "small", "tiny_kd_na"]:
    for p in ["ext2024", "ext2025"]:
        for c in ["cas", "das"]:
            cells = [f"{g[(g.model == m) & (g.part == p) & (g.cls == c) & (g.snr == s)].auroc.iloc[0]:.3f}" for s in [np.inf, 20, 10, 5, 0]]
            lines.append(f"| {NAME[m]} | {PART[p]} | {c.upper()} | " + " | ".join(cells) + " |")
w("tableA2", "\n".join(lines))

if have("x_kd_ablation.csv"):
    kd = pd.read_csv(R("x_kd_ablation.csv"))
    lines = ["| Training of the Tiny network | T | λ | Validation, CAS | Validation, DAS | External 2024, CAS | External 2024, DAS | External 2025, CAS | External 2025, DAS |", sep(22, 4, 4, 11, 11, 11, 11, 11, 11)]
    for cfg, lab in [("tiny", "Hard labels only"), ("kd_T1", "Distillation"), ("tiny_kd", "Distillation (main analysis)"), ("kd_T4", "Distillation"), ("kd_w1", "Soft labels only")]:
        q = kd[kd.config == cfg]
        if q.empty: continue
        cells = [f"{q[(q.part == p) & (q.cls == c)].auroc.iloc[0]:.3f} ± {q[(q.part == p) & (q.cls == c)].sd.iloc[0]:.3f}" for p in ["val", "ext2024", "ext2025"] for c in ["cas", "das"]]
        T, lam = q.iloc[0]["T"], q.iloc[0]["w"]
        lines.append(f"| {lab} | {'–' if cfg == 'tiny' else int(float(T))} | {'0' if cfg == 'tiny' else lam} | " + " | ".join(cells) + " |")
    w("tableA3", "\n".join(lines))

ag = pd.read_csv(R("x_aggregation.csv"))
AL = {"max": "Maximum (main analysis)", "top2": "Mean of the two largest", "lse": "Log-mean-exp", "mean": "Mean"}
lines = ["| Pooling of window logits | Validation, CAS | Validation, DAS | External 2024, CAS | External 2024, DAS | External 2025, CAS | External 2025, DAS |", sep(24, 11, 11, 11, 11, 11, 11)]
for a_ in ["max", "top2", "lse", "mean"]:
    cells = [f"{ag[(ag['agg'] == a_) & (ag.part == p) & (ag.cls == c)].auroc.iloc[0]:.3f}" for p in ["val", "ext2024", "ext2025"] for c in ["cas", "das"]]
    lines.append(f"| {AL[a_]} | " + " | ".join(cells) + " |")
w("tableA4", "\n".join(lines))

wl = pd.read_csv(R("x_window_level.csv"))
lines = ["| Model | Validation, CAS | Validation, DAS | External 2024, CAS | External 2024, DAS | External 2025, CAS | External 2025, DAS |", sep(16, 11, 11, 11, 11, 11, 11)]
for m in ["tiny", "tiny_kd", "small", "medium"]:
    cells = [f"{wl[(wl.model == m) & (wl.part == p) & (wl.cls == c)].auroc.iloc[0]:.3f}" for p in ["val", "ext2024", "ext2025"] for c in ["cas", "das"]]
    lines.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
q = wl[wl.model == "tiny"]
lines.append("| Positive / all windows | " + " | ".join(f"{q[(q.part == p) & (q.cls == c)].n_pos.iloc[0]:,} / {q[(q.part == p) & (q.cls == c)].n_windows.iloc[0]:,}" for p in ["val", "ext2024", "ext2025"] for c in ["cas", "das"]) + " |")
w("tableA5", "\n".join(lines))

sg = pd.read_csv(R("x_subgroups.csv")); sg = sg[sg.variable != "part"]
VL = {"age_band": "Age", "sex_l": "Sex", "site": "Chest location"}
order = {"age_band": ["<3 y", "3 to <6 y", ">=6 y"], "sex_l": ["Female", "Male"], "site": ["Left posterior", "Right posterior", "Left lateral", "Right lateral", "Additional sites"]}
lines = ["| Subgroup | Level | Recordings, n | CAS: positives | CAS: AUROC (95% CI) | DAS: positives | DAS: AUROC (95% CI) |", sep(14, 16, 10, 10, 20, 10, 20)]
for v in ["age_band", "sex_l", "site"]:
    for i, lev in enumerate(order[v]):
        q = sg[(sg.variable == v) & (sg.level == lev)]
        if q.empty: continue
        cells = []
        for c in ["cas", "das"]:
            r = q[q.cls == c]
            cells += ([f"{r.n_pos.iloc[0]}", f"{r.auroc.iloc[0]:.3f} ({r.ci_lo.iloc[0]:.3f}–{r.ci_hi.iloc[0]:.3f})"] if len(r) else ["<10", "–"])
        lines.append(f"| {VL[v] if i == 0 else ''} | {lev.replace('>=', '≥ ').replace('<3', '< 3').replace('<6', '< 6')} | {q.n.iloc[0]:,} | " + " | ".join(cells) + " |")
w("tableA6", "\n".join(lines))

cal = pd.read_csv(R("x_calibration.csv")); cal = cal[cal.model == "tiny_kd"]
lines = ["| Cohort | Class | Prevalence | Mean predicted probability | Calibration slope | Calibration intercept | Brier score |", sep(14, 6, 10, 14, 12, 12, 10)]
for p in ["val", "ext2024", "ext2025"]:
    for c in ["cas", "das"]:
        r = cal[(cal.part == p) & (cal.cls == c)].iloc[0]
        lines.append(f"| {PART[p] if c == 'cas' else ''} | {c.upper()} | {r.prevalence:.3f} | {r.mean_pred:.3f} | {r.slope:.2f} | {r.intercept:+.2f} | {r.brier:.3f} |")
w("tableA7", "\n".join(lines))

dx = pd.read_csv(R("diagnosis_exploratory.csv"))
FN = {"das": "Annotated DAS", "p_das": "Predicted DAS", "cas": "Annotated CAS", "p_cas": "Predicted CAS"}
CL = {"pneumonia vs control": "Pneumonia vs healthy controls", "pneumonia vs all other diagnoses": "Pneumonia vs all other diagnoses", "severe vs non-severe pneumonia": "Severe vs non-severe pneumonia"}
lines = ["| Comparison | Cohort | Children, n | " + " | ".join(FN.values()) + " |", sep(28, 14, 10, 11, 11, 11, 11)]
for comp in dx.comparison.unique():
    for p in ["val", "ext2024"]:
        q = dx[(dx.comparison == comp) & (dx.part == p)]
        if q.empty: continue
        r = q.iloc[0]
        lines.append(f"| {CL[comp]} | {PART[p]} | {r.n_pos} vs {r.n_neg} | " + " | ".join(f"{q[q.feature == f].auroc.iloc[0]:.2f}" for f in FN) + " |")
w("tableA8", "\n".join(lines))
print("tables:", sorted(os.listdir(OUT)))
