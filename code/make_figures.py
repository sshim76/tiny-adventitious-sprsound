"""Figures of the manuscript. Usage: python3 -I code/make_figures.py  (after analyze.py and analyze_extra.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, YEL, MAG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#1f1f1e", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "savefig.dpi": 300})
COL = {"tiny_kd_na": MAG, "acoustic_lr": INK2, "tiny": AQUA, "tiny_kd": BLUE, "small": YEL, "medium": ORANGE}
LAB = {"tiny_kd_na": "Tiny + KD, noise-augmented", "acoustic_lr": "Acoustic features + LR", "tiny": "Tiny", "tiny_kd": "Tiny + KD",
       "small": "Small", "medium": "Reference"}
PART = {"val": "Validation", "ext2024": "External 2024", "ext2025": "External 2025"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".tif"), bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def clean(ax, g="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=g, color=GRID, lw=0.5); ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, fc="#f3f2ef", ec=INK2, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))


# ---------------- Figure 1: study design ----------------
d = load_index(); sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
d = d.drop(columns=["part"]).merge(sp[["stem", "part"]], on="stem")
cnt = d.groupby("part").agg(n=("stem", "size"), p=("patient", "nunique"))
npt, npm = n_params(Net("tiny")), n_params(Net("medium"))
fig, ax = plt.subplots(figsize=(6.85, 3.5)); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.09, 0.97); ax.axis("off")
F = 6.0
box(ax, 0.00, 0.66, 0.27, 0.28, f"Development cohort\nSPRSound 2022 and 2023 releases\n{cnt.loc['train', 'n'] + cnt.loc['val', 'n']:,} recordings, "
    f"{cnt.loc['train', 'p'] + cnt.loc['val', 'p']} patients", fc="#e3eefb", ec=BLUE, fs=F)
box(ax, 0.00, 0.30, 0.125, 0.24, f"Training\n{cnt.loc['train', 'n']:,} rec.\n{cnt.loc['train', 'p']} patients", fc="#e3eefb", ec=BLUE, fs=F)
box(ax, 0.145, 0.30, 0.125, 0.24, f"Validation\n{cnt.loc['val', 'n']:,} rec.\n{cnt.loc['val', 'p']} patients", fc="#e3eefb", ec=BLUE, fs=F)
arrow(ax, 0.065, 0.66, 0.065, 0.54); arrow(ax, 0.205, 0.66, 0.205, 0.54)
ax.text(0.135, 0.245, "patient-level 80/20 split", ha="center", fontsize=6.3, color=INK2)
box(ax, 0.00, 0.01, 0.125, 0.20, f"External 2024\n{cnt.loc['ext2024', 'n']:,} rec.\n{cnt.loc['ext2024', 'p']} patients", fc="#fde8df", ec=ORANGE, fs=F)
box(ax, 0.145, 0.01, 0.125, 0.20, f"External 2025\n{cnt.loc['ext2025', 'n']:,} rec.\n{cnt.loc['ext2025', 'p']} patients", fc="#fde8df", ec=ORANGE, fs=F)
box(ax, 0.31, 0.66, 0.18, 0.28, "Front end\n8-kHz audio\n64-band log-mel\n10-ms frames\n4-s windows, 1-s hop", fs=F)
arrow(ax, 0.27, 0.80, 0.31, 0.80)
box(ax, 0.53, 0.66, 0.20, 0.28, f"Reference CNN\nresidual blocks\n{npm / 1e6:.1f} M parameters\nthree-seed ensemble", fc="#fde8df", ec=ORANGE, fs=F)
box(ax, 0.53, 0.26, 0.20, 0.28, f"Tiny CNN\ndepthwise-separable\n{npt / 1e3:.1f} k parameters", fc="#e3eefb", ec=BLUE, fs=F)
arrow(ax, 0.49, 0.80, 0.53, 0.80); arrow(ax, 0.42, 0.66, 0.53, 0.44)
arrow(ax, 0.63, 0.66, 0.63, 0.54); ax.text(0.638, 0.60, "window-level soft labels (KD)", fontsize=6.0, color=INK2, va="center")
box(ax, 0.77, 0.66, 0.23, 0.28, "Two logits per window\n(CAS, DAS)\nrecording score =\nmaximum over windows", fs=F)
box(ax, 0.77, 0.26, 0.23, 0.28, "Deployment check\nONNX FP32 and INT8\nsingle-thread CPU\nlatency and model size", fc="#e7f6ef", ec=AQUA, fs=F)
arrow(ax, 0.73, 0.40, 0.77, 0.40); arrow(ax, 0.73, 0.80, 0.77, 0.80)
ax.text(0.64, 0.10, "Evaluation on validation and both external releases:\nAUROC and AUPRC, patient-cluster bootstrap CIs, three training seeds,\nnoise robustness, sensitivity by event type", ha="center", va="center", fontsize=6.3, color=INK2)
save(fig, "Figure1")

if not os.path.exists(os.path.join(RES, "discrimination.csv")):
    sys.exit("figure 1 only")
disc = pd.read_csv(os.path.join(RES, "discrimination.csv")); eff = pd.read_csv(os.path.join(RES, "efficiency.csv"))
effm = eff.groupby("model").agg(params=("params", "first"), mflops=("mflops", "first"))
effm.loc["acoustic_lr"] = [98, np.nan]

# ---------------- Figure 2: accuracy versus compute ----------------
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.8), sharey=True)
for ax, c, title in zip(axs, ["cas", "das"], ["Continuous adventitious sounds (CAS)", "Discontinuous adventitious sounds (DAS)"]):
    for mname in ["tiny", "tiny_kd", "small", "medium"]:
        for p, mk in [("ext2024", "o"), ("ext2025", "s")]:
            r = disc[(disc.part == p) & (disc.cls == c) & (disc.model == mname)].iloc[0]
            jit = {"tiny": 0.93, "tiny_kd": 1.07}.get(mname, 1.0)
            ax.errorbar(effm.loc[mname, "mflops"] * jit, r.auroc, yerr=[[r.auroc - r.ci_lo], [r.ci_hi - r.auroc]], fmt=mk,
                        color=COL[mname], ms=6, mec="white", mew=0.8, elinewidth=1.0, capsize=0)
    ax.set_xscale("log"); ax.set_xlabel("MFLOPs per 4-s window (log scale)"); ax.set_title(title, fontsize=8)
    ax.set_xticks([10, 30, 100, 300, 1000]); ax.set_xticklabels(["10", "30", "100", "300", "1000"]); ax.minorticks_off(); ax.set_xlim(8, 1500)
    clean(ax, "both")
axs[0].set_ylabel("AUROC (95% CI)")
h = [Line2D([], [], marker="o", color=COL[m], lw=0, ms=6, label=LAB[m]) for m in ["tiny", "tiny_kd", "small", "medium"]]
h += [Line2D([], [], marker="o", color=INK2, lw=0, ms=5, label="External 2024"), Line2D([], [], marker="s", color=INK2, lw=0, ms=5, label="External 2025")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=7)
save(fig, "Figure2")

# ---------------- Figure 3: ROC curves on the external sets (seed 0..2 averaged) ----------------
preds = pickle.load(open(os.path.join(RES, "dl_preds.pkl"), "rb"))
E = d[d.part.isin(["val", "ext2024", "ext2025"])].reset_index(drop=True)
fig, axs = plt.subplots(2, 2, figsize=(6.85, 6.0))
grid = np.linspace(0, 1, 201)
for i, p in enumerate(["ext2024", "ext2025"]):
    q = E[E.part == p]
    for j, (ci, c) in enumerate([(0, "cas"), (1, "das")]):
        ax = axs[i, j]
        for mname in ["tiny", "tiny_kd", "medium"]:
            tprs = []
            for s in [0, 1, 2]:
                rs = recording_scores({k: preds[(mname, s)][k] for k in q.stem}); sc = np.array([rs[k][ci] for k in q.stem])
                fpr, tpr, _ = roc_curve(q[c], sc); tprs.append(np.interp(grid, fpr, tpr))
            r = disc[(disc.part == p) & (disc.cls == c) & (disc.model == mname)].iloc[0]
            ax.plot(grid, np.mean(tprs, 0), color=COL[mname], lw=1.6, label=f"{LAB[mname]}: {r.auroc:.2f}")
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.8)
        ax.set_title(f"{PART[p]} · {c.upper()} (n = {len(q)}, positives = {int(q[c].sum())})", fontsize=8)
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); clean(ax, "both")
        ax.legend(loc="lower right", frameon=False, fontsize=6.5, title="AUROC (mean of 3 seeds)", title_fontsize=6.5)
fig.tight_layout(); save(fig, "Figure3")

# ---------------- Figure 4: the 2025 release ----------------
R_ = lambda f: os.path.join(RES, f)
if all(os.path.exists(R_(f)) for f in ["x_rolling.csv", "x_cv2025.csv", "x_adaptation_summary.csv"]):
    ro = pd.read_csv(R_("x_rolling.csv")); cv = pd.read_csv(R_("x_cv2025.csv")); ad = pd.read_csv(R_("x_adaptation_summary.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.9), sharey=True, gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0], wspace=0.10))
    ax = axs[0]
    for ci, (c, col, off) in enumerate([("cas", AQUA, -0.12), ("das", ORANGE, 0.12)]):
        r0 = disc[(disc.part == "ext2025") & (disc.cls == c) & (disc.model == "tiny")].iloc[0]
        r1 = ro[(ro.part == "ext2025") & (ro.cls == c)].iloc[0]; r2 = cv[(cv.cls == c) & (cv.base == "tiny")].iloc[0]
        vals = [(r0.auroc, r0.ci_lo, r0.ci_hi), (r1.auroc_roll, r1.ci_lo, r1.ci_hi), (r2.auroc_cv, r2.ci_lo, r2.ci_hi)]
        for i, (v, lo, hi) in enumerate(vals):
            ax.plot([i + off, i + off], [lo, hi], color=col, lw=1.4, solid_capstyle="round")
            ax.plot(i + off, v, "o" if c == "cas" else "s", color=col, ms=6, mec="white", mew=0.8, label=c.upper() if i == 0 else None)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["2022–\n2023", "2022–\n2024", "2025 only\n(5-fold CV)"], fontsize=7)
    ax.set_xlabel("Training data (Tiny network)"); ax.set_ylabel("AUROC on 2025 recordings")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.5, 0.9); clean(ax, "y"); ax.legend(loc="lower left", frameon=False, fontsize=7, ncol=2, handletextpad=0.2, columnspacing=0.8)
    ax.set_title("a  Training regime", loc="left", fontweight="bold", fontsize=8)
    MS = {"threshold": ("Threshold only", INK2, "v"), "head": ("Linear head", BLUE, "o"), "finetune": ("All parameters", MAG, "D")}
    ks = [5, 10, 20, 40]; xk = np.arange(4)
    for ax, c, ttl in [(axs[1], "das", "b  Adaptation, DAS"), (axs[2], "cas", "c  Adaptation, CAS")]:
        src = ad[(ad.method == "source") & (ad.cls == c)].auroc.mean()
        ax.axhline(src, color=INK2, lw=1.0, ls="--", label="No adaptation")
        cvv = cv[(cv.cls == c) & (cv.base == "tiny")].iloc[0].auroc_cv
        ax.axhline(cvv, color=INK2, lw=1.0, ls=":", label="Trained on 2025 only")
        for j, m in enumerate(["head", "finetune"]):
            q = ad[(ad.method == m) & (ad.cls == c)].set_index("k").loc[ks]
            ax.errorbar(xk + (j - 0.5) * 0.14, q.auroc, yerr=q.auroc_sd, color=MS[m][1], marker=MS[m][2], ms=4.5, lw=1.3, mec="white", mew=0.6, label=MS[m][0], capsize=0)
        ax.set_xticks(xk); ax.set_xticklabels([str(k) for k in ks]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.5, 0.9)
        ax.set_xlabel("Patients used for adaptation")
        clean(ax, "y"); ax.set_title(ttl, loc="left", fontweight="bold", fontsize=8)
    hh, ll = axs[1].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False, fontsize=6.8, columnspacing=1.0, handlelength=1.8)
    save(fig, "Figure4")

# ---------------- Figure 5: noise robustness ----------------
nz = pd.read_csv(os.path.join(RES, "noise.csv")); nz["snr_x"] = nz.snr.replace(np.inf, 30)
fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.7), sharey=True)
for ax, c in zip(axs, ["cas", "das"]):
    for mname in ["tiny", "tiny_kd", "small", "tiny_kd_na"]:
        for p, ls in [("ext2024", "-"), ("ext2025", "--")]:
            g = nz[(nz.model == mname) & (nz.cls == c) & (nz.part == p)].groupby("snr_x").auroc.agg(["mean", "std"]).sort_index()
            ax.errorbar(g.index, g["mean"], yerr=g["std"], color=COL[mname], ls=ls, marker="o", ms=4, lw=1.4, capsize=0)
    ax.set_xticks([0, 5, 10, 20, 30]); ax.set_xticklabels(["0", "5", "10", "20", "clean"])
    ax.set_xlabel("Signal-to-noise ratio of added noise (dB)"); ax.set_title(c.upper(), fontsize=8); clean(ax, "both")
axs[0].set_ylabel("AUROC (mean ± SD over 3 seeds)")
h = [Line2D([], [], color=COL[m], lw=1.4, marker="o", ms=4, label=LAB[m]) for m in ["tiny", "tiny_kd", "small", "tiny_kd_na"]]
h += [Line2D([], [], color=INK2, lw=1.4, ls="-", label="External 2024"), Line2D([], [], color=INK2, lw=1.4, ls="--", label="External 2025")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=7)
save(fig, "Figure5")

# ---------------- Figure 6: example recording with event annotation and window probabilities ----------------
if not os.path.exists(os.path.join(ROOT, "data/cache/logmel.npz")):
    sys.exit("Figures 1-5 written; Figure 6 needs the SPRSound audio (run code/build_index.py and code/features.py first)")
ev = load_events(); nrm = json.load(open(os.path.join(RES, "norm.json")))
q = E[(E.part == "ext2024") & (E.das == 1) & (E.event_types.fillna("").str.contains("Normal"))].copy()
q["score"] = [np.mean([preds[("tiny_kd", s)][k].max(0)[1] for s in [0, 1, 2]]) for k in q.stem]
stem = q.sort_values("score", ascending=False).stem.iloc[0]
x = mels()[stem].astype(np.float32); T = x.shape[1] * HOP_S
wl = np.mean([preds[("tiny_kd", s)][stem] for s in [0, 1, 2]], axis=0); pr = 1 / (1 + np.exp(-wl))
st = np.array(starts_for(max(x.shape[1], CROP))) * HOP_S + CROP * HOP_S / 2
fig, axs = plt.subplots(2, 1, figsize=(6.85, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
axs[0].imshow(x, origin="lower", aspect="auto", extent=[0, T, 0, N_MELS], cmap="magma",
              vmin=np.percentile(x, 5), vmax=np.percentile(x, 99.5))
axs[0].set_ylabel("Mel band (lower 32 of 64)")
for s, e, t in ev[stem]:
    col = ORANGE if t in DAS_T and t not in CAS_T else (AQUA if t in CAS_T and t not in DAS_T else (MAG if t in CAS_T else "#bbbbbb"))
    axs[0].add_patch(plt.Rectangle((s, 32.5), e - s, 1.6, color=col, clip_on=False))
axs[0].set_ylim(0, 32)
axs[1].plot(st, pr[:, 0], "-o", color=AQUA, ms=3.5, lw=1.4, label="CAS probability")
axs[1].plot(st, pr[:, 1], "-s", color=ORANGE, ms=3.5, lw=1.4, label="DAS probability")
vi = E[E.part == "val"]; ths = {0: [], 1: []}
for sd_ in [0, 1, 2]:
    rsv = recording_scores({k: preds[("tiny_kd", sd_)][k] for k in vi.stem})
    for ci, c in [(0, "cas"), (1, "das")]:
        fpr, tpr, thr = roc_curve(vi[c], [rsv[k][ci] for k in vi.stem]); ths[ci].append(thr[np.argmax(tpr - fpr)])
for ci, col in [(0, AQUA), (1, ORANGE)]:
    axs[1].axhline(1 / (1 + np.exp(-np.mean(ths[ci]))), color=col, ls=":", lw=1.0)
axs[1].text(0.1, 1 / (1 + np.exp(-np.mean(ths[1]))) + 0.03, "validation thresholds (dotted)", fontsize=6.5, color=INK2)
axs[1].set_ylim(0, 1); axs[1].set_ylabel("Window probability"); axs[1].set_xlabel("Time (s; window centres)")
clean(axs[1], "y"); axs[1].legend(loc="upper right", frameon=False, fontsize=7, ncol=2)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#bbbbbb", AQUA, ORANGE, MAG]]
axs[0].legend(h, ["Normal", "CAS event", "DAS event", "Wheeze+crackle"], loc="upper right", bbox_to_anchor=(1.0, 1.32), ncol=4, frameon=False, fontsize=6.5)
open(os.path.join(RES, "figure6_example.txt"), "w").write(stem)
save(fig, "Figure6")
print("figures written")
