"""Scores under the BioCAS 2022 challenge protocol (Task 1-1, event level). Usage: python3 -I code/analyze_bench.py
Reads results/bench_preds.pkl and results/bench_events.csv; writes results/x_benchmark.csv and x_benchmark_seeds.csv."""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import RES
SEEDS = [0, 1, 2]
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
print(pd.read_csv(os.path.join(RES, "x_benchmark.csv")).round(3).to_string())
