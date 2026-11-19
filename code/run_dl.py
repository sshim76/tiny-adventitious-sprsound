"""Train the CNNs (three seeds each) on the development-training split (BioCAS 2022 + 2023 releases) and store
window-level logits for the development-validation split and the 2024 and 2025 temporal external sets.
Usage: python3 -I code/run_dl.py   -> results/dl_preds.pkl, results/models/*.pt, results/dl.log
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

SEEDS, EPOCHS = [0, 1, 2], 20
os.makedirs(os.path.join(RES, "models"), exist_ok=True)
LOG = open(os.path.join(RES, "dl.log"), "a")
def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


d = load_index(); events = load_events()
# patient-level 80/20 split of the development cohort
rng = np.random.default_rng(2026)
dev_p = np.array(sorted(d[d.cohort == "dev"].patient.unique().tolist()), dtype=object)
rng.shuffle(dev_p)
val_p = set(dev_p[:int(round(0.2 * len(dev_p)))])
d["part"] = np.where(d.cohort != "dev", d.cohort, np.where(d.patient.isin(val_p), "val", "train"))
d[["stem", "patient", "part", "cas", "das"]].to_csv(os.path.join(RES, "split.csv"), index=False)
tr = d[d.part == "train"].reset_index(drop=True)
mu, sd = norm_stats(tr.stem.values)
json.dump({"mu": mu, "sd": sd}, open(os.path.join(RES, "norm.json"), "w"))
for p, g in d.groupby("part"):
    log(f"{p}: {len(g)} recordings, {g.patient.nunique()} patients, CAS {g.cas.mean():.3f}, DAS {g.das.mean():.3f}")
ev_stems = d[d.part != "train"].stem.values
pp = os.path.join(RES, "dl_preds.pkl")
preds = pickle.load(open(pp, "rb")) if os.path.exists(pp) else {}


def run(name, kind, seed, **kw):
    if (name, seed) in preds:
        return
    t0 = time.time()
    m = train_model(tr, events, kind=kind, seed=seed, epochs=EPOCHS, mu=mu, sd=sd, log=log, **kw)
    torch.save(m.state_dict(), os.path.join(RES, "models", f"{name}_s{seed}.pt"))
    preds[(name, seed)] = window_logits(m, ev_stems, mu, sd)
    pickle.dump(preds, open(pp, "wb"))
    log(f"{name} seed {seed} done in {time.time() - t0:.0f}s")


for s in SEEDS:                       # lightweight models first
    run("tiny", "tiny", s)
for s in SEEDS:                       # reference model, also the distillation teacher
    run("medium", "medium", s)
if "teacher_train" not in preds:      # teacher-ensemble logits for every training window on the 1-s grid
    tw = []
    for s in SEEDS:
        t = Net("medium"); t.load_state_dict(torch.load(os.path.join(RES, "models", f"medium_s{s}.pt"))); t.eval()
        tw.append(window_logits(t, tr.stem.values, mu, sd))
    preds["teacher_train"] = {k: np.mean([w[k] for w in tw], axis=0) for k in tr.stem.values}
    pickle.dump(preds, open(pp, "wb"))
for s in SEEDS:
    run("tiny_kd", "tiny", s, soft=preds["teacher_train"], kd_w=0.5)
for s in SEEDS:
    run("small", "small", s)
log("all done")
