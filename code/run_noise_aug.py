"""Post hoc experiment: Tiny + KD trained with additional white-noise augmentation (SNR 0-30 dB, probability 0.5).
Usage: python3 -I code/run_noise_aug.py  (after run_dl.py and analyze.py) -> adds ("tiny_kd_na", seed) to dl_preds.pkl
and appends its rows to results/noise.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from sklearn.metrics import roc_auc_score

LOG = open(os.path.join(RES, "dl.log"), "a")
def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
d = load_index(); events = load_events()
sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
d = d.drop(columns=["part"]).merge(sp[["stem", "part"]], on="stem")
tr = d[d.part == "train"].reset_index(drop=True)
nrm = json.load(open(os.path.join(RES, "norm.json"))); mu, sd = nrm["mu"], nrm["sd"]
pp = os.path.join(RES, "dl_preds.pkl"); preds = pickle.load(open(pp, "rb"))
ev_stems = d[d.part != "train"].stem.values
ext = d[d.part.isin(["ext2024", "ext2025"])].reset_index(drop=True)
rows = []
for s in [0, 1, 2]:
    if ("tiny_kd_na", s) not in preds:
        t0 = time.time()
        m = train_model(tr, events, kind="tiny", seed=s, epochs=20, mu=mu, sd=sd, soft=preds["teacher_train"], kd_w=0.5, log=log, noise_aug=True)
        torch.save(m.state_dict(), os.path.join(RES, "models", f"tiny_kd_na_s{s}.pt"))
        preds[("tiny_kd_na", s)] = window_logits(m, ev_stems, mu, sd); pickle.dump(preds, open(pp, "wb"))
        log(f"tiny_kd_na seed {s} done in {time.time() - t0:.0f}s")
    m = Net("tiny"); m.load_state_dict(torch.load(os.path.join(RES, "models", f"tiny_kd_na_s{s}.pt"))); m.eval()
    for snr in [np.inf, 20, 10, 5, 0]:
        if np.isinf(snr):
            rs = recording_scores({k: preds[("tiny_kd_na", s)][k] for k in ext.stem})
        else:
            rs = recording_scores(window_logits(m, ext.stem.values, mu, sd, noise_snr=snr, rng=np.random.default_rng(100 + s)))
        sc = np.stack([rs[x] for x in ext.stem])
        for p in ["ext2024", "ext2025"]:
            sel = (ext.part == p).values
            for ci, c in enumerate(["cas", "das"]):
                rows.append(dict(model="tiny_kd_na", seed=s, snr=snr, part=p, cls=c, auroc=roc_auc_score(ext[c].values[sel], sc[sel, ci])))
    print("noise eval seed", s, flush=True)
nz = pd.read_csv(os.path.join(RES, "noise.csv")); nz = nz[nz.model != "tiny_kd_na"]
pd.concat([nz, pd.DataFrame(rows)]).to_csv(os.path.join(RES, "noise.csv"), index=False)
print("done")
