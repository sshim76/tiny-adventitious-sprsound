"""Latency re-measurement on an idle machine: median of 1,000 single-window inferences after 50 warm-up runs,
ONNX Runtime, one thread. Also re-times the log-mel front end. Usage: python3 -I code/latency.py
Reads results/efficiency.csv and results/onnx/*.onnx; writes results/efficiency_v2.csv and updates the
front-end row of results/x_memory_frontend.csv.
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import onnxruntime as ort
import librosa

nrm = json.load(open(os.path.join(RES, "norm.json")))
sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
stem = sp[sp.part == "train"].stem.values[0]
x = windows(mels()[stem], nrm["mu"], nrm["sd"])[:1].astype(np.float32)


def lat(path, n=1000, warm=50):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(warm): s.run(None, {"x": x})
    meds = []
    for _ in range(3):                      # three blocks of n runs; keep the fastest block median
        ts = []
        for _ in range(n):
            t = time.perf_counter(); s.run(None, {"x": x}); ts.append((time.perf_counter() - t) * 1000)
        meds.append(statistics.median(ts))
    return min(meds)


eff = pd.read_csv(os.path.join(RES, "efficiency.csv"))
for i, r in eff.iterrows():
    base = os.path.join(RES, "onnx", f"{r.model}_s{int(r.seed)}")
    eff.loc[i, "lat_fp32_ms"] = lat(base + "_fp32.onnx"); eff.loc[i, "lat_int8_ms"] = lat(base + "_int8.onnx")
    print(r.model, int(r.seed), round(eff.loc[i, "lat_fp32_ms"], 3), round(eff.loc[i, "lat_int8_ms"], 3), flush=True)
eff.to_csv(os.path.join(RES, "efficiency_v2.csv"), index=False)

a = np.random.default_rng(0).standard_normal(4 * 8000).astype(np.float32)
def fe():
    return np.log(librosa.feature.melspectrogram(y=a, sr=8000, n_fft=256, win_length=200, hop_length=80, n_mels=64,
                                                 fmin=50, fmax=4000, power=2.0) + 1e-6)
for _ in range(50): fe()
ts = []
for _ in range(1000):
    t = time.perf_counter(); fe(); ts.append((time.perf_counter() - t) * 1000)
mf = pd.read_csv(os.path.join(RES, "x_memory_frontend.csv"))
mf.loc[mf.item == "frontend", "frontend_ms_per_window"] = statistics.median(ts)
mf.to_csv(os.path.join(RES, "x_memory_frontend.csv"), index=False)
print("front end ms per 4-s window:", round(statistics.median(ts), 3))
