"""Deployment metrics: parameters, MFLOPs, ONNX FP32/INT8 size, single-thread CPU latency, and INT8 accuracy.
Usage: python3 -I code/efficiency.py   (after run_dl.py) -> results/efficiency.csv, results/int8_preds.pkl
"""
import os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

OUT = os.path.join(RES, "onnx"); os.makedirs(OUT, exist_ok=True)
nrm = json.load(open(os.path.join(RES, "norm.json"))); mu, sd = nrm["mu"], nrm["sd"]
sp = pd.read_csv(os.path.join(RES, "split.csv"), dtype={"patient": str})
train_stems = sp[sp.part == "train"].stem.values
ev_stems = sp[sp.part.isin(["val", "ext2024", "ext2025"])].stem.values
rng = np.random.default_rng(0)
calib = np.concatenate([windows(mels()[s], mu, sd)[:1] for s in rng.choice(train_stems, 300, replace=False)]).astype(np.float32)


class Reader(CalibrationDataReader):
    def __init__(self): self.i = 0
    def get_next(self):
        if self.i >= len(calib): return None
        self.i += 1; return {"x": calib[self.i - 1:self.i]}


def session(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def latency(sess, n=300):
    x = calib[:1]
    for _ in range(20): sess.run(None, {"x": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter(); sess.run(None, {"x": x}); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


rows, int8 = [], {}
for name, kind in [("tiny", "tiny"), ("tiny_kd", "tiny"), ("small", "small"), ("medium", "medium")]:
    for seed in [0, 1, 2]:
        m = Net(kind); m.load_state_dict(torch.load(os.path.join(RES, "models", f"{name}_s{seed}.pt"))); m.eval()
        f32 = os.path.join(OUT, f"{name}_s{seed}_fp32.onnx"); q8 = os.path.join(OUT, f"{name}_s{seed}_int8.onnx")
        torch.onnx.export(m, torch.zeros(1, 1, N_MELS, CROP), f32, input_names=["x"], output_names=["y"],
                          dynamic_axes={"x": {0: "b"}, "y": {0: "b"}}, opset_version=17, dynamo=False)
        quantize_static(f32, q8, Reader(), quant_format=QuantFormat.QOperator, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
        s32, s8 = session(f32), session(q8)
        r = dict(model=name, seed=seed, params=n_params(m), mflops=mflops(m), wmem_fp32_kb=n_params(m) * 4 / 1024,
                 wmem_int8_kb=n_params(m) / 1024, fp32_kb=os.path.getsize(f32) / 1024,
                 int8_kb=os.path.getsize(q8) / 1024, lat_fp32_ms=latency(s32), lat_int8_ms=latency(s8))
        if name in ("tiny_kd", "tiny", "small"):          # INT8 predictions on all evaluation recordings
            int8[(name, seed)] = {s: s8.run(None, {"x": windows(mels()[s], mu, sd).astype(np.float32)})[0] for s in ev_stems}
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "efficiency.csv"), index=False)
pickle.dump(int8, open(os.path.join(RES, "int8_preds.pkl"), "wb"))
print("done")
