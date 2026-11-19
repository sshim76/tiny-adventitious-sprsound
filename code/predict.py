"""Score one or more WAV files with the 8-bit Tiny + KD network (mean of the three seeds).
Usage: python3 -I code/predict.py recording.wav [more.wav ...]

Prints, for each file, the probability that continuous (CAS) and discontinuous (DAS)
adventitious sounds are present: the sigmoid of the maximum 4-s window logit, as in the paper.
Research code only; it is not a medical device and its outputs are not calibrated for clinical use.
"""
import os, sys, glob, json
import numpy as np
import soundfile as sf, librosa
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import SR, logmel                      # 8 kHz, 64-band log-mel, 10 ms hop
from common import RES, windows

nrm = json.load(open(os.path.join(RES, "norm.json")))
models = sorted(glob.glob(os.path.join(RES, "onnx", "tiny_kd_s*_int8.onnx")))
sessions = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in models]


def load_any(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)                                   # mono
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    x = x - x.mean()
    return x / (np.max(np.abs(x)) + 1e-8)                # peak normalisation, as in features.py


def score(path):
    w = windows(logmel(load_any(path)), nrm["mu"], nrm["sd"]).astype(np.float32)
    logits = np.mean([s.run(None, {"x": w})[0] for s in sessions], axis=0)   # (windows, 2), mean over seeds
    rec = logits.max(axis=0)                             # maximum over windows, per class
    return 1 / (1 + np.exp(-rec)), len(w)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("file\tCAS_probability\tDAS_probability\twindows")
    for p in sys.argv[1:]:
        pr, n = score(p)
        print(f"{os.path.basename(p)}\t{pr[0]:.3f}\t{pr[1]:.3f}\t{n}")
