"""Log-mel spectrograms (cached) and hand-crafted acoustic features for every SPRSound recording.
Usage: python3 -I code/features.py   (reads results/index.csv; writes data/cache/logmel.npz, results/acoustic_features.csv)
"""
import os, warnings
import numpy as np, pandas as pd
import soundfile as sf, librosa

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data/src/sprsound")
SR, N_FFT, WIN, HOP, N_MELS, FMIN, FMAX = 8000, 256, 200, 80, 64, 50, 4000
BANDS = [(50, 150), (150, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4000)]


def load(path):
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR
    x = x - x.mean()
    return x / (np.max(np.abs(x)) + 1e-8)  # peak normalisation per recording


def logmel(x):
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP, n_mels=N_MELS,
                                       fmin=FMIN, fmax=FMAX, power=2.0, center=True)
    return np.log(m + 1e-6).astype(np.float16)


def acoustic(x):
    f = {}
    mf = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=20, n_fft=N_FFT, win_length=WIN, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    d1 = librosa.feature.delta(mf)
    for i in range(20):
        f[f"mfcc{i}_mean"], f[f"mfcc{i}_sd"] = mf[i].mean(), mf[i].std()
        f[f"dmfcc{i}_mean"], f[f"dmfcc{i}_sd"] = d1[i].mean(), d1[i].std()
    S = np.abs(librosa.stft(x, n_fft=N_FFT, win_length=WIN, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    for name, v in [("centroid", librosa.feature.spectral_centroid(S=S, sr=SR)[0]),
                    ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR)[0]),
                    ("rolloff", librosa.feature.spectral_rolloff(S=S, sr=SR)[0]),
                    ("flatness", librosa.feature.spectral_flatness(S=S)[0]),
                    ("zcr", librosa.feature.zero_crossing_rate(x, frame_length=WIN, hop_length=HOP)[0]),
                    ("rms", librosa.feature.rms(y=x, frame_length=WIN, hop_length=HOP)[0])]:
        f[f"{name}_mean"], f[f"{name}_sd"] = float(np.mean(v)), float(np.std(v))
    tot = S.sum() + 1e-12
    for lo, hi in BANDS:
        f[f"band_{lo}_{hi}"] = float(S[(freqs >= lo) & (freqs < hi)].sum() / tot)
    return f


def main():
    idx = pd.read_csv(os.path.join(ROOT, "results/index.csv"), dtype={"patient": str})
    mels, feats = {}, []
    for i, r in idx.iterrows():
        x = load(os.path.join(SRC, r.file))
        mels[r.stem] = logmel(x)
        feats.append(dict(stem=r.stem, **acoustic(x)))
        if i % 500 == 0:
            print(i, flush=True)
    os.makedirs(os.path.join(ROOT, "data/cache"), exist_ok=True)
    np.savez(os.path.join(ROOT, "data/cache/logmel.npz"), **mels)
    pd.DataFrame(feats).to_csv(os.path.join(ROOT, "results/acoustic_features.csv"), index=False)
    print("done", len(feats))


if __name__ == "__main__":
    main()
