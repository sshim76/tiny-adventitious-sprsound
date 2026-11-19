"""Index all SPRSound recordings with patient, demographics, release, annotations and diagnosis.
Usage: python3 -I code/build_index.py"""
import os, glob, json
import pandas as pd, numpy as np
import soundfile as sf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data/src/sprsound")
rows = []
for wav in sorted(glob.glob(os.path.join(SRC, "BioCAS20*", "*_wav", "*.wav"))):
    rel = os.path.relpath(wav, SRC); year = int(rel[6:10]); part = rel.split("/")[1]
    stem = os.path.basename(wav)[:-4]
    js = glob.glob(os.path.join(SRC, f"BioCAS{year}", part.replace("_wav", "_json"), "**", stem + ".json"), recursive=True)
    ann = json.load(open(js[0])) if js else {}
    sub = os.path.basename(os.path.dirname(js[0])) if js else ""
    pid, age, sex, loc, rid = stem.split("_")
    info = sf.info(wav); sr_ = info.samplerate
    ev = ann.get("event_annotation", [])
    types = [e["type"] for e in ev]
    rows.append(dict(file=rel, stem=stem, patient=pid, age=float(age), sex=int(sex), loc=loc, rec_id=rid, release=year, part=part,
                     subset=sub, record_label=ann.get("record_annotation", None), n_events=len(ev),
                     n_adv_events=sum(t != "Normal" for t in types), event_types=";".join(sorted(set(types))),
                     sr=sr_, dur=info.duration, channels=info.channels, subtype=info.subtype))
df = pd.DataFrame(rows)
summ = pd.read_csv(os.path.join(SRC, "Patient Summary", "Grand_Challenge'24_patient_summary.csv"), dtype={"patient_num": str})
s23 = pd.read_csv(os.path.join(SRC, "Patient Summary", "Grand_Challenge'23_patient_summary.csv"), dtype={"patient_num": str})
s22 = pd.read_csv(os.path.join(SRC, "Patient Summary", "SPRSound_patient_summary.csv"), dtype={"patient_num": str})
allsum = pd.concat([summ.assign(src=24), s23.assign(src=23), s22[["patient_num", "disease", "age", "gender"]].assign(src=22)])
allsum["disease"] = allsum.disease.str.strip()
dmap = allsum.drop_duplicates("patient_num", keep="first").set_index("patient_num").disease
df["diagnosis"] = df.patient.map(dmap)
df.to_csv(os.path.join(ROOT, "results/index.csv"), index=False)
print(df.shape, df.patient.nunique())
print(df.groupby("release").agg(n=("file", "size"), pts=("patient", "nunique"), dx=("diagnosis", lambda s: s.notna().mean()), sr=("sr", "median"), dur=("dur", "median")))
print(df.record_label.value_counts(dropna=False))
