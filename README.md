# Tiny adventitious lung-sound detector on SPRSound

Code, patient-level splits, trained networks and 8-bit ONNX models for the paper

> Seong-O Shim, Hezam Albaqami, Safa Habibullah, Lal Hussain.
> *A 7.7k-parameter detector of adventitious lung sounds in children: distillation, 8-bit quantisation and temporal validation across yearly releases of SPRSound.* (under review)

The scripts reproduce every number, table and figure in the paper. A depthwise-separable CNN with 7,722 parameters
(7.5 kB of INT8 weights) detects continuous (CAS) and discontinuous (DAS) adventitious sounds in paediatric recordings.
It is trained on the 2022 and 2023 releases of SPRSound and tested on the 2024 and 2025 releases, which contain new patients.

## Repository layout

| Path | Contents |
|:--|:--|
| `code/` | All scripts (data indexing, features, training, quantisation, analysis, tables, figures, inference) |
| `results/split.csv` | Patient-level split of the development cohort (training / validation) and the external cohorts |
| `results/*.csv` | Result files behind every table and figure |
| `results/*.pkl` | Window-level predictions of all networks and seeds (lets tables and figures be rebuilt without retraining) |
| `results/models/` | Trained PyTorch weights (`*.pt`), three seeds per network |
| `results/onnx/` | Tiny + KD network in FP32 and static INT8 ONNX format, three seeds |
| `results/*.log` | Training and analysis logs |

The SPRSound audio is **not** redistributed here; download it from the official repository (below).

## Installation

Python 3.13 on CPU was used. Other recent versions are likely to work.

```bash
git clone https://github.com/sshim76/tiny-adventitious-sprsound
cd tiny-adventitious-sprsound
pip install -r requirements.txt
```

## Quick start: rebuild tables and figures from the included results (no data, no training)

```bash
python3 -I code/make_tables.py     # -> tables/*.md   (Tables 1-8 and appendix tables)
python3 -I code/make_figures.py    # -> figures/*.png, *.tif (Figures 1-5; Figure 6 also needs the audio)
```

## Score your own recording

```bash
python3 -I code/predict.py recording.wav
```

The script resamples to 8 kHz, cuts 4-s windows, runs the three INT8 Tiny + KD models and prints the probability
that CAS and DAS are present in the recording. It is research code, not a medical device.

## Full reproduction

### 1. Data

```bash
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound data/src/sprsound
cd data/src/sprsound && git checkout bca1e51 && cd ../../..     # CC BY 4.0
```

Only the folders `BioCAS2022`, `BioCAS2023`, `BioCAS2024`, `BioCAS2025` and `Patient Summary` are used.

### 2. Main analysis (CPU only; about 4 h on 2 cores)

```bash
python3 -I code/build_index.py      # results/index.csv
python3 -I code/features.py         # data/cache/logmel.npz, results/acoustic_features.csv
python3 -I code/run_dl.py           # Tiny, Reference, Tiny + KD and Small networks, 3 seeds -> results/dl_preds.pkl
python3 -I code/efficiency.py       # ONNX FP32 / INT8 export, INT8 predictions
python3 -I code/analyze.py          # discrimination, paired tests, event types, noise, diagnosis
python3 -I code/run_noise_aug.py    # noise-augmented Tiny + KD; then run analyze.py again
```

### 3. Additional experiments

```bash
python3 -I code/run_extra.py        # rolling-origin retraining, within-2025 cross-validation, few-patient adaptation, KD ablation
python3 -I code/analyze_extra.py    # pooling rules, window-level AUROC, subgroups, calibration, annotation structure, duplicates
python3 -I code/run_benchmark.py    # BioCAS 2022 challenge protocol (Task 1-1, event level)
python3 -I code/analyze_bench.py    # challenge scores -> results/x_benchmark.csv
python3 -I code/latency.py          # latency on an idle machine (ONNX Runtime, one thread)
```

### 4. Tables and figures

```bash
python3 -I code/make_tables.py
python3 -I code/make_figures.py
```

## Where each result comes from

| Paper item | Result file(s) | Produced by |
|:--|:--|:--|
| Table 1 (cohorts) | `split.csv`, `x_table1_values.csv` | `run_dl.py`, `analyze_extra.py` |
| Table 2 (size, memory, latency) | `efficiency.csv`, `efficiency_v2.csv`, `x_memory_frontend.csv` | `efficiency.py`, `latency.py` |
| Tables 3-5 (AUROC, paired differences, operating points) | `discrimination.csv`, `paired.csv` | `analyze.py` |
| Table 6 (challenge protocol) | `x_benchmark.csv` | `run_benchmark.py`, `analyze_bench.py` |
| Table 7 (annotation practice) | `x_annotation_structure.csv` | `analyze_extra.py` |
| Table 8 and Appendix Table A9 (2025 release) | `x_rolling.csv`, `x_cv2025.csv`, `cv2025_folds.csv`, `x_adaptation_summary.csv` | `run_extra.py`, `analyze_extra.py` |
| Appendix tables | `event_type_sensitivity.csv`, `noise.csv`, `diagnosis_exploratory.csv`, `x_kd_ablation.csv`, `x_aggregation.csv`, `x_window_level.csv`, `x_subgroups.csv`, `x_calibration.csv` | `analyze.py`, `analyze_extra.py` |
| Figures 1-6 | as above, plus `dl_preds.pkl`, `noise.csv` | `make_figures.py` |

## Licence and data attribution

The code is released under the MIT licence (see `LICENSE`).
SPRSound is distributed by its creators under CC BY 4.0. Files derived from it (`index.csv`, `split.csv`, `events.pkl`,
predictions and trained weights) are shared under the same terms; please cite the SPRSound papers when you use them.

## Citation

If you use this code, please cite the paper (details will be added on publication) and this repository (`CITATION.cff`).

## Contact

Seong-O Shim, University of Jeddah: sshim@uj.edu.sa
