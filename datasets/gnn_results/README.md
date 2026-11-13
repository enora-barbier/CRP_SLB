# GNN carbon / recall results (Colorado, North Sea, UCI HAR)

This folder documents how the GNN rows of the benchmark tables were produced on a
Google Cloud VM (n1-standard-8, 1x Tesla T4, `europe-west4`; Python 3.10, PyTorch + DGL,
CodeCarbon with the country pinned to Norway), and rebuilds the summary workbook from the raw
result CSVs committed next to each dataset.

Two GNN architectures were run per dataset, `GraphSAGE` and `ImprovedGNN`, for **15 seeds
(0-14) each**, 15 epochs (HAR additionally at 30). Stage 2 is an Isolation Forest only (the VGAE
variant is not part of these runs).

## Which script produced which result

| Dataset | Training script (as run on GCP) | Recall script | Result CSVs |
|---|---|---|---|
| Colorado | `datasets/colorado/gnn_stage2_anomaly_colorado_local.py` | `datasets/colorado/recover_boundary_recall_colorado.py` | `datasets/colorado/results/` |
| North Sea | `datasets/north_sea/GNN/gnn_stage2_anomaly.py` | `datasets/north_sea/GNN/recover_boundary_recall.py` | `datasets/north_sea/GNN/results/` |
| UCI HAR | `datasets/HAR/gnn_stage2_anomaly_har.py` | `datasets/HAR/recover_transition_recall_har.py` | `datasets/HAR/results/` |

**Filename caveats - please read before reusing these scripts:**

* **Colorado:** the script that produced the results is the `*_local.py` file (dataset
  caching, GPU placement, `stage1_recall`/`if_recall`, ImprovedGNN, per-arch aggregation). The
  un-suffixed `gnn_stage2_anomaly_colorado.py` is an older, independently developed version
  (no caching fix, no recall columns, no ImprovedGNN) and was **not** used for these numbers.
* **North Sea:** on the VM the script was deployed under the name `gnn_stage2_anomaly_shetland.py`,
  but its content is `datasets/north_sea/GNN/gnn_stage2_anomaly.py`. The repo file called
  `gnn_stage2_anomaly_shetland.py` is a different, older script and was **not** used.
* **HAR:** the 30-epoch run used the same script with only the epoch constant changed. That constant
  is now the `EPOCHS` environment variable (default 15), so no code edit is needed to reproduce it.

Changes relative to the earlier scripts that matter for the numbers: the graph datasets are cached after
first build (previously every well was rebuilt from pandas on every access, which dominated runtime);
models and graphs are moved to the GPU explicitly; and `stage1.to(DEVICE)` was added to `fuse_well()`
(North Sea) so the post-loop diagnostic plot does not hit a CPU/GPU device mismatch.

## How the runs were launched

Run from a directory that contains the dataset files; `CKPT_DIR` receives checkpoints and the carbon log
(checkpoints and `.pkl`/`.npz` are git-ignored and not committed). On the GCP VM, `source ~/.bashrc` first
so `LD_LIBRARY_PATH` finds the CUDA libraries used by torch/dgl.

```bash
# Colorado  (needs logs.parquet, tops.parquet, logs_50.parquet, tops_50.csv in the cwd)
CKPT_DIR=/tmp/colorado_stage2_ckpt_v2 python3 -u gnn_stage2_anomaly_colorado_local.py
CKPT_DIR=/tmp/colorado_stage2_ckpt_v2 python3 recover_boundary_recall_colorado.py

# North Sea (needs log_train/log_valid.parquet and tops_train/tops_valid.parquet in the cwd)
CKPT_DIR=/home/jupyter/stage2_ckpt python3 -u gnn_stage2_anomaly.py                                  # GraphSAGE + ImprovedGNN
CKPT_DIR=/home/jupyter/stage2_ckpt ARCHS_OVERRIDE="ImprovedGNN" python3 -u gnn_stage2_anomaly.py     # re-run one arch only
CKPT_DIR=/home/jupyter/stage2_ckpt python3 recover_boundary_recall.py

# UCI HAR (needs warm_state.json, train.npz, val.npz, test.npz in DATA_DIR, built by
# datasets/medical/Classification/combined_data_split_medical.ipynb)
CKPT_DIR=/home/jupyter/har_stage2_ckpt_v2   python3 -u gnn_stage2_anomaly_har.py                     # 15 epochs
EPOCHS=30 CKPT_DIR=/home/jupyter/har_stage2_ckpt_30ep python3 -u gnn_stage2_anomaly_har.py           # 30 epochs
CKPT_DIR=/home/jupyter/har_stage2_ckpt_v2   python3 recover_transition_recall_har.py
CKPT_DIR=/home/jupyter/har_stage2_ckpt_30ep python3 recover_transition_recall_har.py
```

Other knobs: `N_REPEATS`, `BASE_SEED`, `ARCHS_OVERRIDE`, `OUT_DIR`, `ELECTRICITYMAPS_API_KEY`. Use a **fresh
`CKPT_DIR` per run**: checkpoint filenames are deterministic per `(arch, seed)` and are overwritten, so a
re-run into an old folder silently replaces earlier checkpoints.

The recall scripts reload the saved checkpoints; no retraining is needed to (re)compute recall.

## Result files

| File | Contents |
|---|---|
| `datasets/colorado/results/colorado_stage2_carbon_log.csv` | per-seed carbon/energy/timing/quality log (30 rows) |
| `datasets/colorado/results/colorado_stage2_boundary_recall_recovery.csv` | per-seed Recall@50ft (30 rows) |
| `datasets/north_sea/GNN/results/stage2_carbon_log.csv` | per-seed log, **33 rows** (see duplicate note) |
| `datasets/north_sea/GNN/results/stage2_boundary_recall_recovery.csv` | per-seed Recall@50ft (30 rows) |
| `datasets/HAR/results/har_stage2_carbon_log_{15,30}ep.csv` | per-seed log per epoch setting (30 rows each) |
| `datasets/HAR/results/har_stage2_transition_recall_recovery_{15,30}ep.csv` | per-seed transition recall (30 rows each) |

All CSVs are the raw files from the VM, unmodified.

**North Sea duplicate rows:** GraphSAGE seeds 5, 6 and 7 each appear twice in `stage2_carbon_log.csv` (an
earlier overlapping resume run plus the full run). The later-timestamped row is the trustworthy one; the
workbook script drops the earlier three.

## Rebuilding the workbook

```bash
python3 datasets/gnn_results/build_results_workbook.py
```

writes `GNN_carbon_recall_deliverable.xlsx` here: a `Paper_Table` sheet (mean +/- std over 15 seeds, plus a
Notes block) and per-seed / summary sheets for Colorado, North Sea, HAR 15-epoch and HAR 30-epoch.

## Metric definitions and comparability

* **Recall, Colorado / North Sea** - Recall@Tolerance(+/-50ft): a true formation boundary counts as found if a
  predicted boundary in the same validation well lies within 50 ft; pooled over wells. Boundaries must be
  stable for at least 15 ft on both sides, so single-sample classification flips are not counted.
* **Recall, UCI HAR** - `transition_recall` as in `evaluate_medical()` of `experiment_training_medical.ipynb`:
  accuracy on test windows whose true label is a transition onset. This is a **different metric** from the
  well-log recall and is only comparable to the Sliding-family HAR rows, not to the Seq2Seq transition recall.
* **Inference time** - `s_per_well = final_eval_time_s / n_eval_wells` (1 sample = 1 well, the convention of the
  paper's other well-log rows). `ms_per_sample` in the CSVs divides by depth-rows instead and must not be
  compared with the paper's inference-time column. It also still includes the whole per-well pipeline
  (cleaning, graph build, forward pass, IsolationForest scoring) rather than an isolated model forward.
* **Energy scope** - `cc_energy_kwh` is **training only** (Stage 1 training + Isolation Forest fit).
  Evaluation energy is separate (`cc_eval_energy_kwh`: 16-point threshold search + final pass), and
  `cc_total_emissions_g = (cc_energy_kwh + cc_eval_energy_kwh) x 30.08 g/kWh`. Evaluation is about 90% of the
  total GNN energy on Colorado, about 65-70% on North Sea and only 2-4% on HAR. When comparing against other
  models, use the energy column whose scope matches theirs; the workbook reports both.
* **Epochs** - the GNN uses 15 epochs. Training F1 (not validation F1) shows strongly diminishing returns by
  roughly epoch 10-12 on both well-log datasets; the exception is North Sea ImprovedGNN, which is still
  creeping up slightly at epoch 15. The Seq2Seq transformers in the paper train for 30 epochs, so energy
  comparisons against them are partly a training-budget comparison. On HAR, 30 epochs helped GraphSAGE
  (0.691 -> 0.790) but hurt ImprovedGNN (0.792 -> 0.735).
