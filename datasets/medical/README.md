# Medical Dataset: UCI HAR (Human Activity Recognition Using Smartphones)

## Description

30 subjects performed 6 activities (walking, walking upstairs, walking downstairs,
sitting, standing, laying) while wearing a waist-mounted smartphone. The 9 inertial-sensor
channels are windowed into 128-timestep segments (50 Hz, 2.56 s windows, 50% overlap).

In this project the dataset is used as a **transition-centred** detection task (mirroring
the North Sea formation-top pipeline), *not* independent window classification: the models
detect the activity being **entered** at each change point, with a `None` class for stable
regions. Class scheme (7 classes): `0=None, 1=WALKING, 2=WALKING_UPSTAIRS,
3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING`.

## Data Access

The raw data is **not** included in this repository. Download and unzip it once into this
`datasets/medical/` folder.

1. Download from the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
   The **Download** button gives `human+activity+recognition+using+smartphones.zip`.

2. **Note the nested zip.** The downloaded archive contains a *second* zip plus a `.names`
   file:
   ```
   human+activity+recognition+using+smartphones.zip
   ├── UCI HAR Dataset.zip      <- the actual data, unzip THIS one
   └── UCI HAR Dataset.names
   ```
   Extract the outer archive, then extract the inner `UCI HAR Dataset.zip` **into this
   `datasets/medical/` folder**. The folder name must be exactly `UCI HAR Dataset`
   (with the spaces).

   > A `__MACOSX/` folder or a stray `UCI HAR Dataset.names` left alongside can be
   > deleted. Neither is used.

The notebooks find the data automatically: they locate the **repository root** (by walking
up for `setup.py` + `geoxts/`) and read `datasets/medical/UCI HAR Dataset`. To use data
stored elsewhere (for example on a VM where it lives outside the repo), set the
`UCI_HAR_DIR` environment variable to its path. It takes priority over everything else:

```
export UCI_HAR_DIR="/path/to/UCI HAR Dataset"     # bash/VM
# or inside the notebook, before the path cell:
# import os; os.environ["UCI_HAR_DIR"] = "/path/to/UCI HAR Dataset"
```

Plain relative fallbacks (`../UCI HAR Dataset`, `UCI HAR Dataset`,
`datasets/medical/UCI HAR Dataset`) are also tried last, so a standalone notebook beside
the data still works.

**Run each notebook from its own folder.** Finding the raw data works from anywhere in the
repo, but the outputs (`har_transition/`, `har_seq/`, model checkpoints, logs) are written
relative to the working directory, and the later notebooks expect to find them there.
The Transformer data split must be run from `Transformer/src/`.

## Environment setup

From the repository root, install the local `geoxts` package (used by the training
notebooks) in editable mode:

```
pip install -e .
```

The Classification notebooks also add the repo root to `sys.path` themselves (walking up
for `setup.py` + `geoxts/`), so this is a convenience rather than a hard requirement.

## File Structure

```
datasets/medical/
├── README.md                 <- this file
├── Classification/           <- transition-centred window classification notebooks
├── Transformer/              <- seq2seq transformer pipeline
│   ├── uci_har_seq2seq.ipynb
│   ├── src/
│   │   ├── seq2seq_transformer_data_split.ipynb
│   │   ├── PatchTST_backbone.py
│   │   ├── PatchTST_layers.py
│   │   └── RevIN.py
│   ├── data/                 <- written by the data split (warm_state.json + .npz files)
│   └── metrics/              <- saved transformer metrics
└── UCI HAR Dataset/          <- unzipped raw data (download it, not in the repo)
    ├── activity_labels.txt
    ├── features.txt
    ├── train/
    │   ├── X_train.txt
    │   ├── y_train.txt
    │   ├── subject_train.txt
    │   └── Inertial Signals/ <- the 9 raw sensor channels
    └── test/
        ├── X_test.txt
        ├── y_test.txt
        ├── subject_test.txt
        └── Inertial Signals/
```

## Classification notebooks

All live in [`Classification/`](Classification/). Run in this order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [`combined_data_split_medical.ipynb`](Classification/combined_data_split_medical.ipynb) | Reads the raw UCI HAR data, builds the transition-centred dataset, writes `har_transition/` (`train/val/test.npz` + `warm_state.json`). |
| 2 | [`experiment_training_medical.ipynb`](Classification/experiment_training_medical.ipynb) | Loads `har_transition/` and runs the architecture / loss / feature-subset sweep. To split the sweep across machines, set `N_SHARDS` to the number of machines and `MY_SHARD` on each. Appends to `experiment_log_medical__shard{N}.csv`. |
| 3 | [`experiment_carbon_tracking_medical_combined.ipynb`](Classification/experiment_carbon_tracking_medical_combined.ipynb) | Per-model CO₂eq, with training and inference measured together. |
| 3 | [`experiment_carbon_tracking_medical_separate.ipynb`](Classification/experiment_carbon_tracking_medical_separate.ipynb) | Per-model CO₂eq, with training and inference measured separately. |
| (separate) | [`sequence_classification_medical.ipynb`](Classification/sequence_classification_medical.ipynb) | Independent track: reads the raw UCI HAR data directly; FCN + Transformer with a `SEQ_LEN` context-window sweep; writes `har_seq/`. |

## Key Variables

- **Inputs:** 9 inertial channels × 128 timesteps per window: `body_acc_{x,y,z}`,
  `body_gyro_{x,y,z}`, `total_acc_{x,y,z}`.
- **Target:** 7-class transition label (`None` + the 6 activities above).
- **Primary metric:** `transition_recall` (analogue of North Sea `recall_tops`); secondary
  `median_error` (median |pred - true| class distance on true-transition windows).

## Generated outputs

The notebooks write `har_transition/`, `har_seq/`, `models_medical/`,
`models_medical_carbon/` and `carbon_logs_medical/`. These are not included in the
repository; run the notebooks to regenerate them. The small `experiment_log_*.csv` result
logs can be committed if you want to share results.

## Transformer Pipeline

The [`Transformer/`](Transformer/) folder contains the sequence-to-sequence transformer
implementation used to evaluate whether a modern time-series transformer can solve the
transition-detection task directly, without sliding-window classification.

| File | Description |
|--------|-------------|
| [`src/seq2seq_transformer_data_split.ipynb`](Transformer/src/seq2seq_transformer_data_split.ipynb) | Converts the raw UCI HAR data into subject-level train/validation/test splits and writes the sequence-to-sequence datasets to `Transformer/data/`. Run it first, from `Transformer/src/`. |
| [`uci_har_seq2seq.ipynb`](Transformer/uci_har_seq2seq.ipynb) | Trains and evaluates a PatchTST-based sequence-to-sequence transformer on those splits. Produces dense activity predictions and transition-detection metrics analogous to the North Sea formation-top task. |

The transformer treats activity recognition as a dense sequence-labelling problem rather
than independent window classification. Each subject's recording is one sequence, and every
window position gets one of the six activity labels. Transitions are taken from the points
where the label changes (plus a per-activity transition target), and evaluated using the
same methodology developed for geological formation-top detection.

## References

- UCI HAR dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- Anguita et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones.* ESANN.
- Nie et al. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).* ICLR.
- Kim et al. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift (RevIN).* ICLR.
- Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR, 9268–9277.
- Anthony, Kanding & Selvan (2020). *Carbontracker: Tracking and Predicting the Carbon Footprint of Training Deep Learning Models.* ICML Workshop on Challenges in Deploying and Monitoring ML Systems.
