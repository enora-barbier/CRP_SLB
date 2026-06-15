# Medical Dataset — UCI HAR (Human Activity Recognition Using Smartphones)

## Description

30 subjects performed 6 activities (walking, walking upstairs, walking downstairs,
sitting, standing, laying) while wearing a waist-mounted smartphone. The 9 inertial-sensor
channels are windowed into 128-timestep segments (50 Hz, 2.56 s windows, 50 % overlap).

In this project the dataset is used as a **transition-centred** detection task (mirroring
the North Sea formation-top pipeline), *not* independent window classification: the models
detect the activity being **entered** at each change point, with a `None` class for stable
regions. Class scheme (7 classes): `0=None, 1=WALKING, 2=WALKING_UPSTAIRS,
3=WALKING_DOWNSTAIRS, 4=SITTING, 5=STANDING, 6=LAYING`.

## Data Access

The raw data is **not** committed (it is `.gitignore`d) — download and unzip it once into
this `datasets/medical/` folder.

1. Download from the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
   The **Download** button gives `human+activity+recognition+using+smartphones.zip`.

2. **Note the nested zip.** The downloaded archive contains a *second* zip plus a `.names`
   file:
   ```
   human+activity+recognition+using+smartphones.zip
   ├── UCI HAR Dataset.zip      <- the actual data — unzip THIS one
   └── UCI HAR Dataset.names
   ```
   Extract the outer archive, then extract the inner `UCI HAR Dataset.zip` **into this
   `datasets/medical/` folder**. The folder name must be exactly `UCI HAR Dataset`
   (with the spaces).

   > A `__MACOSX/` folder or a stray `UCI HAR Dataset.names` left alongside can be
   > ignored or deleted — both are unused and already `.gitignore`d.

The notebooks find the data automatically: they anchor to the **repository root** (located
by walking up for `setup.py` + `geoxts/`) and read `datasets/medical/UCI HAR Dataset`, so
they work from **any** working directory inside the repo. To use data stored elsewhere
(for example on a VM where it lives outside the repo), set the `UCI_HAR_DIR` environment
variable to its path — it takes priority over everything else:

```
export UCI_HAR_DIR="/path/to/UCI HAR Dataset"     # bash/VM
# or inside the notebook, before the path cell:
# import os; os.environ["UCI_HAR_DIR"] = "/path/to/UCI HAR Dataset"
```

Plain CWD-relative fallbacks (`../UCI HAR Dataset`, `UCI HAR Dataset`,
`datasets/medical/UCI HAR Dataset`) are also tried last, so a standalone notebook beside
the data still works.

## Environment setup

From the repository root, install the local `geoxts` package (used by the training
notebook) in editable mode:

```
pip install -e .
```

The notebooks also self-bootstrap `geoxts` onto `sys.path` (walking up to the repo root
for `setup.py` + `geoxts/`), so this is a convenience rather than a hard requirement when
running from `Classification/`.

## File Structure

```
datasets/medical/
├── README.md                 <- this file
├── Classification/           <- sliding-window classification notebooks
├── Transformer/              <- seq2seq transformer pipeline
│   ├── seq2seq_transformer_data_split.ipynb
│   └── uci_har_seq2seq.ipynb
└── UCI HAR Dataset/          <- unzipped raw data
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

## Notebooks

All live in `Classification/`. Run in this order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `combined_data_split_medical.ipynb` | Reads `../UCI HAR Dataset/`, builds the transition-centred dataset, writes `har_transition/` (`train/val/test.npz` + `warm_state.json`). |
| 2 | `experiment_training_medical.ipynb` | Cold-loads `har_transition/`; architecture / loss / feature-subset sweep. Set `MY_SHARD` (0–3) per machine; appends to `experiment_log_medical__shard{N}.csv`. |
| 3 | `experiment_carbon_tracking_medical_combined.ipynb` | Per-model CO₂eq (training + inference) over `har_transition/`. |
| 3 | `experiment_carbon_tracking_medical_separate.ipynb` | Per-model CO₂eq (training only) over `har_transition/`. |
| — | `sequence_classification_medical.ipynb` | Independent track: reads raw `../UCI HAR Dataset/` directly; FCN + Transformer with a `SEQ_LEN` context-window sweep; writes `har_seq/`. |

## Key Variables

- **Inputs:** 9 inertial channels × 128 timesteps per window — `body_acc_{x,y,z}`,
  `body_gyro_{x,y,z}`, `total_acc_{x,y,z}`.
- **Target:** 7-class transition label (`None` + the 6 activities above).
- **Primary metric:** `transition_recall` (analogue of North Sea `recall_tops`); secondary
  `median_error` (median |pred − true| class distance on true-transition windows).

## Generated outputs (all `.gitignore`d)

`har_transition/`, `har_seq/`, `models_medical/`, `models_medical_carbon/`,
`carbon_logs_medical/`, and any `*.npz` / `*.pth`. The small `experiment_log_*.csv` result
logs are not ignored and may be committed if you want to share results.

## Transformer Pipeline

The `Transformer/` folder contains the sequence-to-sequence transformer implementation
used to evaluate whether a modern time-series transformer can solve the transition-detection
task directly, without sliding-window classification.

| File | Description |
|--------|-------------|
| `seq2seq_transformer_data_split.ipynb` | Converts the raw UCI HAR data into subject-level train/validation/test splits and generates the sequence-to-sequence datasets used by the transformer. |
| `uci_har_seq2seq.ipynb` | Trains and evaluates a PatchTST-based sequence-to-sequence transformer on the generated splits. Produces dense activity predictions and transition-detection metrics analogous to the North Sea formation-top task. |

The transformer treats activity recognition as a dense sequence-labelling problem rather
than independent window classification. Each timestep is assigned either a background
(`None`) label or one of the six activities, allowing transition locations to be extracted
and evaluated using the same methodology developed for geological formation-top detection.

## References

- UCI HAR dataset: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
