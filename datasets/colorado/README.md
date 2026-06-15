# Colorado Well Logs Dataset

## Description

Geological well-log dataset from Colorado, used as a secondary **formation-top
detection** benchmark alongside the North Sea data. Each well provides a gamma-ray (GR)
log recorded as a function of depth, together with the well location (latitude,
longitude). Three formation tops are annotated, in increasing-depth (stratigraphic)
order:

```text
NIOBARA → CODELL → FORT_HAYS
```

As with the North Sea benchmark, this is a sparse event-detection problem: most depth
samples are background geology and only a handful of depths correspond to formation tops.
The Colorado data is smaller and geologically distinct from the North Sea, so it is used
to check how well the same pipelines transfer to a different basin.

## Data Access

The raw data is **not** committed (it is `.gitignore`d) — it is not publicly distributed.
Contact the project maintainer for access, then place it under this `datasets/colorado/`
folder so the notebooks can find it.

The notebooks expect the parquet/CSV files in a local `data/` folder with this layout:

```text
datasets/colorado/data/
├── training/
│   ├── logs.parquet      # well-log measurements (GR, DEPTH)
│   ├── loc.parquet       # well locations (lat, lon)
│   ├── tops.parquet      # formation tops
│   └── training_well_list.npy
└── Validation/
    ├── logs_100.parquet  # 100-well validation/test set
    ├── loc_100.parquet
    └── tops_100.csv
```

The `notebooks/` read these paths relative to the repository root
(`datasets/colorado/data/...`), so run them from the repo root (or adjust the paths at
the top of the notebook if you keep the data elsewhere).

## Environment setup

From the repository root, install the local `geoxts` package (used by the classification
and benchmark notebooks) in editable mode:

```
pip install -e .
```

## File Structure

```
datasets/colorado/
├── README.md                 <- this file
├── notebooks/                <- preprocessing + benchmark + marker-propagation notebooks
│   ├── 01_preprocessing.ipynb
│   ├── 02_benchmark.ipynb
│   ├── 03_marker_propagation.ipynb
│   ├── 04_industrial_baseline.ipynb
│   └── 05_tutorial_marker_propagation.ipynb
├── Transformer/              <- seq2seq transformer pipeline
│   ├── seq2seq-transformer-v2.colorado_aligned.ipynb
│   ├── train.py              <- PatchTST well-log segmentation pipeline
│   ├── src/                  <- PatchTST backbone, layers, RevIN
│   │   ├── PatchTST_backbone.py
│   │   ├── PatchTST_layers.py
│   │   └── RevIN.py
│   ├── data/                 <- predefined train/valid/test well splits
│   │   ├── train_wells.json
│   │   ├── valid_wells.json
│   │   └── test_wells.json
│   └── metrics/              <- saved experiment outputs and evaluation metrics
│       ├── metrics_seq2seq_v2_colorado.json
│       └── metrics_npd2_lithoformer_g2_colorado.json
└── data/                     <- raw parquet/CSV data (gitignored, see Data Access)
```

## Notebooks

All in `notebooks/`. Run in this order:

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_preprocessing.ipynb` | ETL and data quality control; reads `data/training/` + `data/Validation/`. |
| 02 | `02_benchmark.ipynb` | Deep-learning model benchmark (FCN, LSTM, InceptionTime, XCM variants). |
| 03 | `03_marker_propagation.ipynb` | Geological marker propagation using DTW. |
| 04 | `04_industrial_baseline.ipynb` | DTW-based industrial baseline comparison. |
| 05 | `05_tutorial_marker_propagation.ipynb` | Step-by-step marker-propagation tutorial. |

## Key Variables

- **Inputs:** `GR`, `Depth`, `Latitude`, `Longitude`.
- **Target:** formation tops (`NIOBARA`, `CODELL`, `FORT_HAYS`).
- **Window size:** 201 depth samples.

## Transformer Pipeline

The `Transformer/` folder contains the same sequence-to-sequence PatchTST implementation
used for the North Sea, aligned to the Colorado formation tops. It treats top detection
as a dense, per-depth segmentation problem: every depth sample is assigned a zone label,
and transition (top) locations are extracted from the predicted label sequence.

| File | Description |
|------|-------------|
| `seq2seq-transformer-v2.colorado_aligned.ipynb` | Trains and evaluates the PatchTST seq2seq transformer on the Colorado splits, with CodeCarbon / CarbonTracker energy tracking. |
| `train.py` | Standalone PatchTST well-log segmentation pipeline (dataset, model, training loop) imported/reused by the notebook. |
| `src/` | PatchTST backbone, attention/embedding layers, and RevIN normalization. |
| `data/` | Predefined `train_wells.json` / `valid_wells.json` / `test_wells.json` splits. |
| `metrics/` | Saved JSON metrics from the seq2seq and Lithoformer runs. |

The notebook assumes Jupyter sets the working directory to the `Transformer/` folder: it
imports the backbone from the adjacent `src/` folder and reads the parquet data from a
`data/` folder next to the notebook. A Kaggle code path is also kept for portability.

## Generated outputs (gitignored)

Saved model weights, carbon logs, and any intermediate `*.npz` / `*.pth` / `*.parquet`
produced by the notebooks. The small JSON metric files in `Transformer/metrics/` are
committed so results can be compared.

## References

- Salimath, Bugiotti & Wlodarczyk (2025). *GeoTS: A TSC Framework for Estimating Geological Formation to Model Carbon Storage Reservoirs.* KDD '25, 4773–4783. doi:10.1145/3711896.3737228
- Sakoe & Chiba (1978). *Dynamic Programming Algorithm Optimization for Spoken Word Recognition (DTW).* IEEE Trans. Acoustics, Speech, and Signal Processing 26(1), 43–49.
- Nilo et al. (2023). *Semi-Automatic Well-Log Correlation Using the Dynamic Time Warping Algorithm.* 18th Int. Congress of the Brazilian Geophysical Society & EXPOGEf.
- Wang, Yan & Oates (2017). *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline (FCN).* IJCNN, 1578–1585.
- Ismail Fawaz et al. (2020). *InceptionTime: Finding AlexNet for Time Series Classification.* Data Mining and Knowledge Discovery 34(6), 1936–1962.
- Fauvel et al. (2021). *XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification.* Mathematics 9(23), 3137.
- Nie et al. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).* ICLR.
