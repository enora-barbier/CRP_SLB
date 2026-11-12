# North Sea Dataset

## Description

The North Sea benchmark is the primary dataset used throughout this project. It is derived
from public Norwegian Petroleum Directorate (NPD) well-log data and contains approximately
1,765 wells with multiple log modalities and six geological formation tops.

The task is **formation-top detection**: given a set of well-log curves recorded as a
function of depth, predict the depths at which key geological formations begin.

The six target markers follow a fixed stratigraphic order:

```text
HORDALAND → ROGALAND → SHETLAND → VIKING → VESTLAND → DUNLIN
```

Unlike standard time-series classification, this is a sparse event-detection problem. Most
depth samples correspond to background geology, while only a handful of depths correspond
to formation tops.

The dataset is intentionally challenging due to:

- Large geological variability between wells
- Missing log channels
- Missing marker annotations
- Similar geological patterns occurring at very different depths

As a result, the North Sea benchmark serves as the main industrial-style evaluation
dataset for this project.

## Data Access

The raw NPD-derived well logs are **not** committed (they are `.gitignore`d) and are not
redistributed here. The committed material consists of the prepared train/validation/test
splits (`warm_state*.json`, well-list JSONs, split manifests) plus the result logs and
metrics; the per-well `.npz` / `.parquet` tensors are regenerated locally by the
preprocessing and data-split notebooks. Contact the project maintainer for access to the
raw NPD export, then re-run `notebooks/01_preprocessing.ipynb` followed by the
`Classification/combined_data_split.ipynb` (and the Transformer data-split cell) to rebuild
the per-well arrays.

## Environment setup

From the repository root, install the local `geoxts` package (used by the classification
and transformer pipelines) in editable mode:

```
pip install -e .
```

## Repository Structure

```text
north_sea/
├── README.md                 <- this file
├── notebooks/
│   └── 01_preprocessing.ipynb            <- initial data cleaning / preprocessing
├── Classification/                       <- GeoXTS sliding-window baseline
│   ├── combined_data_split.ipynb
│   ├── experiment_training.ipynb
│   ├── experiment_log__shard*.csv
│   ├── sweep_results_combined.csv
│   ├── depth/                            <- splits + warm state WITH depth feature
│   └── no_depth/                         <- splits + warm state WITHOUT depth feature
├── GNN/
│   └── GNN_markers_north_sea.ipynb       <- GraphSAGE formation-top segmentation
└── Transformer/                          <- seq2seq transformer pipeline
    ├── seq2seq-transformer-v2.litho_aligned.ipynb
    ├── src/                              <- PatchTST backbone, layers, RevIN
    │   ├── PatchTST_backbone.py
    │   ├── PatchTST_layers.py
    │   ├── RevIN.py
    │   └── __init__.py
    ├── data/                             <- predefined splits + NPD metadata
    │   ├── train_wells.json
    │   ├── valid_wells.json
    │   ├── test_wells.json
    │   ├── split_manifest.json
    │   ├── build_summary.json
    │   └── npd_metadata/                 <- DTW output + conditioned-top correlations
    └── metrics/                          <- saved experiment outputs and metrics
        ├── metrics_seq2seq_v2_litho_aligned.json
        └── metrics_npd2_lithoformer_g2.json
```

### Preprocessing

- `notebooks/01_preprocessing.ipynb` – initial data cleaning and preprocessing workflow.

### Classification

Contains the GeoXTS-based sliding-window classification pipeline used as the main baseline.

**Key files**

- `combined_data_split.ipynb` – dataset construction and train/validation/test splits.
- `experiment_training.ipynb` – training and hyperparameter sweep notebook.
- `experiment_log__shard*.csv` – distributed experiment logs (one per machine/shard).
- `sweep_results_combined.csv` – aggregated benchmark results.
- `depth/`, `no_depth/` – the two feature regimes (with vs. without an explicit depth
  channel); each holds its own `train/val/test.npz`, `warm_state*.json`, and warm
  tops/parquet files.

### GNN

Contains the Graph Neural Network experiments.

**Key files**

- `GNN_markers_north_sea.ipynb` – GraphSAGE-based formation-top segmentation pipeline.

### Transformer

Contains the sequence-to-sequence transformer implementation developed during the CRP.

**Key files**

- `seq2seq-transformer-v2.litho_aligned.ipynb` – training and evaluation notebook
  (with CodeCarbon / CarbonTracker energy tracking).
- `src/` – PatchTST implementation, attention/embedding layers, and RevIN utilities.
- `data/` – predefined train/validation/test splits, the split manifest, and NPD metadata
  (DTW alignment output and conditioned-top correlations).
- `metrics/` – saved experiment outputs and evaluation metrics.

The notebook assumes Jupyter sets the working directory to the `Transformer/` folder: it
imports the backbone from the adjacent `src/` folder and reads the splits from the adjacent
`data/` folder.

## Key Variables

- **Inputs:** multiple well-log curves (e.g. GR and other modalities) recorded vs. depth;
  optionally an explicit depth channel (`Classification/depth` vs. `no_depth`).
- **Target:** six ordered formation tops
  (`HORDALAND`, `ROGALAND`, `SHETLAND`, `VIKING`, `VESTLAND`, `DUNLIN`).
- **Primary metric:** `recall_tops` (fraction of true formation tops detected), with a
  median depth-error secondary metric.

## Note on the Complex Transformer

The repository includes the transformer implementation developed as part of this project
("simple transformer").

The higher-performing **complex transformer** referenced in the report is **not included in
this repository**. That model was developed independently by the project supervisors as part
of ongoing research and has not yet been publicly released. The project benchmarked that
model under the same evaluation framework, but only the student-developed implementation is
distributed here.

## References

- Salimath, Bugiotti & Wlodarczyk (2025). *GeoTS: A TSC Framework for Estimating Geological Formation to Model Carbon Storage Reservoirs.* KDD '25, 4773–4783. doi:10.1145/3711896.3737228
- Wang, Yan & Oates (2017). *Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline (FCN).* IJCNN, 1578–1585.
- Ismail Fawaz et al. (2020). *InceptionTime: Finding AlexNet for Time Series Classification.* Data Mining and Knowledge Discovery 34(6), 1936–1962.
- Fauvel et al. (2021). *XCM: An Explainable Convolutional Neural Network for Multivariate Time Series Classification.* Mathematics 9(23), 3137.
- Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 30.
- Nie et al. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).* ICLR.
- Kim et al. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift (RevIN).* ICLR.
- Cui et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR, 9268–9277.
- Anthony, Kanding & Selvan (2020). *Carbontracker: Tracking and Predicting the Carbon Footprint of Training Deep Learning Models.* ICML Workshop on Challenges in Deploying and Monitoring ML Systems.
