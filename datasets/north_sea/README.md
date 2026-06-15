# North Sea Dataset

## Description

The North Sea benchmark is the primary dataset used throughout this project. It is derived from public Norwegian Petroleum Directorate (NPD) well-log data and contains approximately 1,765 wells with multiple log modalities and six geological formation tops.

The task is **formation-top detection**: given a set of well-log curves recorded as a function of depth, predict the depths at which key geological formations begin.

The six target markers follow a fixed stratigraphic order:

```text
HORDALAND → ROGALAND → SHETLAND → VIKING → VESTLAND → DUNLIN
```

Unlike standard time-series classification, this is a sparse event-detection problem. Most depth samples correspond to background geology, while only a handful of depths correspond to formation tops.

The dataset is intentionally challenging due to:

- Large geological variability between wells
- Missing log channels
- Missing marker annotations
- Similar geological patterns occurring at very different depths

As a result, the North Sea benchmark serves as the main industrial-style evaluation dataset for this project.

## Repository Structure

```text
north_sea/
├── Classification/
├── GNN/
├── Transformer/
└── notebooks/
```

### Classification

Contains the GeoXTS-based sliding-window classification pipeline used as the main baseline.

**Key files**

- `combined_data_split.ipynb` – dataset construction and train/validation/test splits
- `experiment_training.ipynb` – training and hyperparameter sweep notebook
- `experiment_log__shard*.csv` – distributed experiment logs
- `sweep_results_combined.csv` – aggregated benchmark results

### GNN

Contains the Graph Neural Network experiments.

**Key files**

- `GNN_markers_north_sea.ipynb` – GraphSAGE-based formation-top segmentation pipeline

### Transformer

Contains the sequence-to-sequence transformer implementation developed during the CRP.

**Key files**

- `seq2seq-transformer-v2.litho_aligned.ipynb` – training and evaluation notebook
- `src/` – PatchTST implementation, layers, and RevIN utilities
- `data/` – predefined train/validation/test splits and metadata
- `metrics/` – saved experiment outputs and evaluation metrics

### Preprocessing

- `notebooks/01_preprocessing.ipynb` – initial data cleaning and preprocessing workflow

## Note on the Complex Transformer

The repository includes the transformer implementation developed as part of this project ("simple transformer").

The higher-performing **complex transformer** referenced in the report is **not included in this repository**. That model was developed independently by the project supervisors as part of ongoing research and has not yet been publicly released. The project benchmarked that model under the same evaluation framework, but only the student-developed implementation is distributed here.
