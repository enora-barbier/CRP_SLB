# Corporate Research Project (CRP) - SLB

Corporate Research Project (CRP) carried out in collaboration with **SLB**, on
**geological formation-top detection** from well logs. Given a set of well-log curves
recorded against depth, the goal is to predict the depths at which key geological
formations begin — a sparse event-detection problem where most depth samples are
background geology and only a handful mark a formation top.

The project compares several families of approaches under one evaluation framework, and
tracks the **energy and carbon cost** of training/inference for each:

- **Sliding-window time-series classification** (FCN, LSTM, InceptionTime, XCM) — the main
  baseline, built on the external **GeoXTS** library.
- **Sequence-to-sequence transformers** (PatchTST + RevIN) that label every depth densely.
- **Graph neural networks** (GraphSAGE) for formation-top segmentation.
- **Dynamic Time Warping** baselines for marker propagation / well-log correlation.

Three datasets are studied: **North Sea** (NPD well logs — the primary, industrial-style
benchmark), **Colorado** (a second, geologically distinct well-log basin), and **medical /
UCI HAR** (human activity recognition, used as a transition-detection analogue of the
formation-top task to test how well the same pipelines transfer beyond geology).

> **GeoXTS** is an external dependency this project builds on, not part of this repository:
> <https://gitlab.lisn.upsaclay.fr/salimath/geox>. It provides the GradCAM-equipped TSC
> baselines; `pip install -e .` installs it locally.

## Installation

```bash
pip install -e .
```

## Project Structure

```
geoxts/                  ← GeoXTS library (external dependency, installed locally)
datasets/                ← one self-documented study per dataset (see each README.md)
├── north_sea/           ← North Sea NPD well logs — PRIMARY benchmark
│                          Classification/ (TSC baseline), GNN/, Transformer/, notebooks/
├── colorado/            ← Colorado well logs — notebooks/ (TSC + DTW) + Transformer/
├── medical/             ← UCI HAR medical analogue — Classification/ + Transformer/
└── _template/           ← Copy this when adding a new dataset
benchmarks/
├── ucr/                 ← UCR archive benchmark
└── uea/                 ← UEA archive benchmark
tutorials/               ← Method tutorials (dataset-agnostic, e.g. gradcam_tutorial.ipynb)
docs/images/             ← Figures and diagrams
```

Each dataset folder has its own `README.md` describing the data, layout, notebooks,
and references.

## Adding a New Dataset

1. Copy `datasets/_template/` to `datasets/<your_dataset_name>/`
2. Fill in `README.md` with dataset description and download instructions
3. Add data files to `data/training/` and `data/test/`
4. Number your notebooks: `01_preprocessing.ipynb`, `02_benchmark.ipynb`, etc.
5. All data paths in notebooks should be relative to the **repo root**,
   e.g. `datasets/<your_dataset_name>/data/training/logs.parquet`
6. Saved models go in `models/` inside the dataset folder — these are gitignored

## References

Per-dataset references live in each dataset's `README.md`. The methods most central to
the project as a whole — one per approach family:

- Salimath, Bugiotti & Wlodarczyk (2025). *GeoTS: A TSC Framework for Estimating Geological Formation to Model Carbon Storage Reservoirs.* KDD '25, 4773–4783. doi:10.1145/3711896.3737228 — framework this project builds on.
- Ismail Fawaz et al. (2019). *Deep Learning for Time Series Classification: A Review.* Data Mining and Knowledge Discovery 33(4), 917–963 — sliding-window TSC baselines.
- Nie et al. (2023). *A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers (PatchTST).* ICLR — seq2seq transformer.
- Hamilton, Ying & Leskovec (2017). *Inductive Representation Learning on Large Graphs (GraphSAGE).* NeurIPS 30 — GNN.
- Sakoe & Chiba (1978). *Dynamic Programming Algorithm Optimization for Spoken Word Recognition (DTW).* IEEE Trans. ASSP 26(1), 43–49 — DTW marker propagation.
- Anthony, Kanding & Selvan (2020). *Carbontracker: Tracking and Predicting the Carbon Footprint of Training Deep Learning Models.* ICML Workshop on Challenges in Deploying and Monitoring ML Systems — energy/carbon tracking.

## Licence

CC BY-NC-SA 4.0 — see [LICENCE](LICENCE)
