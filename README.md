# GeoXTS — CRP/SLB Research Project

Benchmark library for time series classification with GradCAM visualisations,
focused on geological and medical data. Developed as part of the Corporate Research
Project (CRP) in collaboration with SLB.

## Installation

```bash
pip install -e .
```

## Project Structure

```
geoxts/                  ← core Python library
datasets/
├── colorado/            ← Colorado well logs (geological)
│   ├── data/
│   ├── notebooks/
│   └── models/          ← models trained on this dataset (not tracked by git)
├── north_sea/           ← North Sea well logs (geological)
├── medical/             ← Medical time series
└── _template/           ← Copy this when adding a new dataset
benchmarks/
├── ucr/                 ← UCR archive benchmark
└── uea/                 ← UEA archive benchmark
tutorials/               ← Method tutorials (dataset-agnostic)
docs/images/             ← Figures and diagrams
```

## Adding a New Dataset

1. Copy `datasets/_template/` to `datasets/<your_dataset_name>/`
2. Fill in `README.md` with dataset description and download instructions
3. Add data files to `data/training/` and `data/test/`
4. Number your notebooks: `01_preprocessing.ipynb`, `02_benchmark.ipynb`, etc.
5. All data paths in notebooks should be relative to the **repo root**,
   e.g. `datasets/<your_dataset_name>/data/training/logs.parquet`
6. Saved models go in `models/` inside the dataset folder — these are gitignored

## Licence

CC BY-NC-SA 4.0 — see [LICENCE](LICENCE)
