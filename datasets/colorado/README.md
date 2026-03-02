# Colorado Well Logs Dataset

## Description
Geological well log dataset from Colorado. Contains gamma ray (GR) measurements
with depth, latitude, and longitude. Formation tops: NIOBARA, CODELL, FORT_HAYS.

## Data Access
Data files are stored in `data/training/` and `data/test/`. Not publicly distributed —
contact the project maintainer for access.

## File Structure
```
colorado/
├── data/
│   ├── training/
│   │   ├── logs.parquet              # well log measurements (GR, DEPTH)
│   │   ├── loc.parquet               # well locations (lat, lon)
│   │   ├── tops.parquet              # formation tops
│   │   └── training_well_list.npy   # list of training well names
│   └── test/
│       ├── logs_50.parquet           # 50-well test set
│       ├── logs_100.parquet          # 100-well test set
│       ├── loc_50.parquet
│       ├── loc_100.parquet
│       ├── tops_50.csv
│       └── tops_100.csv
├── notebooks/
├── models/                           # saved weights (gitignored)
└── README.md
```

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 01 | preprocessing | ETL and data quality control |
| 02 | benchmark | Deep learning model benchmark (FCN, LSTM, InceptionTime, XCM variants) |
| 03 | marker_propagation | Geological marker propagation using DTW |
| 04 | industrial_baseline | DTW-based industrial baseline comparison |
| 05 | tutorial_marker_propagation | Step-by-step marker propagation tutorial |

## Key Variables
- Input: `GR`, `Depth`, `Latitude`, `Longitude`
- Target: formation tops (`NIOBARA`, `CODELL`, `FORT_HAYS`)
- Window size: 201 depth samples

## References
<!-- Add relevant papers or reports -->
