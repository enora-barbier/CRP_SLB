# North Sea Well Logs Dataset

## Description
Well log dataset from the North Sea (hackathon training data). Contains gamma ray
measurements. Formation tops: MARCEL, SYLVAIN, CONRAD.

## Data Access
Data files go in `data/training/`. Source: hackathon_training_data.
Contact the project maintainer for access.

## File Structure
```
data/
└── training/
    ├── logs.parquet     # well log measurements
    └── tops.parquet     # formation tops (MARCEL, SYLVAIN, CONRAD)
```

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 01 | preprocessing | Data loading and quality control |

## Key Variables
- Input: `GR`, `DEPTH`
- Target: formation tops (`MARCEL`, `SYLVAIN`, `CONRAD`)

## References
<!-- Add relevant papers or reports -->
