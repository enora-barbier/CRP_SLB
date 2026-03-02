# UCR Time Series Benchmark

## Description
Standard UCR time series classification archive, used as a reference benchmark
for comparing geoxts models against published baselines.

## Data Access
Data is downloaded automatically via `tsai`'s `get_UCR_data(dsid)` function.
No local files needed.

## Notebooks
| # | Notebook | Description |
|---|----------|-------------|
| 01 | benchmark | Run all models across UCR datasets, record accuracy and timing |

## Output Files
- `UCR_dataset.csv` — dataset feature summary
- `UCR_benchmark_accuracy.csv` — per-dataset accuracy results
- `UCR_benchmark_time.csv` — per-dataset training times

## References
- [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
