# Review notes before publication

Issues found while tidying the medical notebooks for the conference release. Changes to the medical folder are on the `george-review` branch. The items below are in folders other people are working on, so they have been left for the owners to fix.

## North Sea and Colorado Transformer notebooks

Files:
- `datasets/north_sea/Transformer/seq2seq-transformer-v2.litho_aligned.ipynb`
- `datasets/colorado/Transformer/seq2seq-transformer-v2.colorado_aligned.ipynb`

Both notebooks refer to files that are not in the repository, so readers cannot follow them:

| Cell | Reference | Suggested fix |
|---|---|---|
| Depth bounds helper (`# Lithoformer-aligned: ...`) | `cell 41 of NPD_2_Lithoformer_G2_final_copy_carbon.ipynb` | Describe the rule instead, e.g. "hard 2000-14000 ft band, as in the Lithoformer G2 notebook" |
| `# __CARBON_SETUP__` | `NPD_2_Lithoformer_G2_final_copy_carbon.ipynb` and `Enora/[np]carbon_tracking.ipynb` | "same setup as the other carbon-tracking notebooks in the project" |
| `# __CARBON_METRICS_DUMP__` | `NPD_2_Lithoformer_G2_final_copy_carbon.ipynb` | "same format as the Lithoformer G2 notebook" |
| `_parse_carbontracker_log` docstring | `Enora/[np]carbon_tracking.ipynb` | "same parser as in the carbon-tracking notebooks" |

`Enora/...` is a personal path and `NPD_2_Lithoformer_G2_final_copy_carbon.ipynb` was never committed. Either add the referenced notebooks to the repo or remove the filenames.

## UCR benchmark folder

`benchmarks/ucr/notebooks/01_benchmark.ipynb` runs the 25 multivariate **UEA** datasets (ArticularyWordRecognition, NATOPS, PEMS-SF, ...), not the UCR archive. The function is called `get_UCR_data` only because tsai uses that name for both archives. Reviewers may notice, so it is worth checking against `benchmarks/uea/` and renaming or moving the notebook.

## Repository-wide

- **API key in git history.** An Electricity Maps API key is hard-coded in the medical carbon-tracking notebooks (`ELECTRICITYMAPS_API_KEY`). Even after it is removed from the code it stays in the commit history, so it should be revoked or rotated in the Electricity Maps account before the repo is made public. No other hard-coded keys were found.
- **`.gitignore` on `main`** does not exclude the local data folders (`datasets/north_sea/data/`, `datasets/colorado/data/`). Anyone running `git add .` will stage the raw data. Worth adding these before release.
