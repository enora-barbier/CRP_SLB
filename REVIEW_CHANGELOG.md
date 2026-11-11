# Review changelog

A record of the pre-publication clean-up, so the next person (or AI tool) working on the repo knows what has been done, how, and what is left.

- **Branch:** `george-review`, created from `main` at `62f980e`
- **Scope so far:** comments and markdown only. No code, logic, hyperparameters or saved outputs were changed, unless listed below.
- **Open issues for other folders:** see [`REVIEW_NOTES.md`](REVIEW_NOTES.md)

## Conventions used

Follow these when continuing the clean-up, so the notebooks stay consistent.

**Comments**
- Few and short. Comment only where a reader would otherwise have to stop and work something out: non-obvious choices, fallbacks, training rules hidden in arguments, abbreviations, long runs, units.
- No comment on self-explanatory code (plain imports, small helpers, saving CSVs).
- Remove decorative banners (`# ── Section ──────`), dead commented-out code and empty cells.
- Keep a commented-out line only if it is a real alternative, and say when to use it.
- Plain wording, so comments do not read as AI-generated: no semicolons joining clauses, no em dashes, no "serves as", "leverages", "ensures", no SHOUTY CAPS, no narrating the obvious, no polished docstrings in notebooks that otherwise have none.
- UK English spelling.
- Never add deliberate typos to look "human".

**References**
- Every filename mentioned in a comment or markdown cell must exist in the repo. Link to it with a relative path where possible.
- No personal paths (e.g. `Enora/...`) or notebooks that were never committed.

**Editing notebooks safely**
- Edit cell `source` fields with a small Python script over the notebook JSON, matching the exact old text, so outputs and metadata are preserved. Do not re-run cells to make edits.
- Keep each cell's `source` in the format it was stored in (a single string or a list of lines). Converting between them is harmless to Jupyter but bloats the git diff.
- Discard metadata-only changes VS Code makes when a notebook is opened (kernel name, Python version).
- Only edit folders you own. Colleagues are working on `datasets/north_sea/` and `datasets/colorado/`; log issues there in `REVIEW_NOTES.md` instead.

## Changes by file

### `benchmarks/ucr/notebooks/01_benchmark.ipynb`
- Rewrote comments to the conventions above (about 9 short comments in total).
- Fixed a misleading comment: `#optinal download if using any of the functions` sat above plain fastai imports.
- Replaced the `dataset_result()` docstring with a one-line comment.
- Removed dead commented-out lines (`#print(dsid)`, `#dsid = type[i]`, an unused split and sort), a cell holding only a commented-out import, and two empty cells.
- Saved outputs (accuracy and time tables) unchanged.

### `datasets/medical/Classification/combined_data_split_medical.ipynb`
- Removed all section banners, replacing them with one plain line where the reason was not obvious.
- Shortened the `load_raw_split` and `extract_transition_dataset` docstrings and removed em dashes.
- Removed comments restating the code (per-class numbers, shapes, `# 7`, `# 9`, per-key notes in `warm_state`).
- Corrected the validation share comment from "~19%" to "about 20%" (1,450 of 7,352 windows).
- The stale `FileNotFoundError` output in the config cell was cleared.

### `datasets/medical/Classification/experiment_carbon_tracking_medical_combined.ipynb`
- Code comments: removed all 15 banners, shortened the long explanations (deterministic CUDA, combined tracking, log backup, grid sanity check), removed comments duplicating the markdown, removed the empty last cell.
- References: `experiment_training_medical_final.ipynb` (does not exist) now links to `experiment_training_medical.ipynb`. The private `[np]carbon_tracking.ipynb` reference was replaced with "the North Sea carbon-tracking run".
- Markdown: log filename corrected from `..._carbon_canonical.csv` to `..._carbon_combined.csv` (what the code writes). Chart 1 heading now says "training + inference". Em dashes removed.

### `datasets/medical/Classification/experiment_carbon_tracking_medical_separate.ipynb`
- Same reference and markdown fixes as the combined notebook. Log filename corrected to `..._carbon_separate.csv`.
- Two code comments pointing at `experiment_training_medical_final` corrected.
- **Not yet done:** the full comment clean-up (banners, long explanations) has not been applied to this notebook's code cells.

### `datasets/medical/Classification/experiment_training_medical.ipynb`
- Code comments: removed all banners, replaced the long North Sea vs medical recipe table with a 3-line note on the one real difference (the loss), removed inline comments that repeat the `evaluate_medical` docstring, removed the wrong "Evaluate all 6 metrics" comment (7 are printed).
- `evaluate_medical` docstring: em dashes replaced with colons, `all_preds` / `all_trues` added to the list of returned keys.
- Shard comment corrected: it said "set to 0, 1, 2, or 3 on each VM" but `N_SHARDS = 1`. It now explains how `N_SHARDS` and `MY_SHARD` work together.
- `load_dataset` docstring: US "artifacts" replaced with a plain description.
- Markdown intro: analogy table corrected to match the code. The MAE analogue is `median_error`, not macro-F1, and the recall analogue is `transition_recall`. Shard instructions updated as above. Em dash removed.
- Markdown "Results viewer": said the secondary ranking was `f1_macro`, but the code ranks by `transition_recall` then `median_error`. Corrected.

### `datasets/medical/Classification/sequence_classification_medical.ipynb`
- Code comments: removed all banners, shortened the path, normalisation, sequence-building and transition-metric explanations, matched the wording used in `combined_data_split_medical.ipynb`.
- Added notes that the learning-curve, evaluation, transition-metric and summary cells show the **last** variant of the sweep, which was not obvious.
- Fixed a wrong shape comment: logits are `(B, S, n_classes)`, not `(B, S, C)` (C is channels elsewhere).
- Removed comments that restated "1 window = 2.56 s".
- Markdown intro: the comparison table described the old independent-window approach ("Transition detection: Not possible"). It now describes the current transition-centred `experiment_training_medical.ipynb`. Em dashes removed.

### `datasets/medical/Transformer/src/seq2seq_transformer_data_split.ipynb`
- Removed the numbered `# ==== / # N. Title / # ====` banners. Three became one-line comments (subject split, z-score, reload check) that match the wording in the Classification notebooks.
- Path comments: the old block said the notebook "runs from any working directory", but `SAVE_DIR` is built from the working directory, so it only lands in `Transformer/data/` when run from `Transformer/src/`. The comment now says that.
- Shortened the label, feature-mode and transition-target comments (US "Standardize" replaced, semicolons removed). The feature-mode comment now gives the actual sizes (54 features for `stats`, 1,152 for `flatten`).
- Markdown: added `lengths` to the list of arrays saved in each `.npz`.

### `datasets/medical/README.md`
- File structure: `seq2seq_transformer_data_split.ipynb` was shown at the top of `Transformer/`, but it lives in `Transformer/src/`. Added `src/`, `data/` and `metrics/` to the tree.
- Wording about the raw data and generated outputs: it said they were "`.gitignore`d", but the root `.gitignore` on `main` only ignores file types (`*.npz`, `*.pth`, ...), not those folders. The intended point is that they are not included in the repo, so the text now says that instead of describing ignore rules.
- "Works from any working directory": true for finding the raw data, but outputs are written relative to the working directory. Added a "Run each notebook from its own folder" note.
- Notebook table: `MY_SHARD (0–3)` replaced with how `N_SHARDS` / `MY_SHARD` actually work. The `_separate` carbon notebook was described as "training only"; it measures training and inference separately. Notebook names are now links.
- Transformer section: it said each timestep gets "a background (`None`) label or one of the six activities". The transformer labels each window position with one of the six activities; transitions come from label changes. Corrected.
- Em dashes removed throughout.

### Repo root
- Added [`REVIEW_NOTES.md`](REVIEW_NOTES.md): issues in folders owned by others.
- Added this changelog.

## Still to do

- [ ] Full comment clean-up of `experiment_carbon_tracking_medical_separate.ipynb`
- [ ] `experiment_training_medical.ipynb`: plot titles in the best-model and confusion cells use em dashes (code strings, left unchanged)
- [ ] Decide whether `N_SHARDS = 1` in `experiment_training_medical.ipynb` is intended. The original notebook text described a 4-shard run (`MY_SHARD` 0 to 3)
- [ ] Remove the hard-coded Electricity Maps API key from both medical carbon notebooks (read it from the environment instead), and revoke the key, as it is in the git history
- [ ] Fix print messages in the combined carbon notebook that name the wrong output files (`medical_per_model_carbon_summary.csv` and `medical_inference_latency.csv`; the files actually saved have a `combined_` prefix)
- [ ] Remaining markdown in the medical notebooks: em dashes in the `combined_data_split_medical` intro, and its "(incorrect)" label for the alternative approach
- [ ] **`datasets/medical/Transformer/data/warm_state.json` (committed) contains a personal absolute path**: `HAR_ROOT` points into a team member's home folder on their laptop. Its `SAVE_DIR` (`har_seq2seq_transformer`) also does not match the current notebook, so the file was produced by an older version. Regenerate it, or replace the two path values with relative ones, before publishing.
- [ ] `seq2seq_transformer_data_split.ipynb`: `SAVE_DIR = Path.cwd().parent / "data"` only works when run from `Transformer/src/`. Could use `REPO_ROOT / "datasets/medical/Transformer/data"` instead, like the data path. It also writes absolute paths (`HAR_ROOT`, `SAVE_DIR`) into `warm_state.json`, which is how the personal path above got there.
- [ ] Items in [`REVIEW_NOTES.md`](REVIEW_NOTES.md) for the North Sea, Colorado and benchmark owners
