"""
Builds GNN_carbon_recall_deliverable.xlsx from the raw GCP result CSVs that are
committed next to each dataset:

  datasets/colorado/results/            colorado_stage2_carbon_log.csv
                                        colorado_stage2_boundary_recall_recovery.csv
  datasets/north_sea/GNN/results/       stage2_carbon_log.csv
                                        stage2_boundary_recall_recovery.csv
  datasets/HAR/results/                 har_stage2_carbon_log_{15,30}ep.csv
                                        har_stage2_transition_recall_recovery_{15,30}ep.csv

Run from anywhere:  python3 datasets/gnn_results/build_results_workbook.py
See datasets/gnn_results/README.md for how every column is defined.
"""
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "GNN_carbon_recall_deliverable.xlsx"

CO = ROOT / "datasets/colorado/results"
NS = ROOT / "datasets/north_sea/GNN/results"
HAR = ROOT / "datasets/HAR/results"

QUALITY_COLS = ["stage1_acc", "stage1_wf1", "stage1_mf1", "stage1_recall",
                "if_acc", "if_wf1", "if_mf1", "if_recall"]


def load_carbon(path, dedupe_latest=False):
    """Load a carbon log; optionally keep only the latest-timestamp row per (arch, seed)."""
    df = pd.read_csv(path)
    dropped = 0
    if dedupe_latest:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        before = len(df)
        df = df.sort_values("timestamp").drop_duplicates(["arch", "seed"], keep="last")
        dropped = before - len(df)
    assert not df.duplicated(["arch", "seed"]).any(), f"duplicate (arch, seed) rows in {path.name}"
    # carbon = (train energy + eval energy) x grid intensity, so expose the total energy too
    df["cc_total_energy_kwh"] = df["cc_energy_kwh"] + df["cc_eval_energy_kwh"]
    return df, dropped


def merge(carbon, recall, carbon_cols, recall_cols, label):
    out = (pd.merge(carbon[["arch", "seed"] + carbon_cols], recall[["arch", "seed"] + recall_cols],
                    on=["arch", "seed"], how="inner")
           .sort_values(["arch", "seed"]).reset_index(drop=True))
    assert len(out) == 30, f"{label}: merged rows = {len(out)}, expected 30"
    return out


def summarize(merged, family, recall_col, with_inference):
    spec = dict(
        recall_mean=(recall_col, "mean"), recall_std=(recall_col, "std"),
        energy_train_kwh_mean=("cc_energy_kwh", "mean"), energy_train_kwh_std=("cc_energy_kwh", "std"),
        energy_total_kwh_mean=("cc_total_energy_kwh", "mean"), energy_total_kwh_std=("cc_total_energy_kwh", "std"),
        carbon_g_mean=("cc_total_emissions_g", "mean"), carbon_g_std=("cc_total_emissions_g", "std"),
        n_reps=("seed", "count"),
    )
    if with_inference:
        spec.update(s_per_well_mean=("s_per_well", "mean"), s_per_well_std=("s_per_well", "std"),
                    ms_per_sample_mean=("ms_per_sample", "mean"), ms_per_sample_std=("ms_per_sample", "std"))
    s = merged.groupby("arch").agg(**spec).reset_index()
    s.insert(0, "family", family)
    return s


# ── Colorado ──────────────────────────────────────────────────────────────────
co_carbon, _ = load_carbon(CO / "colorado_stage2_carbon_log.csv")
co_recall = pd.read_csv(CO / "colorado_stage2_boundary_recall_recovery.csv")
WELL_CARBON = ["ms_per_sample", "s_per_well", "n_eval_wells", "cc_energy_kwh", "cc_eval_energy_kwh",
               "cc_total_energy_kwh", "cc_total_emissions_g"] + QUALITY_COLS
BOUNDARY_RECALL = ["recall_at_tol50ft", "n_matched", "n_true_boundaries"]
co_merged = merge(co_carbon, co_recall, WELL_CARBON, BOUNDARY_RECALL, "Colorado")
co_summary = summarize(co_merged, "Colorado", "recall_at_tol50ft", True)

# ── North Sea ─────────────────────────────────────────────────────────────────
# GraphSAGE seeds 5, 6, 7 each appear twice in the raw log (an earlier overlapping
# resume run + the good full run). Keep the later-timestamped row per (arch, seed).
ns_carbon, ns_dropped = load_carbon(NS / "stage2_carbon_log.csv", dedupe_latest=True)
ns_recall = pd.read_csv(NS / "stage2_boundary_recall_recovery.csv")
ns_merged = merge(ns_carbon, ns_recall, WELL_CARBON, BOUNDARY_RECALL, "North Sea")
ns_summary = summarize(ns_merged, "North Sea", "recall_at_tol50ft", True)

# ── UCI HAR (two runs: 15 and 30 epochs) ──────────────────────────────────────
HAR_CARBON = ["cc_energy_kwh", "cc_eval_energy_kwh", "cc_total_energy_kwh", "cc_total_emissions_g"] + QUALITY_COLS
HAR_RECALL = ["transition_recall", "transition_precision", "transition_f1", "f1_transition_only",
              "median_error", "n_true_transitions", "n_pred_transitions"]
har = {}
for ep in (15, 30):
    carbon, _ = load_carbon(HAR / f"har_stage2_carbon_log_{ep}ep.csv")
    assert (carbon["n_epochs_s1"] == ep).all(), f"HAR {ep}ep carbon log has unexpected n_epochs_s1"
    recall = pd.read_csv(HAR / f"har_stage2_transition_recall_recovery_{ep}ep.csv")
    merged = merge(carbon, recall, HAR_CARBON, HAR_RECALL, f"HAR {ep}ep")
    har[ep] = (merged, summarize(merged, f"UCI HAR ({ep} epochs)", "transition_recall", False))


# ── Paper-style combined table ────────────────────────────────────────────────
def fmt(mean, std, digits):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def table_row(r, recall_note="", inference="n/a (no well concept for HAR)"):
    return {
        "Family": r["family"], "Model": r["arch"],
        "Recall": fmt(r["recall_mean"], r["recall_std"], 4) + recall_note,
        "Avg inference time / well (s)": inference,
        "Energy, training only (kWh)": fmt(r["energy_train_kwh_mean"], r["energy_train_kwh_std"], 6),
        "Energy, train+eval (kWh)": fmt(r["energy_total_kwh_mean"], r["energy_total_kwh_std"], 6),
        "Carbon, train+eval (gCO2eq)": fmt(r["carbon_g_mean"], r["carbon_g_std"], 5),
        "n_reps": r["n_reps"],
    }


rows = []
for summary in (co_summary, ns_summary):
    for _, r in summary.iterrows():
        rows.append(table_row(r, inference=fmt(r["s_per_well_mean"], r["s_per_well_std"], 5)))
for ep in (15, 30):
    for _, r in har[ep][1].iterrows():
        rows.append(table_row(r, recall_note=" [diff. metric: see notes]"))
combined = pd.DataFrame(rows)

notes = pd.DataFrame({"Note": [
    "RECALL (Colorado, North Sea) = Recall@Tolerance(+/-50ft): a true formation boundary counts as found if any "
    "predicted boundary in that validation well falls within +/-50ft; pooled over wells. Computed by the checkpoint-"
    "reload scripts recover_boundary_recall_colorado.py / recover_boundary_recall.py, with a 15ft minimum-run-length "
    "filter so single-sample classification flips are not counted as boundaries.",
    "",
    "RECALL (UCI HAR) = transition_recall as defined by evaluate_medical() in experiment_training_medical.ipynb (the "
    "reference for the Sliding family, which trains on the same curated har_transition/ data as the GNN): accuracy "
    "restricted to test windows whose TRUE label is a transition onset (label > 0). This is NOT the same metric as "
    "the Colorado/North Sea column and the two should not be read as directly comparable.",
    "",
    "INFERENCE TIME / WELL = s_per_well = final_eval_time_s / n_eval_wells, i.e. 1 sample = 1 well, matching the "
    "convention used for the paper's other model families on the well-log datasets. ms_per_sample (per depth-row) is "
    "kept in the per-seed sheets for reference only and is NOT comparable to the paper's other rows. Residual caveat: "
    "this still times the whole fuse_well() pipeline (cleaning + graph build + model forward + IsolationForest "
    "scoring) with no warm-up / cuda-sync / repeated passes, unlike the isolated model-only benchmark in the "
    "reference carbon-tracking notebooks.",
    "",
    "ENERGY SCOPE (read before comparing against other tables): 'Energy, training only' = cc_energy_kwh = CodeCarbon "
    "energy of Stage 1 training + IsolationForest fit, EXCLUDING evaluation. 'Energy, train+eval' adds "
    "cc_eval_energy_kwh (evaluation: the 16-point threshold search plus the final pass over the validation wells). "
    "Carbon is always (train + eval energy) x 30.08 gCO2eq/kWh (Norway pin), so the Carbon column corresponds to the "
    "train+eval energy column, not the training-only one. On Colorado, evaluation is roughly 90% of total GNN "
    "energy and on North Sea roughly 65-70%; on HAR it is only 2-4%. Use the column whose scope matches the "
    "other models' energy numbers.",
    "",
    f"NORTH SEA carbon log: GraphSAGE seeds 5, 6, 7 each had two rows (earlier overlapping resume run + the good "
    f"full run). Deduplicated by keeping the later-timestamped row per (arch, seed); {ns_dropped} rows dropped, 30 "
    "remain. The committed raw CSV is unmodified.",
    "",
    "UCI HAR was run twice, at 15 and at 30 epochs (EPOCHS env var), in separate checkpoint folders; both are shown "
    "rather than one replacing the other. See HAR_15ep_summary vs HAR_30ep_summary.",
    "",
    "CARBON TRACKING: CodeCarbon OfflineEmissionsTracker(country_iso_code='NOR'); mean +/- std over 15 seeds per "
    "architecture.",
]})

# ── Write workbook ────────────────────────────────────────────────────────────
with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
    combined.to_excel(xl, sheet_name="Paper_Table", index=False)
    notes.to_excel(xl, sheet_name="Paper_Table", index=False, startrow=len(combined) + 3)
    co_merged.to_excel(xl, sheet_name="Colorado_per_seed", index=False)
    co_summary.to_excel(xl, sheet_name="Colorado_summary", index=False)
    ns_merged.to_excel(xl, sheet_name="NorthSea_per_seed", index=False)
    ns_summary.to_excel(xl, sheet_name="NorthSea_summary", index=False)
    for ep in (15, 30):
        har[ep][0].to_excel(xl, sheet_name=f"HAR_{ep}ep_per_seed", index=False)
        har[ep][1].to_excel(xl, sheet_name=f"HAR_{ep}ep_summary", index=False)

wb = load_workbook(OUT)
for ws in wb.worksheets:
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)
wb.save(OUT)

print(f"North Sea: dropped {ns_dropped} duplicate (arch, seed) row(s), kept latest timestamp")
print(f"Wrote {OUT.relative_to(ROOT)}  sheets={wb.sheetnames}\n")
print(combined.to_string(index=False))
