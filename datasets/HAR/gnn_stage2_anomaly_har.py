"""
GNN + Isolation Forest for UCI HAR activity transitions.

HAR counterpart of the Colorado / North Sea GNN scripts, with the same carbon
tracking (CodeCarbon and CarbonTracker, training and evaluation measured
separately, N_REPEATS seeds for mean +/- std).

How the well-log setup maps onto HAR:
  well            -> subject
  depth sample    -> timestep (1/50 s)
  log curves      -> 9 inertial channels
  formation       -> activity class
  formation top   -> activity transition
  background      -> class 0 ("None", no transition)

Stage 1: GNN over each 128-step window, mean-pooled, then a 7-class linear head.
Stage 2: Isolation Forest fitted on the None-class training embeddings. Windows
         predicted as None but flagged anomalous are reassigned to the nearest
         class centroid (centroids come from training embeddings only).

Environment variables: DATA_DIR, CKPT_DIR, OUT_DIR, N_REPEATS, BASE_SEED, EPOCHS,
ARCHS_OVERRIDE, ELECTRICITYMAPS_API_KEY.
"""

import warnings, os, re, time, pickle, json, csv, socket, subprocess, uuid, random
warnings.filterwarnings("ignore")
os.environ["CODECARBON_LOG_LEVEL"] = "error"
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.nn.pytorch as dglnn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker
from carbontracker.tracker import CarbonTracker

# Paths: environment variables first, otherwise relative to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR  = os.environ.get("DATA_DIR",  SCRIPT_DIR)
DATA_DIR  = os.path.join(DATA_DIR, "")  # guarantee a trailing slash

CKPT_DIR  = os.environ.get("CKPT_DIR",  os.path.join(SCRIPT_DIR, "har_stage2_ckpt"))
OUT_DIR   = os.environ.get("OUT_DIR",   CKPT_DIR)
CARBON_DIR = Path(CKPT_DIR) / "carbon_logs"

CONF_PATH = os.path.join(OUT_DIR, "har_stage2_confusion.png")
CARBON_FIG_PATH = os.path.join(OUT_DIR, "har_stage2_carbon_charts.png")
RES_PATH  = os.path.join(OUT_DIR, "har_stage2_results.pkl")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)
CARBON_DIR.mkdir(exist_ok=True)

print(f"DATA_DIR  : {DATA_DIR}")
print(f"CKPT_DIR  : {CKPT_DIR}")
print(f"OUT_DIR   : {OUT_DIR}")

MACHINE = socket.gethostname()
try:
    GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    GIT_SHA = "unknown"

# Carbon tracking is pinned to Norway so results do not depend on where the VM
# happens to sit; CodeCarbon runs offline with an explicit country code.
CARBON_COUNTRY_ISO3 = "NOR"
_ELECTRICITYMAPS_KEY = os.environ.get("ELECTRICITYMAPS_API_KEY")
_ct_api_keys = {"electricitymaps": _ELECTRICITYMAPS_KEY} if _ELECTRICITYMAPS_KEY else None
if _ct_api_keys is None:
    print("  [carbon] ELECTRICITYMAPS_API_KEY not set in the environment — CarbonTracker "
          "will fall back to its static Norway average instead of live grid data.")

# CarbonTracker always geolocates the machine from its outbound IP (carbontracker
# 2.4.4 has no override), which would give the Netherlands on a europe-west4 VM.
# Patch the lookup to return Norway before any tracker is built.
import types
from carbontracker.emissions.intensity import intensity as _ct_intensity

def _forced_norway_location(self):
    return types.SimpleNamespace(
        ok=True, address="Oslo, Norway (forced — see _forced_norway_location)",
        country="NO", lat=59.9139, lng=10.7522,
    )
_ct_intensity.IntensityService._fetch_geo_location = _forced_norway_location

# One independent run per seed; N_REPEATS=1 gives a quick smoke test.
N_REPEATS = int(os.environ.get("N_REPEATS", "15"))
BASE_SEED = int(os.environ.get("BASE_SEED", "0"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  DEVICE    : {DEVICE}")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── Carbon helpers ──────────────────────────────────────────────────────────
def _parse_carbontracker_log(log_dir: Path, prefix: str):
    files = sorted(
        Path(log_dir).glob(f"{prefix}*carbontracker*.log"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not files:
        return np.nan, np.nan
    for path in files:
        text = path.read_text(errors="ignore")
        for heading in ("Actual consumption", "Predicted consumption", "Final emissions", "Total"):
            m = re.search(
                heading + r".*?Energy:\s*([\d.eE+-]+)\s*kWh.*?(?:CO2eq|CO2):\s*([\d.eE+-]+)\s*g",
                text, re.DOTALL | re.IGNORECASE,
            )
            if m:
                return float(m.group(1)), float(m.group(2))
    text = files[0].read_text(errors="ignore")
    kwh_matches = re.findall(r"([\d.eE+-]+)\s*kWh", text)
    co2_matches = re.findall(r"([\d.eE+-]+)\s*g\s*(?:CO2eq|CO2|of\s*CO2)", text, re.IGNORECASE)
    if kwh_matches and co2_matches:
        return float(kwh_matches[-1]), float(co2_matches[-1])
    return np.nan, np.nan

def _read_codecarbon_csv(path):
    empty = {"emissions_kg": np.nan, "energy_kwh": np.nan,
             "cpu_kwh": np.nan, "gpu_kwh": np.nan, "ram_kwh": np.nan,
             "country": None, "region": None}
    if not Path(path).exists():
        return empty
    df = pd.read_csv(path)
    if df.empty:
        return empty
    r = df.iloc[-1]
    return {
        "emissions_kg": float(r.get("emissions", np.nan)),
        "energy_kwh":   float(r.get("energy_consumed", np.nan)),
        "cpu_kwh":      float(r.get("cpu_energy", np.nan)),
        "gpu_kwh":      float(r.get("gpu_energy", np.nan)),
        "ram_kwh":      float(r.get("ram_energy", np.nan)),
        "country":      r.get("country_name", None),
        "region":       r.get("region", None),
    }

def _tracker_spread(ct_co2_g, cc_emissions_kg):
    cc_g = cc_emissions_kg * 1000 if not np.isnan(cc_emissions_kg) else np.nan
    if np.isnan(ct_co2_g) or np.isnan(cc_g) or (ct_co2_g + cc_g) == 0:
        return np.nan
    return 100.0 * abs(cc_g - ct_co2_g) / (0.5 * (ct_co2_g + cc_g))

def _grid_tag(g_per_kwh):
    if np.isnan(g_per_kwh):       return "?? (no data)"
    if 5   <= g_per_kwh <= 80:    return "Norway-like  OK"
    if 200 <= g_per_kwh <= 320:   return "Netherlands-like  FAILED PIN"
    if 350 <= g_per_kwh <= 480:   return "USA-like  FAILED PIN"
    return f"unknown grid (~{g_per_kwh:.0f} g/kWh)  NOT Norway"

def _build_cc(run_name: str) -> OfflineEmissionsTracker:
    return OfflineEmissionsTracker(
        country_iso_code=CARBON_COUNTRY_ISO3,
        project_name=f"har_gnn_{run_name}",
        output_dir=str(CARBON_DIR),
        output_file=f"{run_name}_codecarbon.csv",
        measure_power_secs=5,
        tracking_mode="machine",
        log_level="error",
        save_to_file=True,
    )

def _build_ct(n_epochs: int, run_name: str) -> CarbonTracker:
    return CarbonTracker(
        epochs=n_epochs, monitor_epochs=1, epochs_before_pred=1,
        log_dir=str(CARBON_DIR), log_file_prefix=run_name + "_",
        api_keys=_ct_api_keys, verbose=1, ignore_errors=True,
    )

def _write_carbon_row(row: dict, csv_path: Path):
    df1 = pd.DataFrame([row])
    if csv_path.exists():
        with open(csv_path) as f:
            hdr = f.readline().rstrip("\n").split(",")
        if hdr != list(df1.columns):
            bak = csv_path.with_name(f"{csv_path.stem}__pre_{datetime.now():%Y%m%d_%H%M%S}.bak.csv")
            csv_path.rename(bak)
            print(f"  [log] schema changed — archived old log -> {bak.name}")
    df1.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

CARBON_CSV = Path(CKPT_DIR) / "har_stage2_carbon_log.csv"

# ── Data ────────────────────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "warm_state.json")) as f:
    ws = json.load(f)

CLASS_NAMES = ws["PRED_COLUMN"]           # ['None','WALKING',...]
N_CLASSES   = ws["N_CLASSES"]             # 7
WINDOW_LEN  = ws["WINDOW_LEN"]            # 128
N_CHANNELS  = len(ws["FEATURES_GEOX"])   # 9
DETECTORS   = [("IF only", "if")]
LABELS      = list(range(N_CLASSES))

train_d = np.load(os.path.join(DATA_DIR, "train.npz"))
val_d   = np.load(os.path.join(DATA_DIR, "val.npz"))
test_d  = np.load(os.path.join(DATA_DIR, "test.npz"))
X_train, y_train = train_d["X"], train_d["y"]  # (N, 9, 128), (N,)
X_val,   y_val   = val_d["X"],   val_d["y"]
X_test,  y_test  = test_d["X"],  test_d["y"]

print(f"  Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")
print(f"  Classes: {CLASS_NAMES}")

# ── Graph construction ──────────────────────────────────────────────────────
# Each (9, 128) window becomes a DGL graph: 128 nodes joined as a bidirectional
# chain, each carrying 9 features.
_SRC = list(range(WINDOW_LEN - 1)) + list(range(1, WINDOW_LEN))
_DST = list(range(1, WINDOW_LEN)) + list(range(WINDOW_LEN - 1))

def window_to_graph(x: np.ndarray) -> dgl.DGLGraph:
    """Turn a (9, 128) window into a graph with (128, 9) node features and self-loops."""
    g = dgl.graph((_SRC, _DST))
    g = dgl.add_self_loop(g)  # must happen before batching, it breaks batched graphs
    g.ndata["features"] = torch.tensor(x.T, dtype=torch.float32)
    return g

class HARGraphDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.graphs = [window_to_graph(X[i]) for i in range(len(X))]
        self.labels = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.labels[idx]

def collate_fn(batch):
    graphs, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(labels)

print("\nBuilding datasets...")
train_ds  = HARGraphDataset(X_train, y_train)
val_ds    = HARGraphDataset(X_val,   y_val)
test_ds   = HARGraphDataset(X_test,  y_test)

BATCH = 32
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")

# ── Class weights ───────────────────────────────────────────────────────────
_counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
print("\nTraining class distribution + weights:")
for c, (name, cnt) in enumerate(zip(CLASS_NAMES, _counts)):
    print(f"  class {c} ({name:<22}): {int(cnt):>4}")
# Square-root inverse frequency: a milder correction than full inverse frequency,
# so rare activities are not over-predicted at the cost of None recall.
_inv = np.sqrt(_counts.max() / np.where(_counts == 0, 1, _counts))
class_weights = torch.tensor(_inv, dtype=torch.float32).to(DEVICE)
criterion     = nn.CrossEntropyLoss(weight=class_weights)
print(f"  Weights (sqrt inv-freq): { {n: f'{w:.2f}' for n,w in zip(CLASS_NAMES, _inv)} }")

# ── Models ──────────────────────────────────────────────────────────────────
class GraphSAGEClassifier(nn.Module):
    """Three GraphSAGE layers, mean pooling over each window's nodes, linear head."""
    def __init__(self, in_feats: int, hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.c1  = dglnn.SAGEConv(in_feats, hidden, "mean")
        self.c2  = dglnn.SAGEConv(hidden,   hidden, "mean")
        self.c3  = dglnn.SAGEConv(hidden,   hidden, "mean")
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_classes)

    def embed(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(F.relu(self.bn1(self.c1(g, x))))
        h = self.drop(F.relu(self.bn2(self.c2(g, h))))
        h = self.drop(F.relu(self.bn3(self.c3(g, h))))
        # Per-graph mean pool, done by hand so it is correct for batched graphs.
        sizes  = g.batch_num_nodes().tolist()
        splits = torch.split(h, sizes, dim=0)
        return torch.stack([s.mean(0) for s in splits])  # (batch_size, hidden)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(g, x))


class ImprovedGNNClassifier(nn.Module):
    """Stacked GraphConv, BatchNorm, LeakyReLU and dropout, then mean pooling and a
    linear head. Same layer stack as ImprovedGNN in the Colorado / North Sea
    scripts, but pooled to one prediction per window."""
    def __init__(self, in_feats: int, hidden_sizes: list, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        prev = in_feats
        for h in hidden_sizes:
            self.convs.append(dglnn.GraphConv(prev, h))
            self.bns.append(nn.BatchNorm1d(h))
            prev = h
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(prev, n_classes)

    def embed(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = self.drop(F.leaky_relu(bn(conv(g, h))))
        sizes  = g.batch_num_nodes().tolist()
        splits = torch.split(h, sizes, dim=0)
        return torch.stack([s.mean(0) for s in splits])

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(g, x))


def _build_stage1(arch_name: str, nc: int) -> nn.Module:
    if arch_name == 'GraphSAGE':
        return GraphSAGEClassifier(N_CHANNELS, H, nc)
    elif arch_name == 'ImprovedGNN':
        return ImprovedGNNClassifier(N_CHANNELS, [128, 256, 256, 128, 64], nc)
    raise ValueError(f"Unknown arch: {arch_name}")


# ── Helpers (use the module-level `model`, rebuilt for every repeat) ────────
def get_embeddings(dl: DataLoader):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for bg, lbl in dl:
            bg = bg.to(DEVICE)
            embs.append(model.embed(bg, bg.ndata["features"]).cpu().numpy())
            labels.append(lbl.numpy())
    return np.vstack(embs), np.concatenate(labels)

def full_s1_preds(dl):
    model.eval(); preds = []
    with torch.no_grad():
        for bg, _ in dl:
            bg = bg.to(DEVICE)
            preds += model(bg, bg.ndata["features"]).argmax(1).tolist()
    return np.array(preds)

def fuse(s1_preds: np.ndarray, embs: np.ndarray, score: np.ndarray,
         threshold: float, centroids: dict) -> np.ndarray:
    """Reassign anomalous windows predicted as None to the nearest activity centroid."""
    fused = s1_preds.copy()
    for i in np.where(score >= threshold)[0]:
        if fused[i] == 0:
            dists   = [np.linalg.norm(embs[i] - centroids[c])
                       for c in range(1, N_CLASSES)]
            fused[i] = int(np.argmin(dists)) + 1
    return fused


# ── Training and evaluation loop: N_REPEATS runs per architecture ───────────
# EPOCHS is read from the environment, e.g. EPOCHS=30 CKPT_DIR=... python <this file>
H = 128; EPOCHS = int(os.environ.get("EPOCHS", "15")); LR = 0.001

# ARCHS_OVERRIDE runs a subset, e.g. ARCHS_OVERRIDE="GraphSAGE" BASE_SEED=5 N_REPEATS=10
# to finish the remaining seeds of one architecture.
_archs_override = os.environ.get("ARCHS_OVERRIDE")
ARCHS = [a.strip() for a in _archs_override.split(",")] if _archs_override else ['GraphSAGE', 'ImprovedGNN']

print(f"\n{'='*62}")
print(f"  {len(ARCHS)} arch(s) x {N_REPEATS} repeats, seeds {BASE_SEED}..{BASE_SEED+N_REPEATS-1}")
print(f"{'='*62}")

# Write the CSV header up front so the file exists as soon as the script starts.
# The column order must match the `row` dict built below, because
# _write_carbon_row() archives the log if the header ever differs.
_CARBON_CSV_COLUMNS = (
    ["run_id", "arch", "seed", "rep", "timestamp", "machine", "git_sha",
     "n_epochs_s1", "train_time_s", "eval_time_s",
     "stage1_acc", "stage1_wf1", "stage1_mf1", "stage1_recall"]
    + [f"{key}_{m}" for _, key in DETECTORS for m in ("thresh", "acc", "wf1", "mf1", "recall")]
    + ["ct_energy_kwh", "ct_emissions_g", "cc_energy_kwh", "cc_emissions_g",
       "cc_cpu_kwh", "cc_gpu_kwh", "cc_ram_kwh",
       "cc_eval_energy_kwh", "cc_eval_emissions_g", "cc_total_emissions_g",
       "cc_country", "cc_region", "cc_implied_g_per_kwh", "ct_implied_g_per_kwh",
       "grid_ok", "tracker_spread_pct", "stage1_model_path"]
)
if not CARBON_CSV.exists():
    with open(CARBON_CSV, "w", newline="") as f:
        csv.writer(f).writerow(_CARBON_CSV_COLUMNS)
    print(f"  [carbon] initialized empty log -> {CARBON_CSV}")
else:
    print(f"  [carbon] appending to existing log -> {CARBON_CSV}")

AGG_COLS = ["ct_energy_kwh", "ct_emissions_g", "cc_energy_kwh", "cc_emissions_g",
            "cc_cpu_kwh", "cc_gpu_kwh", "cc_ram_kwh", "cc_eval_energy_kwh",
            "cc_eval_emissions_g", "cc_total_emissions_g", "tracker_spread_pct",
            "train_time_s", "eval_time_s",
            "stage1_acc", "stage1_wf1", "stage1_mf1", "stage1_recall"] + \
           [f"{key}_{m}" for _, key in DETECTORS for m in ("acc", "wf1", "mf1", "recall")]

rep_rows = []
arch_results = {}

for arch_name in ARCHS:
    arch_key = arch_name.lower()
    print(f"\n{'#'*62}")
    print(f"  ARCHITECTURE: {arch_name}")
    print(f"{'#'*62}")

    arch_rep_rows = []

    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        print(f"\n--- {arch_name}  rep {rep+1}/{N_REPEATS}  seed={seed} ---")
        set_seed(seed)

        model = _build_stage1(arch_name, N_CLASSES).to(DEVICE)

        # Stage 1 training (tracked)
        s1_run = f"stage1_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        s1_path = Path(CKPT_DIR) / f"stage1_har_{arch_key}_seed{seed}.pt"
        cc_s1 = _build_cc(s1_run); ct_s1 = _build_ct(EPOCHS, s1_run)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        best_val_f1 = 0.0
        cc_s1.start(); t0 = time.time()
        try:
            for ep in range(EPOCHS):
                ct_s1.epoch_start()
                model.train(); tl = 0; tr_p = []; tr_l = []
                for bg, lbl in train_dl:
                    bg = bg.to(DEVICE); lbl = lbl.to(DEVICE)
                    opt.zero_grad()
                    out = model(bg, bg.ndata["features"])
                    loss = criterion(out, lbl)
                    if torch.isnan(loss): continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); tl += loss.item()
                    tr_p += out.argmax(1).tolist(); tr_l += lbl.tolist()
                sch.step()

                model.eval(); vp = []; vl = []
                with torch.no_grad():
                    for bg, lbl in val_dl:
                        bg = bg.to(DEVICE)
                        vp += model(bg, bg.ndata["features"]).argmax(1).tolist()
                        vl += lbl.tolist()

                tr_f1 = f1_score(tr_l, tr_p, average="weighted", zero_division=0)
                vf1   = f1_score(vl,   vp,   average="weighted", zero_division=0)
                print(f"    ep {ep+1:2d}/{EPOCHS}  loss={tl/len(train_dl):.4f}"
                      f"  tr_f1={tr_f1:.4f}  val_f1={vf1:.4f}")
                if vf1 > best_val_f1:
                    best_val_f1 = vf1
                    torch.save(model.state_dict(), s1_path)
                ct_s1.epoch_end()
        finally:
            try: ct_s1.stop()
            except Exception: pass
            try: cc_s1.stop()
            except Exception: pass
        elapsed_s1 = time.time() - t0
        # The loop leaves `model` at its last epoch; reload the best-val-F1 checkpoint.
        model.load_state_dict(torch.load(s1_path, map_location="cpu"))
        print(f"    Best val F1={best_val_f1:.4f}  →  {s1_path}")

        # Window embeddings
        train_embs, train_lbls = get_embeddings(train_dl)
        val_embs,   val_lbls   = get_embeddings(val_dl)
        test_embs,  test_lbls  = get_embeddings(test_dl)

        # Stage 2: Isolation Forest (tracked)
        if_run = f"stage2a_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_if = _build_cc(if_run); ct_if = _build_ct(1, if_run)
        cc_if.start(); ct_if.epoch_start(); t0_if = time.time()
        try:
            none_embs = train_embs[train_lbls == 0]
            iso = IsolationForest(n_estimators=200, contamination=0.05,
                                  random_state=seed, n_jobs=-1)
            iso.fit(none_embs)
        finally:
            ct_if.epoch_end()
            try: ct_if.stop()
            except Exception: pass
            try: cc_if.stop()
            except Exception: pass
        elapsed_if = time.time() - t0_if
        print(f"    IF: trained on {len(none_embs)} None-class embeddings")

        # Class centroids, from training embeddings only
        centroids = {
            c: train_embs[train_lbls == c].mean(axis=0)
            for c in range(1, N_CLASSES)
            if (train_lbls == c).sum() > 0
        }

        # Evaluation (tracked separately): threshold search on val, final pass on test
        eval_run = f"eval_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_eval = _build_cc(eval_run)
        cc_eval.start(); t0_eval = time.time()
        try:
            val_if_sc  = minmax_scale(-iso.score_samples(val_embs))
            val_s1     = full_s1_preds(val_dl)

            test_if_sc = minmax_scale(-iso.score_samples(test_embs))
            test_s1    = full_s1_preds(test_dl)

            val_scores  = {"if": val_if_sc}
            test_scores = {"if": test_if_sc}

            best_thresh = {key: 0.5 for _, key in DETECTORS}
            best_f1     = {key: 0.0 for _, key in DETECTORS}
            for thresh in np.arange(0.05, 0.85, 0.05):
                for _, key in DETECTORS:
                    fp  = fuse(val_s1, val_embs, val_scores[key], thresh, centroids)
                    mf1 = f1_score(val_lbls, fp, average="macro", zero_division=0)
                    if mf1 > best_f1[key]: best_f1[key] = mf1; best_thresh[key] = thresh

            all_fused = {}
            for _, key in DETECTORS:
                all_fused[key] = fuse(test_s1, test_embs, test_scores[key], best_thresh[key], centroids)
        finally:
            try: cc_eval.stop()
            except Exception: pass
        elapsed_eval = time.time() - t0_eval

        perf = {"stage1_acc": accuracy_score(test_lbls, test_s1),
                "stage1_wf1": f1_score(test_lbls, test_s1, average="weighted", zero_division=0),
                "stage1_mf1": f1_score(test_lbls, test_s1, average="macro", zero_division=0),
                "stage1_recall": recall_score(test_lbls, test_s1, average="weighted", zero_division=0)}
        for _, key in DETECTORS:
            perf[f"{key}_thresh"] = float(best_thresh[key])
            perf[f"{key}_acc"] = accuracy_score(test_lbls, all_fused[key])
            perf[f"{key}_wf1"] = f1_score(test_lbls, all_fused[key], average="weighted", zero_division=0)
            perf[f"{key}_mf1"] = f1_score(test_lbls, all_fused[key], average="macro", zero_division=0)
            perf[f"{key}_recall"] = recall_score(test_lbls, all_fused[key], average="weighted", zero_division=0)
        print(f"    [perf] stage1 acc={perf['stage1_acc']:.4f}  "
              + "  ".join(f"{key}: acc={perf[f'{key}_acc']:.4f} mf1={perf[f'{key}_mf1']:.4f} thr={perf[f'{key}_thresh']:.2f}"
                           for _, key in DETECTORS))

        # Read both trackers back and check the grid intensity
        ct_kwh_s1, ct_co2_s1 = _parse_carbontracker_log(CARBON_DIR, prefix=s1_run + "_")
        cc_blob_s1 = _read_codecarbon_csv(CARBON_DIR / f"{s1_run}_codecarbon.csv")
        ct_kwh_if, ct_co2_if = _parse_carbontracker_log(CARBON_DIR, prefix=if_run + "_")
        cc_blob_if = _read_codecarbon_csv(CARBON_DIR / f"{if_run}_codecarbon.csv")
        cc_blob_eval = _read_codecarbon_csv(CARBON_DIR / f"{eval_run}_codecarbon.csv")

        cc_train_kwh = np.nansum([cc_blob_s1["energy_kwh"], cc_blob_if["energy_kwh"]])
        cc_train_g   = np.nansum([cc_blob_s1["emissions_kg"], cc_blob_if["emissions_kg"]]) * 1000
        cc_eval_g    = np.nan_to_num(cc_blob_eval["emissions_kg"], nan=0.0) * 1000
        cc_total_g   = cc_train_g + cc_eval_g
        ct_train_kwh = np.nansum([ct_kwh_s1, ct_kwh_if])
        ct_train_g   = np.nansum([ct_co2_s1, ct_co2_if])

        cc_implied = (cc_train_g / (cc_train_kwh * 1000)) * 1000 if cc_train_kwh > 0 else np.nan
        ct_implied = ct_train_g / ct_train_kwh if ct_train_kwh > 0 else np.nan
        spread = _tracker_spread(ct_train_g, cc_train_g / 1000)
        cc_tag = _grid_tag(cc_implied); ct_tag = _grid_tag(ct_implied)
        grid_ok = bool(not np.isnan(cc_implied) and 5 <= cc_implied <= 80)

        print(f"    [CC-train] energy={cc_train_kwh*1000:6.2f} Wh  emissions={cc_train_g:6.3f} g  "
              f"-> implied {cc_implied:6.1f} g/kWh  [{cc_tag}]")
        print(f"    [CT-train] energy={ct_train_kwh*1000:6.2f} Wh  emissions={ct_train_g:6.3f} g  "
              f"-> implied {ct_implied:6.1f} g/kWh  [{ct_tag}]  spread={spread:.1f}%")
        print(f"    [CC-eval]  energy={cc_blob_eval['energy_kwh']*1000:6.2f} Wh  emissions={cc_eval_g:6.3f} g   "
              f"|  [CC total train+infer] {cc_total_g:6.3f} g")

        row = {
            "run_id": str(uuid.uuid4()), "arch": arch_name, "seed": int(seed), "rep": int(rep),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "machine": MACHINE, "git_sha": GIT_SHA,
            "n_epochs_s1": EPOCHS,
            "train_time_s": int(elapsed_s1 + elapsed_if), "eval_time_s": int(elapsed_eval),
            **perf,
            "ct_energy_kwh": float(ct_train_kwh), "ct_emissions_g": float(ct_train_g),
            "cc_energy_kwh": float(cc_train_kwh), "cc_emissions_g": float(cc_train_g),
            "cc_cpu_kwh": float(np.nansum([cc_blob_s1["cpu_kwh"], cc_blob_if["cpu_kwh"]])),
            "cc_gpu_kwh": float(np.nansum([cc_blob_s1["gpu_kwh"], cc_blob_if["gpu_kwh"]])),
            "cc_ram_kwh": float(np.nansum([cc_blob_s1["ram_kwh"], cc_blob_if["ram_kwh"]])),
            "cc_eval_energy_kwh": float(cc_blob_eval["energy_kwh"]) if not np.isnan(cc_blob_eval["energy_kwh"]) else np.nan,
            "cc_eval_emissions_g": float(cc_eval_g),
            "cc_total_emissions_g": float(cc_total_g),
            "cc_country": cc_blob_s1["country"], "cc_region": cc_blob_s1["region"],
            "cc_implied_g_per_kwh": float(cc_implied), "ct_implied_g_per_kwh": float(ct_implied),
            "grid_ok": grid_ok,
            "tracker_spread_pct": float(spread),
            "stage1_model_path": str(s1_path),
        }
        arch_rep_rows.append(row)
        rep_rows.append(row)
        _write_carbon_row(row, CARBON_CSV)

        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    # Mean +/- std over this architecture's repeats (Norway-like grid rows only)
    df_arch = pd.DataFrame(arch_rep_rows)
    bad = df_arch[~df_arch["grid_ok"]]
    if len(bad):
        print(f"\n! {arch_name}: {len(bad)}/{len(df_arch)} rep(s) NOT Norway-like "
              f"(failed country pin) — excluded from the aggregate below.")
    df_arch_ok = df_arch[df_arch["grid_ok"]] if len(bad) < len(df_arch) else df_arch

    means = df_arch_ok[AGG_COLS].mean(); stds = df_arch_ok[AGG_COLS].std()
    print(f"\n{arch_name}: {len(df_arch_ok)}/{N_REPEATS} valid rep(s), mean +/- std:")
    for c in AGG_COLS:
        print(f"  {c:<24} {means[c]:>10.5f}  +/- {stds[c]:>9.5f}")

    # Reports below use the last repeat (its predictions are still in scope)
    print(f"\n── {arch_name}: Stage 1 only (last rep) ────────────────────────")
    print(classification_report(test_lbls, test_s1, target_names=CLASS_NAMES,
                                 labels=LABELS, zero_division=0))
    for name, key in DETECTORS:
        print(f"\n── {arch_name}: Stage 1 + {name} (last rep) {'─'*(38-len(name))}")
        print(classification_report(test_lbls, all_fused[key],
              target_names=CLASS_NAMES, labels=LABELS, zero_division=0))

    arch_results[arch_name] = {
        'true': test_lbls, 's1': test_s1, 'fused': dict(all_fused),
        'best_thresh': dict(best_thresh), 'n_reps_valid': len(df_arch_ok),
        'co2_total_mean_g': float(means['cc_total_emissions_g']),
        'co2_total_std_g':  float(stds['cc_total_emissions_g']),
    }

# ── Comparison across architectures (last repeat of each) ───────────────────
print("\n\n" + "="*70)
print("  COMBINED ARCHITECTURE COMPARISON  (perf: last repeat | carbon: mean of valid reps)")
print("="*70)
print(f"{'Arch':<14} {'Model':<16} {'Acc':>6} {'wF1':>6} {'macF1':>7}  {'CO2 total(g) mean+/-std':>26}  {'n_reps':>6}")
print("-"*90)
for arch_name, res in arch_results.items():
    for label, preds in [("Stage1", res['s1'])] + [(name, res['fused'][key]) for name, key in DETECTORS]:
        at  = res['true']
        acc = accuracy_score(at, preds)
        wf1 = f1_score(at, preds, average='weighted', zero_division=0)
        mf1 = f1_score(at, preds, average='macro', zero_division=0)
        if label != "Stage1":
            co2_str = f"{res['co2_total_mean_g']:.5f} +/- {res['co2_total_std_g']:.5f}"
        else:
            co2_str = "—"
        print(f"  {arch_name:<12} {label:<16} {acc:>6.4f} {wf1:>6.4f} {mf1:>7.4f}  {co2_str:>26}  {res['n_reps_valid']:>6}")

# ── Confusion matrices (one row per architecture: Stage 1, Stage 1 + IF) ────
n_methods = 1 + len(DETECTORS)
fig, axes = plt.subplots(len(ARCHS), n_methods, figsize=(7*n_methods, 6*len(ARCHS)), squeeze=False)
for row_idx, arch_name in enumerate(ARCHS):
    res = arch_results[arch_name]
    plots = [("Stage 1 only", res['s1'])] + [(name, res['fused'][key]) for name, key in DETECTORS]
    for col_idx, (title, preds) in enumerate(plots):
        ax = axes[row_idx][col_idx]
        cm = confusion_matrix(res['true'], preds, labels=LABELS)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f"{arch_name}: {title}", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=40)
plt.suptitle("HAR: GraphSAGE vs ImprovedGNN — Stage 1 vs Stage 1 + IF (last rep, test set)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(CONF_PATH, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix → {CONF_PATH}")
print(f"Carbon log → {CARBON_CSV}")

# ── Carbon charts (all architectures pooled) ────────────────────────────────
df_rep = pd.DataFrame(rep_rows)
df_ok  = df_rep[df_rep["grid_ok"]] if (~df_rep["grid_ok"]).any() and df_rep["grid_ok"].any() else df_rep
if len(df_ok):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.bar(["CarbonTracker\n(GPU)"], [df_ok["ct_emissions_g"].mean()],
           yerr=[df_ok["ct_emissions_g"].std()], capsize=4, label="CT")
    ax.bar(["CodeCarbon\n(CPU+GPU+RAM)"], [df_ok["cc_emissions_g"].mean()],
           yerr=[df_ok["cc_emissions_g"].std()], capsize=4, label="CC")
    ax.set_ylabel("Training emissions (gCO2eq)"); ax.set_title("Training emissions: CT vs CC")

    ax = axes[1]
    means_e = [df_ok["cc_cpu_kwh"].mean()*1000, df_ok["cc_gpu_kwh"].mean()*1000, df_ok["cc_ram_kwh"].mean()*1000]
    ax.bar(["CPU", "GPU", "RAM"], means_e, color=["#4472C4", "#70AD47", "#FFC000"])
    ax.set_ylabel("Energy (Wh)"); ax.set_title("CodeCarbon energy breakdown")

    ax = axes[2]
    ax.bar(["Training"], [df_ok["cc_emissions_g"].mean()], color="#4472C4")
    ax.bar(["Training"], [df_ok["cc_eval_emissions_g"].mean()],
           bottom=[df_ok["cc_emissions_g"].mean()], color="#FF7F0E")
    ax.set_ylabel("Emissions (gCO2eq)"); ax.set_title("Training vs inference emissions (stacked)")

    plt.tight_layout()
    plt.savefig(CARBON_FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Carbon charts → {CARBON_FIG_PATH}")

# ── Save results ────────────────────────────────────────────────────────────
with open(RES_PATH, "wb") as f:
    pickle.dump({
        "arch_results": arch_results,
        "class_names":  CLASS_NAMES,
        "rep_rows":     rep_rows,
    }, f)
print(f"Results → {RES_PATH}")
