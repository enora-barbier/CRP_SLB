"""
GNN + Isolation Forest formation-boundary detection on the Colorado wells.

Same pipeline and carbon tracking as the North Sea script (CodeCarbon and
CarbonTracker, training and evaluation measured separately, N_REPEATS seeds for
mean +/- std).

Differences from North Sea:
  features : gamma ray (GR_CLEAN) and normalised depth only
  markers  : NIOBARA, FORT_HAYS, CODELL. There is no background class: each well
             is cropped to start at its shallowest known top.
  well col : wellName
  depth col: DEPTH
  splits   : 600 training wells split 80/20 into train/val, plus a separate test set

Stage 1: node-level GNN classifier (GraphSAGE or ImprovedGNN).
Stage 2: Isolation Forest on the node embeddings; confident Stage 1 predictions
         are used to correct boundaries per well.

Environment variables: CKPT_DIR, OUT_DIR, N_REPEATS, BASE_SEED,
ELECTRICITYMAPS_API_KEY.
"""

import warnings, os, re, time, pickle, csv, socket, subprocess, uuid, random
warnings.filterwarnings("ignore")
os.environ["CODECARBON_LOG_LEVEL"] = "error"
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.nn.pytorch as dglnn
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)
from sklearn.preprocessing import minmax_scale
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker
from carbontracker.tracker import CarbonTracker

# Models and graph batches are moved to this device explicitly with .to(DEVICE);
# neither PyTorch nor DGL does it for you.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_TRAIN = "./"
BASE_TEST = "./"

# Checkpoints, logs and figures go under CKPT_DIR unless OUT_DIR is set.
CKPT_DIR  = os.environ.get("CKPT_DIR",  "/tmp/colorado_stage2_ckpt")
OUT_DIR   = os.environ.get("OUT_DIR",   CKPT_DIR)
CARBON_DIR = Path(CKPT_DIR) / "carbon_logs"
CONF_PATH = os.path.join(OUT_DIR, "colorado_stage2_confusion.png")
CARBON_FIG_PATH = os.path.join(OUT_DIR, "colorado_stage2_carbon_charts.png")
RES_PATH  = os.path.join(OUT_DIR, "colorado_stage2_results.pkl")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)
CARBON_DIR.mkdir(exist_ok=True)

print(f"BASE_TRAIN : {BASE_TRAIN}")
print(f"BASE_TEST  : {BASE_TEST}")
print(f"CKPT_DIR   : {CKPT_DIR}")
print(f"OUT_DIR    : {OUT_DIR}")
print(f"DEVICE     : {DEVICE}" + ("" if torch.cuda.is_available() else "  (no CUDA GPU detected -- running on CPU)"))

MACHINE = socket.gethostname()
try:
    GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    GIT_SHA = "unknown"

# Carbon accounting is pinned to Norway for the whole project, whatever the
# physical location of the machine. CodeCarbon runs offline with an explicit
# country code; CarbonTracker reads live grid intensity from Electricity Maps
# when ELECTRICITYMAPS_API_KEY is set.
CARBON_COUNTRY_ISO3 = "NOR"
_ELECTRICITYMAPS_KEY = os.environ.get("ELECTRICITYMAPS_API_KEY")
_ct_api_keys = {"electricitymaps": _ELECTRICITYMAPS_KEY} if _ELECTRICITYMAPS_KEY else None
if _ct_api_keys is None:
    print("  [carbon] ELECTRICITYMAPS_API_KEY not set in the environment — CarbonTracker "
          "will fall back to its built-in average grid-intensity estimate instead of "
          "live data. Set the env var to match the notebook's methodology exactly.")

# CarbonTracker always geolocates the machine from its outbound IP (carbontracker
# 2.4.4 has no override). Patch the lookup to return Norway before any tracker
# is built.
import types
from carbontracker.emissions.intensity import intensity as _ct_intensity

def _forced_norway_location(self):
    return types.SimpleNamespace(
        ok=True, address="Oslo, Norway (forced — see _forced_norway_location)",
        country="NO", lat=59.9139, lng=10.7522,
    )
_ct_intensity.IntensityService._fetch_geo_location = _forced_norway_location

# CarbonTracker's Electricity Maps request has no timeout, so an unresponsive API
# would hang the run instead of falling back to the static average. Replace it
# with a version that times out after 10 s.
from carbontracker.emissions.intensity.fetchers import electricitymaps as _ct_em
from carbontracker import exceptions as _ct_exceptions
import requests as _requests

def _carbon_intensity_by_location_with_timeout(self, lon=None, lat=None, zone=None):
    params = (("zone", zone),) if zone is not None else (("lon", lon), ("lat", lat))
    headers = {"auth-token": self._api_key}
    response = _requests.get(_ct_em.API_URL, headers=headers, params=params, timeout=10)
    if not response.ok:
        try: error_details = response.json()
        except Exception: error_details = "Bad response received from api. Could not parse json"
        raise _ct_exceptions.CarbonIntensityFetcherError(error_details)
    return response.json()["carbonIntensity"]

_ct_em.ElectricityMap._carbon_intensity_by_location = _carbon_intensity_by_location_with_timeout

# One independent run per seed; N_REPEATS=1 gives a quick smoke test.
N_REPEATS = int(os.environ.get("N_REPEATS", "15"))
BASE_SEED = int(os.environ.get("BASE_SEED", "0"))

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── Carbon helpers ──────────────────────────────────────────────────────────
def _parse_carbontracker_log(log_dir: Path, prefix: str):
    """Read (kWh, gCO2) from a CarbonTracker log.

    The file may be *_carbontracker.log or *_carbontracker_output.log depending on
    the version, and the "Actual consumption" block is missing if training stops
    before `monitor_epochs`. So try "Actual", then "Predicted", and finally fall
    back to the last kWh / gCO2 numbers in the file."""
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
        project_name=f"colorado_gnn_{run_name}",
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

CARBON_CSV = Path(CKPT_DIR) / "colorado_stage2_carbon_log.csv"

# ── Schema ────────────────────────────────────────────────────────────────────
MARKER_COLS     = ['NIOBARA', 'FORT_HAYS', 'CODELL']
INPUT_FEATURES  = ['GR_CLEAN', 'DEPTH_NORM']
# There is no background class. Each well is cropped to start at its shallowest
# known top (see WellGraphDataset.__getitem__ and fuse_well), so every node has a
# formation label, and wells with no known tops are skipped.
N_CLASSES      = len(MARKER_COLS)   # 3
MARKER_TO_CLASS = {'NIOBARA': 0, 'FORT_HAYS': 1, 'CODELL': 2}
CLASS_NAMES     = ['NIOBARA', 'FORT_HAYS', 'CODELL']
CLASS_COLORS    = ['#4472C4', '#70AD47', '#FFC000']
WELL_COL        = 'wellName'
DEPTH_COL       = 'DEPTH'
GR_SENTINEL     = -9999.0             # marks a missing GR value
GR_CLIP_LO, GR_CLIP_HI = 0.0, 300.0   # valid GR range; anything outside is masked

# ── Load data ─────────────────────────────────────────────────────────────────
df_logs_raw = pd.read_parquet(os.path.join(BASE_TRAIN, "logs.parquet"))
df_tops_raw = pd.read_parquet(os.path.join(BASE_TRAIN, "tops.parquet"))

# Well names are floats in the raw data; use strings in both tables.
df_logs_raw[WELL_COL]   = df_logs_raw[WELL_COL].astype(str)
df_tops_raw['well_name'] = df_tops_raw['well_name'].astype(str)
df_tops_raw = df_tops_raw.set_index('well_name')[MARKER_COLS]

# Test set (50 wells)
df_logs_test  = pd.read_parquet(os.path.join(BASE_TEST, "logs_50.parquet"))
df_tops_test  = pd.read_csv(os.path.join(BASE_TEST, "tops_50.csv"))
df_logs_test[WELL_COL]    = df_logs_test[WELL_COL].astype(str)
df_tops_test['wellName']  = df_tops_test['wellName'].astype(str)
df_tops_test = df_tops_test.set_index('wellName')[MARKER_COLS]

all_wells = sorted(df_logs_raw[WELL_COL].unique())
np.random.seed(42)
perm      = np.random.permutation(len(all_wells))
n_val     = int(len(all_wells) * 0.20)
val_wells  = set(np.array(all_wells)[perm[:n_val]])
train_wells = set(np.array(all_wells)[perm[n_val:]])

df_log_train = df_logs_raw[df_logs_raw[WELL_COL].isin(train_wells)].copy()
df_log_valid = df_logs_raw[df_logs_raw[WELL_COL].isin(val_wells)].copy()
df_tops_train = df_tops_raw.loc[df_tops_raw.index.isin(train_wells)]
df_tops_valid = df_tops_raw.loc[df_tops_raw.index.isin(val_wells)]

print(f"  Train wells={len(train_wells)}  Val wells={len(val_wells)}  Test wells={len(df_tops_test)}")
print(f"  Train rows={len(df_log_train):,}  Val rows={len(df_log_valid):,}  Test rows={len(df_logs_test):,}")

# ── Helpers ────────────────────────────────────────────────────────────────────
GLOBAL_DEPTH_MIN = df_logs_raw[DEPTH_COL].min()
GLOBAL_DEPTH_MAX = df_logs_raw[DEPTH_COL].max()

def well_log_cleaning(wn, df_log):
    df = df_log[df_log[WELL_COL] == wn][[DEPTH_COL, 'GR']].copy()
    if df.empty: return df
    df = df.sort_values(DEPTH_COL).reset_index(drop=True)
    # Mask the sentinel and out-of-range GR values
    df['GR'] = df['GR'].replace(GR_SENTINEL, np.nan)
    df.loc[df['GR'] < GR_CLIP_LO, 'GR'] = np.nan
    df.loc[df['GR'] > GR_CLIP_HI, 'GR'] = np.nan
    df['GR_CLEAN'] = df['GR'].ffill().bfill().fillna(df['GR'].median())
    df['DEPTH_NORM'] = ((df[DEPTH_COL] - GLOBAL_DEPTH_MIN) /
                        (GLOBAL_DEPTH_MAX - GLOBAL_DEPTH_MIN))
    return df.dropna(subset=[DEPTH_COL])


def step_function(s, e, tops):
    if not tops or s >= e: return [(s, 1)]
    if not all(tops[i] < tops[i+1] for i in range(len(tops)-1)): return [(s, 1)]
    depths = np.arange(s, e + 0.25, 0.5); vals = []; cur = 1
    for d in depths:
        for t in list(tops):
            if abs(d - t) < 0.26:
                cur += 1; tops = [x for x in tops if x != t]; break
        vals.append((round(d, 4), cur))
    return vals


def create_graph(df, sv):
    n = min(len(df), len(sv))
    if n == 0: return dgl.graph(([], []))
    df = df.iloc[:n]; sv = sv[:n]
    g  = dgl.graph((list(range(n-1)), list(range(1, n))))
    g.ndata['features'] = torch.tensor(df[INPUT_FEATURES].values.astype(np.float32))
    g.ndata['pos']      = torch.tensor(df[[DEPTH_COL]].values.astype(np.float32))
    g.ndata['label']    = torch.tensor([t[1] for t in sv], dtype=torch.int64)
    return g


def assign_labels(depths, sorted_tops):
    """Label each depth with the class of the deepest top at or above it.

    `sorted_tops` is a depth-sorted list of (marker, depth). Callers crop the well
    to start at the first top, so every depth is at or below it."""
    labels = []
    for d in depths:
        active = sorted_tops[0][0]
        for marker, top_depth in sorted_tops:
            if d >= top_depth: active = marker
            else: break
        labels.append(MARKER_TO_CLASS[active])
    return labels


class WellGraphDataset(Dataset):
    def __init__(self, df_log, df_tops, max_wells=None):
        lw = set(df_log[WELL_COL].unique())
        tw = set(df_tops.index)
        self.wells = sorted(lw & tw)[:max_wells] if max_wells else sorted(lw & tw)
        self.df_log = df_log; self.df_tops = df_tops
        # Building a graph (cleaning, step function, DGL) is slow and the result is
        # deterministic and seed-independent, so each well is built once and cached.
        self._cache = {}
        print(f"  {len(self.wells)} wells")

    def __len__(self): return len(self.wells)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        g = self._build(idx)
        self._cache[idx] = g
        return g

    def _build(self, idx):
        w  = self.wells[idx]
        df = well_log_cleaning(w, self.df_log)
        if df.empty: return self._empty()

        tops_row   = self.df_tops.loc[w].dropna()
        well_tops  = tops_row.to_dict()
        if not well_tops: return self._empty()  # no known top, skip the well
        sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
        tops_list   = [v for _, v in sorted_tops]

        df = df[df[DEPTH_COL] >= min(tops_list)].reset_index(drop=True)  # drop everything above the first top
        if df.empty: return self._empty()

        sv = step_function(float(df[DEPTH_COL].min()),
                           float(df[DEPTH_COL].max()), sorted(tops_list))
        g  = create_graph(df, sv)
        if g.num_nodes() == 0: return self._empty()

        # Flag nodes within 15 ft of any formation top
        depths = g.ndata['pos'][:, 0].numpy()
        is_near_top = np.zeros(len(depths), dtype=bool)
        for t in tops_list:
            is_near_top |= (np.abs(depths - t) < 15.0)
        g.ndata['is_near_top'] = torch.tensor(is_near_top, dtype=torch.bool)

        g.ndata['label'] = torch.tensor(
            assign_labels(depths, sorted_tops), dtype=torch.int64)
        return g

    def _empty(self):
        # Must have zero nodes so collate() drops it; a placeholder node would end
        # up in training as a fake sample.
        return dgl.graph(([], []))


def collate(s):
    v = [g for g in s if g.num_nodes() > 0]
    return dgl.batch(v) if v else dgl.graph(([], []))


print("\nBuilding datasets...")
train_ds = WellGraphDataset(df_log_train, df_tops_train)
valid_ds = WellGraphDataset(df_log_valid, df_tops_valid)
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,  collate_fn=collate)
valid_dl = DataLoader(valid_ds, batch_size=4, shuffle=False, collate_fn=collate)

# ── Class weights ──────────────────────────────────────────────────────────────
print("Computing class weights from training labels...")
_counts = np.zeros(N_CLASSES, dtype=np.int64)
for g in train_ds:
    if g.num_nodes() == 0: continue
    for c in range(N_CLASSES):
        _counts[c] += int((g.ndata['label'].numpy() == c).sum())
for c, (n, cnt) in enumerate(zip(CLASS_NAMES, _counts)):
    print(f"  class {c} ({n:<12}): {cnt:>12,}")
_inv     = _counts.max() / np.where(_counts == 0, 1, _counts).astype(np.float32)
supports = torch.tensor(_inv, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=supports)
print(f"  Weights: { {n: f'{w:.2f}' for n, w in zip(CLASS_NAMES, _inv)} }")

# ── Models ─────────────────────────────────────────────────────────────────────
class GraphSAGEEmbed(nn.Module):
    def __init__(self, inf, h, nc, dropout=0.1):
        super().__init__()
        self.c1  = dglnn.SAGEConv(inf, h, 'mean')
        self.c2  = dglnn.SAGEConv(h,   h, 'mean')
        self.c3  = dglnn.SAGEConv(h,   h, 'mean')
        self.bn1 = nn.BatchNorm1d(h)
        self.bn2 = nn.BatchNorm1d(h)
        self.bn3 = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(h, nc)

    def embed(self, g, x):
        h = self.drop(F.relu(self.bn1(self.c1(g, x))))
        h = self.drop(F.relu(self.bn2(self.c2(g, h))))
        h = self.drop(F.relu(self.bn3(self.c3(g, h))))
        return h

    def forward(self, g, x): return self.fc(self.embed(g, x))


class ImprovedGNN(nn.Module):
    """Stacked GraphConv, BatchNorm, LeakyReLU and dropout, with an embed() method
    for Stage 2. Identical to ImprovedGNN in the North Sea script."""
    def __init__(self, inf, hidden_sizes, nc, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        prev = inf
        for h in hidden_sizes:
            self.convs.append(dglnn.GraphConv(prev, h))
            self.bns.append(nn.BatchNorm1d(h))
            prev = h
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(prev, nc)

    def embed(self, g, x):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = self.drop(F.leaky_relu(bn(conv(g, h))))
        return h

    def forward(self, g, x): return self.fc(self.embed(g, x))


def _build_stage1(arch_name, nc):
    if arch_name == 'GraphSAGE':
        return GraphSAGEEmbed(len(INPUT_FEATURES), 128, nc)
    elif arch_name == 'ImprovedGNN':
        return ImprovedGNN(len(INPUT_FEATURES), [128, 256, 256, 128, 64], nc)
    raise ValueError(f"Unknown arch: {arch_name}")


# ── Per-well fusion (uses the module-level stage1 and iso, rebuilt every repeat) ──
DETECTORS = [("IF only", "if")]
RARE_CLASSES = [0, 1, 2]   # every class is a formation


def _apply_fusion(s1_preds, depths, emb, score, threshold):
    """Correct boundaries in one well using its confident Stage 1 predictions."""
    is_anomaly     = score >= threshold
    fused          = s1_preds.copy()
    confident_mask = ~is_anomaly

    if is_anomaly.sum() > 0 and confident_mask.sum() > 0:
        conf_preds  = s1_preds[confident_mask]
        conf_depths = depths[confident_mask]
        class_depths = {c: conf_depths[conf_preds == c]
                        for c in range(N_CLASSES) if (conf_preds == c).sum() > 0}
        if len(class_depths) >= 2:
            sorted_cls = sorted(class_depths, key=lambda c: class_depths[c].mean())
            boundaries = [
                (class_depths[sorted_cls[i]].max() +
                 class_depths[sorted_cls[i+1]].min()) / 2
                for i in range(len(sorted_cls)-1)
            ]
            for i in np.where(is_anomaly)[0]:
                d = depths[i]
                cls_idx = len(sorted_cls) - 1
                for j, b in enumerate(boundaries):
                    if d < b: cls_idx = j; break
                fused[i] = sorted_cls[cls_idx]
    return fused


def fuse_well(well, df_log, df_tops, threshold, smooth=11):
    df = well_log_cleaning(well, df_log)
    if df.empty: return None

    well_tops   = df_tops.loc[well].dropna().to_dict() if well in df_tops.index else {}
    if not well_tops: return None  # no known top, nothing to evaluate
    sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
    tops_list   = [v for _, v in sorted_tops]

    df = df[df[DEPTH_COL] >= min(tops_list)].reset_index(drop=True)  # drop everything above the first top
    if df.empty: return None

    sv = step_function(float(df[DEPTH_COL].min()),
                       float(df[DEPTH_COL].max()), sorted(tops_list))
    g = create_graph(df, sv)
    if g.num_nodes() == 0: return None
    g = dgl.add_self_loop(g)

    depths = g.ndata['pos'][:, 0].numpy()
    g.ndata['label'] = torch.tensor(
        assign_labels(depths, sorted_tops), dtype=torch.int64)
    g = g.to(DEVICE)  # labels are assigned on CPU first

    stage1.eval()
    with torch.no_grad():
        feat     = g.ndata['features']
        logits   = stage1(g, feat).cpu().numpy()
        emb      = stage1.embed(g, feat).cpu().numpy()

    true_lbl = g.ndata['label'].cpu().numpy()

    # Smoothed Stage 1 predictions
    exp_l  = np.exp(logits - logits.max(1, keepdims=True))
    probs  = exp_l / exp_l.sum(1, keepdims=True)
    s_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_pred = s_probs.argmax(axis=1)

    # Anomaly scores
    if_sc = minmax_scale(-iso.score_samples(emb))
    if_sc = scipy.ndimage.uniform_filter1d(if_sc, size=smooth)

    scores = {'if': if_sc}
    fused  = {key: _apply_fusion(s1_pred, depths, emb, sc, threshold)
              for key, sc in scores.items()}

    return true_lbl, s1_pred, fused, depths, scores


# ── Training and evaluation loop: N_REPEATS runs per architecture ─────────────
ARCHS = ['GraphSAGE', 'ImprovedGNN']
H = 128; EPOCHS = 15; LR = 0.005
LABELS = list(range(N_CLASSES))

print(f"\n{'='*62}")
print(f"  {len(ARCHS)} arch(s) x {N_REPEATS} repeats, seeds {BASE_SEED}..{BASE_SEED+N_REPEATS-1}")
print(f"{'='*62}")

# One repeat can take 45+ minutes, so write the CSV header up front. The column
# order must match the `row` dict built below, because _write_carbon_row()
# archives the log if the header ever differs.
_CARBON_CSV_COLUMNS = (
    ["run_id", "arch", "seed", "rep", "timestamp", "machine", "git_sha",
     "n_epochs_s1", "train_time_s", "eval_time_s",
     "thresh_search_time_s", "final_eval_time_s",
     "n_eval_wells", "n_eval_samples", "ms_per_sample", "s_per_well",
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
            "train_time_s", "eval_time_s", "thresh_search_time_s", "final_eval_time_s",
            "ms_per_sample", "s_per_well",
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

        stage1 = _build_stage1(arch_name, N_CLASSES).to(DEVICE)

        # Stage 1 training (tracked)
        s1_run = f"stage1_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_s1 = _build_cc(s1_run); ct_s1 = _build_ct(EPOCHS, s1_run)
        opt = torch.optim.Adam(stage1.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        cc_s1.start(); t0 = time.time()
        try:
            for ep in range(EPOCHS):
                ct_s1.epoch_start()
                stage1.train(); tl = 0; preds = []; labs = []
                for bg in train_dl:
                    if bg.num_nodes() == 0: continue
                    bg = dgl.add_self_loop(bg).to(DEVICE); opt.zero_grad()
                    lbl = torch.clamp(bg.ndata['label'], 0, N_CLASSES-1)
                    out = stage1(bg, bg.ndata['features'])
                    loss = criterion(out, lbl)
                    if torch.isnan(loss): continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(stage1.parameters(), 1.0)
                    opt.step(); tl += loss.item()
                    preds += out.argmax(1).tolist(); labs += lbl.tolist()
                ct_s1.epoch_end()
                sch.step()
                acc = accuracy_score(labs, preds)
                f1  = f1_score(labs, preds, average='weighted', zero_division=0)
                print(f"    ep {ep+1:2d}/{EPOCHS}  loss={tl/max(len(train_dl),1):.4f}"
                      f"  acc={acc:.4f}  f1={f1:.4f}")
        finally:
            try: ct_s1.stop()
            except Exception: pass
            try: cc_s1.stop()
            except Exception: pass
        elapsed_s1 = time.time() - t0
        s1_path = Path(CKPT_DIR) / f"stage1_colorado_{arch_key}_seed{seed}.pt"
        torch.save(stage1.state_dict(), s1_path)

        # Stage 2: Isolation Forest (tracked)
        if_run = f"stage2a_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_if = _build_cc(if_run); ct_if = _build_ct(1, if_run)
        cc_if.start(); ct_if.epoch_start(); t0_if = time.time()
        try:
            stage1.eval(); embeds = []; is_near_top = []
            with torch.no_grad():
                for bg in train_dl:
                    if bg.num_nodes() == 0: continue
                    bg  = dgl.add_self_loop(bg).to(DEVICE)
                    emb = stage1.embed(bg, bg.ndata['features']).cpu().numpy()
                    embeds.append(emb)
                    nt = bg.ndata.get('is_near_top', torch.zeros(bg.num_nodes(), dtype=torch.bool))
                    is_near_top.append(nt.cpu().numpy())
            embeds      = np.vstack(embeds)
            is_near_top = np.concatenate(is_near_top)
            iso = IsolationForest(n_estimators=200, contamination=0.05,
                                  random_state=seed, n_jobs=-1)
            iso.fit(embeds[~is_near_top])
        finally:
            ct_if.epoch_end()
            try: ct_if.stop()
            except Exception: pass
            try: cc_if.stop()
            except Exception: pass
        elapsed_if = time.time() - t0_if
        print(f"    IF: {len(embeds):,} total, {is_near_top.sum():,} near-top, "
              f"{(~is_near_top).sum():,} interior")

        # Evaluation (tracked separately). The threshold search makes 16 extra passes
        # over the validation wells, so it is timed apart from the final pass to keep
        # the per-sample latency meaningful.
        eval_run = f"eval_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_eval = _build_cc(eval_run)
        cc_eval.start(); t0_eval = time.time()
        try:
            best_thresh = {key: 0.5 for _, key in DETECTORS}
            best_f1     = {key: 0.0 for _, key in DETECTORS}
            t0_thresh = time.time()
            for thresh in np.arange(0.05, 0.85, 0.05):
                buckets = {key: {'true': [], 'fused': []} for _, key in DETECTORS}
                for w in valid_ds.wells:
                    res = fuse_well(w, df_log_valid, df_tops_valid, thresh)
                    if res is None: continue
                    tl, _, fused, _, _ = res
                    for _, key in DETECTORS:
                        buckets[key]['true'].extend(tl)
                        buckets[key]['fused'].extend(fused[key])
                for _, key in DETECTORS:
                    b = buckets[key]
                    mf1 = f1_score(b['true'], b['fused'], average='macro', zero_division=0)
                    if mf1 > best_f1[key]: best_f1[key] = mf1; best_thresh[key] = thresh
            thresh_search_time_s = time.time() - t0_thresh

            all_true = []; all_s1 = []; all_fused = {key: [] for _, key in DETECTORS}
            n_eval_wells = 0
            t0_final = time.time()
            for w in valid_ds.wells:
                res = fuse_well(w, df_log_valid, df_tops_valid, best_thresh['if'])
                if res is None: continue
                tl, s1, fused, _, _ = res
                all_true.extend(tl); all_s1.extend(s1)
                for _, key in DETECTORS:
                    all_fused[key].extend(fused[key])
                n_eval_wells += 1
            final_eval_time_s = time.time() - t0_final
        finally:
            try: cc_eval.stop()
            except Exception: pass
        elapsed_eval = time.time() - t0_eval

        all_true = np.array(all_true); all_s1 = np.array(all_s1)
        for key in all_fused: all_fused[key] = np.array(all_fused[key])

        n_eval_samples = len(all_true)
        ms_per_sample = (final_eval_time_s * 1000 / n_eval_samples) if n_eval_samples > 0 else float("nan")
        s_per_well    = (final_eval_time_s / n_eval_wells) if n_eval_wells > 0 else float("nan")

        perf = {"stage1_acc": accuracy_score(all_true, all_s1),
                "stage1_wf1": f1_score(all_true, all_s1, average='weighted', zero_division=0),
                "stage1_mf1": f1_score(all_true, all_s1, average='macro', zero_division=0),
                "stage1_recall": recall_score(all_true, all_s1, average='weighted', zero_division=0)}
        for _, key in DETECTORS:
            perf[f"{key}_thresh"] = float(best_thresh[key])
            perf[f"{key}_acc"] = accuracy_score(all_true, all_fused[key])
            perf[f"{key}_wf1"] = f1_score(all_true, all_fused[key], average='weighted', zero_division=0)
            perf[f"{key}_mf1"] = f1_score(all_true, all_fused[key], average='macro', zero_division=0)
            perf[f"{key}_recall"] = recall_score(all_true, all_fused[key], average='weighted', zero_division=0)
        print(f"    [perf] stage1 acc={perf['stage1_acc']:.4f}  "
              + "  ".join(f"{key}: acc={perf[f'{key}_acc']:.4f} mf1={perf[f'{key}_mf1']:.4f} thr={perf[f'{key}_thresh']:.2f}"
                           for _, key in DETECTORS))
        print(f"    [inference] {n_eval_wells} wells / {n_eval_samples:,} samples  "
              f"final_eval={final_eval_time_s:.2f}s  ms/sample={ms_per_sample:.5f}  s/well={s_per_well:.4f}  "
              f"(thresh_search={thresh_search_time_s:.1f}s, excluded from the per-sample numbers above)")

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
            "thresh_search_time_s": float(thresh_search_time_s),
            "final_eval_time_s": float(final_eval_time_s),
            "n_eval_wells": int(n_eval_wells), "n_eval_samples": int(n_eval_samples),
            "ms_per_sample": float(ms_per_sample), "s_per_well": float(s_per_well),
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
    print(classification_report(all_true, all_s1, target_names=CLASS_NAMES,
                                 labels=LABELS, zero_division=0))
    for name, key in DETECTORS:
        print(f"\n── {arch_name}: Stage 1 + {name} (last rep) {'─'*(38-len(name))}")
        print(classification_report(all_true, all_fused[key],
              target_names=CLASS_NAMES, labels=LABELS, zero_division=0))

    arch_results[arch_name] = {
        'true': all_true, 's1': all_s1, 'fused': all_fused,
        'best_thresh': best_thresh, 'n_reps_valid': len(df_arch_ok),
        'co2_total_mean_g': float(means['cc_total_emissions_g']),
        'co2_total_std_g':  float(stds['cc_total_emissions_g']),
    }

# ── Comparison across architectures (last repeat of each) ──────────────────────
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

# ── Confusion matrices (one row per architecture: Stage 1, Stage 1 + IF) ───────
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
        ax.set_title(f"{arch_name} — {title}", fontsize=10)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=30)
plt.suptitle('Colorado: GraphSAGE vs ImprovedGNN (last rep, validation set)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CONF_PATH, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix → {CONF_PATH}")
print(f"Carbon log → {CARBON_CSV}")

# ── Carbon charts ───────────────────────────────────────────────────────────────
df_rep_all = pd.DataFrame(rep_rows)
df_ok_all = df_rep_all[df_rep_all["grid_ok"]]
if len(df_ok_all):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))

    ax = axes[0]
    g = df_ok_all.groupby("arch")[["ct_emissions_g", "cc_emissions_g"]].agg(['mean', 'std'])
    x = np.arange(len(g)); w = 0.35
    ax.bar(x - w/2, g[("ct_emissions_g", "mean")], w, yerr=g[("ct_emissions_g", "std")], capsize=4, label="CT (GPU)")
    ax.bar(x + w/2, g[("cc_emissions_g", "mean")], w, yerr=g[("cc_emissions_g", "std")], capsize=4, label="CC (CPU+GPU+RAM)")
    ax.set_xticks(x); ax.set_xticklabels(g.index)
    ax.set_ylabel("Training emissions (gCO2eq)"); ax.set_title("Training emissions: CT vs CC"); ax.legend(fontsize=8)

    ax = axes[1]
    g2 = df_ok_all.groupby("arch")[["cc_cpu_kwh", "cc_gpu_kwh", "cc_ram_kwh"]].mean() * 1000
    bottom = np.zeros(len(g2))
    for col, label, color in [("cc_cpu_kwh", "CPU", "#4472C4"), ("cc_gpu_kwh", "GPU", "#70AD47"), ("cc_ram_kwh", "RAM", "#FFC000")]:
        ax.bar(g2.index, g2[col], bottom=bottom, label=label, color=color)
        bottom += g2[col].values
    ax.set_ylabel("Energy (Wh)"); ax.set_title("CodeCarbon energy breakdown"); ax.legend(fontsize=8)

    ax = axes[2]
    g3 = df_ok_all.groupby("arch")[["cc_emissions_g", "cc_eval_emissions_g"]].mean()
    ax.bar(g3.index, g3["cc_emissions_g"], label="Training", color="#4472C4")
    ax.bar(g3.index, g3["cc_eval_emissions_g"], bottom=g3["cc_emissions_g"], label="Inference", color="#FF7F0E")
    ax.set_ylabel("Emissions (gCO2eq)"); ax.set_title("Training vs inference emissions (stacked)"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(CARBON_FIG_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Carbon charts → {CARBON_FIG_PATH}")

with open(RES_PATH, 'wb') as f:
    pickle.dump({'arch_results': arch_results, 'rep_rows': rep_rows, 'class_names': CLASS_NAMES}, f)
print(f"Results → {RES_PATH}")
