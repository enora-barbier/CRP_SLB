"""
GNN + Isolation Forest formation-boundary detection on the North Sea wells,
with carbon tracking, for GraphSAGE and ImprovedGNN.

Stage 1: node-level GNN classifier over six formations.
Stage 2: Isolation Forest on the node embeddings, used to correct boundaries.

Carbon tracking follows north_sea_carbon_tracking.ipynb:
  - CodeCarbon runs offline pinned to Norway; CarbonTracker reads live grid
    intensity from Electricity Maps.
  - Training and evaluation are measured with separate trackers.
  - N_REPEATS seeds per architecture, reported as mean +/- std.
  - Any run whose implied intensity falls outside Norway's ~5-80 g/kWh band is
    flagged (a failed country pin) and excluded from the aggregates.
  - Results are appended to a CSV that is archived if its columns ever change.

Environment variables: CKPT_DIR, OUT_DIR, N_REPEATS, BASE_SEED, ARCHS_OVERRIDE,
ELECTRICITYMAPS_API_KEY.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "codecarbon>=2.3.0"])

import warnings, os, re, time, pickle, uuid, socket, random
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
from codecarbon import OfflineEmissionsTracker
from carbontracker.tracker import CarbonTracker
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Models and graph batches are moved to this device explicitly with .to(DEVICE);
# neither PyTorch nor DGL does it for you.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR   = os.environ.get("CKPT_DIR", "/home/jupyter/stage2_ckpt")
CARBON_DIR = Path(CKPT_DIR) / "carbon_logs"
os.makedirs(CKPT_DIR, exist_ok=True)
CARBON_DIR.mkdir(exist_ok=True)

print(f"DEVICE     : {DEVICE}" + ("" if torch.cuda.is_available() else "  (no CUDA GPU detected -- running on CPU)"))

MACHINE = socket.gethostname()
try:
    GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    GIT_SHA = "unknown"

# Carbon accounting is pinned to Norway: CodeCarbon runs offline with an explicit
# country code, and CarbonTracker reads live grid intensity from Electricity Maps
# when ELECTRICITYMAPS_API_KEY is set. Both read the GPU through NVML;
# CarbonTracker covers the GPU only, CodeCarbon covers CPU, GPU and RAM.
CARBON_COUNTRY_ISO3 = "NOR"
_ELECTRICITYMAPS_KEY = os.environ.get("ELECTRICITYMAPS_API_KEY")
_ct_api_keys = {"electricitymaps": _ELECTRICITYMAPS_KEY} if _ELECTRICITYMAPS_KEY else None
if _ct_api_keys is None:
    print("  [carbon] ELECTRICITYMAPS_API_KEY not set in the environment — CarbonTracker "
          "will fall back to its built-in average grid-intensity estimate instead of "
          "live data. Set the env var to match the notebook's methodology exactly.")

# One independent run per seed, 15 by default as in the LSTM carbon notebook;
# N_REPEATS=1 gives a quick smoke test.
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
    """Return dict with energy_kwh, cpu_kwh, gpu_kwh, ram_kwh, emissions_kg, country, region."""
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
    """Percentage spread between CarbonTracker and CodeCarbon readings."""
    cc_g = cc_emissions_kg * 1000 if not np.isnan(cc_emissions_kg) else np.nan
    if np.isnan(ct_co2_g) or np.isnan(cc_g) or (ct_co2_g + cc_g) == 0:
        return np.nan
    return 100.0 * abs(cc_g - ct_co2_g) / (0.5 * (ct_co2_g + cc_g))

def _grid_tag(g_per_kwh):
    """Label an implied gCO2/kWh value as Norway-like or not. This catches a failed
    country pin, e.g. CodeCarbon falling back to IP geolocation."""
    if np.isnan(g_per_kwh):       return "?? (no data)"
    if 5   <= g_per_kwh <= 80:    return "Norway-like  OK"
    if 200 <= g_per_kwh <= 320:   return "Netherlands-like  FAILED PIN"
    if 350 <= g_per_kwh <= 480:   return "USA-like  FAILED PIN"
    return f"unknown grid (~{g_per_kwh:.0f} g/kWh)  NOT Norway"

def _build_cc(run_name: str) -> OfflineEmissionsTracker:
    return OfflineEmissionsTracker(
        country_iso_code=CARBON_COUNTRY_ISO3,
        project_name=f"gnn_stage2_{run_name}",
        output_dir=str(CARBON_DIR),
        output_file=f"{run_name}_codecarbon.csv",
        measure_power_secs=5,
        tracking_mode="machine",
        log_level="error",
        save_to_file=True,
    )

def _build_ct(n_epochs: int, run_name: str) -> CarbonTracker:
    # monitor_epochs=1 and epochs_before_pred=1 make CarbonTracker log a prediction
    # after the first epoch and the actual figure at the end, so a reading exists
    # even if a run stops early.
    return CarbonTracker(
        epochs=n_epochs, monitor_epochs=1, epochs_before_pred=1,
        log_dir=str(CARBON_DIR), log_file_prefix=run_name + "_",
        api_keys=_ct_api_keys, verbose=1, ignore_errors=True,
    )

def _write_carbon_row(row: dict, csv_path: Path):
    """Append one row to csv_path. If the existing log has different columns it is
    renamed to a .bak.csv first, so the file never ends up with ragged rows."""
    df1 = pd.DataFrame([row])
    if csv_path.exists():
        with open(csv_path) as f:
            hdr = f.readline().rstrip("\n").split(",")
        if hdr != list(df1.columns):
            bak = csv_path.with_name(f"{csv_path.stem}__pre_{datetime.now():%Y%m%d_%H%M%S}.bak.csv")
            csv_path.rename(bak)
            print(f"  [log] schema changed — archived old log -> {bak.name}")
    df1.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

# One row is appended per repeat
CARBON_CSV = Path(CKPT_DIR) / "stage2_carbon_log.csv"

# ── Data ──────────────────────────────────────────────────────────────────
base_dir = "."
df_log_train  = pd.read_parquet(os.path.join(base_dir, "log_train.parquet"))
df_tops_train = pd.read_parquet(os.path.join(base_dir, "tops_train.parquet"))
df_log_valid  = pd.read_parquet(os.path.join(base_dir, "log_valid.parquet"))
df_tops_valid = pd.read_parquet(os.path.join(base_dir, "tops_valid.parquet"))

MARKER_COLS     = ['DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
INPUT_FEATURES  = ['GR_CLEAN', 'RHOB_CLEAN', 'DEEPRES_CLEAN', 'DEPTH_NORM']
# There is no background class. Each well is cropped to start at its shallowest
# known top (see WellGraphDataset.__getitem__ and fuse_well), so every node has a
# formation label, and wells with no known tops are skipped.
N_CLASSES      = len(MARKER_COLS)   # 6
MARKER_TO_CLASS = {'DUNLIN':0,'HORDALAND':1,'ROGALAND':2,
                   'SHETLAND':3,'VESTLAND':4,'VIKING':5}
CLASS_NAMES     = ['DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
CLASS_COLORS    = ['#4472C4','#70AD47','#FFC000','#FF4D4D','#7030A0','#8B4513']
df_tops_train   = df_tops_train[MARKER_COLS]
df_tops_valid   = df_tops_valid[MARKER_COLS]
print("Data loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────
def well_log_cleaning(wn, df_log, feats, depth_col='DS_INDEX', well_col='well_name'):
    existing_feats = [f for f in feats if f in df_log.columns]
    df = df_log[df_log[well_col]==wn][existing_feats + [depth_col]].copy()
    if df.empty: return df
    df = df.sort_values(depth_col).reset_index(drop=True)
    df[existing_feats] = df[existing_feats].ffill().bfill().fillna(df[existing_feats].median())
    GLOBAL_MIN = 100.16
    GLOBAL_MAX = 20377.16
    df['DEPTH_NORM'] = (df[depth_col] - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    return df.dropna(subset=[depth_col])

def step_function(s, e, tops):
    if not tops or s>=e: return [(s,1)]
    if not all(tops[i]<tops[i+1] for i in range(len(tops)-1)): return [(s,1)]
    depths=np.arange(s,e+0.25,0.5); vals=[]; cur=1
    for d in depths:
        for t in list(tops):
            if abs(d-t)<0.26: cur+=1; tops=[x for x in tops if x!=t]; break
        vals.append((round(d,4),cur))
    return vals

def create_graph(df, sv, feats=INPUT_FEATURES):
    n=min(len(df),len(sv))
    if n==0: return dgl.graph(([],[]))
    df=df.iloc[:n]; sv=sv[:n]
    g=dgl.graph((list(range(n-1)),list(range(1,n))))
    g.ndata['features']=torch.tensor(df[feats].values.astype(np.float32))
    g.ndata['pos']=torch.tensor(df[['DS_INDEX']].values.astype(np.float32))
    g.ndata['label']=torch.tensor([t[1] for t in sv],dtype=torch.int64)
    return g

class WellGraphDataset(Dataset):
    def __init__(self, df_log, df_tops, feats, max_wells=None, with_marker=False):
        lw=set(df_log['well_name'].unique()); tw=set(df_tops.index)
        self.wells=sorted(lw&tw)[:max_wells] if max_wells else sorted(lw&tw)
        self.df_log=df_log; self.df_tops=df_tops
        self.feats=feats; self.with_marker=with_marker
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
        w=self.wells[idx]
        df=well_log_cleaning(w, self.df_log, self.feats)
        if df.empty: return self._empty()
        tops=self.df_tops.loc[w].dropna().tolist() if w in self.df_tops.index else []
        if not tops: return self._empty()  # no known top, skip the well
        df = df[df['DS_INDEX'] >= min(tops)].reset_index(drop=True)  # drop everything above the first top
        if df.empty: return self._empty()
        sv=step_function(float(df['DS_INDEX'].min()),float(df['DS_INDEX'].max()),sorted(tops))
        g=create_graph(df, sv)
        if g.num_nodes()==0: return self._empty()
        depths = g.ndata['pos'][:, 0].numpy()
        is_near_top = np.zeros(len(depths), dtype=bool)
        for t in tops:
            is_near_top |= (np.abs(depths - t) < 15.0)
        g.ndata['is_near_top'] = torch.tensor(is_near_top, dtype=torch.bool)
        tops_series = self.df_tops.loc[w] if w in self.df_tops.index else pd.Series()
        well_tops = tops_series.dropna().to_dict()
        sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
        fixed_labels = []
        for d in depths:
            active_marker = sorted_tops[0][0]  # the crop above guarantees d >= the first top
            for marker_name, top_depth in sorted_tops:
                if d >= top_depth: active_marker = marker_name
                else: break
            fixed_labels.append(MARKER_TO_CLASS[active_marker])
        g.ndata['label'] = torch.tensor(fixed_labels, dtype=torch.int64)
        if self.with_marker and 'marker_label' in self.df_log.columns:
            raw=self.df_log[self.df_log['well_name']==w].sort_values('DS_INDEX')
            ml=raw['marker_label'].values[:g.num_nodes()]
            if len(ml)<g.num_nodes():
                ml=np.pad(ml,(0,g.num_nodes()-len(ml)),'edge')
            g.ndata['marker_label']=torch.tensor(
                [MARKER_TO_CLASS.get(l, 0) for l in ml], dtype=torch.int64)
        return g
    def _empty(self):
        # Must have zero nodes so collate() drops it; a placeholder node would end
        # up in training as a fake sample.
        return dgl.graph(([],[]))

def collate(s):
    v=[g for g in s if g.num_nodes()>0]
    return dgl.batch(v) if v else dgl.graph(([],[]))

print("\nBuilding datasets...")
train_ds = WellGraphDataset(df_log_train,df_tops_train,INPUT_FEATURES,with_marker=True)
valid_ds = WellGraphDataset(df_log_valid,df_tops_valid,INPUT_FEATURES)
train_dl = DataLoader(train_ds,batch_size=4,shuffle=True, collate_fn=collate)
valid_dl = DataLoader(valid_ds,batch_size=4,shuffle=False,collate_fn=collate)

# ── Formation depth priors (median top over training wells only) ──────────
PRIOR_TOPS = {col: float(df_tops_train[col].median())
              for col in MARKER_COLS if df_tops_train[col].notna().sum() > 0}
PRIOR_SORTED_TOPS = sorted(PRIOR_TOPS.items(), key=lambda x: x[1])
print(f"  Formation depth priors: { {k: f'{v:.0f}ft' for k,v in PRIOR_SORTED_TOPS} }")

# ── Models ────────────────────────────────────────────────────────────────
class GraphSAGEEmbed(nn.Module):
    """Three GraphSAGE layers, with an embed() method for Stage 2."""
    def __init__(self, inf, h, nc, dropout=0.1):
        super().__init__()
        self.c1=dglnn.SAGEConv(inf,h,'mean'); self.c2=dglnn.SAGEConv(h,h,'mean')
        self.c3=dglnn.SAGEConv(h,h,'mean')
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.bn3=nn.BatchNorm1d(h)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(h,nc)
    def embed(self, g, x):
        h=self.drop(F.relu(self.bn1(self.c1(g,x))))
        h=self.drop(F.relu(self.bn2(self.c2(g,h))))
        h=self.drop(F.relu(self.bn3(self.c3(g,h))))
        return h
    def forward(self, g, x): return self.fc(self.embed(g, x))

class ImprovedGNN(nn.Module):
    """Stacked GraphConv, BatchNorm, LeakyReLU and dropout, with an embed() method
    for Stage 2."""
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

class VGAE(nn.Module):
    def __init__(self,inf,hidden,latent):
        super().__init__()
        self.enc1  =dglnn.GraphConv(inf,hidden)
        self.mu    =dglnn.GraphConv(hidden,latent)
        self.logvar=dglnn.GraphConv(hidden,latent)
        self.dec   =nn.Sequential(nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,inf))
    def encode(self,g,x):
        h=F.relu(self.enc1(g,x)); return self.mu(g,h), self.logvar(g,h)
    def reparameterise(self,mu,lv):
        return mu+torch.exp(0.5*lv)*torch.randn_like(mu) if self.training else mu
    def forward(self,g,x):
        mu,lv=self.encode(g,x); z=self.reparameterise(mu,lv); x_hat=self.dec(z)
        recon=F.mse_loss(x_hat,x,reduction='none').mean(dim=1)
        kl=-0.5*(1+lv-mu**2-lv.exp()).mean(dim=1)
        return recon+0.01*kl, z.detach()

def _build_stage1(arch_name, nc):
    if arch_name == 'GraphSAGE':
        return GraphSAGEEmbed(len(INPUT_FEATURES), 128, nc)
    elif arch_name == 'ImprovedGNN':
        return ImprovedGNN(len(INPUT_FEATURES), [128, 256, 256, 128, 64], nc)
    raise ValueError(f"Unknown arch: {arch_name}")

# ── Hyperparameters and class weights (shared by both architectures) ─────
EPOCHS=15; LR=0.005; NC=N_CLASSES

print("Computing class weights from training label frequencies...")
_counts = np.zeros(N_CLASSES, dtype=np.int64)
for _g in train_ds:
    if _g.num_nodes() == 0: continue
    _lbl = _g.ndata['label'].numpy()
    for _c in range(N_CLASSES): _counts[_c] += int((_lbl == _c).sum())
for _c, (_n, _cnt) in enumerate(zip(CLASS_NAMES, _counts)):
    print(f"  class {_c} ({_n:12s}): {_cnt:>10,}")
_inv = _counts.max() / np.where(_counts == 0, 1, _counts).astype(np.float32)
supports  = torch.tensor(_inv, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=supports)
print(f"  Weights: { {n: f'{w:.2f}' for n,w in zip(CLASS_NAMES, _inv)} }")

# ── Per-well fusion (uses the module-level stage1 and iso, rebuilt every repeat) ──
def fuse_well(well, df_log, df_tops, threshold, smooth=11):
    df = well_log_cleaning(well, df_log, INPUT_FEATURES)
    if df.empty: return None
    tops = df_tops.loc[well].dropna().tolist() if well in df_tops.index else []
    if not tops: return None  # no known top, nothing to evaluate
    df = df[df['DS_INDEX'] >= min(tops)].reset_index(drop=True)  # drop everything above the first top
    if df.empty: return None
    shetland_depth = None
    if well in df_tops.index and 'SHETLAND' in df_tops.columns:
        shet_val = df_tops.loc[well, 'SHETLAND']
        if not pd.isna(shet_val): shetland_depth = shet_val
    sv = step_function(float(df['DS_INDEX'].min()), float(df['DS_INDEX'].max()), sorted(tops))
    g = create_graph(df, sv)
    if g.num_nodes() == 0: return None
    g = dgl.add_self_loop(g)
    well_tops = df_tops.loc[well].dropna().to_dict() if well in df_tops.index else {}
    sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
    depths = g.ndata['pos'][:, 0].numpy()
    fixed_labels = []
    for d in depths:
        active_marker = sorted_tops[0][0]  # the crop above guarantees d >= the first top
        for marker_name, top_depth in sorted_tops:
            if d >= top_depth: active_marker = marker_name
            else: break
        fixed_labels.append(MARKER_TO_CLASS[active_marker])
    g.ndata['label'] = torch.tensor(fixed_labels, dtype=torch.int64)
    g = g.to(DEVICE)  # labels are assigned on CPU first
    # stage1 is moved back to the CPU after every repeat to free GPU memory, and
    # plot_well() calls this after the loop, so put the model back on DEVICE here.
    stage1.to(DEVICE)
    stage1.eval()
    with torch.no_grad():
        feat   = g.ndata['features']
        logits = stage1(g, feat).cpu().numpy()
        emb    = stage1.embed(g, feat).cpu().numpy()
    true_lbl = g.ndata['label'].cpu().numpy()
    exp_l        = np.exp(logits - logits.max(1, keepdims=True))
    probs        = exp_l / exp_l.sum(1, keepdims=True)
    smooth_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_preds     = smooth_probs.argmax(axis=1)
    if_sc = minmax_scale(-iso.score_samples(emb))
    combo = scipy.ndimage.uniform_filter1d(if_sc, size=smooth)
    local_mean = scipy.ndimage.uniform_filter1d(combo, size=201)
    is_anomaly = combo >= (local_mean + 0.15)
    class_depths   = {c: depths[s1_preds==c] for c in range(NC) if (s1_preds==c).sum()>0}
    sorted_classes = sorted(class_depths, key=lambda c: class_depths[c].mean())
    boundaries = [(class_depths[sorted_classes[i]].max() + class_depths[sorted_classes[i+1]].min()) / 2
                  for i in range(len(sorted_classes)-1)]
    fused = s1_preds.copy()
    for i in np.where(is_anomaly)[0]:
        d = depths[i]
        if shetland_depth is not None and d < shetland_depth: continue
        class_idx = len(sorted_classes) - 1
        for j, b in enumerate(boundaries):
            if d < b: class_idx = j; break
        fused[i] = sorted_classes[class_idx]
    return true_lbl, s1_preds, fused, depths, combo

def _spans(depths, mask):
    spans=[]; start=None
    for i, m in enumerate(mask):
        if m and start is None: start=depths[i]
        elif not m and start is not None:
            spans.append((start,depths[i-1])); start=None
    if start is not None: spans.append((start,depths[-1]))
    return spans

def plot_well(well, df_log, df_tops, threshold, smooth=11, out_path=None):
    result = fuse_well(well, df_log, df_tops, threshold, smooth)
    if result is None: print(f"No data for well {well}"); return None
    true_lbl, s1_preds, fused, depths, combo = result
    df_raw   = well_log_cleaning(well, df_log, INPUT_FEATURES)
    log_vals = df_raw[INPUT_FEATURES].values[:len(depths)]
    tops     = df_tops.loc[well].dropna().tolist() if well in df_tops.index else []
    fig, axes = plt.subplots(1, 7, figsize=(22, 14), sharey=True)
    for i, (fname, color) in enumerate(zip(INPUT_FEATURES, ['#2ca02c','#1f77b4','#d62728','#ff7f0e'])):
        ax = axes[i]
        ax.plot(log_vals[:, i], depths, color=color, lw=0.6)
        ax.set_title(fname, fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.2)
        for t in tops: ax.axhline(t, color='red', lw=1, ls='--', alpha=0.6)
    for ax, preds, title in [(axes[3], true_lbl, 'True'),
                              (axes[4], s1_preds, 'Stage 1'),
                              (axes[5], fused,    'Fused')]:
        for c in range(NC):
            for s, e in _spans(depths, preds==c):
                ax.fill_betweenx([s,e], 0, 1, color=CLASS_COLORS[c], alpha=0.85)
        ax.set_xlim(0,1); ax.set_xticks([]); ax.set_title(title, fontsize=8)
        for t in tops: ax.axhline(t, color='red', lw=1, ls='--', alpha=0.7)
    ax = axes[6]
    ax.fill_betweenx(depths, 0, combo, where=combo>=threshold, color='red', alpha=0.35)
    ax.plot(combo, depths, 'k-', lw=0.6)
    ax.axvline(threshold, color='orange', lw=1.2, ls='--')
    ax.set_xlim(0,1); ax.set_title('Anomaly', fontsize=8); ax.grid(True, alpha=0.2)
    for t in tops: ax.axhline(t, color='red', lw=1, ls='--', alpha=0.6)
    axes[0].set_ylabel('Depth', fontsize=9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in range(NC)],
               loc='lower center', ncol=NC, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle(f'{well}  |  thresh={threshold:.2f}', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    out = out_path or f'{CKPT_DIR}/well_{well}_stage2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Plot saved → {out}")
    return out

# ── Training and evaluation loop: N_REPEATS runs per architecture ───────────
# ARCHS_OVERRIDE runs a subset, e.g. ARCHS_OVERRIDE="GraphSAGE" BASE_SEED=5
# N_REPEATS=10 to finish GraphSAGE from seed 5 onward.
_archs_override = os.environ.get("ARCHS_OVERRIDE")
ARCHS = [a.strip() for a in _archs_override.split(",")] if _archs_override else ['GraphSAGE', 'ImprovedGNN']
arch_results    = {}
all_carbon_rows = []

for arch_name in ARCHS:
    print(f"\n{'='*62}")
    print(f"  ARCHITECTURE: {arch_name}   ({N_REPEATS} repeats, seeds {BASE_SEED}..{BASE_SEED+N_REPEATS-1})")
    print(f"{'='*62}")

    arch_key = arch_name.lower()
    rep_rows = []

    for rep in range(N_REPEATS):
        seed = BASE_SEED + rep
        print(f"\n  --- {arch_name}  rep {rep+1}/{N_REPEATS}  seed={seed} ---")
        set_seed(seed)

        stage1 = _build_stage1(arch_name, NC).to(DEVICE)
        opt = torch.optim.Adam(stage1.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        # Stage 1 training (tracked)
        s1_run = f"stage1_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_s1 = _build_cc(s1_run)
        ct_s1 = _build_ct(EPOCHS, s1_run)
        cc_s1.start(); t0 = time.time()
        try:
            for ep in range(EPOCHS):
                ct_s1.epoch_start()
                stage1.train(); tl=0; preds=[]; labs=[]
                for bg in train_dl:
                    if bg.num_nodes()==0: continue
                    bg=dgl.add_self_loop(bg).to(DEVICE); opt.zero_grad()
                    lbl=torch.clamp(bg.ndata['label'],0,NC-1)
                    out=stage1(bg,bg.ndata['features']); loss=criterion(out,lbl)
                    if torch.isnan(loss): continue
                    loss.backward(); torch.nn.utils.clip_grad_norm_(stage1.parameters(),1.0)
                    opt.step(); tl+=loss.item(); preds+=out.argmax(1).tolist(); labs+=lbl.tolist()
                ct_s1.epoch_end()
                acc=accuracy_score(labs,preds); f1=f1_score(labs,preds,average='weighted',zero_division=0)
                print(f"    ep {ep+1:2d}/{EPOCHS}  loss={tl/max(len(train_dl),1):.4f}  acc={acc:.4f}  f1={f1:.4f}")
                sch.step()
        finally:
            try: ct_s1.stop()
            except Exception: pass
            try: cc_s1.stop()
            except Exception: pass
        elapsed_s1 = time.time() - t0

        s1_path = Path(CKPT_DIR) / f"stage1_{arch_key}_seed{seed}.pt"
        torch.save(stage1.state_dict(), s1_path)

        # Stage 2: Isolation Forest (tracked)
        stage1.eval(); embeds=[]; is_near_top_list=[]
        with torch.no_grad():
            for bg in train_dl:
                if bg.num_nodes()==0: continue
                bg=dgl.add_self_loop(bg).to(DEVICE)
                emb=stage1.embed(bg,bg.ndata['features']).cpu().numpy()
                embeds.append(emb)
                is_near_top_list.append(bg.ndata['is_near_top'].cpu().numpy()
                                         if 'is_near_top' in bg.ndata
                                         else np.zeros(emb.shape[0], bool))
        embeds = np.vstack(embeds); is_near_top = np.concatenate(is_near_top_list)

        if_run = f"stage2a_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_if = _build_cc(if_run); ct_if = _build_ct(1, if_run)
        cc_if.start(); ct_if.epoch_start(); t0_if = time.time()
        try:
            iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed, n_jobs=-1)
            iso.fit(embeds[~is_near_top])
        finally:
            ct_if.epoch_end()
            try: ct_if.stop()
            except Exception: pass
            try: cc_if.stop()
            except Exception: pass
        elapsed_if = time.time() - t0_if

        # Evaluation (tracked separately, as in the notebook's train/inference split).
        # The threshold search makes extra passes over the validation wells, so it is
        # timed apart from the final pass to keep the per-sample latency meaningful.
        eval_run = f"eval_{arch_key}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_eval = _build_cc(eval_run)
        cc_eval.start(); t0_eval = time.time()
        try:
            best_thresh=0.5; best_f1=0
            t0_thresh = time.time()
            for thresh in np.arange(0.30, 0.85, 0.05):
                _at=[]; _af=[]
                for w in valid_ds.wells:
                    res=fuse_well(w, df_log_valid, df_tops_valid, thresh)
                    if res is None: continue
                    tl,_,fp,_,_ = res; _at.extend(tl); _af.extend(fp)
                wf1 = f1_score(_at, _af, average='weighted', zero_division=0)
                if wf1>best_f1: best_f1=wf1; best_thresh=thresh
            thresh_search_time_s = time.time() - t0_thresh

            all_true=[]; all_s1=[]; all_fused=[]; n_eval_wells=0
            t0_final = time.time()
            for w in valid_ds.wells:
                res=fuse_well(w, df_log_valid, df_tops_valid, best_thresh)
                if res is None: continue
                tl,s1p,fp,_,_ = res
                all_true.extend(tl); all_s1.extend(s1p); all_fused.extend(fp); n_eval_wells+=1
            final_eval_time_s = time.time() - t0_final
        finally:
            try: cc_eval.stop()
            except Exception: pass
        elapsed_eval = time.time() - t0_eval

        all_true=np.array(all_true); all_s1=np.array(all_s1); all_fused=np.array(all_fused)
        n_eval_samples = len(all_true)
        ms_per_sample = (final_eval_time_s * 1000 / n_eval_samples) if n_eval_samples > 0 else float("nan")
        s_per_well    = (final_eval_time_s / n_eval_wells) if n_eval_wells > 0 else float("nan")

        stage1_acc = accuracy_score(all_true, all_s1)
        stage1_wf1 = f1_score(all_true, all_s1, average='weighted', zero_division=0)
        stage1_mf1 = f1_score(all_true, all_s1, average='macro', zero_division=0)
        stage1_recall = recall_score(all_true, all_s1, average='weighted', zero_division=0)
        if_acc = accuracy_score(all_true, all_fused)
        if_wf1 = f1_score(all_true, all_fused, average='weighted', zero_division=0)
        if_mf1 = f1_score(all_true, all_fused, average='macro', zero_division=0)
        if_recall = recall_score(all_true, all_fused, average='weighted', zero_division=0)

        # Read both trackers back
        ct_kwh_s1, ct_co2_s1 = _parse_carbontracker_log(CARBON_DIR, prefix=s1_run + "_")
        cc_blob_s1   = _read_codecarbon_csv(CARBON_DIR / f"{s1_run}_codecarbon.csv")
        ct_kwh_if, ct_co2_if = _parse_carbontracker_log(CARBON_DIR, prefix=if_run + "_")
        cc_blob_if   = _read_codecarbon_csv(CARBON_DIR / f"{if_run}_codecarbon.csv")
        cc_blob_eval = _read_codecarbon_csv(CARBON_DIR / f"{eval_run}_codecarbon.csv")

        cc_train_kwh = np.nansum([cc_blob_s1["energy_kwh"], cc_blob_if["energy_kwh"]])
        cc_train_g   = np.nansum([cc_blob_s1["emissions_kg"], cc_blob_if["emissions_kg"]]) * 1000
        cc_eval_g    = np.nan_to_num(cc_blob_eval["emissions_kg"], nan=0.0) * 1000
        cc_total_g   = cc_train_g + cc_eval_g
        ct_train_kwh = np.nansum([ct_kwh_s1, ct_kwh_if])
        ct_train_g   = np.nansum([ct_co2_s1, ct_co2_if])

        # Grid-intensity check from the measured numbers, not the tracker's metadata
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
        print(f"    [perf]  {n_eval_wells} wells / {n_eval_samples:,} samples  "
              f"stage1 acc={stage1_acc:.4f}  if: acc={if_acc:.4f} mf1={if_mf1:.4f} thresh={best_thresh:.2f}  "
              f"train={elapsed_s1+elapsed_if:.1f}s  eval={elapsed_eval:.1f}s  "
              f"ms/sample={ms_per_sample:.5f}  s/well={s_per_well:.4f}")

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
            "stage1_acc": float(stage1_acc), "stage1_wf1": float(stage1_wf1), "stage1_mf1": float(stage1_mf1),
            "stage1_recall": float(stage1_recall),
            "if_thresh": float(best_thresh),
            "if_acc": float(if_acc), "if_wf1": float(if_wf1), "if_mf1": float(if_mf1),
            "if_recall": float(if_recall),
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
        rep_rows.append(row)
        _write_carbon_row(row, CARBON_CSV)
        all_carbon_rows.append(row)

        stage1.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()

    # Mean +/- std over this architecture's repeats (Norway-like grid rows only)
    df_rep = pd.DataFrame(rep_rows)
    bad = df_rep[~df_rep["grid_ok"]]
    if len(bad):
        print(f"\n  ! {len(bad)}/{len(df_rep)} rep(s) NOT Norway-like (failed country pin) "
              f"— excluded from the aggregate below.")
    df_ok = df_rep[df_rep["grid_ok"]] if len(bad) < len(df_rep) else df_rep

    AGG_COLS = ["ct_energy_kwh","ct_emissions_g","cc_energy_kwh","cc_emissions_g",
                "cc_cpu_kwh","cc_gpu_kwh","cc_ram_kwh","cc_eval_energy_kwh",
                "cc_eval_emissions_g","cc_total_emissions_g","tracker_spread_pct",
                "train_time_s","eval_time_s","thresh_search_time_s","final_eval_time_s",
                "ms_per_sample","s_per_well",
                "stage1_acc","stage1_wf1","stage1_mf1","stage1_recall",
                "if_acc","if_wf1","if_mf1","if_recall"]
    means = df_ok[AGG_COLS].mean(); stds = df_ok[AGG_COLS].std()
    print(f"\n  {arch_name} — {len(df_ok)}/{N_REPEATS} valid rep(s), mean +/- std:")
    for c in AGG_COLS:
        print(f"    {c:<24} {means[c]:>10.5f}  +/- {stds[c]:>9.5f}")

    # Reports below use the last repeat's model and predictions
    # (fuse_well and plot_well read `stage1` and `iso` as module-level names).
    LABELS=list(range(N_CLASSES))
    print(f"\n── {arch_name} Stage 1 only ──────────────────────────────────────")
    print(classification_report(all_true, all_s1, target_names=CLASS_NAMES, labels=LABELS, zero_division=0))
    print(f"\n── {arch_name} Stage 1 + Anomaly (last rep) ──────────────────────")
    print(classification_report(all_true, all_fused, target_names=CLASS_NAMES, labels=LABELS, zero_division=0))

    print(f"\n── {arch_name} Rare-class summary (last rep) ─────────────────────")
    print(f"{'Class':<12} {'S1 F1':>7} {'S1 Rec':>8} {'Fused F1':>10} {'Fused Rec':>10} {'Delta F1':>10}")
    print("-"*58)
    for ci,cn in [(MARKER_TO_CLASS['DUNLIN'],'DUNLIN'), (MARKER_TO_CLASS['VIKING'],'VIKING')]:
        s1f=f1_score(all_true==ci, all_s1==ci,   average='binary', zero_division=0)
        fuf=f1_score(all_true==ci, all_fused==ci, average='binary', zero_division=0)
        s1r=recall_score(all_true==ci, all_s1==ci,   zero_division=0)
        fur=recall_score(all_true==ci, all_fused==ci, zero_division=0)
        arrow='↑' if fuf>s1f else ('↓' if fuf<s1f else '=')
        print(f"{cn:<12} {s1f:>7.4f} {s1r:>8.4f} {fuf:>10.4f} {fur:>10.4f} {arrow} {abs(fuf-s1f):>8.4f}")

    arch_results[arch_name] = {
        'true': all_true, 's1': all_s1, 'fused': all_fused,
        'best_thresh': best_thresh,
        'co2': {'total_mean_g': float(means['cc_total_emissions_g']),
                'total_std_g':  float(stds['cc_total_emissions_g'])},
        'n_reps_valid': len(df_ok), 'carbon_rows': rep_rows,
    }

    # Example well plot from the last repeat's model
    plot_well(valid_ds.wells[0], df_log_valid, df_tops_valid, best_thresh,
              out_path=f'{CKPT_DIR}/well_{valid_ds.wells[0]}_{arch_key}_stage2.png')

# ── Comparison across architectures ───────────────────────────────────────
print("\n\n" + "="*70)
print("  COMBINED ARCHITECTURE COMPARISON  (perf: last repeat | carbon: mean of valid reps)")
print("="*70)
print(f"{'Arch':<14} {'Model':<16} {'Acc':>6} {'wF1':>6} {'Rec':>6}  {'CO2 total(g) mean+/-std':>28}  {'n_reps':>6}")
print("-"*90)
for arch_name, res in arch_results.items():
    for label, preds in [('Stage1', res['s1']), ('Stage1+Anomaly', res['fused'])]:
        at  = res['true']
        acc = accuracy_score(at, preds)
        wf1 = f1_score(at, preds, average='weighted', zero_division=0)
        wr  = recall_score(at, preds, average='weighted', zero_division=0)
        if label == 'Stage1+Anomaly':
            co2_str = f"{res['co2']['total_mean_g']:.5f} +/- {res['co2']['total_std_g']:.5f}"
        else:
            co2_str = "—"
        print(f"  {arch_name:<12} {label:<16} {acc:>6.4f} {wf1:>6.4f} {wr:>6.4f}  {co2_str:>28}  {res['n_reps_valid']:>6}")

# ── Confusion matrices (2 architectures x Stage 1 / Stage 1 + IF) ──────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
for row_idx, (arch_name, res) in enumerate(arch_results.items()):
    for col_idx, (preds, title) in enumerate([
        (res['s1'],    f"{arch_name} — Stage 1 only"),
        (res['fused'], f"{arch_name} — Stage 1 + Anomaly (thresh={res['best_thresh']:.2f})")
    ]):
        ax = axes[row_idx][col_idx]
        cm = confusion_matrix(res['true'], preds, labels=list(range(N_CLASSES)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=30)
plt.suptitle('GraphSAGE vs ImprovedGNN — Stage1 and Stage1+Anomaly (last repeat)', fontsize=13, fontweight='bold')
plt.tight_layout()
out_fig = f'{CKPT_DIR}/stage2_confusion_comparison.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nConfusion matrix comparison → {out_fig}")
print(f"Carbon log → {CARBON_CSV}")

# ── Carbon charts (same layout as the LSTM carbon-tracking notebook) ───────
df_carbon_all = pd.DataFrame(all_carbon_rows)
if len(df_carbon_all):
    df_ok_all = df_carbon_all[df_carbon_all["grid_ok"]]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1. Training emissions per architecture, CarbonTracker vs CodeCarbon
    ax = axes[0]
    g = df_ok_all.groupby("arch")[["ct_emissions_g", "cc_emissions_g"]].agg(['mean', 'std'])
    x = np.arange(len(g)); w = 0.35
    ax.bar(x - w/2, g[("ct_emissions_g","mean")], w, yerr=g[("ct_emissions_g","std")],
           label="CarbonTracker (GPU)", capsize=3)
    ax.bar(x + w/2, g[("cc_emissions_g","mean")], w, yerr=g[("cc_emissions_g","std")],
           label="CodeCarbon (CPU+GPU+RAM)", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(g.index); ax.set_ylabel("Training emissions (gCO2eq)")
    ax.set_title("Training emissions: CT vs CC"); ax.legend(fontsize=8)

    # 2. CodeCarbon energy split into CPU, GPU and RAM
    ax = axes[1]
    g2 = df_ok_all.groupby("arch")[["cc_cpu_kwh", "cc_gpu_kwh", "cc_ram_kwh"]].mean() * 1000
    bottom = np.zeros(len(g2))
    for col, label, color in [("cc_cpu_kwh","CPU","#4472C4"), ("cc_gpu_kwh","GPU","#70AD47"), ("cc_ram_kwh","RAM","#FFC000")]:
        ax.bar(g2.index, g2[col], bottom=bottom, label=label, color=color)
        bottom += g2[col].values
    ax.set_ylabel("Energy (Wh)"); ax.set_title("CodeCarbon energy breakdown"); ax.legend(fontsize=8)

    # 3. Training vs inference emissions, stacked
    ax = axes[2]
    g3 = df_ok_all.groupby("arch")[["cc_emissions_g", "cc_eval_emissions_g"]].mean()
    ax.bar(g3.index, g3["cc_emissions_g"], label="Training", color="#4472C4")
    ax.bar(g3.index, g3["cc_eval_emissions_g"], bottom=g3["cc_emissions_g"], label="Inference", color="#FF7F0E")
    ax.set_ylabel("Emissions (gCO2eq)"); ax.set_title("Training vs inference emissions"); ax.legend(fontsize=8)

    plt.tight_layout()
    out_carbon_fig = f'{CKPT_DIR}/stage2_carbon_charts.png'
    plt.savefig(out_carbon_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Carbon charts → {out_carbon_fig}")

    print("\nPer-arch carbon summary (mean +/- std over Norway-like reps):")
    summary = (df_ok_all.groupby("arch")
               .agg(n_reps=("run_id","count"),
                    cc_emissions_g_mean=("cc_emissions_g","mean"), cc_emissions_g_std=("cc_emissions_g","std"),
                    cc_eval_emissions_g_mean=("cc_eval_emissions_g","mean"), cc_eval_emissions_g_std=("cc_eval_emissions_g","std"),
                    cc_total_emissions_g_mean=("cc_total_emissions_g","mean"), cc_total_emissions_g_std=("cc_total_emissions_g","std"),
                    tracker_spread_pct_mean=("tracker_spread_pct","mean")))
    print(summary.round(4).to_string())

# ── Save combined results ──────────────────────────────────────────────────
with open(f'{CKPT_DIR}/stage2_results.pkl','wb') as f:
    pickle.dump({'arch_results': arch_results, 'all_carbon_rows': all_carbon_rows}, f)
