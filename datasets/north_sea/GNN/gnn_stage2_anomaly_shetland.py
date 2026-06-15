"""
Stage 2 Anomaly Detection — dual carbon tracking (CarbonTracker + CodeCarbon),
matching the methodology of experiment_carbon_tracking.ipynb.
Runs both GraphSAGE and ImprovedGNN and compares results.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "codecarbon>=2.3.0"])

import warnings, os, re, time, pickle, uuid, socket
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
from codecarbon import EmissionsTracker
from carbontracker.tracker import CarbonTracker
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CKPT_DIR   = "/home/jupyter/stage2_ckpt"
CARBON_DIR = Path(CKPT_DIR) / "carbon_logs"
os.makedirs(CKPT_DIR, exist_ok=True)
CARBON_DIR.mkdir(exist_ok=True)

MACHINE = socket.gethostname()
# CI is read dynamically from the CodeCarbon CSV after each run (auto-detected by region).
# Fallback only used if CC CSV is missing.
CI_G_PER_KWH_FALLBACK = 28.0  # gCO2/kWh — Norway (~95% hydropower)

# ── Carbon helper functions (from experiment_carbon_tracking.ipynb) ───────
_RE_ENERGY = re.compile(r"Energy:\s*([0-9eE+\-.]+)\s*kWh")
_RE_CO2    = re.compile(r"CO2eq:\s*([0-9eE+\-.]+)\s*g")

def _parse_carbontracker_log(prefix: str):
    """Parse the most recent {prefix}*_carbontracker_output.log. Returns (kwh, co2_g)."""
    candidates = sorted(CARBON_DIR.glob(f"{prefix}*_carbontracker_output.log"))
    if not candidates:
        return np.nan, np.nan
    text = candidates[-1].read_text(errors="ignore")
    energies = _RE_ENERGY.findall(text)
    co2s     = _RE_CO2.findall(text)
    if not energies or not co2s:
        return np.nan, np.nan
    try:
        return float(energies[-1]), float(co2s[-1])
    except Exception:
        return np.nan, np.nan

def _read_codecarbon_csv(run_name: str):
    """Return dict with energy_kwh, cpu_kwh, gpu_kwh, ram_kwh, emissions_kg, country, region, carbon_intensity."""
    path = CARBON_DIR / f"{run_name}_codecarbon.csv"
    empty = {"emissions_kg": np.nan, "energy_kwh": np.nan,
             "cpu_kwh": np.nan, "gpu_kwh": np.nan, "ram_kwh": np.nan,
             "country": "", "region": "", "carbon_intensity": np.nan}
    if not path.exists():
        return empty
    df = pd.read_csv(path)
    if df.empty:
        return empty
    row = df.iloc[-1]
    emissions_kg = float(row.get("emissions",       np.nan))
    energy_kwh   = float(row.get("energy_consumed", np.nan))
    ci = float(row.get("emissions_rate", np.nan))
    if np.isnan(ci) and not np.isnan(emissions_kg) and not np.isnan(energy_kwh) and energy_kwh > 0:
        ci = (emissions_kg * 1000) / energy_kwh
    return {
        "emissions_kg":     emissions_kg,
        "energy_kwh":       energy_kwh,
        "cpu_kwh":          float(row.get("cpu_energy",      np.nan)),
        "gpu_kwh":          float(row.get("gpu_energy",      np.nan)),
        "ram_kwh":          float(row.get("ram_energy",      np.nan)),
        "country":          str(row.get("country_iso_code",  "")),
        "region":           str(row.get("region",            "")),
        "carbon_intensity": ci,
    }

def _tracker_spread(ct_co2_g, cc_emissions_kg):
    """Percentage spread between CarbonTracker and CodeCarbon readings."""
    cc_g = cc_emissions_kg * 1000 if not np.isnan(cc_emissions_kg) else np.nan
    if np.isnan(ct_co2_g) or np.isnan(cc_g) or (ct_co2_g + cc_g) == 0:
        return np.nan
    return 100.0 * abs(cc_g - ct_co2_g) / (0.5 * (ct_co2_g + cc_g))

def _build_cc(run_name: str) -> EmissionsTracker:
    kwargs = dict(
        project_name=f"gnn_stage2_{run_name}",
        output_dir=str(CARBON_DIR),
        output_file=f"{run_name}_codecarbon.csv",
        measure_power_secs=5,
        tracking_mode="machine",
        log_level="error",
        save_to_file=True,
    )
    try:
        return EmissionsTracker(**kwargs, country_iso_code="NOR")
    except TypeError:
        print("  [carbon] country_iso_code not supported — using auto-detect")
        return EmissionsTracker(**kwargs)

def _build_ct(n_epochs: int, run_name: str) -> CarbonTracker:
    return CarbonTracker(
        epochs=n_epochs,
        monitor_epochs=n_epochs,
        epochs_before_pred=0,
        log_dir=str(CARBON_DIR),
        log_file_prefix=run_name + "_",
        verbose=1,
    )

# Incremental CSV log (same schema as experiment_carbon_tracking.ipynb)
CARBON_CSV = Path(CKPT_DIR) / "stage2_carbon_log.csv"

# ── Data ──────────────────────────────────────────────────────────────────
base_dir = "."
df_log_train  = pd.read_parquet(os.path.join(base_dir, "log_train.parquet"))
df_tops_train = pd.read_parquet(os.path.join(base_dir, "tops_train.parquet"))
df_log_valid  = pd.read_parquet(os.path.join(base_dir, "log_valid.parquet"))
df_tops_valid = pd.read_parquet(os.path.join(base_dir, "tops_valid.parquet"))

MARKER_COLS     = ['DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
INPUT_FEATURES  = ['GR_CLEAN', 'RHOB_CLEAN', 'DEEPRES_CLEAN', 'DEPTH_NORM']
N_CLASSES       = len(MARKER_COLS) + 1   # 7
MARKER_TO_CLASS = {'background':0,'DUNLIN':1,'HORDALAND':2,
                   'ROGALAND':3,'SHETLAND':4,'VESTLAND':5,'VIKING':6}
CLASS_NAMES     = ['background','DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
CLASS_COLORS    = ['#d0d0d0','#4472C4','#70AD47','#FFC000','#FF4D4D','#7030A0','#8B4513']
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
        print(f"  {len(self.wells)} wells")
    def __len__(self): return len(self.wells)
    def __getitem__(self, idx):
        w=self.wells[idx]
        df=well_log_cleaning(w, self.df_log, self.feats)
        if df.empty: return self._empty()
        tops=self.df_tops.loc[w].dropna().tolist() if w in self.df_tops.index else []
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
            active_marker = 'background'
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
        g=dgl.graph(([],[])); g.add_nodes(1)
        g.ndata['features']=torch.zeros(1,len(self.feats))
        g.ndata['pos']=torch.zeros(1,1)
        g.ndata['label']=torch.tensor([0],dtype=torch.int64)
        g.ndata['is_near_top']=torch.tensor([False],dtype=torch.bool)
        if self.with_marker:
            g.ndata['marker_label']=torch.tensor([0],dtype=torch.int64)
        return g

def collate(s):
    v=[g for g in s if g.num_nodes()>0]
    return dgl.batch(v) if v else dgl.graph(([],[]))

print("\nBuilding datasets...")
train_ds = WellGraphDataset(df_log_train,df_tops_train,INPUT_FEATURES,with_marker=True)
valid_ds = WellGraphDataset(df_log_valid,df_tops_valid,INPUT_FEATURES)
train_dl = DataLoader(train_ds,batch_size=4,shuffle=True, collate_fn=collate)
valid_dl = DataLoader(valid_ds,batch_size=4,shuffle=False,collate_fn=collate)

# ── Formation depth priors from TRAINING wells only ───────────────────────
PRIOR_TOPS = {col: float(df_tops_train[col].median())
              for col in MARKER_COLS if df_tops_train[col].notna().sum() > 0}
PRIOR_SORTED_TOPS = sorted(PRIOR_TOPS.items(), key=lambda x: x[1])
print(f"  Formation depth priors: { {k: f'{v:.0f}ft' for k,v in PRIOR_SORTED_TOPS} }")

# ── Models ────────────────────────────────────────────────────────────────
class GraphSAGEEmbed(nn.Module):
    """3-layer GraphSAGE with embed() hook for Stage 2."""
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
    """Stacked GraphConv with LeakyReLU, BN, dropout, and an embed() hook for Stage 2."""
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

# ── Shared hyperparams & class weights (same for all archs) ───────────────
EPOCHS=12; LR=0.005; NC=N_CLASSES

print("Computing class weights from training label frequencies...")
_counts = np.zeros(N_CLASSES, dtype=np.int64)
for _g in train_ds:
    if _g.num_nodes() == 0: continue
    _lbl = _g.ndata['label'].numpy()
    for _c in range(N_CLASSES): _counts[_c] += int((_lbl == _c).sum())
for _c, (_n, _cnt) in enumerate(zip(CLASS_NAMES, _counts)):
    print(f"  class {_c} ({_n:12s}): {_cnt:>10,}")
_inv = _counts.max() / np.where(_counts == 0, 1, _counts).astype(np.float32)
supports  = torch.tensor(_inv, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=supports)
print(f"  Weights: { {n: f'{w:.2f}' for n,w in zip(CLASS_NAMES, _inv)} }")

# ── Per-well fusion (uses module-level stage1, vgae, iso — reassigned per arch) ──
def fuse_well(well, df_log, df_tops, threshold, smooth=11):
    df = well_log_cleaning(well, df_log, INPUT_FEATURES)
    if df.empty: return None
    tops = df_tops.loc[well].dropna().tolist() if well in df_tops.index else []
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
        active_marker = 'background'
        for marker_name, top_depth in sorted_tops:
            if d >= top_depth: active_marker = marker_name
            else: break
        fixed_labels.append(MARKER_TO_CLASS[active_marker])
    g.ndata['label'] = torch.tensor(fixed_labels, dtype=torch.int64)
    stage1.eval()
    with torch.no_grad():
        feat   = g.ndata['features']
        logits = stage1(g, feat).cpu().numpy()
        emb    = stage1.embed(g, feat).cpu().numpy()
    true_lbl = g.ndata['label'].numpy()
    exp_l        = np.exp(logits - logits.max(1, keepdims=True))
    probs        = exp_l / exp_l.sum(1, keepdims=True)
    smooth_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_preds     = smooth_probs.argmax(axis=1)
    last_formation = None
    cleaned_preds = s1_preds.copy()
    for i in range(len(cleaned_preds)):
        if cleaned_preds[i] != 0: last_formation = cleaned_preds[i]
        elif cleaned_preds[i] == 0 and last_formation is not None:
            cleaned_preds[i] = last_formation
    s1_preds = cleaned_preds
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
    out = out_path or f'/home/jupyter/stage2_ckpt/well_{well}_stage2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Plot saved → {out}")
    return out

# ── Per-architecture sweep ────────────────────────────────────────────────
ARCHS = ['GraphSAGE', 'ImprovedGNN']
arch_results    = {}
all_carbon_rows = []

for arch_name in ARCHS:
    print(f"\n{'='*62}")
    print(f"  ARCHITECTURE: {arch_name}")
    print(f"{'='*62}")

    arch_key  = arch_name.lower()
    S1_PATH   = os.path.join(CKPT_DIR, f"stage1_{arch_key}.pt")
    IF_PATH   = os.path.join(CKPT_DIR, f"isolation_forest_{arch_key}.pkl")
    VGAE_PATH = os.path.join(CKPT_DIR, f"vgae_{arch_key}.pt")

    co2_s1=co2_if=0.0
    elapsed_s1=elapsed_if=0.0
    _carbon_rows = []

    stage1 = _build_stage1(arch_name, NC)

    # ── Stage 1: train or load ─────────────────────────────────────────────
    if os.path.exists(S1_PATH):
        stage1.load_state_dict(torch.load(S1_PATH))
        print(f"  Stage 1 loaded from {S1_PATH}  (no carbon tracked)")
    else:
        print(f"\n  Training {arch_name} ({EPOCHS} epochs, lr={LR})")
        opt = torch.optim.Adam(stage1.parameters(), lr=LR)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        _s1_run = f"stage1_{arch_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_s1 = _build_cc(_s1_run)
        ct_s1 = _build_ct(EPOCHS, _s1_run)
        cc_s1.start(); t0 = time.time()

        _s1_co2_prev = 0.0
        for ep in range(EPOCHS):
            ep_t0 = time.time()
            ct_s1.epoch_start()
            stage1.train(); tl=0; preds=[]; labs=[]
            for bg in train_dl:
                if bg.num_nodes()==0: continue
                bg=dgl.add_self_loop(bg); opt.zero_grad()
                lbl=torch.clamp(bg.ndata['label'],0,NC-1)
                out=stage1(bg,bg.ndata['features']); loss=criterion(out,lbl)
                if torch.isnan(loss): continue
                loss.backward(); torch.nn.utils.clip_grad_norm_(stage1.parameters(),1.0)
                opt.step(); tl+=loss.item(); preds+=out.argmax(1).tolist(); labs+=lbl.tolist()
            ct_s1.epoch_end()
            acc=accuracy_score(labs,preds)
            f1 =f1_score(labs,preds,average='weighted',zero_division=0)
            _s1_co2_cum = (cc_s1.flush() or 0.0) * 1000
            _s1_co2_ep  = _s1_co2_cum - _s1_co2_prev
            _s1_co2_prev = _s1_co2_cum
            ep_dt = time.time() - ep_t0
            print(f"  ep {ep+1:2d}/{EPOCHS}  loss={tl/max(len(train_dl),1):.4f}  acc={acc:.4f}  f1={f1:.4f}"
                  f"  t={ep_dt:.1f}s  cc_co2={_s1_co2_ep:.5f}g (cum={_s1_co2_cum:.5f}g)")
            sch.step()

        try: ct_s1.stop()
        except Exception: pass
        cc_s1.stop()
        elapsed_s1 = time.time() - t0
        ct_kwh_s1, ct_co2_s1 = _parse_carbontracker_log(_s1_run)
        cc_blob_s1 = _read_codecarbon_csv(_s1_run)
        co2_s1 = np.nan_to_num(cc_blob_s1["emissions_kg"], nan=0.0) * 1000
        spread_s1 = _tracker_spread(ct_co2_s1, cc_blob_s1["emissions_kg"])
        print(f"\n  {arch_name} Stage 1:")
        print(f"    CC → {co2_s1:.5f} gCO2eq  |  {cc_blob_s1['energy_kwh']:.5f} kWh"
              f"  (CPU {cc_blob_s1['cpu_kwh']:.5f} / GPU {cc_blob_s1['gpu_kwh']:.5f} / RAM {cc_blob_s1['ram_kwh']:.5f} kWh)")
        print(f"    CT → {ct_co2_s1:.5f} gCO2eq  |  {ct_kwh_s1:.5f} kWh  spread={spread_s1:.1f}%  t={elapsed_s1:.1f}s")
        _row = {"run_id": str(uuid.uuid4()), "component": f"Stage1_{arch_name}",
                "arch": arch_name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "machine": MACHINE, "n_epochs": EPOCHS, "train_time_s": int(elapsed_s1),
                "ct_energy_kwh": ct_kwh_s1, "ct_emissions_g": ct_co2_s1,
                "cc_emissions_kg": cc_blob_s1["emissions_kg"], "cc_emissions_g": co2_s1,
                "cc_energy_kwh": cc_blob_s1["energy_kwh"],
                "cc_cpu_kwh": cc_blob_s1["cpu_kwh"], "cc_gpu_kwh": cc_blob_s1["gpu_kwh"],
                "cc_ram_kwh": cc_blob_s1["ram_kwh"],
                "cc_country": cc_blob_s1["country"], "cc_region": cc_blob_s1["region"],
                "carbon_intensity": cc_blob_s1["carbon_intensity"],
                "tracker_spread_pct": spread_s1}
        _carbon_rows.append(_row)
        pd.DataFrame([_row]).to_csv(CARBON_CSV, mode="a", header=not CARBON_CSV.exists(), index=False)
        torch.save(stage1.state_dict(), S1_PATH)
        print(f"  Saved → {S1_PATH}")

    # ── Stage 2a: Isolation Forest ─────────────────────────────────────────
    if os.path.exists(IF_PATH):
        with open(IF_PATH,'rb') as f: iso = pickle.load(f)
        print(f"  IF loaded from {IF_PATH}")
    else:
        print(f"\n  Extracting {arch_name} embeddings...")
        stage1.eval(); embeds=[]; is_near_top_list=[]
        with torch.no_grad():
            for bg in train_dl:
                if bg.num_nodes()==0: continue
                bg=dgl.add_self_loop(bg)
                emb=stage1.embed(bg,bg.ndata['features']).cpu().numpy()
                embeds.append(emb)
                if 'is_near_top' in bg.ndata:
                    is_near_top_list.append(bg.ndata['is_near_top'].cpu().numpy())
                else:
                    is_near_top_list.append(np.zeros(emb.shape[0],bool))
        embeds=np.vstack(embeds); is_near_top=np.concatenate(is_near_top_list)
        print(f"  Total: {len(embeds):,}  Near-top: {is_near_top.sum():,}  Stable: {(~is_near_top).sum():,}")

        _if_run = f"stage2a_{arch_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cc_if = _build_cc(_if_run); ct_if = _build_ct(1, _if_run)
        cc_if.start(); ct_if.epoch_start(); t0 = time.time()
        print("  Training Isolation Forest...")
        iso=IsolationForest(n_estimators=200,contamination=0.05,random_state=42,n_jobs=-1)
        iso.fit(embeds[~is_near_top])
        ct_if.epoch_end()
        try: ct_if.stop()
        except Exception: pass
        cc_if.stop()
        elapsed_if = time.time() - t0
        ct_kwh_if, ct_co2_if = _parse_carbontracker_log(_if_run)
        cc_blob_if = _read_codecarbon_csv(_if_run)
        co2_if = np.nan_to_num(cc_blob_if["emissions_kg"], nan=0.0) * 1000
        spread_if = _tracker_spread(ct_co2_if, cc_blob_if["emissions_kg"])
        print(f"  CC → {co2_if:.5f} gCO2eq  |  CT → {ct_co2_if:.5f} gCO2eq  spread={spread_if:.1f}%  t={elapsed_if:.1f}s")
        _row = {"run_id": str(uuid.uuid4()), "component": f"Stage2a_IF_{arch_name}",
                "arch": arch_name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "machine": MACHINE, "n_epochs": 1, "train_time_s": int(elapsed_if),
                "ct_energy_kwh": ct_kwh_if, "ct_emissions_g": ct_co2_if,
                "cc_emissions_kg": cc_blob_if["emissions_kg"], "cc_emissions_g": co2_if,
                "cc_energy_kwh": cc_blob_if["energy_kwh"],
                "cc_cpu_kwh": cc_blob_if["cpu_kwh"], "cc_gpu_kwh": cc_blob_if["gpu_kwh"],
                "cc_ram_kwh": cc_blob_if["ram_kwh"],
                "cc_country": cc_blob_if["country"], "cc_region": cc_blob_if["region"],
                "carbon_intensity": cc_blob_if["carbon_intensity"],
                "tracker_spread_pct": spread_if}
        _carbon_rows.append(_row)
        pd.DataFrame([_row]).to_csv(CARBON_CSV, mode="a", header=not CARBON_CSV.exists(), index=False)
        with open(IF_PATH,'wb') as f: pickle.dump(iso,f)
        print(f"  Saved → {IF_PATH}")

    # Stage 2b VGAE skipped — anomaly score uses Isolation Forest only.

    # ── Threshold search ───────────────────────────────────────────────────
    print(f"\n  Threshold search ({arch_name})...")
    best_thresh=0.5; best_f1=0
    t_thresh = time.time()
    for thresh in np.arange(0.30, 0.85, 0.05):
        _at=[]; _af=[]
        for w in valid_ds.wells:
            res=fuse_well(w, df_log_valid, df_tops_valid, thresh)
            if res is None: continue
            tl,_,fp,_,_ = res; _at.extend(tl); _af.extend(fp)
        wf1 = f1_score(_at, _af, average='weighted', zero_division=0)
        df1 = f1_score(_at, _af, labels=[1], average='macro', zero_division=0)
        vf1 = f1_score(_at, _af, labels=[6], average='macro', zero_division=0)
        print(f"  thresh={thresh:.2f}  overall_f1={wf1:.4f}  dunlin_f1={df1:.4f}  viking_f1={vf1:.4f}")
        if wf1>best_f1: best_f1=wf1; best_thresh=thresh
    elapsed_thresh = time.time() - t_thresh
    print(f"  Best: {best_thresh:.2f}  (F1={best_f1:.4f})  t={elapsed_thresh:.1f}s")

    # ── Final evaluation ───────────────────────────────────────────────────
    print(f"\n  Final evaluation ({arch_name})...")
    t_eval = time.time()
    all_true=[]; all_s1=[]; all_fused=[]; n_eval_wells=0
    for w in valid_ds.wells:
        res=fuse_well(w, df_log_valid, df_tops_valid, best_thresh)
        if res is None: continue
        tl,s1p,fp,_,_ = res
        all_true.extend(tl); all_s1.extend(s1p); all_fused.extend(fp); n_eval_wells+=1
    elapsed_eval = time.time() - t_eval
    n_eval_samples  = len(all_true)
    time_per_sample = elapsed_eval / n_eval_samples if n_eval_samples > 0 else np.nan
    time_per_well   = elapsed_eval / n_eval_wells   if n_eval_wells   > 0 else np.nan
    print(f"  {n_eval_wells} wells  |  {n_eval_samples:,} samples  |  "
          f"time/sample={time_per_sample:.5f}s  |  time/well={time_per_well:.3f}s")

    all_true=np.array(all_true); all_s1=np.array(all_s1); all_fused=np.array(all_fused)
    LABELS=list(range(N_CLASSES))

    print(f"\n── {arch_name} Stage 1 only ──────────────────────────────────────")
    print(classification_report(all_true, all_s1, target_names=CLASS_NAMES, labels=LABELS, zero_division=0))
    print(f"\n── {arch_name} Stage 1 + Anomaly (IF+VGAE) ──────────────────────")
    print(classification_report(all_true, all_fused, target_names=CLASS_NAMES, labels=LABELS, zero_division=0))

    print(f"\n── {arch_name} Rare-class summary ───────────────────────────────")
    print(f"{'Class':<12} {'S1 F1':>7} {'S1 Rec':>8} {'Fused F1':>10} {'Fused Rec':>10} {'Delta F1':>10}")
    print("-"*58)
    for ci,cn in [(1,'DUNLIN'),(6,'VIKING')]:
        s1f=f1_score(all_true==ci, all_s1==ci,   average='binary', zero_division=0)
        fuf=f1_score(all_true==ci, all_fused==ci, average='binary', zero_division=0)
        s1r=recall_score(all_true==ci, all_s1==ci,   zero_division=0)
        fur=recall_score(all_true==ci, all_fused==ci, zero_division=0)
        arrow='↑' if fuf>s1f else ('↓' if fuf<s1f else '=')
        print(f"{cn:<12} {s1f:>7.4f} {s1r:>8.4f} {fuf:>10.4f} {fur:>10.4f} {arrow} {abs(fuf-s1f):>8.4f}")

    # ── Carbon summary (per arch) ──────────────────────────────────────────
    total_co2 = co2_s1 + co2_if
    def _rv(comp, key):
        for r in _carbon_rows:
            if r["component"]==comp: return r.get(key, np.nan)
        return np.nan
    def _f(v): return f"{v:.5f}" if not np.isnan(v) else "—"
    print(f"\n── {arch_name} Carbon & Timing ──────────────────────────────────")
    hdr = f"  {'Component':<30}  {'CC_gCO2':>9}  {'CC_kWh':>8}  {'CT_gCO2':>9}  {'CT_kWh':>8}  {'Spread%':>8}  {'Time(s)':>8}"
    print(hdr); print("  " + "-"*len(hdr))
    for comp, co2_g, t_s in [
            (f"Stage1_{arch_name}",     co2_s1, elapsed_s1),
            (f"Stage2a_IF_{arch_name}", co2_if, elapsed_if)]:
        ct_g  = _rv(comp,"ct_emissions_g"); cc_kwh = _rv(comp,"cc_energy_kwh")
        ct_kw = _rv(comp,"ct_energy_kwh"); sprd   = _rv(comp,"tracker_spread_pct")
        t_str = f"{t_s:>8.1f}" if t_s > 0 else f"{'—':>8}"
        print(f"  {comp:<30}  {_f(co2_g):>9}  {_f(cc_kwh):>8}  {_f(ct_g):>9}  {_f(ct_kw):>8}  {_f(sprd):>8}  {t_str}")
    print("  " + "-"*len(hdr))
    print(f"  {'TOTAL':<30}  {total_co2:>9.5f}")
    _ci_vals = [r.get("carbon_intensity", np.nan) for r in _carbon_rows
                if not np.isnan(r.get("carbon_intensity", np.nan))]
    _ci_str  = f"{np.mean(_ci_vals):.1f} gCO2/kWh" if _ci_vals else f"{CI_G_PER_KWH_FALLBACK:.1f} gCO2/kWh (Norway fallback)"
    _loc = next(((r["cc_country"], r["cc_region"]) for r in _carbon_rows if r.get("cc_country")), ("?","?"))
    print(f"  CI: {_ci_str}  |  location: {_loc[0]}/{_loc[1]}")
    print(f"  Inference: thresh={elapsed_thresh:.1f}s  eval={elapsed_eval:.1f}s  "
          f"per_sample={time_per_sample:.5f}s  per_well={time_per_well:.3f}s")

    all_carbon_rows.extend(_carbon_rows)
    arch_results[arch_name] = {
        'true': all_true, 's1': all_s1, 'fused': all_fused,
        'best_thresh': best_thresh,
        'co2': {'stage1': co2_s1, 'if': co2_if, 'total': total_co2},
        'time_s': {'stage1': elapsed_s1, 'if': elapsed_if,
                   'thresh_search': elapsed_thresh, 'final_eval': elapsed_eval,
                   'per_sample_s': time_per_sample, 'per_well_s': time_per_well},
        'carbon_rows': _carbon_rows,
    }

    # per-well plot
    plot_well(valid_ds.wells[0], df_log_valid, df_tops_valid, best_thresh,
              out_path=f'/home/jupyter/stage2_ckpt/well_{valid_ds.wells[0]}_{arch_key}_stage2.png')

# ── Combined architecture comparison ──────────────────────────────────────
print("\n\n" + "="*70)
print("  COMBINED ARCHITECTURE COMPARISON")
print("="*70)
print(f"{'Arch':<14} {'Model':<16} {'Acc':>6} {'wF1':>6} {'Prec':>6} {'Rec':>6}  {'CO2_total(g)':>13}")
print("-"*70)
for arch_name, res in arch_results.items():
    for label, preds in [('Stage1', res['s1']), ('Stage1+Anomaly', res['fused'])]:
        at  = res['true']
        acc = accuracy_score(at, preds)
        wf1 = f1_score(at, preds, average='weighted', zero_division=0)
        wp  = precision_score(at, preds, average='weighted', zero_division=0)
        wr  = recall_score(at, preds, average='weighted', zero_division=0)
        co2_str = f"{res['co2']['total']:>13.5f}" if label=='Stage1+Anomaly' else f"{'—':>13}"
        print(f"  {arch_name:<12} {label:<16} {acc:>6.4f} {wf1:>6.4f} {wp:>6.4f} {wr:>6.4f}  {co2_str}")

# ── Combined confusion matrices (2 archs × 2 preds = 4 subplots) ──────────
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
plt.suptitle('GraphSAGE vs ImprovedGNN — Stage1 and Stage1+Anomaly', fontsize=13, fontweight='bold')
plt.tight_layout()
out_fig = '/home/jupyter/stage2_ckpt/stage2_confusion_comparison.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nConfusion matrix comparison → {out_fig}")
if CARBON_CSV.exists():
    print(f"Carbon log → {CARBON_CSV}")

# ── Save combined results ──────────────────────────────────────────────────
with open('/home/jupyter/stage2_ckpt/stage2_results.pkl','wb') as f:
    pickle.dump({'arch_results': arch_results, 'all_carbon_rows': all_carbon_rows}, f)
print("Results → /home/jupyter/stage2_ckpt/stage2_results.pkl")