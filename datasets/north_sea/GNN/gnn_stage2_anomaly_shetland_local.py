"""
Stage 2 Anomaly Detection — with checkpointing so we never retrain on crash.
"""

import warnings, os, time, pickle
warnings.filterwarnings("ignore")
os.environ["CODECARBON_LOG_LEVEL"] = "error"

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

CKPT_DIR = "/tmp/stage2_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────
base_dir = "/Users/fokia/CRP_SLB/datasets/north_sea/Transformer/"
df_log_train  = pd.read_parquet(os.path.join(base_dir, "log_train.parquet"))
df_tops_train = pd.read_parquet(os.path.join(base_dir, "tops_train.parquet"))
df_log_valid  = pd.read_parquet(os.path.join(base_dir, "log_valid.parquet"))
df_tops_valid = pd.read_parquet(os.path.join(base_dir, "tops_valid.parquet"))

MARKER_COLS    = ['DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
INPUT_FEATURES = ['GR_CLEAN', 'RHOB_CLEAN', 'DEEPRES_CLEAN', 'DEPTH_NORM']
# Background is a genuine geological zone (overburden above the first formation top).
# DUNLIN is never the shallowest formation (0% of wells), so merging background
# into any formation only pollutes that class.  Keep it as class 0.
N_CLASSES      = len(MARKER_COLS) + 1   # 7
MARKER_TO_CLASS = {'background':0,'DUNLIN':1,'HORDALAND':2,'ROGALAND':3,'SHETLAND':4,'VESTLAND':5,'VIKING':6}
CLASS_NAMES     = ['background','DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
CLASS_COLORS    = ['#d0d0d0','#4472C4','#70AD47','#FFC000','#FF4D4D','#7030A0','#8B4513']
df_tops_train  = df_tops_train[MARKER_COLS]
df_tops_valid  = df_tops_valid[MARKER_COLS]
print("Data loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────
def well_log_cleaning(wn, df_log, feats, depth_col='DS_INDEX', well_col='well_name'):
    existing_feats = [f for f in feats if f in df_log.columns]
    
    df = df_log[df_log[well_col]==wn][existing_feats + [depth_col]].copy()
    if df.empty: return df
    df = df.sort_values(depth_col).reset_index(drop=True)
    df[existing_feats] = df[existing_feats].ffill().bfill().fillna(df[existing_feats].median())
    
    # Scale depth globally
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
        
        # --- NEW: Flag nodes within +/- 15 ft of any geological boundary top ---
        depths = g.ndata['pos'][:, 0].numpy()
        is_near_top = np.zeros(len(depths), dtype=bool)
        for t in tops:
            is_near_top |= (np.abs(depths - t) < 15.0)
        g.ndata['is_near_top'] = torch.tensor(is_near_top, dtype=torch.bool)
        # -----------------------------------------------------------------------
        tops_series = self.df_tops.loc[w] if w in self.df_tops.index else pd.Series()
        well_tops = tops_series.dropna().to_dict()
        sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])  # shallowest first

        fixed_labels = []
        for d in depths:
            active_marker = 'background'
            for marker_name, top_depth in sorted_tops:
                if d >= top_depth:
                    active_marker = marker_name
                else:
                    break
            fixed_labels.append(MARKER_TO_CLASS[active_marker])
            
        g.ndata['label'] = torch.tensor(fixed_labels, dtype=torch.int64)
        if self.with_marker and 'marker_label' in self.df_log.columns:
            raw=self.df_log[self.df_log['well_name']==w].sort_values('DS_INDEX')
            ml=raw['marker_label'].values[:g.num_nodes()]
            if len(ml)<g.num_nodes():
                ml=np.pad(ml,(0,g.num_nodes()-len(ml)),'edge')
            g.ndata['marker_label']=torch.tensor(
                [MARKER_TO_CLASS.get(l, 0) for l in ml],
                dtype=torch.int64)
        return g
        
    def _empty(self):
        g=dgl.graph(([],[])); g.add_nodes(1)
        g.ndata['features']=torch.zeros(1,len(self.feats))
        g.ndata['pos']=torch.zeros(1,1)
        g.ndata['label']=torch.tensor([0],dtype=torch.int64)
        g.ndata['is_near_top']=torch.tensor([False],dtype=torch.bool)  # NEW
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
# Used in fusion correction to avoid leaking validation tops into the evaluation.
# For each marker, we use the median top depth across all training wells.
PRIOR_TOPS = {
    col: float(df_tops_train[col].median())
    for col in MARKER_COLS if df_tops_train[col].notna().sum() > 0
}
PRIOR_SORTED_TOPS = sorted(PRIOR_TOPS.items(), key=lambda x: x[1])
print(f"  Formation depth priors (training medians): { {k: f'{v:.0f}ft' for k,v in PRIOR_SORTED_TOPS} }")

# ── Models ────────────────────────────────────────────────────────────────
class GraphSAGEEmbed(nn.Module):
    def __init__(self,inf,h,nc,dropout=0.1):
        super().__init__()
        self.c1=dglnn.SAGEConv(inf,h,'mean'); self.c2=dglnn.SAGEConv(h,h,'mean')
        self.c3=dglnn.SAGEConv(h,h,'mean')
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.bn3=nn.BatchNorm1d(h)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(h,nc)
    def embed(self,g,x):
        h=self.drop(F.relu(self.bn1(self.c1(g,x))))
        h=self.drop(F.relu(self.bn2(self.c2(g,h))))
        h=self.drop(F.relu(self.bn3(self.c3(g,h))))
        return h
    def forward(self,g,x): return self.fc(self.embed(g,x))

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

# ── Stage 1: train or load ────────────────────────────────────────────────
S1_PATH   = os.path.join(CKPT_DIR, "stage1_graphsage_fixed.pt")

# Compute class weights from actual training label frequencies
print("Computing class weights from training label frequencies...")
_names = CLASS_NAMES  # ['DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
_counts = np.zeros(N_CLASSES, dtype=np.int64)
for _g in train_ds:
    if _g.num_nodes() == 0: continue
    _lbl = _g.ndata['label'].numpy()
    for _c in range(N_CLASSES):
        _counts[_c] += int((_lbl == _c).sum())
for _c, (_n, _cnt) in enumerate(zip(_names, _counts)):
    print(f"  class {_c} ({_n:12s}): {_cnt:>10,}")
# Inverse-frequency weights, normalised so the most common class = weight 1.0
_inv = _counts.max() / np.where(_counts == 0, 1, _counts).astype(np.float32)
supports  = torch.tensor(_inv, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=supports)
print(f"  Weights: { {n: f'{w:.2f}' for n,w in zip(_names, _inv)} }")

EPOCHS=15; LR=0.005; H=128; NC=N_CLASSES

stage1 = GraphSAGEEmbed(len(INPUT_FEATURES),H,NC)

if os.path.exists(S1_PATH):
    stage1.load_state_dict(torch.load(S1_PATH))
    print(f"\nStage 1 loaded from {S1_PATH}")
else:
    print("\n" + "="*55)
    print(f"  Stage 1: Training GraphSAGE ({EPOCHS} epochs, lr={LR})")
    print("="*55)
    opt=torch.optim.Adam(stage1.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    for ep in range(EPOCHS):
        stage1.train(); tl=0; preds=[]; labs=[]
        for bg in train_dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg); opt.zero_grad()
            lbl=torch.clamp(bg.ndata['label'],0,NC-1)
            out=stage1(bg,bg.ndata['features']); loss=criterion(out,lbl)
            if torch.isnan(loss): continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(stage1.parameters(),1.0)
            opt.step(); tl+=loss.item(); preds+=out.argmax(1).tolist(); labs+=lbl.tolist()
        acc=accuracy_score(labs,preds)
        f1=f1_score(labs,preds,average='weighted',zero_division=0)
        print(f"  ep {ep+1:2d}/{EPOCHS}  loss={tl/max(len(train_dl),1):.4f}  acc={acc:.4f}  f1={f1:.4f}")
        sch.step()
    torch.save(stage1.state_dict(), S1_PATH)
    print(f"  Saved → {S1_PATH}")

# ── Stage 2a: Isolation Forest (train or load) ────────────────────────────
IF_PATH = os.path.join(CKPT_DIR, "isolation_forest_boundary.pkl")


if os.path.exists(IF_PATH):
    with open(IF_PATH,'rb') as f: iso = pickle.load(f)
    print(f"Isolation Forest loaded from {IF_PATH}")
else:
    print("\nExtracting training embeddings...")
    stage1.eval(); embeds=[]; is_near_top=[]
    with torch.no_grad():
        for bg in train_dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg)
            emb=stage1.embed(bg,bg.ndata['features']).cpu().numpy()
            embeds.append(emb)
            if 'is_near_top' in bg.ndata:
                is_near_top.append(bg.ndata['is_near_top'].cpu().numpy())
            else:
                is_near_top.append(np.zeros(emb.shape[0],bool))
    embeds=np.vstack(embeds); is_near_top=np.concatenate(is_near_top)
    print(f"  Total: {len(embeds):,}  Anomalous (Near Top): {is_near_top.sum():,}  Normal (Stable Interior): {(~is_near_top).sum():,}")
    print("Training Isolation Forest on stable interiors...")
    iso=IsolationForest(n_estimators=200,contamination=0.05,random_state=42,n_jobs=-1)
    iso.fit(embeds[~is_near_top])  # Train on stable interiors
    with open(IF_PATH,'wb') as f: pickle.dump(iso,f)
    print(f"  Saved → {IF_PATH}")


# ── Stage 2b: VGAE (train or load) ───────────────────────────────────────
VGAE_PATH = os.path.join(CKPT_DIR, "vgae_boundary.pt")
vgae = VGAE(len(INPUT_FEATURES),64,32)

if os.path.exists(VGAE_PATH):
    vgae.load_state_dict(torch.load(VGAE_PATH))
    print(f"VGAE loaded from {VGAE_PATH}")
else:
    print("\nTraining VGAE (20 epochs)...")
    opt_v=torch.optim.Adam(vgae.parameters(),lr=0.001)
    for ep in range(20):
        vgae.train(); tl=0
        for bg in train_dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg)
            # Train only on stable interiors (is_near_top is False)
            mask = ~bg.ndata['is_near_top'] if 'is_near_top' in bg.ndata \
                  else torch.ones(bg.num_nodes(),dtype=torch.bool)
            if mask.sum()==0: continue
            opt_v.zero_grad()
            loss,_=vgae(bg,bg.ndata['features'])
            loss=loss[mask].mean()
            if torch.isnan(loss): continue
            loss.backward(); opt_v.step(); tl+=loss.item()
        if (ep+1)%5==0:
            print(f"  ep {ep+1:2d}/20  loss={tl/max(len(train_dl),1):.4f}")
    torch.save(vgae.state_dict(), VGAE_PATH)
    print(f"  Saved → {VGAE_PATH}")

# ── Per-well fusion function ──────────────────────────────────────────────
def fuse_well(well, df_log, df_tops, threshold, smooth=11):
    df = well_log_cleaning(well, df_log, INPUT_FEATURES)
    if df.empty: return None
    tops = df_tops.loc[well].dropna().tolist() if well in df_tops.index else []

    # Get the top depth of SHETLAND for this well if available
    shetland_depth = None
    if well in df_tops.index and 'SHETLAND' in df_tops.columns:
        shet_val = df_tops.loc[well, 'SHETLAND']
        if not pd.isna(shet_val):
            shetland_depth = shet_val

    sv = step_function(float(df['DS_INDEX'].min()), float(df['DS_INDEX'].max()), sorted(tops))
    g = create_graph(df, sv)
    if g.num_nodes() == 0: return None
    g = dgl.add_self_loop(g)
    
    # --- Assign per-depth labels (no background class) ---
    well_tops = df_tops.loc[well].dropna().to_dict() if well in df_tops.index else {}
    sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])  # shallowest first
    depths = g.ndata['pos'][:, 0].numpy()

    fixed_labels = []
    for d in depths:
        active_marker = 'background'
        for marker_name, top_depth in sorted_tops:
            if d >= top_depth:
                active_marker = marker_name
            else:
                break
        fixed_labels.append(MARKER_TO_CLASS[active_marker])
    g.ndata['label'] = torch.tensor(fixed_labels, dtype=torch.int64)
    # -------------------------------------------------------------

    stage1.eval(); vgae.eval()
    with torch.no_grad():
        feat = g.ndata['features']
        logits = stage1(g, feat).cpu().numpy()
        emb    = stage1.embed(g, feat).cpu().numpy()
        vgae_loss, _ = vgae(g, feat)
        vgae_sc = vgae_loss.cpu().numpy()
    
    # Use the unshifted 0-indexed labels
    true_lbl = g.ndata['label'].numpy()

    # Stage 1 smoothed predictions
    exp_l       = np.exp(logits - logits.max(1, keepdims=True))
    probs       = exp_l / exp_l.sum(1, keepdims=True)
    smooth_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_preds    = smooth_probs.argmax(axis=1)

    # background (Class 0) can only exist before the first top is met.
    # In between tops, the most recently encountered formation remains active.
    last_formation = None
    cleaned_preds = s1_preds.copy()
    for i in range(len(cleaned_preds)):
        if cleaned_preds[i] != 0:
            last_formation = cleaned_preds[i]
        elif cleaned_preds[i] == 0 and last_formation is not None:
            cleaned_preds[i] = last_formation
    s1_preds = cleaned_preds

    # ── 1. Three anomaly scores: IF-only, VGAE-only, combined ──
    if_sc    = minmax_scale(-iso.score_samples(emb))
    vgae_sc  = minmax_scale(vgae_sc)
    combo_sc = 0.5 * if_sc + 0.5 * vgae_sc
    scores = {
        'if':    scipy.ndimage.uniform_filter1d(if_sc,    size=smooth),
        'vgae':  scipy.ndimage.uniform_filter1d(vgae_sc,  size=smooth),
        'combo': scipy.ndimage.uniform_filter1d(combo_sc, size=smooth),
    }

    # ── 2. For each score, fuse using per-well adaptive boundaries ──
    def _apply_fusion(score, threshold):
        is_anomaly = score >= threshold
        fused = s1_preds.copy()
        confident_mask = ~is_anomaly
        if is_anomaly.sum() > 0 and confident_mask.sum() > 0:
            conf_preds  = s1_preds[confident_mask]
            conf_depths = depths[confident_mask]
            class_depths = {c: conf_depths[conf_preds == c]
                            for c in range(NC) if (conf_preds == c).sum() > 0}
            if len(class_depths) >= 2:
                sorted_classes = sorted(class_depths, key=lambda c: class_depths[c].mean())
                boundaries = [
                    (class_depths[sorted_classes[i]].max() +
                     class_depths[sorted_classes[i + 1]].min()) / 2
                    for i in range(len(sorted_classes) - 1)
                ]
                for i in np.where(is_anomaly)[0]:
                    d = depths[i]
                    class_idx = len(sorted_classes) - 1
                    for j, b in enumerate(boundaries):
                        if d < b:
                            class_idx = j
                            break
                    fused[i] = sorted_classes[class_idx]
        return fused, is_anomaly

    fused_if,    is_anom_if    = _apply_fusion(scores['if'],    threshold)
    fused_vgae,  is_anom_vgae  = _apply_fusion(scores['vgae'],  threshold)
    fused_combo, is_anom_combo = _apply_fusion(scores['combo'], threshold)

    return true_lbl, s1_preds, fused_if, fused_vgae, fused_combo, depths, scores


def _spans(depths, mask):
    spans = []; start = None
    for i, m in enumerate(mask):
        if m and start is None: start = depths[i]
        elif not m and start is not None:
            spans.append((start, depths[i - 1])); start = None
    if start is not None: spans.append((start, depths[-1]))
    return spans

def plot_well(well, df_log, df_tops, threshold, smooth=11, out_path=None):
    result = fuse_well(well, df_log, df_tops, threshold, smooth)
    if result is None: print(f"No data for well {well}"); return None
    true_lbl, s1_preds, _, _, fused, depths, scores = result
    combo = scores['combo']

    df_raw   = well_log_cleaning(well, df_log, INPUT_FEATURES)
    log_vals = df_raw[INPUT_FEATURES].values[:len(depths)]
    tops     = df_tops.loc[well].dropna().tolist() if well in df_tops.index else []

    fig, axes = plt.subplots(1, 7, figsize=(22, 14), sharey=True)

    # Log tracks
    for i, (fname, color) in enumerate(zip(INPUT_FEATURES, ['#2ca02c','#1f77b4','#d62728'])):
        ax = axes[i]
        ax.plot(log_vals[:, i], depths, color=color, lw=0.6)
        ax.set_title(fname, fontsize=8); ax.invert_yaxis(); ax.grid(True, alpha=0.2)
        for t in tops: ax.axhline(t, color='red', lw=1, ls='--', alpha=0.6)

    # Label tracks: True / Stage1 / Fused
    for ax, preds, title in [(axes[3], true_lbl, 'True'),
                              (axes[4], s1_preds, 'Stage 1'),
                              (axes[5], fused,    'Fused')]:
        for c in range(NC):
            for s, e in _spans(depths, preds == c):
                ax.fill_betweenx([s, e], 0, 1, color=CLASS_COLORS[c], alpha=0.85)
        ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_title(title, fontsize=8)
        for t in tops: ax.axhline(t, color='red', lw=1, ls='--', alpha=0.7)

    # Anomaly score track
    ax = axes[6]
    ax.fill_betweenx(depths, 0, combo, where=combo >= threshold, color='red', alpha=0.35)
    ax.plot(combo, depths, 'k-', lw=0.6)
    ax.axvline(threshold, color='orange', lw=1.2, ls='--')
    ax.set_xlim(0, 1); ax.set_title('Anomaly', fontsize=8); ax.grid(True, alpha=0.2)
    for t in tops: ax.axhline(t, color='red', lw=1, ls='--', alpha=0.6)

    axes[0].set_ylabel('Depth', fontsize=9)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=CLASS_COLORS[c], label=CLASS_NAMES[c]) for c in range(NC)],
               loc='lower center', ncol=NC, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    plt.suptitle(f'Well: {well}  |  thresh={threshold:.2f}', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    out = out_path or f'/tmp/well_{well}_stage2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Plot saved → {out}")
    return out

# ── Threshold search ──────────────────────────────────────────────────────
RARE_CLASSES = [1, 3, 5]  # DUNLIN, ROGALAND, VESTLAND

LABELS = list(range(N_CLASSES))
DETECTORS = [('IF only', 'if'), ('VGAE only', 'vgae'), ('IF+VGAE', 'combo')]

# ── Threshold search — per detector ──────────────────────────────────────
print("\nThreshold search...")
best_thresh = {key: 0.5 for _, key in DETECTORS}
best_f1     = {key: 0.0 for _, key in DETECTORS}

for thresh in np.arange(0.05, 0.85, 0.05):
    buckets = {key: {'true': [], 'fused': [], 'flagged': 0, 'total': 0}
               for _, key in DETECTORS}
    for w in valid_ds.wells:
        res = fuse_well(w, df_log_valid, df_tops_valid, thresh)
        if res is None: continue
        tl, _, f_if, f_vgae, f_combo, _, scores = res
        for key, fused in [('if', f_if), ('vgae', f_vgae), ('combo', f_combo)]:
            buckets[key]['true'].extend(tl)
            buckets[key]['fused'].extend(fused)
            buckets[key]['flagged'] += int((scores[key] >= thresh).sum())
            buckets[key]['total']   += len(tl)

    row = f"  thresh={thresh:.2f}"
    for name, key in DETECTORS:
        b   = buckets[key]
        frac = b['flagged'] / max(b['total'], 1)
        mf1  = f1_score(b['true'], b['fused'], average='macro', zero_division=0)
        row += f"  | {name}  flagged={frac:5.1%}  macro_f1={mf1:.4f}"
        if mf1 > best_f1[key]: best_f1[key] = mf1; best_thresh[key] = thresh
    print(row)

print("\n  Best thresholds:")
for name, key in DETECTORS:
    print(f"    {name:<10}  thresh={best_thresh[key]:.2f}  macro_f1={best_f1[key]:.4f}")

# ── Final evaluation — all three detectors at their best threshold ────────
print("\nFinal evaluation...")
results = {key: {'true': [], 's1': [], 'fused': []} for _, key in DETECTORS}
for w in valid_ds.wells:
    # Run at combo best_thresh (use combo threshold as reference for all three
    # so they're compared on equal footing; each detector uses its own score)
    res = fuse_well(w, df_log_valid, df_tops_valid, best_thresh['combo'])
    if res is None: continue
    tl, s1, f_if, f_vgae, f_combo, _, _ = res
    for key, fused in [('if', f_if), ('vgae', f_vgae), ('combo', f_combo)]:
        results[key]['true'].extend(tl)
        results[key]['s1'].extend(s1)
        results[key]['fused'].extend(fused)

all_true = np.array(results['combo']['true'])
all_s1   = np.array(results['combo']['s1'])

print("\n── Stage 1 only (GraphSAGE baseline) ────────────────────────────────")
print(classification_report(all_true, all_s1, target_names=CLASS_NAMES,
                             labels=LABELS, zero_division=0))

for name, key in DETECTORS:
    all_fused = np.array(results[key]['fused'])
    print(f"\n── Stage 1 + {name} ──────────────────────────────────────────────────")
    print(classification_report(all_true, all_fused, target_names=CLASS_NAMES,
                                 labels=LABELS, zero_division=0))

# ── Summary table ─────────────────────────────────────────────────────────
print("\n── Overall comparison ────────────────────────────────────────────────")
print(f"  {'Method':<20} {'Acc':>6}  {'wF1':>6}  {'macF1':>7}  {'DUNLIN-F1':>10}  {'VIKING-F1':>10}")
print("  " + "-"*68)
for label, preds in [("Stage 1 only", all_s1)] + \
                    [(name, np.array(results[key]['fused'])) for name, key in DETECTORS]:
    acc  = accuracy_score(all_true, preds)
    wf1  = f1_score(all_true, preds, average='weighted', zero_division=0)
    mf1  = f1_score(all_true, preds, average='macro',    zero_division=0)
    df1  = f1_score(all_true, preds, labels=[1], average='macro', zero_division=0)
    vf1  = f1_score(all_true, preds, labels=[6], average='macro', zero_division=0)
    print(f"  {label:<20} {acc:>6.4f}  {wf1:>6.4f}  {mf1:>7.4f}  {df1:>10.4f}  {vf1:>10.4f}")

# ── Confusion matrices ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(28, 6))
plots = [("Stage 1 only", all_s1)] + \
        [(name, np.array(results[key]['fused'])) for name, key in DETECTORS]
for ax, (title, preds) in zip(axes, plots):
    cm = confusion_matrix(all_true, preds, labels=LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title(title, fontsize=10); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Stage 1 vs IF-only vs VGAE-only vs IF+VGAE', fontsize=12, fontweight='bold')
plt.tight_layout()
out_fig = '/tmp/stage2_confusion.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix → {out_fig}")

with open('/tmp/stage2_results.pkl', 'wb') as f:
    pickle.dump({'stage1': all_s1, 'true': all_true,
                 'fused_if': np.array(results['if']['fused']),
                 'fused_vgae': np.array(results['vgae']['fused']),
                 'fused_combo': np.array(results['combo']['fused']),
                 'best_thresh': best_thresh}, f)
print("Results → /tmp/stage2_results.pkl")


# ── Per-well visualisation ─────────────────────────────────────────────────
sample_well = valid_ds.wells[0]
plot_well(sample_well, df_log_valid, df_tops_valid, best_thresh['combo'])