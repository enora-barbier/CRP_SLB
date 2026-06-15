"""
Colorado — GNN Stage 1 + Stage 2 Anomaly Detection
====================================================
Direct adaptation of gnn_stage2_anomaly_shetland.py for the Colorado dataset.

Differences vs North Sea:
  Features  : GR + DEPTH_NORM only (2 features, GR_CLEAN via sentinel masking)
  Markers   : NIOBARA, FORT_HAYS, CODELL  (3 formations → 4 classes inc. background)
  Well col  : wellName  (float IDs)
  Depth col : DEPTH
  Splits    : 600 training wells split 80/20 → train/val; separate test set (50 or 100 wells)

Stage 1 : GraphSAGE node-level classifier
Stage 2 : IF-only / VGAE-only / IF+VGAE anomaly detection
          Correction: per-well adaptive boundaries from confident Stage 1 predictions
"""

import warnings, os, time, pickle, csv
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
from codecarbon import EmissionsTracker
from datetime import datetime
# ── Paths ──────────────────────────────────────────────────────────────────────
# ── Path resolution (env vars → relative fallback) ────────────────────────────
# Cloud orchestrators (Vertex AI, SageMaker, AzureML, etc.) inject paths via
# environment variables.  Local runs fall back to paths relative to this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Normalise with os.path.join so a missing trailing slash never causes
# "Traininglogs.parquet"-style concatenation bugs on cloud envs.
BASE_TRAIN="./" 
BASE_TEST="./"
# Output / checkpoint root
CKPT_DIR  = os.environ.get("CKPT_DIR",  "/tmp/colorado_stage2_ckpt")
OUT_DIR   = os.environ.get("OUT_DIR",   CKPT_DIR)
LOG_PATH  = os.path.join(OUT_DIR, "colorado_stage2_timeline.csv")
CONF_PATH = os.path.join(OUT_DIR, "colorado_stage2_confusion.png")
RES_PATH  = os.path.join(OUT_DIR, "colorado_stage2_results.pkl")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

S1_PATH   = os.path.join(CKPT_DIR, "stage1_colorado.pt")
IF_PATH   = os.path.join(CKPT_DIR, "isolation_forest_colorado.pkl")
VGAE_PATH = os.path.join(CKPT_DIR, "vgae_colorado.pt")

print(f"BASE_TRAIN : {BASE_TRAIN}")
print(f"BASE_TEST  : {BASE_TEST}")
print(f"CKPT_DIR   : {CKPT_DIR}")
# ── Timeline + energy logger ───────────────────────────────────────────────────
_timeline: list[dict] = []

def _kwh(tracker: EmissionsTracker) -> float:
    try:    return float(tracker.final_emissions_data.energy_consumed)
    except: return 0.0

def log_event(event: str, kwh: float = 0.0, **extra):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": ts, "event": event, "kwh": f"{kwh:.8f}", **extra}
    _timeline.append(row)
    suffix    = f"  kWh={kwh:.6f}" if kwh else ""
    extra_str = "  " + "  ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    print(f"[{ts}] {event}{suffix}{extra_str}")

def save_timeline():
    fields = ["timestamp", "event", "kwh", "elapsed_s", "notes"]
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(_timeline)
    print(f"Timeline → {LOG_PATH}")

# ── Schema ────────────────────────────────────────────────────────────────────
MARKER_COLS     = ['NIOBARA', 'FORT_HAYS', 'CODELL']
INPUT_FEATURES  = ['GR_CLEAN', 'DEPTH_NORM']
N_CLASSES       = len(MARKER_COLS) + 1          # 4: background + 3 formations
MARKER_TO_CLASS = {'background': 0, 'NIOBARA': 1, 'FORT_HAYS': 2, 'CODELL': 3}
CLASS_NAMES     = ['background', 'NIOBARA', 'FORT_HAYS', 'CODELL']
CLASS_COLORS    = ['#d0d0d0', '#4472C4', '#70AD47', '#FFC000']
WELL_COL        = 'wellName'
DEPTH_COL       = 'DEPTH'
GR_SENTINEL     = -9999.0          # invalid GR value → mask as NaN
GR_CLIP_LO, GR_CLIP_HI = 0.0, 300.0   # physically plausible GR range

# ── Load data ─────────────────────────────────────────────────────────────────
log_event("pipeline_start")
df_logs_raw = pd.read_parquet(BASE_TRAIN + "logs.parquet")
df_tops_raw = pd.read_parquet(BASE_TRAIN + "tops.parquet")

# Normalise well name to string in both tables
df_logs_raw[WELL_COL]   = df_logs_raw[WELL_COL].astype(str)
df_tops_raw['well_name'] = df_tops_raw['well_name'].astype(str)
df_tops_raw = df_tops_raw.set_index('well_name')[MARKER_COLS]

# Test set (50 wells)
df_logs_test  = pd.read_parquet(BASE_TEST + "logs_50.parquet")
df_tops_test  = pd.read_csv(BASE_TEST + "tops_50.csv")
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

log_event("data_loaded",
          notes=f"train_wells={len(train_wells)} val_wells={len(val_wells)} test_wells={len(df_tops_test)}")
print(f"  Train wells={len(train_wells)}  Val wells={len(val_wells)}  Test wells={len(df_tops_test)}")
print(f"  Train rows={len(df_log_train):,}  Val rows={len(df_log_valid):,}  Test rows={len(df_logs_test):,}")

# ── Helpers ────────────────────────────────────────────────────────────────────
GLOBAL_DEPTH_MIN = df_logs_raw[DEPTH_COL].min()
GLOBAL_DEPTH_MAX = df_logs_raw[DEPTH_COL].max()

def well_log_cleaning(wn, df_log):
    df = df_log[df_log[WELL_COL] == wn][[DEPTH_COL, 'GR']].copy()
    if df.empty: return df
    df = df.sort_values(DEPTH_COL).reset_index(drop=True)
    # Mask sentinel and out-of-range GR values
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
    """Assign MARKER_TO_CLASS labels given sorted (marker, depth) list."""
    labels = []
    for d in depths:
        active = 'background'
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
        print(f"  {len(self.wells)} wells")

    def __len__(self): return len(self.wells)

    def __getitem__(self, idx):
        w  = self.wells[idx]
        df = well_log_cleaning(w, self.df_log)
        if df.empty: return self._empty()

        tops_row   = self.df_tops.loc[w].dropna()
        well_tops  = tops_row.to_dict()
        sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
        tops_list   = [v for _, v in sorted_tops]

        sv = step_function(float(df[DEPTH_COL].min()),
                           float(df[DEPTH_COL].max()), sorted(tops_list))
        g  = create_graph(df, sv)
        if g.num_nodes() == 0: return self._empty()

        # Flag nodes near any formation boundary (±15 ft)
        depths = g.ndata['pos'][:, 0].numpy()
        is_near_top = np.zeros(len(depths), dtype=bool)
        for t in tops_list:
            is_near_top |= (np.abs(depths - t) < 15.0)
        g.ndata['is_near_top'] = torch.tensor(is_near_top, dtype=torch.bool)

        # Correct labels using proper depth-based assignment
        g.ndata['label'] = torch.tensor(
            assign_labels(depths, sorted_tops), dtype=torch.int64)
        return g

    def _empty(self):
        g = dgl.graph(([], [])); g.add_nodes(1)
        g.ndata['features']    = torch.zeros(1, len(INPUT_FEATURES))
        g.ndata['pos']         = torch.zeros(1, 1)
        g.ndata['label']       = torch.tensor([0], dtype=torch.int64)
        g.ndata['is_near_top'] = torch.tensor([False], dtype=torch.bool)
        return g


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
supports = torch.tensor(_inv, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=supports)
print(f"  Weights: { {n: f'{w:.2f}' for n, w in zip(CLASS_NAMES, _inv)} }")

# ── Model definitions ──────────────────────────────────────────────────────────
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


class VGAE(nn.Module):
    def __init__(self, inf, hidden, latent):
        super().__init__()
        self.enc1   = dglnn.GraphConv(inf, hidden)
        self.mu     = dglnn.GraphConv(hidden, latent)
        self.logvar = dglnn.GraphConv(hidden, latent)
        self.dec    = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(),
                                    nn.Linear(hidden, inf))

    def encode(self, g, x):
        h = F.relu(self.enc1(g, x))
        return self.mu(g, h), self.logvar(g, h)

    def reparameterise(self, mu, lv):
        return mu + torch.exp(0.5*lv)*torch.randn_like(mu) if self.training else mu

    def forward(self, g, x):
        mu, lv = self.encode(g, x)
        z      = self.reparameterise(mu, lv)
        x_hat  = self.dec(z)
        recon  = F.mse_loss(x_hat, x, reduction='none').mean(dim=1)
        kl     = -0.5 * (1 + lv - mu**2 - lv.exp()).mean(dim=1)
        return recon + 0.01*kl, z.detach()


# ── Stage 1 ────────────────────────────────────────────────────────────────────
H = 128; EPOCHS = 15; LR = 0.005
stage1 = GraphSAGEEmbed(len(INPUT_FEATURES), H, N_CLASSES)

if os.path.exists(S1_PATH):
    stage1.load_state_dict(torch.load(S1_PATH))
    log_event("stage1_loaded_from_checkpoint")
    print(f"\nStage 1 loaded from {S1_PATH}")
else:
    log_event("stage1_training_start")
    tracker = EmissionsTracker(project_name="Colorado_GNN_Stage1",
                               output_dir=CKPT_DIR, log_level="error")
    tracker.start(); t0 = time.time()

    print(f"\n{'='*55}")
    print(f"  Stage 1: GraphSAGE ({EPOCHS} epochs, lr={LR})")
    print(f"{'='*55}")
    opt = torch.optim.Adam(stage1.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    for ep in range(EPOCHS):
        stage1.train(); tl = 0; preds = []; labs = []
        for bg in train_dl:
            if bg.num_nodes() == 0: continue
            bg = dgl.add_self_loop(bg); opt.zero_grad()
            lbl = torch.clamp(bg.ndata['label'], 0, N_CLASSES-1)
            out = stage1(bg, bg.ndata['features'])
            loss = criterion(out, lbl)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stage1.parameters(), 1.0)
            opt.step(); tl += loss.item()
            preds += out.argmax(1).tolist(); labs += lbl.tolist()
        sch.step()
        acc = accuracy_score(labs, preds)
        f1  = f1_score(labs, preds, average='weighted', zero_division=0)
        print(f"  ep {ep+1:2d}/{EPOCHS}  loss={tl/max(len(train_dl),1):.4f}"
              f"  acc={acc:.4f}  f1={f1:.4f}")

    torch.save(stage1.state_dict(), S1_PATH)
    kwh = _kwh(tracker); tracker.stop()
    log_event("stage1_training_done", kwh=kwh,
              elapsed_s=f"{time.time()-t0:.1f}")
    print(f"  Saved → {S1_PATH}")

# ── Stage 2a: Isolation Forest ─────────────────────────────────────────────────
if os.path.exists(IF_PATH):
    with open(IF_PATH, 'rb') as f: iso = pickle.load(f)
    log_event("IF_loaded_from_checkpoint")
    print(f"IF loaded from {IF_PATH}")
else:
    log_event("IF_training_start")
    tracker = EmissionsTracker(project_name="Colorado_GNN_IF",
                               output_dir=CKPT_DIR, log_level="error")
    tracker.start(); t0 = time.time()

    print("\nExtracting training embeddings for IF...")
    stage1.eval(); embeds = []; is_near_top = []
    with torch.no_grad():
        for bg in train_dl:
            if bg.num_nodes() == 0: continue
            bg  = dgl.add_self_loop(bg)
            emb = stage1.embed(bg, bg.ndata['features']).cpu().numpy()
            embeds.append(emb)
            nt = bg.ndata.get('is_near_top', torch.zeros(bg.num_nodes(), dtype=torch.bool))
            is_near_top.append(nt.cpu().numpy())

    embeds      = np.vstack(embeds)
    is_near_top = np.concatenate(is_near_top)
    print(f"  Total: {len(embeds):,}  Near-top: {is_near_top.sum():,}"
          f"  Interior: {(~is_near_top).sum():,}")

    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1)
    iso.fit(embeds[~is_near_top])
    with open(IF_PATH, 'wb') as f: pickle.dump(iso, f)

    kwh = _kwh(tracker); tracker.stop()
    log_event("IF_training_done", kwh=kwh,
              elapsed_s=f"{time.time()-t0:.1f}")
    print(f"  Saved → {IF_PATH}")

# ── Stage 2b: VGAE ─────────────────────────────────────────────────────────────
vgae = VGAE(len(INPUT_FEATURES), 64, 32)

if os.path.exists(VGAE_PATH):
    vgae.load_state_dict(torch.load(VGAE_PATH))
    log_event("VGAE_loaded_from_checkpoint")
    print(f"VGAE loaded from {VGAE_PATH}")
else:
    log_event("VGAE_training_start")
    tracker = EmissionsTracker(project_name="Colorado_GNN_VGAE",
                               output_dir=CKPT_DIR, log_level="error")
    tracker.start(); t0 = time.time()

    print("\nTraining VGAE on stable interiors (10 epochs)...")
    opt_v = torch.optim.Adam(vgae.parameters(), lr=0.001)
    for ep in range(10):
        vgae.train(); tl = 0; nb = 0
        for bg in train_dl:
            if bg.num_nodes() == 0: continue
            bg   = dgl.add_self_loop(bg)
            mask = (~bg.ndata['is_near_top']) if 'is_near_top' in bg.ndata \
                   else torch.ones(bg.num_nodes(), dtype=torch.bool)
            if mask.sum() == 0: continue
            opt_v.zero_grad()
            loss, _ = vgae(bg, bg.ndata['features'])
            loss = loss[mask].mean()
            if torch.isnan(loss): continue
            loss.backward(); opt_v.step(); tl += loss.item(); nb += 1
        if (ep+1) % 2 == 0:
            print(f"  ep {ep+1:2d}/10  loss={tl/max(nb,1):.4f}")
    torch.save(vgae.state_dict(), VGAE_PATH)

    kwh = _kwh(tracker); tracker.stop()
    log_event("VGAE_training_done", kwh=kwh,
              elapsed_s=f"{time.time()-t0:.1f}")
    print(f"  Saved → {VGAE_PATH}")

# ── Per-well fusion ────────────────────────────────────────────────────────────
DETECTORS = [("IF only", "if"), ("VGAE only", "vgae"), ("IF+VGAE", "combo")]
RARE_CLASSES = [1, 2, 3]   # all formation classes (non-background)


def _apply_fusion(s1_preds, depths, emb, score, threshold):
    """Per-well adaptive correction using confident Stage 1 predictions."""
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
    sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
    tops_list   = [v for _, v in sorted_tops]
    sv = step_function(float(df[DEPTH_COL].min()),
                       float(df[DEPTH_COL].max()), sorted(tops_list))
    g = create_graph(df, sv)
    if g.num_nodes() == 0: return None
    g = dgl.add_self_loop(g)

    depths = g.ndata['pos'][:, 0].numpy()
    g.ndata['label'] = torch.tensor(
        assign_labels(depths, sorted_tops), dtype=torch.int64)

    stage1.eval(); vgae.eval()
    with torch.no_grad():
        feat     = g.ndata['features']
        logits   = stage1(g, feat).cpu().numpy()
        emb      = stage1.embed(g, feat).cpu().numpy()
        vgae_raw, _ = vgae(g, feat)
        vgae_sc  = vgae_raw.cpu().numpy()

    true_lbl = g.ndata['label'].numpy()

    # Stage 1 smoothed predictions
    exp_l  = np.exp(logits - logits.max(1, keepdims=True))
    probs  = exp_l / exp_l.sum(1, keepdims=True)
    s_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_pred = s_probs.argmax(axis=1)

    # Monotonicity cleanup
    last_form = None
    cleaned   = s1_pred.copy()
    for i in range(len(cleaned)):
        if cleaned[i] != 0:
            last_form = cleaned[i]
        elif last_form is not None:
            cleaned[i] = last_form
    s1_pred = cleaned

    # Anomaly scores
    if_sc    = minmax_scale(-iso.score_samples(emb))
    vgae_sc  = minmax_scale(vgae_sc)
    combo_sc = scipy.ndimage.uniform_filter1d(
        0.5*if_sc + 0.5*vgae_sc, size=smooth)
    if_sc   = scipy.ndimage.uniform_filter1d(if_sc,   size=smooth)
    vgae_sc = scipy.ndimage.uniform_filter1d(vgae_sc, size=smooth)

    scores = {'if': if_sc, 'vgae': vgae_sc, 'combo': combo_sc}
    fused  = {key: _apply_fusion(s1_pred, depths, emb, sc, threshold)
              for key, sc in scores.items()}

    return true_lbl, s1_pred, fused, depths, scores


# ── Threshold search ───────────────────────────────────────────────────────────
LABELS = list(range(N_CLASSES))

log_event("inference_start")
tracker_inf = EmissionsTracker(project_name="Colorado_GNN_Inference",
                               output_dir=CKPT_DIR, log_level="error")
tracker_inf.start(); t0_inf = time.time()

best_thresh = {key: 0.5 for _, key in DETECTORS}
best_f1     = {key: 0.0 for _, key in DETECTORS}

print("\nThreshold search (validation set)...")
for thresh in np.arange(0.05, 0.85, 0.05):
    buckets = {key: {'true': [], 'fused': [], 'flagged': 0, 'total': 0}
               for _, key in DETECTORS}
    for w in valid_ds.wells:
        res = fuse_well(w, df_log_valid, df_tops_valid, thresh)
        if res is None: continue
        tl, _, fused, _, scores = res
        for _, key in DETECTORS:
            buckets[key]['true'].extend(tl)
            buckets[key]['fused'].extend(fused[key])
            buckets[key]['flagged'] += int((scores[key] >= thresh).sum())
            buckets[key]['total']   += len(tl)

    row = f"  thresh={thresh:.2f}"
    for name, key in DETECTORS:
        b    = buckets[key]
        frac = b['flagged'] / max(b['total'], 1)
        mf1  = f1_score(b['true'], b['fused'], average='macro', zero_division=0)
        row += f"  | {name}  flagged={frac:5.1%}  macro_f1={mf1:.4f}"
        if mf1 > best_f1[key]: best_f1[key] = mf1; best_thresh[key] = thresh
    print(row)

print("\n  Best thresholds:")
for name, key in DETECTORS:
    print(f"    {name:<10}  thresh={best_thresh[key]:.2f}"
          f"  macro_f1={best_f1[key]:.4f}")

# ── Final evaluation on validation set ────────────────────────────────────────
print("\nFinal evaluation (validation set)...")
all_true = []; all_s1 = []; all_fused = {key: [] for _, key in DETECTORS}

for w in valid_ds.wells:
    res = fuse_well(w, df_log_valid, df_tops_valid, best_thresh['combo'])
    if res is None: continue
    tl, s1, fused, _, _ = res
    all_true.extend(tl); all_s1.extend(s1)
    for _, key in DETECTORS:
        all_fused[key].extend(fused[key])

kwh_inf = _kwh(tracker_inf); tracker_inf.stop()
log_event("inference_done", kwh=kwh_inf,
          elapsed_s=f"{time.time()-t0_inf:.1f}")

all_true = np.array(all_true)
all_s1   = np.array(all_s1)
for key in all_fused: all_fused[key] = np.array(all_fused[key])

print("\n── Stage 1 only (GraphSAGE baseline) ─────────────────────────────────")
print(classification_report(all_true, all_s1, target_names=CLASS_NAMES,
                             labels=LABELS, zero_division=0))

for name, key in DETECTORS:
    print(f"\n── Stage 1 + {name} {'─'*(48-len(name))}")
    print(classification_report(all_true, all_fused[key],
          target_names=CLASS_NAMES, labels=LABELS, zero_division=0))

print("\n── Overall comparison ─────────────────────────────────────────────────")
print(f"  {'Method':<22} {'Acc':>6}  {'wF1':>6}  {'macF1':>7}")
print("  " + "─"*45)
for label, preds in [("Stage 1 only", all_s1)] + \
                    [(name, all_fused[key]) for name, key in DETECTORS]:
    acc = accuracy_score(all_true, preds)
    wf1 = f1_score(all_true, preds, average='weighted', zero_division=0)
    mf1 = f1_score(all_true, preds, average='macro',    zero_division=0)
    print(f"  {label:<22} {acc:>6.4f}  {wf1:>6.4f}  {mf1:>7.4f}")

# ── Confusion matrices ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(26, 6))
plots = [("Stage 1 only", all_s1)] + \
        [(name, all_fused[key]) for name, key in DETECTORS]
for ax, (title, preds) in zip(axes, plots):
    cm = confusion_matrix(all_true, preds, labels=LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=30)
plt.suptitle('Colorado: Stage 1 vs IF / VGAE / IF+VGAE (validation set)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CONF_PATH, dpi=150, bbox_inches='tight')
print(f"\nConfusion matrix → {CONF_PATH}")

with open(RES_PATH, 'wb') as f:
    pickle.dump({'true': all_true, 'stage1': all_s1,
                 **{f'fused_{k}': all_fused[k] for k in all_fused},
                 'best_thresh': best_thresh, 'class_names': CLASS_NAMES}, f)
print(f"Results → {RES_PATH}")

log_event("pipeline_complete")
save_timeline()
