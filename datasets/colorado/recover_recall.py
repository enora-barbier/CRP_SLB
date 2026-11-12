"""
Recovers stage1_recall / if_recall for already-completed Colorado reps whose
checkpoints were saved but whose recall wasn't logged in the original CSV.

Reloads each stage1_colorado_{arch}_seed{N}.pt checkpoint, re-fits IsolationForest
on its embeddings (deterministic given the same seed + eval-mode embeddings, so
this reproduces the original `iso` exactly -- it was never saved to disk), reuses
the if_thresh already logged in colorado_stage2_carbon_log.csv (skips the 16-pass
threshold search), and does ONE eval pass to compute recall (+ acc/f1 as a sanity
check against the original log).

Run this on the VM (needs logs.parquet/tops.parquet + the checkpoints, both only
there): python3 recover_recall.py
"""
import os, re, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.nn.pytorch as dglnn
import scipy.ndimage
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import minmax_scale

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_TRAIN = "."
BASE_TEST = "."
CKPT_DIR = os.environ.get("CKPT_DIR", "/tmp/colorado_stage2_ckpt")
CARBON_CSV = Path(CKPT_DIR) / "colorado_stage2_carbon_log.csv"
OUT_CSV = Path(CKPT_DIR) / "colorado_stage2_recall_recovery.csv"

MARKER_COLS     = ['NIOBARA', 'FORT_HAYS', 'CODELL']
INPUT_FEATURES  = ['GR_CLEAN', 'DEPTH_NORM']
N_CLASSES       = len(MARKER_COLS)
MARKER_TO_CLASS = {'NIOBARA': 0, 'FORT_HAYS': 1, 'CODELL': 2}
WELL_COL        = 'wellName'
DEPTH_COL       = 'DEPTH'
GR_SENTINEL     = -9999.0
GR_CLIP_LO, GR_CLIP_HI = 0.0, 300.0

df_logs_raw = pd.read_parquet(os.path.join(BASE_TRAIN, "logs.parquet"))
df_tops_raw = pd.read_parquet(os.path.join(BASE_TRAIN, "tops.parquet"))
df_logs_raw[WELL_COL] = df_logs_raw[WELL_COL].astype(str)
df_tops_raw['well_name'] = df_tops_raw['well_name'].astype(str)
df_tops_raw = df_tops_raw.set_index('well_name')[MARKER_COLS]

all_wells = sorted(df_logs_raw[WELL_COL].unique())
np.random.seed(42)
perm = np.random.permutation(len(all_wells))
n_val = int(len(all_wells) * 0.20)
val_wells = set(np.array(all_wells)[perm[:n_val]])
train_wells = set(np.array(all_wells)[perm[n_val:]])
df_log_train = df_logs_raw[df_logs_raw[WELL_COL].isin(train_wells)].copy()
df_log_valid = df_logs_raw[df_logs_raw[WELL_COL].isin(val_wells)].copy()
df_tops_train = df_tops_raw.loc[df_tops_raw.index.isin(train_wells)]
df_tops_valid = df_tops_raw.loc[df_tops_raw.index.isin(val_wells)]

GLOBAL_DEPTH_MIN = df_logs_raw[DEPTH_COL].min()
GLOBAL_DEPTH_MAX = df_logs_raw[DEPTH_COL].max()

def well_log_cleaning(wn, df_log):
    df = df_log[df_log[WELL_COL] == wn][[DEPTH_COL, 'GR']].copy()
    if df.empty: return df
    df = df.sort_values(DEPTH_COL).reset_index(drop=True)
    df['GR'] = df['GR'].replace(GR_SENTINEL, np.nan)
    df.loc[df['GR'] < GR_CLIP_LO, 'GR'] = np.nan
    df.loc[df['GR'] > GR_CLIP_HI, 'GR'] = np.nan
    df['GR_CLEAN'] = df['GR'].ffill().bfill().fillna(df['GR'].median())
    df['DEPTH_NORM'] = (df[DEPTH_COL] - GLOBAL_DEPTH_MIN) / (GLOBAL_DEPTH_MAX - GLOBAL_DEPTH_MIN)
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
    g = dgl.graph((list(range(n-1)), list(range(1, n))))
    g.ndata['features'] = torch.tensor(df[INPUT_FEATURES].values.astype(np.float32))
    g.ndata['pos'] = torch.tensor(df[[DEPTH_COL]].values.astype(np.float32))
    g.ndata['label'] = torch.tensor([t[1] for t in sv], dtype=torch.int64)
    return g

def assign_labels(depths, sorted_tops):
    labels = []
    for d in depths:
        active = sorted_tops[0][0]
        for marker, top_depth in sorted_tops:
            if d >= top_depth: active = marker
            else: break
        labels.append(MARKER_TO_CLASS[active])
    return labels

class WellGraphDataset:
    def __init__(self, df_log, df_tops):
        lw = set(df_log[WELL_COL].unique()); tw = set(df_tops.index)
        self.wells = sorted(lw & tw)
        self.df_log = df_log; self.df_tops = df_tops
    def __len__(self): return len(self.wells)
    def __getitem__(self, idx):
        w = self.wells[idx]
        df = well_log_cleaning(w, self.df_log)
        if df.empty: return dgl.graph(([], []))
        tops_row = self.df_tops.loc[w].dropna()
        well_tops = tops_row.to_dict()
        if not well_tops: return dgl.graph(([], []))
        sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
        tops_list = [v for _, v in sorted_tops]
        df = df[df[DEPTH_COL] >= min(tops_list)].reset_index(drop=True)
        if df.empty: return dgl.graph(([], []))
        sv = step_function(float(df[DEPTH_COL].min()), float(df[DEPTH_COL].max()), sorted(tops_list))
        g = create_graph(df, sv)
        if g.num_nodes() == 0: return dgl.graph(([], []))
        depths = g.ndata['pos'][:, 0].numpy()
        is_near_top = np.zeros(len(depths), dtype=bool)
        for t in tops_list: is_near_top |= (np.abs(depths - t) < 15.0)
        g.ndata['is_near_top'] = torch.tensor(is_near_top, dtype=torch.bool)
        g.ndata['label'] = torch.tensor(assign_labels(depths, sorted_tops), dtype=torch.int64)
        return g

def collate(v):
    v = [g for g in v if g.num_nodes() > 0]
    return dgl.batch(v) if v else dgl.graph(([], []))

train_ds = WellGraphDataset(df_log_train, df_tops_train)
valid_ds = WellGraphDataset(df_log_valid, df_tops_valid)
train_batches = [collate([train_ds[i] for i in range(j, min(j+4, len(train_ds)))])
                  for j in range(0, len(train_ds), 4)]

class GraphSAGEEmbed(nn.Module):
    def __init__(self, inf, h, nc, dropout=0.1):
        super().__init__()
        self.c1 = dglnn.SAGEConv(inf, h, 'mean'); self.c2 = dglnn.SAGEConv(h, h, 'mean')
        self.c3 = dglnn.SAGEConv(h, h, 'mean')
        self.bn1 = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h); self.bn3 = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(dropout); self.fc = nn.Linear(h, nc)
    def embed(self, g, x):
        h = self.drop(F.relu(self.bn1(self.c1(g, x))))
        h = self.drop(F.relu(self.bn2(self.c2(g, h))))
        h = self.drop(F.relu(self.bn3(self.c3(g, h))))
        return h
    def forward(self, g, x): return self.fc(self.embed(g, x))

class ImprovedGNN(nn.Module):
    def __init__(self, inf, hidden_sizes, nc, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        prev = inf
        for h in hidden_sizes:
            self.convs.append(dglnn.GraphConv(prev, h)); self.bns.append(nn.BatchNorm1d(h)); prev = h
        self.drop = nn.Dropout(dropout); self.fc = nn.Linear(prev, nc)
    def embed(self, g, x):
        h = x
        for conv, bn in zip(self.convs, self.bns): h = self.drop(F.leaky_relu(bn(conv(g, h))))
        return h
    def forward(self, g, x): return self.fc(self.embed(g, x))

def build_model(arch_name):
    if arch_name == 'graphsage':
        return GraphSAGEEmbed(len(INPUT_FEATURES), 128, N_CLASSES)
    return ImprovedGNN(len(INPUT_FEATURES), [128, 256, 256, 128, 64], N_CLASSES)

def fuse_well(well, df_log, df_tops, threshold, stage1, iso, smooth=11):
    df = well_log_cleaning(well, df_log)
    if df.empty: return None
    well_tops = df_tops.loc[well].dropna().to_dict() if well in df_tops.index else {}
    if not well_tops: return None
    sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])
    tops_list = [v for _, v in sorted_tops]
    df = df[df[DEPTH_COL] >= min(tops_list)].reset_index(drop=True)
    if df.empty: return None
    sv = step_function(float(df[DEPTH_COL].min()), float(df[DEPTH_COL].max()), sorted(tops_list))
    g = create_graph(df, sv)
    if g.num_nodes() == 0: return None
    g = dgl.add_self_loop(g)
    depths = g.ndata['pos'][:, 0].numpy()
    g.ndata['label'] = torch.tensor(assign_labels(depths, sorted_tops), dtype=torch.int64)
    g = g.to(DEVICE)
    stage1.eval()
    with torch.no_grad():
        feat = g.ndata['features']
        logits = stage1(g, feat).cpu().numpy()
        emb = stage1.embed(g, feat).cpu().numpy()
    true_lbl = g.ndata['label'].cpu().numpy()
    exp_l = np.exp(logits - logits.max(1, keepdims=True))
    probs = exp_l / exp_l.sum(1, keepdims=True)
    s_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_pred = s_probs.argmax(axis=1)
    if_sc = minmax_scale(-iso.score_samples(emb))
    if_sc = scipy.ndimage.uniform_filter1d(if_sc, size=smooth)
    is_anomaly = if_sc >= threshold
    fused = s1_pred.copy()
    confident = ~is_anomaly
    if is_anomaly.sum() > 0 and confident.sum() > 0:
        conf_preds = s1_pred[confident]; conf_depths = depths[confident]
        class_depths = {c: conf_depths[conf_preds == c] for c in range(N_CLASSES) if (conf_preds == c).sum() > 0}
        if len(class_depths) >= 2:
            sorted_cls = sorted(class_depths, key=lambda c: class_depths[c].mean())
            boundaries = [(class_depths[sorted_cls[i]].max() + class_depths[sorted_cls[i+1]].min()) / 2
                          for i in range(len(sorted_cls)-1)]
            for i in np.where(is_anomaly)[0]:
                d = depths[i]; cls_idx = len(sorted_cls) - 1
                for j, b in enumerate(boundaries):
                    if d < b: cls_idx = j; break
                fused[i] = sorted_cls[cls_idx]
    return true_lbl, s1_pred, fused

orig = pd.read_csv(CARBON_CSV) if CARBON_CSV.exists() else pd.DataFrame(
    columns=["arch", "seed", "if_thresh", "stage1_acc", "if_acc"])

results = []
ckpts = sorted(glob.glob(f"{CKPT_DIR}/stage1_colorado_graphsage_seed*.pt")) + \
        sorted(glob.glob(f"{CKPT_DIR}/stage1_colorado_improvedgnn_seed*.pt"))

for ckpt_path in ckpts:
    m = re.search(r"stage1_colorado_(\w+)_seed(\d+)\.pt", ckpt_path)
    arch_key, seed = m.group(1), int(m.group(2))
    arch_name = "GraphSAGE" if arch_key == "graphsage" else "ImprovedGNN"

    row_match = orig[(orig["arch"] == arch_name) & (orig["seed"] == seed)]
    if row_match.empty:
        print(f"  [skip] no matching CSV row for {arch_name} seed={seed}")
        continue
    if_thresh = float(row_match.iloc[0]["if_thresh"])

    print(f"{arch_name} seed={seed}  loading {ckpt_path} ...")
    stage1 = build_model(arch_key).to(DEVICE)
    stage1.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    stage1.eval()

    embeds, is_near_top = [], []
    with torch.no_grad():
        for bg in train_batches:
            if bg.num_nodes() == 0: continue
            bg = dgl.add_self_loop(bg).to(DEVICE)
            emb = stage1.embed(bg, bg.ndata['features']).cpu().numpy()
            embeds.append(emb)
            is_near_top.append(bg.ndata.get('is_near_top', torch.zeros(bg.num_nodes(), dtype=torch.bool)).cpu().numpy())
    embeds = np.vstack(embeds); is_near_top = np.concatenate(is_near_top)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed, n_jobs=-1)
    iso.fit(embeds[~is_near_top])

    all_true, all_s1, all_fused = [], [], []
    for w in valid_ds.wells:
        res = fuse_well(w, df_log_valid, df_tops_valid, if_thresh, stage1, iso)
        if res is None: continue
        tl, s1p, fp = res
        all_true.extend(tl); all_s1.extend(s1p); all_fused.extend(fp)
    all_true = np.array(all_true); all_s1 = np.array(all_s1); all_fused = np.array(all_fused)

    stage1_acc = accuracy_score(all_true, all_s1)
    if_acc = accuracy_score(all_true, all_fused)
    stage1_recall = recall_score(all_true, all_s1, average='weighted', zero_division=0)
    if_recall = recall_score(all_true, all_fused, average='weighted', zero_division=0)

    class_labels = list(range(N_CLASSES))
    stage1_recall_per_class = recall_score(all_true, all_s1, average=None, labels=class_labels, zero_division=0)
    if_recall_per_class = recall_score(all_true, all_fused, average=None, labels=class_labels, zero_division=0)
    class_names = [k for k, v in sorted(MARKER_TO_CLASS.items(), key=lambda kv: kv[1])]

    orig_stage1_acc = float(row_match.iloc[0]["stage1_acc"])
    orig_if_acc = float(row_match.iloc[0]["if_acc"])
    match_ok = abs(stage1_acc - orig_stage1_acc) < 0.01 and abs(if_acc - orig_if_acc) < 0.01

    print(f"  stage1_recall={stage1_recall:.4f}  if_recall={if_recall:.4f}  "
          f"[sanity: stage1_acc {stage1_acc:.4f} vs orig {orig_stage1_acc:.4f}, "
          f"if_acc {if_acc:.4f} vs orig {orig_if_acc:.4f}  -> {'OK' if match_ok else 'MISMATCH'}]")
    print("    per-class recall (stage1 / if):  " +
          "  ".join(f"{cname}={s1r:.4f}/{ifr:.4f}"
                     for cname, s1r, ifr in zip(class_names, stage1_recall_per_class, if_recall_per_class)))

    row_out = {
        "arch": arch_name, "seed": seed, "if_thresh": if_thresh,
        "stage1_recall": stage1_recall, "if_recall": if_recall,
        "stage1_acc_recomputed": stage1_acc, "stage1_acc_original": orig_stage1_acc,
        "if_acc_recomputed": if_acc, "if_acc_original": orig_if_acc,
        "sanity_match": match_ok,
    }
    for cname, r in zip(class_names, stage1_recall_per_class):
        row_out[f"stage1_recall_{cname.lower()}"] = r
    for cname, r in zip(class_names, if_recall_per_class):
        row_out[f"if_recall_{cname.lower()}"] = r
    results.append(row_out)

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"\nDone. {len(results)} rows -> {OUT_CSV}")
