"""
Computes Recall@Tolerance(+/-50ft) for Stage 1 predictions, for every already-
saved Colorado checkpoint (stage1_colorado_{arch}_seed{N}.pt), without needing
the IsolationForest fusion step or any threshold from the CSV logs -- just
reloads each checkpoint and runs Stage 1 inference once.

Recall@Tolerance: a true formation-boundary (a depth where the label
sequence changes value) counts as "found" if ANY predicted boundary in that
well falls within +/- tolerance_ft of it. Recall = matched / total true
boundaries, pooled across all wells for that (arch, seed).

Run this on the VM (needs logs.parquet/tops.parquet + the checkpoints, both
only there): python3 recover_boundary_recall_colorado.py
"""
import os, re, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.nn.pytorch as dglnn
import scipy.ndimage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_TRAIN = "."
CKPT_DIR = os.environ.get("CKPT_DIR", "/tmp/colorado_stage2_ckpt")
OUT_CSV = Path(CKPT_DIR) / "colorado_stage2_boundary_recall_recovery.csv"
TOLERANCE_FT = 50.0

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
df_log_valid = df_logs_raw[df_logs_raw[WELL_COL].isin(val_wells)].copy()
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

def build_model(arch_key):
    if arch_key == 'graphsage':
        return GraphSAGEEmbed(len(INPUT_FEATURES), 128, N_CLASSES)
    return ImprovedGNN(len(INPUT_FEATURES), [128, 256, 256, 128, 64], N_CLASSES)

def predict_well(well, df_log, df_tops, stage1, smooth=11):
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
    true_lbl = g.ndata['label'].cpu().numpy()
    exp_l = np.exp(logits - logits.max(1, keepdims=True))
    probs = exp_l / exp_l.sum(1, keepdims=True)
    s_probs = scipy.ndimage.uniform_filter1d(probs, size=smooth, axis=0)
    s1_pred = s_probs.argmax(axis=1)
    return true_lbl, s1_pred, depths

def boundary_depths(labels, depths, min_run_ft=15.0):
    """Depths where the label sequence changes to a NEW value that then holds
    stable for at least min_run_ft on both sides -- rejects single-sample
    classification noise (a raw np.diff()!=0 edge detector would count every
    flip in a noisy per-sample prediction sequence as a "boundary", which
    makes a tolerance-window recall metric trivially easy to satisfy
    regardless of real model quality)."""
    if len(labels) < 2:
        return np.array([])
    # collapse consecutive identical labels into runs: (label, start_idx, end_idx)
    change_idx = np.where(np.diff(labels) != 0)[0] + 1
    run_starts = np.concatenate([[0], change_idx])
    run_ends   = np.concatenate([change_idx, [len(labels)]])
    run_labels = labels[run_starts]
    run_len_ft = depths[run_ends - 1] - depths[run_starts]

    boundaries = []
    for i in range(1, len(run_starts)):
        if run_labels[i] == run_labels[i-1]:
            continue  # shouldn't happen given run construction, guard anyway
        if run_len_ft[i-1] >= min_run_ft and run_len_ft[i] >= min_run_ft:
            boundaries.append(depths[run_starts[i]])
    return np.array(boundaries)

def recall_at_tolerance(true_boundaries, pred_boundaries, tol_ft):
    if len(true_boundaries) == 0:
        return None, 0, 0
    if len(pred_boundaries) == 0:
        return 0.0, 0, len(true_boundaries)
    matched = 0
    for tb in true_boundaries:
        if np.min(np.abs(pred_boundaries - tb)) <= tol_ft:
            matched += 1
    return matched / len(true_boundaries), matched, len(true_boundaries)

results = []
ckpts = sorted(glob.glob(f"{CKPT_DIR}/stage1_colorado_graphsage_seed*.pt")) + \
        sorted(glob.glob(f"{CKPT_DIR}/stage1_colorado_improvedgnn_seed*.pt"))

for ckpt_path in ckpts:
    m = re.search(r"stage1_colorado_(\w+)_seed(\d+)\.pt", ckpt_path)
    arch_key, seed = m.group(1), int(m.group(2))
    arch_name = "GraphSAGE" if arch_key == "graphsage" else "ImprovedGNN"

    print(f"{arch_name} seed={seed}  loading {ckpt_path} ...")
    stage1 = build_model(arch_key).to(DEVICE)
    stage1.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    stage1.eval()

    total_matched = 0
    total_true = 0
    wells = sorted(set(df_log_valid[WELL_COL].unique()) & set(df_tops_valid.index))
    for w in wells:
        res = predict_well(w, df_log_valid, df_tops_valid, stage1)
        if res is None: continue
        true_lbl, s1_pred, depths = res
        tb = boundary_depths(true_lbl, depths)
        pb = boundary_depths(s1_pred, depths)
        _, matched, n_true = recall_at_tolerance(tb, pb, TOLERANCE_FT)
        total_matched += matched
        total_true += n_true

    recall_tol = total_matched / total_true if total_true > 0 else float("nan")
    print(f"  Recall@{TOLERANCE_FT:.0f}ft = {recall_tol:.4f}  ({total_matched}/{total_true} boundaries)")

    results.append({
        "arch": arch_name, "seed": seed,
        "recall_at_tol50ft": recall_tol,
        "n_matched": total_matched, "n_true_boundaries": total_true,
    })

df_res = pd.DataFrame(results)
df_res.to_csv(OUT_CSV, index=False)
print(f"\nDone. {len(df_res)} rows -> {OUT_CSV}")
print("\nPer-arch mean +/- std:")
print(df_res.groupby("arch")["recall_at_tol50ft"].agg(['mean', 'std']).round(4).to_string())
