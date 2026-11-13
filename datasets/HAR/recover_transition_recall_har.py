"""
transition_recall for the saved HAR Stage 1 checkpoints
(stage1_har_{arch}_seed{N}.pt).

The metrics follow evaluate_medical() in experiment_training_medical.ipynb, which
produces the Sliding-family numbers (LSTM, LSTM-FCN, InceptionTime). Those models
train on the same curated har_transition/ data as the GNN (built by
combined_data_split_medical.ipynb), so the results are comparable. They are not
comparable to the Seq2Seq / Transformer transition recall, which is computed on
uncurated sequences.

Metrics:
  transition_recall    accuracy on windows whose true label is a transition (> 0):
                       the share classified as exactly the right activity
  transition_precision share of windows predicted as a transition (> 0) that truly
                       are one
  transition_f1        harmonic mean of the two
  f1_transition_only   macro-F1 over classes 1-6, on true-transition windows only
  median_error, max_class_error
                       median / max of |predicted class - true class| on
                       true-transition windows

Only the curated, shuffled test.npz and the checkpoints are needed, not the raw
signals or any temporal order:  python3 recover_transition_recall_har.py
"""
import os, re, glob, json
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import dgl, dgl.nn.pytorch as dglnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.environ.get("DATA_DIR", ".")
CKPT_DIR = os.environ.get("CKPT_DIR", "/home/jupyter/har_stage2_ckpt_v2")
OUT_CSV = Path(CKPT_DIR) / "har_stage2_transition_recall_recovery.csv"

with open(os.path.join(DATA_DIR, "warm_state.json")) as f:
    ws = json.load(f)

CLASS_NAMES = ws["PRED_COLUMN"]
N_CLASSES   = ws["N_CLASSES"]
WINDOW_LEN  = ws["WINDOW_LEN"]
N_CHANNELS  = len(ws["FEATURES_GEOX"])

test_d = np.load(os.path.join(DATA_DIR, "test.npz"))
X_test, y_test = test_d["X"], test_d["y"]
print(f"test.npz: X={X_test.shape}  y={y_test.shape}  classes={CLASS_NAMES}")

_SRC = list(range(WINDOW_LEN - 1)) + list(range(1, WINDOW_LEN))
_DST = list(range(1, WINDOW_LEN)) + list(range(WINDOW_LEN - 1))

def window_to_graph(x):
    g = dgl.graph((_SRC, _DST))
    g = dgl.add_self_loop(g)
    g.ndata["features"] = torch.tensor(x.T, dtype=torch.float32)
    return g

class HARGraphDataset(Dataset):
    def __init__(self, X, y):
        self.graphs = [window_to_graph(X[i]) for i in range(len(X))]
        self.labels = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx], self.labels[idx]

def collate_fn(batch):
    graphs, labels = zip(*batch)
    return dgl.batch(graphs), torch.stack(labels)

test_ds = HARGraphDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

H = 128

class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_feats, hidden, n_classes, dropout=0.3):
        super().__init__()
        self.c1  = dglnn.SAGEConv(in_feats, hidden, "mean")
        self.c2  = dglnn.SAGEConv(hidden,   hidden, "mean")
        self.c3  = dglnn.SAGEConv(hidden,   hidden, "mean")
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_classes)
    def embed(self, g, x):
        h = self.drop(F.relu(self.bn1(self.c1(g, x))))
        h = self.drop(F.relu(self.bn2(self.c2(g, h))))
        h = self.drop(F.relu(self.bn3(self.c3(g, h))))
        sizes  = g.batch_num_nodes().tolist()
        splits = torch.split(h, sizes, dim=0)
        return torch.stack([s.mean(0) for s in splits])
    def forward(self, g, x): return self.fc(self.embed(g, x))

class ImprovedGNNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_sizes, n_classes, dropout=0.1):
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
    def embed(self, g, x):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = self.drop(F.leaky_relu(bn(conv(g, h))))
        sizes  = g.batch_num_nodes().tolist()
        splits = torch.split(h, sizes, dim=0)
        return torch.stack([s.mean(0) for s in splits])
    def forward(self, g, x): return self.fc(self.embed(g, x))

def build_model(arch_key):
    if arch_key == 'graphsage':
        return GraphSAGEClassifier(N_CHANNELS, H, N_CLASSES)
    return ImprovedGNNClassifier(N_CHANNELS, [128, 256, 256, 128, 64], N_CLASSES)

def full_preds(model, dl):
    model.eval(); preds = []
    with torch.no_grad():
        for bg, _ in dl:
            bg = bg.to(DEVICE)
            preds += model(bg, bg.ndata["features"]).argmax(1).tolist()
    return np.array(preds)

def evaluate_medical(preds, trues, n_classes):
    """Port of the metric logic in evaluate_medical() from experiment_training_medical.ipynb."""
    true_trans_mask = trues > 0
    pred_trans_mask = preds > 0

    if true_trans_mask.sum() > 0:
        t_recall = float((preds[true_trans_mask] == trues[true_trans_mask]).mean())
    else:
        t_recall = float('nan')

    if pred_trans_mask.sum() > 0:
        t_precision = float((trues[pred_trans_mask] > 0).mean())
    else:
        t_precision = float('nan')

    if not (np.isnan(t_recall) or np.isnan(t_precision)) and (t_recall + t_precision) > 0:
        t_f1 = 2 * t_precision * t_recall / (t_precision + t_recall)
    else:
        t_f1 = float('nan')

    if true_trans_mask.sum() > 0:
        f1_trans_only = float(f1_score(
            trues[true_trans_mask], preds[true_trans_mask],
            labels=list(range(1, n_classes)), average='macro', zero_division=0,
        ))
    else:
        f1_trans_only = float('nan')

    f1_mac = float(f1_score(trues, preds, average='macro', zero_division=0))

    if true_trans_mask.sum() > 0:
        abs_errors = np.abs(preds[true_trans_mask].astype(int) - trues[true_trans_mask].astype(int))
        median_error    = float(np.median(abs_errors))
        max_class_error = float(np.max(abs_errors))
    else:
        median_error    = float('nan')
        max_class_error = float('nan')

    return {
        "transition_recall": t_recall, "transition_precision": t_precision,
        "transition_f1": t_f1, "f1_transition_only": f1_trans_only,
        "f1_macro": f1_mac, "median_error": median_error, "max_class_error": max_class_error,
        "n_true_transitions": int(true_trans_mask.sum()),
        "n_pred_transitions": int(pred_trans_mask.sum()),
    }

results = []
ckpts = sorted(glob.glob(f"{CKPT_DIR}/stage1_har_graphsage_seed*.pt")) + \
        sorted(glob.glob(f"{CKPT_DIR}/stage1_har_improvedgnn_seed*.pt"))

for ckpt_path in ckpts:
    m = re.search(r"stage1_har_(\w+)_seed(\d+)\.pt", ckpt_path)
    arch_key, seed = m.group(1), int(m.group(2))
    arch_name = "GraphSAGE" if arch_key == "graphsage" else "ImprovedGNN"

    print(f"{arch_name} seed={seed}  loading {ckpt_path} ...")
    model = build_model(arch_key).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    preds = full_preds(model, test_dl)
    trues = y_test

    metrics = evaluate_medical(preds, trues, N_CLASSES)
    print(f"  transition_recall={metrics['transition_recall']:.4f}  "
          f"precision={metrics['transition_precision']:.4f}  "
          f"f1={metrics['transition_f1']:.4f}  "
          f"({metrics['n_true_transitions']} true transition windows)")

    results.append({"arch": arch_name, "seed": seed, **metrics})

df_res = pd.DataFrame(results)
df_res.to_csv(OUT_CSV, index=False)
print(f"\nDone. {len(df_res)} rows -> {OUT_CSV}")
print("\nPer-arch mean +/- std:")
print(df_res.groupby("arch")[["transition_recall", "transition_precision", "transition_f1",
                               "f1_transition_only", "median_error"]]
      .agg(['mean', 'std']).round(4).to_string())
