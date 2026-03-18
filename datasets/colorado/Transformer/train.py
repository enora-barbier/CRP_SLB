"""
PatchTST Well Log Segmentation Pipelines
=================================================================
Input  : GR log resampled to a fixed length  (bs, 1, seq_len)
Output : zone label at every depth point     (bs, seq_len)
Loss   : Cross-entropy


"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Tuple
from sklearn.metrics import classification_report, confusion_matrix

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import random
def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



# ---------------------------------------------------------------------------
# 1.  DATASET
# ---------------------------------------------------------------------------

class WellLogDataset(Dataset):
    """
    Each sample is one well:
        X : float32  (1, seq_len)   – resampled GR log
        y : int64    (seq_len,)     – zone label per depth point  (tops_seq)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (n_wells, seq_len)  or  (n_wells, 1, seq_len)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]          # → (n_wells, 1, seq_len)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 2.  SEGMENTATION HEAD
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """
    Maps backbone output → per-depth-point class logits.

    Backbone (TSTiEncoder + classification_head) outputs:
        (bs, n_vars, n_markers+2, patch_num)

    We flatten patch_num and n_markers+2 into a feature vector,
    then upsample to seq_len via a single Linear layer,
    giving (bs, seq_len, n_classes) logits.

    Parameters
    ----------
    head_nf     : (n_markers+2) * patch_num  backbone feature dim per channel
    n_vars      : c_in (1 for GR-only)
    seq_len     : target output sequence length
    n_classes   : number of zone labels  (n_tops + 1)
    dropout     : dropout before the linear
    """
    def __init__(self, head_nf: int, n_vars: int, seq_len: int,
                 n_classes: int, dropout: float = 0.1):
        super().__init__()
        flat_dim = head_nf * n_vars          # flatten vars too
        self.flatten = nn.Flatten()          # (bs, n_vars*head_nf)
        self.drop    = nn.Dropout(dropout)
        # project to seq_len * n_classes in one shot
        self.linear  = nn.Linear(flat_dim, seq_len * n_classes)
        self.seq_len  = seq_len
        self.n_classes = n_classes

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (bs, n_vars, n_markers+2, patch_num)
        bs = z.size(0)
        z = z.reshape(bs, -1)                          # (bs, n_vars*head_nf)
        z = self.drop(z)
        z = self.linear(z)                             # (bs, seq_len*n_classes)
        z = z.reshape(bs, self.seq_len, self.n_classes)# (bs, seq_len, n_classes)
        return z


# ---------------------------------------------------------------------------
# 3.  FULL MODEL
# ---------------------------------------------------------------------------

class PatchTSTSegmenter(nn.Module):
    """
    PatchTST backbone + segmentation head for well-log zone prediction.

    Input  : (bs, 1, seq_len)   – GR log
    Output : (bs, seq_len, n_classes)  – logits per depth point
    """
    def __init__(
        self,
        seq_len:    int,
        n_classes:  int,
        patch_len:  int = 16,
        stride:     int = 8,
        d_model:    int = 128,
        n_heads:    int = 8,
        n_layers:   int = 3,
        d_ff:       int = 256,
        n_markers:  int = 0,
        dropout:    float = 0.1,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        from PatchTST_backbone import PatchTST_backbone

        # patch_num computed same way as backbone does it (with 'end' padding)
        patch_num = int((seq_len - patch_len) / stride + 1) + 1  # +1 for 'end' pad
        head_nf   = (n_markers + 2) * patch_num

        # backbone with a dummy target_window – its Flatten_Head is NOT used;
        # we bypass it with our own SegmentationHead below
        self.backbone = PatchTST_backbone(
            c_in           = 1,
            context_window = seq_len,
            target_window  = seq_len,     # placeholder, head is replaced
            patch_len      = patch_len,
            n_markers      = n_markers,
            stride         = stride,
            d_model        = d_model,
            n_heads        = n_heads,
            n_layers       = n_layers,
            d_ff           = d_ff,
            norm           = "BatchNorm",
            dropout        = dropout,
            fc_dropout     = dropout,
            head_dropout   = head_dropout,
            head_type      = "flatten",
            individual     = False,
            revin          = False,       # as requested
            padding_patch  = "end",
            pe             = "zeros",
            learn_pe       = True,
            res_attention  = True,
            rope           = False,
            alibi          = False,
        )

        # Remove the backbone's own head – we only want the encoder features
        # The backbone's TSTiEncoder already applies the classification_head
        # which maps d_model → (n_markers+2), so backbone output is
        # (bs, n_vars, n_markers+2, patch_num)  i.e. head_nf per var
        del self.backbone.head

        self.seg_head = SegmentationHead(
            head_nf   = head_nf,
            n_vars    = 1,
            seq_len   = seq_len,
            n_classes = n_classes,
            dropout   = head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (bs, 1, seq_len)
        returns logits : (bs, seq_len, n_classes)
        """
        # Run backbone up to (just before) its head
        z = self._encode(x)       # (bs, 1, n_markers+2, patch_num)
        return self.seg_head(z)   # (bs, seq_len, n_classes)

    def _encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Replicate backbone.forward() but skip the final self.backbone.head call.
        revin=False so no normalisation layer.
        """
        b = self.backbone

        # padding
        if b.padding_patch == "end":
            z = b.padding_patch_layer(z)

        # unfold into patches
        z = z.unfold(dimension=-1, size=b.patch_len, step=b.stride)  # (bs,1,patch_num,patch_len)
        z = z.permute(0, 1, 3, 2)                                     # (bs,1,patch_len,patch_num)

        # key padding mask
        if b.key_padding_mask == "auto":
            z, kpm = b.key_padding_masking(z)
        else:
            kpm = None

        z = torch.nan_to_num(z)

        # backbone encoder  (TSTiEncoder)
        z = b.backbone(z, key_padding_mask_tensor=kpm)  # (bs,1,n_markers+2,patch_num)
        return z


# ---------------------------------------------------------------------------
# 4.  DATA PREPARATION
# ---------------------------------------------------------------------------

def build_tops_seq(depth_arr: np.ndarray, top_depths: np.ndarray) -> np.ndarray:
    """
    Build a 1-D integer zone-label array matching tops_seq from
    generate_sequences_resampling_markers.

    zone 0 = above first top
    zone k = between top k-1 and top k
    zone N = below last top
    """
    labels = np.zeros(len(depth_arr), dtype=np.int64)
    for k, td in enumerate(sorted(top_depths)):
        labels[depth_arr >= td] = k + 1
    return labels

def crop_to_tops_region(
    depth_arr: np.ndarray,
    gr_arr: np.ndarray,
    top_depths: np.ndarray,
    margin_ft: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop log to [min_top - margin, max_top + margin]."""
    min_top = top_depths.min() - margin_ft
    max_top = top_depths.max() + margin_ft
    mask = (depth_arr >= min_top) & (depth_arr <= max_top)
    return depth_arr[mask], gr_arr[mask]

def prepare_sequences(
    df_log:     pd.DataFrame,
    df_tops:    pd.DataFrame,
    top_cols:   List[str],
    srate:      float = 4.0,
    gr_clip:    Tuple[float, float] = (-1.0, 400.0),
    well_list:  Optional[List] = None,
    pad_value:  float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample every well's GR log to a regular depth grid (srate ft spacing),
    build the zone-label sequence, and pad to the same length.

    Mirrors generate_sequences_resampling_markers logic from the screenshot.

    Returns
    -------
    X          : float32  (n_wells, seq_len)   – resampled GR
    y          : int64    (n_wells, seq_len)   – zone labels (tops_seq)
    depth_axes : list of 1-D depth arrays, one per well (before padding)
    """
    wells = well_list if (well_list is not None and len(well_list) > 0) \
            else list(df_tops.index)
    log_wells = set(float(w) for w in df_log["wellName"].unique())
    tops_wells = set(float(w) for w in df_tops.index)
    wells = [w for w in wells
             if float(w) in log_wells
             and float(w) in tops_wells
             and not df_tops.loc[w, top_cols].isnull().any()]

    all_gr, all_labels, depth_axes = [], [], []

    for well in wells:
        wdf = df_log[df_log["wellName"] == well].sort_values("DEPTH")
        depth_raw = wdf["DEPTH"].values.astype(np.float32)
        gr_raw    = np.clip(wdf["GR"].values.astype(np.float32),
                            gr_clip[0], gr_clip[1])

        min_d = depth_raw.min()
        max_d = depth_raw.max()

        # resample to regular grid at srate ft spacing
        depth_grid = np.arange(min_d, max_d + srate, srate, dtype=np.float32)
        gr_resampled = np.interp(depth_grid, depth_raw, gr_raw).astype(np.float32)

        top_depths = df_tops.loc[well, top_cols].values.astype(float)
        # Crop to tops region
        depth_grid, gr_resampled = crop_to_tops_region(
            depth_grid, gr_resampled, top_depths, margin_ft=500.0
        )
        labels = build_tops_seq(depth_grid, top_depths)

        all_gr.append(gr_resampled)
        all_labels.append(labels)
        depth_axes.append(depth_grid)

    # Pad all sequences to the same length
    seq_len = max(len(g) for g in all_gr)
    X = np.full((len(all_gr), seq_len), pad_value, dtype=np.float32)
    y = np.zeros((len(all_gr), seq_len), dtype=np.int64)

    for i, (gr, lbl) in enumerate(zip(all_gr, all_labels)):
        X[i, :len(gr)] = gr
        y[i, :len(lbl)] = lbl
        # pad label with last valid label (not 0) to avoid loss on padding
        y[i, len(lbl):] = lbl[-1]

    print(f"Prepared {len(wells)} wells  |  seq_len={seq_len}  |  "
          f"n_classes={len(top_cols)+1}")
    return X, y, depth_axes


# ---------------------------------------------------------------------------
# 5.  TRAINING
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, device, class_weights=None):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)   # (bs, 1, seq_len)
        y_batch = y_batch.to(device)   # (bs, seq_len)
        optimiser.zero_grad()
        logits = model(X_batch)        # (bs, seq_len, n_classes)
        # CrossEntropyLoss expects (bs, n_classes, seq_len)
        loss = criterion(logits.permute(0, 2, 1), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)                          # (bs, seq_len, n_classes)
        loss = criterion(logits.permute(0, 2, 1), y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=-1)                    # (bs, seq_len)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.numel()
    return total_loss / len(loader.dataset), correct / total


def train(
    model:       PatchTSTSegmenter,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_valid:     np.ndarray,
    y_valid:     np.ndarray,
    epochs:      int   = 30,
    batch_size:  int   = 16,
    lr:          float = 1e-4,
    weight_decay:float = 1e-4,
    save_path:   str   = "patchtst_seg.pt",
    device_str:  str   = "auto",
) -> PatchTSTSegmenter:

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device_str == "auto" else torch.device(device_str)
    print(f"Training on {device}")

    model = model.to(device)

    n_classes = model.seg_head.n_classes
    class_counts = np.bincount(y_train.flatten(), minlength=n_classes).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = torch.tensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum() * n_classes
    print(f"Class weights: {class_weights.cpu().numpy()}")

    train_dl = DataLoader(WellLogDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True,  num_workers=0)
    valid_dl = DataLoader(WellLogDataset(X_valid, y_valid),
                          batch_size=batch_size, shuffle=False, num_workers=0)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_acc, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimiser, device)
        val_loss, val_acc = evaluate(model, valid_dl, device)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f"  ✓ Saved  (val_acc={best_acc:.4f})")

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# 6.  INFERENCE
# ---------------------------------------------------------------------------

def enforce_monotonic(zone_preds: np.ndarray) -> np.ndarray:
    """Force the step function to be non-decreasing with depth."""
    out = zone_preds.copy()
    for i in range(1, len(out)):
        if out[i] < out[i-1]:
            out[i] = out[i-1]
    return out

@torch.no_grad()
def predict(
    model:      PatchTSTSegmenter,
    X:          np.ndarray,          # (n_wells, seq_len)
    depth_axes: List[np.ndarray],    # original (unpadded) depth arrays
    batch_size: int = 16,
) -> List[pd.DataFrame]:
    """
    Run inference and return one DataFrame per well with columns:
        DEPTH, GR, predicted_zone
    Padding tokens are trimmed using depth_axes.
    """
    device = next(model.parameters()).device
    model.eval()

    if X.ndim == 2:
        X = X[:, np.newaxis, :]       # (n_wells, 1, seq_len)

    results = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start: start + batch_size]).to(device)
        logits = model(batch)          # (bs, seq_len, n_classes)
        preds  = logits.argmax(dim=-1).cpu().numpy()  # (bs, seq_len)

        for i, pred_seq in enumerate(preds):
            well_idx  = start + i
            depth_arr = depth_axes[well_idx]
            valid_len = len(depth_arr)
            zone_preds = enforce_monotonic(pred_seq[:valid_len])
            df = pd.DataFrame({
                "DEPTH":          depth_arr,
                "GR":             X[well_idx, 0, :valid_len],
                "predicted_zone": zone_preds,
            })
            results.append(df)

    return results


# ---------------------------------------------------------------------------
# 7.  EVALUATION  –  top-depth picking accuracy
# ---------------------------------------------------------------------------

def top_depths_from_step_function(
    depth_arr:  np.ndarray,
    zone_preds: np.ndarray,
    n_tops:     int,
) -> np.ndarray:
    """
    Recover predicted top depths from the step function.
    Top k = first depth where zone label transitions from k-1 → k.
    """
    est = []
    for k in range(1, n_tops + 1):
        idx = np.where((zone_preds[:-1] < k) & (zone_preds[1:] >= k))[0]
        est.append(float(depth_arr[idx[0]]) if len(idx) else np.nan)
    return np.array(est)

def evaluate_top_picking(
    results:    List[pd.DataFrame],
    well_names: List[str],
    df_tops:    pd.DataFrame,
    top_cols:   List[str],
) -> pd.DataFrame:
    """
    Compare predicted vs ground-truth top depths.
    Prints mean absolute error per top and returns a summary DataFrame.
    """
    rows = []
    for well, df in zip(well_names, results):
        if well not in df_tops.index:
            continue
        est = top_depths_from_step_function(
            df["DEPTH"].values, df["predicted_zone"].values, len(top_cols)
        )
        gt = df_tops.loc[well, top_cols].values.astype(float)
        row = {"well": well}
        for i, col in enumerate(top_cols):
            row[f"{col}_gt"]     = gt[i]
            row[f"{col}_pred"]   = est[i]
            row[f"{col}_err_ft"] = abs(est[i] - gt[i]) if not np.isnan(est[i]) else np.nan
        rows.append(row)

    df_eval = pd.DataFrame(rows)
    print("\nTop-picking mean absolute error:")
    for col in top_cols:
        print(f"  {col:15s}  {df_eval[f'{col}_err_ft'].mean():.1f} ft")
    return df_eval

def evaluate_segmentation_metrics(
    results:    List[pd.DataFrame],
    well_names: List[str],
    df_tops:    pd.DataFrame,
    top_cols:   List[str],
    srate:      float = 4.0,
    gr_clip:    Tuple[float, float] = (-1.0, 400.0),
):
    """
    Compute per-class precision, recall, F1 and overall accuracy
    by comparing predicted zones to ground-truth zones at every depth point.
    """
    all_preds, all_true = [], []

    for well, df in zip(well_names, results):
        if well not in df_tops.index:
            continue
        top_depths = df_tops.loc[well, top_cols].values.astype(float)
        depth_arr  = df["DEPTH"].values
        true_labels = build_tops_seq(depth_arr, top_depths)
        pred_labels = df["predicted_zone"].values

        all_true.extend(true_labels.tolist())
        all_preds.extend(pred_labels.tolist())

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)

    # Overall accuracy
    accuracy = (all_true == all_preds).mean()
    print(f"\nOverall depth-sample accuracy: {accuracy:.4f}")

    # Per-class precision, recall, F1
    zone_names = ["zone_0"] + [f"{col}" for col in top_cols]
    print("\nPer-class metrics:")
    print(classification_report(all_true, all_preds, target_names=zone_names))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=zone_names, columns=zone_names))

    return accuracy


# ---------------------------------------------------------------------------
# 8.  VISUALISATION
# ---------------------------------------------------------------------------

def plot_prediction(
    df_result:  pd.DataFrame,
    df_tops:    Optional[pd.DataFrame] = None,
    top_cols:   Optional[List[str]]    = None,
    well_name:  str = "",
    save_fig:   Optional[str] = None,
):
    """
    Plot GR log (left) and predicted step function (right),
    with ground-truth tops overlaid as dashed lines.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    depth = df_result["DEPTH"].values
    gr    = df_result["GR"].values
    zone  = df_result["predicted_zone"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10), sharey=True)

    ax1.plot(gr, depth, color="steelblue", lw=0.8)
    ax1.set_xlabel("GR (API)")
    ax1.set_ylabel("Depth (ft)")
    ax1.set_title(f"GR Log  |  {well_name}")
    ax1.invert_yaxis()

    ax2.step(zone, depth, where="post", color="steelblue", lw=1.5)
    ax2.set_xlabel("Predicted zone")
    ax2.set_title("Zone step function")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ground-truth tops
    if df_tops is not None and top_cols and well_name in df_tops.index:
        colors = plt.cm.tab10.colors
        for i, col in enumerate(top_cols):
            td = df_tops.loc[well_name, col]
            if not np.isnan(td):
                for ax in (ax1, ax2):
                    ax.axhline(td, color=colors[i % 10], ls="--", lw=1, alpha=0.8,
                               label=col)
        ax2.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 9.  MAIN PIPELINE  (drop-in replacement for existing notebook cells)
# ---------------------------------------------------------------------------

def run_pipeline(
    # --- data ---
    df_log_train:         pd.DataFrame,
    df_tops_train:        pd.DataFrame,
    training_well_array:  np.ndarray,
    df_log_valid:         pd.DataFrame,
    df_tops_valid:        pd.DataFrame,
    df_log_test:          Optional[pd.DataFrame] = None,
    df_tops_test:         Optional[pd.DataFrame] = None,
    top_cols:             Optional[List[str]]    = None,
    # --- resampling ---
    srate:                float = 4.0,
    gr_clip:              Tuple[float, float] = (-1.0, 400.0),
    # --- model ---
    patch_len:            int   = 16,
    stride:               int   = 8,
    d_model:              int   = 128,
    n_heads:              int   = 8,
    n_layers:             int   = 3,
    d_ff:                 int   = 256,
    n_markers:            int   = 0,
    dropout:              float = 0.1,
    # --- training ---
    epochs:               int   = 30,
    batch_size:           int   = 16,
    lr:                   float = 1e-4,
    save_path:            str   = "patchtst_seg.pt",
):
    """
    Full end-to-end pipeline.

    Returns
    -------
    model    : trained PatchTSTSegmenter
    results  : list of DataFrames  [DEPTH, GR, predicted_zone]  per test well
    df_eval  : top-picking accuracy table
    """
    if top_cols is None:
        top_cols = list(df_tops_train.columns)
    n_classes = len(top_cols) + 1

    # ── 1. Prepare sequences ──────────────────────────────────────────────
    print("Preparing training sequences …")
    X_train, y_train, _ = prepare_sequences(
        df_log_train, df_tops_train, top_cols, srate, gr_clip,
        well_list=list(training_well_array),
    )

    print("Preparing validation sequences …")
    X_valid, y_valid, _ = prepare_sequences(
        df_log_valid, df_tops_valid, top_cols, srate, gr_clip,
    )

    # Pad both splits to the same seq_len
    seq_len = max(X_train.shape[1], X_valid.shape[1])

    def pad_to(X, y, L):
        if X.shape[1] < L:
            pad = np.zeros((len(X), L - X.shape[1]), dtype=X.dtype)
            ypad = y[:, -1:].repeat(L - y.shape[1], axis=1)
            X = np.concatenate([X, pad], axis=1)
            y = np.concatenate([y, ypad], axis=1)
        return X, y

    X_train, y_train = pad_to(X_train, y_train, seq_len)
    X_valid, y_valid = pad_to(X_valid, y_valid, seq_len)

    # ── 2. Build model ────────────────────────────────────────────────────
    print(f"\nBuilding PatchTSTSegmenter  seq_len={seq_len}  n_classes={n_classes}")
    model = PatchTSTSegmenter(
        seq_len   = seq_len,
        n_classes = n_classes,
        patch_len = patch_len,
        stride    = stride,
        d_model   = d_model,
        n_heads   = n_heads,
        n_layers  = n_layers,
        d_ff      = d_ff,
        n_markers = n_markers,
        dropout   = dropout,
    )

    # ── 3. Train ──────────────────────────────────────────────────────────
    model = train(
        model, X_train, y_train, X_valid, y_valid,
        epochs=epochs, batch_size=batch_size, lr=lr, save_path=save_path,
    )

    # ── 4. Inference on test set ──────────────────────────────────────────
    df_log_inf  = df_log_test  if df_log_test  is not None else df_log_valid
    df_tops_inf = df_tops_test if df_tops_test is not None else df_tops_valid

    print("\nPreparing test sequences …")
    X_test, y_test, depth_axes_test = prepare_sequences(
        df_log_inf, df_tops_inf, top_cols, srate, gr_clip,
    )
    X_test, y_test = pad_to(X_test, y_test, seq_len)

    test_wells = [
        w for w in (list(df_tops_inf.index))
        if w in set(df_log_inf["wellName"].unique())
        and not df_tops_inf.loc[w, top_cols].isnull().any()
    ]

    print("Running inference …")
    results = predict(model, X_test, depth_axes_test, batch_size=batch_size)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    df_eval = evaluate_top_picking(results, test_wells, df_tops_inf, top_cols)
    evaluate_segmentation_metrics(
    results, test_wells, df_tops_inf, top_cols)

    return model, results, df_eval


# ---------------------------------------------------------------------------
# EXAMPLE USAGE  (mirrors existing notebook)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if os.path.exists("patchtst_seg.pt"):
        os.remove("patchtst_seg.pt")
        print("Deleted old checkpoint")

    reset_seeds(42)

    # Load data
    df_log = pd.read_parquet("data/Training/logs.parquet",
                             engine="fastparquet")
    df_log.loc[df_log["GR"] < -1, "GR"] = -1
    df_loc = pd.read_parquet("data/Training/loc.parquet",
                             engine="fastparquet").reset_index()
    df_log = df_loc.merge(df_log, on="wellName")

    df_tops = pd.read_parquet("data/Training/tops.parquet",
                              engine="fastparquet").set_index("well_name")
    top_cols = ["NIOBARA", "CODELL", "FORT_HAYS"]
    df_tops = df_tops[top_cols].dropna()
    
    #_loaded = np.load("data/Training/training_well_list.npy", allow_pickle=True)
    #training_well_array = [float(w) for w in _loaded[0, 0]]

    df_valid_log = pd.read_parquet("data/testdata/logs_50.parquet",
                                   engine="fastparquet")
    df_valid_log.loc[df_valid_log["GR"] < -1,  "GR"] = -1
    df_valid_log.loc[df_valid_log["GR"] > 400, "GR"] = 400
    df_valid_loc = pd.read_parquet("data/testdata/loc_50.parquet",
                                   engine="fastparquet").reset_index()
    df_valid_log = df_valid_loc.merge(df_valid_log, on="wellName")
    df_valid_tops = pd.read_csv("data/testdata/tops_50.csv"
                                ).set_index("wellName")[top_cols]

    df_test_log = pd.read_parquet("data/testdata/logs_100.parquet",
                                  engine="fastparquet")
    df_test_log.loc[df_test_log["GR"] < -1,  "GR"] = -1
    df_test_log.loc[df_test_log["GR"] > 400, "GR"] = 400
    df_test_loc = pd.read_parquet("data/testdata/loc_100.parquet",
                                  engine="fastparquet").reset_index()
    df_test_log = df_test_loc.merge(df_test_log, on="wellName")
    df_test_tops = pd.read_csv("data/testdata/tops_100.csv"
                               ).set_index("wellName")[top_cols]
    print(type(df_log["wellName"].iloc[0]))
    print(df_log["wellName"].iloc[0])

    # Run pipeline
    model, results, df_eval = run_pipeline(
        df_log_train        = df_log,
        df_tops_train       = df_tops,
        training_well_array = list(df_tops.index),
        df_log_valid        = df_valid_log,
        df_tops_valid       = df_valid_tops,
        df_log_test         = df_test_log,
        df_tops_test        = df_test_tops,
        top_cols            = top_cols,
        srate               = 4.0,
        patch_len           = 16,
        stride              = 8,
        d_model             = 128,
        n_heads             = 8,
        n_layers            = 3,
        epochs              = 30,
        batch_size          = 16,
        lr                  = 1e-4,
    )

    print(df_eval.to_string(index=False))

    # Plot first test well
    plot_prediction(
        results[0],
        df_tops    = df_test_tops,
        top_cols   = top_cols,
        well_name  = str(df_test_tops.index[0]),
    )