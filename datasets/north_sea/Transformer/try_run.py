import json
import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# I am currently running it on kaggle gpu 
# at the moment which is why I have a kaggle path
#os.chdir("/kaggle/working")

def reset_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100

DEFAULT_LOG_COLS = ["GR_CLEAN", "RHOB_CLEAN", "DEEPRES_CLEAN"]

DEFAULT_TOP_COLS = [
    "HORDALAND",
    "ROGALAND",
    "SHETLAND",
    "VIKING",
    "VESTLAND",
    "DUNLIN",
]

# Half-width (in depth samples) around each boundary used to build edge labels.
# At srate=2.0m this gives a ±50m window around each top — wide enough for the
# model to see the signal but not so wide it blurs adjacent boundaries.
EDGE_HALF_WIDTH = 25


# ---------------------------------------------------------------------------
# 1.  DATASET
# ---------------------------------------------------------------------------

class WellLogDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, y_edge: np.ndarray):
        assert X.ndim == 3, f"Expected (n_wells, n_channels, seq_len), got {X.shape}"
        self.X      = torch.from_numpy(X.astype(np.float32))
        self.y      = torch.from_numpy(y.astype(np.int64))
        self.y_edge = torch.from_numpy(y_edge.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_edge[idx]


# ---------------------------------------------------------------------------
# SEGMENTATION HEAD
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """
    Per-patch MLP → upsample to seq_len.
    Input:  (bs, n_vars, feat_dim, patch_num)
    Output: (bs, seq_len, n_classes)
    """
    def __init__(self, feat_dim: int, n_vars: int, patch_num: int,
                 seq_len: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        in_dim = feat_dim * n_vars
        hidden = 256

        self.seq_len   = seq_len
        self.n_classes = n_classes
        self.patch_num = patch_num

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.upsample = nn.Upsample(size=seq_len, mode="linear", align_corners=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bs = z.size(0)
        z = z.permute(0, 3, 1, 2).reshape(bs, self.patch_num, -1)  # (bs, patch_num, in_dim)
        z = self.mlp(z)                                              # (bs, patch_num, n_classes)
        z = self.upsample(z.permute(0, 2, 1))                       # (bs, n_classes, seq_len)
        return z.permute(0, 2, 1)                                    # (bs, seq_len, n_classes)


# ---------------------------------------------------------------------------
# EDGE HEAD
# ---------------------------------------------------------------------------

class EdgeHead(nn.Module):
    """
    Binary head predicting P(depth sample is near a zone boundary).
    Shares backbone features with the segmentation head.
    Input:  (bs, n_vars, feat_dim, patch_num)
    Output: (bs, seq_len)   — raw logits (apply sigmoid for probability)
    """
    def __init__(self, feat_dim: int, n_vars: int, patch_num: int,
                 seq_len: int, dropout: float = 0.1):
        super().__init__()
        in_dim = feat_dim * n_vars
        hidden = 128

        self.seq_len   = seq_len
        self.patch_num = patch_num

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.upsample = nn.Upsample(size=seq_len, mode="linear", align_corners=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        bs = z.size(0)
        z = z.permute(0, 3, 1, 2).reshape(bs, self.patch_num, -1)  # (bs, patch_num, in_dim)
        z = self.mlp(z).squeeze(-1)                                  # (bs, patch_num)
        z = self.upsample(z.unsqueeze(1)).squeeze(1)                 # (bs, seq_len)
        return z


# ---------------------------------------------------------------------------
# FULL MODEL
# ---------------------------------------------------------------------------

class PatchTSTSegmenter(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_classes: int,
        n_channels: int = 1,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        n_markers: int = 0,
        dropout: float = 0.1,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from Transformer.PatchTST_backbone import PatchTST_backbone
        except ModuleNotFoundError:
            from PatchTST_backbone import PatchTST_backbone

        patch_num = int((seq_len - patch_len) / stride + 1) + 1
        feat_dim  = n_markers + 2

        self.backbone = PatchTST_backbone(
            c_in=n_channels,
            context_window=seq_len,
            target_window=seq_len,
            patch_len=patch_len,
            n_markers=n_markers,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            norm="BatchNorm",
            dropout=dropout,
            fc_dropout=dropout,
            head_dropout=head_dropout,
            head_type="flatten",
            individual=False,
            revin=False,
            padding_patch="end",
            pe="zeros",
            learn_pe=True,
            res_attention=True,
            rope=False,
            alibi=False,
        )
        del self.backbone.head

        self.seg_head  = SegmentationHead(
            feat_dim=feat_dim, n_vars=n_channels, patch_num=patch_num,
            seq_len=seq_len, n_classes=n_classes, dropout=head_dropout,
        )
        self.edge_head = EdgeHead(
            feat_dim=feat_dim, n_vars=n_channels, patch_num=patch_num,
            seq_len=seq_len, dropout=head_dropout,
        )

    def _encode(self, z: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        if b.padding_patch == "end":
            z = b.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=b.patch_len, step=b.stride)
        z = z.permute(0, 1, 3, 2)
        if b.key_padding_mask == "auto":
            z, kpm = b.key_padding_masking(z)
        else:
            kpm = None
        z = torch.nan_to_num(z)
        z = b.backbone(z, key_padding_mask_tensor=kpm)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (seg_logits, edge_logits) — shapes (bs, seq_len, n_classes) and (bs, seq_len)."""
        z = self._encode(x)
        return self.seg_head(z), self.edge_head(z)


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def build_tops_seq(
    depth_arr: np.ndarray,
    top_depths: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    n_tops = len(top_depths)
    labels = np.full(len(depth_arr), ignore_index, dtype=np.int64)

    known     = ~np.isnan(top_depths)
    known_idx = np.where(known)[0]

    if len(known_idx) == 0:
        return labels

    first_k = known_idx[0]
    if first_k == 0:
        labels[depth_arr < top_depths[0]] = 0

    last_k = known_idx[-1]
    if last_k == n_tops - 1:
        labels[depth_arr >= top_depths[last_k]] = n_tops

    for j in range(len(known_idx) - 1):
        ka = known_idx[j]
        kb = known_idx[j + 1]
        region = (depth_arr >= top_depths[ka]) & (depth_arr < top_depths[kb])
        if kb == ka + 1:
            labels[region] = ka + 1

    return labels


def build_edge_seq(
    zone_labels: np.ndarray,
    half_width: int = EDGE_HALF_WIDTH,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """
    Binary sequence: 1 where a zone boundary falls within `half_width` samples,
    0 elsewhere. Masked positions (ignore_index) remain -1 so we can skip them
    in the loss.
    """
    edge = np.zeros(len(zone_labels), dtype=np.float32)
    mask = zone_labels == ignore_index

    # Detect transitions between known labels
    valid = zone_labels.copy()
    valid[mask] = -999          # sentinel; transitions into/out of masked regions are ignored

    transitions = np.where(np.diff(valid) != 0)[0] + 1   # indices where zone changes
    # Filter to transitions between two known zones (not involving -999/ignore)
    real_trans = [t for t in transitions
                  if valid[t - 1] != -999 and valid[t] != -999]

    for t in real_trans:
        lo = max(0, t - half_width)
        hi = min(len(edge), t + half_width)
        edge[lo:hi] = 1.0

    # Mask unknown positions with -1 so BCE loss can ignore them
    edge[mask] = -1.0
    return edge


def compute_global_depth_bounds(
    df_tops: pd.DataFrame,
    top_cols: List[str],
    margin: float = 500.0,
    lower_pct: float = 25.0,
    upper_pct: float = 75.0,
) -> Tuple[float, float]:
    min_top_per_well = df_tops[top_cols].min(axis=1).dropna()
    max_top_per_well = df_tops[top_cols].max(axis=1).dropna()
    global_min = float(np.percentile(min_top_per_well, lower_pct)) - margin
    global_max = float(np.percentile(max_top_per_well, upper_pct)) + margin
    return global_min, global_max


def prepare_sequences(
    df_log: pd.DataFrame,
    df_tops: pd.DataFrame,
    top_cols: List[str],
    global_min: float,
    global_max: float,
    log_cols: List[str] = DEFAULT_LOG_COLS,
    depth_col: str = "DS_INDEX",
    well_col: str = "well_name",
    srate: float = 4.0,
    pad_value: float = 0.0,
    ignore_index: int = IGNORE_INDEX,
    well_list: Optional[List] = None,
    min_top_fraction: float = 0.65,
    edge_half_width: int = EDGE_HALF_WIDTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List]:
    wells = well_list if (well_list is not None and len(well_list) > 0) else list(df_tops.index)

    log_wells  = set(df_log[well_col].unique())
    tops_wells = set(df_tops.index)
    min_tops   = int(np.ceil(min_top_fraction * len(top_cols)))

    wells = [
        w for w in wells
        if w in log_wells
        and w in tops_wells
        and df_tops.loc[w, top_cols].notna().sum() >= min_tops
    ]

    depth_grid = np.arange(global_min, global_max + srate, srate, dtype=np.float32)
    seq_len    = len(depth_grid)
    n_channels = len(log_cols)

    X      = np.full((len(wells), n_channels, seq_len), pad_value, dtype=np.float32)
    y      = np.full((len(wells), seq_len), ignore_index, dtype=np.int64)
    y_edge = np.full((len(wells), seq_len), -1.0, dtype=np.float32)   # -1 = unknown
    depth_axes = []

    for i, well in enumerate(wells):
        wdf        = df_log[df_log[well_col] == well].sort_values(depth_col)
        depth_raw  = wdf[depth_col].values.astype(np.float32)

        if len(depth_raw) == 0:
            depth_axes.append(np.array([], dtype=np.float32))
            continue

        valid_mask = (depth_grid >= depth_raw.min()) & (depth_grid <= depth_raw.max())

        for c, col in enumerate(log_cols):
            values   = wdf[col].values.astype(np.float32)
            not_nan  = ~np.isnan(values)
            if not_nan.sum() > 1 and valid_mask.any():
                X[i, c, valid_mask] = np.interp(
                    depth_grid[valid_mask], depth_raw[not_nan], values[not_nan],
                ).astype(np.float32)

        top_depths  = df_tops.loc[well, top_cols].values.astype(float)
        labels_full = build_tops_seq(depth_grid, top_depths, ignore_index=ignore_index)

        labels = np.full(seq_len, ignore_index, dtype=np.int64)
        valid_and_labelled = valid_mask & (labels_full != ignore_index)
        labels[valid_and_labelled] = labels_full[valid_and_labelled]

        y[i]      = labels
        y_edge[i] = build_edge_seq(labels, half_width=edge_half_width, ignore_index=ignore_index)
        depth_axes.append(depth_grid[valid_mask].copy())

    print(
        f"Prepared {len(wells)} wells | seq_len={seq_len} | "
        f"n_channels={n_channels} | n_classes={len(top_cols) + 1} | "
        f"global range=[{global_min:.1f}, {global_max:.1f}]"
    )
    return X, y, y_edge, depth_axes, wells


# ---------------------------------------------------------------------------
# CLASS-BALANCED LOSS
# ---------------------------------------------------------------------------

# Using effective number of samples instead of inverse frequency for more balanced approach

def effective_number_weights(
    y: np.ndarray,
    n_classes: int,
    beta: float = 0.9999,
    ignore_index: int = IGNORE_INDEX,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    valid        = y[y != ignore_index].flatten()
    class_counts = np.bincount(valid, minlength=n_classes).astype(np.float64)
    class_counts = np.where(class_counts == 0, 1, class_counts)

    effective = 1.0 - np.power(beta, class_counts)
    weights   = (1.0 - beta) / effective

    print(f"Class counts:  {dict(enumerate(class_counts.astype(int)))}")
    print(f"Effective num: {dict(enumerate((effective / (1 - beta)).astype(int)))}")

    return torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def _edge_loss(edge_logits: torch.Tensor, y_edge: torch.Tensor) -> torch.Tensor:
    """BCE loss ignoring positions where y_edge == -1."""
    mask = y_edge >= 0
    if not mask.any():
        return edge_logits.sum() * 0.0
    return nn.functional.binary_cross_entropy_with_logits(
        edge_logits[mask], y_edge[mask],
    )


def train_one_epoch(
    model, loader, optimiser, device,
    class_weights=None, ignore_index=IGNORE_INDEX, edge_weight=1.0,
):
    model.train()
    seg_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    total_loss = 0.0

    for X_batch, y_batch, y_edge_batch in loader:
        X_batch      = X_batch.to(device)
        y_batch      = y_batch.to(device)
        y_edge_batch = y_edge_batch.to(device)

        optimiser.zero_grad()
        seg_logits, edge_logits = model(X_batch)

        loss_seg  = seg_criterion(seg_logits.permute(0, 2, 1), y_batch)
        loss_edge = _edge_loss(edge_logits, y_edge_batch)
        loss      = loss_seg + edge_weight * loss_edge

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, ignore_index=IGNORE_INDEX):
    model.eval()
    seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        seg_logits, _ = model(X_batch)
        loss           = seg_criterion(seg_logits.permute(0, 2, 1), y_batch)
        total_loss    += loss.item() * len(y_batch)

        preds = seg_logits.argmax(dim=-1)
        valid = y_batch != ignore_index
        correct += ((preds == y_batch) & valid).sum().item()
        total   += valid.sum().item()

    acc = correct / total if total > 0 else np.nan
    return total_loss / len(loader.dataset), acc


def train_model(
    model: PatchTSTSegmenter,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_edge_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    y_edge_valid: np.ndarray,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    beta: float = 0.9999,
    edge_weight: float = 1.0,
    save_path: str = "patchtst_seg.pt",
    device_str: str = "auto",
    ignore_index: int = IGNORE_INDEX,
) -> PatchTSTSegmenter:

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"Training on {device}")
    model     = model.to(device)
    n_classes = model.seg_head.n_classes

    class_weights = effective_number_weights(
        y_train, n_classes, beta=beta, ignore_index=ignore_index, device=device,
    )
    print(f"Class weights (beta={beta}): {class_weights.cpu().numpy()}")

    train_dl = DataLoader(
        WellLogDataset(X_train, y_train, y_edge_train),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    valid_dl = DataLoader(
        WellLogDataset(X_valid, y_valid, y_edge_valid),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_acc   = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_dl, optimiser, device,
            class_weights=class_weights, ignore_index=ignore_index,
            edge_weight=edge_weight,
        )
        val_loss, val_acc = evaluate(model, valid_dl, device, ignore_index=ignore_index)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
            print(f"  -> Saved best (val_acc={best_acc:.4f})")

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint.")

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def enforce_monotonic(zone_preds: np.ndarray, smooth_window: int = 11) -> np.ndarray:
    """
    Soft monotonic enforcement:
      1. Median-smooth the raw predictions to remove single-sample spikes —
         a noisy spike to zone 5 followed by a drop forces the hard clamp to
         skip zone 4 entirely.  Smoothing prevents this.
      2. Then apply the hard monotonic clamp (sequence can never decrease).

    smooth_window is in samples; at srate=2.0 the default 11 = ±10m.
    Keep it odd so the window is symmetric.
    """
    from scipy.ndimage import median_filter
    smoothed = median_filter(zone_preds.astype(np.float32), size=smooth_window).astype(np.int64)
    out = smoothed.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


@torch.no_grad()
def predict(
    model: PatchTSTSegmenter,
    X: np.ndarray,
    depth_axes: List[np.ndarray],
    log_cols: List[str] = DEFAULT_LOG_COLS,
    batch_size: int = 16,
) -> List[pd.DataFrame]:
    device = next(model.parameters()).device
    model.eval()
    assert X.ndim == 3
    n_classes = model.seg_head.n_classes

    results = []
    for start in range(0, len(X), batch_size):
        batch             = torch.from_numpy(X[start:start + batch_size]).to(device)
        seg_logits, edge_logits = model(batch)

        seg_probs  = torch.softmax(seg_logits, dim=-1).cpu().numpy()   # (bs, seq_len, n_classes)
        edge_probs = torch.sigmoid(edge_logits).cpu().numpy()           # (bs, seq_len)
        preds      = seg_logits.argmax(dim=-1).cpu().numpy()

        for i, pred_seq in enumerate(preds):
            well_idx  = start + i
            depth_arr = depth_axes[well_idx]
            valid_len = len(depth_arr)

            zone_preds = enforce_monotonic(pred_seq[:valid_len])

            row_dict = {"DEPTH": depth_arr}
            for c, col in enumerate(log_cols):
                row_dict[col] = X[well_idx, c, :valid_len]
            row_dict["predicted_zone"] = zone_preds
            row_dict["edge_prob"]      = edge_probs[i, :valid_len]
            for k in range(n_classes):
                row_dict[f"prob_zone_{k}"] = seg_probs[i, :valid_len, k]

            results.append(pd.DataFrame(row_dict))

    return results


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def top_depths_from_step_function(
    depth_arr: np.ndarray,
    zone_preds: np.ndarray,
    n_tops: int,
) -> np.ndarray:
    est = []
    for k in range(1, n_tops + 1):
        idx = np.where((zone_preds[:-1] < k) & (zone_preds[1:] >= k))[0]
        est.append(float(depth_arr[idx[0]]) if len(idx) else np.nan)
    return np.array(est)


def evaluate_top_picking(
    results: List[pd.DataFrame],
    well_names: List,
    df_tops: pd.DataFrame,
    top_cols: List[str],
) -> pd.DataFrame:
    rows = []
    for well, df in zip(well_names, results):
        if well not in df_tops.index:
            continue
        est = top_depths_from_step_function(
            df["DEPTH"].values, df["predicted_zone"].values, len(top_cols),
        )
        gt  = df_tops.loc[well, top_cols].values.astype(float)
        row = {"well": well}
        for i, col in enumerate(top_cols):
            row[f"{col}_gt"]   = gt[i]
            row[f"{col}_pred"] = est[i]
            row[f"{col}_err"]  = abs(est[i] - gt[i]) if not (np.isnan(gt[i]) or np.isnan(est[i])) else np.nan
        rows.append(row)

    df_eval = pd.DataFrame(rows)

    print("\nTop-picking errors (only where ground truth exists):")
    print(f"  {'Top':<15} {'MAE':>8}  {'Median':>8}  {'n':>6}")
    print(f"  {'-'*15} {'-'*8}  {'-'*8}  {'-'*6}")
    for col in top_cols:
        err_col = f"{col}_err"
        if err_col in df_eval.columns:
            valid = df_eval[err_col].dropna()
            mae    = valid.mean()
            median = valid.median()
            print(f"  {col:<15} {mae:>8.1f}  {median:>8.1f}  {len(valid):>6}")

    return df_eval


def evaluate_segmentation_metrics(
    results: List[pd.DataFrame],
    well_names: List,
    df_tops: pd.DataFrame,
    top_cols: List[str],
):
    all_preds, all_true = [], []

    for well, df in zip(well_names, results):
        if well not in df_tops.index:
            continue
        top_depths  = df_tops.loc[well, top_cols].values.astype(float)
        true_labels = build_tops_seq(df["DEPTH"].values, top_depths, ignore_index=IGNORE_INDEX)
        pred_labels = df["predicted_zone"].values

        known = true_labels != IGNORE_INDEX
        all_true.extend(true_labels[known].tolist())
        all_preds.extend(pred_labels[known].tolist())

    all_true  = np.array(all_true)
    all_preds = np.array(all_preds)

    n_classes     = len(top_cols) + 1
    zone_names    = [f"zone_{i}" for i in range(n_classes)]
    present       = sorted(set(all_true) | set(all_preds))
    present_names = [zone_names[c] for c in present if c < n_classes]

    accuracy = (all_true == all_preds).mean()

    # Per-class recall + macro average
    per_class_recall = {}
    for c in present:
        mask_c = all_true == c
        per_class_recall[c] = float((all_preds[mask_c] == c).mean()) if mask_c.sum() > 0 else np.nan
    macro_recall = float(np.nanmean(list(per_class_recall.values())))

    print(f"\nOverall depth-sample accuracy : {accuracy:.4f}  (on {len(all_true):,} labelled points)")
    print(f"Macro-average recall          : {macro_recall:.4f}")
    recall_str = "  ".join(f"{zone_names[c]}={per_class_recall[c]:.3f}" for c in present if c < n_classes)
    print(f"Per-class recall              : {recall_str}")

    print("\nPer-class metrics:")
    print(classification_report(
        all_true, all_preds,
        labels=present,
        target_names=present_names,
        zero_division=0,
    ))

    cm = confusion_matrix(all_true, all_preds, labels=list(range(n_classes)))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=zone_names, columns=zone_names))

    return accuracy, macro_recall


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------

def plot_prediction(
    df_result: pd.DataFrame,
    log_cols: List[str] = DEFAULT_LOG_COLS,
    df_tops: Optional[pd.DataFrame] = None,
    top_cols: Optional[List[str]] = None,
    well_name: str = "",
    save_fig: Optional[str] = None,
    show_probs: bool = True,
):
    """
    Plot well logs, predicted zone step function, class probability curves,
    and (optionally) the edge probability curve.

    Columns expected in df_result:
      DEPTH, <log_cols>, predicted_zone, edge_prob, prob_zone_0, prob_zone_1, ...
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    depth      = df_result["DEPTH"].values
    zone       = df_result["predicted_zone"].values
    edge_prob  = df_result.get("edge_prob", None)

    prob_cols  = [c for c in df_result.columns if c.startswith("prob_zone_")]
    n_logs     = len([c for c in log_cols if c in df_result.columns])

    # Axes layout: logs | zone step | prob curves | edge prob
    n_prob_axes  = 1 if (show_probs and prob_cols) else 0
    n_edge_axes  = 1 if (show_probs and edge_prob is not None) else 0
    n_axes       = n_logs + 1 + n_prob_axes + n_edge_axes

    fig, axes = plt.subplots(1, n_axes, figsize=(3 * n_axes, 10), sharey=True)
    if n_axes == 1:
        axes = [axes]

    colors   = plt.cm.tab10.colors
    ax_idx   = 0

    # ---- Log tracks ----
    for col in log_cols:
        if col not in df_result.columns:
            continue
        ax = axes[ax_idx]
        ax.plot(df_result[col].values, depth, color="steelblue", lw=0.8)
        ax.set_xlabel(col)
        ax.set_title(col)
        if ax_idx == 0:
            ax.set_ylabel("Depth")
        ax.invert_yaxis()
        ax_idx += 1

    # ---- Zone step function ----
    ax_zone = axes[ax_idx]
    ax_zone.step(zone, depth, where="post", color="steelblue", lw=1.5)
    ax_zone.set_xlabel("Predicted zone")
    ax_zone.set_title("Zone")
    ax_zone.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_zone.invert_yaxis()
    ax_idx += 1

    # ---- Probability curves ----
    if n_prob_axes:
        ax_prob = axes[ax_idx]
        for k, pc in enumerate(prob_cols):
            ax_prob.plot(df_result[pc].values, depth,
                         color=colors[k % 10], lw=0.8, label=pc.replace("prob_", ""))
        ax_prob.set_xlim(0, 1)
        ax_prob.set_xlabel("Prob")
        ax_prob.set_title("Zone probs")
        ax_prob.legend(fontsize=6, loc="lower right")
        ax_prob.invert_yaxis()
        ax_idx += 1

    # ---- Edge probability ----
    if n_edge_axes:
        ax_edge = axes[ax_idx]
        ax_edge.plot(df_result["edge_prob"].values, depth, color="crimson", lw=0.9)
        ax_edge.set_xlim(0, 1)
        ax_edge.set_xlabel("Edge prob")
        ax_edge.set_title("Edge")
        ax_edge.invert_yaxis()

    # ---- Overlay GT tops on every axis ----
    if df_tops is not None and top_cols and well_name in df_tops.index:
        for i, col in enumerate(top_cols):
            td = df_tops.loc[well_name, col]
            if not np.isnan(td):
                for ax in axes:
                    ax.axhline(td, color=colors[i % 10], ls="--", lw=1, alpha=0.8, label=col)
        axes[-1].legend(fontsize=6, loc="lower right")

    fig.suptitle(f"Well {well_name}", fontsize=12)
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(
    # --- data ---
    df_log_train: pd.DataFrame,
    df_tops_train: pd.DataFrame,
    training_well_list: List,
    df_log_valid: pd.DataFrame,
    df_tops_valid: pd.DataFrame,
    df_log_test: Optional[pd.DataFrame] = None,
    df_tops_test: Optional[pd.DataFrame] = None,
    top_cols: Optional[List[str]] = None,
    log_cols: List[str] = DEFAULT_LOG_COLS,
    depth_col: str = "DS_INDEX",
    well_col: str = "well_name",
    # --- resampling ---
    srate: float = 4,
    margin: float = 500.0,
    min_top_fraction: float = 0.65,
    # --- depth window percentiles ---
    lower_pct: float = 25.0,
    upper_pct: float = 75.0,
    # --- model ---
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    d_ff: int = 256,
    n_markers: int = 0,
    dropout: float = 0.1,
    # --- training ---
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    beta: float = 0.9999,
    edge_weight: float = 1.0,
    save_path: str = "patchtst_seg.pt",
):
    if top_cols is None:
        top_cols = DEFAULT_TOP_COLS

    n_classes  = len(top_cols) + 1
    n_channels = len(log_cols)

    # 1. Global depth bounds
    global_min, global_max = compute_global_depth_bounds(
        df_tops_train, top_cols, margin=margin,
        lower_pct=lower_pct, upper_pct=upper_pct,
    )
    print(f"Depth window (pct=[{lower_pct},{upper_pct}]): [{global_min:.1f}, {global_max:.1f}]")

    common_kw = dict(
        top_cols=top_cols,
        global_min=global_min,
        global_max=global_max,
        log_cols=log_cols,
        depth_col=depth_col,
        well_col=well_col,
        srate=srate,
        ignore_index=IGNORE_INDEX,
        min_top_fraction=min_top_fraction,
    )

    # 2. Prepare sequences
    print("\n--- Training ---")
    X_train, y_train, y_edge_train, _, train_wells = prepare_sequences(
        df_log_train, df_tops_train, well_list=training_well_list, **common_kw,
    )

    print("\n--- Validation ---")
    X_valid, y_valid, y_edge_valid, _, valid_wells = prepare_sequences(
        df_log_valid, df_tops_valid, **common_kw,
    )

    seq_len = X_train.shape[2]
    assert X_valid.shape[2] == seq_len, "Sequence length mismatch"

    # 3. Build model
    print(f"\nBuilding PatchTSTSegmenter | seq_len={seq_len} | "
          f"n_channels={n_channels} | n_classes={n_classes}")

    model = PatchTSTSegmenter(
        seq_len=seq_len,
        n_classes=n_classes,
        n_channels=n_channels,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_markers=n_markers,
        dropout=dropout,
    )

    # 4. Train
    model = train_model(
        model=model,
        X_train=X_train, y_train=y_train, y_edge_train=y_edge_train,
        X_valid=X_valid, y_valid=y_valid, y_edge_valid=y_edge_valid,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        beta=beta,
        edge_weight=edge_weight,
        save_path=save_path,
        ignore_index=IGNORE_INDEX,
    )

    # 5. Inference set
    df_log_inf  = df_log_test  if df_log_test  is not None else df_log_valid
    df_tops_inf = df_tops_test if df_tops_test is not None else df_tops_valid

    print("\n--- Inference ---")
    X_inf, y_inf, _, depth_axes_inf, inf_wells = prepare_sequences(
        df_log_inf, df_tops_inf, **common_kw,
    )
    assert X_inf.shape[2] == seq_len, "Inference seq_len mismatch"

    # 6. Predict
    results = predict(model, X_inf, depth_axes_inf, log_cols=log_cols, batch_size=batch_size)

    # 7. Evaluate
    df_eval = evaluate_top_picking(results, inf_wells, df_tops_inf, top_cols)
    accuracy, macro_recall = evaluate_segmentation_metrics(results, inf_wells, df_tops_inf, top_cols)

    return model, results, df_eval, inf_wells, accuracy, macro_recall


if __name__ == "__main__":
    print("Loading prepared Parquet datasets...")
    base_dir = "/Users/fokia/CRP_SLB/datasets/north_sea/Transformer/"
    
    df_log_train  = pd.read_parquet(os.path.join(base_dir, "log_train.parquet"))
    df_tops_train = pd.read_parquet(os.path.join(base_dir, "tops_train.parquet"))
    df_log_valid  = pd.read_parquet(os.path.join(base_dir, "log_valid.parquet"))
    df_tops_valid = pd.read_parquet(os.path.join(base_dir, "tops_valid.parquet"))

    print("Launching Transformer pipeline...")
    
    log_cols = ["GR_CLEAN", "RHOB_CLEAN", "DEEPRES_CLEAN"]  

    model, results, df_eval, inf_wells, accuracy, macro_recall = run_pipeline(
        df_log_train=df_log_train,
        df_tops_train=df_tops_train,
        training_well_list=list(df_tops_train.index),
        df_log_valid=df_log_valid,
        df_tops_valid=df_tops_valid,
        log_cols=log_cols,
        depth_col="DS_INDEX",
        well_col="well_name",
        epochs=100,
        batch_size=8,
        lr=1e-4,
        beta=0.99999999,    # stronger minority upweighting
        edge_weight=6.0, # push harder to find zone boundaries
        patch_len=16,    # larger patch = more geological context per token
        stride=8,
    )
    
    print("\nTraining completed successfully!")

    # --- Experiment Tracking ---
    try:
        comment = input("\nExperiment Comment (press Enter to skip): ")
    except EOFError:
        comment = ""
        
    # Calculate average MAE
    err_cols = [c for c in df_eval.columns if c.endswith("_err")]
    avg_mae = df_eval[err_cols].mean().mean()
    
    import csv
    from datetime import datetime
    csv_file = os.path.join(base_dir, "experiment_log.csv")
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Comment", "Accuracy", "Macro_Recall", "Avg_MAE", "Features"])
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            comment,
            f"{accuracy:.4f}",
            f"{macro_recall:.4f}",
            f"{avg_mae:.2f}",
            "|".join(log_cols)
        ])
    print(f"Logged run metrics to {csv_file}")

    # Plot a random well from the validation/inference set
    import random
    if inf_wells and results:
        idx = random.randint(0, len(inf_wells) - 1)
        rand_well = inf_wells[idx]
        rand_result = results[idx]
        
        print(f"\nPlotting predictions for random well: {rand_well}")
        # Save it to the active directory so you can view it
        save_path = os.path.join(base_dir, f"prediction_{rand_well.replace('/', '_')}.png")
        
        plot_prediction(
            df_result=rand_result,
            log_cols=log_cols,
            df_tops=df_tops_valid,
            top_cols=DEFAULT_TOP_COLS,
            well_name=rand_well,
            save_fig=save_path,
            show_probs=True
        )
        print(f"Saved plot to: {save_path}")
