import warnings, time, os, pickle
warnings.filterwarnings("ignore")
os.environ["CODECARBON_LOG_LEVEL"] = "error"

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import dgl, dgl.nn.pytorch as dglnn
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, classification_report)
from codecarbon import EmissionsTracker

# ── Data ──────────────────────────────────────────────────────────────────
base_dir = "/Users/fokia/CRP_SLB/datasets/north_sea/Transformer/"
df_log_train  = pd.read_parquet(os.path.join(base_dir, "log_train.parquet"))
df_tops_train = pd.read_parquet(os.path.join(base_dir, "tops_train.parquet"))
df_log_valid  = pd.read_parquet(os.path.join(base_dir, "log_valid.parquet"))
df_tops_valid = pd.read_parquet(os.path.join(base_dir, "tops_valid.parquet"))

MARKER_COLS    = ['DUNLIN','HORDALAND','ROGALAND','SHETLAND','VESTLAND','VIKING']
INPUT_FEATURES = ['GR_CLEAN','RHOB_CLEAN','DEEPRES_CLEAN']
N_CLASSES      = len(MARKER_COLS) + 1   # 7

df_tops_train = df_tops_train[MARKER_COLS]
df_tops_valid = df_tops_valid[MARKER_COLS]
print("Data loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────
def well_log_cleaning(wn, df_log, feats, depth_col='DS_INDEX', well_col='well_name'):
    df = df_log[df_log[well_col]==wn][feats+[depth_col]].copy()
    if df.empty: return df
    df = df.sort_values(depth_col).reset_index(drop=True)
    df[feats] = df[feats].ffill().bfill().fillna(df[feats].median())
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

def create_graph(df, sv, depth_col='DS_INDEX', feats=INPUT_FEATURES):
    n=min(len(df),len(sv))
    if n==0: return dgl.graph(([],[]))
    df=df.iloc[:n]; sv=sv[:n]
    g=dgl.graph((list(range(n-1)),list(range(1,n))))
    g.ndata['features']=torch.tensor(df[feats].values.astype(np.float32))
    g.ndata['pos']=torch.tensor(df[[depth_col]].values.astype(np.float32))
    g.ndata['label']=torch.tensor([t[1] for t in sv],dtype=torch.int64)
    return g

class WellGraphDataset(Dataset):
    def __init__(self,df_log,df_tops,feats,max_wells=None):
        lw=set(df_log['well_name'].unique()); tw=set(df_tops.index)
        wells=sorted(lw&tw)
        if max_wells: wells=wells[:max_wells]
        self.wells=wells; self.df_log=df_log; self.df_tops=df_tops; self.feats=feats
        print(f"  {len(wells)} wells")
    def __len__(self): return len(self.wells)
    def __getitem__(self, idx):
        w = self.wells[idx]
        df = well_log_cleaning(w, self.df_log, self.feats)
        if df.empty: return self._empty()
        
        # --- NEW: Fixed Global Label Mapping ---
        MARKER_TO_CLASS = {
            'background': 0,
            'DUNLIN': 1,
            'HORDALAND': 2,
            'ROGALAND': 3,
            'SHETLAND': 4,
            'VESTLAND': 5,
            'VIKING': 6
        }
        tops_series = self.df_tops.loc[w] if w in self.df_tops.index else pd.Series()
        well_tops = tops_series.dropna().to_dict()
        sorted_tops = sorted(well_tops.items(), key=lambda x: x[1])  # Sort tops by depth
        
        depths = df['DS_INDEX'].values
        fixed_labels = []
        for d in depths:
            active_marker = 'background'
            for marker_name, top_depth in sorted_tops:
                if d >= top_depth:
                    active_marker = marker_name
                else:
                    break
            fixed_labels.append(MARKER_TO_CLASS[active_marker])
            
        g = create_graph(df, [(depths[i], fixed_labels[i]) for i in range(len(depths))])
        g.ndata['label'] = torch.tensor(fixed_labels, dtype=torch.int64)
        return g

    def _empty(self):
        g=dgl.graph(([],[])); g.add_nodes(1)
        g.ndata['features']=torch.zeros(1,len(self.feats))
        g.ndata['pos']=torch.zeros(1,1)
        g.ndata['label']=torch.tensor([0],dtype=torch.int64); return g

def collate(s):
    v=[g for g in s if g.num_nodes()>0]
    return dgl.batch(v) if v else dgl.graph(([],[]))

print("Building datasets...")
train_ds = WellGraphDataset(df_log_train,df_tops_train,INPUT_FEATURES)
valid_ds  = WellGraphDataset(df_log_valid, df_tops_valid, INPUT_FEATURES)
train_dl  = DataLoader(train_ds,batch_size=4,shuffle=True, collate_fn=collate)
valid_dl  = DataLoader(valid_ds,batch_size=4,shuffle=False,collate_fn=collate)

# ── Post-processing ────────────────────────────────────────────────────────
def predict_with_mono(logits, window=11, threshold=0.5):
    """Robust dynamic boundary snapping + overburden lock."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    smoothed = scipy.ndimage.uniform_filter1d(probs, size=window, axis=0)
    s1_preds = smoothed.argmax(axis=1)
    
    # 1. Geological Overburden Lock
    last_formation = None
    cleaned_preds = s1_preds.copy()
    for i in range(len(cleaned_preds)):
        if cleaned_preds[i] != 0:
            last_formation = cleaned_preds[i]
        elif cleaned_preds[i] == 0 and last_formation is not None:
            cleaned_preds[i] = last_formation
    s1_preds = cleaned_preds

    # 2. Dynamic Monotonic Boundary Alignment
    depths = np.arange(len(s1_preds))
    class_depths = {c: depths[s1_preds == c] for c in range(NC) if (s1_preds == c).sum() > 0}
    sorted_classes = sorted(class_depths, key=lambda c: class_depths[c].mean())

    boundaries = [
        (class_depths[sorted_classes[i]].max() + class_depths[sorted_classes[i + 1]].min()) / 2
        for i in range(len(sorted_classes) - 1)
    ]

    fused = s1_preds.copy()
    for i in range(len(fused)):
        d = depths[i]
        class_idx = len(sorted_classes) - 1
        for j, b in enumerate(boundaries):
            if d < b:
                class_idx = j
                break
        fused[i] = sorted_classes[class_idx]
        
    return fused


# ── Post-processing ────────────────────────────────────────────────────────
def predict_no_mono(logits, window=11):
    """Smooth probabilities then take argmax — no ordering constraint."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    smoothed = scipy.ndimage.uniform_filter1d(probs, size=window, axis=0)
    return smoothed.argmax(axis=1)

def predict_with_mono(logits, window=11, threshold=0.5):
    """Robust dynamic boundary snapping + overburden lock."""
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    smoothed = scipy.ndimage.uniform_filter1d(probs, size=window, axis=0)
    s1_preds = smoothed.argmax(axis=1)
    
    # 1. Geological Overburden Lock
    last_formation = None
    cleaned_preds = s1_preds.copy()
    for i in range(len(cleaned_preds)):
        if cleaned_preds[i] != 0:
            last_formation = cleaned_preds[i]
        elif cleaned_preds[i] == 0 and last_formation is not None:
            cleaned_preds[i] = last_formation
    s1_preds = cleaned_preds

    # 2. Dynamic Monotonic Boundary Alignment
    depths = np.arange(len(s1_preds))
    class_depths = {c: depths[s1_preds == c] for c in range(NC) if (s1_preds == c).sum() > 0}
    sorted_classes = sorted(class_depths, key=lambda c: class_depths[c].mean())

    boundaries = [
        (class_depths[sorted_classes[i]].max() + class_depths[sorted_classes[i + 1]].min()) / 2
        for i in range(len(sorted_classes) - 1)
    ]

    fused = s1_preds.copy()
    for i in range(len(fused)):
        d = depths[i]
        class_idx = len(sorted_classes) - 1
        for j, b in enumerate(boundaries):
            if d < b:
                class_idx = j
                break
        fused[i] = sorted_classes[class_idx]
        
    return fused


# ── Models ────────────────────────────────────────────────────────────────
class ImprovedGNN(nn.Module):
    def __init__(self,inf,hs,nc,dropout=0.1):
        super().__init__()
        self.convs=nn.ModuleList(); self.bns=nn.ModuleList(); prev=inf
        for h in hs:
            self.convs.append(dglnn.GraphConv(prev,h))
            self.bns.append(nn.BatchNorm1d(h)); prev=h
        self.fc=nn.Linear(prev,nc); self.drop=nn.Dropout(dropout)
    def forward(self,g,x):
        h=x
        for c,b in zip(self.convs,self.bns): h=self.drop(F.leaky_relu(b(c(g,h))))
        return self.fc(h)

class ExtendedGAT(nn.Module):
    def __init__(self,inf,h,nc,heads=4,dropout=0.1):
        super().__init__()
        self.g1=dgl.nn.GATConv(inf,h,heads); self.g2=dgl.nn.GATConv(h*heads,h,heads)
        self.g3=dgl.nn.GATConv(h*heads,h,heads)
        self.bn1=nn.BatchNorm1d(h*heads); self.bn2=nn.BatchNorm1d(h*heads)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(h*heads,nc)
    def forward(self,g,x):
        h=self.drop(self.bn1(F.elu(self.g1(g,x)).view(x.size(0),-1)))
        h=self.drop(self.bn2(F.elu(self.g2(g,h)).view(h.size(0),-1)))
        h=self.drop(F.elu(self.g3(g,h)).view(h.size(0),-1))
        return self.fc(h)

class GraphSAGE(nn.Module):
    def __init__(self,inf,h,nc,dropout=0.1):
        super().__init__()
        self.c1=dglnn.SAGEConv(inf,h,'mean'); self.c2=dglnn.SAGEConv(h,h,'mean')
        self.c3=dglnn.SAGEConv(h,h,'mean')
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.bn3=nn.BatchNorm1d(h)
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(h,nc)
    def forward(self,g,x):
        h=self.drop(F.relu(self.bn1(self.c1(g,x))))
        h=self.drop(F.relu(self.bn2(self.c2(g,h))))
        h=self.drop(F.relu(self.bn3(self.c3(g,h))))
        return self.fc(h)

class ExtendedGraphSAGE(nn.Module):
    def __init__(self,inf,h,nc,dropout=0.1):
        super().__init__()
        self.convs=nn.ModuleList([dglnn.SAGEConv(inf if i==0 else h,h,'mean') for i in range(5)])
        self.bns=nn.ModuleList([nn.BatchNorm1d(h) for _ in range(5)])
        self.drop=nn.Dropout(dropout); self.fc=nn.Linear(h,nc)
    def forward(self,g,x):
        h=x; skip=None
        for i,(c,b) in enumerate(zip(self.convs,self.bns)):
            hn=self.drop(F.relu(b(c(g,h))))
            if i==1: skip=hn
            if i==3 and skip is not None: hn=hn+skip
            h=hn
        return self.fc(h)

# ── Training & evaluation ─────────────────────────────────────────────────
supports  = torch.tensor([1000,800,600,500,400,300,200],dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=supports.max()/supports)
EPOCHS=15; LR=0.005; NC=N_CLASSES; H=128

def make_model(name):
    inf=len(INPUT_FEATURES)
    if name=='ImprovedGNN':       return ImprovedGNN(inf,[128,256,256,128,64],NC)
    if name=='ExtendedGAT':       return ExtendedGAT(inf,64,NC,heads=4)
    if name=='GraphSAGE':         return GraphSAGE(inf,H,NC)
    if name=='ExtendedGraphSAGE': return ExtendedGraphSAGE(inf,H,NC)

def train_model(m, dl, epochs):
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    m.train()
    for ep in range(epochs):
        tl=0; preds=[]; labs=[]
        for bg in dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg); opt.zero_grad()
            feat=bg.ndata['features']
            lbl=torch.clamp(bg.ndata['label']-1,0,NC-1)
            out=m(bg,feat); loss=criterion(out,lbl)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step()
            tl+=loss.item(); preds+=out.argmax(1).tolist(); labs+=lbl.tolist()
        from sklearn.metrics import accuracy_score, f1_score
        avg=tl/max(len(dl),1)
        acc=accuracy_score(labs,preds)
        f1=f1_score(labs,preds,average='weighted',zero_division=0)
        print(f"    ep {ep+1:2d}/{epochs}  loss={avg:.4f}  acc={acc:.4f}  f1={f1:.4f}")
        sch.step()

def collect_logits(m, dl):
    m.eval(); all_logits=[]; all_labels=[]
    with torch.no_grad():
        for bg in dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg)
            logits=m(bg,bg.ndata['features']).cpu().numpy()
            labs=np.clip(bg.ndata['label'].cpu().numpy()-1,0,NC-1)
            all_logits.append(logits); all_labels.append(labs)
    return np.vstack(all_logits), np.concatenate(all_labels)

def score(preds, labels):
    return dict(
        acc  = accuracy_score(labels,preds),
        f1   = f1_score(labels,preds,average='weighted',zero_division=0),
        prec = precision_score(labels,preds,average='weighted',zero_division=0),
        rec  = recall_score(labels,preds,average='weighted',zero_division=0),
    )

# ── Main sweep ─────────────────────────────────────────────────────────────
MODEL_NAMES = ['ImprovedGNN','ExtendedGAT','GraphSAGE','ExtendedGraphSAGE']
results = {}

for name in MODEL_NAMES:
    print(f"\n{'='*55}")
    print(f"  {name}  |  lr={LR}  |  epochs={EPOCHS}")
    print(f"{'='*55}")
    m = make_model(name)

    tracker = EmissionsTracker(project_name=f"GNN_{name}",
                               output_dir="/tmp", log_level="error",
                               save_to_file=False)
    tracker.start(); t0=time.time()
    train_model(m, train_dl, EPOCHS)
    co2     = tracker.stop()
    elapsed = time.time()-t0

    print("  Evaluating...")
    logits, labels = collect_logits(m, valid_dl)

    preds_no_mono   = predict_no_mono(logits,   window=11)
    preds_with_mono = predict_with_mono(logits, window=11, threshold=0.5)

    s_no   = score(preds_no_mono,   labels)
    s_with = score(preds_with_mono, labels)

    results[name] = {
        'no_mono':   s_no,
        'with_mono': s_with,
        'train_s':   elapsed,
        'co2_g':     (co2 or 0)*1000,
    }

print(f"\n  No-mono  : acc={s_no['acc']:.4f}  f1={s_no['f1']:.4f}  "
        f"prec={s_no['prec']:.4f}  rec={s_no['rec']:.4f}")
print(f"  Mono     : acc={s_with['acc']:.4f}  f1={s_with['f1']:.4f}  "
        f"prec={s_with['prec']:.4f}  rec={s_with['rec']:.4f}")
print(f"  Time: {elapsed:.1f}s   CO2: {(co2 or 0)*1000:.5f} g")

# ── NEW: Print detailed per-class classification reports ──
CLASS_NAMES = ['background', 'DUNLIN', 'HORDALAND', 'ROGALAND', 'SHETLAND', 'VESTLAND', 'VIKING']
print(f"\n  ── {name} Classification Report (No Monotonicity) ──")
print(classification_report(labels, preds_no_mono, target_names=CLASS_NAMES, labels=list(range(NC)), zero_division=0))

print(f"  ── {name} Classification Report (With Dynamic Monotonicity) ──")
print(classification_report(labels, preds_with_mono, target_names=CLASS_NAMES, labels=list(range(NC)), zero_division=0))


# ── Summary table ─────────────────────────────────────────────────────────
print("\n\n" + "="*85)
print(f"{'Model':<22}  {'── No Monotonicity ──':^28}  {'── With Monotonicity ──':^28}")
print(f"{'':22}  {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}  "
      f"{'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}  {'CO2(g)':>9}")
print("="*85)
for name, r in results.items():
    n=r['no_mono']; w=r['with_mono']
    print(f"{name:<22}  {n['acc']:>6.4f} {n['f1']:>6.4f} {n['prec']:>6.4f} {n['rec']:>6.4f}  "
          f"{w['acc']:>6.4f} {w['f1']:>6.4f} {w['prec']:>6.4f} {w['rec']:>6.4f}  "
          f"{r['co2_g']:>9.5f}")
print("="*85)

# Save
with open('/tmp/gnn_mono_compare.pkl','wb') as f: pickle.dump(results,f)
print("\nSaved to /tmp/gnn_mono_compare.pkl")
