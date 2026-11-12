import warnings, time, os, sys
warnings.filterwarnings("ignore")
os.environ["CODECARBON_LOG_LEVEL"] = "error"

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
import dgl, dgl.nn.pytorch as dglnn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codecarbon import EmissionsTracker
import scipy.ndimage

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
    src=list(range(n-1)); dst=list(range(1,n))
    g=dgl.graph((src,dst))
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
    def __getitem__(self,idx):
        w=self.wells[idx]
        df=well_log_cleaning(w,self.df_log,self.feats)
        if df.empty: return self._empty()
        tops=self.df_tops.loc[w].dropna().tolist() if w in self.df_tops.index else []
        sv=step_function(float(df['DS_INDEX'].min()),float(df['DS_INDEX'].max()),sorted(tops))
        g=create_graph(df,sv)
        return g if g.num_nodes()>0 else self._empty()
    def _empty(self):
        g=dgl.graph(([],[])); g.add_nodes(1)
        g.ndata['features']=torch.zeros(1,len(self.feats)); g.ndata['pos']=torch.zeros(1,1)
        g.ndata['label']=torch.tensor([0],dtype=torch.int64); return g

def collate(s):
    v=[g for g in s if g.num_nodes()>0]
    return dgl.batch(v) if v else dgl.graph(([],[]))

print("Building datasets...")
train_ds = WellGraphDataset(df_log_train,df_tops_train,INPUT_FEATURES)
valid_ds = WellGraphDataset(df_log_valid,df_tops_valid,INPUT_FEATURES)
train_dl = DataLoader(train_ds,batch_size=4,shuffle=True,collate_fn=collate)
valid_dl = DataLoader(valid_ds,batch_size=4,shuffle=False,collate_fn=collate)

# ── Models ─────────────────────────────────────────────────────────────────
class ImprovedGNN(nn.Module):
    def __init__(self,inf,hs,nc,dropout=0.1):
        super().__init__()
        self.convs=nn.ModuleList(); self.bns=nn.ModuleList(); prev=inf
        for h in hs:
            self.convs.append(dglnn.GraphConv(prev,h)); self.bns.append(nn.BatchNorm1d(h)); prev=h
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

# ── Training ───────────────────────────────────────────────────────────────
supports=torch.tensor([1000,800,600,500,400,300,200],dtype=torch.float32)
criterion=nn.CrossEntropyLoss(weight=supports.max()/supports)
EPOCHS=15; LR=0.005; NC=N_CLASSES

def train_one(model,dl,opt,sched,epochs,nc):
    model.train()
    lh,ah,fh=[],[],[]
    for ep in range(epochs):
        tl=0; preds=[]; labs=[]
        for bg in dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg); opt.zero_grad()
            feat=bg.ndata['features']; lbl=torch.clamp(bg.ndata['label']-1,0,nc-1)
            out=model(bg,feat); loss=criterion(out,lbl)
            if torch.isnan(loss): continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            tl+=loss.item(); preds+=out.argmax(1).tolist(); labs+=lbl.tolist()
        avg=tl/max(len(dl),1)
        acc=accuracy_score(labs,preds)
        f1=f1_score(labs,preds,average='weighted',zero_division=0)
        lh.append(avg); ah.append(acc); fh.append(f1)
        print(f"  Epoch {ep+1:2d}/{epochs}  loss={avg:.4f}  acc={acc:.4f}  f1={f1:.4f}")
        sched.step()
    return lh,ah,fh

def validate(model,dl,nc):
    model.eval(); preds=[]; labs=[]
    with torch.no_grad():
        for bg in dl:
            if bg.num_nodes()==0: continue
            bg=dgl.add_self_loop(bg)
            out=model(bg,bg.ndata['features']).cpu().numpy()
            lb=np.clip(bg.ndata['label'].cpu().numpy()-1,0,nc-1)
            preds+=out.argmax(1).tolist(); labs+=lb.tolist()
    return (accuracy_score(labs,preds),
            f1_score(labs,preds,average='weighted',zero_division=0),
            precision_score(labs,preds,average='weighted',zero_division=0),
            recall_score(labs,preds,average='weighted',zero_division=0))

def make_model(name):
    inf=len(INPUT_FEATURES)
    if name=='ImprovedGNN':      return ImprovedGNN(inf,[128,256,256,128,64],NC)
    if name=='ExtendedGAT':      return ExtendedGAT(inf,64,NC,heads=4)
    if name=='GraphSAGE':        return GraphSAGE(inf,128,NC)
    if name=='ExtendedGraphSAGE':return ExtendedGraphSAGE(inf,128,NC)

# ── Sweep ──────────────────────────────────────────────────────────────────
MODEL_NAMES=['ImprovedGNN','ExtendedGAT','GraphSAGE','ExtendedGraphSAGE']
results={}

for name in MODEL_NAMES:
    print(f"\n{'='*55}")
    print(f"  {name}  |  lr={LR}  |  epochs={EPOCHS}")
    print(f"{'='*55}")
    m=make_model(name)
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=0.5)
    tracker=EmissionsTracker(project_name=f"GNN_{name}",output_dir="/tmp",
                             log_level="error",save_to_file=False)
    tracker.start(); t0=time.time()
    lh,ah,fh=train_one(m,train_dl,opt,sch,EPOCHS,NC)
    co2=tracker.stop(); elapsed=time.time()-t0
    va,vf,vp,vr=validate(m,valid_dl,NC)
    results[name]={'val_acc':va,'val_f1':vf,'val_prec':vp,'val_rec':vr,
                   'train_s':elapsed,'co2_g':(co2 or 0)*1000,
                   'loss':lh,'acc':ah,'f1':fh}
    print(f"\n  VAL  acc={va:.4f}  f1={vf:.4f}  prec={vp:.4f}  rec={vr:.4f}")
    print(f"  TIME {elapsed:.1f}s   CO2 {(co2 or 0)*1000:.5f} g")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print(f"{'Model':<22} {'Val Acc':>8} {'Val F1':>8} {'Prec':>8} {'Rec':>8} {'Time(s)':>9} {'CO2(g)':>10}")
print("="*70)
for n,r in results.items():
    print(f"{n:<22} {r['val_acc']:>8.4f} {r['val_f1']:>8.4f} {r['val_prec']:>8.4f} {r['val_rec']:>8.4f} {r['train_s']:>9.1f} {r['co2_g']:>10.5f}")
print("="*70)

import pickle
with open('/tmp/gnn_sweep_results.pkl','wb') as f: pickle.dump(results,f)
print("\nResults saved to /tmp/gnn_sweep_results.pkl")
