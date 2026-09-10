# =====================================================================
#  NEURAL-QDC  ·  37-taxon mammalian  ·  FULL pipeline + output files
#  ONE cell for Kaggle.  Train the model AND write all 7 output files.
#
#  Before running:
#    Settings -> Internet ON , Accelerator = GPU (T4 x2 / P100)
#    Add Input -> upload 37-taxon.zip as a Dataset
#  Run this single cell.  Downloads appear in the right-side Output panel.
# =====================================================================

# ---------------- setup ----------------
!pip -q install torch_geometric
!pip -q install dendropy
!apt-get -qq install -y default-jre-headless 2>/dev/null
!git clone -q --depth 1 https://github.com/Mahim1997/wQFM-2020.git /kaggle/working/wQFM-2020
WQFM = "/kaggle/working/wQFM-2020"; JAR = "wQFM-v1.4.jar"

# ---------------- imports ----------------
import os, glob, zipfile, subprocess, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import dendropy
from dendropy.calculate import treecompare
from itertools import combinations
from collections import defaultdict
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", dev)

# ---------------- locate data ----------------
hits = glob.glob("/kaggle/input/**/mammalian-model-species.tre", recursive=True)
if not hits:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        if "37" in os.path.basename(z):
            with zipfile.ZipFile(z) as zf: zf.extractall("/kaggle/working/m37")
    hits = glob.glob("/kaggle/working/**/mammalian-model-species.tre", recursive=True)
assert hits, "37-taxon data not found — Add Input -> upload 37-taxon.zip"
BASE = os.path.dirname(hits[0]); ROOT = os.path.dirname(BASE)
print("dataset base:", BASE)

# ---------------- taxa + true species tree ----------------
sp0 = dendropy.Tree.get(path=f"{BASE}/mammalian-model-species.tre", schema="newick")
names = sorted(x.taxon.label for x in sp0.leaf_nodes())
TNS = dendropy.TaxonNamespace(names)
TRUE_SP = dendropy.Tree.get(path=f"{BASE}/mammalian-model-species.tre", schema="newick", taxon_namespace=TNS)
print(len(names), "taxa")

# ---------------- data functions ----------------
def _res(k):
    t0,t1,t2,t3=k
    return [((t0,t1),(t2,t3)),((t0,t2),(t1,t3)),((t0,t3),(t1,t2))]
def load_trees(paths):
    tl=dendropy.TreeList(taxon_namespace=TNS)
    for p in paths:
        try: tl.read(path=p,schema="newick")
        except Exception: pass
    return tl
def qcounts(trees):
    c=defaultdict(lambda: defaultdict(int))
    for gt in trees:
        pdm=gt.phylogenetic_distance_matrix(); tx={t.label:t for t in gt.taxon_namespace}
        for a,b,cc,d in combinations(names,4):
            if not all(x in tx for x in (a,b,cc,d)): continue
            g=lambda x,y: pdm.path_edge_count(tx[x],tx[y])
            r=[g(a,b)+g(cc,d), g(a,cc)+g(b,d), g(a,d)+g(b,cc)]
            c[(a,b,cc,d)][_res((a,b,cc,d))[r.index(min(r))]]+=1
    return c
def counts_37(cond, rep):                      # -> (QD_E counts, QD_T counts)
    est = load_trees(sorted(glob.glob(f"{ROOT}/{cond}/{rep}/*/raxmlboot.gtrgamma/RAxML_bipartitions.final.f200")))
    tru = load_trees(sorted(glob.glob(f"{ROOT}/true-genetrees/mammalian-1X-truegt/1X-200-true/{rep}/*/true.gt")))
    return qcounts(est), qcounts(tru)

def wqrts_to_graph(qe, qt):                     # node=4-taxon set, feat=3 norm weights, edge=share 3 taxa
    keys=sorted(combinations(names,4))
    def feat(counts):
        F_=[]
        for k in keys:
            w=[counts[k].get(r,0) for r in _res(k)]; tot=sum(w) or 1
            F_.append([x/tot for x in w])
        return F_
    idx={k:i for i,k in enumerate(keys)}; src=[]; dst=[]; ts=set(names)
    for i,k in enumerate(keys):
        for trio in combinations(k,3):
            for e in ts-set(k):
                j=idx.get(tuple(sorted(set(trio)|{e})))
                if j is not None and j>i: src.append(i); dst.append(j)
    x=torch.tensor(feat(qe),dtype=torch.float); y=torch.tensor(feat(qt),dtype=torch.float)
    ei=torch.tensor([src+dst,dst+src],dtype=torch.long)
    return Data(x=x,y=y,edge_index=ei)

# ---------------- model (GAT, memory-safe for T4) ----------------
class NeuralQDC(nn.Module):
    def __init__(self, hid=16, heads=2):
        super().__init__()
        self.g1  = GATConv(3, hid, heads=heads, add_self_loops=False)
        self.g2  = GATConv(hid*heads, hid, heads=heads, add_self_loops=False)
        self.lin = nn.Linear(hid*heads, 3)
    def forward(self, x, ei):
        h=F.elu(self.g1(x,ei)); h=F.elu(self.g2(h,ei))
        return F.softmax(torch.log(x.clamp_min(1e-6))+self.lin(h), dim=1)

# ---------------- build graphs (1X-200-250, R1..R12 ; train R1-R8, test R9-R12) ----------------
COND = "1X-200-250"
reps = [f"R{i}" for i in range(1,13)]
store = {}
print("building graphs (~1.5 min each)...")
for rep in reps:
    qe,qt = counts_37(COND, rep)
    store[rep] = (wqrts_to_graph(qe,qt), qe, qt)     # graphs stay on CPU
    print("  ", rep, "done")
train = [store[r][0] for r in reps[:8]]
test  = [(r,)+store[r] for r in reps[8:]]
print(f"{len(train)} train, {len(test)} test")

# ---------------- train (one graph on GPU at a time) ----------------
model = NeuralQDC(hid=16, heads=2).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=0.002)
best=1e9; best_state=None
for ep in range(200):
    model.train(); random.shuffle(train); tot=0
    for g in train:
        g=g.to(dev)
        opt.zero_grad(); out=model(g.x,g.edge_index)
        loss=F.mse_loss(out,g.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=loss.item()
        g=g.cpu(); torch.cuda.empty_cache()
    model.eval(); gm=0
    with torch.no_grad():
        for _,g,_,_ in test:
            g=g.to(dev); gm+=F.mse_loss(model(g.x,g.edge_index),g.y).item(); g=g.cpu()
    gm/=len(test); torch.cuda.empty_cache()
    if gm<best: best=gm; best_state={k:v.clone() for k,v in model.state_dict().items()}
    if ep%25==0 or ep==199: print(f"epoch {ep:3d}  train {tot/len(train):.5f}  test {gm:.5f}")
model.load_state_dict(best_state); print("trained.")

# =====================================================================
#  OUTPUT FILES  ·  QD_E , QD' , QD_T   (distribution + dominant + comparison)
#  for one condition + replicate (choose below).  Saved to /kaggle/working.
# =====================================================================
OUT_COND = "1X-200-250"     # 1X-200-250 / 1X-200-500 / 1X-200-1000
OUT_REP  = "R9"             # use a HELD-OUT replicate (R9-R12) for a fair result
NGENES   = 200

print(f"\nwriting output files for {OUT_COND} {OUT_REP} ...")
QDE, QDT = counts_37(OUT_COND, OUT_REP)

def gnn_correct(counts):                        # QD_E -> QD' (as integer weights, prob*NGENES)
    keys=sorted(combinations(names,4))
    feats=[[x/(sum(w) or 1) for x in w] for w in ([counts[k].get(r,0) for r in _res(k)] for k in keys)]
    idx={k:i for i,k in enumerate(keys)}; src=[];dst=[]; ts=set(names)
    for i,k in enumerate(keys):
        for trio in combinations(k,3):
            for e in ts-set(k):
                j=idx.get(tuple(sorted(set(trio)|{e})))
                if j is not None and j>i: src.append(i);dst.append(j)
    x=torch.tensor(feats,dtype=torch.float); ei=torch.tensor([src+dst,dst+src],dtype=torch.long)
    d=Data(x=x,edge_index=ei).to(dev); model.eval()
    with torch.no_grad(): out=model(d.x,d.edge_index).cpu().tolist()
    del d; torch.cuda.empty_cache()
    QDP=defaultdict(dict)
    for i,k in enumerate(keys):
        for j,r in enumerate(_res(k)): QDP[k][r]=int(round(out[i][j]*NGENES))
    return QDP
QDP = gnn_correct(QDE)

def lab(res): return "".join(res[0])+"|"+"".join(res[1])
def dominant(QD,k): return max(_res(k), key=lambda r: QD[k].get(r,0))
def write_distribution(QD, path):
    with open(path,"w") as f:
        f.write(f"{'4-taxon set':<26}{'resolution':<22}{'weight'}\n")
        for k in combinations(names,4):
            for r in _res(k): f.write(f"{'{'+','.join(k)+'}':<26}{lab(r):<22}{QD[k].get(r,0)}\n")
def write_dominant(QD, path):
    with open(path,"w") as f:
        f.write(f"{'4-taxon set':<26}{'dominant quartet':<22}{'weight'}\n")
        for k in combinations(names,4):
            b=dominant(QD,k); f.write(f"{'{'+','.join(k)+'}':<26}{lab(b):<22}{QD[k].get(b,0)}\n")

write_distribution(QDE,"estimated_quartet_distribution.txt"); write_dominant(QDE,"estimated_dominant_quartets.txt")
write_distribution(QDP,"improved_quartet_distribution.txt");  write_dominant(QDP,"improved_dominant_quartets.txt")
write_distribution(QDT,"true_quartet_distribution.txt");      write_dominant(QDT,"true_dominant_quartets.txt")

# ---- comparison : dominant-quartet agreement with the truth ----
allsets=list(combinations(names,4)); N=len(allsets); matchE=matchP=changed=0; rows=[]
for k in allsets:
    de=dominant(QDE,k); dp=dominant(QDP,k); dt=dominant(QDT,k)
    me=(de==dt); mp=(dp==dt); matchE+=me; matchP+=mp
    if de!=dp: changed+=1; rows.append((k,lab(de),lab(dp),lab(dt),me,mp))
with open("comparison_summary.txt","w") as f:
    f.write("Neural-QDC : dominant-quartet agreement with the TRUE gene trees\n")
    f.write(f"condition {OUT_COND}   replicate {OUT_REP}   total 4-taxon sets = {N}\n\n")
    f.write(f"QD_E (estimated) dominant matches QD_T (true) : {matchE}/{N} = {100*matchE/N:.2f}%\n")
    f.write(f"QD'  (GNN output) dominant matches QD_T (true): {matchP}/{N} = {100*matchP/N:.2f}%\n")
    f.write(f"change in agreement                            : {100*(matchP-matchE)/N:+.2f} percentage points\n")
    f.write(f"sets where GNN changed the dominant quartet    : {changed}\n\n")
    f.write("--- sample of changed sets (set | QD_E dom | QD' dom | QD_T dom | E=T | '=T) ---\n")
    for k,e,p,t,me,mp in rows[:50]:
        f.write(f"{'{'+','.join(k)+'}':<24} {e:<16} {p:<16} {t:<16} {int(me)}   {int(mp)}\n")

print("\n=== DONE ===")
print(f"QD_E matches truth : {100*matchE/N:.2f}%")
print(f"QD'  matches truth : {100*matchP/N:.2f}%   ({100*(matchP-matchE)/N:+.2f} pts)")
for fn in ["estimated_quartet_distribution.txt","estimated_dominant_quartets.txt",
           "improved_quartet_distribution.txt","improved_dominant_quartets.txt",
           "true_quartet_distribution.txt","true_dominant_quartets.txt","comparison_summary.txt"]:
    print("  saved:", fn)
