# =====================================================================
#  NEURAL-QDC  ·  15-taxon  ·  FULL pipeline in ONE cell  (Kaggle)
#  Before running: Settings -> Internet ON , Accelerator = GPU (T4 x2 / P100)
#  Add Input -> upload 15-taxon.zip as a Dataset.  Then run this single cell.
# =====================================================================

# ---- setup ----
!pip -q install torch_geometric
!pip -q install dendropy
!apt-get -qq install -y default-jre-headless 2>/dev/null
!git clone -q --depth 1 https://github.com/Mahim1997/wQFM-2020.git /kaggle/working/wQFM-2020
WQFM = "/kaggle/working/wQFM-2020"; JAR = "wQFM-v1.4.jar"

# ---- imports ----
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import dendropy, subprocess, glob, os, zipfile
from dendropy.calculate import treecompare
from itertools import combinations
from collections import defaultdict
import numpy as np
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", dev)

# ---- locate 15-taxon data (works whether Kaggle extracted the zip or not) ----
hits = glob.glob("/kaggle/input/**/true-species.tre", recursive=True)
if not hits:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        if "15" in os.path.basename(z):
            with zipfile.ZipFile(z) as zf: zf.extractall("/kaggle/working/real")
    hits = glob.glob("/kaggle/working/**/true-species.tre", recursive=True)
assert hits, "15-taxon data not found — Add Input -> upload 15-taxon.zip"
BASE = os.path.dirname(hits[0]); print("dataset base:", BASE)

# ---- taxa + true species tree ----
names = [chr(ord('A')+i) for i in range(15)]
TNS   = dendropy.TaxonNamespace(names)
TRUE_SP = dendropy.Tree.get(path=f"{BASE}/true-species.tre", schema="newick", taxon_namespace=TNS)

# ---- data functions ----
def load_trees(paths):
    tl = dendropy.TreeList(taxon_namespace=TNS)
    for p in paths: tl.read(path=p, schema="newick")
    return tl

def qcounts_real(trees):                       # real gene tree leaves are 'A'..'O'
    c = defaultdict(lambda: defaultdict(int))
    for gt in trees:
        pdm = gt.phylogenetic_distance_matrix(); tx = {t.label: t for t in gt.taxon_namespace}
        for a,b,cc,d in combinations(names,4):
            dd=lambda x,y: pdm.path_edge_count(tx[x],tx[y])
            r =[dd(a,b)+dd(cc,d), dd(a,cc)+dd(b,d), dd(a,d)+dd(b,cc)]
            pr=[((a,b),(cc,d)),((a,cc),(b,d)),((a,d),(b,cc))]
            c[(a,b,cc,d)][pr[r.index(min(r))]] += 1
    return c

def real_counts(cond, rep, ngenes):            # -> (QD_E counts, QD_T counts)
    true = load_trees([f"{BASE}/true-genetrees/{rep}/{ngenes}_gt.tre"])
    ep   = sorted(glob.glob(f"{BASE}/{cond}/estimated-genetrees/{rep}/*/RAxML_bipartitions.final.f100"))
    est  = load_trees(ep)
    return qcounts_real(est), qcounts_real(true)

def _res(k):
    t0,t1,t2,t3=k
    return [((t0,t1),(t2,t3)),((t0,t2),(t1,t3)),((t0,t3),(t1,t2))]

def wqrts_to_graph(qe_counts, qt_counts):      # node=4-taxon set, feat=3 norm weights, edge=share 3 taxa
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
    x =torch.tensor(feat(qe_counts),dtype=torch.float)
    y =torch.tensor(feat(qt_counts),dtype=torch.float)
    ei=torch.tensor([src+dst,dst+src],dtype=torch.long)
    return Data(x=x,y=y,edge_index=ei)

def run_wqfm(inp, out):
    subprocess.run(["java","-jar",JAR,"-i",inp,"-o",out], cwd=WQFM,
                   capture_output=True, text=True, timeout=300)
    return open(f"{WQFM}/{out}").read().strip()

def rf_true(nwk):                              # RF to the fixed true species tree
    t=dendropy.Tree.get(data=nwk if nwk.endswith(";") else nwk+";",schema="newick",taxon_namespace=TNS)
    t.deroot(); t.encode_bipartitions()
    sp=TRUE_SP.clone(depth=1); sp.deroot(); sp.encode_bipartitions()
    return treecompare.symmetric_difference(sp, t)

def write_counts(c, path):
    with open(path,"w") as f:
        for k in combinations(names,4):
            for (p,q) in _res(k): f.write(f"(({p[0]},{p[1]}),({q[0]},{q[1]})); {c[k].get((p,q),0)}\n")

def gnn_correct_and_write(c, path, scale=1000):     # QD_E -> GNN -> write QD' .wqrts
    keys=sorted(combinations(names,4))
    feats=[[x/(sum(w) or 1) for x in w] for w in ([c[k].get(r,0) for r in _res(k)] for k in keys)]
    idx={k:i for i,k in enumerate(keys)}; src=[];dst=[]; ts=set(names)
    for i,k in enumerate(keys):
        for trio in combinations(k,3):
            for e in ts-set(k):
                j=idx.get(tuple(sorted(set(trio)|{e})))
                if j is not None and j>i: src.append(i);dst.append(j)
    x=torch.tensor(feats,dtype=torch.float); ei=torch.tensor([src+dst,dst+src],dtype=torch.long)
    d=Data(x=x,edge_index=ei).to(dev); model.eval()
    with torch.no_grad(): out=model(d.x,d.edge_index).cpu().tolist()
    with open(path,"w") as f:
        for i,k in enumerate(keys):
            for (p,q),pv in zip(_res(k), out[i]):
                f.write(f"(({p[0]},{p[1]}),({q[0]},{q[1]})); {int(round(pv*scale))}\n")

# ---- model ----
class NeuralQDC(nn.Module):                    # 2-hop GAT + residual
    def __init__(self, hid=64, heads=4):
        super().__init__()
        self.g1  = GATConv(3, hid, heads=heads)
        self.g2  = GATConv(hid*heads, hid, heads=heads)
        self.lin = nn.Linear(hid*heads, 3)
    def forward(self, x, ei):
        h=F.elu(self.g1(x,ei)); h=F.elu(self.g2(h,ei))
        return F.softmax(torch.log(x.clamp_min(1e-6))+self.lin(h), dim=1)

# ---- build graphs (2 conditions, R1-R10 ; train R1-R8, test R9-R10) ----
conds = [("100gene-100bp",100), ("100gene-1000bp",100)]
reps  = [f"R{i}" for i in range(1,11)]
data_by_key = {}
print("building real graphs (a few minutes)...")
for cond,ng in conds:
    for rep in reps:
        qe,qt = real_counts(cond,rep,ng)
        data_by_key[(cond,rep)] = (wqrts_to_graph(qe,qt).to(dev), qe, qt)
test_reps = ("R9","R10")
train = [g for (c,r),(g,_,_) in data_by_key.items() if r not in test_reps]
test  = [(c,r,g,qe,qt) for (c,r),(g,qe,qt) in data_by_key.items() if r in test_reps]
print(f"{len(train)} train graphs, {len(test)} test cases")

# ---- train ----
model=NeuralQDC().to(dev); opt=torch.optim.Adam(model.parameters(), lr=0.002)
best=1e9; best_state=None
import random
for ep in range(300):
    model.train(); random.shuffle(train); tot=0
    for g in train:
        opt.zero_grad(); out=model(g.x,g.edge_index)
        loss=F.mse_loss(out,g.y); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=loss.item()
    model.eval(); gm=0
    with torch.no_grad():
        for _,_,g,_,_ in test: gm+=F.mse_loss(model(g.x,g.edge_index),g.y).item()
    gm/=len(test)
    if gm<best: best=gm; best_state={k:v.clone() for k,v in model.state_dict().items()}
    if ep%40==0 or ep==299: print(f"epoch {ep:3d}  train {tot/len(train):.5f}  test {gm:.5f}")
model.load_state_dict(best_state); print("trained.")

# ---- result: RF + MSE on held-out cases ----
rows=[]
for cond,rep,g,qe,qt in test:
    write_counts(qe, f"{WQFM}/e.wqrts"); write_counts(qt, f"{WQFM}/t.wqrts")
    gnn_correct_and_write(qe, f"{WQFM}/c.wqrts")
    rE=rf_true(run_wqfm("e.wqrts","te.tree"))
    rC=rf_true(run_wqfm("c.wqrts","tc.tree"))
    rT=rf_true(run_wqfm("t.wqrts","tt.tree"))
    with torch.no_grad():
        mid=F.mse_loss(g.x,g.y).item(); mgn=F.mse_loss(model(g.x,g.edge_index),g.y).item()
    rows.append((rE,rC,rT))
    print(f"{cond} {rep}:  RF(QD_E)={rE:2d} RF(GNN)={rC:2d} RF(QD_T)={rT:2d}   | MSE id={mid:.4f} GNN={mgn:.4f}")
a=np.array(rows,float).mean(0)
print(f"\n=== 15-TAXON avg over {len(rows)} held-out cases (lower=better) ===")
print(f"QD_E={a[0]:.2f}   Neural-QDC={a[1]:.2f}   QD_T={a[2]:.2f}")
print(f"RF reduction vs QD_E: {100*(a[0]-a[1])/max(a[0],1e-9):.1f}%")
mid=np.mean([F.mse_loss(g.x,g.y).item() for _,_,g,_,_ in test])
mgn=np.mean([F.mse_loss(model(g.x,g.edge_index),g.y).item() for _,_,g,_,_ in test])
print(f"AVERAGE MSE: identity {mid:.4f} -> GNN {mgn:.4f}  ({100*(mid-mgn)/mid:.1f}% reduction)")
