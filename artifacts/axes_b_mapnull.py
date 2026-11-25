"""R² of each GZ axis score from 18 directions: the physical set (map's own), 18 shuffled-target w's, and 18 random directions in the cleaned space's covariance."""
import sys, json; sys.path.insert(0, "artifacts")
import numpy as np, axes_a_ordinal as A
from pathlib import Path
AXS,d=A.load()
B=Path('runs/almagest/axes_phys/b')
m=json.load(open('docs/axes/b_map.json')); names=m['names']; gz=names[:m['n_gz']]; strong=m['strong_physical']
X=d.clean['A1']; te=np.isin(d.union,np.asarray(d.setup.test_ids))&d.ok
rng=np.random.default_rng(1)
S=np.linalg.cholesky(np.cov(X[te].T)+1e-6*np.eye(384))
real=np.column_stack([X@np.load(B/'fits'/(n.replace('/','_').replace(' ','_')+'.npz'))['cleaned_we'] for n in strong])
nulls=np.column_stack([X@np.load(B/'fits'/(n.replace('/','_').replace(' ','_')+'.npz'))['null_we'][k] for n in strong for k in (0,)])
rd=np.column_stack([X@(np.linalg.inv(S).T@rng.normal(size=384)) for _ in range(18)])
out={}
for f in gz:
    a=np.load(AXS.AX/f'{f}.npz'); s=X@a['we']+float(a['be'])
    e=np.zeros(len(X),bool); e[d.rows(f)[1]]=True
    o=np.flatnonzero(e&te); h=rng.permutation(len(o))<len(o)//2
    r=[]
    for M0 in (real,nulls,rd):
        M=np.column_stack([np.ones(len(o)),M0[o]]); M[:,1:]=(M[:,1:]-M[h,1:].mean(0))/M[h,1:].std(0)
        c=np.linalg.solve(M[h].T@M[h]+1e-3*np.eye(M.shape[1]),M[h].T@s[o][h]); y=s[o][~h]
        r.append(1-((y-M[~h]@c)**2).sum()/((y-y.mean())**2).sum())
    out[f]=r
json.dump({"note": __doc__, "r2": {f: {"physical": float(r[0]), "shuffled_target_directions": float(r[1]), "random_directions": float(r[2])} for f, r in out.items()}}, open("docs/axes/b_map_null.json", "w"), indent=1)
