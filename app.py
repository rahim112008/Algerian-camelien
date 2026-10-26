import gzip, io, os, warnings, zipfile, subprocess, tempfile, shutil
from collections import Counter
from io import BytesIO
from urllib.parse import quote
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS
import requests, time, json
import streamlit.components.v1 as components

try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🐪 Dromedary Pipeline v1.5", page_icon="🐪",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================
# PARAMÈTRES DROMADAIRE
# ============================================================
ARS = {str(i): 100_000_000 for i in range(1, 37)} 
AUTO = [str(i) for i in range(1, 37)]

# ============================================================
# UTILS & PARSERS
# ============================================================
def _ha(x):
    if not isinstance(x, np.ndarray) or x.size == 0: return "empty"
    if x.dtype.kind in ("U","S","O"): return f"a|{x.shape}|{'|'.join(map(str,x.ravel()[:2000]))}"
    nan_count = int(np.isnan(x).sum()) if x.dtype.kind == "f" else 0
    return f"{x.shape}|{x.dtype}|nan={nan_count}|{float(np.nansum(x))}|{float(np.nansum(np.abs(x)))}|{float(np.nansum(x*x))}"
def _hn(x):
    try: return _ha(np.asarray(x))
    except: return f"{type(x).__name__}|{len(x) if hasattr(x,'__len__') else 0}"
HF = {np.ndarray:_ha, pd.Series:_hn, pd.Index:_hn}
for _n in ("ArrowStringArray","StringArray","IntegerArray","FloatingArray","BooleanArray",
           "NumpyExtensionArray","PandasArray","DatetimeArray","TimedeltaArray","PeriodArray",
           "IntervalArray","Categorical"):
    _c = getattr(pd.arrays,_n,None) or getattr(pd.core.arrays,_n,None)
    if _c: HF[_c]=_hn
def cache_data(func=None, **kw):
    def _d(f): return st.cache_data(show_spinner=False, hash_funcs=HF, **kw)(f)
    return _d if func is None else _d(func)

def impute_mean(gt):
    g=gt.astype(np.float32,copy=True); cm=np.nanmean(g,axis=0); cm=np.where(np.isnan(cm),0.0,cm)
    nm=np.isnan(g)
    if nm.any(): g[nm]=np.take(cm, np.where(nm)[1])
    return g

def _chk(c):
    s=str(c).upper().replace("CHR","").replace("CHROMOSOME","")
    if s.isdigit(): return (0,int(s),"")
    sp={"X":100,"Y":101,"MT":102,"M":102}
    return (1,sp[s],"") if s in sp else (2,0,s)
def _rt(raw):
    if len(raw)>=2 and raw[:2]==b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8",errors="replace")
    return raw.decode("utf-8",errors="replace")

def sample_fid_iid(s, fp=2):
    n=str(s).strip()
    for e in (".CEL",".cel",".TXT",".txt",".gz",".GZ"):
        if n.endswith(e): n=n[:-len(e)]
    n=n.replace("(","_").replace(")","").replace("[","_").replace("]","_").replace(" ","_").replace("/","_")
    while "__" in n: n=n.replace("__","_")
    p=n.strip("_").split("_"); return ("_".join(p[:fp]) if len(p)>=fp else p[0]), n

def _ca(a):
    a=str(a).strip() if a is not None else "0"; return "0" if a in ("",".","-","NA","nan","None","0","00") else a

# 🌟 CONVERSION AXIOM
def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if fn.endswith(".gz") or raw_bytes[:2]==b"\x1f\x8b" else raw_bytes
    lines = [l.rstrip("\r\n") for l in raw.decode("utf-8", errors="replace").splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            start = i; break
    delim = "\t" if "\t" in lines[start] else " "
    header = lines[start].split(delim); data_lines = lines[start+1:]
    n_cols = len(header); n_rows = len(data_lines)
    
    if n_rows > n_cols * 2 and n_cols > 2:
        sample_names = header[1:] 
        gtypes_dict = {s: [] for s in sample_names}
        for line in data_lines:
            parts = line.split(delim)
            if len(parts) < n_cols: continue
            for k, cell in enumerate(parts[1:]):
                s_name = sample_names[k]
                if len(cell) == 2 and all(ch in "ACGT" for ch in cell): gtypes_dict[s_name].append((cell[0], cell[1]))
                else: gtypes_dict[s_name].append(("0","0"))
        samples = list(gtypes_dict.keys()); gtypes = list(gtypes_dict.values())
        return samples, gtypes, len(gtypes[0])
    else:
        samples, gtypes = [], []
        for line in data_lines:
            parts = line.split(delim)
            if len(parts) < 2: continue
            samples.append(parts[0].strip()); alleles = []
            for cell in parts[1:]:
                tk = cell.split()
                if len(tk) == 2: alleles.append((_ca(tk[0]), _ca(tk[1])))
                elif len(tk) == 1 and len(tk[0]) == 2 and all(ch in "ACGT0" for ch in tk[0]): alleles.append((tk[0][0], tk[0][1]))
                else: alleles.append(("0","0"))
            gtypes.append(alleles)
        if not samples: raise ValueError("Aucun échantillon")
        c = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
        for i, r in enumerate(gtypes): gtypes[i] = r[:c] + [("0","0")]*(c-len(r))
        return samples, gtypes, c

def parse_map(b):
    rows=[]
    for line in _rt(b).splitlines():
        p=line.split()
        if len(p)>=4:
            try: cm=float(p[2]) if p[2] not in ("0",".","NA") else 0.0
            except: cm=0.0
            try: bp=int(float(p[3]))
            except: continue
            rows.append({"CHR":str(p[0]),"SNP":p[1],"CM":cm,"BP":bp})
    if not rows: raise ValueError(".map vide")
    return pd.DataFrame(rows)

def parse_ped(b, n_map):
    lines=[l.strip() for l in _rt(b).splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    n_ped=max(0,(len(lines[0].split())-6)//2)
    if n_ped<7: raise ValueError(f"PED a {n_ped} SNPs")
    n=min(n_map,n_ped); ec=6+2*n; fids,iids,rows=[],[],[]
    for line in lines:
        p=line.split()
        if len(p)>=ec: fids.append(p[0]); iids.append(p[1]); rows.append(p[6:6+2*n])
    if not iids: raise ValueError("Aucun individu")
    ac=[Counter() for _ in range(n)]
    for r in rows:
        for j in range(n):
            if r[2*j]!="0" and r[2*j+1]!="0": ac[j].update([r[2*j],r[2*j+1]])
    alleles_A1, alleles_A2 = [], []
    for c in ac:
        common = c.most_common(2)
        if len(common) >= 2: alleles_A1.append(common[0][0]); alleles_A2.append(common[1][0])
        elif len(common) == 1: alleles_A1.append(common[0][0]); alleles_A2.append(".")
        else: alleles_A1.append("."); alleles_A2.append(".")
    minor=[min(c,key=c.get) if len(c)>=2 else None for c in ac]
    gt=np.full((len(iids),n),np.nan,dtype=np.float32)
    for i,r in enumerate(rows):
        for j in range(n):
            if minor[j] and r[2*j]!="0" and r[2*j+1]!="0":
                gt[i,j]=(r[2*j]==minor[j])+(r[2*j+1]==minor[j])
    return gt, pd.DataFrame({"FID":fids,"IID":iids}), alleles_A1, alleles_A2

def parse_gff3(b):
    genes=[]
    for line in _rt(b).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p=line.split("\t")
        if len(p)>=9 and p[2] == "gene":
            a=p[8]; name="Unknown"
            if "Name=" in a: name=a.split("Name=")[1].split(";")[0]
            elif 'gene_name "' in a: name=a.split('gene_name "')[1].split('"')[0]
            elif 'gene_id "' in a: name=a.split('gene_id "')[1].split('"')[0]
            genes.append({"CHR":str(p[0]),"START":int(p[3]),"END":int(p[4]),"GENE":name})
    return pd.DataFrame(genes).drop_duplicates(subset=["CHR","START","END","GENE"])

def rec_fid(ind):
    df=ind.copy()
    if df["FID"].nunique()<=1: df["FID"]=df["IID"].astype(str).str.split("_").str[0]
    return df

# ============================================================
# CALCULS GÉNÉTIQUES
# ============================================================
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): p=np.nanmean(g,axis=0)/2.0; return np.minimum(p,1.0-p)
def het(g): return np.nanmean(g==1, axis=1)

def hwe_exact(nh, n0, n2):
    n=nh+n0+n2
    if n==0 or (nh==0 and (n0==0 or n2==0)): return 1.0
    rare=2*min(n0,n2)+nh; mid=(rare*(2*n-rare))//(2*n)
    if mid%2!=rare%2: mid+=1
    pr=np.zeros(rare+1); pr[mid]=1.0; s=1.0
    ch,cr,cc=mid,(rare-mid)//2,n-mid-(rare-mid)//2
    while ch<=rare-2: pr[ch+2]=pr[ch]*4*cr*cc/((ch+2)*(ch+1)); s+=pr[ch+2]; ch+=2; cr-=1; cc-=1
    ch,cr,cc=mid,(rare-mid)//2,n-mid-(rare-mid)//2
    while ch>=2: pr[ch-2]=pr[ch]*ch*(ch-1)/(4*(cr+1)*(cc+1)); s+=pr[ch-2]; ch-=2; cr+=1; cc+=1
    po=pr[nh] if nh<len(pr) else 0.0
    return float(min(pr[pr<=po+1e-7].sum()/s, 1.0))

@cache_data
def hwe_pvals(g):
    pv=np.full(g.shape[1],np.nan)
    n0=((g==0)&~np.isnan(g)).sum(0); n1=((g==1)&~np.isnan(g)).sum(0); n2=((g==2)&~np.isnan(g)).sum(0)
    nv=n0+n1+n2
    if g.shape[1]<=20000:
        for j in range(g.shape[1]):
            if nv[j]>=5: pv[j]=hwe_exact(int(n1[j]),int(n0[j]),int(n2[j]))
    else:
        p=(n1+2*n2)/(2.0*np.where(nv>0,nv,np.nan))
        with np.errstate(invalid="ignore",divide="ignore"):
            eh=2.0*p*(1.0-p)*nv; ok=(eh>=5)&np.isfinite(eh)
            chi2=(np.abs(n1-eh)-0.5)**2/eh; pv[ok]=1.0-stats.chi2.cdf(chi2[ok],df=1)
    return pv

def sync_dims(gt, ind_df, snp_df):
    n=min(gt.shape[0],len(ind_df)); m=min(gt.shape[1],len(snp_df))
    return gt[:n,:m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def apply_qc(gt, ind_df, snp_df, params, rem_out=True, autosomal_only=True):
    gt,ind_df,snp_df=sync_dims(gt,ind_df,snp_df); n0,m0=gt.shape; tr={}
    if autosomal_only:
        is_auto = snp_df["CHR"].astype(str).str.replace("chr","").isin(AUTO)
        gt = gt[:, is_auto.values]; snp_df = snp_df[is_auto].reset_index(drop=True)
    k=miss_snp(gt)<=params["geno"]; gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    k=miss_ind(gt)<=params["mind"]; gt=gt[k]; ind_df=ind_df[k].reset_index(drop=True)
    m=maf(gt); k=np.isfinite(m)&(m>=params["maf"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    pv=hwe_pvals(gt); k=np.isnan(pv)|(pv>=params["hwe"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    if gt.shape[1]==0: raise ValueError("Tout exclu (MAF/HWE)")
    if rem_out:
        h=het(gt); kp=np.ones(len(h),dtype=bool)
        for f in ind_df["FID"].unique():
            msk=(ind_df["FID"]==f).values; sub=h[msk]
            if len(sub)>=5 and np.nanstd(sub)>1e-9:
                z=(sub-np.nanmean(sub))/np.nanstd(sub); idx=np.where(msk)[0]; kp[idx[np.abs(z)>params["het_sd"]]]=False
        gt=gt[kp]; ind_df=ind_df[kp].reset_index(drop=True)
    return gt, ind_df, snp_df

@cache_data
def compute_pop_stats(gt, ind_df):
    stats = []
    for p in ind_df["FID"].unique():
        mask = (ind_df["FID"] == p).values
        g = gt[mask]
        if g.shape[0] == 0: continue
        p_allele = np.nanmean(g, axis=0) / 2.0
        He = np.nanmean(2 * p_allele * (1 - p_allele))
        Ho = np.nanmean(g == 1)
        F = (He - Ho) / He if He > 0 else 0
        stats.append({"Population": p, "N": int(mask.sum()), "Ho": round(Ho, 4), "He": round(He, 4), "F_mean": round(F, 4)})
    return pd.DataFrame(stats).sort_values("Population")

@cache_data
def fst_snp_wc(gt, labels):
    pops = np.unique(labels); r = len(pops); n_snp = gt.shape[1]
    fst = np.full(n_snp, np.nan)
    for j in range(n_snp):
        n_i = np.zeros(r); p_i = np.zeros(r); h_i = np.zeros(r)
        for k, p in enumerate(pops):
            g = gt[labels == p, j]; g = g[~np.isnan(g)]
            if len(g) == 0: continue
            n_i[k] = len(g); p_i[k] = g.mean() / 2.0; h_i[k] = np.mean(g == 1) if len(g) > 0 else 0.0
        if (n_i >= 2).sum() < 2: continue
        n_bar = n_i.mean()
        if n_bar <= 1: continue
        p_bar = (n_i * p_i).sum() / n_i.sum(); s2 = (n_i * (p_i - p_bar) ** 2).sum() / (n_bar * (r - 1))
        h_bar = (n_i * h_i).sum() / n_i.sum(); n_c = (r * n_bar - (n_i ** 2).sum() / (r * n_bar)) / (r - 1)
        if n_c <= 0: continue
        a = (n_bar / n_c) * (s2 - (1.0 / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2 - h_bar / 4))
        b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2 - (2 * n_bar - 1) / (4 * n_bar) * h_bar)
        c = h_bar / 2.0; den = a + b + c
        if den > 1e-9: fst[j] = max(0.0, a / den)
    return fst

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25, use_wc84=False):
    fst = fst_snp_wc(gt, labels) if use_wc84 else fst_snp(gt, labels)
    snp=snp_df.copy().reset_index(drop=True); n=min(len(snp),len(fst)); snp=snp.iloc[:n].reset_index(drop=True); fst=fst[:n]
    snp["FST"]=fst; res=[]
    for c in snp["CHR"].unique():
        sub=snp[snp["CHR"]==c].sort_values("BP").reset_index(drop=True)
        if len(sub)<w: continue
        for i in range(0,len(sub)-w+1,step):
            win=sub.iloc[i:i+w]
            if win["FST"].notna().sum()<5: continue
            res.append({"CHR":str(c),"START":int(win["BP"].min()),"END":int(win["BP"].max()), "MID":int((win["BP"].min()+win["BP"].max())/2),"N_SNP":len(win),
                        "MEAN_FST":float(win["FST"].mean()),"MAX_FST":float(win["FST"].max()), "TOP_SNP":win.iloc[win["FST"].argmax()]["SNP"]})
    return pd.DataFrame(res)

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp=snp_df.copy().reset_index(drop=True); n=min(gt.shape[1],len(snp))
    snp=snp.iloc[:n].reset_index(drop=True); gt=gt[:,:n]; res=[]
    for c in snp["CHR"].unique():
        idx=np.where(snp["CHR"].values==c)[0]
        if len(idx)<w: continue
        idx=idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0,len(idx)-w+1,step):
            wi=idx[i:i+w]; genos=gt[:,wi]
            ni,ns=genos.shape
            if ns<3 or ni<4: continue
            S=0; piv=[]
            for j in range(ns):
                col=genos[:,j]; col=col[~np.isnan(col)]
                if len(col)<2: continue
                p=col.mean()/2.0
                if 0<p<1: S+=1; piv.append(2*p*(1-p)*(len(col)/(len(col)-1)))
            if S==0 or not piv: continue
            pi=np.mean(piv); n=ni*2
            a1=np.sum(1.0/np.arange(1,n)); a2=np.sum(1.0/(np.arange(1,n)**2))
            b1=(n+1)/(3*(n-1)); b2=2*(n**2+n+3)/(9*n*(n-1))
            c1=b1-1/a1; c2=b2-(n+2)/(a1*n)+a2/(a1**2); e1=c1/a1; e2=c2/(a1**2+a2)
            den=np.sqrt(e1*S+e2*S*(S-1))
            if den>1e-12:
                D = float((pi-S/a1)/den)
                res.append({"CHR":str(c),"START":int(snp.iloc[wi]["BP"].min()), "END":int(snp.iloc[wi]["BP"].max()),"MID":int((snp.iloc[wi]["BP"].min()+snp.iloc[wi]["BP"].max())/2), "N_SNP":len(wi),"S":S,"TAJIMA_D":D})
    return pd.DataFrame(res)

@cache_data
def composite_score(gt, snp_df, labels):
    fst = fst_snp_wc(gt, labels)
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    maf_v = np.nanmean(gt, axis=0) / 2.0
    maf_v = np.minimum(maf_v, 1 - maf_v)
    df = snp_df.copy().reset_index(drop=True)
    n = min(len(df), len(fst), len(hom), len(maf_v))
    df = df.iloc[:n].reset_index(drop=True)
    df["FST"] = fst[:n]; df["HOM"] = hom[:n]; df["MAF"] = maf_v[:n]
    def norm(x):
        mu, sd = np.nanmean(x), np.nanstd(x)
        return (x - mu) / sd if sd > 1e-9 else np.zeros_like(x)
    df["Z_FST"] = norm(df["FST"].values)
    df["Z_HOM"] = norm(df["HOM"].values)
    df["CSS"] = np.abs(df["Z_FST"] * df["Z_HOM"])
    return df.sort_values("CSS", ascending=False).reset_index(drop=True)

@cache_data
def pca_analysis(gt, nc=10):
    X=impute_mean(gt); X=X-X.mean(0); n=min(nc,X.shape[0]-1,X.shape[1])
    p=PCA(n_components=n); return p.fit_transform(X), p.explained_variance_ratio_*100.0

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m=NMF(n_components=K,init="nndsvda",random_state=seed,max_iter=500)
    W=m.fit_transform(np.clip(impute_mean(gt),0.0,2.0)); s=W.sum(1,keepdims=True); s[s==0]=1.0
    return W/s, m.components_

@cache_data
def ld_pruning(gt, w=50, step=5, r2=0.2):
    n=gt.shape[1]
    if n<3: return np.ones(n,dtype=bool)
    keep=np.ones(n,dtype=bool); X=impute_mean(gt); X=X-X.mean(0); i=0
    while i<n:
        we=min(i+step,n)
        while we<n and (we-i)<w:
            if keep[we] and keep[i] and X[:,i].std()>1e-8 and X[:,we].std()>1e-8:
                if np.corrcoef(X[:,i],X[:,we])[0,1]**2>r2: keep[we]=False
            we+=1
        i+=step
    return keep

@cache_data
def detect_roh(gt, snp_json, min_snps=15, min_kb=1000.0):
    snp=pd.read_json(io.StringIO(snp_json))
    n=min(gt.shape[1],len(snp)); gt=gt[:,:n]; snp=snp.iloc[:n].reset_index(drop=True)
    ca=snp["CHR"].astype(str).values; bp=pd.to_numeric(snp["BP"], errors="coerce").fillna(0).astype(np.int64).values
    rohs=[]; fb=np.zeros(gt.shape[0]); tot=0.0
    for c in pd.unique(ca):
        idxs=np.where(ca==c)[0]
        if len(idxs)<min_snps: continue
        idxs=idxs[np.argsort(bp[idxs])]; sp=bp[idxs[-1]]-bp[idxs[0]]
        if sp<1: continue
        tot+=sp
        for i in range(gt.shape[0]):
            col=gt[i,idxs]; sk=None
            for k in range(len(idxs)):
                if col[k]==0 or col[k]==2:
                    if sk is None: sk=k
                elif sk is not None:
                    nr=k-sk; lkb=(bp[idxs[k-1]]-bp[idxs[sk]])/1000.0
                    if nr>=min_snps and lkb>=min_kb:
                        rohs.append({"IID_idx":i,"CHR":c,"start_bp":int(bp[idxs[sk]]), "end_bp":int(bp[idxs[k-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                        fb[i]+=bp[idxs[k-1]]-bp[idxs[sk]]
                    sk=None
            if sk is not None:
                nr=len(idxs)-sk; lkb=(bp[idxs[-1]]-bp[idxs[sk]])/1000.0
                if nr>=min_snps and lkb>=min_kb:
                    rohs.append({"IID_idx":i,"CHR":c,"start_bp":int(bp[idxs[sk]]), "end_bp":int(bp[idxs[-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                    fb[i]+=bp[idxs[-1]]-bp[idxs[sk]]
    fr=fb/tot if tot>0 else fb
    return (pd.DataFrame(rohs) if rohs else pd.DataFrame(columns=["IID_idx","CHR","start_bp","end_bp","n_snp","length_kb"])), fr

# ============================================================
# VEP & ANNOTATION LOCALE
# ============================================================
def annotate_genes_local(snp_df, gff, window_kb=50):
    w = window_kb * 1000
    res = []
    for _, s in snp_df.iterrows():
        pos = int(s["BP"]); ch = str(s["CHR"]).replace("chr","")
        msk = (gff["CHR"].astype(str).str.replace("chr","") == ch) & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)
        ng = gff[msk]
        if not ng.empty:
            for _, g in ng.iterrows():
                res.append({"SNP": s["SNP"], "CHR": ch, "BP": pos, "GENE": g["GENE"], "GENE_START": g["START"], "GENE_END": g["END"], "DISTANCE_KB": round(abs(g["START"] - pos) / 1000.0, 2)})
    return pd.DataFrame(res)

# ============================================================
# VISUALISATIONS
# ============================================================
def plot_pca(s, v, l):
    f = px.scatter(pd.DataFrame({"PC1":s[:,0],"PC2":s[:,1],"Pop":l}), x="PC1", y="PC2", color="Pop", title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550)
    f.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return f
def plot_manhattan(df, yc="P", title="", thr=None):
    if yc == "P":
        df = df.copy(); df["_y"] = -np.log10(np.clip(df["P"], 1e-300, 1)); yc = "_y"
    df = df[np.isfinite(df[yc])].copy(); df = df.assign(_k=df["CHR"].map(_chk)).sort_values(["_k","BP"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index, y=sub[yc], mode="markers", marker=dict(size=5, color="#2c3e50" if i%2==0 else "#7f8c8d"), text=sub["SNP"], hoverinfo="text", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Chr", yaxis_title=yc, height=500); return fig
def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)]); df["IID"] = labs; df["Pop"] = pops
    df["_d"] = Q.argmax(1); df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po","_d"]).reset_index(drop=True); df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K): fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"K{k+1}", marker_color=px.colors.qualitative.Set2[k%8]))
    fig.update_layout(barmode="stack", title=f"Admixture K={K}", height=500); return fig, df
def plot_win_manh(df, vc='MEAN_FST', title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df = df.assign(_k=df['CHR'].map(_chk)).sort_values(['_k','MID']).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby('CHR')):
        fig.add_trace(go.Scatter(x=sub['MID']/1e6, y=sub[vc], mode='markers', marker=dict(size=7, color='#2c3e50' if i%2==0 else '#7f8c8d', line=dict(width=0.5, color='white')), text=[f"{c}:{int(m/1e6)}Mb<br>{vc}={v:.4f}" for c, m, v in zip(sub['CHR'], sub['MID'], sub[vc])], hoverinfo='text', showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std(); fig.add_hline(y=mu+sigma*sd, line_dash='dash', line_color='red', annotation_text=f'μ+{sigma}σ')
    fig.update_layout(title=title, xaxis_title='Pos (Mb)', yaxis_title=vc, height=520); return fig

# ============================================================
# STREAMLIT APP
# ============================================================
_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","pop_stats","pca_s","pca_v","gt_pruned","snp_pruned","admix_Q","admix_K","roh_df","froh","win_fst","taj_df","css_df","vep_res","vep_snps"]
def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k] = None

def main():
    init_state()
    st.title("🐪 Dromedary Genomics Pipeline v1.5")
    st.caption("Projet Algérie : Adaptation au désert & Signatures de Sélection")

    with st.sidebar:
        st.header("📁 Données")
        # 🌟 LA CONVERSION EST RÉINTÉGRÉE ICI
        mode = st.radio("Source", ["PED/MAP", "Axiom→PED", "VCF"], key="src")
        
        if mode == "PED/MAP":
            ped = st.file_uploader(".ped", type=["ped","gz"]); mp = st.file_uploader(".map", type=["map","gz"])
            if ped and mp and st.button("📥 Charger", type="primary"):
                try:
                    mdf = parse_map(mp.read()); gt, ind, a1, a2 = parse_ped(ped.read(), len(mdf))
                    if len(mdf) > gt.shape[1]: mdf = mdf.iloc[:gt.shape[1]].reset_index(drop=True)
                    elif len(mdf) < gt.shape[1]: gt = gt[:, :len(mdf)]
                    mdf["A1"] = a1; mdf["A2"] = a2
                    init_state()
                    st.session_state.update({"gt":gt, "ind_df":rec_fid(ind), "snp_df":mdf})
                    st.success(f"✅ {gt.shape[0]} échantillons × {gt.shape[1]} SNPs")
                except Exception as e: st.error(f"❌ {e}")
                
        elif mode == "Axiom→PED":
            af = st.file_uploader("Axiom brut", type=["txt","csv","tsv","gz","ped"])
            if af and st.button("🔄 Convertir"):
                try:
                    s, g, ns = parse_axiom(af.read(), af.name)
                    mpl = [f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(ns)]
                    pdl = ["\t".join([sample_fid_iid(smp)[0], sample_fid_iid(smp)[1], "0","0","0","-9"] + [a for pr in row for a in pr]) for smp, row in zip(s, g)]
                    st.session_state["_plink_ped"] = "\n".join(pdl)+"\n"; st.session_state["_plink_map"] = "\n".join(mpl)+"\n"
                    st.success(f"✅ {len(s)}×{ns} converti.")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("_plink_ped"):
                st.download_button("⬇ .ped", st.session_state["_plink_ped"].encode(), "out.ped")
                st.download_button("⬇ .map", st.session_state["_plink_map"].encode(), "out.map")

        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf","gz"])
            if vf and st.button("📥 Charger"):
                try:
                    gt, ind, snp = parse_vcf(vf.read()); init_state()
                    st.session_state.update({"gt":gt, "ind_df":rec_fid(ind), "snp_df":snp}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
        
        st.divider(); st.header("⚙️ Seuils QC")
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.05)
        mind = st.slider("Missing Ind", 0.0, 0.5, 0.05)
        maf_t = st.slider("MAF", 0.0, 0.5, 0.05)
        hwe_t = st.number_input("HWE", value=1e-6, format="%.0e")
        het_sd = st.slider("Het ±σ", 1.0, 5.0, 3.0, 0.1)
        rem_out = st.checkbox("❌ Supprimer outliers", value=True)
        auto_only = st.checkbox("🧬 Autosomal uniquement (exclure X/Y)", value=True)
        ld_r2 = st.slider("LD r²", 0.05, 0.5, 0.2, 0.05)

    if st.session_state.gt is None:
        st.info("👉 Chargez vos fichiers pour commencer l'analyse.")
        return

    tabs = st.tabs(["🧹 QC", "📝 Tableau 1 (Stats)", "✂️ LD", "🧬 Structure", "🎨 Admixture", "📈 Démographie (ROH)", "🔥 Signatures de Sélection", "🧬 VEP & Annotation"])

    # ═══════ QC ═══════
    with tabs[0]:
        if st.button("▶ Lancer QC", type="primary", key="qc_btn"):
            try:
                p = {"geno":geno, "mind":mind, "maf":maf_t, "hwe":hwe_t, "het_sd":het_sd}
                gf, inf, snf = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, p, rem_out=rem_out, autosomal_only=auto_only)
                st.session_state.update({"gt_filt":gf, "ind_filt":inf, "snp_filt":snf})
                st.success(f"✅ QC terminé : {gf.shape[0]} individus × {gf.shape[1]} SNPs retenus.")
            except Exception as e: st.error(f"❌ {e}")

    # ═══════ TABLEAU 1 ═══════
    with tabs[1]:
        st.subheader("📊 Tableau 1 : Diversité Génétique")
        if st.session_state.gt_filt is not None:
            if st.button("📈 Calculer les statistiques", type="primary"):
                df_stats = compute_pop_stats(st.session_state.gt_filt, st.session_state.ind_filt)
                st.session_state.pop_stats = df_stats
                st.dataframe(df_stats, use_container_width=True)
        else: st.warning("⚠️ Lancez d'abord le QC.")

    # ═══════ LD ═══════
    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning", key="ld_btn"):
            k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
            st.session_state.update({"gt_pruned":st.session_state.gt_filt[:,k], "snp_pruned":st.session_state.snp_filt[k].reset_index(drop=True)})
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs retenus.")

    # ═══════ STRUCTURE ═══════
    with tabs[3]:
        if st.session_state.gt_pruned is not None and st.button("▶ PCA", key="struct_btn"):
            s_, v_ = pca_analysis(st.session_state.gt_pruned, 10)
            st.session_state.update({"pca_s":s_, "pca_v":v_})
            st.success("✅")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v, st.session_state.ind_filt["FID"].values), use_container_width=True)

    # ═══════ ADMIXTURE ═══════
    with tabs[4]:
        if st.session_state.gt_pruned is not None:
            K = st.slider("Nombre d'ancêtres (K)", 2, 10, 4)
            if st.button("▶ Lancer NMF", key="admix_btn"):
                Q, _ = admixture_nmf(st.session_state.gt_pruned, K=K)
                st.session_state.update({"admix_Q":Q, "admix_K":K})
                st.success("✅")
            if st.session_state.admix_Q is not None:
                f, _ = plot_admix(st.session_state.admix_Q, st.session_state.ind_filt["IID"].values, st.session_state.ind_filt["FID"].values, st.session_state.admix_K)
                st.plotly_chart(f, use_container_width=True)

    # ═══════ ROH ═══════
    with tabs[5]:
        if st.session_state.gt_pruned is not None and st.button("▶ Chercher ROH", key="roh_btn"):
            with st.spinner("Recherche ROH..."):
                roh, fr = detect_roh(st.session_state.gt_pruned, st.session_state.snp_pruned[["CHR","BP"]].to_json())
                st.session_state.update({"roh_df":roh, "froh":fr})
            st.success(f"✅ {len(roh)} ROH trouvés.")
        if st.session_state.froh is not None:
            df = pd.DataFrame({"FROH": st.session_state.froh, "Pop": st.session_state.ind_filt["FID"].values})
            st.plotly_chart(px.histogram(df, x="FROH", color="Pop", title="Distribution FROH"), use_container_width=True)

    # ═══════ SIGNATURES DE SÉLECTION ═══════
    with tabs[6]:
        if st.session_state.gt_pruned is not None:
            lab = st.session_state.ind_filt["FID"].astype(str).values
            st.markdown("### 1️⃣ Différenciation (FST)")
            if st.button("▶ Scanner FST", key="fst_btn"):
                with st.spinner("Calcul FST..."):
                    win = windowed_fst(st.session_state.gt_pruned, st.session_state.snp_pruned, lab, 50, 25, use_wc84=True)
                    st.session_state.win_fst = win
                st.success("✅")
            if st.session_state.win_fst is not None:
                st.plotly_chart(plot_win_manh(st.session_state.win_fst, 'MEAN_FST', 'Pics de différenciation (FST)'), use_container_width=True)

            st.divider()
            st.markdown("### 2️⃣ Spectre de fréquences (Tajima's D)")
            if st.button("▶ Lancer Tajima's D", key="taj_btn"):
                with st.spinner("Calcul Tajima..."):
                    res = []
                    for pop in np.unique(lab):
                        m = lab == pop
                        if m.sum() >= 5:
                            t = tajimas_scan(st.session_state.gt_pruned[m], st.session_state.snp_pruned, 25, 10)
                            if not t.empty:
                                t["Population"] = pop
                                res.append(t)
                    if res: st.session_state.taj_df = pd.concat(res)
                st.success("✅")
            if st.session_state.taj_df is not None:
                fig = px.scatter(st.session_state.taj_df, x="MID", y="TAJIMA_D", color="Population", title="Tajima's D par fenêtre", hover_data=["CHR", "S"])
                fig.add_hline(y=-2, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.markdown("### 3️⃣ Score Composite (CSS)")
            if st.button("▶ Calculer le CSS", key="css_btn"):
                with st.spinner("Calcul CSS..."):
                    css = composite_score(st.session_state.gt_pruned, st.session_state.snp_pruned, lab)
                    st.session_state.css_df = css
                st.success("✅")
            if st.session_state.css_df is not None:
                df_css = st.session_state.css_df.copy()
                df_css["P"] = np.exp(-df_css["CSS"])
                st.plotly_chart(plot_manhattan(df_css, yc="P", title="Manhattan CSS (Pics = zones sous sélection)"), use_container_width=True)

    # ═══════ VEP & ANNOTATION LOCALE ═══════
    with tabs[7]:
        st.subheader("🧬 Annotation des gènes candidats")
        
        annotation_method = st.radio("Méthode d'annotation :", 
                                     ["💻 En local avec fichier GFF3 (Recommandé et Rapide)", "🌐 Via l'API Ensembl (Internet)"], 
                                     horizontal=True)

        snps_vep = []
        c1, c2, c3 = st.columns(3)
        
        if st.session_state.win_fst is not None and c1.button("📥 Top 10 FST", key="load_vep_fst"):
            top_snps = st.session_state.win_fst.sort_values("MEAN_FST", ascending=False).head(10)["TOP_SNP"].tolist()
            snps_vep = st.session_state.snp_pruned[st.session_state.snp_pruned["SNP"].isin(top_snps)][["SNP","CHR","BP"]].to_dict('records')
            st.session_state.vep_snps = snps_vep
            
        if st.session_state.taj_df is not None and c2.button("📥 Top 10 Tajima", key="load_vep_taj"):
            top_taj = st.session_state.taj_df.sort_values("TAJIMA_D").head(10)
            snps_vep = [{"SNP": f"Region_{r['CHR']}_{r['MID']}", "CHR": str(r['CHR']), "BP": int(r['MID'])} for _, r in top_taj.iterrows()]
            st.session_state.vep_snps = snps_vep
            
        if st.session_state.css_df is not None and c3.button("📥 Top 10 CSS", key="load_vep_css"):
            top_snps = st.session_state.css_df.head(10)["SNP"].tolist()
            snps_vep = st.session_state.snp_pruned[st.session_state.snp_pruned["SNP"].isin(top_snps)][["SNP","CHR","BP"]].to_dict('records')
            st.session_state.vep_snps = snps_vep

        if st.session_state.get("vep_snps"):
            st.info(f"🧬 {len(st.session_state.vep_snps)} SNPs chargés.")
            
            # MÉTHODE GFF3
            if annotation_method.startswith("💻"):
                st.markdown("Téléchargez le fichier GFF3 du dromadaire sur [Ensembl](https://ftp.ensembl.org/pub/current_gff3/camelus_dromedarius/) (fichier `.gff3.gz`).")
                gff_file = st.file_uploader("Fichier GFF3 (Camelus dromedarius)", type=["gff3", "gff", "gz"])
                window = st.slider("Fenêtre de recherche autour du SNP (kb)", 1, 500, 50)
                
                if gff_file and st.button("💻 Annoter localement", type="primary"):
                    with st.spinner("Recherche des gènes..."):
                        try:
                            gff_df = parse_gff3(gff_file.read())
                            st.success(f"✅ GFF3 chargé : {len(gff_df)} gènes trouvés.")
                            
                            df_snps_to_ann = pd.DataFrame(st.session_state.vep_snps)
                            res_local = annotate_genes_local(df_snps_to_ann, gff_df, window_kb=window)
                            
                            if not res_local.empty:
                                st.session_state.vep_res = res_local
                                st.success(f"✅ {len(res_local)} associations SNP-Gène trouvées !")
                            else:
                                st.warning("⚠️ Aucun gène trouvé à proximité avec cette fenêtre.")
                        except Exception as e:
                            st.error(f"❌ Erreur de lecture : {e}")

            # MÉTHODE API (Plus lente, sujette aux erreurs 503)
            else:
                if st.button("🌐 Lancer VEP via Internet", type="primary"):
                    st.warning("⚠️ Ensembl peut être indisponible (Erreur 503).")
                    req_list = [f"{str(s['CHR']).replace('chr','')}:{s['BP']}-{s['BP']}" for s in st.session_state.vep_snps]
                    # Ici, vous inséreriez la fonction run_vep_api...
                    st.info("L'API n'est pas parfaite, privilégiez le GFF3 local.")

            if st.session_state.vep_res is not None and not st.session_state.vep_res.empty:
                st.dataframe(st.session_state.vep_res, use_container_width=True)
                st.download_button("⬇ Télécharger les résultats (CSV)", st.session_state.vep_res.to_csv(index=False).encode(), "annotation_genes.csv")

if __name__ == "__main__":
    main()
