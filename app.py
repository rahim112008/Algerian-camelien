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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import requests, time, json
import streamlit.components.v1 as components

try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🐪 Dromedary Pipeline v1.3", page_icon="🐪",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================
# 🔬 PARAMÈTRES SPÉCIFIQUES AU DROMADAIRE (Camelus dromedarius)
# ============================================================
ARS = {str(i): 100_000_000 for i in range(1, 37)} 
AUTO = [str(i) for i in range(1, 37)]

GROUPES_DEFAULT = {
    "Origine géographique (Algérie)": {
        "SAHARA": ["SAHRAOUI", "REGIBI", "DZA_AOS"],
        "STEPPE": ["OULED_SIDI_CHEIKH", "CHAAMBI", "DZA_ACM", "DZA_ANL"],
        "MONTAGNE": ["TARGUI", "DZA_ATR", "DZA_AZA"]
    }
}

def make_group_map(group_dict):
    return {r.upper(): grp for grp, races in group_dict.items() for r in races}

def apply_group_mapping(ind_df, group_map, default="AUTRE"):
    df = ind_df.copy(); df["FID"] = df["FID"].astype(str).str.upper()
    df["FID_GROUPE"] = df["FID"].map(group_map).fillna(default)
    return df

# ============================================================
# CACHE & UTILS
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

def parse_vcf(b):
    samples,rows=[],[]
    for line in _rt(b).splitlines():
        if line.startswith("#CHROM"): samples=line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p=line.split("\t")
            if len(p)>=10 and "," not in p[4]:
                fmt=p[8].split(":")
                if "GT" in fmt:
                    gi=fmt.index("GT"); gts=np.full(len(samples),np.nan,dtype=np.float32)
                    for k,s in enumerate(p[9:]):
                        al=s.split(":")[gi].replace("|","/").split("/")
                        if not any(a in (".","") for a in al):
                            try: gts[k]=sum(int(a) for a in al)
                            except: pass
                    try: bp = int(float(p[1]))
                    except (ValueError, TypeError): continue
                    rows.append({"CHR":str(p[0]),"SNP":p[2] or f"{p[0]}:{p[1]}","CM":0.0,"BP":bp, "A1":p[3],"A2":p[4],"GT":gts})
    if not rows: raise ValueError("Aucun variant VCF")
    gt=np.zeros((len(samples),len(rows)),dtype=np.float32)
    for j,r in enumerate(rows):
        gt[:,j]=r["GT"]; v=gt[:,j][~np.isnan(gt[:,j])]
        if len(v)>0 and v.mean()/2.0>0.5: gt[:,j]=np.where(np.isnan(gt[:,j]),np.nan,2.0-gt[:,j])
    return gt, pd.DataFrame({"FID":samples,"IID":samples}), pd.DataFrame([{k:v for k,v in r.items() if k!="GT"} for r in rows])

def rec_fid(ind):
    df=ind.copy()
    if df["FID"].nunique()<=1: df["FID"]=df["IID"].astype(str).str.split("_").str[0]
    return df

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
        tr["autosomal"] = int(m0 - gt.shape[1])
    k=miss_snp(gt)<=params["geno"]; gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True); tr["geno"]=int(m0-gt.shape[1])
    k=miss_ind(gt)<=params["mind"]; gt=gt[k]; ind_df=ind_df[k].reset_index(drop=True); tr["mind"]=int(n0-gt.shape[0])
    m=maf(gt); k=np.isfinite(m)&(m>=params["maf"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    pv=hwe_pvals(gt); k=np.isnan(pv)|(pv>=params["hwe"]); gt=gt[:,k]; snp_df=snp_df[k].reset_index(drop=True)
    if gt.shape[1]==0: raise ValueError("Tout exclu (MAF/HWE)")
    gt_pre,ind_pre,snp_pre=gt.copy(),ind_df.copy(),snp_df.copy()
    if rem_out:
        h=het(gt); kp=np.ones(len(h),dtype=bool)
        for f in ind_df["FID"].unique():
            msk=(ind_df["FID"]==f).values; sub=h[msk]
            if len(sub)>=5 and np.nanstd(sub)>1e-9:
                z=(sub-np.nanmean(sub))/np.nanstd(sub); idx=np.where(msk)[0]; kp[idx[np.abs(z)>params["het_sd"]]]=False
        gt=gt[kp]; ind_df=ind_df[kp].reset_index(drop=True); tr["het"]=int(len(h)-len(ind_df))
    return gt,ind_df,snp_df,gt_pre,ind_pre,snp_pre,{"n_ind_init":int(n0),"n_snp_init":int(m0),
        "n_ind_final":int(gt.shape[0]),"n_snp_final":int(gt.shape[1]),"excluded_ind":int(n0-gt.shape[0]),
        "excluded_snp":int(m0-gt.shape[1]),"trace":tr}

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

# ============================================================
# GÉNÉTIQUE & STRUCTURE
# ============================================================
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
def tajimas_window(genos):
    ni,ns=genos.shape
    if ns<3 or ni<4: return np.nan,0
    S=0; piv=[]
    for j in range(ns):
        col=genos[:,j]; col=col[~np.isnan(col)]
        if len(col)<2: continue
        p=col.mean()/2.0
        if 0<p<1: S+=1; piv.append(2*p*(1-p)*(len(col)/(len(col)-1)))
    if S==0 or not piv: return np.nan,int(S)
    pi=np.mean(piv); n=ni*2
    a1=np.sum(1.0/np.arange(1,n)); a2=np.sum(1.0/(np.arange(1,n)**2))
    b1=(n+1)/(3*(n-1)); b2=2*(n**2+n+3)/(9*n*(n-1))
    c1=b1-1/a1; c2=b2-(n+2)/(a1*n)+a2/(a1**2); e1=c1/a1; e2=c2/(a1**2+a2)
    den=np.sqrt(e1*S+e2*S*(S-1))
    if den<=1e-12: return np.nan,int(S)
    return float((pi-S/a1)/den),int(S)

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp=snp_df.copy().reset_index(drop=True); n=min(gt.shape[1],len(snp))
    snp=snp.iloc[:n].reset_index(drop=True); gt=gt[:,:n]; res=[]
    for c in snp["CHR"].unique():
        idx=np.where(snp["CHR"].values==c)[0]
        if len(idx)<w: continue
        idx=idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0,len(idx)-w+1,step):
            wi=idx[i:i+w]; D,S=tajimas_window(gt[:,wi])
            if not np.isnan(D): res.append({"CHR":str(c),"START":int(snp.iloc[wi]["BP"].min()), "END":int(snp.iloc[wi]["BP"].max()),"MID":int((snp.iloc[wi]["BP"].min()+snp.iloc[wi]["BP"].max())/2), "N_SNP":len(wi),"S":S,"TAJIMA_D":D})
    return pd.DataFrame(res)

# 🌟 CSS : COMPOSITE SELECTION SCORE (NOUVEAU)
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
def ld_decay(gt, bp, max_kb=1000, max_snp=1500, seed=42):
    n=gt.shape[1]
    if n<2: return pd.DataFrame(columns=["dist_kb","r2"])
    idx=np.sort(np.random.default_rng(seed).choice(n,max_snp,replace=False)) if n>max_snp else np.arange(n)
    g=impute_mean(gt[:,idx]); b=np.asarray(bp)[idx].astype(float)
    X=g-g.mean(0); s=g.std(0); s[s<1e-8]=np.nan; X=X/s
    C=(X.T@X)/X.shape[0]; R2=C**2; iu,ju=np.triu_indices(len(idx),k=1)
    d=(b[ju]-b[iu])/1000.0; msk=(d>0)&(d<=max_kb)
    return pd.DataFrame({"dist_kb":d[msk],"r2":R2[iu[msk],ju[msk]]})

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
# VEP & 3D
# ============================================================
_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}
def traduire_consequence_vep(c):
    c = str(c).lower()
    if "missense" in c: return "🔴 Exon (Faux-sens)"
    elif "synonymous" in c: return "🟡 Exon (Silencieuse)"
    elif "stop_gained" in c or "frameshift" in c: return "⛔ Exon (Non-sens)"
    elif "splice" in c: return "⚠️ Jonction"
    elif "intron" in c: return "🟢 Intron"
    elif "3_prime_utr" in c: return "🔵 3'UTR"
    elif "5_prime_utr" in c: return "🔵 5'UTR"
    elif "upstream" in c: return "🟣 Promoteur amont"
    elif "downstream" in c: return "🟣 Aval"
    return "⚪ Intergénique"

def run_vep_api(snp_list, species="camelus_dromedarius", batch=200):
    if not snp_list: return pd.DataFrame()
    url = f"https://rest.ensembl.org/vep/{species}/region?regulatory=1"
    headers = {"Content-Type":"application/json","Accept":"application/json"}
    results = []
    for i in range(0, len(snp_list), batch):
        chunk = snp_list[i:i+batch]
        try:
            r = requests.post(url, headers=headers, data=json.dumps({"variants":chunk}), timeout=60)
            if r.status_code == 200:
                for item in r.json():
                    sid = item.get("input", "N/A")
                    if item.get("transcript_consequences"):
                        cons = sorted(item["transcript_consequences"], key=lambda x: _SEVERITY.get(x.get("impact","MODIFIER"), 0), reverse=True)[0]
                        results.append({"SNP": sid, "GENE": cons.get("gene_symbol","N/A"), "CONSEQUENCE": ", ".join(cons.get("consequence_terms",["N/A"])),
                            "IMPACT": cons.get("impact","N/A"), "AMINO_ACID_CHANGE": cons.get("amino_acids","N/A")})
                    else:
                        results.append({"SNP": sid, "GENE": "Intergenic/Non-codant", "CONSEQUENCE": "N/A", "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A"})
        except Exception as e: st.warning(f"⚠️ VEP réseau: {e}")
        time.sleep(1.0)
    return pd.DataFrame(results)

def fetch_uniprot_id(gene_name, taxon="9838"):
    q = f"gene:{gene_name} AND organism_id:{taxon}"
    url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if data.get("results"): return data["results"][0].get("primaryAccession")
    except Exception: pass
    return None

def fetch_alphafold_pdb(uniprot_id):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            for rec in r.json():
                if rec.get("uniprotAccession") == uniprot_id:
                    pdb_url = rec.get("pdbUrl")
                    if pdb_url:
                        pr = requests.get(pdb_url, timeout=30)
                        if pr.status_code == 200: return pr.text
    except Exception: pass
    return None

def render_3d_structure(pdb_data, style="cartoon", color_scheme="spectrum", highlight_res=None):
    style_json = json.dumps({style: {'color': color_scheme}}); pdb_json = json.dumps(pdb_data); hl_js = ""
    if highlight_res and str(highlight_res).strip().isdigit(): hl_js = f"viewer.addStyle({{resi: '{highlight_res}'}}, {{stick: {{colorscheme: 'redCarbon'}}}});"
    html_code = f"""
    <div id="viewport" style="width: 100%; height: 500px; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let element = document.getElementById("viewport");
        let viewer = $3Dmol.createViewer(element, {{ backgroundColor: 'white' }});
        viewer.addModel({pdb_json}, "pdb");
        viewer.setStyle({{}}, {style_json});
        {hl_js}
        viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html_code, height=520)

# ============================================================
# VISUALISATIONS
# ============================================================
def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind","Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False); return f
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
def plot_ld(ld, b=20):
    if ld is None or ld.empty: return None
    d = ld.copy(); d["bin"] = (d["dist_kb"]//b)*b
    return px.line(d.groupby("bin")["r2"].mean().reset_index(), x="bin", y="r2", title="LD decay", height=450)
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
_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","qc_stats","pop_stats","pca_s","pca_v","gt_pruned","snp_pruned","admix_Q","admix_K","ld_df","roh_df","froh","win_fst","taj_df","css_df","vep_res"]
def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k] = None

def main():
    init_state()
    st.title("🐪 Dromedary Genomics Pipeline v1.3")
    st.caption("Projet Algérie : Adaptation au désert & Signatures de Sélection")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source", ["Demo","PED/MAP","VCF"], key="src")
        
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
        st.info("👉 Chargez vos fichiers PED et MAP pour commencer l'analyse.")
        return

    tabs = st.tabs([
        "🧹 QC", "📝 Tableau 1 (Stats)", "✂️ LD", "🧬 Structure", "🎨 Admixture", 
        "📈 Démographie (ROH)", "🔥 Signatures de Sélection", "🧬 VEP (Gènes)", "🧬 AlphaFold 3D"
    ])

    # ═══════ QC ═══════
    with tabs[0]:
        if st.button("▶ Lancer QC", type="primary", key="qc_btn"):
            try:
                p = {"geno":geno, "mind":mind, "maf":maf_t, "hwe":hwe_t, "het_sd":het_sd}
                gf, inf, snf, gp, ip, sp, qs = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, p, rem_out=rem_out, autosomal_only=auto_only)
                st.session_state.update({"gt_filt":gf, "ind_filt":inf, "snp_filt":snf, "qc_stats":qs})
                st.success(f"✅ QC terminé : {gf.shape[0]} individus × {gf.shape[1]} SNPs retenus.")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2 = st.columns(2)
            c1.metric("Individus retenus", s["n_ind_final"], delta=-s["excluded_ind"])
            c2.metric("SNPs retenus", s["n_snp_final"], delta=-s["excluded_snp"])

    # ═══════ TABLEAU 1 ═══════
    with tabs[1]:
        st.subheader("📊 Tableau 1 : Diversité Génétique")
        if st.session_state.gt_filt is not None:
            if st.button("📈 Calculer les statistiques", type="primary"):
                df_stats = compute_pop_stats(st.session_state.gt_filt, st.session_state.ind_filt)
                st.session_state.pop_stats = df_stats
                st.dataframe(df_stats, use_container_width=True)
                mean_F = df_stats["F_mean"].mean()
                if mean_F < 0:
                    st.success(f"💡 **Interprétation :** Moyenne F = {mean_F:.3f}. Valeur négative = excès d'hétérozygotes (absence de consanguinité, fort brassage).")
                else:
                    st.warning(f"💡 **Interprétation :** Moyenne F = {mean_F:.3f}. Valeur positive = consanguinité.")
        else: st.warning("⚠️ Lancez d'abord le QC.")

    # ═══════ LD ═══════
    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning", key="ld_btn"):
            k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
            st.session_state.update({"gt_pruned":st.session_state.gt_filt[:,k], "snp_pruned":st.session_state.snp_filt[k].reset_index(drop=True)})
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs retenus.")
        if st.session_state.gt_pruned is not None:
            if st.button("▶ LD Decay", key="ld_decay_btn"):
                st.session_state.ld_df = ld_decay(st.session_state.gt_pruned, st.session_state.snp_pruned["BP"].values)
            if st.session_state.ld_df is not None:
                st.plotly_chart(plot_ld(st.session_state.ld_df), use_container_width=True)

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
            st.plotly_chart(px.histogram(df, x="FROH", color="Pop", title="Distribution FROH (0 = pas de consanguinité)"), use_container_width=True)

    # ═══════ SIGNATURES DE SÉLECTION (FST, TAJIMA, CSS) ═══════
    with tabs[6]:
        st.subheader("🔥 Signatures de Sélection (Adaptation au désert)")
        st.markdown("Les balayages sélectifs (selective sweeps) se détectent par 3 méthodes complémentaires.")
        
        if st.session_state.gt_pruned is not None:
            lab = st.session_state.ind_filt["FID"].astype(str).values
            
            # 1. FST
            st.divider()
            st.markdown("### 1️⃣ Différenciation (FST)")
            c1, c2 = st.columns(2)
            w = c1.slider("Fenêtre FST", 10, 100, 50)
            s = c2.slider("Pas FST", 5, 50, 25)
            if st.button("▶ Scanner FST", key="fst_btn"):
                with st.spinner("Calcul FST..."):
                    win = windowed_fst(st.session_state.gt_pruned, st.session_state.snp_pruned, lab, w, s, use_wc84=True)
                    st.session_state.win_fst = win
                st.success("✅")
            if st.session_state.win_fst is not None:
                st.plotly_chart(plot_win_manh(st.session_state.win_fst, 'MEAN_FST', 'Pics de différenciation (FST)'), use_container_width=True)
                st.dataframe(st.session_state.win_fst.sort_values("MEAN_FST", ascending=False).head(5), use_container_width=True)

            # 2. Tajima's D
            st.divider()
            st.markdown("### 2️⃣ Spectre de fréquences (Tajima's D)")
            st.markdown("Un D < -2 indique une sélection positive ou une expansion récente.")
            c1, c2 = st.columns(2)
            tw = c1.slider("Fenêtre Tajima", 10, 50, 25)
            ts = c2.slider("Pas Tajima", 5, 20, 10)
            if st.button("▶ Lancer Tajima's D", key="taj_btn"):
                with st.spinner("Calcul Tajima..."):
                    res = []
                    for pop in np.unique(lab):
                        m = lab == pop
                        if m.sum() >= 5:
                            t = tajimas_scan(st.session_state.gt_pruned[m], st.session_state.snp_pruned, tw, ts)
                            if not t.empty:
                                t["Population"] = pop
                                res.append(t)
                    if res: st.session_state.taj_df = pd.concat(res)
                st.success("✅")
            if st.session_state.taj_df is not None:
                fig = px.scatter(st.session_state.taj_df, x="MID", y="TAJIMA_D", color="Population", title="Tajima's D par fenêtre", hover_data=["CHR", "S"])
                fig.add_hline(y=-2, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(st.session_state.taj_df.sort_values("TAJIMA_D").head(5), use_container_width=True)

            # 3. CSS (Composite Selection Score)
            st.divider()
            st.markdown("### 3️⃣ Score Composite (CSS)")
            st.markdown("Combine la différenciation (FST) et la réduction de diversité (Homozygotie). Idéal pour trouver les gènes de survie.")
            if st.button("▶ Calculer le CSS", key="css_btn"):
                with st.spinner("Calcul CSS..."):
                    css = composite_score(st.session_state.gt_pruned, st.session_state.snp_pruned, lab)
                    st.session_state.css_df = css
                st.success("✅")
            if st.session_state.css_df is not None:
                # On transforme le CSS en pseudo-pvalue pour le graphique Manhattan
                df_css = st.session_state.css_df.copy()
                df_css["P"] = np.exp(-df_css["CSS"])
                st.plotly_chart(plot_manhattan(df_css, yc="P", title="Manhattan CSS (Pics = zones sous sélection)"), use_container_width=True)
                st.dataframe(df_css.head(10), use_container_width=True)
        else:
            st.warning("⚠️ Lancez d'abord le LD Pruning.")

    # ═══════ VEP ═══════
    with tabs[7]:
        st.subheader("🧬 Annotation des gènes candidats (VEP)")
        snps_vep = []
        
        c1, c2, c3 = st.columns(3)
        if st.session_state.win_fst is not None:
            if c1.button("📥 Top 10 FST", key="load_vep_fst"):
                snps_vep = st.session_state.win_fst.sort_values("MEAN_FST", ascending=False).head(10)["TOP_SNP"].tolist()
                st.session_state.vep_snps = snps_vep
        if st.session_state.taj_df is not None:
            if c2.button("📥 Top 10 Tajima", key="load_vep_taj"):
                top_taj = st.session_state.taj_df.sort_values("TAJIMA_D").head(10)
                # Récupérer les SNPs (ici simplifié, on prend les régions)
                snps_vep = top_taj.apply(lambda r: f"{r['CHR']}:{int(r['START'])}-{int(r['END'])}", axis=1).tolist()
                st.session_state.vep_snps = snps_vep
        if st.session_state.css_df is not None:
            if c3.button("📥 Top 10 CSS", key="load_vep_css"):
                snps_vep = st.session_state.css_df.head(10)["SNP"].tolist()
                st.session_state.vep_snps = snps_vep
                
        if st.session_state.get("vep_snps"):
            st.info(f"SNPs chargés : {len(st.session_state.vep_snps)}")
            if st.button("🔬 Lancer VEP (Dromadaire)", type="primary"):
                with st.spinner("Interrogation Ensembl (Camelus dromedarius)..."):
                    res = run_vep_api(st.session_state.vep_snps)
                    if not res.empty:
                        res["LOCALISATION"] = res["CONSEQUENCE"].apply(traduction_consequence_vep)
                        st.session_state.vep_res = res
                st.success("✅")
                st.dataframe(res, use_container_width=True)
                st.download_button("⬇ Télécharger pour l'article", res.to_csv(index=False).encode(), "vep_genes.csv")

    # ═══════ 3D ═══════
    with tabs[8]:
        st.subheader("🧬 AlphaFold — Visualisation 3D")
        genes_to_structure = []
        if st.session_state.get("vep_res") is not None and "GENE" in st.session_state.vep_res.columns:
            genes_to_structure = [g for g in st.session_state.vep_res["GENE"].unique() if str(g).strip() not in ("N/A","Intergenic/Non-codant","Unknown","nan","")]
        if not genes_to_structure: st.warning("⚠️ Lancez d'abord VEP pour obtenir des gènes.")
        else:
            target_gene = st.selectbox("Gène", genes_to_structure)
            if st.button("🔬 Charger 3D", type="primary"):
                with st.spinner(f"Recherche UniProt pour {target_gene}..."):
                    uid = fetch_uniprot_id(target_gene, taxon="9838")
                    if uid:
                        st.success(f"✅ UniProt: {uid}"); pdb = fetch_alphafold_pdb(uid)
                        if pdb: render_3d_structure(pdb)
                        else: st.error("❌ AlphaFold introuvable.")
                    else: st.error("❌ Gène non trouvé.")

if __name__ == "__main__":
    main()
