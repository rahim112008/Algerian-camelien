"""🐪 Camel Genomics Pipeline v22.0 — Basé sur Bovin v12.0 (Corrigé)"""
import gzip, io, os, warnings, zipfile, subprocess, tempfile, shutil, gc
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

st.set_page_config(page_title="🐪 Camel Pipeline v22.0", page_icon="🐪",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CONSTANTES GÉNOMIQUES — Camelus dromedarius (2n=74)
# ============================================================
ARS = {"1": 124.7, "2": 123.8, "3": 120.7, "4": 68.3, "5": 97.5, "6": 97.5, "7": 91.2, "8": 89.4,
       "9": 86.7, "10": 84.1, "11": 81.5, "12": 78.9, "13": 76.3, "14": 73.8, "15": 71.2, "16": 68.6,
       "17": 66.1, "18": 63.5, "19": 60.9, "20": 58.4, "21": 55.8, "22": 53.2, "23": 50.7, "24": 48.1,
       "25": 45.5, "26": 43.0, "27": 40.4, "28": 37.8, "29": 35.3, "30": 32.7, "31": 30.1, "32": 27.6,
       "33": 25.0, "34": 22.4, "35": 19.9, "36": 17.3, "X": 140.0, "Y": 45.0, "MT": 16.6}
AUTO = [str(i) for i in range(1, 37)]

GROUPES_DEFAULT = {
    "Niveau de production laitière": {"HAUTE_PRODUCTION_LAITIERE": ["MOJAHEEM", "MAGHATEER", "WADAH", "AWARIK"],
                                       "FAIBLE_PRODUCTION_LAITIERE": ["BIKANERI", "JAISALMERI", "KACHCHHI", "MEWARI"]},
    "Type de production principal": {"LAIT": ["MOJAHEEM", "MAGHATEER", "WADAH"], "VIANDE": ["JANDAWEL", "NABUL"], "MIXTE": ["KAZAKH_BACTRIAN", "ARVANA"]},
    "Adaptation climatique": {"CHAUD": ["MOJAHEEM", "BIKANERI"], "FROID": ["KAZAKH_BACTRIAN", "ARVANA"]},
    "Sous-espèce": {"DROMADAIRE": ["MOJAHEEM", "MAGHATEER", "WADAH", "BIKANERI"], "BACTRIANE": ["KAZAKH_BACTRIAN", "ARVANA"]}
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

def _autw():
    L=np.array([ARS[c] for c in AUTO],dtype=float); return L/L.sum()
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

# ============================================================
# PARSERS
# ============================================================
def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if fn.endswith(".gz") or raw_bytes[:2]==b"\x1f\x8b" else raw_bytes
    lines=[l.rstrip("\r\n") for l in raw.decode("utf-8",errors="replace").splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    start = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample") or "Sample Filename" in lines[0][:50]) else 0
    samples, gtypes = [], []
    for line in lines[start:]:
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts)<2: continue
        samples.append(parts[0].strip()); alleles=[]
        for cell in parts[1:]:
            tk=cell.split()
            if len(tk)==2: alleles.append((_ca(tk[0]),_ca(tk[1])))
            elif len(tk)==1 and len(tk[0])==2 and all(ch in "ACGT0" for ch in tk[0]): alleles.append((tk[0][0],tk[0][1]))
            else: alleles.append(("0","0"))
        gtypes.append(alleles)
    if not samples: raise ValueError("Aucun échantillon")
    c=Counter([len(r) for r in gtypes]).most_common(1)[0][0]
    for i,r in enumerate(gtypes): gtypes[i]=r[:c]+[("0","0")]*(c-len(r))
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
    df = pd.DataFrame(rows)
    # 🔬 CORRECTION DES UNITÉS (Mb/Kb -> bp)
    if df["BP"].max() < 10000: df["BP"] = df["BP"] * 1_000_000
    elif df["BP"].max() < 10_000_000: df["BP"] = df["BP"] * 1_000
    return df

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
                    rows.append({"CHR":str(p[0]),"SNP":p[2] or f"{p[0]}:{p[1]}","CM":0.0,"BP":bp,
                                 "A1":p[3],"A2":p[4],"GT":gts})
    if not rows: raise ValueError("Aucun variant VCF")
    gt=np.zeros((len(samples),len(rows)),dtype=np.float32)
    for j,r in enumerate(rows):
        gt[:,j]=r["GT"]; v=gt[:,j][~np.isnan(gt[:,j])]
        if len(v)>0 and v.mean()/2.0>0.5: gt[:,j]=np.where(np.isnan(gt[:,j]),np.nan,2.0-gt[:,j])
    df_snp = pd.DataFrame([{k:v for k,v in r.items() if k!="GT"} for r in rows])
    if df_snp["BP"].max() < 10000: df_snp["BP"] = df_snp["BP"] * 1_000_000
    elif df_snp["BP"].max() < 10_000_000: df_snp["BP"] = df_snp["BP"] * 1_000
    return gt, pd.DataFrame({"FID":samples,"IID":samples}), df_snp

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

def parse_gmt(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace"); sets = {}
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        term = parts[0]; genes = [g for g in parts[2:] if g]
        if genes: sets[term] = genes
    return sets

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8",errors="replace"); sep = "\t" if text.count("\t")>text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    if mapping_df is not None:
        p,m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal","IID","id","ID"] if c in p.columns), None)
        if id_c is None: raise ValueError("Colonne ID absente.")
        p[id_c]=p[id_c].astype(str)
        map_a = next((c for c in m.columns if c!="IID"), None)
        m[map_a]=m[map_a].astype(str); m["IID"]=m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left")
        ind = ind_df.copy(); ind["_k"]=ind["IID"].astype(str); pm["_k"]=pm["IID"].astype(str)
        return ind[["_k","FID","IID"]].merge(pm.drop(columns=["IID"],errors="ignore"),on="_k",how="left").drop(columns="_k"), id_c
    id_c = next((c for c in ["IID","id","ID","sample","Animal","animal"] if c in pheno_df.columns), None)
    if id_c is None: raise ValueError("Aucune colonne ID.")
    p = pheno_df.copy(); p[id_c]=p[id_c].astype(str)
    ind = ind_df.copy(); ind["_k"]=ind["IID"].astype(str); p["_k"]=p[id_c]
    return ind[["_k","FID","IID"]].merge(p.drop(columns=[id_c]),on="_k",how="left").drop(columns="_k"), id_c

def gen_demo(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng=np.random.default_rng(seed); pops=["MOJAHEEM", "MAGHATEER", "WADAH", "AWARIK", "BIKANERI", "JAISALMERI", "KAZAKH_BACTRIAN", "ARVANA"][:min(n_pop,8)]
    pa=rng.beta(0.6,0.6,n_snp); gt=np.zeros((n_ind,n_snp),dtype=np.float32); ir=[]; k=0
    for p in pops:
        pk=np.clip(pa+rng.normal(0,0.15,n_snp),0.02,0.98)
        for i in range(n_ind//len(pops)+(1 if pops.index(p)<n_ind%len(pops) else 0)):
            gt[k]=(rng.random(n_snp)<pk).astype(np.int8)+(rng.random(n_snp)<pk).astype(np.int8)
            ir.append({"FID":p,"IID":f"{p}_{i+1:03d}"}); k+=1
    gt[rng.random(gt.shape)<0.02]=np.nan
    ch=rng.choice(AUTO,n_snp,p=_autw()); bp=np.array([rng.integers(1,ARS[c]*1_000_000) for c in ch],dtype=np.int64)
    snp=pd.DataFrame({"CHR":ch,"SNP":[f"rs{i:07d}" for i in range(n_snp)],"CM":bp/1e6,"BP":bp})
    return gt, pd.DataFrame(ir), snp.sort_values(by=["CHR","BP"],
        key=lambda x: x.map(_chk) if x.name=="CHR" else x).reset_index(drop=True)

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

def apply_qc(gt, ind_df, snp_df, params, keep_autosomes=True, rem_out=True):
    gt,ind_df,snp_df=sync_dims(gt,ind_df,snp_df); n0,m0=gt.shape; tr={}
    
    if keep_autosomes:
        mask = snp_df["CHR"].astype(str).str.upper().isin([a.upper() for a in AUTO])
        gt = gt[:, mask]; snp_df = snp_df[mask].reset_index(drop=True)
        tr["autosomes"] = int(m0 - gt.shape[1])
        
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
    gc.collect()
    return gt,ind_df,snp_df,gt_pre,ind_pre,snp_pre,{"n_ind_init":int(n0),"n_snp_init":int(m0),
        "n_ind_final":int(gt.shape[0]),"n_snp_final":int(gt.shape[1]),"excluded_ind":int(n0-gt.shape[0]),
        "excluded_snp":int(m0-gt.shape[1]),"trace":tr}

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
            n_i[k] = len(g); p_i[k] = g.mean() / 2.0
            h_i[k] = np.mean(g == 1) if len(g) > 0 else 0.0
        if (n_i >= 2).sum() < 2: continue
        n_bar = n_i.mean()
        if n_bar <= 1: continue
        p_bar = (n_i * p_i).sum() / n_i.sum()
        s2 = (n_i * (p_i - p_bar) ** 2).sum() / (n_bar * (r - 1))
        h_bar = (n_i * h_i).sum() / n_i.sum()
        n_c = (r * n_bar - (n_i ** 2).sum() / (r * n_bar)) / (r - 1)
        if n_c <= 0: continue
        a = (n_bar / n_c) * (s2 - (1.0 / (n_bar - 1)) *
             (p_bar * (1 - p_bar) - (r - 1) / r * s2 - h_bar / 4))
        b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2
             - (2 * n_bar - 1) / (4 * n_bar) * h_bar)
        c = h_bar / 2.0
        den = a + b + c
        if den > 1e-9: fst[j] = max(0.0, a / den)
    return fst

@cache_data
def pca_analysis(gt, nc=10):
    X=impute_mean(gt); X=X-X.mean(0); n=min(nc,X.shape[0]-1,X.shape[1])
    p=PCA(n_components=n); return p.fit_transform(X), p.explained_variance_ratio_*100.0

@cache_data
def mds_analysis(gt, nc=5):
    X=impute_mean(gt); n=X.shape[0]; D=np.zeros((n,n),dtype=np.float32)
    for i in range(n): D[i]=np.abs(X-X[i]).sum(1)/X.shape[1]
    return SklearnMDS(n_components=nc,dissimilarity="precomputed",random_state=42,
                      n_init=1,max_iter=300,normalized_stress=False).fit_transform(D)

@cache_data
def king_mat(gt):
    h = np.nansum(gt == 1, axis=1).astype(np.float64)
    H = (gt == 1).astype(np.float32); BH = H @ H.T; d = h[:, None] + h[None, :]
    with np.errstate(divide="ignore", invalid="ignore"): K = (d - 2 * BH) / d
    K[~np.isfinite(K)] = 0.0; np.fill_diagonal(K, 0.5); return K

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m=NMF(n_components=K,init="nndsvda",random_state=seed,max_iter=500)
    W=m.fit_transform(np.clip(impute_mean(gt),0.0,2.0)); s=W.sum(1,keepdims=True); s[s==0]=1.0
    return W/s, m.components_

@cache_data
def choose_best_k(gt, k_min=2, k_max=8, seed=42):
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    rows = []
    for k in range(k_min, k_max + 1):
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        W = m.fit_transform(X); H = m.components_
        rmse = float(np.sqrt(np.mean((X - W @ H) ** 2)))
        rows.append({"K": k, "RMSE": rmse})
    return pd.DataFrame(rows)

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

# 🔬 NOUVELLE FONCTION LD CORRIGÉE (Fenêtre glissante)
@cache_data
def ld_decay(gt, snp_df, max_kb=1000, max_snp=3000):
    n = gt.shape[1]
    if n < 10: return pd.DataFrame(columns=["dist_kb","r2","count"])
    
    df_snp = snp_df.copy()
    df_snp["CHR"] = df_snp["CHR"].astype(str)
    df_snp["BP"] = df_snp["BP"].astype(float)
    
    df_snp["CHR_num"] = df_snp["CHR"].map(lambda x: _chk(x)[1])
    df_snp = df_snp.sort_values(["CHR_num", "BP"]).reset_index(drop=True)
    
    if n > max_snp:
        idx = np.linspace(0, n-1, max_snp).astype(int)
    else:
        idx = np.arange(n)
        
    gt = gt[:, idx]; df_snp = df_snp.iloc[idx].reset_index(drop=True)
    n = len(idx)

    X = impute_mean(gt); X = X - X.mean(0)
    stds = X.std(0); stds[stds == 0] = 1e-9; X = X / stds
    
    distances = []; r2_vals = []
    bp = df_snp["BP"].values; chrs = df_snp["CHR"].values; max_bp = max_kb * 1000
    
    for i in range(n):
        j_end = min(i+100, n) 
        same_chr = (chrs[i] == chrs[i+1:j_end])
        if not np.any(same_chr): continue
        dists = np.abs(bp[i] - bp[i+1:j_end])
        valid = same_chr & (dists <= max_bp) & (dists > 0)
        if np.any(valid):
            r = np.dot(X[:, i], X[:, i+1:j_end][:, valid]) / X.shape[0]
            distances.extend(dists[valid] / 1000.0)
            r2_vals.extend(r**2)
            
    if not distances: return pd.DataFrame(columns=["dist_kb","r2","count"])
    
    df = pd.DataFrame({"dist_kb": distances, "r2": r2_vals})
    bins = np.linspace(0, max_kb, 30)
    df["bin"] = pd.cut(df["dist_kb"], bins=bins, labels=False, include_lowest=True)
    res = df.groupby("bin").agg(dist_kb=("dist_kb", "mean"), r2=("r2", "mean"), count=("r2", "count")).dropna().sort_values("dist_kb")
    return res

@cache_data
def detect_roh(gt, snp_json, min_snps=30, min_kb=500.0):
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
                        rohs.append({"IID_idx":i,"CHR":c,"start_bp":int(bp[idxs[sk]]),
                                     "end_bp":int(bp[idxs[k-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                        fb[i]+=bp[idxs[k-1]]-bp[idxs[sk]]
                    sk=None
            if sk is not None:
                nr=len(idxs)-sk; lkb=(bp[idxs[-1]]-bp[idxs[sk]])/1000.0
                if nr>=min_snps and lkb>=min_kb:
                    rohs.append({"IID_idx":i,"CHR":c,"start_bp":int(bp[idxs[sk]]),
                                 "end_bp":int(bp[idxs[-1]]),"n_snp":int(nr),"length_kb":float(lkb)})
                    fb[i]+=bp[idxs[-1]]-bp[idxs[sk]]
    fr=fb/tot if tot>0 else fb
    return (pd.DataFrame(rohs) if rohs else pd.DataFrame(columns=["IID_idx","CHR","start_bp","end_bp","n_snp","length_kb"])), fr

@cache_data
def fst_pairwise(gt, labels, use_wc84=False):
    pops=sorted(np.unique(labels)); K=len(pops); M=np.full((K,K),np.nan); np.fill_diagonal(M,0.0)
    masks={p:(labels==p) for p in pops}
    for i in range(K):
        for j in range(i+1,K):
            sub = gt[masks[pops[i]] | masks[pops[j]]]
            sub_lab = labels[masks[pops[i]] | labels[pops[j]]]
            if use_wc84:
                v = fst_snp_wc(sub, sub_lab)
                M[i,j]=M[j,i]=float(np.nanmean(v))
            else:
                g1=gt[masks[pops[i]]]; g2=gt[masks[pops[j]]]
                p1=np.nanmean(g1,0)/2.0; p2=np.nanmean(g2,0)/2.0
                n1=(~np.isnan(g1)).sum(0); n2=(~np.isnan(g2)).sum(0)
                dn=np.where(n1+n2>0,n1+n2,np.nan); pb=(p1*n1+p2*n2)/dn
                hs=(2*p1*(1-p1)*n1+2*p2*(1-p2)*n2)/dn; ht=2*pb*(1-pb)
                with np.errstate(divide="ignore",invalid="ignore"):
                    fj=np.clip((ht-hs)/ht, 0.0, None)
                M[i,j]=M[j,i]=float(np.nanmean(fj))
    return M, pops

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25, use_wc84=False):
    fst = fst_snp_wc(gt, labels) if use_wc84 else fst_snp_wc(gt, labels)
    snp=snp_df.copy().reset_index(drop=True)
    n=min(len(snp),len(fst)); snp=snp.iloc[:n].reset_index(drop=True); fst=fst[:n]
    snp["FST"]=fst; res=[]
    for c in snp["CHR"].unique():
        sub=snp[snp["CHR"]==c].sort_values("BP").reset_index(drop=True)
        if len(sub)<w: continue
        for i in range(0,len(sub)-w+1,step):
            win=sub.iloc[i:i+w]
            if win["FST"].notna().sum()<5: continue
            res.append({"CHR":str(c),"START":int(win["BP"].min()),"END":int(win["BP"].max()),
                        "MID":int((win["BP"].min()+win["BP"].max())/2),"N_SNP":len(win),
                        "MEAN_FST":float(win["FST"].mean()),"MAX_FST":float(win["FST"].max()),
                        "TOP_SNP":win.iloc[win["FST"].argmax()]["SNP"]})
    return pd.DataFrame(res)

@cache_data
def detect_outliers(win_df, z_thr=3.0):
    df=win_df.copy(); mu,sd=df["MEAN_FST"].mean(),df["MEAN_FST"].std()
    df["Z_FST"]=(df["MEAN_FST"]-mu)/sd if sd>1e-9 else 0.0
    df["OUTLIER"]=df["Z_FST"]>=z_thr
    return df.sort_values("MEAN_FST",ascending=False).reset_index(drop=True)

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
            if not np.isnan(D): res.append({"CHR":str(c),"START":int(snp.iloc[wi]["BP"].min()),
                "END":int(snp.iloc[wi]["BP"].max()),"MID":int((snp.iloc[wi]["BP"].min()+snp.iloc[wi]["BP"].max())/2),
                "N_SNP":len(wi),"S":S,"TAJIMA_D":D})
    return pd.DataFrame(res)

@cache_data
def composite_score(gt, snp_df, labels):
    fst=fst_snp_wc(gt,labels); hom=np.nanmean((gt==0)|(gt==2),axis=0)
    maf_v=np.nanmean(gt,axis=0)/2.0; maf_v=np.minimum(maf_v,1-maf_v)
    df=snp_df.copy().reset_index(drop=True); n=min(len(df),len(fst),len(hom),len(maf_v))
    df=df.iloc[:n].reset_index(drop=True); df["FST"]=fst[:n]; df["HOM"]=hom[:n]; df["MAF"]=maf_v[:n]
    def norm(x): mu,sd=np.nanmean(x),np.nanstd(x); return (x-mu)/sd if sd>1e-9 else np.zeros_like(x)
    df["Z_FST"]=norm(df["FST"].values); df["Z_HOM"]=norm(df["HOM"].values)
    df["CSS"]=np.abs(df["Z_FST"]*df["Z_HOM"])
    return df.sort_values("CSS",ascending=False).reset_index(drop=True)

@cache_data
def reynolds(gt, labels):
    pops=sorted(np.unique(labels)); K=len(pops)
    ph={p:np.nanmean(gt[labels==p],0)/2.0 for p in pops}
    D=np.zeros((K,K))
    for i in range(K):
        for j in range(i+1,K):
            pi,pj=ph[pops[i]],ph[pops[j]]; num=np.nansum((pi-pj)**2)
            dn=np.nansum(pi*(1-pj)+pj*(1-pi))
            if dn>0: D[i,j]=D[j,i]=num/dn
    return D,pops

@cache_data
def upgma_bootstrap(gt, snp_df, labels, n_boot=100, seed=42):
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    labels=np.asarray(labels).astype(str); pops=sorted(np.unique(labels)); K=len(pops); n_snp=gt.shape[1]
    def cR(g):
        ph={p:np.nanmean(g[labels==p],0)/2.0 for p in pops}; D=np.zeros((K,K))
        for i in range(K):
            for j in range(i+1,K):
                pi,pj=ph[pops[i]],ph[pops[j]]; num=np.nansum((pi-pj)**2)
                dn=np.nansum(pi*(1-pj)+pj*(1-pi))
                if dn>0: D[i,j]=D[j,i]=num/dn
        return D
    Do=cR(gt); Zo=linkage(squareform(Do,checks=False),method="average"); nl=K
    def gc(Z):
        cl={}
        for k,row in enumerate(Z):
            i,j=int(row[0]),int(row[1])
            l=frozenset([i]) if i<nl else cl[i]; r=frozenset([j]) if j<nl else cl[j]
            cl[nl+k]=frozenset(l|r)
        return cl
    co=gc(Zo); sc=np.zeros(len(Zo)); rng=np.random.default_rng(seed)
    for b in range(n_boot):
        idx=rng.choice(n_snp,n_snp,replace=True); Db=cR(gt[:,idx])
        Zb=linkage(squareform(Db,checks=False),method="average"); cb=set(gc(Zb).values())
        for k,cl in co.items():
            zi=k-nl
            if 0<=zi<len(Zo) and cl in cb: sc[zi]+=1
    supp=(sc/n_boot*100).round(0); tr=to_tree(Zo)
    def nw(node):
        if node.is_leaf(): return f"{pops[node.id]}"
        l=nw(node.get_left()); r=nw(node.get_right())
        if node.id>=nl: return f"({l},{r}){int(supp[node.id-nl])}:{node.dist:.4f}"
        return f"({l},{r}):{node.dist:.4f}"
    return {"tree":tr,"Z":Zo,"newick":nw(tr)+";","support":supp,"D":Do,"pops":pops}

@cache_data
def midpoint_info(D, labels):
    D=np.asarray(D,dtype=float); n=len(labels); D=(D+D.T)/2; np.fill_diagonal(D,0)
    im,jm,md=0,1,0.0
    for i in range(n):
        for j in range(i+1,n):
            if D[i,j]>md: md=D[i,j]; im,jm=i,j
    return {"leaf_i":labels[im],"leaf_j":labels[jm],"max_dist":float(md),"midpoint":float(md/2)}

# ============================================================
# GWAS
# ============================================================
def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9; H = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        try: step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError: return beta[-1], None
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30); mu = 1.0/(1.0+np.exp(-eta)); W = mu*(1-mu)+1e-9
    try:
        cov = np.linalg.inv((X.T * W) @ X + 1e-8 * np.eye(X.shape[1]))
        se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError: return float(beta[-1]), None
    return float(beta[-1]), se

@cache_data
def gwas_pairwise(gt, snp_df, labels, ra, rb, maf_min=0.05, method="allelic", covs=None):
    labels=np.asarray(labels).astype(str); mask=np.isin(labels,[ra,rb])
    if mask.sum()<20: raise ValueError(f"<20 individus pour {ra} vs {rb}")
    gab=gt[mask]; y=(labels[mask]==ra).astype(int)
    snp=snp_df.reset_index(drop=True); n=min(gab.shape[1],len(snp))
    gab=gab[:,:n]; snp=snp.iloc[:n].reset_index(drop=True); rows=[]
    if covs is not None: covs=covs[mask]
    for j in range(n):
        g=gab[:,j]; gA=g[y==1]; gB=g[y==0]; gA=gA[~np.isnan(gA)]; gB=gB[~np.isnan(gB)]
        if len(gA)<5 or len(gB)<5: continue
        pA=gA.mean()/2.0; pB=gB.mean()/2.0; tot=(gA.sum()+gB.sum())/(2*(len(gA)+len(gB))); mj=min(tot,1-tot)
        if mj<maf_min: continue
        if method=="allelic":
            tab=np.array([[2*len(gA)-gA.sum(),gA.sum()],[2*len(gB)-gB.sum(),gB.sum()]])
            try: chi2,p,_,_=stats.chi2_contingency(tab,correction=False)
            except: continue
        else:
            X=np.column_stack([np.ones(len(y))] + ([covs] if covs is not None else []) + [g])
            val=~np.isnan(X).any(1); Xv=X[val]; yv=y[val]
            beta_snp, se = _logistic_irls(Xv, yv)
            if se is None or se <= 0: continue
            zv = beta_snp / se; chi2 = zv**2
            p = float(stats.norm.sf(abs(zv))*2)
        hs=(2*pA*(1-pA)*len(gA)+2*pB*(1-pB)*len(gB))/(len(gA)+len(gB))
        pb=(pA*len(gA)+pB*len(gB))/(len(gA)+len(gB)); ht=2*pb*(1-pb)
        fv=max(0.0,(ht-hs)/ht) if ht>1e-9 else 0.0
        rows.append({"SNP":snp.iloc[j]["SNP"],"CHR":str(snp.iloc[j]["CHR"]),"BP":int(snp.iloc[j]["BP"]),
                     "freq_A":round(pA,4),"freq_B":round(pB,4),"delta_freq":round(pA-pB,4),
                     "MAF":round(mj,4),"chi2":round(chi2,4),"P":p,"FST_snp":round(fv,4),
                     "n_A":len(gA),"n_B":len(gB)})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values("P").reset_index(drop=True)
    return df

@cache_data
def gwas_pairwise_group(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None):
    groups=np.asarray(groups).astype(str); mask=np.isin(groups,[ga,gb])
    if mask.sum()<20: raise ValueError(f"<20 individus pour {ga} vs {gb}")
    gab=gt[mask]; y=(groups[mask]==ga).astype(int)
    snp=snp_df.reset_index(drop=True); n=min(gab.shape[1],len(snp))
    gab=gab[:,:n]; snp=snp.iloc[:n].reset_index(drop=True); rows=[]
    if covs is not None: covs=covs[mask]
    for j in range(n):
        g=gab[:,j]; gA=g[y==1]; gB=g[y==0]; gA=gA[~np.isnan(gA)]; gB=gB[~np.isnan(gB)]
        if len(gA)<5 or len(gB)<5: continue
        pA=gA.mean()/2.0; pB=gB.mean()/2.0; tot=(gA.sum()+gB.sum())/(2*(len(gA)+len(gB))); mj=min(tot,1-tot)
        if mj<maf_min: continue
        if method=="allelic":
            tab=np.array([[2*len(gA)-gA.sum(),gA.sum()],[2*len(gB)-gB.sum(),gB.sum()]])
            try: chi2,p,_,_=stats.chi2_contingency(tab,correction=False)
            except: continue
        else:
            X=np.column_stack([np.ones(len(y))] + ([covs] if covs is not None else []) + [g])
            val=~np.isnan(X).any(1); Xv=X[val]; yv=y[val]
            beta_snp, se = _logistic_irls(Xv, yv)
            if se is None or se <= 0: continue
            zv = beta_snp / se; chi2 = zv**2
            p = float(stats.norm.sf(abs(zv))*2)
        hs=(2*pA*(1-pA)*len(gA)+2*pB*(1-pB)*len(gB))/(len(gA)+len(gB))
        pb=(pA*len(gA)+pB*len(gB))/(len(gA)+len(gB)); ht=2*pb*(1-pb)
        fv=max(0.0,(ht-hs)/ht) if ht>1e-9 else 0.0
        rows.append({"SNP":snp.iloc[j]["SNP"],"CHR":str(snp.iloc[j]["CHR"]),"BP":int(snp.iloc[j]["BP"]),
                     "freq_A":round(pA,4),"freq_B":round(pB,4),"delta_freq":round(pA-pB,4),
                     "MAF":round(mj,4),"chi2":round(chi2,4),"P":p,"FST_snp":round(fv,4),
                     "n_A":len(gA),"n_B":len(gB)})
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values("P").reset_index(drop=True)
    return df

@cache_data
def get_pairs(labels):
    r=sorted(pd.unique(labels).astype(str))
    return r,[(r[i],r[j]) for i in range(len(r)) for j in range(i+1,len(r))]

def _vc_delta(Y, X0, K):
    n=len(Y); d,U=np.linalg.eigh(K); d=np.maximum(d,0); Yt=U.T@Y; X0t=U.T@X0
    def nll(dl):
        if dl<=0: return 1e18
        w=1.0/np.sqrt(dl*d+1.0); Yw=Yt*w; Xw=X0t*w[:,None]
        try: b,*_=np.linalg.lstsq(Xw,Yw,rcond=None)
        except: return 1e18
        s2=float(np.sum((Yw-Xw@b)**2)/n)
        return 1e18 if s2<=0 else -(-0.5*(n*np.log(s2)+np.sum(np.log(dl*d+1.0))))
    grid=np.logspace(-3,3,30); lls=np.array([nll(dl) for dl in grid]); ib=int(np.argmin(lls))
    try:
        return max(float(minimize_scalar(nll,bounds=(grid[max(0,ib-1)],grid[min(29,ib+1)]),
                                         method="bounded").x),1e-6)
    except: return max(float(grid[ib]),1e-6)

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_bin=False):
    valid = np.isfinite(pheno)
    if valid.sum() < 10: raise ValueError(f"<10 individus valides")
    if K is None or not isinstance(K, np.ndarray) or K.shape[0] != gt.shape[0]:
        K = np.eye(gt.shape[0])
    Y = pheno[valid]; gv = gt[valid]; Kv = K[np.ix_(valid, valid)]
    X0 = np.ones((len(Y), 1)) if covs is None or covs.size == 0 \
         else np.column_stack([np.ones(len(Y)), np.asarray(covs)[valid]])
    X0 = X0[:, np.std(X0, axis=0) > 1e-10]

    if is_bin:
        X0_final, Yt, U, w = X0, Y, None, None
    elif use_lmm:
        dl = _vc_delta(Y, X0, Kv)
        d, U = np.linalg.eigh(Kv); d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(dl * d + 1.0)
        Yt = (U.T @ Y) * w; X0_final = (U.T @ X0) * w[:, None]
    else:
        X0_final, Yt, U, w = X0, Y, None, None

    rows = []
    for j in range(gv.shape[1]):
        g = gv[:, j].astype(float)
        if np.isnan(g).mean() > miss_max: continue
        g = np.where(np.isnan(g), np.nanmean(g), g)
        p = g.mean() / 2.0; mj = min(p, 1 - p)
        if mj < maf_min: continue
        if is_bin:
            X = np.column_stack([X0_final, g])
            beta_snp, se = _logistic_irls(X, Y)
            if se is None or se <= 0: continue
            tv = beta_snp / se
            pval = float(2 * stats.norm.sf(abs(tv)))
            rows.append({"SNP_idx": j, "MAF": mj, "BETA": beta_snp, "SE": se, "T": float(tv), "P": pval})
        else:
            X = np.column_stack([X0_final, (U.T @ g) * w if use_lmm else g])
            try: b, *_ = np.linalg.lstsq(X, Yt, rcond=None)
            except Exception: continue
            res = Yt - X @ b
            s2 = float(np.sum(res ** 2) / max(len(Y) - X.shape[1], 1))
            if s2 <= 0: continue
            try: se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
            except Exception: continue
            if se <= 0: continue
            tv = b[-1] / se
            pval = float(2 * t_dist.sf(abs(tv), df=max(len(Y) - X.shape[1], 1)))
            rows.append({"SNP_idx": j, "MAF": mj, "BETA": b[-1], "SE": se, "T": float(tv), "P": pval})
    return pd.DataFrame(rows)

def annotate_gwas(r, snp_df):
    if r.empty: return r
    idx = r["SNP_idx"].astype(int).values; r = r.copy()
    for c in ["SNP", "CHR", "BP", "CM"]:
        dt = str if c in ("SNP","CHR") else float if c=="CM" else int
        r[c] = snp_df[c].astype(dt).values[idx]
    return r[["SNP_idx","SNP","CHR","BP","CM","MAF","BETA","SE","T","P"]]

def lambda_gc(pvals):
    p = np.asarray(pvals); p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return np.nan
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1))

def postprocess_gwas(r, apply_gc=True):
    if r is None or r.empty: return r
    r = r.copy()
    if HAS_STATSMODELS:
        try:
            _, fdr, _, _ = multipletests(r["P"].values, method="fdr_bh")
            r["FDR"] = fdr
        except Exception: r["FDR"] = np.nan
    else: r["FDR"] = np.nan
    r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0:
            r["P_GC"] = stats.chi2.sf(stats.chi2.isf(r["P"].clip(1e-300,1), df=1) / lam, df=1)
            r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

def fine_mapping(r):
    if r.empty: return r
    df = r.copy().sort_values("P"); v = 0.04
    df["ABF"] = np.exp(0.5*(df["BETA"]**2/(df["SE"]**2+v)) - 0.5*np.log(1+v/df["SE"]**2))
    df["PIP"] = df["ABF"]/df["ABF"].sum()
    return df.sort_values("PIP", ascending=False)

def run_gblup(gt, pheno, kf=5):
    valid = np.isfinite(pheno)
    if valid.sum() < 20: raise ValueError("<20 phénotypes")
    kf = min(kf, len(pheno) - 1)
    Y = pheno[valid]; X = impute_mean(gt[valid]); p = X.mean(0)/2.0; Z = X - 2*p
    G = (Z@Z.T)/(2*np.sum(p*(1-p))) + np.eye(len(Y))*1e-5
    I = np.eye(len(Y)); bh, bl = 0.5, -np.inf
    for h2 in np.linspace(0.01, 0.99, 20):
        lam = (1-h2)/h2
        try:
            ll = -0.5*(np.linalg.slogdet(G+lam*I)[1] + len(Y)*np.log(Y.T@np.linalg.inv(G+lam*I)@Y))
            if ll > bl: bl, bh = ll, h2
        except Exception: pass
    lam = (1-bh)/bh; gebv = G@np.linalg.inv(G+lam*I)@Y
    idx = np.arange(len(Y)); np.random.shuffle(idx); folds = np.array_split(idx, kf)
    yp = np.zeros_like(Y)
    for te in folds:
        tr = np.setdiff1d(idx, te)
        Gtt = G[np.ix_(tr, tr)]; Gvt = G[np.ix_(te, tr)]
        yp[te] = Gvt@np.linalg.inv(Gtt+lam*np.eye(len(tr)))@Y[tr]
    corr = np.corrcoef(Y, yp)[0, 1] if len(Y) > 2 else 0.0
    return pd.DataFrame({"IID": np.where(valid)[0], "Phenotype": Y, "GEBV": gebv}), \
           {"h2": bh, "corr": corr, "r2": corr**2, "rmse": float(np.sqrt(np.mean((Y-yp)**2)))}

# ============================================================
# MÉTA-ANALYSE
# ============================================================
def combine_pvalues_fisher(res_race, res_group):
    if res_race is None or res_group is None: return None
    m = res_race[["SNP","CHR","BP","P"]].rename(columns={"P":"P_race"}).merge(
        res_group[["SNP","P"]].rename(columns={"P":"P_group"}), on="SNP", how="inner")
    if m.empty: return None
    comb = []
    for row in m[["P_race","P_group"]].values:
        try: _, p = stats.combine_pvalues(row, method="fisher")
        except: p = np.nan
        comb.append(p)
    m["P_combined"] = comb
    m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

def combine_stouffer(res_race, res_group, n_race, n_group):
    if res_race is None or res_group is None: return None
    m = res_race[["SNP","CHR","BP","P"]].rename(columns={"P":"P_race"}).merge(
        res_group[["SNP","P"]].rename(columns={"P":"P_group"}), on="SNP", how="inner")
    if m.empty: return None
    w_r = np.sqrt(n_race); w_g = np.sqrt(n_group)
    z_r = stats.norm.isf(m["P_race"].values / 2)
    z_g = stats.norm.isf(m["P_group"].values / 2)
    z_comb = (w_r*z_r + w_g*z_g) / np.sqrt(w_r**2 + w_g**2)
    m["P_combined"] = 2 * stats.norm.sf(np.abs(z_comb))
    return m.sort_values("P_combined").reset_index(drop=True)

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42, max_snp=5000):
    rng = np.random.default_rng(seed)
    if snp_df.shape[1] > max_snp:
        idx = np.sort(rng.choice(snp_df.shape[1], max_snp, replace=False))
        snp_df = snp_df.iloc[idx].reset_index(drop=True); gt = gt[:, idx]
    original = gwas_pairwise(gt, snp_df, labels, ra, rb)
    if original.empty: return pd.DataFrame(), 1.0, 0, []
    n_obs = int((original["P"] < 0.05/len(original)).sum())
    mask = np.isin(labels, [ra, rb])
    labels_sub = labels[mask].copy(); gt_sub = gt[mask]; perm_counts = []
    for _ in range(n_perm):
        rng.shuffle(labels_sub)
        r = gwas_pairwise(gt_sub, snp_df, labels_sub, ra, rb)
        perm_counts.append(int((r["P"] < 0.05/len(r)).sum()) if not r.empty else 0)
    p_emp = (np.sum(np.array(perm_counts) >= n_obs) + 1) / (n_perm + 1)
    return original, float(p_emp), n_obs, perm_counts

def gwas_threshold_permutation(gt, snp_df, labels, ra, rb, n_perm=200, seed=42):
    rng = np.random.default_rng(seed)
    mask = np.isin(labels, [ra, rb])
    labels_sub = labels[mask].copy(); gt_sub = gt[mask]
    obs = gwas_pairwise(gt_sub, snp_df, labels_sub, ra, rb)
    if obs.empty: return np.nan, np.nan
    min_perm = []
    for _ in range(n_perm):
        rng.shuffle(labels_sub)
        r = gwas_pairwise(gt_sub, snp_df, labels_sub, ra, rb)
        if not r.empty: min_perm.append(r["P"].min())
    return float(np.percentile(min_perm, 5)), float(obs["P"].min())

# ============================================================
# VEP / NCBI / 3D / ALIGN / LIFTOVER
# ============================================================
_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if "missense" in c: return "🔴 Exon (Faux-sens - Protéine modifiée)"
    elif "synonymous" in c: return "🟡 Exon (Silencieuse - Protéine inchangée)"
    elif "stop_gained" in c or "frameshift" in c: return "⛔ Exon (Non-sens - Protéine tronquée)"
    elif "splice" in c: return "⚠️ Jonction Exon/Intron"
    elif "intron" in c: return "🟢 Intron (Régulation)"
    elif "3_prime_utr" in c: return "🔵 Région 3'UTR"
    elif "5_prime_utr" in c: return "🔵 Région 5'UTR"
    elif "upstream" in c: return "🟣 Régulateur (Promoteur amont)"
    elif "downstream" in c: return "🟣 Régulateur (Aval)"
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
                        cons = sorted(item["transcript_consequences"],
                                      key=lambda x: _SEVERITY.get(x.get("impact","MODIFIER"), 0),
                                      reverse=True)[0]
                        results.append({"SNP": sid, "GENE": cons.get("gene_symbol","N/A"),
                            "CONSEQUENCE": ", ".join(cons.get("consequence_terms",["N/A"])),
                            "IMPACT": cons.get("impact","N/A"),
                            "AMINO_ACID_CHANGE": cons.get("amino_acids","N/A"),
                            "SIFT": cons.get("sift_prediction","N/A"),
                            "POLYPHEN": cons.get("polyphen_prediction","N/A")})
                    elif item.get("regulatory_feature_consequences"):
                        cons = item["regulatory_feature_consequences"][0]
                        results.append({"SNP": sid, "GENE": "Région régulatrice",
                            "CONSEQUENCE": cons.get("consequence_terms",["N/A"])[0],
                            "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A",
                            "SIFT": "N/A", "POLYPHEN": "N/A"})
                    else:
                        results.append({"SNP": sid, "GENE": "Intergenic/Non-codant",
                            "CONSEQUENCE": "N/A", "IMPACT": "MODIFIER",
                            "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
            else:
                st.warning(f"⚠️ VEP API batch {i//batch+1}: code {r.status_code}")
        except Exception as e:
            st.warning(f"⚠️ VEP réseau batch {i//batch+1}: {e}")
        time.sleep(1.0)
    return pd.DataFrame(results)

def fetch_ncbi_gene_info(gene_list, taxon="9838"):
    if not gene_list: return pd.DataFrame()
    results = []
    for gene in gene_list:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/{quote(str(gene))}/taxon/{taxon}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json(); reports = data.get("reports") or [{}]
                gi = reports[0].get("gene", {}) if reports else {}
                results.append({"GENE": gene,
                    "Nom complet": gi.get("description","N/A"),
                    "Chromosome": gi.get("chromosomes",["N/A"])[0],
                    "Résumé": (gi.get("summary","N/A")[:200]+"...") if gi.get("summary") else "N/A"})
            else:
                results.append({"GENE": gene, "Nom complet": "Non trouvé",
                                "Chromosome": "N/A", "Résumé": "N/A"})
        except Exception as e:
            results.append({"GENE": gene, "Nom complet": "Erreur API",
                            "Chromosome": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
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
    style_json = json.dumps({style: {'color': color_scheme}})
    pdb_json = json.dumps(pdb_data)
    hl_js = ""
    if highlight_res and str(highlight_res).strip().isdigit():
        hl_js = f"viewer.addStyle({{resi: '{highlight_res}'}}, {{stick: {{colorscheme: 'redCarbon'}}}});"
    html_code = f"""
    <div id="viewport" style="width: 100%; height: 500px; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let element = document.getElementById("viewport");
        let config = {{ backgroundColor: 'white' }};
        let viewer = $3Dmol.createViewer(element, config);
        let pdbData = {pdb_json};
        viewer.addModel(pdbData, "pdb");
        viewer.setStyle({{}}, {style_json});
        {hl_js}
        viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html_code, height=520)

def fetch_ensembl_sequence(chrom, start, end, species="camelus_dromedarius"):
    url = f"https://rest.ensembl.org/sequence/region/{species}/{chrom}:{start}..{end}:1?content-type=text/plain"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200: return r.text.upper()
    except Exception: pass
    return None

def align_and_highlight(seq_ref, seq_fast, seq_slow):
    if not HAS_BIO: return "<p>⚠️ Biopython non installé.</p>"
    aligner = Align.PairwiseAligner(); aligner.mode = 'global'
    af = aligner.align(seq_ref, seq_fast)[0]; asl = aligner.align(seq_ref, seq_slow)[0]
    html = "<div style='font-family: monospace; font-size: 16px; line-height: 1.5;'>"
    def fmt(aln, name, color):
        ref_aln, seq_aln = aln[0], aln[1]
        s = f"<b>{name:10}</b> | "
        for r, q in zip(ref_aln, seq_aln):
            if r == '-' or q == '-': s += f"<span style='color: orange; font-weight: bold;'>{q}</span>"
            elif r != q: s += f"<span style='color: red; font-weight: bold;'>{q}</span>"
            else: s += f"<span style='color: {color};'>{q}</span>"
        return s + "<br>"
    html += fmt(af, "Rapide", "green")
    html += f"<b>{'Référence':10}</b> | " + "".join([f"<span style='color: grey;'>{r}</span>" for r in af[0]]) + "<br>"
    html += fmt(asl, "Lente", "blue")
    return html + "</div>"

def lift_over(chrom, bp, species_to="human"):
    url = f"https://rest.ensembl.org/map/{species_to}/GRCh38/{chrom}:{bp}-{bp}:1?content-type=application/json"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            m = r.json().get("mappings", [])
            if m:
                mp = m[0].get("mapped", {})
                return mp.get("seq_region_name"), mp.get("start")
    except Exception: pass
    return None, None

# ============================================================
# ANNOTATION
# ============================================================
def annotate_genes(snp_df, gff, wk=50):
    w = wk*1000; res = []
    for _, s in snp_df.iterrows():
        pos = s.get("BP", s.get("START", 0)); ch = str(s["CHR"])
        msk = (gff["CHR"]==ch) & (gff["START"]<=pos+w) & (gff["END"]>=pos-w)
        ng = gff[msk]
        if not ng.empty:
            for _, g in ng.iterrows():
                res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch, "SNP_POS": pos,
                            "GENE": g["GENE"], "GENE_START": g["START"], "GENE_END": g["END"],
                            "DIST_KB": round(abs(g["START"]-pos)/1000.0, 2)})
    return pd.DataFrame(res)

def _snp_cm(snp_df_full, snp_id):
    try:
        row = snp_df_full[snp_df_full["SNP"].astype(str) == str(snp_id)]
        if row.empty: return None
        cm = row["CM"].iloc[0]
        return float(cm) if pd.notna(cm) else None
    except Exception: return None

def _cm_window_to_bp(snp_df_full, chrom, center_cm, window_cm):
    sub = snp_df_full[snp_df_full["CHR"].astype(str) == str(chrom)].copy()
    if sub.empty: return None
    cm_vals = sub["CM"].values.astype(float)
    if np.all(cm_vals == 0): return None
    sub = sub.sort_values("CM").reset_index(drop=True)
    cm_s = sub["CM"].values; bp_s = sub["BP"].values
    lo = max(cm_s.min(), center_cm - window_cm); hi = min(cm_s.max(), center_cm + window_cm)
    bp_lo = float(np.interp(lo, cm_s, bp_s)); bp_hi = float(np.interp(hi, cm_s, bp_s))
    return int(min(bp_lo, bp_hi)), int(max(bp_lo, bp_hi))

def _bp_to_cm(snp_df_full, chrom, bp):
    sub = snp_df_full[snp_df_full["CHR"].astype(str) == str(chrom)].copy()
    if sub.empty: return None
    cm_vals = sub["CM"].values.astype(float)
    if np.all(cm_vals == 0): return None
    sub = sub.sort_values("BP").reset_index(drop=True)
    return float(np.interp(bp, sub["BP"].values, sub["CM"].values))

def annotate_genes_cm(snp_target, snp_df_full, gff, window_cm=10, window_kb_fallback=100):
    results = []; has_cm = not np.all(snp_df_full["CM"].astype(float).values == 0)
    for _, s in snp_target.iterrows():
        chrom = str(s["CHR"]); snp_id = s.get("SNP", ""); bp_pos = int(s.get("BP", 0))
        cm_pos = s.get("CM", None)
        if cm_pos is None or pd.isna(cm_pos): cm_pos = _snp_cm(snp_df_full, snp_id)
        bp_range = None; method = ""
        if has_cm and cm_pos is not None and not pd.isna(cm_pos):
            bp_range = _cm_window_to_bp(snp_df_full, chrom, float(cm_pos), window_cm)
            if bp_range is not None: method = f"{window_cm} cM"
        if bp_range is None:
            w = window_kb_fallback * 1000
            bp_range = (bp_pos - w, bp_pos + w); method = f"±{window_kb_fallback} kb"
        msk = ((gff["CHR"].astype(str) == chrom) & (gff["END"] >= bp_range[0]) & (gff["START"] <= bp_range[1]))
        ng = gff[msk]
        for _, g in ng.iterrows():
            dist_bp = min(abs(g["START"]-bp_pos), abs(g["END"]-bp_pos)); dist_kb = dist_bp/1000.0
            dist_cm = None
            if method.endswith("cM"):
                cm_snp = _bp_to_cm(snp_df_full, chrom, bp_pos)
                cm_gene = _bp_to_cm(snp_df_full, chrom, (g["START"]+g["END"])/2)
                if cm_snp is not None and cm_gene is not None: dist_cm = abs(cm_gene - cm_snp)
            results.append({"SNP": snp_id, "CHR": chrom, "SNP_POS_BP": bp_pos,
                "SNP_POS_CM": round(float(cm_pos), 2) if cm_pos else None,
                "GENE": g["GENE"], "GENE_START": int(g["START"]), "GENE_END": int(g["END"]),
                "DIST_KB": round(dist_kb, 2),
                "DIST_CM": round(dist_cm, 2) if dist_cm is not None else None,
                "WINDOW": method})
    df = pd.DataFrame(results)
    if not df.empty:
        sort_col = "DIST_CM" if has_cm else "DIST_KB"
        df = df.sort_values(["SNP", sort_col]).reset_index(drop=True)
    return df

# ============================================================
# ENRICHISSEMENT / STRING
# ============================================================
def run_enrichment(gene_list, gene_sets=None, organism="Human", outdir=None):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3: return pd.DataFrame()
        if gene_sets is None: gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism,
                         outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None: return pd.DataFrame()
        df = enr.results.copy()
        for old, new in {"Term":"TERM","Adjusted P-value":"FDR","P-value":"PVAL",
                          "Overlap":"OVERLAP","Genes":"GENES","Combined Score":"SCORE"}.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True)
    except Exception as e:
        st.error(f"❌ Enrichr : {e}")
        return pd.DataFrame()

def query_string_network(genes, species=9838, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([str(g).strip() for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(data)
        edges = df[["preferredName_A","preferredName_B","score"]].copy()
        edges.columns = ["SOURCE","TARGET","SCORE"]
        nodes = pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
        return edges, nodes
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def plot_string_network(edges, nodes):
    if edges is None or edges.empty: return None
    try: import networkx as nx
    except ImportError:
        st.error("⚠️ `networkx` non installé."); return None
    G = nx.Graph()
    for _, r in edges.iterrows(): G.add_edge(r["SOURCE"], r["TARGET"], weight=float(r["SCORE"]))
    pos = nx.spring_layout(G, seed=42, k=0.6)
    ex, ey = [], []
    for s, t in G.edges():
        x0, y0 = pos[s]; x1, y1 = pos[t]; ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(width=0.6, color="#cccccc"),
                             hoverinfo="skip", showlegend=False))
    nx_, ny_ = [], []
    for n in G.nodes(): x, y = pos[n]; nx_.append(x); ny_.append(y)
    deg = [d for _, d in G.degree()]
    fig.add_trace(go.Scatter(x=nx_, y=ny_, mode="markers+text",
                             marker=dict(size=[8+3*d for d in deg], color=deg,
                                         colorscale="Viridis", line=dict(width=1, color="white")),
                             text=list(G.nodes()), textposition="top center", hoverinfo="text"))
    fig.update_layout(showlegend=False, height=600,
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False))
    return fig

def plot_tree(Z, labels, support=None, title="", mid=None):
    from scipy.cluster.hierarchy import leaves_list
    n = len(labels); labels = [str(l) for l in labels]
    lo = leaves_list(Z); ly = {int(l): i for i, l in enumerate(lo)}
    cy = {i: ly[i] for i in range(n)}; cx = {i: 0.0 for i in range(n)}; fig = go.Figure()
    for k, row in enumerate(Z):
        i, j, d = int(row[0]), int(row[1]), float(row[2])
        yi, yj = cy[i], cy[j]; xi, xj = cx[i], cx[j]
        if support is not None and k < len(support):
            s = support[k]; col = "#2ecc71" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
        else: col = "#3498db"
        fig.add_trace(go.Scatter(x=[xi,d], y=[yi,yi], mode='lines', line=dict(color=col,width=2), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[xj,d], y=[yj,yj], mode='lines', line=dict(color=col,width=2), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[d,d], y=[yi,yj], mode='lines', line=dict(color=col,width=2), hoverinfo='skip', showlegend=False))
        ni = n+k; cy[ni] = (yi+yj)/2; cx[ni] = d
        if support is not None and k < len(support):
            s = support[k]
            fig.add_annotation(x=d, y=(yi+yj)/2, text=f"<b>{int(s)}</b>", showarrow=False,
                               font=dict(size=10, color=col), yshift=8, bgcolor="white", borderpad=2)
    for leaf, y in ly.items():
        fig.add_annotation(x=0, y=y, text=f"<b>{labels[leaf]}</b>", showarrow=False,
                           xanchor='right', xshift=-8, font=dict(size=12, color="#2c3e50"))
    if mid: fig.add_vline(x=mid["midpoint"], line_dash="dot", line_color="purple")
    fig.update_layout(title=title, xaxis_title="Distance de Reynolds",
                      yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                      height=max(500, n*35), margin=dict(l=120,r=40,t=80,b=40), plot_bgcolor='white')
    return fig

# ============================================================
# EXPORT
# ============================================================
def build_plink(gt, ind_df, snp_df, prefix="camel"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        n, m = gt.shape; nb = (n+3)//4; buf = bytearray([0x6C,0x1B,0x01])
        for j in range(m):
            bits = np.zeros(n, dtype=np.uint8); bits[np.isnan(gt[:,j])] = 0b01
            bits[gt[:,j]==0] = 0b11; bits[gt[:,j]==1] = 0b10; bits[gt[:,j]==2] = 0b00
            pk = np.zeros(nb, dtype=np.uint8)
            for k in range(n): pk[k//4] |= (bits[k] & 0x03) << (2*(k%4))
            buf.extend(pk.tobytes())
        zf.writestr(f"{prefix}.bed", bytes(buf))
        zf.writestr(f"{prefix}.bim", "\n".join(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t{int(r['BP'])}\tA\tG" for _, r in snp_df.iterrows())+"\n")
        zf.writestr(f"{prefix}.fam", "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _, r in ind_df.iterrows())+"\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Camel"):
    h = ["##fileformat=VCFv4.2", f"##source=Camel-{proj}",
         '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
         "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ind_df["IID"].astype(str))]
    b = []
    for j in range(gt.shape[1]):
        r = snp_df.iloc[j]
        gts = ["./." if np.isnan(gt[i,j]) else "0/0" if gt[i,j]==0 else "0/1" if gt[i,j]==1 else "1/1"
               for i in range(gt.shape[0])]
        b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(h + b) + "\n"

# ============================================================
# VISUALISATIONS
# ============================================================
def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind","Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False); return f
def plot_pca(s, v, l):
    f = px.scatter(pd.DataFrame({"PC1":s[:,0],"PC2":s[:,1],"Pop":l}),
                   x="PC1", y="PC2", color="Pop",
                   title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550)
    f.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return f
def plot_mds(m, l):
    f = px.scatter(pd.DataFrame({"MDS1":m[:,0],"MDS2":m[:,1],"Pop":list(l)}),
                   x="MDS1", y="MDS2", color="Pop", title="MDS Reynolds", height=520)
    f.update_traces(marker=dict(size=11, line=dict(width=1, color="white"))); return f
def plot_manhattan(df, yc="P", title="", thr=None):
    if yc == "P":
        df = df.copy(); df["_y"] = -np.log10(np.clip(df["P"], 1e-300, 1)); yc = "_y"
    df = df[np.isfinite(df[yc])].copy()
    df = df.assign(_k=df["CHR"].map(_chk)).sort_values(["_k","BP"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index, y=sub[yc], mode="markers",
                                 marker=dict(size=5, color="#2c3e50" if i%2==0 else "#7f8c8d"),
                                 text=sub["SNP"], hoverinfo="text", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Chr", yaxis_title=yc, height=500)
    return fig
def plot_ld(ld, b=20):
    if ld is None or ld.empty: return None
    d = ld.copy(); d["bin"] = (d["dist_kb"]//b)*b
    return px.line(d.groupby("bin")["r2"].mean().reset_index(), x="bin", y="r2", title="LD decay", height=450)
def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)]); df["IID"] = labs; df["Pop"] = pops
    df["_d"] = Q.argmax(1); df["_po"] = pd.Categorical(pops, categories=sorted(set(pops)))
    df = df.sort_values(["_po","_d"]).reset_index(drop=True); df["x"] = np.arange(len(df))
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k+1}"], name=f"K{k+1}",
                             marker_color=px.colors.qualitative.Set2[k%8]))
    fig.update_layout(barmode="stack", title=f"Admixture K={K}", height=500)
    return fig, df
def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP")
    fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1); p = np.poly1d(z)
            xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=p(xr), mode="lines", name="OLS",
                                     line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig
def plot_pip(df):
    return px.bar(df.head(20), x="SNP", y="PIP", title="PIP (Posterior Inclusion Probability)",
                  color="PIP", color_continuous_scale="Viridis")
def plot_win_manh(df, vc='MEAN_FST', title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df = df.assign(_k=df['CHR'].map(_chk)).sort_values(['_k','MID']).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby('CHR')):
        fig.add_trace(go.Scatter(x=sub['MID']/1e6, y=sub[vc], mode='markers',
            marker=dict(size=7, color='#2c3e50' if i%2==0 else '#7f8c8d',
                        line=dict(width=0.5, color='white')),
            text=[f"{c}:{int(m/1e6)}Mb<br>{vc}={v:.4f}" for c, m, v in zip(sub['CHR'], sub['MID'], sub[vc])],
            hoverinfo='text', showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std()
    fig.add_hline(y=mu+sigma*sd, line_dash='dash', line_color='red',
                  annotation_text=f'μ+{sigma}σ')
    fig.update_layout(title=title, xaxis_title='Pos (Mb)', yaxis_title=vc, height=520)
    return fig
def plot_gene_track(si, gff, wk=50):
    if gff is None or gff.empty: return None
    ch = si["CHR"]; pos = si.get("BP", si.get("START", 0)); w = wk*1000
    rg = gff[(gff["CHR"]==ch) & (gff["START"]<=pos+w) & (gff["END"]>=pos-w)]
    fig = go.Figure()
    for i, (_, g) in enumerate(rg.iterrows()):
        fig.add_trace(go.Scatter(x=[g["START"]/1e6, g["END"]/1e6], y=[i,i], mode="lines",
                                 line=dict(width=10, color="royalblue"), name=g["GENE"],
                                 text=[g["GENE"], g["GENE"]], hoverinfo="text"))
    fig.add_vline(x=pos/1e6, line_dash="dash", line_color="red",
                  annotation_text=f"SNP: {si.get('SNP','')}")
    fig.update_layout(title=f"Gènes CHR{ch}", xaxis_title="Pos (Mb)",
                      yaxis=dict(showticklabels=False), height=400+len(rg)*20)
    return fig

# ============================================================
# STREAMLIT
# ============================================================
_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","gt_pre","ind_pre","snp_pre",
         "gt_pruned","snp_pruned","qc_stats","pca_s","pca_v","mds_c","king_K","fst","fst_M",
         "fst_pops","admix_Q","admix_K","cv_res","ld_df","roh_df","froh","ne_dict","reyn_D",
         "reyn_pops","pheno_df","gwas_res","gwas_pheno","lambda_gc","gebv_df","gblup_stats",
         "ann_df","_plink","_vcf","mapping_df","fst_snp_df","win_fst","taj_df","taj_race",
         "css_df","pair_res","pair_res_race","pair_res_group","pair_name_race","pair_name_group",
         "nj_result","nj_mid","sel_gff","sel_ann","sel_ann_race","sel_ann_group","scan_df",
         "vep_res","ncbi_gene_res","lift_res"]

def reset_derived():
    for k in _KEYS:
        if k not in ("gt","ind_df","snp_df"): st.session_state[k] = None
def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k] = None

def main():
    init_state()
    st.title("🐪 Camel Genomics Pipeline v22.0")
    st.caption("Consolidé : FST WC84 · GWAS logistique IRLS · FDR · UPGMA · VEP · 3D · ML · Haplotypes · LiftOver")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source", ["Demo","PED/MAP","VCF","Axiom→PED"], key="src")
        if mode == "Demo":
            if st.button("🎲 Générer démo", use_container_width=True):
                gt, i, s = gen_demo(); reset_derived()
                st.session_state.update({"gt":gt, "ind_df":rec_fid(i), "snp_df":s}); st.success("✅")
        elif mode == "PED/MAP":
            ped = st.file_uploader(".ped", type=["ped","gz"]); mp = st.file_uploader(".map", type=["map","gz"])
            if ped and mp and st.button("📥 Charger"):
                try:
                    mdf = parse_map(mp.read()); gt, ind, a1, a2 = parse_ped(ped.read(), len(mdf))
                    if len(mdf) > gt.shape[1]: mdf = mdf.iloc[:gt.shape[1]].reset_index(drop=True)
                    elif len(mdf) < gt.shape[1]: gt = gt[:, :len(mdf)]
                    mdf["A1"] = a1; mdf["A2"] = a2
                    reset_derived(); st.session_state.update({"gt":gt, "ind_df":rec_fid(ind), "snp_df":mdf})
                    st.success(f"✅ {gt.shape[0]}×{gt.shape[1]}")
                except Exception as e: st.error(f"❌ {e}")
        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf","gz"])
            if vf and st.button("📥 Charger"):
                try:
                    gt, ind, snp = parse_vcf(vf.read()); reset_derived()
                    st.session_state.update({"gt":gt, "ind_df":rec_fid(ind), "snp_df":snp}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
        else:
            af = st.file_uploader("Axiom brut", type=["txt","csv","tsv","gz"])
            if af and st.button("🔄 Convertir"):
                try:
                    s, g, ns = parse_axiom(af.read(), af.name)
                    mpl = [f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(ns)]
                    pdl = ["\t".join([sample_fid_iid(smp)[0], sample_fid_iid(smp)[1], "0","0","0","-9"] +
                            [a for pr in row for a in pr]) for smp, row in zip(s, g)]
                    st.session_state["_plink_ped"] = "\n".join(pdl)+"\n"
                    st.session_state["_plink_map"] = "\n".join(mpl)+"\n"
                    st.success(f"✅ {len(s)}×{ns}")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("_plink_ped"):
                st.download_button("⬇ .ped", st.session_state["_plink_ped"].encode(), "out.ped")
                st.download_button("⬇ .map", st.session_state["_plink_map"].encode(), "out.map")
        st.divider(); st.header("⚙️ Seuils QC")
        keep_auto = st.checkbox("🧬 Exclure Chromosomes Sexuels (X, Y, MT)", value=True)
        rem_out = st.checkbox("🚫 Supprimer outliers", value=True)
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.05)
        mind = st.slider("Missing Ind", 0.0, 0.5, 0.05)
        maf_t = st.slider("MAF", 0.0, 0.5, 0.05)
        hwe_t = st.number_input("HWE", value=1e-6, format="%.0e")
        het_sd = st.slider("Het ±σ", 1.0, 5.0, 3.0, 0.1)
        ld_r2 = st.slider("LD r²", 0.05, 0.5, 0.2, 0.05)

    if st.session_state.gt is None:
        st.info("👉 Chargez des données (Demo recommandé pour tester)."); return

    tabs = st.tabs([
        "🧹 QC", "✂️ LD", "🧬 Structure", "🎨 Admixture", "📈 Démographie",
        "🔍 Sélection", "🧪 GWAS Classique", "🎯 Post-GWAS", "📈 GBLUP",
        "🧬 Annotation", "🧬 VEP & NCBI", "🧬 Structure 3D", "🧬 Alignement",
        "📤 Export", "🔬 Méta-analyse", "🧠 ML", "🧬 Haplotypes", "🌍 LiftOver"
    ])

    # ═══════ QC ═══════
    with tabs[0]:
        if st.button("▶ Lancer QC", key="qc_btn"):
            try:
                p = {"geno":geno, "mind":mind, "maf":maf_t, "hwe":hwe_t, "het_sd":het_sd}
                gf, inf, snf, gp, ip, sp, qs = apply_qc(
                    st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, p, keep_autosomes=keep_auto, rem_out=rem_out)
                st.session_state.update({"gt_filt":gf, "ind_filt":inf, "snp_filt":snf,
                    "gt_pre":gp, "ind_pre":ip, "snp_pre":sp, "qc_stats":qs})
                st.success(f"✅ {gf.shape[0]}×{gf.shape[1]}")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus", s["n_ind_final"], delta=-s["excluded_ind"])
            c2.metric("SNPs", s["n_snp_final"], delta=-s["excluded_snp"])
            c3.metric("Pré-outliers", f"{st.session_state.gt_pre.shape[1]}" if st.session_state.gt_pre is not None else "N/A")
            c4.metric("Outliers", s["trace"].get("het", 0))
            st.plotly_chart(plot_missing(miss_ind(st.session_state.gt), miss_snp(st.session_state.gt)),
                            use_container_width=True, key="plot_missing_qc")

    # ═══════ LD ═══════
    with tabs[1]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning", key="ld_btn"):
            k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
            st.session_state.update({"gt_pruned":st.session_state.gt_filt[:,k],
                                     "snp_pruned":st.session_state.snp_filt[k].reset_index(drop=True)})
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs")
        if st.session_state.gt_pruned is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avant", st.session_state.gt_filt.shape[1])
            c2.metric("Après", st.session_state.gt_pruned.shape[1])
            c3.metric("Réduction", f"{100*(1-st.session_state.gt_pruned.shape[1]/st.session_state.gt_filt.shape[1]):.1f}%")
            if st.button("▶ LD Decay", key="ld_decay_btn"):
                st.session_state.ld_df = ld_decay(st.session_state.gt_pruned, st.session_state.snp_pruned)
            if st.session_state.ld_df is not None:
                f = plot_ld(st.session_state.ld_df)
                if f: st.plotly_chart(f, use_container_width=True, key="plot_ld_decay")

    # ═══════ Structure ═══════
    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS + KING", key="struct_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                s_, v_ = pca_analysis(gu, 10); st.session_state.update({"pca_s":s_, "pca_v":v_})
                st.session_state.mds_c = mds_analysis(gu, 5); st.session_state.king_K = king_mat(gu)
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v,
                                     st.session_state.ind_filt["FID"].values),
                            use_container_width=True, key="plot_pca_tab")
            st.plotly_chart(plot_mds(st.session_state.mds_c, st.session_state.ind_filt["FID"].values),
                            use_container_width=True, key="plot_mds_tab")

    # ═══════ Admixture ═══════
    with tabs[3]:
        if st.session_state.gt_filt is not None:
            st.subheader("📊 Choix de K par courbe d'erreur NMF")
            c1, c2 = st.columns(2)
            kmin = c1.slider("K min", 2, 5, 2, key="kmin_sl")
            kmax = c2.slider("K max", 4, 10, 8, key="kmax_sl")
            if st.button("📈 Évaluer tous les K", key="eval_k_btn"):
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                st.session_state.cv_res = choose_best_k(gu, kmin, kmax)
            if st.session_state.cv_res is not None:
                df = st.session_state.cv_res
                st.dataframe(df, use_container_width=True)
                st.plotly_chart(px.line(df, x="K", y="RMSE",
                                        title="Erreur NMF par K (choisir l'elbow)"),
                                use_container_width=True, key="plot_k_curve")
            st.divider()
            K = st.slider("K choisi", 2, 10, 4, key="admix_K_slider")
            if st.button("▶ Admixture NMF", key="admix_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    Q, _ = admixture_nmf(gu, K=K)
                    st.session_state.update({"admix_Q":Q, "admix_K":K}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.admix_Q is not None:
                f, _ = plot_admix(st.session_state.admix_Q, st.session_state.ind_filt["IID"].values,
                                  st.session_state.ind_filt["FID"].values, st.session_state.admix_K)
                st.plotly_chart(f, use_container_width=True, key="plot_admix_tab")

    # ═══════ Démographie ═══════
    with tabs[4]:
        if st.session_state.gt_filt is not None and st.button("▶ ROH + Ne", key="dem_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                roh, fr = detect_roh(gu, su[["CHR","BP"]].to_json())
                st.session_state.update({"roh_df":roh, "froh":fr})
                ne = {}
                for f in st.session_state.ind_filt["FID"].unique():
                    m = (st.session_state.ind_filt["FID"] == f).values
                    if m.sum() >= 5: ne[str(f)] = ne_historical(gu[m], su)
                st.session_state.ne_dict = ne; st.success(f"✅ {len(roh)} ROH")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.froh is not None:
            df = pd.DataFrame({"FROH": st.session_state.froh, "Pop": st.session_state.ind_filt["FID"].values})
            st.plotly_chart(px.histogram(df, x="FROH", color="Pop", nbins=50, title="FROH"),
                            use_container_width=True, key="plot_roh_hist")

    # ═══════ Sélection ═══════
    with tabs[5]:
        st.subheader("🔍 Sélection & structure des populations")
        use_wc = st.checkbox("Utiliser Weir & Cockerham (1984) — recommandé", value=True, key="use_wc84")
        if st.button("▶ FST pairwise + Reynolds", key="fst_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                lab = st.session_state.ind_filt["FID"].astype(str).values
                st.session_state.fst = (fst_snp_wc(gu, lab) if use_wc else fst_snp(gu, lab))
                st.session_state.fst_snp_df = su.copy()
                M, pops = fst_pairwise(gu, lab, use_wc84=use_wc)
                st.session_state.update({"fst_M":M, "fst_pops":pops})
                D, rp = reynolds(gu, lab); st.session_state.update({"reyn_D":D, "reyn_pops":rp})
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.fst is not None:
            sf = st.session_state.fst_snp_df if st.session_state.get("fst_snp_df") is not None else st.session_state.snp_filt
            nf = min(len(st.session_state.fst), len(sf))
            dff = pd.DataFrame({"SNP":sf["SNP"].values[:nf], "CHR":sf["CHR"].values[:nf],
                                "BP":sf["BP"].values[:nf], "FST":st.session_state.fst[:nf]})
            st.plotly_chart(plot_manhattan(dff, yc="FST", title="FST par SNP"),
                            use_container_width=True, key="plot_manh_fst_snp")

        st.divider(); st.subheader("🌳 Arbre UPGMA (bootstrap)")
        st.caption("⚠️ UPGMA suppose une horloge moléculaire — interpréter avec prudence.")
        if st.session_state.reyn_D is not None:
            c1, c2 = st.columns(2)
            nb = c1.slider("Bootstrap", 10, 500, 100, 10, key="nj_boot")
            ro = c2.selectbox("Root", ["Non-enraciné","Midpoint"], key="nj_root")
            if st.button("🌲 Construire l'arbre", type="primary", key="nj_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    lab = st.session_state.ind_filt["FID"].astype(str).values
                    with st.spinner(f"UPGMA + {nb} boot..."):
                        r = upgma_bootstrap(gu, su, lab, n_boot=int(nb)); st.session_state.nj_result = r
                        st.session_state.nj_mid = midpoint_info(r["D"], r["pops"])
                    st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("nj_result") is not None:
                r = st.session_state.nj_result; mp = st.session_state.get("nj_mid")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Races", len(r["pops"])); c2.metric("Nœuds", len(r["support"]))
                c3.metric("Supp≥70%", f"{(r['support']>=70).sum()}/{len(r['support'])}")
                c4.metric("Dist max", f"{mp['max_dist']:.4f}" if mp else "N/A")
                if mp: st.info(f"Midpoint entre **{mp['leaf_i']}** et **{mp['leaf_j']}**")
                f = plot_tree(r["Z"], r["pops"], r["support"],
                              title=f"UPGMA ({nb} boot)", mid=mp if ro=="Midpoint" else None)
                st.plotly_chart(f, use_container_width=True, key="plot_nj_tree")

        st.divider(); st.subheader("🧬 GWAS case-control")
        if st.session_state.gt_filt is not None:
            gu = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt)
            su = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt)
            iu = st.session_state.ind_filt
            covs_pair = st.session_state.pca_s[:, :5] if st.session_state.pca_s is not None else None
            mode_gwas = st.radio("🎯 Mode", ["🏷️ Race × Race", "📊 Groupe × Groupe", "⚖️ Comparer"],
                                 horizontal=True, key="gwas_mode")
            st.divider()
            if mode_gwas.startswith("🏷️"):
                races, _ = get_pairs(iu["FID"].astype(str).values)
                c1, c2 = st.columns(2)
                ra = c1.selectbox("Race A", races, index=0, key="pa")
                rb = c2.selectbox("Race B", races, index=1 if len(races)>1 else 0, key="pb")
                c1, c2 = st.columns(2)
                mtd = c1.selectbox("Méthode", ["allelic","logistic"], key="pm")
                mf = c2.slider("MAF", 0.0, 0.3, 0.05, key="pmf")
                if ra != rb and st.button(f"🚀 {ra} vs {rb}", type="primary", key="pr_btn"):
                    try:
                        with st.spinner(f"{ra} vs {rb}..."):
                            r = gwas_pairwise(gu, su, iu["FID"].astype(str).values, ra, rb, float(mf), mtd, covs_pair)
                            st.session_state["pair_res_race"] = r
                            st.session_state["pair_name_race"] = f"{ra} vs {rb}"
                        if not r.empty:
                            st.success(f"✅ {len(r)} SNPs · {(r['P']<0.05/len(r)).sum()} Bonf")
                            st.dataframe(r.head(10), use_container_width=True)
                    except Exception as e: st.error(f"❌ {e}")
            elif mode_gwas.startswith("📊"):
                critere = st.selectbox("Critère", list(GROUPES_DEFAULT.keys()) + ["➕ CSV"], key="grp_critere")
                if critere == "➕ CSV":
                    f = st.file_uploader("CSV (RACE, GROUPE)", type=["csv"], key="grp_csv")
                    if f:
                        mdf = pd.read_csv(f)
                        group_map = dict(zip(mdf["RACE"].astype(str).str.upper(), mdf["GROUPE"].astype(str)))
                    else: st.stop()
                else: group_map = make_group_map(GROUPES_DEFAULT[critere])
                iu_g = apply_group_mapping(iu, group_map)
                groups = sorted([g for g in iu_g["FID_GROUPE"].unique() if g != "AUTRE"])
                c1, c2 = st.columns(2)
                ga = c1.selectbox("Groupe A", groups, index=0, key="grp_a")
                gb = c2.selectbox("Groupe B", groups, index=1 if len(groups)>1 else 0, key="grp_b")
                mtd_g = c1.selectbox("Méthode", ["allelic","logistic"], key="pm_grp")
                mf_g = c2.slider("MAF", 0.0, 0.3, 0.05, key="pmf_grp")
                nA = (iu_g["FID_GROUPE"]==ga).sum(); nB = (iu_g["FID_GROUPE"]==gb).sum()
                rA = iu_g[iu_g["FID_GROUPE"]==ga]["FID"].nunique()
                rB = iu_g[iu_g["FID_GROUPE"]==gb]["FID"].nunique()
                st.info(f"📊 **{ga}** ({nA} ind, {rA} races) vs **{gb}** ({nB} ind, {rB} races)")
                if ga != gb and st.button(f"🚀 {ga} vs {gb}", type="primary", key="pr_btn_grp"):
                    try:
                        with st.spinner(f"{ga} vs {gb}..."):
                            r = gwas_pairwise_group(gu, su, iu_g["FID_GROUPE"].values, ga, gb, float(mf_g), mtd_g, covs_pair)
                            st.session_state["pair_res_group"] = r
                            st.session_state["pair_name_group"] = f"{ga} vs {gb} ({critere})"
                        if not r.empty:
                            st.success(f"✅ {len(r)} SNPs · {(r['P']<0.05/len(r)).sum()} Bonf")
                            st.dataframe(r.head(10), use_container_width=True)
                    except Exception as e: st.error(f"❌ {e}")
            else:
                res_r = st.session_state.get("pair_res_race")
                res_g = st.session_state.get("pair_res_group")
                if res_r is None or res_g is None:
                    st.warning("⚠️ Lancez RACE puis GROUPE pour comparer.")
                else:
                    name_r = st.session_state.get("pair_name_race", "Race")
                    name_g = st.session_state.get("pair_name_group", "Groupe")
                    c1, c2 = st.columns(2)
                    c1.metric(f"🏷️ {name_r}", f"{len(res_r)} SNPs",
                              delta=f"{(res_r['P']<0.05/len(res_r)).sum()} Bonf")
                    c2.metric(f"📊 {name_g}", f"{len(res_g)} SNPs",
                              delta=f"{(res_g['P']<0.05/len(res_g)).sum()} Bonf")
                    common = set(res_r["SNP"]).intersection(set(res_g["SNP"]))
                    st.metric("SNPs communs", len(common))
                    if common:
                        dfc = res_r[res_r["SNP"].isin(common)][["SNP","CHR","BP","FST_snp","P"]].rename(
                            columns={"P":"P_race","FST_snp":"FST_race"})
                        dfc = dfc.merge(res_g[res_g["SNP"].isin(common)][["SNP","P","FST_snp"]].rename(
                            columns={"P":"P_group","FST_snp":"FST_group"}), on="SNP")
                        st.dataframe(dfc, use_container_width=True)
                        st.download_button("⬇ SNPs communs", dfc.to_csv(index=False).encode(),
                                           "snps_communs.csv", "text/csv")

        st.divider(); st.subheader("🧬 Signatures de sélection")
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            lab = st.session_state.ind_filt["FID"].astype(str).values
            ms = st.radio("Méthode", ["8.3.1 — FST par fenêtres", "8.3.2 — Tajima's D", "8.3.3 — CSS"],
                          horizontal=True, key="sel_m")
            if ms.startswith("8.3.1"):
                c1, c2, c3 = st.columns(3)
                ws = c1.slider("Fenêtre", 10, 200, 50, key="wf")
                sp = c2.slider("Pas", 5, 100, 25, key="wp")
                zt = c3.slider("Seuil z", 2.0, 5.0, 3.0, 0.1, key="wz")
                if st.button("▶ Scanner FST", type="primary", key="wf_btn"):
                    with st.spinner("Scan FST..."):
                        w = windowed_fst(gu, su, lab, ws, sp, use_wc84=use_wc)
                        w = detect_outliers(w, zt); st.session_state.win_fst = w
                    st.success(f"✅ {len(w)} fenêtres")
                if st.session_state.get("win_fst") is not None:
                    w = st.session_state.win_fst
                    st.plotly_chart(plot_win_manh(w, 'MEAN_FST', f'FST fenêtres'),
                                    use_container_width=True, key="plot_win_fst")
                    st.dataframe(w.head(20)[['CHR','START','END','N_SNP','MEAN_FST','MAX_FST','Z_FST','TOP_SNP']],
                                 use_container_width=True)
            elif ms.startswith("8.3.2"):
                rs = st.selectbox("Race", sorted(np.unique(lab)), key="tr")
                c1, c2 = st.columns(2)
                wt = c1.slider("Fenêtre", 10, 100, 25, key="wt")
                stp = c2.slider("Pas", 5, 50, 10, key="stp")
                if st.button(f"▶ Tajima's D — {rs}", type="primary", key="t_btn"):
                    m = lab == rs
                    if m.sum() < 5: st.error("⚠️ < 5 individus")
                    else:
                        with st.spinner("Tajima..."):
                            t = tajimas_scan(gu[m], su, wt, stp)
                            st.session_state.taj_df = t; st.session_state.taj_race = rs
                        st.success(f"✅ {len(t)} fenêtres")
                if st.session_state.get("taj_df") is not None:
                    t = st.session_state.taj_df
                    st.plotly_chart(plot_win_manh(t, 'TAJIMA_D', f"Tajima D"),
                                    use_container_width=True, key="plot_tajima")
                    st.dataframe(t.sort_values('TAJIMA_D').head(20), use_container_width=True)
            else:
                if st.button("▶ CSS", type="primary", key="css_btn"):
                    with st.spinner("CSS..."):
                        c = composite_score(gu, su, lab); st.session_state.css_df = c
                    st.success(f"✅ {len(c)} SNPs")
                if st.session_state.get("css_df") is not None:
                    c = st.session_state.css_df; cp = c.copy()
                    cp["P_pseudo"] = np.exp(-cp["CSS"])
                    st.plotly_chart(plot_manhattan(cp.rename(columns={"P_pseudo":"P"}),
                                                    yc="P", title="CSS"),
                                    use_container_width=True, key="plot_css_manh")
                    st.dataframe(c.head(20), use_container_width=True)

        st.divider(); st.subheader("🧬 Annotation des gènes (cM ou kb)")
        snp_full = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt)
        has_cm_data = (snp_full is not None and "CM" in snp_full.columns
                       and not np.all(snp_full["CM"].astype(float).values == 0))
        if has_cm_data: st.success("✅ cM valides détectées — annotation en cM disponible")
        else: st.warning("⚠️ cM=0 → fallback kb")
        gff_f = st.file_uploader("GFF3/GTF", type=["gff","gff3","gtf","gz"], key="sel_gff_file")
        sources = []
        if st.session_state.get("win_fst") is not None: sources.append("FST outliers")
        if st.session_state.get("taj_df") is not None: sources.append("Tajima extrêmes")
        if st.session_state.get("css_df") is not None: sources.append("CSS top")
        if st.session_state.get("pair_res_race") is not None: sources.append("Race Pairwise")
        if st.session_state.get("pair_res_group") is not None: sources.append("Group Pairwise")
        sources.append("Manuel")
        if gff_f is not None:
            try:
                with st.spinner("Parsing GFF3..."):
                    gff = parse_gff3(gff_f.read()); st.session_state.sel_gff = gff
                st.success(f"✅ {len(gff)} gènes")
                c1, c2 = st.columns(2)
                metric = c1.radio("Unité", ["cM (génétique)","kb (physique)"],
                                  index=0 if has_cm_data else 1, horizontal=True, key="ann_metric")
                if metric.startswith("cM"):
                    window_cm = c2.slider("Fenêtre cM", 1, 50, 10, 1); wk_fallback = 100
                else:
                    wk_fallback = c2.slider("Fenêtre kb", 1, 500, 50, 5); window_cm = 10
                src = st.selectbox("Source", sources, key="ann_src")
                to_ann = None
                if src == "FST outliers":
                    to_ann = st.session_state.win_fst.head(50)[["CHR","START","TOP_SNP"]].rename(
                        columns={"START":"BP","TOP_SNP":"SNP"})
                elif src == "Tajima extrêmes":
                    ex = st.session_state.taj_df.sort_values("TAJIMA_D").head(20)
                    to_ann = ex[["CHR","MID"]].rename(columns={"MID":"BP"})
                    to_ann["SNP"] = [f"Taj_{i+1}" for i in range(len(to_ann))]
                elif src == "CSS top":
                    to_ann = st.session_state.css_df.head(30)[["SNP","CHR","BP"]]
                elif src in ("Race Pairwise", "Group Pairwise"):
                    key = "pair_res_race" if src=="Race Pairwise" else "pair_res_group"
                    to_ann = st.session_state[key].head(30)[["SNP","CHR","BP"]]
                else:
                    c1, c2 = st.columns(2)
                    chm = c1.text_input("Chr", "14", key="ann_mc")
                    pm = c2.number_input("Pos", 0, 200_000_000, 25_000_000, 100_000, key="ann_mp")
                    to_ann = pd.DataFrame({"SNP":[f"m_{chm}_{pm}"], "CHR":[str(chm)], "BP":[int(pm)]})
                if to_ann is not None and len(to_ann) > 0 and st.button("🔍 Annoter", type="primary", key="ann_btn"):
                    with st.spinner("Annotation..."):
                        if metric.startswith("cM"):
                            ann = annotate_genes_cm(to_ann, snp_full, gff,
                                                    window_cm=window_cm, window_kb_fallback=wk_fallback)
                        else:
                            ann = annotate_genes(to_ann, gff, wk=wk_fallback)
                            if not ann.empty:
                                ann["WINDOW"] = f"±{wk_fallback} kb"; ann["DIST_CM"] = None
                        st.session_state.sel_ann = ann
                        if src == "Race Pairwise": st.session_state.sel_ann_race = ann
                        elif src == "Group Pairwise": st.session_state.sel_ann_group = ann
                    if ann.empty: st.warning("Aucun gène")
                    else: st.success(f"✅ {len(ann)} associations")
                if st.session_state.get("sel_ann") is not None and len(st.session_state.sel_ann) > 0:
                    ann = st.session_state.sel_ann
                    st.dataframe(ann, use_container_width=True)
                    st.download_button("⬇ CSV", ann.to_csv(index=False).encode(),
                                       "sel_ann.csv", "text/csv")
                    sc = ann["SNP"].unique().tolist()
                    ss = st.selectbox("Visualiser", sc, key="ann_track")
                    row = ann[ann["SNP"]==ss].iloc[0].to_dict()
                    si = {"SNP":row["SNP"], "CHR":row["CHR"],
                          "BP":row.get("SNP_POS_BP", row.get("SNP_POS", 0))}
                    wk_viz = int(window_cm*10) if metric.startswith("cM") else wk_fallback
                    f = plot_gene_track(si, gff, wk_viz)
                    if f: st.plotly_chart(f, use_container_width=True, key="plot_gene_track")
            except Exception as e: st.error(f"❌ {e}")
        else:
            st.info("💡 [GFF3 camelin Ensembl](https://ftp.ensembl.org/pub/current_gff3/camelus_dromedarius/)")

    # ═══════ GWAS Classique ═══════
    with tabs[6]:
        st.markdown("### 1️⃣ Mapping ID")
        mf = st.file_uploader("CSV mapping (Animal + IID)", type=["csv","txt"], key="gwas_map")
        if mf:
            try:
                st.session_state.mapping_df = pd.read_csv(mf)
                st.success("✅"); st.dataframe(st.session_state.mapping_df.head(3))
            except Exception as e: st.error(f"❌ {e}")
        st.markdown("### 2️⃣ Phénotypes")
        pf = st.file_uploader("Phénotypes", type=["csv","txt","gz"], key="gwas_pheno_file")
        c1, c2 = st.columns(2)
        with c1:
            if pf and st.button("📥 Charger", key="load_ph_btn", use_container_width=True):
                try:
                    mg, _ = match_pheno(st.session_state.ind_filt, load_pheno(pf.read(), pf.name),
                                        st.session_state.mapping_df)
                    st.session_state.pheno_df = mg
                    tc = next((c for c in mg.columns if c not in ("FID","IID","Animal")), None)
                    n_ok = pd.to_numeric(mg[tc], errors="coerce").notna().sum() if tc else 0
                    if n_ok < 10: st.error(f"⚠️ {n_ok} matchés.")
                    else: st.success(f"✅ {n_ok} ind")
                except Exception as e: st.error(f"❌ {e}")
        with c2:
            if st.button("🎲 Simuler", key="sim_ph_btn", use_container_width=True):
                ind = st.session_state.ind_filt; n = len(ind)
                rng = np.random.default_rng(42)
                tr = (st.session_state.pca_s[:,0]*2 + rng.normal(0,0.5,n)
                      if (st.session_state.pca_s is not None and len(st.session_state.pca_s)==n)
                      else rng.normal(0,1,n))
                st.session_state.pheno_df = pd.DataFrame({"FID":ind["FID"].values,
                                                          "IID":ind["IID"].values, "trait":tr})
                st.success("✅")
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            if cols:
                pc = st.selectbox("Phénotype", cols); ib = st.checkbox("Binaire")
                ps = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                nv = ps.notna().sum()
                if nv < 10: st.error(f"❌ {nv} valides")
                else:
                    st.info(f"✅ {nv} valides")
                    if st.button("🚀 Lancer GWAS", type="primary", key="gwas_btn"):
                        try:
                            gg = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                            sg = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                            cv = st.session_state.pca_s[:,:5] if st.session_state.pca_s is not None else np.empty((gg.shape[0],0))
                            with st.spinner("GWAS..."):
                                r = run_gwas(gg, ps.values, st.session_state.king_K, cv, is_bin=ib)
                                r = annotate_gwas(r, sg)
                                r = postprocess_gwas(r, apply_gc=True)
                                st.session_state.gwas_res = r
                                st.session_state.lambda_gc = lambda_gc(r["P"].values) if len(r) else np.nan
                            st.success(f"✅ {len(r)} SNPs · λGC={st.session_state.lambda_gc:.3f}")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("SNPs FDR<0.05", int((r.get("FDR", pd.Series([1])*1) < 0.05).sum()) if "FDR" in r.columns else "N/A")
                            c2.metric("Bonferroni", int((r["P"] < 0.05/len(r)).sum()) if len(r) else 0)
                            c3.metric("λGC", f"{st.session_state.lambda_gc:.3f}" if np.isfinite(st.session_state.lambda_gc) else "N/A")
                            yc_choice = st.radio("Axe -log10 P", ["P", "P_GC"] if "P_GC" in r.columns else ["P"],
                                                horizontal=True, key="manh_yc")
                            st.plotly_chart(plot_manhattan(r, yc=yc_choice, title="Manhattan"),
                                            use_container_width=True, key="plot_gwas_manh_classic")
                            show_cols = [c for c in ["SNP","CHR","BP","MAF","BETA","SE","T","P","FDR"] if c in r.columns]
                            st.dataframe(r[show_cols].head(20), use_container_width=True)
                        except Exception as e: st.error(f"❌ {e}")

    # ═══════ Post-GWAS ═══════
    with tabs[7]:
        if st.session_state.gwas_res is not None:
            st.subheader("🎯 Fine-mapping (ABF/PIP)")
            pip = fine_mapping(st.session_state.gwas_res)
            st.dataframe(pip[["SNP","P","PIP"]].head(10), use_container_width=True)
            st.plotly_chart(plot_pip(pip), use_container_width=True, key="plot_pip_postgwas")
        else: st.warning("Lancez GWAS d'abord")

    # ═══════ GBLUP ═══════
    with tabs[8]:
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            pc = st.selectbox("Phénotype GBLUP", cols, key="gb_pc")
            if st.button("▶ GBLUP + validation", key="gb_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    ps = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                    with st.spinner("REML..."):
                        df, st_ = run_gblup(gu, ps.values)
                        st.session_state.gebv_df = df; st.session_state.gblup_stats = st_
                    st.success("✅"); st.plotly_chart(plot_gblup(df), use_container_width=True, key="plot_gblup")
                    st.json(st_)
                except Exception as e: st.error(f"❌ {e}")
        else: st.warning("Chargez phénotypes")

    # ═══════ Annotation Dédiée ═══════
    with tabs[9]:
        gff_f = st.file_uploader("GFF3/GTF", type=["gff","gff3","gtf","gz"], key="ann_ded_gff")
        src = st.radio("Source", ["Sélection","GWAS"], horizontal=True, key="ann_ded_src")
        wk = st.slider("Fenêtre kb", 1, 500, 50, key="ann_ded_wk")
        if gff_f:
            try:
                gff = parse_gff3(gff_f.read()); st.success(f"✅ {len(gff)} gènes")
                snps = None
                if src == "Sélection" and st.session_state.win_fst is not None:
                    snps = st.session_state.win_fst.head(20).rename(
                        columns={"START":"BP","TOP_SNP":"SNP"})[["SNP","CHR","BP"]]
                elif src == "GWAS" and st.session_state.gwas_res is not None:
                    snps = st.session_state.gwas_res.sort_values("P").head(20)
                if snps is not None and st.button("🔍 Annoter", key="ann_ded_btn"):
                    st.session_state.ann_df = annotate_genes(snps, gff, wk); st.success("✅")
                if st.session_state.ann_df is not None:
                    st.dataframe(st.session_state.ann_df, use_container_width=True)
            except Exception as e: st.error(f"❌ {e}")

    # ═══════ VEP & NCBI ═══════
    with tabs[10]:
        st.subheader("🧬 Annotation Fonctionnelle (VEP & NCBI)")
        sources_snp = []
        if st.session_state.get("pair_res_group") is not None: sources_snp.append("Top SNPs Groupe")
        if st.session_state.get("pair_res_race") is not None: sources_snp.append("Top SNPs Race")
        if st.session_state.get("gwas_res") is not None: sources_snp.append("Top SNPs GWAS")
        if st.session_state.get("win_fst") is not None: sources_snp.append("Top SNPs FST")
        sources_snp.append("Saisie manuelle")
        src_snp = st.selectbox("Source des SNPs", sources_snp, key="vep_src")
        snps_to_analyze = []
        if src_snp == "Top SNPs Groupe": snps_to_analyze = st.session_state.pair_res_group["SNP"].head(20).tolist()
        elif src_snp == "Top SNPs Race": snps_to_analyze = st.session_state.pair_res_race["SNP"].head(20).tolist()
        elif src_snp == "Top SNPs GWAS": snps_to_analyze = st.session_state.gwas_res.sort_values("P")["SNP"].head(20).tolist()
        elif src_snp == "Top SNPs FST": snps_to_analyze = st.session_state.win_fst["TOP_SNP"].head(20).tolist()
        else:
            txt = st.text_area("Liste de SNPs (1/ligne)", key="vep_manual")
            if txt.strip(): snps_to_analyze = [s.strip() for s in txt.splitlines() if s.strip()]

        if snps_to_analyze:
            st.info(f"📊 {len(snps_to_analyze)} SNPs sélectionnés.")
            snp_full = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            vep_input = []
            if snp_full is not None:
                for s in snps_to_analyze:
                    if str(s).startswith("rs"): vep_input.append(s)
                    else:
                        row = snp_full[snp_full["SNP"].astype(str) == str(s)]
                        if not row.empty:
                            r = row.iloc[0]
                            ref = r.get("A1","."); alt = r.get("A2",".")
                            if ref in (".","","nan",None) or alt in (".","","nan",None):
                                vep_input.append(f"{r['CHR']}:{int(r['BP'])}-{int(r['BP'])}")
                            else:
                                vep_input.append(f"{r['CHR']}:{int(r['BP'])}:{ref}:{alt}")
                        else: vep_input.append(s)
            if st.button("🔬 Lancer VEP", type="primary", key="run_vep_btn"):
                with st.spinner("Interrogation Ensembl VEP..."):
                    vep_res = run_vep_api(vep_input)
                    if not vep_res.empty:
                        vep_res["LOCALISATION"] = vep_res["CONSEQUENCE"].apply(traduire_consequence_vep)
                        st.session_state.vep_res = vep_res
                        st.success("✅ VEP terminé.")
            if st.session_state.get("vep_res") is not None and not st.session_state.vep_res.empty:
                st.dataframe(st.session_state.vep_res[["SNP","GENE","LOCALISATION","CONSEQUENCE","IMPACT","AMINO_ACID_CHANGE"]],
                             use_container_width=True)
                st.download_button("⬇ CSV VEP", st.session_state.vep_res.to_csv(index=False).encode(),
                                   "vep_results.csv", "text/csv")

            st.markdown("### NCBI Gene")
            genes_to_fetch = []
            if st.session_state.get("vep_res") is not None and "GENE" in st.session_state.vep_res.columns:
                genes_to_fetch = st.session_state.vep_res["GENE"].dropna().unique().tolist()
            elif st.session_state.get("sel_ann") is not None and "GENE" in st.session_state.sel_ann.columns:
                genes_to_fetch = st.session_state.sel_ann["GENE"].dropna().unique().tolist()
            genes_to_fetch = [g for g in genes_to_fetch if str(g).strip() not in
                             ("N/A","Intergenic/Non-codant","Unknown","nan","")]
            if genes_to_fetch:
                st.info(f"🧬 {len(genes_to_fetch)} gènes valides.")
                if st.button("🏛️ Récupérer NCBI", type="primary", key="run_ncbi_btn"):
                    with st.spinner("NCBI Datasets..."):
                        ncbi_res = fetch_ncbi_gene_info(genes_to_fetch[:20])
                        st.session_state.ncbi_gene_res = ncbi_res
                        if not ncbi_res.empty: st.success("✅ NCBI OK")
                if st.session_state.get("ncbi_gene_res") is not None and not st.session_state.ncbi_gene_res.empty:
                    st.dataframe(st.session_state.ncbi_gene_res, use_container_width=True)
                    st.download_button("⬇ CSV NCBI", st.session_state.ncbi_gene_res.to_csv(index=False).encode(),
                                       "ncbi_gene_info.csv", "text/csv")

    # ═══════ Structure 3D ═══════
    with tabs[11]:
        st.subheader("🧬 AlphaFold — Visualisation 3D")
        genes_to_structure = []
        for key in ("vep_res","sel_ann","ann_df"):
            v = st.session_state.get(key)
            if v is not None and hasattr(v, "columns") and "GENE" in v.columns and not v.empty:
                genes_to_structure = [g for g in v["GENE"].unique()
                                      if str(g).strip() not in ("N/A","Intergenic/Non-codant","Unknown","nan","")]
                if genes_to_structure: break
        if not genes_to_structure:
            st.warning("⚠️ Lancez d'abord VEP ou Annotation.")
        else:
            c1, c2, c3 = st.columns(3)
            target_gene = c1.selectbox("Gène", genes_to_structure, key="gene_3d")
            view_style = c2.selectbox("Style", ["cartoon","stick","sphere","line"], key="style_3d")
            color_scheme = c3.selectbox("Coloration", ["spectrum","chainHetatm","ssPyMol"], key="color_3d")
            highlight_res = st.text_input("Surligner résidu (ex: 123)", key="hl_res")
            if st.button("🔬 Charger 3D", type="primary", key="btn_load_3d"):
                with st.spinner(f"Recherche pour {target_gene}..."):
                    uid = fetch_uniprot_id(target_gene)
                    if uid:
                        st.success(f"✅ UniProt: {uid}")
                        pdb = fetch_alphafold_pdb(uid)
                        if pdb:
                            st.info("💡 Cliquer-glisser pour tourner, molette pour zoomer.")
                            render_3d_structure(pdb, style=view_style, color_scheme=color_scheme,
                                                highlight_res=highlight_res)
                        else: st.error("❌ AlphaFold introuvable.")
                    else: st.error("❌ Gène non trouvé dans UniProt.")

    # ═══════ Alignement ═══════
    with tabs[12]:
        st.subheader("🧬 Alignement Séquences (Rapide vs Lente)")
        if not HAS_BIO:
            st.error("⚠️ Biopython requis : `pip install biopython`")
        else:
            has_race = st.session_state.get("pair_res_race") is not None
            has_group = st.session_state.get("pair_res_group") is not None
            has_gwas = st.session_state.get("gwas_res") is not None
            if has_race or has_group or has_gwas:
                if has_group: snps_to_align = st.session_state.pair_res_group.head(10)
                elif has_race: snps_to_align = st.session_state.pair_res_race.head(10)
                else: snps_to_align = st.session_state.gwas_res.head(10)
                snp_to_choose = st.selectbox("SNP à visualiser :", snps_to_align["SNP"].tolist())
                if st.button("🔍 Générer l'alignement", type="primary"):
                    snp_info = snps_to_align[snps_to_align["SNP"] == snp_to_choose].iloc[0]
                    chrom = str(snp_info["CHR"]).replace("chr",""); bp = int(snp_info["BP"])
                    start = max(1, bp-25); end = bp+25
                    with st.spinner(f"Téléchargement CHR{chrom}:{start}-{end}..."):
                        seq_ref = fetch_ensembl_sequence(chrom, start, end)
                        if seq_ref:
                            snp_pos = bp - start
                            seq_fast = list(seq_ref); seq_slow = list(seq_ref)
                            ref_b = seq_ref[snp_pos]; alt_b = "A" if ref_b != "A" else "G"
                            seq_fast[snp_pos] = alt_b; seq_slow[snp_pos] = ref_b
                            st.markdown(align_and_highlight(seq_ref, "".join(seq_fast), "".join(seq_slow)),
                                        unsafe_allow_html=True)
                        else: st.error("❌ Ensembl n'a pas retourné la séquence.")
            else:
                st.warning("⚠️ Lancez d'abord un GWAS ou une analyse Groupe.")

    # ═══════ Export ═══════
    with tabs[13]:
        if st.session_state.gt_filt is not None:
            if st.button("📦 PLINK", key="pl_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    data = build_plink(gu, st.session_state.ind_filt, su)
                    st.download_button("⬇ PLINK", data, "camel.zip", "application/zip")
                except Exception as e: st.error(f"❌ {e}")
            if st.button("📄 VCF", key="vcf_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                    data = build_vcf(gu, st.session_state.ind_filt, su)
                    st.download_button("⬇ VCF", data.encode(), "camel.vcf", "text/plain")
                except Exception as e: st.error(f"❌ {e}")

    # ═══════ Méta-analyse ═══════
    with tabs[14]:
        st.subheader("🔬 Méta-analyse & Enrichissement")
        st.markdown("## 1️⃣ Permutation GWAS")
        if st.session_state.gt_filt is not None:
            gu_perm = (st.session_state.gt_pruned if st.session_state.gt_pruned is not None
                       else st.session_state.gt_filt)
            su_perm = (st.session_state.snp_pruned if st.session_state.snp_pruned is not None
                       else st.session_state.snp_filt)
            iu_perm = st.session_state.ind_filt
            races_perm, _ = get_pairs(iu_perm["FID"].astype(str).values)
            c1, c2 = st.columns(2)
            ra_p = c1.selectbox("Race A", races_perm, index=0, key="perm_ra")
            rb_p = c2.selectbox("Race B", races_perm, index=1 if len(races_perm)>1 else 0, key="perm_rb")
            c1, c2 = st.columns(2)
            n_perm = c1.slider("N permutations", 50, 1000, 100, 50, key="perm_n")
            maf_p = c2.slider("MAF", 0.0, 0.3, 0.05, key="perm_maf")
            if ra_p != rb_p and st.button("🧪 Permutation test", type="primary", key="perm_btn"):
                try:
                    with st.spinner(f"{n_perm} permutations..."):
                        orig, p_emp, n_obs, perm_counts = permutation_test_race(
                            gu_perm, su_perm, iu_perm["FID"].astype(str).values, ra_p, rb_p, n_perm=n_perm)
                    if orig.empty: st.warning("Aucun SNP testé.")
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Observés", n_obs); c2.metric("p-value empirique", f"{p_emp:.4f}")
                        c3.metric("Seuil", "p<0.05" if p_emp < 0.05 else "p≥0.05")
                        st.plotly_chart(px.histogram(x=perm_counts, nbins=20,
                                                      labels={"x":"Nb SNPs significatifs"},
                                                      title="Distribution nulle"),
                                        use_container_width=True, key="perm_hist")
                except Exception as e: st.error(f"❌ {e}")

        st.divider(); st.markdown("## 2️⃣ Combinaison P-values")
        st.caption("Fisher suppose indépendance (ce qui n'est pas exact ici). Stouffer pondéré est plus robuste.")
        res_r = st.session_state.get("pair_res_race"); res_g = st.session_state.get("pair_res_group")
        if res_r is None or res_g is None:
            st.warning("⚠️ Lancez RACE puis GROUPE.")
        else:
            method = st.radio("Méthode", ["Fisher","Stouffer pondéré"], horizontal=True, key="meta_meth")
            if st.button("🧪 Combiner", type="primary", key="meta_btn"):
                if method == "Fisher":
                    fish = combine_pvalues_fisher(res_r, res_g)
                else:
                    n_r = len(res_r); n_g = len(res_g)
                    fish = combine_stouffer(res_r, res_g, n_r, n_g)
                if fish is None: st.warning("Aucun SNP commun.")
                else:
                    st.success(f"✅ {len(fish)} SNPs combinés")
                    st.dataframe(fish.head(30), use_container_width=True)
                    st.plotly_chart(px.scatter(fish, x="P_race", y="P_group",
                                                hover_data=["SNP"], title="P_race vs P_group"),
                                    use_container_width=True, key="fisher_scatter")
                    st.download_button("⬇ CSV", fish.to_csv(index=False).encode(),
                                       "meta_analysis.csv", "text/csv")

        st.divider(); st.markdown("## 3️⃣ Enrichissement fonctionnel")
        sources_genes = []
        if res_r is not None and not res_r.empty: sources_genes.append("SNPs Race (top 50)")
        if res_g is not None and not res_g.empty: sources_genes.append("SNPs Groupe (top 50)")
        if st.session_state.get("sel_ann") is not None: sources_genes.append("Gènes annotés (sel_ann)")
        sources_genes.append("Liste manuelle")
        if sources_genes:
            c1, c2 = st.columns(2)
            src_g = c1.selectbox("Source gènes", sources_genes, key="enr_src")
            outdir = c2.text_input("Dossier sortie", "", key="enr_outdir")
            gmt_file = st.file_uploader("GMT (optionnel)", type=["gmt"], key="enr_gmt")
            gene_sets = None
            if gmt_file is not None:
                gene_sets = parse_gmt(gmt_file.read()); st.success(f"✅ {len(gene_sets)} gene sets")
            genes_to_enrich = None
            if src_g.startswith("SNPs"):
                if st.session_state.get("vep_res") is not None:
                    genes_to_enrich = st.session_state.vep_res["GENE"].dropna().unique().tolist()
                else: st.warning("⚠️ Lancez d'abord VEP.")
            elif src_g == "Gènes annotés (sel_ann)":
                ann = st.session_state.get("sel_ann")
                if ann is not None and "GENE" in ann.columns:
                    genes_to_enrich = ann["GENE"].dropna().unique().tolist()
            else:
                txt = st.text_area("Liste de gènes (1/ligne)", key="enr_manual")
                if txt.strip(): genes_to_enrich = [g.strip() for g in txt.splitlines() if g.strip()]
            if genes_to_enrich:
                genes_to_enrich = [g for g in genes_to_enrich
                                   if str(g).strip() not in ("N/A","Intergenic/Non-codant","Unknown","nan","")]
            if genes_to_enrich and st.button("🔬 Enrichissement", type="primary", key="enr_btn"):
                try:
                    with st.spinner("Enrichissement (orthologues humains)..."):
                        enr = run_enrichment(genes_to_enrich, gene_sets=gene_sets,
                                             organism="Human", outdir=outdir or None)
                    if enr.empty: st.warning("Aucun terme enrichi.")
                    else:
                        st.success(f"✅ {len(enr)} termes")
                        st.dataframe(enr.head(30), use_container_width=True)
                        st.download_button("⬇ CSV", enr.to_csv(index=False).encode(),
                                           "enrichment.csv", "text/csv")
                except Exception as e: st.error(f"❌ {e}")

        st.divider(); st.markdown("## 4️⃣ Réseau STRING")
        genes_string = []
        if st.session_state.get("sel_ann") is not None and "GENE" in st.session_state.sel_ann.columns:
            genes_string = st.session_state.sel_ann["GENE"].dropna().unique().tolist()
        st.info(f"Gènes disponibles : {len(genes_string)}")
        c1, c2 = st.columns(2)
        score_str = c1.selectbox("Score STRING", [150,400,700,900], index=1, key="string_score")
        max_genes = c2.slider("Nb max", 5, 100, 30, key="string_max")
        if st.button("🕸️ Réseau STRING", type="primary", key="string_btn"):
            try:
                if len(genes_string) < 2: st.warning("Il faut ≥ 2 gènes.")
                else:
                    with st.spinner("STRING..."):
                        edges, nodes = query_string_network(genes_string[:max_genes],
                                                            species=9838,
                                                            required_score=int(score_str))
                    if edges.empty: st.warning("Aucune interaction.")
                    else:
                        st.success(f"✅ {len(edges)} interactions")
                        fig = plot_string_network(edges, nodes)
                        if fig: st.plotly_chart(fig, use_container_width=True, key="string_network_plot")
                        st.dataframe(edges, use_container_width=True)
                        st.download_button("⬇ CSV", edges.to_csv(index=False).encode(),
                                           "string_network.csv", "text/csv")
            except Exception as e: st.error(f"❌ {e}")

    # ═══════ ML ═══════
    with tabs[15]:
        st.subheader("🧠 Random Forest")
        if st.session_state.gt_filt is not None and st.session_state.get("pheno_df") is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            if cols:
                pc = st.selectbox("Phénotype", cols, key="ml_pc")
                if st.button("🚀 Entraîner", type="primary", key="ml_btn"):
                    try:
                        gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                        sg = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                        ps = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                        valid = np.isfinite(ps.values)
                        if valid.sum() < 20: st.error("❌ Pas assez d'individus.")
                        else:
                            X = impute_mean(gu[valid]); y = ps.values[valid]
                            with st.spinner("Random Forest..."):
                                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                                rf.fit(X_tr, y_tr)
                                y_pr = rf.predict(X_te); r2_rf = r2_score(y_te, y_pr)
                                imp = rf.feature_importances_
                                df_imp = pd.DataFrame({"SNP":sg["SNP"], "Importance":imp}
                                                       ).sort_values("Importance", ascending=False).head(15)
                            st.success(f"✅ R² (RF) = {r2_rf:.3f}")
                            st.info("💡 Comparez avec le R² GBLUP pour arbitrer.")
                            fig = px.bar(df_imp, x="Importance", y="SNP", orientation='h',
                                         title="Top 15 SNPs")
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e: st.error(f"❌ {e}")
        else: st.warning("⚠️ Chargez données + phénotypes.")

    # ═══════ Haplotypes ═══════
    with tabs[16]:
        st.subheader("🧬 Blocs d'haplotypes")
        if st.session_state.gt_filt is not None:
            gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            iu = st.session_state.ind_filt
            chroms = sorted(su["CHR"].astype(str).unique())
            sel_chr = st.selectbox("Chromosome", chroms, key="hap_chr")
            sub_idx = su[su["CHR"].astype(str) == sel_chr].index
            if len(sub_idx) > 0:
                if len(sub_idx) > 50: sub_idx = sub_idx[:50]
                snps_names = su.loc[sub_idx, "SNP"].values
                gt_sub = gu[:, sub_idx]
                groups = iu["FID"].astype(str).values
                sort_idx = np.argsort(groups)
                gt_sorted = gt_sub[sort_idx]; groups_sorted = groups[sort_idx]
                fig = px.imshow(gt_sorted,
                    labels=dict(x="Position des SNPs", y="Individus", color="Génotype"),
                    x=snps_names, y=groups_sorted,
                    color_continuous_scale=["#F0F0F0","#3498db","#e74c3c"],
                    title=f"Haplotypes CHR{sel_chr}", aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("🖱️ Survolez. Lignes continues = blocs partagés.")
            else: st.warning("Aucun SNP sur ce chromosome.")

    # ═══════ LiftOver ═══════
    with tabs[17]:
        st.subheader("🌍 LiftOver Camelin → Humain")
        if st.session_state.get("snp_filt") is not None:
            su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
            target_snp = st.selectbox("SNP à convertir", su["SNP"].tolist(), key="lift_snp")
            if st.button("🔄 Convertir", type="primary", key="lift_btn"):
                row = su[su["SNP"] == target_snp].iloc[0]
                chrom, bp = str(row["CHR"]), int(row["BP"])
                with st.spinner(f"Recherche orthologue pour {chrom}:{bp}..."):
                    hc, hb = lift_over(chrom, bp)
                    if hc:
                        st.success(f"✅ Trouvé chez Homme : chr{hc}:{hb}")
                        st.info("💡 Interroger GWAS Catalog pour cette position.")
                    else: st.error("❌ Aucune correspondance.")
        else: st.warning("Chargez des données.")

if __name__ == "__main__":
    main()
