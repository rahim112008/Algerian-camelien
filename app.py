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

st.set_page_config(page_title="🐪 Dromedary Pipeline v1.0", page_icon="🐪",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================
# 🔬 PARAMÈTRES SPÉCIFIQUES AU DROMADAIRE (Camelus dromedarius)
# ============================================================
# Tailles approximatives des 35 autosomes (Assembly CamDro3)
ARS = {
    "1": 105_000_000, "2": 80_000_000, "3": 75_000_000, "4": 70_000_000, "5": 65_000_000,
    "6": 60_000_000, "7": 58_000_000, "8": 55_000_000, "9": 50_000_000, "10": 48_000_000,
    "11": 45_000_000, "12": 42_000_000, "13": 40_000_000, "14": 38_000_000, "15": 35_000_000,
    "16": 33_000_000, "17": 32_000_000, "18": 30_000_000, "19": 28_000_000, "20": 26_000_000,
    "21": 24_000_000, "22": 22_000_000, "23": 20_000_000, "24": 18_000_000, "25": 16_000_000,
    "26": 15_000_000, "27": 14_000_000, "28": 13_000_000, "29": 12_000_000, "30": 11_000_000,
    "31": 10_000_000, "32": 9_000_000, "33": 8_000_000, "34": 7_000_000, "35": 6_000_000
}
AUTO = [str(i) for i in range(1, 36)]

# Races de dromadaires (à adapter selon vos données)
GROUPES_DEFAULT = {
    "Origine géographique (Algérie)": {
        "SAHARA": ["SAHRAOUI", "REGIBI"],
        "STEPPE": ["OULED_SIDI_CHEIKH", "CHAAMBI"],
        "MONTAGNE": ["TARGUI"]
    },
    "Type de production": {
        "LAIT": ["SAHRAOUI", "TARGUI"],
        "VIANDE": ["CHAAMBI", "REGIBI"]
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

# 🌟 PARSEUR AXIOM (Correction automatique du fichier transposé)
def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if fn.endswith(".gz") or raw_bytes[:2]==b"\x1f\x8b" else raw_bytes
    lines = [l.rstrip("\r\n") for l in raw.decode("utf-8", errors="replace").splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    
    start = 0
    for i, line in enumerate(lines):
        if not line.startswith("#"):
            start = i
            break
            
    delim = "\t" if "\t" in lines[start] else " "
    header = lines[start].split(delim)
    data_lines = lines[start+1:]
    n_cols = len(header)
    n_rows = len(data_lines)
    
    if n_rows > n_cols * 2 and n_cols > 2:
        # Format Transposé
        sample_names = header[1:] 
        gtypes_dict = {s: [] for s in sample_names}
        for line in data_lines:
            parts = line.split(delim)
            if len(parts) < n_cols: continue
            for k, cell in enumerate(parts[1:]):
                s_name = sample_names[k]
                if len(cell) == 2 and all(ch in "ACGT" for ch in cell):
                    gtypes_dict[s_name].append((cell[0], cell[1]))
                else:
                    gtypes_dict[s_name].append(("0","0"))
        samples = list(gtypes_dict.keys())
        gtypes = list(gtypes_dict.values())
        return samples, gtypes, len(gtypes[0])
    else:
        # Format Classique
        samples, gtypes = [], []
        for line in data_lines:
            parts = line.split(delim)
            if len(parts) < 2: continue
            samples.append(parts[0].strip())
            alleles = []
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
                    rows.append({"CHR":str(p[0]),"SNP":p[2] or f"{p[0]}:{p[1]}","CM":0.0,"BP":bp,
                                 "A1":p[3],"A2":p[4],"GT":gts})
    if not rows: raise ValueError("Aucun variant VCF")
    gt=np.zeros((len(samples),len(rows)),dtype=np.float32)
    for j,r in enumerate(rows):
        gt[:,j]=r["GT"]; v=gt[:,j][~np.isnan(gt[:,j])]
        if len(v)>0 and v.mean()/2.0>0.5: gt[:,j]=np.where(np.isnan(gt[:,j]),np.nan,2.0-gt[:,j])
    return gt, pd.DataFrame({"FID":samples,"IID":samples}), pd.DataFrame([{k:v for k,v in r.items() if k!="GT"} for r in rows])

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
    rng=np.random.default_rng(seed); pops=["SAHRAOUI", "TARGUI", "REGIBI", "CHAAMBI", "OULED_SIDI_CHEIKH"][:min(n_pop,5)]
    pa=rng.beta(0.6,0.6,n_snp); gt=np.zeros((n_ind,n_snp),dtype=np.float32); ir=[]; k=0
    for p in pops:
        pk=np.clip(pa+rng.normal(0,0.15,n_snp),0.02,0.98)
        for i in range(n_ind//len(pops)+(1 if pops.index(p)<n_ind%len(pops) else 0)):
            gt[k]=(rng.random(n_snp)<pk).astype(np.int8)+(rng.random(n_snp)<pk).astype(np.int8)
            ir.append({"FID":p,"IID":f"{p}_{i+1:03d}"}); k+=1
    gt[rng.random(gt.shape)<0.02]=np.nan
    ch=rng.choice(AUTO,n_snp,p=_autw()); bp=np.array([rng.integers(1,ARS[c]) for c in ch],dtype=np.int64)
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

def apply_qc(gt, ind_df, snp_df, params, rem_out=True):
    gt,ind_df,snp_df=sync_dims(gt,ind_df,snp_df); n0,m0=gt.shape; tr={}
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
        a = (n_bar / n_c) * (s2 - (1.0 / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2 - h_bar / 4))
        b = (n_bar / (n_bar - 1)) * (p_bar * (1 - p_bar) - (r - 1) / r * s2 - (2 * n_bar - 1) / (4 * n_bar) * h_bar)
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
    return SklearnMDS(n_components=nc,dissimilarity="precomputed",random_state=42, n_init=1,max_iter=300,normalized_stress=False).fit_transform(D)

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
    X = np.clip(impute_mean(gt), 0.0, 2.0); rows = []
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

@cache_data
def fst_pairwise(gt, labels, use_wc84=False):
    pops=sorted(np.unique(labels)); K=len(pops); M=np.full((K,K),np.nan); np.fill_diagonal(M,0.0)
    masks={p:(labels==p) for p in pops}
    for i in range(K):
        for j in range(i+1,K):
            sub = gt[masks[pops[i]] | masks[pops[j]]]; sub_lab = labels[masks[pops[i]] | masks[pops[j]]]
            if use_wc84: M[i,j]=M[j,i]=float(np.nanmean(fst_snp_wc(sub, sub_lab)))
            else:
                g1=gt[masks[pops[i]]]; g2=gt[masks[pops[j]]]
                p1=np.nanmean(g1,0)/2.0; p2=np.nanmean(g2,0)/2.0
                n1=(~np.isnan(g1)).sum(0); n2=(~np.isnan(g2)).sum(0)
                dn=np.where(n1+n2>0,n1+n2,np.nan); pb=(p1*n1+p2*n2)/dn
                hs=(2*p1*(1-p1)*n1+2*p2*(1-p2)*n2)/dn; ht=2*pb*(1-pb)
                with np.errstate(divide="ignore",invalid="ignore"): fj=np.clip((ht-hs)/ht, 0.0, None)
                M[i,j]=M[j,i]=float(np.nanmean(fj))
    return M, pops

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
def detect_outliers(win_df, z_thr=3.0):
    df=win_df.copy(); mu,sd=df["MEAN_FST"].mean(),df["MEAN_FST"].std()
    df["Z_FST"]=(df["MEAN_FST"]-mu)/sd if sd>1e-9 else 0.0
    df["OUTLIER"]=df["Z_FST"]>=z_thr
    return df.sort_values("MEAN_FST",ascending=False).reset_index(drop=True)

# ============================================================
# GWAS & GBLUP
# ============================================================
def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9
        H = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        try: step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError: return beta[-1], None
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30); mu = 1.0/(1.0+np.exp(-eta)); W = mu*(1-mu)+1e-9
    try:
        cov = np.linalg.inv((X.T * W) @ X + 1e-8 * np.eye(X.shape[1])); se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError: return float(beta[-1]), None
    return float(beta[-1]), se

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
    try: return max(float(minimize_scalar(nll,bounds=(grid[max(0,ib-1)],grid[min(29,ib+1)]), method="bounded").x),1e-6)
    except: return max(float(grid[ib]),1e-6)

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_bin=False):
    valid = np.isfinite(pheno)
    if valid.sum() < 10: raise ValueError(f"<10 individus valides")
    if K is None or not isinstance(K, np.ndarray) or K.shape[0] != gt.shape[0]: K = np.eye(gt.shape[0])
    Y = pheno[valid]; gv = gt[valid]; Kv = K[np.ix_(valid, valid)]
    X0 = np.ones((len(Y), 1)) if covs is None or covs.size == 0 else np.column_stack([np.ones(len(Y)), np.asarray(covs)[valid]])
    X0 = X0[:, np.std(X0, axis=0) > 1e-10]
    if is_bin: X0_final, Yt, U, w = X0, Y, None, None
    elif use_lmm:
        dl = _vc_delta(Y, X0, Kv); d, U = np.linalg.eigh(Kv); d = np.maximum(d, 0)
        w = 1.0 / np.sqrt(dl * d + 1.0); Yt = (U.T @ Y) * w; X0_final = (U.T @ X0) * w[:, None]
    else: X0_final, Yt, U, w = X0, Y, None, None
    rows = []
    for j in range(gv.shape[1]):
        g = gv[:, j].astype(float)
        if np.isnan(g).mean() > miss_max: continue
        g = np.where(np.isnan(g), np.nanmean(g), g)
        p = g.mean() / 2.0; mj = min(p, 1 - p)
        if mj < maf_min: continue
        if is_bin:
            X = np.column_stack([X0_final, g]); beta_snp, se = _logistic_irls(X, Y)
            if se is None or se <= 0: continue
            tv = beta_snp / se; pval = float(2 * stats.norm.sf(abs(tv)))
            rows.append({"SNP_idx": j, "MAF": mj, "BETA": beta_snp, "SE": se, "T": float(tv), "P": pval})
        else:
            X = np.column_stack([X0_final, (U.T @ g) * w if use_lmm else g])
            try: b, *_ = np.linalg.lstsq(X, Yt, rcond=None)
            except Exception: continue
            res = Yt - X @ b; s2 = float(np.sum(res ** 2) / max(len(Y) - X.shape[1], 1))
            if s2 <= 0: continue
            try: se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
            except Exception: continue
            if se <= 0: continue
            tv = b[-1] / se; pval = float(2 * t_dist.sf(abs(tv), df=max(len(Y) - X.shape[1], 1)))
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
        try: _, fdr, _, _ = multipletests(r["P"].values, method="fdr_bh"); r["FDR"] = fdr
        except Exception: r["FDR"] = np.nan
    else: r["FDR"] = np.nan
    r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0:
            r["P_GC"] = stats.chi2.sf(stats.chi2.isf(r["P"].clip(1e-300,1), df=1) / lam, df=1)
            r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

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
# VEP / NCBI / 3D (ADAPTÉS AU DROMADAIRE)
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
                        cons = sorted(item["transcript_consequences"], key=lambda x: _SEVERITY.get(x.get("impact","MODIFIER"), 0), reverse=True)[0]
                        results.append({"SNP": sid, "GENE": cons.get("gene_symbol","N/A"), "CONSEQUENCE": ", ".join(cons.get("consequence_terms",["N/A"])),
                            "IMPACT": cons.get("impact","N/A"), "AMINO_ACID_CHANGE": cons.get("amino_acids","N/A"), "SIFT": cons.get("sift_prediction","N/A"), "POLYPHEN": cons.get("polyphen_prediction","N/A")})
                    else:
                        results.append({"SNP": sid, "GENE": "Intergenic/Non-codant", "CONSEQUENCE": "N/A", "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
        except Exception as e: st.warning(f"⚠️ VEP réseau: {e}")
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
                results.append({"GENE": gene, "Nom complet": gi.get("description","N/A"), "Chromosome": gi.get("chromosomes",["N/A"])[0], "Résumé": (gi.get("summary","N/A")[:200]+"...") if gi.get("summary") else "N/A"})
            else: results.append({"GENE": gene, "Nom complet": "Non trouvé", "Chromosome": "N/A", "Résumé": "N/A"})
        except Exception as e: results.append({"GENE": gene, "Nom complet": "Erreur API", "Chromosome": "N/A", "Résumé": str(e)})
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

def query_string_network(genes, species=9838, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([str(g).strip() for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(data); edges = df[["preferredName_A","preferredName_B","score"]].copy(); edges.columns = ["SOURCE","TARGET","SCORE"]
        nodes = pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
        return edges, nodes
    except Exception: return pd.DataFrame(), pd.DataFrame()

def annotate_genes(snp_df, gff, wk=50):
    w = wk*1000; res = []
    for _, s in snp_df.iterrows():
        pos = s.get("BP", s.get("START", 0)); ch = str(s["CHR"])
        msk = (gff["CHR"]==ch) & (gff["START"]<=pos+w) & (gff["END"]>=pos-w); ng = gff[msk]
        if not ng.empty:
            for _, g in ng.iterrows():
                res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch, "SNP_POS": pos, "GENE": g["GENE"], "GENE_START": g["START"], "GENE_END": g["END"], "DIST_KB": round(abs(g["START"]-pos)/1000.0, 2)})
    return pd.DataFrame(res)

def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind","Missing/SNP"))
    f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1)
    f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False); return f
def plot_pca(s, v, l):
    f = px.scatter(pd.DataFrame({"PC1":s[:,0],"PC2":s[:,1],"Pop":l}), x="PC1", y="PC2", color="Pop", title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550)
    f.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return f
def plot_mds(m, l):
    f = px.scatter(pd.DataFrame({"MDS1":m[:,0],"MDS2":m[:,1],"Pop":list(l)}), x="MDS1", y="MDS2", color="Pop", title="MDS Reynolds", height=520)
    f.update_traces(marker=dict(size=11, line=dict(width=1, color="white"))); return f
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
def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP"); fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1); p = np.poly1d(z); xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100)
            fig.add_trace(go.Scatter(x=xr, y=p(xr), mode="lines", name="OLS", line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig
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
_KEYS = ["gt","ind_df","snp_df","gt_filt","ind_filt","snp_filt","gt_pre","ind_pre","snp_pre", "gt_pruned","snp_pruned","qc_stats","pca_s","pca_v","mds_c","king_K","fst","fst_M", "fst_pops","admix_Q","admix_K","cv_res","ld_df","roh_df","froh","ne_dict","reyn_D", "reyn_pops","pheno_df","gwas_res","gwas_pheno","lambda_gc","gebv_df","gblup_stats", "ann_df","_plink","_vcf","mapping_df","fst_snp_df","win_fst","taj_df","taj_race", "css_df","pair_res","pair_res_race","pair_res_group","pair_name_race","pair_name_group", "nj_result","nj_mid","sel_gff","sel_ann","sel_ann_race","sel_ann_group","scan_df", "vep_res","ncbi_gene_res","lift_res"]

def reset_derived():
    for k in _KEYS:
        if k not in ("gt","ind_df","snp_df"): st.session_state[k] = None
def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k] = None

def main():
    init_state()
    st.title("🐪 Dromedary Genomics Pipeline v1.0")
    st.caption("Adapté pour *Camelus dromedarius* (Algérie) · QC · LD · Structure · GWAS · VEP · AlphaFold")

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
                    pdl = ["\t".join([sample_fid_iid(smp)[0], sample_fid_iid(smp)[1], "0","0","0","-9"] + [a for pr in row for a in pr]) for smp, row in zip(s, g)]
                    st.session_state["_plink_ped"] = "\n".join(pdl)+"\n"; st.session_state["_plink_map"] = "\n".join(mpl)+"\n"
                    st.success(f"✅ {len(s)}×{ns}")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.get("_plink_ped"):
                st.download_button("⬇ .ped", st.session_state["_plink_ped"].encode(), "out.ped")
                st.download_button("⬇ .map", st.session_state["_plink_map"].encode(), "out.map")
        st.divider(); st.header("⚙️ Seuils QC")
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.05); mind = st.slider("Missing Ind", 0.0, 0.5, 0.05)
        maf_t = st.slider("MAF", 0.0, 0.5, 0.05); hwe_t = st.number_input("HWE", value=1e-6, format="%.0e")
        het_sd = st.slider("Het ±σ", 1.0, 5.0, 3.0, 0.1); rem_out = st.checkbox("❌ Supprimer outliers", value=True)
        ld_r2 = st.slider("LD r²", 0.05, 0.5, 0.2, 0.05)

    if st.session_state.gt is None:
        st.info("👉 Chargez des données (Demo recommandé pour tester)."); return

    tabs = st.tabs(["🧹 QC", "✂️ LD", "🧬 Structure", "🎨 Admixture", "📈 Démographie", "🔍 Sélection", "🧪 GWAS", "🧬 VEP & NCBI", "🧬 3D", "📤 Export"])

    with tabs[0]:
        if st.button("▶ Lancer QC", key="qc_btn"):
            try:
                p = {"geno":geno, "mind":mind, "maf":maf_t, "hwe":hwe_t, "het_sd":het_sd}
                gf, inf, snf, gp, ip, sp, qs = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, p, rem_out=rem_out)
                st.session_state.update({"gt_filt":gf, "ind_filt":inf, "snp_filt":snf, "gt_pre":gp, "ind_pre":ip, "snp_pre":sp, "qc_stats":qs})
                st.success(f"✅ {gf.shape[0]}×{gf.shape[1]}")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.qc_stats:
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Individus", s["n_ind_final"], delta=-s["excluded_ind"]); c2.metric("SNPs", s["n_snp_final"], delta=-s["excluded_snp"])
            st.plotly_chart(plot_missing(miss_ind(st.session_state.gt), miss_snp(st.session_state.gt)), use_container_width=True, key="plot_missing_qc")

    with tabs[1]:
        if st.session_state.gt_filt is not None and st.button("▶ LD Pruning", key="ld_btn"):
            k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
            st.session_state.update({"gt_pruned":st.session_state.gt_filt[:,k], "snp_pruned":st.session_state.snp_filt[k].reset_index(drop=True)})
            st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs")
        if st.session_state.gt_pruned is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avant", st.session_state.gt_filt.shape[1]); c2.metric("Après", st.session_state.gt_pruned.shape[1])
            if st.button("▶ LD Decay", key="ld_decay_btn"): st.session_state.ld_df = ld_decay(st.session_state.gt_pruned, st.session_state.snp_pruned["BP"].values)
            if st.session_state.ld_df is not None:
                f = plot_ld(st.session_state.ld_df)
                if f: st.plotly_chart(f, use_container_width=True, key="plot_ld_decay")

    with tabs[2]:
        if st.session_state.gt_filt is not None and st.button("▶ PCA + MDS", key="struct_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                s_, v_ = pca_analysis(gu, 10); st.session_state.update({"pca_s":s_, "pca_v":v_})
                st.session_state.mds_c = mds_analysis(gu, 5); st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.pca_s is not None:
            st.plotly_chart(plot_pca(st.session_state.pca_s, st.session_state.pca_v, st.session_state.ind_filt["FID"].values), use_container_width=True, key="plot_pca_tab")

    with tabs[3]:
        if st.session_state.gt_filt is not None:
            K = st.slider("K choisi", 2, 10, 4, key="admix_K_slider")
            if st.button("▶ Admixture NMF", key="admix_btn"):
                try:
                    gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                    Q, _ = admixture_nmf(gu, K=K); st.session_state.update({"admix_Q":Q, "admix_K":K}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if st.session_state.admix_Q is not None:
                f, _ = plot_admix(st.session_state.admix_Q, st.session_state.ind_filt["IID"].values, st.session_state.ind_filt["FID"].values, st.session_state.admix_K)
                st.plotly_chart(f, use_container_width=True, key="plot_admix_tab")

    with tabs[4]:
        if st.session_state.gt_filt is not None and st.button("▶ ROH", key="dem_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                roh, fr = detect_roh(gu, su[["CHR","BP"]].to_json()); st.session_state.update({"roh_df":roh, "froh":fr})
                st.success(f"✅ {len(roh)} ROH")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.froh is not None:
            df = pd.DataFrame({"FROH": st.session_state.froh, "Pop": st.session_state.ind_filt["FID"].values})
            st.plotly_chart(px.histogram(df, x="FROH", color="Pop", nbins=50, title="FROH"), use_container_width=True, key="plot_roh_hist")

    with tabs[5]:
        st.subheader("🔍 Sélection")
        use_wc = st.checkbox("Utiliser Weir & Cockerham (1984)", value=True, key="use_wc84")
        if st.button("▶ FST pairwise", key="fst_btn"):
            try:
                gu = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                su = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                lab = st.session_state.ind_filt["FID"].astype(str).values
                st.session_state.fst = (fst_snp_wc(gu, lab) if use_wc else fst_snp(gu, lab)); st.session_state.fst_snp_df = su.copy()
                st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.fst is not None:
            sf = st.session_state.fst_snp_df if st.session_state.get("fst_snp_df") is not None else st.session_state.snp_filt
            nf = min(len(st.session_state.fst), len(sf))
            dff = pd.DataFrame({"SNP":sf["SNP"].values[:nf], "CHR":sf["CHR"].values[:nf], "BP":sf["BP"].values[:nf], "FST":st.session_state.fst[:nf]})
            st.plotly_chart(plot_manhattan(dff, yc="FST", title="FST par SNP"), use_container_width=True, key="plot_manh_fst_snp")

    with tabs[6]:
        st.markdown("### 🧪 GWAS")
        pf = st.file_uploader("Phénotypes", type=["csv","txt","gz"], key="gwas_pheno_file")
        if pf and st.button("📥 Charger", key="load_ph_btn"):
            try:
                mg, _ = match_pheno(st.session_state.ind_filt, load_pheno(pf.read(), pf.name))
                st.session_state.pheno_df = mg; st.success("✅")
            except Exception as e: st.error(f"❌ {e}")
        if st.session_state.pheno_df is not None:
            cols = [c for c in st.session_state.pheno_df.columns if c not in ("FID","IID","Animal")]
            if cols:
                pc = st.selectbox("Phénotype", cols); ib = st.checkbox("Binaire")
                ps = pd.to_numeric(st.session_state.pheno_df[pc], errors="coerce")
                if st.button("🚀 Lancer GWAS", type="primary", key="gwas_btn"):
                    try:
                        gg = st.session_state.gt_pruned if st.session_state.gt_pruned is not None else st.session_state.gt_filt
                        sg = st.session_state.snp_pruned if st.session_state.snp_pruned is not None else st.session_state.snp_filt
                        cv = st.session_state.pca_s[:,:5] if st.session_state.pca_s is not None else np.empty((gg.shape[0],0))
                        with st.spinner("GWAS..."):
                            r = run_gwas(gg, ps.values, st.session_state.king_K, cv, is_bin=ib); r = annotate_gwas(r, sg); r = postprocess_gwas(r, apply_gc=True)
                            st.session_state.gwas_res = r
                        st.success(f"✅ {len(r)} SNPs")
                        st.plotly_chart(plot_manhattan(r, yc="P", title="Manhattan"), use_container_width=True, key="plot_gwas_manh_classic")
                        st.dataframe(r.head(20), use_container_width=True)
                    except Exception as e: st.error(f"❌ {e}")

    with tabs[7]:
        st.subheader("🧬 Annotation Fonctionnelle (VEP)")
        snps_to_analyze = []
        if st.session_state.get("gwas_res") is not None: snps_to_analyze = st.session_state.gwas_res.sort_values("P")["SNP"].head(20).tolist()
        elif st.session_state.get("fst") is not None: snps_to_analyze = st.session_state.snp_filt.head(20)["SNP"].tolist()
        
        if snps_to_analyze:
            st.info(f"📊 {len(snps_to_analyze)} SNPs sélectionnés.")
            if st.button("🔬 Lancer VEP (Dromadaire)", type="primary", key="run_vep_btn"):
                with st.spinner("Interrogation Ensembl VEP pour Camelus dromedarius..."):
                    vep_res = run_vep_api(snps_to_analyze)
                    if not vep_res.empty:
                        vep_res["LOCALISATION"] = vep_res["CONSEQUENCE"].apply(traduire_consequence_vep); st.session_state.vep_res = vep_res; st.success("✅ VEP terminé.")
            if st.session_state.get("vep_res") is not None and not st.session_state.vep_res.empty:
                st.dataframe(st.session_state.vep_res, use_container_width=True)

    with tabs[8]:
        st.subheader("🧬 AlphaFold — Visualisation 3D")
        genes_to_structure = []
        if st.session_state.get("vep_res") is not None and "GENE" in st.session_state.vep_res.columns:
            genes_to_structure = [g for g in st.session_state.vep_res["GENE"].unique() if str(g).strip() not in ("N/A","Intergenic/Non-codant","Unknown","nan","")]
        if not genes_to_structure: st.warning("⚠️ Lancez d'abord VEP pour obtenir des gènes.")
        else:
            target_gene = st.selectbox("Gène", genes_to_structure, key="gene_3d")
            if st.button("🔬 Charger 3D", type="primary", key="btn_load_3d"):
                with st.spinner(f"Recherche UniProt (Taxon 9838) pour {target_gene}..."):
                    uid = fetch_uniprot_id(target_gene, taxon="9838")
                    if uid:
                        st.success(f"✅ UniProt: {uid}"); pdb = fetch_alphafold_pdb(uid)
                        if pdb: render_3d_structure(pdb)
                        else: st.error("❌ AlphaFold introuvable.")
                    else: st.error("❌ Gène non trouvé dans UniProt pour le dromadaire.")

if __name__ == "__main__":
    main()
