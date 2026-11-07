"""🧬 Universal Livestock Genomics Pipeline v23.6 (Bovin & Camelin) — stats pour publication"""
import gzip, io, warnings, zipfile, gc, hashlib, json, time
from collections import Counter
from io import BytesIO
from urllib.parse import quote
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist
from scipy.cluster.hierarchy import linkage, to_tree, leaves_list
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🧬 Universal Genomics v23.6", page_icon="🧬",
                   layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CONFIGURATION DES ESPÈCES
# ============================================================
# ⚠️ Les longueurs chromosomiques (Mb) servent à FROH et à la démo : à vérifier/remplacer
#    par celles de votre assemblage (ARS-UCD1.2 / UMD3.1 pour le bovin ; CamDro pour le dromadaire).
SPECIES_CONFIG = {
    "Camelin (Dromadaire)": {
        "taxon": "9838", "vep_species": "camelus_dromedarius", "prefix": "camel",
        "autosomes": [str(i) for i in range(1, 37)],
        "ars": {"1": 124.7, "2": 123.8, "3": 120.7, "4": 68.3, "5": 97.5, "6": 97.5, "7": 91.2, "8": 89.4, "9": 86.7, "10": 84.1, "11": 81.5, "12": 78.9, "13": 76.3, "14": 73.8, "15": 71.2, "16": 68.6, "17": 66.1, "18": 63.5, "19": 60.9, "20": 58.4, "21": 55.8, "22": 53.2, "23": 50.7, "24": 48.1, "25": 45.5, "26": 43.0, "27": 40.4, "28": 37.8, "29": 35.3, "30": 32.7, "31": 30.1, "32": 27.6, "33": 25.0, "34": 22.4, "35": 19.9, "36": 17.3, "X": 140.0, "Y": 45.0, "MT": 16.6},
        "groupes": {
            "Production laitière": {"HAUTE_LAITIERE": ["MOJAHEEM", "MAGHATEER", "WADAH", "AWARIK"], "FAIBLE_LAITIERE": ["BIKANERI", "JAISALMERI"]},
            "Sous-espèce": {"DROMADAIRE": ["MOJAHEEM", "MAGHATEER", "WADAH", "BIKANERI"], "BACTRIANE": ["KAZAKH_BACTRIAN", "ARVANA"]}
        }
    },
    "Bovin (Taurin/Zébu)": {
        "taxon": "9913", "vep_species": "bos_taurus", "prefix": "bovine",
        "autosomes": [str(i) for i in range(1, 30)],
        "ars": {"1": 158.5, "2": 136.2, "3": 121.0, "4": 120.0, "5": 120.1, "6": 117.8, "7": 110.7, "8": 113.4, "9": 105.7, "10": 103.3, "11": 107.3, "12": 91.2, "13": 84.2, "14": 84.6, "15": 85.2, "16": 81.7, "17": 75.2, "18": 66.1, "19": 64.1, "20": 72.0, "21": 71.6, "22": 61.4, "23": 52.5, "24": 62.4, "25": 43.0, "26": 51.7, "27": 46.8, "28": 46.3, "29": 51.3, "X": 148.8, "Y": 57.8, "MT": 16.3},
        "groupes": {
            "Production": {"LAIT": ["NAR", "NBD", "PTP", "PRS"], "VIANDE": ["LJR", "SHO", "YBS"]},
            "Sous-espèce": {"ZEBU": ["PTP", "PRS"], "TAURUS": ["LJR", "SHO", "NAR", "NBD"]}
        }
    }
}

def make_group_map(group_dict):
    return {r.upper(): grp for grp, races in group_dict.items() for r in races}

def apply_group_mapping(ind_df, group_map, default="AUTRE"):
    df = ind_df.copy(); df["FID"] = df["FID"].astype(str).str.upper()
    df["FID_GROUPE"] = df["FID"].map(group_map).fillna(default)
    return df

# ============================================================
# UTILS & CACHE
# ============================================================
def _ha(x):
    """Hash exact (blake2b) d'un tableau numpy — évite les collisions du cache."""
    if not isinstance(x, np.ndarray) or x.size == 0: return "empty"
    if x.dtype.kind in ("U", "S", "O"):
        h = hashlib.blake2b("|".join(map(str, x.ravel())).encode("utf-8", "replace"), digest_size=16)
        return f"a|{x.shape}|{h.hexdigest()}"
    return f"{x.shape}|{x.dtype}|{hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=16).hexdigest()}"

def _hn(x):
    try: return _ha(np.asarray(x))
    except Exception: return f"{type(x).__name__}|{len(x) if hasattr(x, '__len__') else 0}"

HF = {np.ndarray: _ha, pd.Series: _hn, pd.Index: _hn}
for _n in ("ArrowStringArray", "StringArray", "IntegerArray", "FloatingArray", "BooleanArray", "NumpyExtensionArray", "PandasArray", "DatetimeArray", "TimedeltaArray", "PeriodArray", "IntervalArray", "Categorical"):
    _c = getattr(pd.arrays, _n, None) or getattr(pd.core.arrays, _n, None)
    if _c: HF[_c] = _hn

def cache_data(func=None, **kw):
    def _d(f): return st.cache_data(show_spinner=False, hash_funcs=HF, **kw)(f)
    return _d if func is None else _d(func)

def impute_mean(gt):
    g = gt.astype(np.float32, copy=True); cm = np.nanmean(g, axis=0); cm = np.where(np.isnan(cm), 0.0, cm); nm = np.isnan(g)
    if nm.any(): g[nm] = np.take(cm, np.where(nm)[1])
    return g

def _std_geno(gt):
    X = impute_mean(gt); X = X - X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return X / sd

def _autw(ars, autosomes):
    L = np.array([ars[c] for c in autosomes], dtype=float); return L / L.sum()

def _chk(c):
    s = str(c).upper().replace("CHROMOSOME", "").replace("CHR", "")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    return (1, sp[s], "") if s in sp else (2, 0, s)

def _chk_num(c):
    t = _chk(c); return t[0] * 1000 + t[1]

def sort_chr_bp(df, bp="BP"):
    return (df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str))
              .sort_values(["_k", "_s", bp], kind="stable").drop(columns=["_k", "_s"]).reset_index(drop=True))

def _norm_chr(s):
    return s.astype(str).str.strip().str.replace(r"^(?i:chr)", "", regex=True)

def _rt(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def sample_fid_iid(s, fp=2):
    n = str(s).strip()
    for e in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(e): n = n[:-len(e)]
    n = n.replace("(", "_").replace(")", "").replace("[", "_").replace("]", "_").replace(" ", "_").replace("/", "_")
    while "__" in n: n = n.replace("__", "_")
    p = n.strip("_").split("_"); return ("_".join(p[:fp]) if len(p) >= fp else p[0]), n

def _ca(a):
    a = str(a).strip() if a is not None else "0"; return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a

def _sample_blocks(n, max_snp, block=200, seed=42):
    """Sous-échantillonne des BLOCS contigus (préserve la densité locale, indispensable pour le LD)."""
    if n <= max_snp: return np.arange(n)
    nb = int(np.ceil(n / block)); k = max(1, max_snp // block)
    ch = np.sort(np.random.default_rng(seed).choice(nb, size=min(k, nb), replace=False))
    return np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in ch])

# ============================================================
# PARSERS
# ============================================================
def parse_axiom(raw_bytes, fn=""):
    raw = gzip.decompress(raw_bytes) if fn.endswith(".gz") or raw_bytes[:2] == b"\x1f\x8b" else raw_bytes
    lines = [l.rstrip("\r\n") for l in raw.decode("utf-8", errors="replace").splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    start = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample") or "Sample Filename" in lines[0][:50]) else 0
    samples, gtypes = [], []
    for line in lines[start:]:
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2: continue
        samples.append(parts[0].strip()); alleles = []
        for cell in parts[1:]:
            tk = cell.split()
            if len(tk) == 2: alleles.append((_ca(tk[0]), _ca(tk[1])))
            elif len(tk) == 1 and len(tk[0]) == 2 and all(ch in "ACGT0" for ch in tk[0]): alleles.append((tk[0][0], tk[0][1]))
            else: alleles.append(("0", "0"))
        gtypes.append(alleles)
    if not samples: raise ValueError("Aucun échantillon")
    c = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
    for i, r in enumerate(gtypes): gtypes[i] = r[:c] + [("0", "0")] * (c - len(r))
    return samples, gtypes, c

# ✅ CORRECTION 1 : Ignorer la ligne d'en-tête (#Chromosome ...)
def parse_map(b):
    rows = []
    for line in _rt(b).splitlines():
        if line.startswith("#"): continue
        p = line.split()
        if len(p) >= 4:
            try: cm = float(p[2]) if p[2] not in ("0", ".", "NA") else 0.0
            except ValueError: cm = 0.0
            try: bp = int(float(p[3]))
            except ValueError: continue
            rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": cm, "BP": bp})
    if not rows: raise ValueError(".map vide")
    df = pd.DataFrame(rows); df["CHR"] = _norm_chr(df["CHR"])
    if df["BP"].max() < 10000: df["BP"] = df["BP"] * 1_000_000   # positions données en Mb
    return df

# ✅ CORRECTION 2 : Parseur PED robuste pour gérer le format sans les 6 colonnes de base
def parse_ped(b, n_map):
    """Parseur PED robuste : gère le format PLINK standard et les exports alternatifs (ex: Axiom)."""
    lines = [l for l in _rt(b).splitlines() if l.strip()]
    if not lines: raise ValueError(".ped vide")
    
    if lines[0].startswith("#") or lines[0].startswith("Sample") or lines[0].startswith("FID"):
        lines = lines[1:]
        if not lines: raise ValueError(".ped vide après en-tête")
        
    first_line = lines[0].split()
    is_standard = False
    
    if len(first_line) >= 6:
        try:
            int(first_line[2]); int(first_line[3]); int(first_line[4])
            is_standard = True
        except ValueError:
            pass
    
    if is_standard:
        n_ped = (len(first_line) - 6) // 2
        start_idx = 6
        id_idx = 1
    else:
        n_ped = (len(first_line) - 1) // 2
        start_idx = 1
        id_idx = 0
        
    if n_ped < 7: raise ValueError(f"PED a {n_ped} SNPs")
    n = min(n_map, n_ped); ec = start_idx + 2 * n; fids, iids, rows = [], [], []
    
    for line in lines:
        p = line.split()
        if len(p) >= ec: 
            fids.append(p[0] if is_standard else "FAM")
            iids.append(p[id_idx])
            rows.append(p[start_idx:start_idx + 2 * n])
            
    if not iids: raise ValueError("Aucun individu")
    
    arr = np.array(rows, dtype=str)
    uniq, inv = np.unique(arr, return_inverse=True); inv = np.asarray(inv).reshape(arr.shape)
    miss_codes = [i for i, u in enumerate(uniq) if u in ("0", ".")]
    a1, a2 = inv[:, 0::2], inv[:, 1::2]
    miss = np.isin(a1, miss_codes) | np.isin(a2, miss_codes); ok = ~miss
    K = len(uniq); cnt = np.zeros((K + 1, n), dtype=np.int32)
    for k in range(K):
        if k in miss_codes: continue
        cnt[k] = ((a1 == k) & ok).sum(0) + ((a2 == k) & ok).sum(0)
    uniq_ext = np.append(uniq, ".")
    order = np.argsort(-cnt, axis=0, kind="stable"); major, minor = order[0], order[1]
    cols = np.arange(n)
    A1 = np.where(cnt[major, cols] > 0, uniq_ext[major], "."); A2 = np.where(cnt[minor, cols] > 0, uniq_ext[minor], ".")
    gt = (a1 == minor[None, :]).astype(np.float32) + (a2 == minor[None, :]).astype(np.float32)
    gt[miss] = np.nan
    return gt, pd.DataFrame({"FID": fids, "IID": iids}), A1.tolist(), A2.tolist()

def parse_vcf(b):
    samples, rows, gts_list = [], [], []
    for line in _rt(b).splitlines():
        if line.startswith("#CHROM"): samples = line.split("\t")[9:]
        elif line.strip() and not line.startswith("#"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 10 and "," not in p[4]:
                fmt = p[8].split(":")
                if "GT" not in fmt: continue
                gi = fmt.index("GT"); gts = np.full(len(samples), np.nan, dtype=np.float32)
                for k, s in enumerate(p[9:9 + len(samples)]):
                    try:
                        al = s.split(":")[gi].replace("|", "/").split("/")
                        if not any(a in (".", "") for a in al): gts[k] = sum(int(a) for a in al)
                    except (ValueError, IndexError): pass
                try: bp = int(float(p[1]))
                except (ValueError, TypeError): continue
                rows.append({"CHR": str(p[0]), "SNP": p[2] if p[2] not in ("", ".") else f"{p[0]}:{p[1]}", "CM": 0.0, "BP": bp, "A1": p[3], "A2": p[4]})
                gts_list.append(gts)
    if not rows: raise ValueError("Aucun variant VCF")
    gt = np.column_stack(gts_list).astype(np.float32)
    for j in range(gt.shape[1]):
        v = gt[:, j][~np.isnan(gt[:, j])]
        if len(v) > 0 and v.mean() / 2.0 > 0.5:
            gt[:, j] = 2.0 - gt[:, j]; rows[j]["A1"], rows[j]["A2"] = rows[j]["A2"], rows[j]["A1"]
    df_snp = pd.DataFrame(rows); df_snp["CHR"] = _norm_chr(df_snp["CHR"])
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), df_snp

def _gff_attr(a, key):
    for part in a.split(";"):
        part = part.strip()
        if part.startswith(key + "="): return part[len(key) + 1:]
    return None

def parse_gff3(b):
    genes = []
    for line in _rt(b).splitlines():
        if line.startswith("#") or not line.strip(): continue
        p = line.split("\t")
        if len(p) < 9 or p[2] not in ("gene", "ncRNA_gene", "pseudogene"): continue
        a = p[8]; name = None
        if 'gene_name "' in a: name = a.split('gene_name "')[1].split('"')[0]
        elif 'gene_id "' in a: name = a.split('gene_id "')[1].split('"')[0]
        else:
            for k in ("Name", "gene", "gene_name", "ID"):
                name = _gff_attr(a, k)
                if name: break
        if not name: name = "Unknown"
        for pre in ("gene:", "gene-"):
            if name.startswith(pre): name = name[len(pre):]
        chrom = p[0][3:] if p[0].lower().startswith("chr") else p[0]
        try: genes.append({"CHR": str(chrom), "START": int(p[3]), "END": int(p[4]), "GENE": name})
        except ValueError: continue
    df = pd.DataFrame(genes, columns=["CHR", "START", "END", "GENE"])
    return df.drop_duplicates().reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_gff(b):
    return parse_gff3(b)

def parse_gmt(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace"); sets = {}
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3: continue
        genes = [g for g in parts[2:] if g]
        if genes: sets[parts[0]] = genes
    return sets

def load_pheno(b, fn=""):
    raw = gzip.decompress(b) if fn.endswith(".gz") else b
    text = raw.decode("utf-8", errors="replace"); sep = "\t" if text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, engine="python")

def match_pheno(ind_df, pheno_df, mapping_df=None):
    """Aligne les phénotypes sur ind_df (même ordre, même longueur — doublons éliminés)."""
    ind = ind_df.copy(); ind["_k"] = ind["IID"].astype(str)
    if mapping_df is not None:
        p, m = pheno_df.copy(), mapping_df.copy()
        id_c = next((c for c in ["Animal", "IID", "id", "ID"] if c in p.columns), None)
        if id_c is None: raise ValueError("Colonne ID absente.")
        if id_c != "IID": p = p.drop(columns=["IID"], errors="ignore")
        p[id_c] = p[id_c].astype(str); map_a = next((c for c in m.columns if c != "IID"), None)
        if map_a is None or "IID" not in m.columns: raise ValueError("Le mapping doit contenir IID + une colonne d'identifiant.")
        m[map_a] = m[map_a].astype(str); m["IID"] = m["IID"].astype(str)
        pm = p.merge(m, left_on=id_c, right_on=map_a, how="left"); pm["_k"] = pm["IID"].astype(str)
        pm = pm.drop_duplicates("_k")
        out = ind[["_k", "FID", "IID"]].merge(pm.drop(columns=["IID", "FID"], errors="ignore"), on="_k", how="left")
        return out.drop(columns="_k"), id_c
    id_c = next((c for c in ["IID", "id", "ID", "sample", "Animal", "animal"] if c in pheno_df.columns), None)
    if id_c is None: raise ValueError("Aucune colonne ID.")
    p = pheno_df.copy(); p["_k"] = p[id_c].astype(str); p = p.drop_duplicates("_k")
    out = ind[["_k", "FID", "IID"]].merge(p.drop(columns=[id_c, "FID"], errors="ignore"), on="_k", how="left")
    return out.drop(columns="_k"), id_c

def gen_demo(n_ind=150, n_snp=800, n_pop=4, seed=42, autosomes=None, ars=None):
    rng = np.random.default_rng(seed)
    if not autosomes or not ars:
        autosomes = [str(i) for i in range(1, 30)]; ars = {c: 100.0 for c in autosomes}
    pops = [f"POP_{i}" for i in range(1, min(n_pop, 10) + 1)]
    pa = rng.beta(0.6, 0.6, n_snp); gt = np.zeros((n_ind, n_snp), dtype=np.float32); ir = []; k = 0
    for pi_, p in enumerate(pops):
        pk = np.clip(pa + rng.normal(0, 0.15, n_snp), 0.02, 0.98)
        for i in range(n_ind // len(pops) + (1 if pi_ < n_ind % len(pops) else 0)):
            gt[k] = (rng.random(n_snp) < pk).astype(np.int8) + (rng.random(n_snp) < pk).astype(np.int8)
            ir.append({"FID": p, "IID": f"{p}_{i + 1:03d}"}); k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    ch = rng.choice(autosomes, n_snp, p=_autw(ars, autosomes)); bp = np.array([rng.integers(1, int(ars[c] * 1_000_000)) for c in ch], dtype=np.int64)
    snp = pd.DataFrame({"CHR": ch.astype(str), "SNP": [f"rs{i:07d}" for i in range(n_snp)], "CM": bp / 1e6, "BP": bp, "A1": "A", "A2": "G"})
    order = np.lexsort((bp, snp["CHR"].map(_chk_num).values))
    return gt[:, order], pd.DataFrame(ir), snp.iloc[order].reset_index(drop=True)

def rec_fid(ind):
    df = ind.copy()
    if df["FID"].nunique() <= 1: df["FID"] = df["IID"].astype(str).str.split("_").str[0]
    return df

# ============================================================
# QC
# ============================================================
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): p = np.nanmean(g, axis=0) / 2.0; return np.minimum(p, 1.0 - p)

def het(g):
    """Taux d'hétérozygotie par individu, calculé sur les génotypes NON manquants."""
    ok = ~np.isnan(g); n = ok.sum(1)
    return np.where(n > 0, ((g == 1) & ok).sum(1) / np.maximum(n, 1), np.nan)

def hwe_exact(nh, n0, n2):
    n = nh + n0 + n2
    if n == 0 or (nh == 0 and (n0 == 0 or n2 == 0)): return 1.0
    rare = 2 * min(n0, n2) + nh; mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    pr = np.zeros(rare + 1); pr[mid] = 1.0; s = 1.0
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2: pr[ch + 2] = pr[ch] * 4 * cr * cc / ((ch + 2) * (ch + 1)); s += pr[ch + 2]; ch += 2; cr -= 1; cc -= 1
    ch, cr, cc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2: pr[ch - 2] = pr[ch] * ch * (ch - 1) / (4 * (cr + 1) * (cc + 1)); s += pr[ch - 2]; ch -= 2; cr += 1; cc += 1
    po = pr[nh] if nh < len(pr) else 0.0
    return float(min(pr[pr <= po + 1e-7].sum() / s, 1.0))

@cache_data
def hwe_pvals(g):
    pv = np.full(g.shape[1], np.nan)
    n0 = ((g == 0) & ~np.isnan(g)).sum(0); n1 = ((g == 1) & ~np.isnan(g)).sum(0); n2 = ((g == 2) & ~np.isnan(g)).sum(0); nv = n0 + n1 + n2
    if g.shape[1] <= 20000:
        for j in range(g.shape[1]):
            if nv[j] >= 5: pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(nv > 0, nv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            eh = 2.0 * p * (1.0 - p) * nv; ok = (eh >= 5) & np.isfinite(eh); chi2 = (np.abs(n1 - eh) - 0.5) ** 2 / eh; pv[ok] = stats.chi2.sf(chi2[ok], df=1)
    return pv

def sync_dims(gt, ind_df, snp_df):
    n = min(gt.shape[0], len(ind_df)); m = min(gt.shape[1], len(snp_df))
    return gt[:n, :m], ind_df.iloc[:n].reset_index(drop=True), snp_df.iloc[:m].reset_index(drop=True)

def apply_qc(gt, ind_df, snp_df, params, autosomes, keep_autosomes=True, rem_out=True):
    gt, ind_df, snp_df = sync_dims(gt, ind_df, snp_df); n0, m0 = gt.shape; tr = {}
    if keep_autosomes:
        auto = {str(a).upper() for a in autosomes}
        mask = snp_df["CHR"].astype(str).str.upper().str.replace(r"^CHR", "", regex=True).isin(auto).values
        gt = gt[:, mask]; snp_df = snp_df[mask].reset_index(drop=True); tr["autosomes"] = int(m0 - gt.shape[1])
    m = gt.shape[1]
    k = miss_snp(gt) <= params["geno"]; gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True); tr["geno"] = int(m - gt.shape[1]); m = gt.shape[1]
    k = miss_ind(gt) <= params["mind"]; gt = gt[k]; ind_df = ind_df[k].reset_index(drop=True); tr["mind"] = int(n0 - gt.shape[0])
    mf = maf(gt); k = np.isfinite(mf) & (mf >= params["maf"]); gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True); tr["maf"] = int(m - gt.shape[1]); m = gt.shape[1]
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus (MAF)")
    if params.get("hwe_pop"):
        pvd = hwe_by_pop(gt, ind_df["FID"].astype(str).values, int(params.get("hwe_min_n", 10)))
        if pvd.shape[1] == 0: raise ValueError("Aucune population avec assez d'individus pour le HWE par population")
        k = (pvd.values < params["hwe"]).sum(1) < int(params.get("hwe_k", 1)); tr["hwe_pops_testées"] = int(pvd.shape[1])
    else:
        pv = hwe_pvals(gt); k = np.isnan(pv) | (pv >= params["hwe"])
    gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True); tr["hwe"] = int(m - gt.shape[1])
    if gt.shape[1] == 0: raise ValueError("Tous les SNPs exclus (HWE)")
    n_pre = int(gt.shape[0])
    if rem_out:
        h = het(gt); kp = np.ones(len(h), dtype=bool)
        for f in ind_df["FID"].unique():
            msk = (ind_df["FID"] == f).values; sub = h[msk]
            if len(sub) >= 5 and np.nanstd(sub) > 1e-9:
                z = (sub - np.nanmean(sub)) / np.nanstd(sub); idx = np.where(msk)[0]; kp[idx[np.abs(z) > params["het_sd"]]] = False
        gt = gt[kp]; ind_df = ind_df[kp].reset_index(drop=True); tr["het"] = int(n_pre - len(ind_df))
    gc.collect()
    return gt, ind_df, snp_df, {"n_ind_init": int(n0), "n_snp_init": int(m0), "n_ind_pre": n_pre, "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
                                "excluded_ind": int(n0 - gt.shape[0]), "excluded_snp": int(m0 - gt.shape[1]), "trace": tr}

# ============================================================
# ALGORITHMES
# ============================================================
def _wc_abc(gt, labels, min_n=2):
    """Composantes a, b, c de Weir & Cockerham (1984), vectorisées sur les SNPs."""
    labels = np.asarray(labels).astype(str); pops = np.unique(labels); m = gt.shape[1]
    n = np.zeros((len(pops), m)); p = np.zeros_like(n); h = np.zeros_like(n)
    for k, pop in enumerate(pops):
        g = gt[labels == pop]; ok = ~np.isnan(g); nn = ok.sum(0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pk = np.where(nn > 0, np.nansum(g, 0) / (2 * nn), 0.0); hk = np.where(nn > 0, ((g == 1) & ok).sum(0) / nn, 0.0)
        use = nn >= min_n
        n[k] = np.where(use, nn, 0.0); p[k] = np.where(use, pk, 0.0); h[k] = np.where(use, hk, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n > 0).sum(0).astype(float); nsum = n.sum(0); nbar = nsum / r; pbar = (n * p).sum(0) / nsum
        s2 = (n * (p - pbar) ** 2).sum(0) / (nbar * (r - 1)); hbar = (n * h).sum(0) / nsum
        nc = (r * nbar - (n ** 2).sum(0) / (r * nbar)) / (r - 1)
        a = (nbar / nc) * (s2 - (1.0 / (nbar - 1)) * (pbar * (1 - pbar) - (r - 1) / r * s2 - hbar / 4))
        b = (nbar / (nbar - 1)) * (pbar * (1 - pbar) - (r - 1) / r * s2 - (2 * nbar - 1) / (4 * nbar) * hbar)
        den = a + b + hbar / 2.0
    valid = (r >= 2) & (nbar > 1) & (nc > 0) & np.isfinite(den) & np.isfinite(a)
    return np.where(valid, a, np.nan), np.where(valid, b, np.nan), np.where(valid, hbar / 2.0, np.nan)

def _wc_components(gt, labels, min_n=2):
    a, b, c = _wc_abc(gt, labels, min_n); return a, a + b + c

@cache_data
def fst_snp_wc(gt, labels):
    a, den = _wc_components(gt, labels)
    with np.errstate(invalid="ignore", divide="ignore"): f = a / den
    return np.where(np.isfinite(f) & (den > 1e-9), np.clip(f, 0.0, None), np.nan)

def fst_global_wc(gt, labels):
    """FST global = rapport des moyennes (Σa / Σ(a+b+c)), recommandé par W&C 1984."""
    a, den = _wc_components(gt, labels); ok = np.isfinite(a) & np.isfinite(den)
    return float(max(0.0, a[ok].sum() / den[ok].sum())) if ok.any() and den[ok].sum() > 0 else np.nan

@cache_data
def pca_analysis(gt, nc=10):
    X = impute_mean(gt); X = X - X.mean(0); n = min(nc, X.shape[0] - 1, X.shape[1]); p = PCA(n_components=n)
    return p.fit_transform(X), p.explained_variance_ratio_ * 100.0

@cache_data
def mds_analysis(gt, nc=5):
    X = impute_mean(gt); n = X.shape[0]; D = np.zeros((n, n), dtype=np.float32)
    for i in range(n): D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    kw = dict(n_components=nc, random_state=42, n_init=1, max_iter=300)
    try: return SklearnMDS(dissimilarity="precomputed", normalized_stress=False, **kw).fit_transform(D)
    except Exception: return SklearnMDS(metric="precomputed", **kw).fit_transform(D)

@cache_data
def compute_diversity(gt, labels):
    """Ho, He, Fis, PIC, MAF moyen par population (NaN correctement exclus)."""
    labels = np.asarray(labels).astype(str); rows = []
    for p in np.unique(labels):
        sub = gt[labels == p]
        if sub.shape[0] < 3: continue
        ok = ~np.isnan(sub); n = ok.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            freq = np.where(n > 0, np.nansum(sub, 0) / (2 * n), np.nan); ho_snp = np.where(n > 0, ((sub == 1) & ok).sum(0) / n, np.nan)
        v = np.isfinite(freq)
        if not v.any(): continue
        f = freq[v]; q = 1 - f; he = float(np.mean(2 * f * q)); ho = float(np.nanmean(ho_snp[v]))
        pic = float(np.mean(1 - (f ** 2 + q ** 2) - 2 * (f ** 2) * (q ** 2)))
        rows.append({"Population": p, "N": int(sub.shape[0]), "H_Observee (Ho)": round(ho, 4), "H_Attendue (He)": round(he, 4),
                     "Fis (Consanguinité)": round(1 - ho / he, 4) if he > 0 else 0.0, "PIC": round(pic, 4),
                     "MAF_moyenne": round(float(np.mean(np.minimum(f, q))), 4), "SNPs_Polymorphes": int(((f > 0) & (f < 1)).sum())})
    return pd.DataFrame(rows)

@cache_data
def king_mat(gt):
    """Kinship KING-robust (Manichaikul 2010) : φ = (N_Aa,Aa − 2·N_AA,aa) / (N_Aa(i) + N_Aa(j))."""
    ok = ~np.isnan(gt); H = ((gt == 1) & ok).astype(np.float32); G0 = ((gt == 0) & ok).astype(np.float32); G2 = ((gt == 2) & ok).astype(np.float32)
    h = H.sum(1).astype(np.float64); BH = H @ H.T; ibs0 = G0 @ G2.T + G2 @ G0.T; d = h[:, None] + h[None, :]
    with np.errstate(divide="ignore", invalid="ignore"): K = (BH - 2 * ibs0) / d
    K[~np.isfinite(K)] = 0.0; np.fill_diagonal(K, 0.5); return K

@cache_data
def grm_vanraden(gt):
    """GRM de VanRaden (2008) — matrice semi-définie positive, utilisée pour le LMM."""
    X = impute_mean(gt); p = X.mean(0) / 2.0; Z = X - 2 * p; den = 2 * np.sum(p * (1 - p))
    if den <= 0: return np.eye(gt.shape[0])
    return ((Z @ Z.T) / den).astype(np.float64)

@cache_data
def admixture_nmf(gt, K=3, seed=42):
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=500); W = m.fit_transform(np.clip(impute_mean(gt), 0.0, 2.0)); s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W / s, m.components_

@cache_data
def choose_best_k(gt, k_min=2, k_max=8, seed=42):
    X = np.clip(impute_mean(gt), 0.0, 2.0); rows = []
    for k in range(k_min, k_max + 1):
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400); W = m.fit_transform(X)
        rows.append({"K": k, "RMSE": float(np.sqrt(np.mean((X - W @ m.components_) ** 2)))})
    return pd.DataFrame(rows)

@cache_data
def ld_pruning(gt, chrs, w=50, step=5, r2=0.2):
    """Élagage LD par fenêtre glissante (style PLINK --indep-pairwise), chromosome par chromosome."""
    n = gt.shape[1]; keep = np.ones(n, dtype=bool)
    if n < 3: return keep
    chrs = np.asarray(chrs).astype(str); X = _std_geno(gt).astype(np.float32); N = X.shape[0]
    for c in pd.unique(chrs):
        idx = np.where(chrs == c)[0]
        for s in range(0, len(idx), step):
            win = idx[s:s + w]; win = win[keep[win]]
            if len(win) < 2: continue
            R2 = ((X[:, win].T @ X[:, win]) / N) ** 2
            for a in range(len(win) - 1):
                if not keep[win[a]]: continue
                hits = np.where(R2[a, a + 1:] > r2)[0] + a + 1; keep[win[hits]] = False
    return keep

def _pair_r2(X, bp, chrs, win=100, max_bp=None):
    n = X.shape[1]; N = X.shape[0]; D, R = [], []
    for i in range(n - 1):
        j1 = min(i + 1 + win, n); same = chrs[i + 1:j1] == chrs[i]
        if not same.any(): continue
        d = np.abs(bp[i + 1:j1] - bp[i]); ok = same & (d > 0)
        if max_bp is not None: ok &= (d <= max_bp)
        if ok.any():
            r = (X[:, i + 1:j1][:, ok].T @ X[:, i]) / N; D.append(d[ok]); R.append(r.astype(np.float64) ** 2)
    return (np.concatenate(D), np.concatenate(R)) if D else (np.array([]), np.array([]))

def _sorted_snps(gt, snp_df):
    s = snp_df.reset_index(drop=True).copy(); s["CHR"] = s["CHR"].astype(str); s["BP"] = s["BP"].astype(float)
    order = np.lexsort((s["BP"].values, s["CHR"].map(_chk_num).values))
    return gt[:, order], s.iloc[order].reset_index(drop=True)

@cache_data
def ld_decay(gt, snp_df, max_kb=1000, max_snp=20000):
    if gt.shape[1] < 10: return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    gt, s = _sorted_snps(gt, snp_df)
    idx = _sample_blocks(gt.shape[1], max_snp); gt = gt[:, idx]; s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt); dist, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 100, max_kb * 1000)
    if len(dist) == 0: return pd.DataFrame(columns=["dist_kb", "r2", "count"])
    df = pd.DataFrame({"dist_kb": dist / 1000.0, "r2": r2}); df["bin"] = pd.cut(df["dist_kb"], bins=np.linspace(0, max_kb, 30), labels=False, include_lowest=True)
    return df.groupby("bin").agg(dist_kb=("dist_kb", "mean"), r2=("r2", "mean"), count=("r2", "count")).dropna().sort_values("dist_kb").reset_index(drop=True)

@cache_data
def ne_historical(gt, snp_df, max_snp=10000):
    """Ne historique à partir du LD (Sved 1971 : E[r²] = 1/(1+4Nc)), t = 1/(2c) générations."""
    N = gt.shape[0]
    if N < 5 or gt.shape[1] < 20: return pd.DataFrame({"GenAgo": [], "Ne": []})
    gt, s = _sorted_snps(gt, snp_df); idx = _sample_blocks(gt.shape[1], max_snp); gt = gt[:, idx]; s = s.iloc[idx].reset_index(drop=True)
    X = _std_geno(gt); d, r2 = _pair_r2(X, s["BP"].values, s["CHR"].values, 50, 50e6); d = d / 1e6
    m = (d >= 0.001) & (d <= 50)
    if m.sum() < 20: return pd.DataFrame({"GenAgo": [], "Ne": []})
    d, r2 = d[m], r2[m]; bins = np.logspace(np.log10(d.min()), np.log10(d.max()), 20); rows = []
    for k in range(len(bins) - 1):
        sel = (d >= bins[k]) & (d < bins[k + 1])
        if sel.sum() >= 10: rows.append((np.sqrt(bins[k] * bins[k + 1]), r2[sel].mean()))
    if len(rows) < 3: return pd.DataFrame({"GenAgo": [], "Ne": []})
    dm, rm = np.array(rows).T; c = dm * 0.01
    r2a = np.clip(rm - 1.0 / N, 1e-6, 0.999); ne = (1.0 / r2a - 1.0) / (4.0 * c); t = 1.0 / (2.0 * c)
    df = pd.DataFrame({"GenAgo": t, "Ne": ne}).replace([np.inf, -np.inf], np.nan).dropna(); df = df[df["Ne"] > 0].sort_values("GenAgo").reset_index(drop=True)
    if len(df) > 3: df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df

@cache_data
def detect_roh(gt, chrs, bps, min_snps=30, min_kb=500.0):
    """ROH = suites de SNPs sans hétérozygote (les manquants ne cassent pas la suite). À lancer sur données NON élaguées."""
    chrs = np.asarray(chrs).astype(str); bps = np.asarray(bps).astype(np.int64); n_ind = gt.shape[0]; rohs = []; fb = np.zeros(n_ind); tot = 0.0
    for c in pd.unique(chrs):
        idxs = np.where(chrs == c)[0]
        if len(idxs) < min_snps: continue
        idxs = idxs[np.argsort(bps[idxs], kind="stable")]; pos = bps[idxs]; sp = pos[-1] - pos[0]
        if sp < 1: continue
        tot += sp; G = gt[:, idxs]
        for i in range(n_ind):
            bnd = np.concatenate(([-1], np.where(G[i] == 1)[0], [len(idxs)]))
            for a, b in zip(bnd[:-1], bnd[1:]):
                s_, e_ = a + 1, b - 1
                if e_ < s_: continue
                nr = e_ - s_ + 1; lkb = (pos[e_] - pos[s_]) / 1000.0
                if nr >= min_snps and lkb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": c, "start_bp": int(pos[s_]), "end_bp": int(pos[e_]), "n_snp": int(nr), "length_kb": float(lkb)}); fb[i] += pos[e_] - pos[s_]
    cols = ["IID_idx", "CHR", "start_bp", "end_bp", "n_snp", "length_kb"]
    return (pd.DataFrame(rohs, columns=cols)), (fb / tot if tot > 0 else fb)

@cache_data
def fst_pairwise(gt, labels, use_wc84=True):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); K = len(pops); M = np.full((K, K), np.nan); np.fill_diagonal(M, 0.0)
    for i in range(K):
        for j in range(i + 1, K):
            mi, mj = labels == pops[i], labels == pops[j]
            if use_wc84: v = fst_global_wc(gt[mi | mj], labels[mi | mj])
            else:
                g1, g2 = gt[mi], gt[mj]; p1 = np.nanmean(g1, 0) / 2.0; p2 = np.nanmean(g2, 0) / 2.0; n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
                dn = np.where(n1 + n2 > 0, n1 + n2, np.nan); pb = (p1 * n1 + p2 * n2) / dn; hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / dn; ht = 2 * pb * (1 - pb)
                ok = np.isfinite(ht) & np.isfinite(hs); v = float(max(0.0, (ht[ok] - hs[ok]).sum() / ht[ok].sum())) if ok.any() and ht[ok].sum() > 0 else np.nan
            M[i, j] = M[j, i] = v
    return M, pops

@cache_data
def windowed_fst(gt, snp_df, labels, w=50, step=25):
    a, den = _wc_components(gt, labels); snp = snp_df.reset_index(drop=True).copy(); n = min(len(snp), len(a)); snp = snp.iloc[:n].copy(); a = a[:n]; den = den[:n]
    snp["_a"] = a; snp["_d"] = den
    with np.errstate(invalid="ignore", divide="ignore"): snp["FST"] = np.where(np.isfinite(den) & (den > 1e-9), np.clip(a / den, 0.0, None), np.nan)
    res = []
    for c in snp["CHR"].astype(str).unique():
        sub = snp[snp["CHR"].astype(str) == c].sort_values("BP").reset_index(drop=True)
        if len(sub) < w: continue
        A, Dn, F, B, S = sub["_a"].values, sub["_d"].values, sub["FST"].values, sub["BP"].values, sub["SNP"].values
        for i in range(0, len(sub) - w + 1, step):
            sl = slice(i, i + w); okm = np.isfinite(A[sl]) & np.isfinite(Dn[sl])
            if okm.sum() < 5: continue
            dd = Dn[sl][okm].sum()
            if dd <= 1e-9: continue
            f_sl = F[sl]; top = int(np.nanargmax(f_sl)) if np.isfinite(f_sl).any() else 0
            res.append({"CHR": c, "START": int(B[sl].min()), "END": int(B[sl].max()), "MID": int((B[sl].min() + B[sl].max()) / 2), "N_SNP": w,
                        "MEAN_FST": float(max(0.0, A[sl][okm].sum() / dd)), "MAX_FST": float(np.nanmax(f_sl)) if np.isfinite(f_sl).any() else np.nan, "TOP_SNP": S[sl][top]})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "MEAN_FST", "MAX_FST", "TOP_SNP"])

@cache_data
def detect_outliers(win_df, z_thr=3.0):
    df = win_df.copy()
    if df.empty: df["Z_FST"] = []; df["OUTLIER"] = []; return df
    mu, sd = df["MEAN_FST"].mean(), df["MEAN_FST"].std(); df["Z_FST"] = (df["MEAN_FST"] - mu) / sd if sd > 1e-9 else 0.0; df["OUTLIER"] = df["Z_FST"] >= z_thr
    return df.sort_values("MEAN_FST", ascending=False).reset_index(drop=True)

@cache_data
def tajimas_window(genos):
    """D de Tajima + π et θ_W de la fenêtre. ⚠️ Sur puce SNP : biais d'ascertainment — interpréter en relatif."""
    ni, ns = genos.shape
    if ns < 3 or ni < 4: return np.nan, 0, np.nan, np.nan
    S = 0; piv = []; thv = []
    for j in range(ns):
        col = genos[:, j]; col = col[~np.isnan(col)]
        if len(col) < 2: continue
        p = col.mean() / 2.0; N = 2 * len(col)
        if 0 < p < 1: S += 1; piv.append(2 * p * (1 - p) * N / (N - 1)); thv.append(1.0 / np.sum(1.0 / np.arange(1, N)))
    if S == 0 or not piv: return np.nan, int(S), np.nan, np.nan
    pi = float(np.sum(piv)); th = float(np.sum(thv)); n = ni * 2; a1 = np.sum(1.0 / np.arange(1, n)); a2 = np.sum(1.0 / (np.arange(1, n) ** 2)); b1 = (n + 1) / (3 * (n - 1)); b2 = 2 * (n ** 2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1; c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 ** 2); e1 = c1 / a1; e2 = c2 / (a1 ** 2 + a2); den = np.sqrt(e1 * S + e2 * S * (S - 1))
    if den <= 1e-12: return np.nan, int(S), pi, th
    return float((pi - S / a1) / den), int(S), pi, th

@cache_data
def tajimas_scan(gt, snp_df, w=25, step=10):
    snp = snp_df.copy().reset_index(drop=True); n = min(gt.shape[1], len(snp)); snp = snp.iloc[:n].reset_index(drop=True); gt = gt[:, :n]; res = []
    for c in snp["CHR"].unique():
        idx = np.where(snp["CHR"].values == c)[0]
        if len(idx) < w: continue
        idx = idx[np.argsort(snp["BP"].values[idx])]
        for i in range(0, len(idx) - w + 1, step):
            wi = idx[i:i + w]; D, S, pis, ths = tajimas_window(gt[:, wi]); bpw = snp["BP"].values[wi]; span_kb = max((bpw.max() - bpw.min()) / 1000.0, 1e-9)
            if not np.isnan(D): res.append({"CHR": str(c), "START": int(bpw.min()), "END": int(bpw.max()), "MID": int((bpw.min() + bpw.max()) / 2), "N_SNP": len(wi), "S": S, "TAJIMA_D": D,
                                            "PI_PAR_SNP": pis / len(wi), "THETA_W_PAR_SNP": ths / len(wi), "PI_PAR_KB": pis / span_kb})
    return pd.DataFrame(res, columns=["CHR", "START", "END", "MID", "N_SNP", "S", "TAJIMA_D", "PI_PAR_SNP", "THETA_W_PAR_SNP", "PI_PAR_KB"])

@cache_data
def composite_score(gt, snp_df, labels):
    fst = fst_snp_wc(gt, labels); ok = ~np.isnan(gt); n = ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        hom = np.where(n > 0, (((gt == 0) | (gt == 2)) & ok).sum(0) / n, np.nan); mv = np.nanmean(gt, axis=0) / 2.0
    mv = np.minimum(mv, 1 - mv); df = snp_df.copy().reset_index(drop=True); m = min(len(df), len(fst)); df = df.iloc[:m].reset_index(drop=True)
    df["FST"] = fst[:m]; df["HOM"] = hom[:m]; df["MAF"] = mv[:m]
    def norm(x): mu, sd = np.nanmean(x), np.nanstd(x); return (x - mu) / sd if sd > 1e-9 else np.zeros_like(x)
    df["Z_FST"] = norm(df["FST"].values); df["Z_HOM"] = norm(df["HOM"].values); df["CSS"] = np.abs(df["Z_FST"] * df["Z_HOM"])
    return df.sort_values("CSS", ascending=False).reset_index(drop=True)

def _reynolds_D(g, labels, pops):
    """Distance de Reynolds (1983) : D = −ln(1 − θ)."""
    ph = {p: np.nanmean(g[labels == p], 0) / 2.0 for p in pops}; K = len(pops); D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi_, pj = ph[pops[i]], ph[pops[j]]; ok = np.isfinite(pi_) & np.isfinite(pj)
            num = np.sum((pi_[ok] - pj[ok]) ** 2); dn = np.sum(pi_[ok] * (1 - pj[ok]) + pj[ok] * (1 - pi_[ok]))
            if dn > 0: D[i, j] = D[j, i] = -np.log(1.0 - min(num / dn, 0.999999))
    return D

@cache_data
def reynolds(gt, labels):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); return _reynolds_D(gt, labels, pops), pops

@cache_data
def upgma_bootstrap(gt, snp_df, labels, n_boot=100, seed=42):
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); K = len(pops); n_snp = gt.shape[1]
    if K < 2: raise ValueError("Au moins 2 populations requises")
    Do = _reynolds_D(gt, labels, pops); Zo = linkage(squareform(Do, checks=False), method="average"); nl = K
    def clusters(Z):
        cl = {}
        for k, row in enumerate(Z):
            i, j = int(row[0]), int(row[1]); l = frozenset([i]) if i < nl else cl[i]; r = frozenset([j]) if j < nl else cl[j]; cl[nl + k] = frozenset(l | r)
        return cl
    co = clusters(Zo); sc = np.zeros(len(Zo)); rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = rng.choice(n_snp, n_snp, replace=True); Zb = linkage(squareform(_reynolds_D(gt[:, idx], labels, pops), checks=False), method="average"); cb = set(clusters(Zb).values())
        for k, cl in co.items():
            if cl in cb: sc[k - nl] += 1
    supp = (sc / n_boot * 100).round(0); tr = to_tree(Zo)
    def nw(node):
        if node.is_leaf(): return f"{pops[node.id]}"
        l = nw(node.get_left()); r = nw(node.get_right())
        return f"({l},{r}){int(supp[node.id - nl])}:{node.dist:.4f}" if node.id >= nl else f"({l},{r}):{node.dist:.4f}"
    return {"tree": tr, "Z": Zo, "newick": nw(tr) + ";", "support": supp, "D": Do, "pops": pops}

@cache_data
def midpoint_info(D, labels):
    D = np.asarray(D, dtype=float); n = len(labels); D = (D + D.T) / 2; np.fill_diagonal(D, 0); im, jm, md = 0, 1, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] > md: md = D[i, j]; im, jm = i, j
    return {"leaf_i": labels[im], "leaf_j": labels[jm], "max_dist": float(md), "midpoint": float(md / 2)}

# ============================================================
# GWAS
# ============================================================
def _logistic_irls(X, y, max_iter=25, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9; H = (X.T * W) @ X + 1e-8 * np.eye(X.shape[1])
        try: step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError: return float(beta[-1]), None
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
        beta = beta_new
    eta = np.clip(X @ beta, -30, 30); mu = 1.0 / (1.0 + np.exp(-eta)); W = mu * (1 - mu) + 1e-9
    try: cov = np.linalg.inv((X.T * W) @ X + 1e-8 * np.eye(X.shape[1])); se = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError: return float(beta[-1]), None
    return float(beta[-1]), se

def _pairwise_core(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None):
    """Case/control entre deux classes (races ou groupes). Test allélique vectorisé ou régression logistique (+ PCs)."""
    groups = np.asarray(groups).astype(str); mask = np.isin(groups, [ga, gb])
    if mask.sum() < 20: raise ValueError(f"<20 individus pour {ga} vs {gb}")
    g = gt[mask]; y = (groups[mask] == ga).astype(int)
    if (y == 1).sum() < 5 or (y == 0).sum() < 5: raise ValueError("Moins de 5 individus dans une des classes")
    snp = snp_df.reset_index(drop=True); n = min(g.shape[1], len(snp)); g = g[:, :n]; snp = snp.iloc[:n].reset_index(drop=True); cv = covs[mask] if covs is not None else None
    gA, gB = g[y == 1], g[y == 0]; nA = (~np.isnan(gA)).sum(0).astype(float); nB = (~np.isnan(gB)).sum(0).astype(float); sA = np.nansum(gA, 0); sB = np.nansum(gB, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pA = sA / (2 * nA); pB = sB / (2 * nB); tot = (sA + sB) / (2 * (nA + nB)); mj = np.minimum(tot, 1 - tot)
        hs = (2 * pA * (1 - pA) * nA + 2 * pB * (1 - pB) * nB) / (nA + nB); pb = (pA * nA + pB * nB) / (nA + nB); ht = 2 * pb * (1 - pb)
        fv = np.where(ht > 1e-9, np.clip((ht - hs) / ht, 0, None), 0.0)
    valid = (nA >= 5) & (nB >= 5) & np.isfinite(mj) & (mj >= maf_min)
    if method == "allelic":
        a1, a0, b1, b0 = sA, 2 * nA - sA, sB, 2 * nB - sB; N = a1 + a0 + b1 + b0
        with np.errstate(invalid="ignore", divide="ignore"):
            den = (a1 + a0) * (b1 + b0) * (a1 + b1) * (a0 + b0); chi2 = N * (a1 * b0 - a0 * b1) ** 2 / den
        valid &= np.isfinite(chi2) & (den > 0); pv = np.where(valid, stats.chi2.sf(np.where(valid, chi2, 0), 1), np.nan)
    else:
        chi2 = np.full(n, np.nan); pv = np.full(n, np.nan)
        for j in np.where(valid)[0]:
            X = np.column_stack([np.ones(len(y))] + ([cv] if cv is not None else []) + [g[:, j]]); val = ~np.isnan(X).any(1)
            b_, se = _logistic_irls(X[val], y[val])
            if se is None or se <= 0: valid[j] = False; continue
            z = b_ / se; chi2[j] = z ** 2; pv[j] = float(2 * stats.norm.sf(abs(z)))
        valid &= np.isfinite(pv)
    idx = np.where(valid)[0]
    if len(idx) == 0: return pd.DataFrame()
    df = pd.DataFrame({"SNP": snp["SNP"].values[idx], "CHR": snp["CHR"].astype(str).values[idx], "BP": snp["BP"].astype(int).values[idx], "freq_A": pA[idx].round(4), "freq_B": pB[idx].round(4),
                       "delta_freq": (pA - pB)[idx].round(4), "MAF": mj[idx].round(4), "chi2": chi2[idx].round(4), "P": pv[idx], "FST_snp": fv[idx].round(4), "n_A": nA[idx].astype(int), "n_B": nB[idx].astype(int)})
    return df.sort_values("P").reset_index(drop=True)

@cache_data
def gwas_pairwise(gt, snp_df, labels, ra, rb, maf_min=0.05, method="allelic", covs=None):
    return _pairwise_core(gt, snp_df, labels, ra, rb, maf_min, method, covs)

@cache_data
def gwas_pairwise_group(gt, snp_df, groups, ga, gb, maf_min=0.05, method="allelic", covs=None):
    return _pairwise_core(gt, snp_df, groups, ga, gb, maf_min, method, covs)

@cache_data
def get_pairs(labels):
    r = sorted(pd.unique(labels).astype(str)); return r, [(r[i], r[j]) for i in range(len(r)) for j in range(i + 1, len(r))]

def _vc_delta(Y, X0, K):
    n = len(Y); d, U = np.linalg.eigh(K); d = np.maximum(d, 0); Yt = U.T @ Y; X0t = U.T @ X0
    def nll(dl):
        if dl <= 0: return 1e18
        w = 1.0 / np.sqrt(dl * d + 1.0); Yw = Yt * w; Xw = X0t * w[:, None]
        try: b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        except Exception: return 1e18
        s2 = float(np.sum((Yw - Xw @ b) ** 2) / n)
        return 1e18 if s2 <= 0 else 0.5 * (n * np.log(s2) + np.sum(np.log(dl * d + 1.0)))
    grid = np.logspace(-3, 3, 30); lls = np.array([nll(dl) for dl in grid]); ib = int(np.argmin(lls))
    try: return max(float(minimize_scalar(nll, bounds=(grid[max(0, ib - 1)], grid[min(29, ib + 1)]), method="bounded").x), 1e-6)
    except Exception: return max(float(grid[ib]), 1e-6)

_GWAS_COLS = ["SNP_idx", "MAF", "BETA", "SE", "T", "P"]

@cache_data
def run_gwas(gt, pheno, K, covs, maf_min=0.05, miss_max=0.05, use_lmm=True, is_bin=False):
    """GWAS quantitatif (LMM avec GRM, ou OLS) vectorisé, ou binaire (logistique + PCs). L'intercept est TOUJOURS conservé."""
    pheno = np.asarray(pheno, dtype=float); valid = np.isfinite(pheno)
    if valid.sum() < 10: raise ValueError("<10 individus valides")
    Y = pheno[valid]; gv = gt[valid]; n = len(Y); miss = np.isnan(gv).mean(0); Gi = impute_mean(gv); p = Gi.mean(0) / 2.0; mj = np.minimum(p, 1 - p)
    keep = np.where((miss <= miss_max) & (mj >= maf_min) & (Gi.std(0) > 1e-8))[0]
    if len(keep) == 0: return pd.DataFrame(columns=_GWAS_COLS)
    X0 = np.ones((n, 1))
    if covs is not None and np.size(covs) > 0:
        C = np.asarray(covs, dtype=float)[valid]; C = C[:, np.nanstd(C, 0) > 1e-10]; X0 = np.column_stack([X0, C])
    if is_bin:
        rows = []
        for j in keep:
            b_, se = _logistic_irls(np.column_stack([X0, Gi[:, j].astype(float)]), Y)
            if se is None or se <= 0: continue
            tv = b_ / se; rows.append({"SNP_idx": int(j), "MAF": float(mj[j]), "BETA": b_, "SE": se, "T": float(tv), "P": float(2 * stats.norm.sf(abs(tv)))})
        return pd.DataFrame(rows, columns=_GWAS_COLS)
    lmm = use_lmm and isinstance(K, np.ndarray) and K.shape[0] == gt.shape[0]
    if lmm:
        Kv = K[np.ix_(valid, valid)]; dl = _vc_delta(Y, X0, Kv); d, U = np.linalg.eigh(Kv); d = np.maximum(d, 0); w = 1.0 / np.sqrt(dl * d + 1.0)
        Yt = (U.T @ Y) * w; X0t = (U.T @ X0) * w[:, None]
    else: Yt, X0t, U, w = Y, X0, None, None
    dof = n - X0t.shape[1] - 1
    if dof < 5: raise ValueError("Trop peu d'individus pour ce nombre de covariables")
    Q, _ = np.linalg.qr(X0t); Yr = Yt - Q @ (Q.T @ Yt); yy = float(Yr @ Yr); out = []
    for s in range(0, len(keep), 4000):
        idx = keep[s:s + 4000]; Gc = Gi[:, idx].astype(np.float64); Gt = (U.T @ Gc) * w[:, None] if lmm else Gc
        Gr = Gt - Q @ (Q.T @ Gt); gg = (Gr ** 2).sum(0); ok = gg > 1e-10; gg_s = np.where(ok, gg, 1.0)
        beta = (Gr.T @ Yr) / gg_s; s2 = np.clip((yy - beta ** 2 * gg_s) / dof, 1e-300, None); se = np.sqrt(s2 / gg_s); tv = beta / se; pv = 2 * t_dist.sf(np.abs(tv), df=dof)
        out.append(pd.DataFrame({"SNP_idx": idx[ok], "MAF": mj[idx][ok], "BETA": beta[ok], "SE": se[ok], "T": tv[ok], "P": pv[ok]}))
    return pd.concat(out, ignore_index=True)

def annotate_gwas(r, snp_df):
    if r.empty: return r
    idx = r["SNP_idx"].astype(int).values; r = r.copy()
    r["SNP"] = snp_df["SNP"].astype(str).values[idx]; r["CHR"] = snp_df["CHR"].astype(str).values[idx]; r["BP"] = snp_df["BP"].astype(float).astype(np.int64).values[idx]
    r["CM"] = snp_df["CM"].astype(float).values[idx] if "CM" in snp_df.columns else 0.0
    return r[["SNP_idx", "SNP", "CHR", "BP", "CM", "MAF", "BETA", "SE", "T", "P"]]

def lambda_gc(pvals):
    p = np.asarray(pvals); p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) < 10: return np.nan
    return float(np.median(stats.chi2.isf(p, df=1)) / stats.chi2.ppf(0.5, df=1))

def _bh(p):
    p = np.asarray(p, dtype=float); n = len(p); o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out

def postprocess_gwas(r, apply_gc=True):
    if r is None or r.empty: return r
    r = r.copy(); r["FDR"] = _bh(r["P"].values); r["log10P"] = -np.log10(r["P"].clip(1e-300, 1))
    if apply_gc:
        lam = lambda_gc(r["P"].values)
        if np.isfinite(lam) and lam > 1.0:
            r["P_GC"] = stats.chi2.sf(stats.chi2.isf(r["P"].clip(1e-300, 1), df=1) / lam, df=1); r["log10P_GC"] = -np.log10(r["P_GC"].clip(1e-300, 1))
    return r.sort_values("P").reset_index(drop=True)

def fine_mapping(r, window_mb=1.0, w_prior=0.04):
    """Fine-mapping ABF de Wakefield (2009) dans une fenêtre autour du SNP principal (un seul variant causal supposé)."""
    if r is None or r.empty: return pd.DataFrame()
    d = r[np.isfinite(r["BETA"]) & np.isfinite(r["SE"]) & (r["SE"] > 0)].copy()
    if d.empty: return d
    lead = d.sort_values("P").iloc[0]
    reg = d[(d["CHR"].astype(str) == str(lead["CHR"])) & ((d["BP"] - lead["BP"]).abs() <= window_mb * 1e6)].copy()
    V = reg["SE"] ** 2; z2 = reg["BETA"] ** 2 / V; reg["logABF"] = 0.5 * np.log(V / (V + w_prior)) + 0.5 * z2 * w_prior / (V + w_prior)
    reg["ABF"] = np.exp(reg["logABF"] - reg["logABF"].max()); reg["PIP"] = reg["ABF"] / reg["ABF"].sum()
    reg = reg.sort_values("PIP", ascending=False).reset_index(drop=True); reg["CUM_PIP"] = reg["PIP"].cumsum(); return reg

def run_gblup(gt, pheno, kf=5, ids=None, seed=42):
    """GBLUP : h² par maximum de vraisemblance (phénotype centré, décomposition spectrale) + validation croisée k-fold."""
    pheno = np.asarray(pheno, dtype=float); valid = np.isfinite(pheno)
    if valid.sum() < 20: raise ValueError("<20 phénotypes")
    Y = pheno[valid]; n = len(Y); kf = max(2, min(kf, n - 1)); X = impute_mean(gt[valid]); p = X.mean(0) / 2.0; Z = X - 2 * p
    G = (Z @ Z.T) / (2 * np.sum(p * (1 - p))) + 1e-5 * np.eye(n); d, U = np.linalg.eigh(G); d = np.maximum(d, 0)
    mu = Y.mean(); Yt = U.T @ (Y - mu)
    def ll(h2): lam = (1 - h2) / h2; return -0.5 * (np.sum(np.log(d + lam)) + n * np.log(np.sum(Yt ** 2 / (d + lam))))
    grid = np.linspace(0.01, 0.99, 50); ib = int(np.argmax([ll(h) for h in grid]))
    try: h2 = float(minimize_scalar(lambda h: -ll(h), bounds=(grid[max(0, ib - 1)], grid[min(49, ib + 1)]), method="bounded").x)
    except Exception: h2 = float(grid[ib])
    lam = (1 - h2) / h2; gebv = U @ (d / (d + lam) * Yt)
    rng = np.random.default_rng(seed); idx = rng.permutation(n); folds = np.array_split(idx, kf); yp = np.zeros(n)
    for te in folds:
        tr = np.setdiff1d(idx, te); mtr = Y[tr].mean(); yp[te] = mtr + G[np.ix_(te, tr)] @ np.linalg.solve(G[np.ix_(tr, tr)] + lam * np.eye(len(tr)), Y[tr] - mtr)
    corr = float(np.corrcoef(Y, yp)[0, 1]) if n > 2 else 0.0
    names = np.asarray(ids)[valid] if ids is not None else np.where(valid)[0]
    return pd.DataFrame({"IID": names, "Phenotype": Y, "GEBV": gebv}), {"h2": h2, "corr": corr, "r2": corr ** 2, "rmse": float(np.sqrt(np.mean((Y - yp) ** 2)))}

# ============================================================
# MÉTA-ANALYSE & API
# ============================================================
def _merge_res(res_race, res_group):
    return res_race[["SNP", "CHR", "BP", "P"]].rename(columns={"P": "P_race"}).merge(res_group[["SNP", "P"]].rename(columns={"P": "P_group"}), on="SNP", how="inner")

def combine_pvalues_fisher(res_race, res_group):
    """⚠️ Fisher suppose des tests indépendants : race×race et groupe×groupe partagent des individus."""
    if res_race is None or res_group is None: return None
    m = _merge_res(res_race, res_group)
    if m.empty: return None
    m["P_combined"] = [stats.combine_pvalues(r, method="fisher")[1] for r in m[["P_race", "P_group"]].values]; m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

def combine_stouffer(res_race, res_group, n_race, n_group):
    """Stouffer pondéré par √N (N = nombre d'INDIVIDUS, pas de SNPs)."""
    if res_race is None or res_group is None: return None
    m = _merge_res(res_race, res_group)
    if m.empty: return None
    wr, wg = np.sqrt(n_race), np.sqrt(n_group); zr = stats.norm.isf(m["P_race"].clip(1e-300, 1).values / 2); zg = stats.norm.isf(m["P_group"].clip(1e-300, 1).values / 2)
    m["P_combined"] = 2 * stats.norm.sf(np.abs((wr * zr + wg * zg) / np.sqrt(wr ** 2 + wg ** 2))); m["logP_combined"] = -np.log10(m["P_combined"].clip(1e-300, 1))
    return m.sort_values("P_combined").reset_index(drop=True)

@cache_data
def permutation_test_race(gt, snp_df, labels, ra, rb, n_perm=100, seed=42, max_snp=5000, maf_min=0.05):
    rng = np.random.default_rng(seed); labels = np.asarray(labels).astype(str)
    if len(snp_df) > max_snp:
        idx = np.sort(rng.choice(len(snp_df), max_snp, replace=False)); snp_df = snp_df.iloc[idx].reset_index(drop=True); gt = gt[:, idx]
    original = _pairwise_core(gt, snp_df, labels, ra, rb, maf_min)
    if original.empty: return pd.DataFrame(), 1.0, 0, [], 1.0
    n_obs = int((original["P"] < 0.05 / len(original)).sum()); min_obs = original["P"].min()
    mask = np.isin(labels, [ra, rb]); lab_sub = labels[mask].copy(); gt_sub = gt[mask]; counts, minps = [], []
    for _ in range(n_perm):
        rng.shuffle(lab_sub); r = _pairwise_core(gt_sub, snp_df, lab_sub, ra, rb, maf_min)
        counts.append(int((r["P"] < 0.05 / len(r)).sum()) if not r.empty else 0); minps.append(float(r["P"].min()) if not r.empty else 1.0)
    return original, float((np.sum(np.array(counts) >= n_obs) + 1) / (n_perm + 1)), n_obs, counts, float((np.sum(np.array(minps) <= min_obs) + 1) / (n_perm + 1))

_SEVERITY = {"HIGH": 4, "MODERATE": 3, "LOW": 2, "MODIFIER": 1}

def traduire_consequence_vep(c):
    c = str(c).lower()
    if any(k in c for k in ("stop_gained", "frameshift", "start_lost", "stop_lost")): return "⛔ Exon (Non-sens / perte de codon)"
    if "missense" in c: return "🔴 Exon (Faux-sens - Protéine modifiée)"
    if "synonymous" in c: return "🟡 Exon (Silencieuse - Protéine inchangée)"
    if "splice" in c: return "⚠️ Jonction Exon/Intron"
    if "intron" in c: return "🟢 Intron (Régulation)"
    if "3_prime_utr" in c: return "🔵 Région 3'UTR"
    if "5_prime_utr" in c: return "🔵 Région 5'UTR"
    if "upstream" in c: return "🟣 Régulateur (Promoteur amont)"
    if "downstream" in c: return "🟣 Régulateur (Aval)"
    if "regulatory" in c or "tf_binding" in c: return "🟣 Région régulatrice"
    return "⚪ Intergénique"

def run_vep_api(items, species, batch=200):
    """items = [(snp_id, 'id'|'region', valeur)]. 'region' = "CHR POS POS REF/ALT 1" ; 'id' = identifiant rsID (endpoint /vep/id)."""
    if not items: return pd.DataFrame()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}; results = []
    def _post(url, payload, mapping):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code == 429: time.sleep(float(r.headers.get("Retry-After", 2))); r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
            if r.status_code != 200: st.warning(f"⚠️ VEP HTTP {r.status_code} : {r.text[:150]}"); return
            for item in r.json():
                sid = mapping.get(item.get("input"), item.get("input", "N/A"))
                if item.get("transcript_consequences"):
                    cons = sorted(item["transcript_consequences"], key=lambda x: _SEVERITY.get(x.get("impact", "MODIFIER"), 0), reverse=True)[0]
                    results.append({"SNP": sid, "GENE": cons.get("gene_symbol", cons.get("gene_id", "N/A")), "CONSEQUENCE": ", ".join(cons.get("consequence_terms", ["N/A"])), "IMPACT": cons.get("impact", "N/A"),
                                    "AMINO_ACID_CHANGE": cons.get("amino_acids", "N/A"), "SIFT": cons.get("sift_prediction", "N/A"), "POLYPHEN": cons.get("polyphen_prediction", "N/A")})
                elif item.get("regulatory_feature_consequences"):
                    cons = item["regulatory_feature_consequences"][0]
                    results.append({"SNP": sid, "GENE": "Région régulatrice", "CONSEQUENCE": cons.get("consequence_terms", ["N/A"])[0], "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
                else: results.append({"SNP": sid, "GENE": "Intergenic/Non-codant", "CONSEQUENCE": item.get("most_severe_consequence", "N/A"), "IMPACT": "MODIFIER", "AMINO_ACID_CHANGE": "N/A", "SIFT": "N/A", "POLYPHEN": "N/A"})
        except Exception as e: st.warning(f"⚠️ VEP réseau : {e}")
    reg = [(s, v) for s, k, v in items if k == "region"]; ids = [(s, v) for s, k, v in items if k == "id"]
    for i in range(0, len(reg), batch):
        ch = reg[i:i + batch]; _post(f"https://rest.ensembl.org/vep/{species}/region?regulatory=1", {"variants": [v for _, v in ch]}, {v: s for s, v in ch}); time.sleep(0.4)
    for i in range(0, len(ids), batch):
        ch = ids[i:i + batch]; _post(f"https://rest.ensembl.org/vep/{species}/id?regulatory=1", {"ids": [v for _, v in ch]}, {v: s for s, v in ch}); time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_ncbi_gene_info(gene_list, taxon):
    results = []
    for gene in gene_list:
        url = f"https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{quote(str(gene))}/taxon/{taxon}"
        try:
            r = requests.get(url, timeout=30, headers={"Accept": "application/json"})
            reports = (r.json().get("reports") or []) if r.status_code == 200 else []
            gi = (reports[0].get("gene") or {}) if reports else {}
            if not gi: results.append({"GENE": gene, "Nom complet": "Non trouvé", "Chromosome": "N/A", "Résumé": "N/A"}); continue
            summ = gi.get("summary")
            if isinstance(summ, list): summ = " ".join(x.get("description", "") for x in summ if isinstance(x, dict))
            results.append({"GENE": gene, "Nom complet": gi.get("description", "N/A"), "Chromosome": (gi.get("chromosomes") or ["N/A"])[0], "Résumé": (str(summ)[:200] + "...") if summ else "N/A"})
        except Exception as e: results.append({"GENE": gene, "Nom complet": "Erreur API", "Chromosome": "N/A", "Résumé": str(e)})
        time.sleep(0.4)
    return pd.DataFrame(results)

def fetch_uniprot_id(gene_name, taxon):
    q = f"gene:{gene_name} AND organism_id:{taxon}"; url = f"https://rest.uniprot.org/uniprotkb/search?query={quote(q)}&format=json&size=1"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and r.json().get("results"): return r.json()["results"][0].get("primaryAccession")
    except Exception: pass
    return None

def fetch_alphafold_pdb(uniprot_id):
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}", timeout=30)
        if r.status_code == 200:
            recs = r.json(); rec = next((x for x in recs if x.get("uniprotAccession") == uniprot_id), recs[0] if recs else None)
            if rec and rec.get("pdbUrl"):
                pr = requests.get(rec["pdbUrl"], timeout=30)
                if pr.status_code == 200: return pr.text
    except Exception: pass
    return None

def render_3d_structure(pdb_data, style="cartoon", color_scheme="spectrum", highlight_res=None):
    sty = {style: ({"color": "spectrum"} if color_scheme == "spectrum" else {"colorscheme": color_scheme})}; hl = ""
    if highlight_res and str(highlight_res).strip().isdigit(): hl = f"viewer.addStyle({{resi: {int(highlight_res)}}}, {{stick: {{colorscheme: 'redCarbon'}}}});"
    html = f"""<div id="viewport" style="width:100%;height:500px;position:relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      let viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{backgroundColor: 'white'}});
      viewer.addModel({json.dumps(pdb_data)}, "pdb"); viewer.setStyle({{}}, {json.dumps(sty)}); {hl}
      viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html, height=520)

def fetch_ensembl_sequence(chrom, start, end, species):
    try:
        r = requests.get(f"https://rest.ensembl.org/sequence/region/{species}/{chrom}:{start}..{end}:1?content-type=text/plain", timeout=30)
        if r.status_code == 200: return r.text.strip().upper()
    except Exception: pass
    return None

def align_and_highlight(seq_ref, seq_a, seq_b, name_a="Allèle A", name_b="Allèle B"):
    if not HAS_BIO: return "<p>⚠️ Biopython non installé.</p>"
    al = Align.PairwiseAligner(); al.mode = "global"; al.match_score = 2; al.mismatch_score = -3; al.open_gap_score = -5; al.extend_gap_score = -2
    def fmt(seq, name, color):
        aln = al.align(seq_ref, seq)[0]; ref_aln, seq_aln = str(aln[0]), str(aln[1]); s = f"<b>{name:10}</b> | "
        for r, q in zip(ref_aln, seq_aln):
            if r == "-" or q == "-": s += f"<span style='color:orange;font-weight:bold;'>{q}</span>"
            elif r != q: s += f"<span style='color:red;font-weight:bold;'>{q}</span>"
            else: s += f"<span style='color:{color};'>{q}</span>"
        return s + "<br>", ref_aln
    ha, ref_aln = fmt(seq_a, name_a, "green"); hb, _ = fmt(seq_b, name_b, "blue")
    return ("<div style='font-family:monospace;font-size:16px;line-height:1.5;'>" + ha + f"<b>{'Référence':10}</b> | " + "".join(f"<span style='color:grey;'>{r}</span>" for r in ref_aln) + "<br>" + hb + "</div>")

def ortholog_lookup(chrom, bp, species, target="homo_sapiens"):
    """Remplace l'ancien « LiftOver » (endpoint /map invalide entre espèces) : gène chevauchant → orthologue humain via Ensembl Compara."""
    base = "https://rest.ensembl.org"; h = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        genes = []
        for w in (0, 50000):
            r = requests.get(f"{base}/overlap/region/{species}/{chrom}:{max(1, bp - w)}-{bp + w}?feature=gene", headers=h, timeout=30)
            genes = r.json() if r.status_code == 200 else []
            if genes: break
        if not genes: return None
        g = next((x for x in genes if x.get("biotype") == "protein_coding"), genes[0]); gid = g["id"]
        r = requests.get(f"{base}/homology/id/{species}/{gid}?target_species={target};type=orthologues", headers=h, timeout=60)
        data = r.json().get("data", []) if r.status_code == 200 else []
        homs = data[0].get("homologies", []) if data else []
        if not homs: return {"gene_src": g.get("external_name", gid), "gene_id": gid, "human_gene": None}
        t = homs[0]["target"]; r = requests.get(f"{base}/lookup/id/{t['id']}", headers=h, timeout=30); info = r.json() if r.status_code == 200 else {}
        return {"gene_src": g.get("external_name", gid), "gene_id": gid, "human_gene": info.get("display_name", t["id"]), "human_id": t["id"], "chr": info.get("seq_region_name"),
                "start": info.get("start"), "end": info.get("end"), "perc_id": t.get("perc_id"), "assembly": info.get("assembly_name")}
    except Exception: return None

def annotate_genes(snp_df, gff, wk=50):
    if gff is None or gff.empty: return pd.DataFrame()
    w = wk * 1000; res = []; gchr = gff["CHR"].astype(str).values
    for _, s in snp_df.iterrows():
        pos = int(s.get("BP", s.get("START", 0))); ch = str(s["CHR"]); ng = gff[(gchr == ch) & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)]
        for _, g in ng.iterrows():
            d = 0 if g["START"] <= pos <= g["END"] else min(abs(g["START"] - pos), abs(g["END"] - pos))
            res.append({"SNP": s.get("SNP", f"{ch}:{pos}"), "CHR": ch, "SNP_POS": pos, "GENE": g["GENE"], "GENE_START": g["START"], "GENE_END": g["END"], "DIST_KB": round(d / 1000.0, 2)})
    return pd.DataFrame(res)

def _snp_cm(snp_df_full, snp_id):
    try:
        row = snp_df_full[snp_df_full["SNP"].astype(str) == str(snp_id)]
        if row.empty: return None
        cm = row["CM"].iloc[0]; return float(cm) if pd.notna(cm) else None
    except Exception: return None

def _cm_window_to_bp(snp_df_full, chrom, center_cm, window_cm):
    sub = snp_df_full[snp_df_full["CHR"].astype(str) == str(chrom)].copy()
    if sub.empty or np.all(sub["CM"].values.astype(float) == 0): return None
    sub = sub.sort_values("CM").reset_index(drop=True); cm_s = sub["CM"].values.astype(float); bp_s = sub["BP"].values.astype(float)
    lo = max(cm_s.min(), center_cm - window_cm); hi = min(cm_s.max(), center_cm + window_cm)
    a, b = float(np.interp(lo, cm_s, bp_s)), float(np.interp(hi, cm_s, bp_s)); return int(min(a, b)), int(max(a, b))

def _bp_to_cm(snp_df_full, chrom, bp):
    sub = snp_df_full[snp_df_full["CHR"].astype(str) == str(chrom)].copy()
    if sub.empty or np.all(sub["CM"].values.astype(float) == 0): return None
    sub = sub.sort_values("BP").reset_index(drop=True); return float(np.interp(bp, sub["BP"].values.astype(float), sub["CM"].values.astype(float)))

def annotate_genes_cm(snp_target, snp_df_full, gff, window_cm=10, window_kb_fallback=100):
    if gff is None or gff.empty: return pd.DataFrame()
    results = []; has_cm = not np.all(snp_df_full["CM"].astype(float).values == 0); gchr = gff["CHR"].astype(str).values
    for _, s in snp_target.iterrows():
        chrom = str(s["CHR"]); snp_id = s.get("SNP", ""); bp_pos = int(s.get("BP", 0)); cm_pos = s.get("CM", None)
        if cm_pos is None or pd.isna(cm_pos): cm_pos = _snp_cm(snp_df_full, snp_id)
        if (cm_pos is None or pd.isna(cm_pos)) and has_cm: cm_pos = _bp_to_cm(snp_df_full, chrom, bp_pos)
        bp_range = None; method = ""
        if has_cm and cm_pos is not None and not pd.isna(cm_pos):
            bp_range = _cm_window_to_bp(snp_df_full, chrom, float(cm_pos), window_cm)
            if bp_range is not None: method = f"{window_cm} cM"
        if bp_range is None: w = window_kb_fallback * 1000; bp_range = (bp_pos - w, bp_pos + w); method = f"±{window_kb_fallback} kb"
        ng = gff[(gchr == chrom) & (gff["END"] >= bp_range[0]) & (gff["START"] <= bp_range[1])]
        for _, g in ng.iterrows():
            dist_bp = 0 if g["START"] <= bp_pos <= g["END"] else min(abs(g["START"] - bp_pos), abs(g["END"] - bp_pos)); dist_cm = None
            if method.endswith("cM"):
                cs = _bp_to_cm(snp_df_full, chrom, bp_pos); cg = _bp_to_cm(snp_df_full, chrom, (g["START"] + g["END"]) / 2)
                if cs is not None and cg is not None: dist_cm = abs(cg - cs)
            results.append({"SNP": snp_id, "CHR": chrom, "SNP_POS_BP": bp_pos, "SNP_POS_CM": round(float(cm_pos), 2) if cm_pos is not None and not pd.isna(cm_pos) else None, "GENE": g["GENE"],
                            "GENE_START": int(g["START"]), "GENE_END": int(g["END"]), "DIST_KB": round(dist_bp / 1000.0, 2), "DIST_CM": round(dist_cm, 2) if dist_cm is not None else None, "WINDOW": method})
    df = pd.DataFrame(results)
    if not df.empty: df = df.sort_values(["SNP", "DIST_CM" if has_cm else "DIST_KB"]).reset_index(drop=True)
    return df

def run_enrichment(gene_list, gene_sets=None, organism="Human", outdir=None):
    try:
        import gseapy as gp
        genes = list({str(g).strip() for g in gene_list if str(g).strip() and str(g) != "nan"})
        if len(genes) < 3: return pd.DataFrame()
        if gene_sets is None: gene_sets = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
        enr = gp.enrichr(gene_list=genes, gene_sets=gene_sets, organism=organism, outdir=outdir, cutoff=1.0)
        if enr is None or enr.results is None: return pd.DataFrame()
        df = enr.results.copy()
        for old, new in {"Term": "TERM", "Adjusted P-value": "FDR", "P-value": "PVAL", "Overlap": "OVERLAP", "Genes": "GENES", "Combined Score": "SCORE"}.items():
            if old in df.columns: df = df.rename(columns={old: new})
        return df.sort_values("FDR").reset_index(drop=True) if "FDR" in df.columns else df
    except Exception as e:
        st.error(f"❌ Enrichr : {e}"); return pd.DataFrame()

def query_string_network(genes, species, required_score=400):
    if not genes or len(genes) < 2: return pd.DataFrame(), pd.DataFrame()
    ids = "%0d".join([quote(str(g).strip()) for g in genes if str(g).strip()])
    url = f"https://string-db.org/api/json/network?identifiers={ids}&species={species}&required_score={required_score}&caller_identity=livestock_genomics_pipeline"
    try:
        r = requests.get(url, timeout=60); r.raise_for_status(); data = r.json()
        if not data: return pd.DataFrame(), pd.DataFrame()
        edges = pd.DataFrame(data)[["preferredName_A", "preferredName_B", "score"]].drop_duplicates(); edges.columns = ["SOURCE", "TARGET", "SCORE"]
        return edges, pd.DataFrame({"ID": sorted(set(edges["SOURCE"]).union(set(edges["TARGET"])))})
    except Exception: return pd.DataFrame(), pd.DataFrame()

# ============================================================
# EXPORTS & GRAPHIQUES
# ============================================================
def _alleles_for_export(snp_df, m):
    a_minor = snp_df["A2"].astype(str).values[:m] if "A2" in snp_df.columns else np.array(["G"] * m)
    a_major = snp_df["A1"].astype(str).values[:m] if "A1" in snp_df.columns else np.array(["A"] * m)
    return a_minor, a_major

def build_plink(gt, ind_df, snp_df, prefix="genome"):
    """BED SNP-major vectorisé. Allèle compté (A1 du .bim) = allèle minoritaire = A2 de l'application."""
    n, m = gt.shape; nb = (n + 3) // 4; code = np.full(gt.shape, 0b01, dtype=np.uint8)
    code[gt == 0] = 0b11; code[gt == 1] = 0b10; code[gt == 2] = 0b00
    code = np.vstack([code, np.zeros((nb * 4 - n, m), dtype=np.uint8)]).reshape(nb, 4, m)
    packed = code[:, 0, :] | (code[:, 1, :] << 2) | (code[:, 2, :] << 4) | (code[:, 3, :] << 6)
    a1, a2 = _alleles_for_export(snp_df, m); clean = lambda a: np.where(np.isin(a, ["", ".", "nan", "None"]), "0", a)
    bim = pd.DataFrame({"c": snp_df["CHR"].values[:m], "s": snp_df["SNP"].values[:m], "cm": snp_df["CM"].values[:m], "bp": snp_df["BP"].astype(int).values[:m], "a1": clean(a1), "a2": clean(a2)})
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", bytes([0x6C, 0x1B, 0x01]) + np.ascontiguousarray(packed.T).tobytes())
        zf.writestr(f"{prefix}.bim", bim.to_csv(sep="\t", header=False, index=False))
        zf.writestr(f"{prefix}.fam", "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9" for _, r in ind_df.iterrows()) + "\n")
    return bio.getvalue()

def build_vcf(gt, ind_df, snp_df, proj="Genome"):
    m = gt.shape[1]; alt, ref = _alleles_for_export(snp_df, m); acgt = np.array(["A", "C", "G", "T"])
    ref = np.where(np.isin(ref, acgt), ref, "A"); alt = np.where(np.isin(alt, acgt), alt, "G")
    h = ["##fileformat=VCFv4.2", f"##source={proj}", '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">', "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(ind_df["IID"].astype(str))]
    lut = np.array(["0/0", "0/1", "1/1"]); strs = np.where(np.isnan(gt), "./.", lut[np.nan_to_num(gt, nan=0).astype(int).clip(0, 2)]); b = []
    for j in range(m):
        r = snp_df.iloc[j]; b.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref[j]}\t{alt[j]}\t.\tPASS\t.\tGT\t" + "\t".join(strs[:, j]))
    return "\n".join(h + b) + "\n"

def plot_missing(mi, ms):
    f = make_subplots(rows=1, cols=2, subplot_titles=("Missing/Ind", "Missing/SNP")); f.add_trace(go.Histogram(x=mi, nbinsx=60, marker_color="skyblue"), 1, 1); f.add_trace(go.Histogram(x=ms, nbinsx=60, marker_color="coral"), 1, 2)
    f.update_layout(height=400, showlegend=False); return f

def plot_pca(s, v, l):
    f = px.scatter(pd.DataFrame({"PC1": s[:, 0], "PC2": s[:, 1], "Pop": l}), x="PC1", y="PC2", color="Pop", title=f"PCA PC1({v[0]:.1f}%) vs PC2({v[1]:.1f}%)", height=550)
    f.update_traces(marker=dict(size=10, line=dict(width=1, color="white"))); return f

def plot_mds(m, l):
    f = px.scatter(pd.DataFrame({"MDS1": m[:, 0], "MDS2": m[:, 1], "Pop": list(l)}), x="MDS1", y="MDS2", color="Pop", title="MDS (distance IBS)", height=520)
    f.update_traces(marker=dict(size=11, line=dict(width=1, color="white"))); return f

def _cum_positions(df, poscol):
    df = df.assign(_k=df["CHR"].map(_chk_num), _s=df["CHR"].astype(str)).sort_values(["_k", "_s", poscol], kind="stable").reset_index(drop=True); offs = {}; ticks = []; cur = 0.0
    for c, sub in df.groupby("CHR", sort=False):
        mx = float(sub[poscol].max()) / 1e6; offs[c] = cur; ticks.append((str(c), cur + mx / 2)); cur += mx + 5
    df["_x"] = df["CHR"].map(offs) + df[poscol].astype(float) / 1e6; return df, ticks

def plot_manhattan(df, yc="P", title="", thr=None):
    df = df.copy(); ylab = yc
    if yc in ("P", "P_GC"): df["_y"] = -np.log10(np.clip(df[yc], 1e-300, 1)); yc = "_y"; ylab = "-log10(P)"
    df = df[np.isfinite(df[yc])]
    if df.empty: return go.Figure()
    df, ticks = _cum_positions(df, "BP"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub[yc], mode="markers", marker=dict(size=5, color="#2c3e50" if i % 2 == 0 else "#7f8c8d"), text=sub["SNP"], hoverinfo="text+y", showlegend=False))
    if thr: fig.add_hline(y=thr, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis=dict(title="Chromosome", tickmode="array", tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]), yaxis_title=ylab, height=500); return fig

def plot_qq(pvals, title="QQ-plot"):
    p = np.sort(np.asarray(pvals)[np.isfinite(pvals)]); p = p[(p > 0) & (p <= 1)]; n = len(p)
    if n == 0: return go.Figure()
    exp = -np.log10((np.arange(1, n + 1) - 0.5) / n); obs = -np.log10(p)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=exp, y=obs, mode="markers", marker=dict(size=4), name="obs")); fig.add_trace(go.Scatter(x=[0, exp.max()], y=[0, exp.max()], mode="lines", line=dict(color="red", dash="dash"), name="attendu"))
    fig.update_layout(title=title, xaxis_title="-log10(P) attendu", yaxis_title="-log10(P) observé", height=420); return fig

def plot_ld(ld):
    if ld is None or ld.empty: return None
    return px.line(ld.sort_values("dist_kb"), x="dist_kb", y="r2", markers=True, title="Décroissance du LD (r² moyen)", labels={"dist_kb": "Distance (kb)", "r2": "r²"}, height=450)

def plot_admix(Q, labs, pops, K):
    df = pd.DataFrame(Q, columns=[f"K{k + 1}" for k in range(K)]); df["IID"] = labs; df["Pop"] = pops; df["_d"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pops, categories=sorted(set(pops))); df = df.sort_values(["_po", "_d"]).reset_index(drop=True); df["x"] = np.arange(len(df)); fig = go.Figure()
    for k in range(K): fig.add_trace(go.Bar(x=df["x"], y=df[f"K{k + 1}"], name=f"K{k + 1}", marker_color=px.colors.qualitative.Set2[k % 8]))
    fig.update_layout(barmode="stack", title=f"Admixture (NMF) K={K}", height=500); return fig, df

def plot_gblup(df):
    fig = px.scatter(df, x="Phenotype", y="GEBV", title="GBLUP"); fig.update_traces(marker=dict(size=10, color="royalblue"))
    if len(df) > 1:
        try:
            z = np.polyfit(df["Phenotype"], df["GEBV"], 1); xr = np.linspace(df["Phenotype"].min(), df["Phenotype"].max(), 100); fig.add_trace(go.Scatter(x=xr, y=np.poly1d(z)(xr), mode="lines", name="OLS", line=dict(color="red", dash="dash")))
        except Exception: pass
    return fig

def plot_pip(df): return px.bar(df.head(20), x="SNP", y="PIP", title="PIP (probabilité d'inclusion a posteriori)", color="PIP", color_continuous_scale="Viridis")

def plot_win_manh(df, vc="MEAN_FST", title="", sigma=3.0):
    df = df[np.isfinite(df[vc])].copy()
    if df.empty: return None
    df, ticks = _cum_positions(df, "MID"); fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR", sort=False)):
        fig.add_trace(go.Scatter(x=sub["_x"], y=sub[vc], mode="markers", marker=dict(size=7, color="#2c3e50" if i % 2 == 0 else "#7f8c8d", line=dict(width=0.5, color="white")),
                                 text=[f"{c}:{int(m / 1e6)}Mb<br>{vc}={v:.4f}" for m, v in zip(sub["MID"], sub[vc])], hoverinfo="text", showlegend=False))
    mu, sd = df[vc].mean(), df[vc].std(); fig.add_hline(y=mu + sigma * sd, line_dash="dash", line_color="red", annotation_text=f"μ+{sigma}σ")
    fig.update_layout(title=title, xaxis=dict(title="Chromosome", tickmode="array", tickvals=[t[1] for t in ticks], ticktext=[t[0] for t in ticks]), yaxis_title=vc, height=520); return fig

def plot_gene_track(si, gff, wk=50):
    if gff is None or gff.empty: return None
    ch = str(si["CHR"]); pos = int(si.get("BP", si.get("START", 0))); w = wk * 1000; rg = gff[(gff["CHR"].astype(str) == ch) & (gff["START"] <= pos + w) & (gff["END"] >= pos - w)]; fig = go.Figure()
    for i, (_, g) in enumerate(rg.iterrows()): fig.add_trace(go.Scatter(x=[g["START"] / 1e6, g["END"] / 1e6], y=[i, i], mode="lines", line=dict(width=10, color="royalblue"), name=g["GENE"], text=[g["GENE"]] * 2, hoverinfo="text"))
    fig.add_vline(x=pos / 1e6, line_dash="dash", line_color="red", annotation_text=f"SNP: {si.get('SNP', '')}")
    fig.update_layout(title=f"Gènes CHR{ch}", xaxis_title="Pos (Mb)", yaxis=dict(showticklabels=False), height=400 + len(rg) * 20); return fig

def plot_string_network(edges, nodes):
    if edges is None or edges.empty: return None
    try: import networkx as nx
    except ImportError: st.error("⚠️ `networkx` non installé."); return None
    G = nx.Graph()
    for _, r in edges.iterrows(): G.add_edge(r["SOURCE"], r["TARGET"], weight=float(r["SCORE"]))
    pos = nx.spring_layout(G, seed=42, k=0.6); ex, ey = [], []
    for s, t in G.edges(): x0, y0 = pos[s]; x1, y1 = pos[t]; ex += [x0, x1, None]; ey += [y0, y1, None]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.6, color="#cccccc"), hoverinfo="skip", showlegend=False))
    deg = [d for _, d in G.degree()]
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], mode="markers+text", marker=dict(size=[8 + 3 * d for d in deg], color=deg, colorscale="Viridis", line=dict(width=1, color="white")), text=list(G.nodes()), textposition="top center", hoverinfo="text"))
    fig.update_layout(showlegend=False, height=600, xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False)); return fig

def plot_tree(Z, labels, support=None, title="", mid=None):
    n = len(labels); labels = [str(l) for l in labels]; lo = leaves_list(Z); ly = {int(l): i for i, l in enumerate(lo)}; cy = {i: ly[i] for i in range(n)}; cx = {i: 0.0 for i in range(n)}; fig = go.Figure()
    for k, row in enumerate(Z):
        i, j, d = int(row[0]), int(row[1]), float(row[2]); yi, yj = cy[i], cy[j]; xi, xj = cx[i], cx[j]
        if support is not None and k < len(support): s = support[k]; col = "#2ecc71" if s >= 70 else "#f39c12" if s >= 50 else "#e74c3c"
        else: col = "#3498db"
        for xs, ys in (([xi, d], [yi, yi]), ([xj, d], [yj, yj]), ([d, d], [yi, yj])): fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2), hoverinfo="skip", showlegend=False))
        cy[n + k] = (yi + yj) / 2; cx[n + k] = d
        if support is not None and k < len(support): fig.add_annotation(x=d, y=(yi + yj) / 2, text=f"<b>{int(support[k])}</b>", showarrow=False, font=dict(size=10, color=col), yshift=8, bgcolor="white", borderpad=2)
    for leaf, y in ly.items(): fig.add_annotation(x=0, y=y, text=f"<b>{labels[leaf]}</b>", showarrow=False, xanchor="right", xshift=-8, font=dict(size=12, color="#2c3e50"))
    if mid: fig.add_vline(x=mid["midpoint"], line_dash="dot", line_color="purple")
    fig.update_layout(title=title, xaxis_title="Distance de Reynolds", yaxis=dict(showticklabels=False, showgrid=False, zeroline=False), height=max(500, n * 35), margin=dict(l=120, r=40, t=80, b=40), plot_bgcolor="white"); return fig


# ============================================================
# STATISTIQUES POUR PUBLICATION (v23.6)
# ============================================================
@cache_data
def hwe_by_pop(gt, labels, min_n=10):
    """P-values HWE exactes (Wigginton 2005) calculées séparément dans chaque population (≥ min_n individus)."""
    labels = np.asarray(labels).astype(str); cols = {}
    for p in sorted(np.unique(labels)):
        m = labels == p
        if m.sum() < min_n: continue
        g = gt[m]; ok = ~np.isnan(g); n0 = ((g == 0) & ok).sum(0); n1 = ((g == 1) & ok).sum(0); n2 = ((g == 2) & ok).sum(0); pv = np.full(g.shape[1], np.nan)
        for j in range(g.shape[1]):
            if n0[j] + n1[j] + n2[j] >= 5: pv[j] = hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
        cols[p] = pv
    return pd.DataFrame(cols)

@cache_data
def private_alleles(gt, labels, min_n=3):
    """Allèles privés : présents dans une seule population (SNPs biallèles : A2 = allèle minoritaire, A1 = majoritaire)."""
    labels = np.asarray(labels).astype(str); pops = [p for p in sorted(np.unique(labels)) if (labels == p).sum() >= min_n]; F = []
    for p in pops:
        g = gt[labels == p]; n = (~np.isnan(g)).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"): F.append(np.where(n > 0, np.nansum(g, 0) / (2.0 * n), np.nan))
    F = np.array(F); rows = []
    for k, p in enumerate(pops):
        oth = np.delete(F, k, 0); fin = np.isfinite(F[k]) & (np.all(np.isfinite(oth), 0) if len(oth) else True)
        pm = fin & (F[k] > 0) & (np.all(oth == 0, 0) if len(oth) else False); pM = fin & (F[k] < 1) & (np.all(oth == 1, 0) if len(oth) else False)
        fr = np.concatenate([F[k][pm], 1 - F[k][pM]])
        rows.append({"Population": p, "N": int((labels == p).sum()), "Allèles privés A2": int(pm.sum()), "Allèles privés A1": int(pM.sum()), "Allèles privés (total)": int(pm.sum() + pM.sum()),
                     "Fréq. moyenne des privés": round(float(fr.mean()), 4) if len(fr) else np.nan})
    return pd.DataFrame(rows)

@cache_data
def diversity_extended(gt, labels, min_n=5, g_rare=None):
    """Na, Ae, Ar (raréfiée, Hurlbert/El Mousadik & Petit), π, θ_W (Watterson), Shannon H et équitabilité J — par population.
    π et θ_W sont exprimés PAR SNP typé (comparables entre populations de tailles différentes)."""
    labels = np.asarray(labels).astype(str); pops = [p for p in sorted(np.unique(labels)) if (labels == p).sum() >= min_n]; info = {}
    if not pops: return pd.DataFrame(), 0
    for p in pops:
        g = gt[labels == p]; n = (~np.isnan(g)).sum(0); N = 2 * n; k = np.nansum(g, 0); info[p] = (n, N, k)
    if g_rare is None: g_rare = int(min(np.percentile(info[p][1][info[p][1] > 0], 5) for p in pops))
    g_rare = max(2, int(g_rare)); maxN = int(max(info[p][1].max() for p in pops)); Hh = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, maxN + 1))]); rows = []
    for p in pops:
        n, N, k = info[p]; typed = N >= 2
        if not typed.any(): continue
        Nt, kt = N[typed].astype(float), k[typed]; pf = kt / Nt; q = 1 - pf; poly = (kt > 0) & (kt < Nt)
        na = 1.0 + poly; ae = 1.0 / (pf ** 2 + q ** 2); pi_s = 2 * pf * q * Nt / (Nt - 1); th_s = poly / Hh[(Nt - 1).astype(int)]
        with np.errstate(divide="ignore", invalid="ignore"): sh = -(np.where(pf > 0, pf * np.log(pf), 0.0) + np.where(q > 0, q * np.log(q), 0.0))
        okr = Nt >= g_rare
        if okr.any():
            Nr, kr = Nt[okr], kt[okr]
            def lc(a, b): return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)
            def miss(c): return np.where(Nr - c >= g_rare, np.exp(lc(Nr - c, g_rare) - lc(Nr, g_rare)), 0.0)
            ar = float(np.mean(2.0 - miss(kr) - miss(Nr - kr)))
        else: ar = np.nan
        rows.append({"Population": p, "N": int((labels == p).sum()), "Na": round(float(na.mean()), 4), "Ae": round(float(ae.mean()), 4), f"Ar (g={g_rare})": round(ar, 4),
                     "π (par SNP)": round(float(pi_s.mean()), 5), "θ_W (par SNP)": round(float(th_s.mean()), 5), "Shannon H": round(float(sh.mean()), 4), "Équitabilité J": round(float(sh.mean() / np.log(2)), 4)})
    return pd.DataFrame(rows), g_rare

@cache_data
def wc_fstats(gt, labels, n_boot=1000, seed=42):
    """Fis, Fst, Fit de Weir & Cockerham (1984) par locus et globaux (rapport des sommes), avec IC95 % par bootstrap sur les loci."""
    a, b, c = _wc_abc(gt, labels); den = a + b + c; ok = np.isfinite(den) & (den > 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"): fst = a / den; fis = b / (b + c); fit = (a + b) / den
    A, B, C, D = a[ok], b[ok], c[ok], den[ok]
    if len(A) == 0: return {"FST": fst, "FIS": fis, "FIT": fit}, {}
    gl = {"Fst": float(A.sum() / D.sum()), "Fis": float(B.sum() / (B + C).sum()), "Fit": float((A + B).sum() / D.sum())}
    gl["Fst_CI"] = _boot_ratio(A, D, n_boot, seed); gl["Fis_CI"] = _boot_ratio(B, B + C, n_boot, seed); gl["Fit_CI"] = _boot_ratio(A + B, D, n_boot, seed); gl["n_loci"] = int(ok.sum())
    return {"FST": fst, "FIS": fis, "FIT": fit}, gl

def _boot_ratio(num, den, n_boot=1000, seed=42):
    num = np.asarray(num, float); den = np.asarray(den, float); ok = np.isfinite(num) & np.isfinite(den); num, den = num[ok], den[ok]; m = len(num)
    if m < 10: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); v = []
    for s in range(0, n_boot, 100):
        idx = rng.integers(0, m, size=(min(100, n_boot - s), m)); v.append(num[idx].sum(1) / den[idx].sum(1))
    v = np.concatenate(v); return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def _wc_pair_a_den(n1, s1, h1, n2, s2, h2):
    """W&C 1984 pour r = 2 populations à partir de comptages (vectorisé sur SNPs × permutations)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        p1 = s1 / (2 * n1); p2 = s2 / (2 * n2); hh1 = h1 / n1; hh2 = h2 / n2; nsum = n1 + n2; nbar = nsum / 2.0; pbar = (n1 * p1 + n2 * p2) / nsum
        s2v = (n1 * (p1 - pbar) ** 2 + n2 * (p2 - pbar) ** 2) / nbar; hbar = (n1 * hh1 + n2 * hh2) / nsum; nc = 2 * nbar - (n1 ** 2 + n2 ** 2) / (2 * nbar)
        a = (nbar / nc) * (s2v - (1.0 / (nbar - 1)) * (pbar * (1 - pbar) - 0.5 * s2v - hbar / 4)); b = (nbar / (nbar - 1)) * (pbar * (1 - pbar) - 0.5 * s2v - (2 * nbar - 1) / (4 * nbar) * hbar); den = a + b + hbar / 2.0
    ok = (n1 >= 2) & (n2 >= 2) & np.isfinite(a) & np.isfinite(den) & (nbar > 1) & (nc > 0)
    return np.where(ok, a, np.nan), np.where(ok, den, np.nan)

@cache_data
def fst_pairwise_perm(gt, labels, n_perm=1000, seed=42, n_boot=1000):
    """FST pairwise (W&C 1984, rapport des moyennes) + p-value par permutation des individus + IC95 % bootstrap sur loci + corrections multi-tests."""
    labels = np.asarray(labels).astype(str); pops = sorted(np.unique(labels)); ok = ~np.isnan(gt)
    OK = ok.astype(np.float32); S = np.nan_to_num(gt).astype(np.float32); Hh = ((gt == 1) & ok).astype(np.float32); rng = np.random.default_rng(seed); rows = []
    for i in range(len(pops)):
        for j in range(i + 1, len(pops)):
            idx = np.where((labels == pops[i]) | (labels == pops[j]))[0]; z0 = (labels[idx] == pops[i]).astype(np.float32)
            if z0.sum() < 2 or (1 - z0).sum() < 2: continue
            O, Sg, Hg = OK[idx], S[idx], Hh[idx]; tn, ts, th = O.sum(0).astype(np.float64), Sg.sum(0).astype(np.float64), Hg.sum(0).astype(np.float64)
            def stat(Z):
                n1 = (O.T @ Z).astype(np.float64); s1 = (Sg.T @ Z).astype(np.float64); h1 = (Hg.T @ Z).astype(np.float64)
                return _wc_pair_a_den(n1, s1, h1, tn[:, None] - n1, ts[:, None] - s1, th[:, None] - h1)
            a, d = stat(z0[:, None]); a, d = a[:, 0], d[:, 0]; okk = np.isfinite(a) & np.isfinite(d)
            if not okk.any() or d[okk].sum() <= 0: continue
            obs = max(0.0, float(a[okk].sum() / d[okk].sum())); ci = _boot_ratio(a[okk], d[okk], n_boot, seed); cnt = 0
            for s in range(0, n_perm, 100):
                k = min(100, n_perm - s); Z = np.stack([rng.permutation(z0) for _ in range(k)], 1); ap, dp = stat(Z)
                with np.errstate(invalid="ignore", divide="ignore"): v = np.nansum(ap, 0) / np.nansum(dp, 0)
                cnt += int(np.sum(v >= obs - 1e-12))
            rows.append({"Pop1": pops[i], "Pop2": pops[j], "FST": round(obs, 5), "IC95_bas": round(max(0.0, ci[0]), 5), "IC95_haut": round(ci[1], 5), "P_perm": (cnt + 1) / (n_perm + 1)})
    df = pd.DataFrame(rows, columns=["Pop1", "Pop2", "FST", "IC95_bas", "IC95_haut", "P_perm"])
    if not df.empty: df["P_Bonferroni"] = np.minimum(1.0, df["P_perm"] * len(df)); df["P_BH"] = _bh(df["P_perm"].values)
    return df

def _ss_within(D2, codes):
    K = int(codes.max()) + 1; M = np.zeros((len(codes), K)); M[np.arange(len(codes)), codes] = 1.0; n = M.sum(0)
    return float((((D2 @ M) * M).sum(0) / (2.0 * np.maximum(n, 1))).sum())

def _amova_components(D2, pc, gop):
    """AMOVA à 3 niveaux (Excoffier, Smouse & Quattro 1992) sur matrice de distances² (déséquilibre géré via n, n', n'')."""
    N = len(pc); P = len(gop); G = int(gop.max()) + 1; grp = gop[pc]
    ss_t = float(D2.sum() / (2.0 * N)); ss_wp = _ss_within(D2, pc); ss_wg = _ss_within(D2, grp); ss_ag = ss_t - ss_wg; ss_ap = ss_wg - ss_wp
    df_ag, df_ap, df_wp = G - 1, P - G, N - P; ms_wp = ss_wp / df_wp; ms_ap = ss_ap / df_ap; ms_ag = ss_ag / df_ag
    n_p = np.bincount(pc, minlength=P).astype(float); Ng = np.bincount(gop, weights=n_p, minlength=G); A = (np.bincount(gop, weights=n_p ** 2, minlength=G) / Ng).sum(); Bv = (n_p ** 2).sum() / N
    n0 = (N - A) / (P - G); n1 = (A - Bv) / (G - 1); n2 = (N - (Ng ** 2).sum() / N) / (G - 1)
    s2c = ms_wp; s2b = (ms_ap - ms_wp) / n0; s2a = (ms_ag - ms_wp - n1 * s2b) / n2; tot = s2a + s2b + s2c
    return {"s2a": s2a, "s2b": s2b, "s2c": s2c, "tot": tot, "PhiCT": s2a / tot, "PhiSC": s2b / (s2b + s2c), "PhiST": (s2a + s2b) / tot,
            "SS": (ss_ag, ss_ap, ss_wp, ss_t), "df": (df_ag, df_ap, df_wp, N - 1), "MS": (ms_ag, ms_ap, ms_wp)}

@cache_data
def amova(gt, pops, groups, n_perm=999, seed=42):
    """AMOVA hiérarchique (groupes › races › individus) sur distances euclidiennes² de génotypes (0/1/2), tests de permutation d'Excoffier."""
    pops = np.asarray(pops).astype(str); groups = np.asarray(groups).astype(str); pu, pc = np.unique(pops, return_inverse=True); pc = np.asarray(pc).reshape(-1)
    gstr = [np.unique(groups[pops == p]) for p in pu]
    if any(len(x) != 1 for x in gstr): raise ValueError("Une race appartient à plusieurs groupes")
    gu, gop = np.unique([x[0] for x in gstr], return_inverse=True); gop = np.asarray(gop).reshape(-1)
    if len(gu) < 2: raise ValueError("Au moins 2 groupes requis")
    if len(pu) <= len(gu): raise ValueError("Il faut plusieurs races par groupe (races > groupes)")
    X = impute_mean(gt).astype(np.float64); sq = (X ** 2).sum(1); D2 = np.clip(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0, None); D2 /= X.shape[1]
    obs = _amova_components(D2, pc, gop); rng = np.random.default_rng(seed); c = {"PhiST": 0, "PhiCT": 0, "PhiSC": 0}; grp_ind = gop[pc]; idx_g = [np.where(grp_ind == g)[0] for g in range(len(gu))]
    for _ in range(n_perm):
        if _amova_components(D2, rng.permutation(pc), gop)["PhiST"] >= obs["PhiST"]: c["PhiST"] += 1
        if _amova_components(D2, pc, rng.permutation(gop))["PhiCT"] >= obs["PhiCT"]: c["PhiCT"] += 1
        pc3 = pc.copy()
        for ix in idx_g: pc3[ix] = rng.permutation(pc[ix])
        if _amova_components(D2, pc3, gop)["PhiSC"] >= obs["PhiSC"]: c["PhiSC"] += 1
    p = {k: (v + 1) / (n_perm + 1) for k, v in c.items()}
    tab = pd.DataFrame({"Source": ["Entre groupes", "Entre races (intra-groupe)", "Intra-race (entre individus)", "Total"], "ddl": obs["df"], "SC": np.round(obs["SS"], 4), "CM": list(np.round(obs["MS"], 4)) + [np.nan],
                        "Composante de variance": [obs["s2a"], obs["s2b"], obs["s2c"], obs["tot"]], "% variation": [100 * obs["s2a"] / obs["tot"], 100 * obs["s2b"] / obs["tot"], 100 * obs["s2c"] / obs["tot"], 100.0]}).round(5)
    return {"table": tab, "PhiCT": obs["PhiCT"], "PhiSC": obs["PhiSC"], "PhiST": obs["PhiST"], "p": p, "n_groups": len(gu), "n_pops": len(pu), "n_perm": n_perm}

def haversine_matrix(lat, lon):
    lat = np.radians(np.asarray(lat, float)); lon = np.radians(np.asarray(lon, float)); dlat = lat[:, None] - lat[None, :]; dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2; return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def mantel_test(A, B, n_perm=999, seed=42):
    A = np.asarray(A, float); B = np.asarray(B, float); iu = np.triu_indices_from(A, 1); a = A[iu]; b = B[iu]; r = float(np.corrcoef(a, b)[0, 1]); rng = np.random.default_rng(seed); n = A.shape[0]; cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if np.corrcoef(a, B[np.ix_(p, p)][iu])[0, 1] >= r: cnt += 1
    return r, (cnt + 1) / (n_perm + 1)

# ============================================================
# INTERFACE STREAMLIT
# ============================================================
_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt", "gt_pre", "ind_pre", "snp_pre", "gt_pruned", "snp_pruned", "qc_stats", "pca_s", "pca_v", "mds_c", "king_K", "fst", "fst_M", "fst_pops", "admix_Q", "admix_K", "cv_res", "ld_df", "roh_df", "froh", "ne_dict", "reyn_D", "reyn_pops",
         "pheno_df", "gwas_res", "gwas_pheno", "lambda_gc", "gebv_df", "gblup_stats", "ann_df", "_plink", "_vcf", "mapping_df", "fst_snp_df", "win_fst", "taj_df", "taj_race", "css_df", "pair_res", "pair_res_race", "pair_res_group", "pair_name_race", "pair_name_group",
         "nj_result", "nj_mid", "sel_gff", "sel_ann", "sel_ann_race", "sel_ann_group", "scan_df", "vep_res", "ncbi_gene_res", "lift_res", "div_df", "perm_res", "meta_res", "enr_res", "string_res", "pdb_gene", "pdb_data", "rf_res", "pub_div", "fst_perm", "fstats", "amova_res", "amova_crit", "mantel_res"]

def reset_derived(keep=("gt", "ind_df", "snp_df")):
    for k in _KEYS:
        if k not in keep: st.session_state[k] = None

def init_state():
    for k in _KEYS:
        if k not in st.session_state: st.session_state[k] = None

def work_data():
    """Données de travail : élaguées (LD) si disponibles, sinon filtrées QC."""
    ss = st.session_state
    gu = ss.gt_pruned if ss.gt_pruned is not None else ss.gt_filt; su = ss.snp_pruned if ss.snp_pruned is not None else ss.snp_filt
    return gu, su, ss.ind_filt

def show_pair(key, name_key, plot_key):
    r = st.session_state.get(key)
    if r is None: return
    nm = st.session_state.get(name_key, "")
    if r.empty: st.warning("Aucun SNP testable."); return
    nb = int((r["P"] < 0.05 / len(r)).sum())
    st.success(f"✅ {nm} : {len(r)} SNPs · {nb} Bonferroni · λGC = {lambda_gc(r['P'].values):.3f}")
    st.plotly_chart(plot_manhattan(r, "P", nm, thr=-np.log10(0.05 / len(r))), use_container_width=True, key=plot_key); st.dataframe(r.head(20), use_container_width=True)

def main():
    init_state(); ss = st.session_state

    with st.sidebar:
        st.title("🧬 Universal Genomics")
        species_choice = st.selectbox("🐾 Sélectionnez l'espèce", list(SPECIES_CONFIG.keys()))
        cfg = SPECIES_CONFIG[species_choice]; AUTO = cfg["autosomes"]; ARS = cfg["ars"]; TAXON = cfg["taxon"]; VEP_SPECIES = cfg["vep_species"]; PREFIX = cfg["prefix"]; GROUPES_DEFAULT = cfg["groupes"]
        if "current_species" not in ss: ss.current_species = species_choice
        elif ss.current_species != species_choice:
            ss.current_species = species_choice; reset_derived(keep=()); st.warning(f"🔄 Espèce changée pour {species_choice}. Données réinitialisées.")

        st.header("📁 Données"); mode = st.radio("Source", ["Demo", "PED/MAP", "VCF", "Axiom→PED"], key="src")
        if mode == "Demo":
            if st.button("🎲 Générer démo", use_container_width=True):
                gt, i, s = gen_demo(autosomes=AUTO, ars=ARS); reset_derived(keep=()); ss.update({"gt": gt, "ind_df": rec_fid(i), "snp_df": s}); st.success("✅ Démo générée")
        elif mode == "PED/MAP":
            ped = st.file_uploader(".ped", type=["ped", "gz"]); mp = st.file_uploader(".map", type=["map", "gz"])
            if ped and mp and st.button("📥 Charger"):
                try:
                    mdf = parse_map(mp.getvalue()); gt, ind, a1, a2 = parse_ped(ped.getvalue(), len(mdf))
                    mdf = mdf.iloc[:gt.shape[1]].reset_index(drop=True); gt = gt[:, :len(mdf)]; mdf["A1"] = a1[:len(mdf)]; mdf["A2"] = a2[:len(mdf)]
                    reset_derived(keep=()); ss.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": mdf}); st.success(f"✅ {gt.shape[0]}×{gt.shape[1]}")
                except Exception as e: st.error(f"❌ {e}")
        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf", "gz"])
            if vf and st.button("📥 Charger"):
                try:
                    gt, ind, snp = parse_vcf(vf.getvalue()); reset_derived(keep=()); ss.update({"gt": gt, "ind_df": rec_fid(ind), "snp_df": snp}); st.success(f"✅ {gt.shape[0]}×{gt.shape[1]}")
                except Exception as e: st.error(f"❌ {e}")
        else:
            af = st.file_uploader("Axiom brut", type=["txt", "csv", "tsv", "gz"])
            map_file = st.file_uploader("Fichier .map réel (optionnel mais très recommandé)", type=["map"])
            
            if af and st.button("🔄 Convertir"):
                try:
                    s, g, ns = parse_axiom(af.getvalue(), af.name)
                    
                    if map_file:
                        mdf = parse_map(map_file.getvalue())
                        n_m = min(len(mdf), ns)
                        mpl = [f"{row['CHR']}\t{row['SNP']}\t{row['CM']}\t{row['BP']}" for _, row in mdf.iloc[:n_m].iterrows()]
                        st.info(f"✅ Utilisation de votre fichier .map réel ({n_m} SNPs). Tous les chromosomes sont conservés.")
                    else:
                        mpl = [f"1\tSNP_{i + 1:06d}\t0\t{i + 1}" for i in range(ns)]
                        st.warning("⚠️ La .map générée est factice (chr 1, positions 1..N) : chargez la vraie carte de la puce pour conserver tous les chromosomes.")
                        
                    pdl = ["\t".join([sample_fid_iid(smp)[0], sample_fid_iid(smp)[1], "0", "0", "0", "-9"] + [a for pr in row for a in pr]) for smp, row in zip(s, g)]
                    ss["_plink_ped"] = "\n".join(pdl) + "\n"; ss["_plink_map"] = "\n".join(mpl) + "\n"; st.success(f"✅ {len(s)}×{ns}")
                except Exception as e: st.error(f"❌ {e}")
            if ss.get("_plink_ped"): st.download_button("⬇ .ped", ss["_plink_ped"].encode(), "out.ped"); st.download_button("⬇ .map", ss["_plink_map"].encode(), "out.map")

        st.divider(); st.header("⚙️ Seuils QC")
        keep_auto = st.checkbox("🧬 Exclure Chromosomes Sexuels (X, Y, MT)", value=True); rem_out = st.checkbox("🚫 Supprimer outliers", value=True)
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.05); mind = st.slider("Missing Ind", 0.0, 0.5, 0.05); maf_t = st.slider("MAF", 0.0, 0.5, 0.05)
        hwe_t = st.number_input("HWE (p min)", value=1e-6, min_value=0.0, max_value=1.0, step=1e-6, format="%.0e"); het_sd = st.slider("Het ±σ", 1.0, 5.0, 3.0, 0.1)
        hwe_pop = st.checkbox("HWE par population (recommandé)", value=True); hwe_k = st.slider("Exclure si HWE rejeté dans ≥ k populations", 1, 10, 2) if hwe_pop else 1
        ld_r2 = st.slider("LD r²", 0.05, 0.5, 0.2, 0.05); ld_w = st.slider("LD fenêtre (SNPs)", 20, 200, 50, 10)

    if ss.gt is None: st.info(f"👉 Chargez des données pour {species_choice} (Demo recommandé)."); return
    st.title(f"🐾 {species_choice} Genomics Pipeline v23.6")
    tabs = st.tabs(["🧹 QC", "✂️ LD", "🧬 Structure", "🎨 Admixture", "📈 Démographie", "🔍 Sélection", "🧪 GWAS Classique", "🎯 Post-GWAS", "📈 GBLUP", "🧬 Annotation", "🧬 VEP & NCBI", "🧬 Structure 3D", "🧬 Alignement", "📤 Export", "🔬 Méta-analyse", "🧠 ML", "🧬 Génotypes/blocs", "🌍 Orthologue humain", "📐 Stats publication"])
    NEED_QC = "⚠️ Lancez d'abord le QC (onglet 🧹)."

    # ---------------- QC ----------------
    with tabs[0]:
        if st.button("▶ Lancer QC", key="qc_btn"):
            try:
                p = {"geno": geno, "mind": mind, "maf": maf_t, "hwe": hwe_t, "het_sd": het_sd, "hwe_pop": hwe_pop, "hwe_k": hwe_k}
                gf, inf, snf, qs = apply_qc(ss.gt, ss.ind_df, ss.snp_df, p, AUTO, keep_autosomes=keep_auto, rem_out=rem_out)
                reset_derived(keep=("gt", "ind_df", "snp_df", "mapping_df")); ss.update({"gt_filt": gf, "ind_filt": inf, "snp_filt": snf, "qc_stats": qs}); st.success(f"✅ {gf.shape[0]}×{gf.shape[1]} (les analyses en aval ont été réinitialisées)")
            except Exception as e: st.error(f"❌ {e}")
        if ss.qc_stats:
            s = ss.qc_stats; c1, c2, c3, c4 = st.columns(4); c1.metric("Individus", s["n_ind_final"], delta=-s["excluded_ind"]); c2.metric("SNPs", s["n_snp_final"], delta=-s["excluded_snp"])
            c3.metric("Individus avant outliers", s["n_ind_pre"]); c4.metric("Outliers (het)", s["trace"].get("het", 0)); st.caption(f"Détail des exclusions SNP/individus : {s['trace']}")
        st.plotly_chart(plot_missing(miss_ind(ss.gt), miss_snp(ss.gt)), use_container_width=True, key="plot_missing_qc"); st.caption("Histogrammes calculés sur les données brutes (avant QC).")

    # ---------------- LD ----------------
    with tabs[1]:
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            if st.button("▶ LD Pruning", key="ld_btn"):
                with st.spinner("Élagage LD par chromosome..."):
                    k = ld_pruning(ss.gt_filt, ss.snp_filt["CHR"].astype(str).values, int(ld_w), 5, ld_r2)
                ss.update({"gt_pruned": ss.gt_filt[:, k], "snp_pruned": ss.snp_filt.iloc[np.where(k)[0]].reset_index(drop=True), "pca_s": None, "pca_v": None, "mds_c": None, "king_K": None, "admix_Q": None}); st.success(f"✅ {ss.gt_pruned.shape[1]} SNPs conservés")
            if ss.gt_pruned is not None:
                c1, c2, c3 = st.columns(3); c1.metric("Avant", ss.gt_filt.shape[1]); c2.metric("Après", ss.gt_pruned.shape[1]); c3.metric("Réduction", f"{100 * (1 - ss.gt_pruned.shape[1] / ss.gt_filt.shape[1]):.1f}%")
            if st.button("▶ LD Decay (sur SNPs non élagués)", key="ld_decay_btn"):
                with st.spinner("Calcul du LD..."): ss.ld_df = ld_decay(ss.gt_filt, ss.snp_filt)
            if ss.ld_df is not None:
                f = plot_ld(ss.ld_df)
                if f: st.plotly_chart(f, use_container_width=True, key="plot_ld_decay")
                else: st.warning("Pas assez de paires de SNPs proches pour estimer le LD.")

    # ---------------- Structure ----------------
    with tabs[2]:
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            gu, su, iu = work_data()
            if st.button("▶ PCA + MDS + KING", key="struct_btn"):
                try:
                    with st.spinner("PCA, MDS, KING..."):
                        s_, v_ = pca_analysis(gu, 10); ss.update({"pca_s": s_, "pca_v": v_, "mds_c": mds_analysis(gu, 5), "king_K": king_mat(gu)})
                    st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if ss.pca_s is not None and len(ss.pca_s) == len(iu):
                st.plotly_chart(plot_pca(ss.pca_s, ss.pca_v, iu["FID"].values), use_container_width=True, key="plot_pca_tab"); st.plotly_chart(plot_mds(ss.mds_c, iu["FID"].values), use_container_width=True, key="plot_mds_tab")
                if ss.king_K is not None:
                    Kt = np.triu(ss.king_K, 1); st.info(f"Paires apparentées (KING φ > 0.177, 1er degré) : **{int((Kt > 0.177).sum())}** · 2e degré (φ > 0.088) : **{int((Kt > 0.088).sum())}**")
            st.divider(); st.subheader("📊 Indices de Diversité Génétique (Ho, He, PIC, Fis)"); st.markdown("Indices standards (Nei, 1973 ; Botstein et al., 1980).")
            if st.button("Calculer la Diversité", type="primary", key="div_btn"):
                try:
                    with st.spinner("Calcul des indices..."): ss.div_df = compute_diversity(gu, iu["FID"].astype(str).values)
                except Exception as e: st.error(f"❌ Erreur lors du calcul : {e}")
            if ss.div_df is not None and not ss.div_df.empty:
                d = ss.div_df; st.dataframe(d, use_container_width=True); st.download_button("⬇ Télécharger le tableau (CSV)", d.to_csv(index=False).encode("utf-8"), "tableau_diversite.csv", "text/csv")
                st.plotly_chart(px.bar(d, x="Population", y=["H_Observee (Ho)", "H_Attendue (He)"], barmode="group", title="Hétérozygotie Observée vs Attendue par Population", labels={"value": "Hétérozygotie", "variable": "Type"}), use_container_width=True, key="plot_div")
                st.info(f"**Moyennes :** He = {d['H_Attendue (He)'].mean():.4f} | Ho = {d['H_Observee (Ho)'].mean():.4f} | PIC = {d['PIC'].mean():.4f} | Fis = {d['Fis (Consanguinité)'].mean():.4f}")
            elif ss.div_df is not None: st.warning("Aucune population avec ≥ 3 individus.")

    # ---------------- Admixture ----------------
    with tabs[3]:
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            gu, su, iu = work_data(); st.subheader("📊 Choix de K par courbe d'erreur NMF"); c1, c2 = st.columns(2); kmin = c1.slider("K min", 2, 5, 2, key="kmin_sl"); kmax = c2.slider("K max", 4, 10, 8, key="kmax_sl")
            if st.button("📈 Évaluer tous les K", key="eval_k_btn"):
                with st.spinner("NMF pour chaque K..."): ss.cv_res = choose_best_k(gu, kmin, kmax)
            if ss.cv_res is not None: st.dataframe(ss.cv_res, use_container_width=True); st.plotly_chart(px.line(ss.cv_res, x="K", y="RMSE", markers=True, title="Erreur NMF par K (choisir l'elbow)"), use_container_width=True, key="plot_k_curve")
            st.divider(); K = st.slider("K choisi", 2, 10, 4, key="admix_K_slider"); st.caption("ℹ️ Approximation par NMF (pas un vrai modèle ADMIXTURE/STRUCTURE) : à valider avec ADMIXTURE pour publication.")
            if st.button("▶ Admixture NMF", key="admix_btn"):
                try:
                    with st.spinner("NMF..."): Q, _ = admixture_nmf(gu, K=K)
                    ss.update({"admix_Q": Q, "admix_K": K}); st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if ss.admix_Q is not None and len(ss.admix_Q) == len(iu):
                f, _ = plot_admix(ss.admix_Q, iu["IID"].values, iu["FID"].values, ss.admix_K); st.plotly_chart(f, use_container_width=True, key="plot_admix_tab")

    # ---------------- Démographie ----------------
    with tabs[4]:
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            c1, c2 = st.columns(2); min_snps = c1.number_input("SNPs min par ROH", 10, 200, 30, key="roh_ms"); min_kb = c2.number_input("Longueur min ROH (kb)", 100, 5000, 500, 100, key="roh_kb")
            st.caption("ROH et Ne sont calculés sur les SNPs filtrés QC mais NON élagués (l'élagage LD détruit le signal).")
            if st.button("▶ ROH + Ne", key="dem_btn"):
                try:
                    gf, sf, inf = ss.gt_filt, ss.snp_filt, ss.ind_filt
                    with st.spinner("Détection des ROH..."): roh, fr = detect_roh(gf, sf["CHR"].astype(str).values, sf["BP"].values, int(min_snps), float(min_kb))
                    ss.update({"roh_df": roh, "froh": fr}); ne = {}
                    with st.spinner("Ne historique..."):
                        for f in inf["FID"].unique():
                            m = (inf["FID"] == f).values
                            if m.sum() >= 5:
                                nd = ne_historical(gf[m], sf)
                                if not nd.empty: ne[str(f)] = nd
                    ss.ne_dict = ne; st.success(f"✅ {len(roh)} ROH")
                except Exception as e: st.error(f"❌ {e}")
            if ss.froh is not None and len(ss.froh) == len(ss.ind_filt):
                df = pd.DataFrame({"FROH": ss.froh, "Pop": ss.ind_filt["FID"].values}); st.plotly_chart(px.histogram(df, x="FROH", color="Pop", nbins=50, title="FROH"), use_container_width=True, key="plot_roh_hist")
                st.dataframe(df.groupby("Pop")["FROH"].agg(["count", "mean", "std", "max"]).round(4), use_container_width=True)
                if ss.roh_df is not None and not ss.roh_df.empty:
                    st.markdown("### 📏 ROH par classes de taille (moyenne par individu)"); roh = ss.roh_df; n_ind = len(ss.ind_filt); L_kb = sum(ARS[c] for c in AUTO) * 1000
                    col1, col2, col3 = st.columns(3)
                    for col, thr, lab in ((col1, 1500, "FROH > 1.5 Mb (Ancien)"), (col2, 5000, "FROH > 5 Mb (Récent)"), (col3, 10000, "FROH > 10 Mb (Très récent)")):
                        col.metric(lab, f"{roh[roh['length_kb'] > thr]['length_kb'].sum() / (L_kb * n_ind):.4f}")
            if ss.ne_dict:
                fig = go.Figure()
                for k, d in ss.ne_dict.items(): fig.add_trace(go.Scatter(x=d["GenAgo"], y=d["Ne"], mode="lines+markers", name=k))
                fig.update_layout(title="Ne historique (LD, Sved 1971)", xaxis_title="Générations passées", yaxis_title="Ne", xaxis_type="log", yaxis_type="log", height=450); st.plotly_chart(fig, use_container_width=True, key="plot_ne")

    # ---------------- Sélection ----------------
    with tabs[5]:
        st.subheader("🔍 Sélection & structure des populations")
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            gu, su, iu = work_data(); lab = iu["FID"].astype(str).values; use_wc = st.checkbox("Utiliser Weir & Cockerham (1984) — recommandé", value=True, key="use_wc84")
            if st.button("▶ FST pairwise + Reynolds", key="fst_btn"):
                try:
                    if len(set(lab)) < 2: raise ValueError("Au moins 2 populations (FID) requises")
                    with st.spinner("FST..."):
                        ss.fst = fst_snp_wc(gu, lab); ss.fst_snp_df = su.copy(); M, pops = fst_pairwise(gu, lab, use_wc84=use_wc); ss.update({"fst_M": M, "fst_pops": pops}); D, rp = reynolds(gu, lab); ss.update({"reyn_D": D, "reyn_pops": rp})
                    st.success("✅")
                except Exception as e: st.error(f"❌ {e}")
            if ss.fst is not None:
                sf = ss.fst_snp_df if ss.fst_snp_df is not None else su; nf = min(len(ss.fst), len(sf))
                dff = pd.DataFrame({"SNP": sf["SNP"].values[:nf], "CHR": sf["CHR"].values[:nf], "BP": sf["BP"].values[:nf], "FST": ss.fst[:nf]})
                st.plotly_chart(plot_manhattan(dff, yc="FST", title="FST par SNP (W&C 1984)"), use_container_width=True, key="plot_manh_fst_snp")
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.imshow(ss.fst_M, x=ss.fst_pops, y=ss.fst_pops, text_auto=".3f", color_continuous_scale="Viridis", title="FST pairwise (global)"), use_container_width=True, key="plot_fst_mat")
                c2.plotly_chart(px.imshow(ss.reyn_D, x=ss.reyn_pops, y=ss.reyn_pops, text_auto=".3f", color_continuous_scale="Viridis", title="Distance de Reynolds"), use_container_width=True, key="plot_reyn_mat")

            st.divider(); st.subheader("🌳 Arbre UPGMA (bootstrap)")
            if ss.reyn_D is not None:
                c1, c2 = st.columns(2); nb = c1.slider("Bootstrap", 10, 500, 100, 10, key="nj_boot"); ro = c2.selectbox("Root", ["Non-enraciné", "Midpoint"], key="nj_root")
                if st.button("🌲 Construire l'arbre", type="primary", key="nj_btn"):
                    try:
                        with st.spinner(f"UPGMA + {nb} boot..."): r = upgma_bootstrap(gu, su, lab, n_boot=int(nb)); ss.nj_result = r; ss.nj_mid = midpoint_info(r["D"], r["pops"])
                        st.success("✅")
                    except Exception as e: st.error(f"❌ {e}")
                if ss.nj_result is not None:
                    r = ss.nj_result; mp = ss.nj_mid; c1, c2, c3, c4 = st.columns(4); c1.metric("Races", len(r["pops"])); c2.metric("Nœuds", len(r["support"])); c3.metric("Supp≥70%", f"{(r['support'] >= 70).sum()}/{len(r['support'])}"); c4.metric("Dist max", f"{mp['max_dist']:.4f}" if mp else "N/A")
                    if mp: st.info(f"Midpoint entre **{mp['leaf_i']}** et **{mp['leaf_j']}**")
                    st.plotly_chart(plot_tree(r["Z"], r["pops"], r["support"], title="UPGMA (bootstrap)", mid=mp if ro == "Midpoint" else None), use_container_width=True, key="plot_nj_tree")
                    with st.expander("Newick"): st.code(r["newick"])

            st.divider(); st.subheader("🧬 GWAS case-control")
            covs_pair = ss.pca_s[:, :5] if (ss.pca_s is not None and len(ss.pca_s) == len(iu)) else None
            mode_gwas = st.radio("🎯 Mode", ["🏷️ Race × Race", "📊 Groupe × Groupe", "⚖️ Comparer"], horizontal=True, key="gwas_mode"); st.divider()
            if mode_gwas.startswith("🏷️"):
                races, _ = get_pairs(iu["FID"].astype(str).values); c1, c2 = st.columns(2); ra = c1.selectbox("Race A", races, index=0, key="pa"); rb = c2.selectbox("Race B", races, index=1 if len(races) > 1 else 0, key="pb")
                c1, c2 = st.columns(2); mtd = c1.selectbox("Méthode", ["allelic", "logistic"], key="pm"); mf = c2.slider("MAF", 0.0, 0.3, 0.05, key="pmf")
                if mtd == "allelic": st.caption("ℹ️ Le test allélique ignore la structure/parenté : λGC élevé possible ; préférez « logistic » (corrigé par 5 PCs).")
                if ra != rb and st.button(f"🚀 {ra} vs {rb}", type="primary", key="pr_btn"):
                    try:
                        with st.spinner(f"{ra} vs {rb}..."): ss["pair_res_race"] = gwas_pairwise(gu, su, iu["FID"].astype(str).values, ra, rb, float(mf), mtd, covs_pair); ss["pair_name_race"] = f"{ra} vs {rb}"
                    except Exception as e: st.error(f"❌ {e}")
                show_pair("pair_res_race", "pair_name_race", "plot_pair_race")
            elif mode_gwas.startswith("📊"):
                critere = st.selectbox("Critère", list(GROUPES_DEFAULT.keys()) + ["➕ CSV"], key="grp_critere"); group_map = None
                if critere == "➕ CSV":
                    f = st.file_uploader("CSV (RACE, GROUPE)", type=["csv"], key="grp_csv")
                    if f is not None:
                        mdf = pd.read_csv(io.BytesIO(f.getvalue())); group_map = dict(zip(mdf["RACE"].astype(str).str.upper(), mdf["GROUPE"].astype(str)))
                else: group_map = make_group_map(GROUPES_DEFAULT[critere])
                if group_map is None: st.info("Chargez un CSV avec les colonnes RACE et GROUPE.")
                else:
                    iu_g = apply_group_mapping(iu, group_map); groups = sorted([g for g in iu_g["FID_GROUPE"].unique() if g != "AUTRE"])
                    if len(groups) < 2: st.warning("Moins de 2 groupes reconnus : vérifiez que les FID correspondent aux races du critère.")
                    else:
                        c1, c2 = st.columns(2); ga = c1.selectbox("Groupe A", groups, index=0, key="grp_a"); gb = c2.selectbox("Groupe B", groups, index=1, key="grp_b"); mtd_g = c1.selectbox("Méthode", ["allelic", "logistic"], key="pm_grp"); mf_g = c2.slider("MAF", 0.0, 0.3, 0.05, key="pmf_grp")
                        nA = (iu_g["FID_GROUPE"] == ga).sum(); nB = (iu_g["FID_GROUPE"] == gb).sum(); rA = iu_g[iu_g["FID_GROUPE"] == ga]["FID"].nunique(); rB = iu_g[iu_g["FID_GROUPE"] == gb]["FID"].nunique()
                        st.info(f"📊 **{ga}** ({nA} ind, {rA} races) vs **{gb}** ({nB} ind, {rB} races)")
                        if ga != gb and st.button(f"🚀 {ga} vs {gb}", type="primary", key="pr_btn_grp"):
                            try:
                                with st.spinner(f"{ga} vs {gb}..."): ss["pair_res_group"] = gwas_pairwise_group(gu, su, iu_g["FID_GROUPE"].values, ga, gb, float(mf_g), mtd_g, covs_pair); ss["pair_name_group"] = f"{ga} vs {gb} ({critere})"
                            except Exception as e: st.error(f"❌ {e}")
                show_pair("pair_res_group", "pair_name_group", "plot_pair_group")
            else:
                res_r = ss.get("pair_res_race"); res_g = ss.get("pair_res_group")
                if res_r is None or res_g is None or res_r.empty or res_g.empty: st.warning("⚠️ Lancez RACE puis GROUPE pour comparer.")
                else:
                    c1, c2 = st.columns(2); c1.metric(f"🏷️ {ss.get('pair_name_race', 'Race')}", f"{len(res_r)} SNPs", delta=f"{(res_r['P'] < 0.05 / len(res_r)).sum()} Bonf"); c2.metric(f"📊 {ss.get('pair_name_group', 'Groupe')}", f"{len(res_g)} SNPs", delta=f"{(res_g['P'] < 0.05 / len(res_g)).sum()} Bonf")
                    common = set(res_r["SNP"]).intersection(set(res_g["SNP"])); st.metric("SNPs communs", len(common))
                    if common:
                        dfc = res_r[res_r["SNP"].isin(common)][["SNP", "CHR", "BP", "FST_snp", "P"]].rename(columns={"P": "P_race", "FST_snp": "FST_race"})
                        dfc = dfc.merge(res_g[res_g["SNP"].isin(common)][["SNP", "P", "FST_snp"]].rename(columns={"P": "P_group", "FST_snp": "FST_group"}), on="SNP"); st.dataframe(dfc, use_container_width=True); st.download_button("⬇ SNPs communs", dfc.to_csv(index=False).encode(), "snps_communs.csv", "text/csv")

            st.divider(); st.subheader("🧬 Signatures de sélection")
            ms = st.radio("Méthode", ["8.3.1 — FST par fenêtres", "8.3.2 — Tajima's D", "8.3.3 — CSS"], horizontal=True, key="sel_m")
            if ms.startswith("8.3.1"):
                c1, c2, c3 = st.columns(3); ws = c1.slider("Fenêtre", 10, 200, 50, key="wf"); sp = c2.slider("Pas", 5, 100, 25, key="wp"); zt = c3.slider("Seuil z", 2.0, 5.0, 3.0, 0.1, key="wz")
                if st.button("▶ Scanner FST", type="primary", key="wf_btn"):
                    with st.spinner("Scan FST..."): w = windowed_fst(gu, su, lab, ws, sp)
                    if w.empty: st.warning("Aucune fenêtre valide (fenêtre trop grande pour le nombre de SNPs par chromosome ?).")
                    else: ss.win_fst = detect_outliers(w, zt); st.success(f"✅ {len(w)} fenêtres · {int(ss.win_fst['OUTLIER'].sum())} outliers")
                if ss.get("win_fst") is not None:
                    w = ss.win_fst; fg = plot_win_manh(w, "MEAN_FST", "FST fenêtres (rapport des moyennes W&C)")
                    if fg: st.plotly_chart(fg, use_container_width=True, key="plot_win_fst")
                    st.dataframe(w.head(20)[["CHR", "START", "END", "N_SNP", "MEAN_FST", "MAX_FST", "Z_FST", "TOP_SNP"]], use_container_width=True)
            elif ms.startswith("8.3.2"):
                rs = st.selectbox("Race", sorted(np.unique(lab)), key="tr"); c1, c2 = st.columns(2); wt = c1.slider("Fenêtre", 10, 100, 25, key="wt"); stp = c2.slider("Pas", 5, 50, 10, key="stp")
                st.caption("⚠️ Sur puce SNP (biais d'ascertainment), D n'a pas sa valeur absolue : repérez les fenêtres extrêmes en relatif.")
                if st.button(f"▶ Tajima's D — {rs}", type="primary", key="t_btn"):
                    m = lab == rs
                    if m.sum() < 5: st.error("⚠️ < 5 individus")
                    else:
                        with st.spinner("Tajima..."): t = tajimas_scan(gu[m], su, wt, stp); ss.taj_df = t; ss.taj_race = rs
                        st.success(f"✅ {len(t)} fenêtres")
                if ss.get("taj_df") is not None and not ss.taj_df.empty:
                    t = ss.taj_df; tv = st.selectbox("Variable à tracer", ["TAJIMA_D", "PI_PAR_SNP", "THETA_W_PAR_SNP", "PI_PAR_KB"], key="taj_var"); fg = plot_win_manh(t, tv, f"{tv} — {ss.get('taj_race', '')}")
                    if fg: st.plotly_chart(fg, use_container_width=True, key="plot_tajima")
                    st.dataframe(t.sort_values("TAJIMA_D").head(20), use_container_width=True)
            else:
                if st.button("▶ CSS", type="primary", key="css_btn"):
                    with st.spinner("CSS..."): ss.css_df = composite_score(gu, su, lab)
                    st.success(f"✅ {len(ss.css_df)} SNPs")
                if ss.get("css_df") is not None:
                    c = ss.css_df; st.plotly_chart(plot_manhattan(c.dropna(subset=["CSS"]), yc="CSS", title="Composite Selection Signal"), use_container_width=True, key="plot_css_manh"); st.dataframe(c.head(20), use_container_width=True)

            st.divider(); st.subheader("🧬 Annotation des gènes (cM ou kb)")
            snp_full = su; has_cm_data = (snp_full is not None and "CM" in snp_full.columns and not np.all(snp_full["CM"].astype(float).values == 0))
            if has_cm_data: st.success("✅ cM valides détectées — annotation en cM disponible")
            else: st.warning("⚠️ cM=0 → fallback kb")
            gff_f = st.file_uploader("GFF3/GTF", type=["gff", "gff3", "gtf", "gz"], key="sel_gff_file"); sources = []
            if ss.get("win_fst") is not None: sources.append("FST outliers")
            if ss.get("taj_df") is not None and not ss.taj_df.empty: sources.append("Tajima extrêmes")
            if ss.get("css_df") is not None: sources.append("CSS top")
            if ss.get("pair_res_race") is not None and not ss.pair_res_race.empty: sources.append("Race Pairwise")
            if ss.get("pair_res_group") is not None and not ss.pair_res_group.empty: sources.append("Group Pairwise")
            sources.append("Manuel")
            if gff_f is not None:
                try:
                    gff = load_gff(gff_f.getvalue()); ss.sel_gff = gff
                    if gff.empty: st.error("Aucun gène lu dans ce fichier (type 'gene' attendu)."); raise ValueError("GFF vide")
                    st.success(f"✅ {len(gff)} gènes"); c1, c2 = st.columns(2); metric = c1.radio("Unité", ["cM (génétique)", "kb (physique)"], index=0 if has_cm_data else 1, horizontal=True, key="ann_metric")
                    if metric.startswith("cM"): window_cm = c2.slider("Fenêtre cM", 1, 50, 10, 1); wk_fallback = 100
                    else: wk_fallback = c2.slider("Fenêtre kb", 1, 500, 50, 5); window_cm = 10
                    src = st.selectbox("Source", sources, key="ann_src"); to_ann = None
                    if src == "FST outliers": to_ann = ss.win_fst.head(50)[["CHR", "MID", "TOP_SNP"]].rename(columns={"MID": "BP", "TOP_SNP": "SNP"})
                    elif src == "Tajima extrêmes": ex = ss.taj_df.sort_values("TAJIMA_D").head(20); to_ann = ex[["CHR", "MID"]].rename(columns={"MID": "BP"}); to_ann["SNP"] = [f"Taj_{i + 1}" for i in range(len(to_ann))]
                    elif src == "CSS top": to_ann = ss.css_df.head(30)[["SNP", "CHR", "BP"]]
                    elif src in ("Race Pairwise", "Group Pairwise"): to_ann = ss["pair_res_race" if src == "Race Pairwise" else "pair_res_group"].head(30)[["SNP", "CHR", "BP"]]
                    else: c1, c2 = st.columns(2); chm = c1.text_input("Chr", "14", key="ann_mc"); pm = c2.number_input("Pos", 0, 200_000_000, 25_000_000, 100_000, key="ann_mp"); to_ann = pd.DataFrame({"SNP": [f"m_{chm}_{pm}"], "CHR": [str(chm)], "BP": [int(pm)]})
                    if to_ann is not None and len(to_ann) > 0 and st.button("🔍 Annoter", type="primary", key="ann_btn"):
                        with st.spinner("Annotation..."):
                            if metric.startswith("cM"): ann = annotate_genes_cm(to_ann, snp_full, gff, window_cm=window_cm, window_kb_fallback=wk_fallback)
                            else:
                                ann = annotate_genes(to_ann, gff, wk=wk_fallback)
                                if not ann.empty: ann["WINDOW"] = f"±{wk_fallback} kb"; ann["DIST_CM"] = None
                            ss.sel_ann = ann
                            if src == "Race Pairwise": ss.sel_ann_race = ann
                            elif src == "Group Pairwise": ss.sel_ann_group = ann
                        if ann.empty: st.warning("Aucun gène (vérifiez la correspondance des noms de chromosomes GFF ↔ SNPs)")
                        else: st.success(f"✅ {len(ann)} associations")
                    if ss.get("sel_ann") is not None and len(ss.sel_ann) > 0:
                        ann = ss.sel_ann; st.dataframe(ann, use_container_width=True); st.download_button("⬇ CSV", ann.to_csv(index=False).encode(), "sel_ann.csv", "text/csv")
                        sc = ann["SNP"].unique().tolist(); sel_s = st.selectbox("Visualiser", sc, key="ann_track"); row = ann[ann["SNP"] == sel_s].iloc[0].to_dict()
                        si = {"SNP": row["SNP"], "CHR": row["CHR"], "BP": row.get("SNP_POS_BP", row.get("SNP_POS", 0))}; f = plot_gene_track(si, gff, int(window_cm * 10) if metric.startswith("cM") else wk_fallback)
                        if f: st.plotly_chart(f, use_container_width=True, key="plot_gene_track")
                except Exception as e:
                    if str(e) != "GFF vide": st.error(f"❌ {e}")
            else: st.info("💡 [GFF3 Ensembl](https://ftp.ensembl.org/pub/current_gff3/)")

    # ---------------- GWAS classique ----------------
    with tabs[6]:
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            st.markdown("### 1️⃣ Mapping ID (optionnel)"); mf = st.file_uploader("CSV mapping (Animal + IID)", type=["csv", "txt"], key="gwas_map")
            if mf is not None:
                try: ss.mapping_df = pd.read_csv(io.BytesIO(mf.getvalue())); st.success("✅ Mapping chargé"); st.dataframe(ss.mapping_df.head(3))
                except Exception as e: st.error(f"❌ {e}")
            st.markdown("### 2️⃣ Phénotypes"); pf = st.file_uploader("Phénotypes", type=["csv", "txt", "gz"], key="gwas_pheno_file"); c1, c2 = st.columns(2)
            with c1:
                if pf and st.button("📥 Charger", key="load_ph_btn", use_container_width=True):
                    try:
                        mg, _ = match_pheno(ss.ind_filt, load_pheno(pf.getvalue(), pf.name), ss.mapping_df); ss.pheno_df = mg; tc = next((c for c in mg.columns if c not in ("FID", "IID", "Animal")), None)
                        n_ok = pd.to_numeric(mg[tc], errors="coerce").notna().sum() if tc else 0
                        if n_ok < 10: st.error(f"⚠️ {n_ok} individus appariés (min 10).")
                        else: st.success(f"✅ {n_ok} ind")
                    except Exception as e: st.error(f"❌ {e}")
            with c2:
                if st.button("🎲 Simuler", key="sim_ph_btn", use_container_width=True):
                    ind = ss.ind_filt; n = len(ind); rng = np.random.default_rng(42)
                    tr = (ss.pca_s[:, 0] * 2 + rng.normal(0, 0.5, n)) if (ss.pca_s is not None and len(ss.pca_s) == n) else rng.normal(0, 1, n)
                    ss.pheno_df = pd.DataFrame({"FID": ind["FID"].values, "IID": ind["IID"].values, "trait": tr}); st.success("✅")
            if ss.pheno_df is not None and len(ss.pheno_df) == len(ss.ind_filt):
                cols = [c for c in ss.pheno_df.columns if c not in ("FID", "IID", "Animal")]
                if cols:
                    pc = st.selectbox("Phénotype", cols, key="gw_pc"); ib = st.checkbox("Binaire (0/1)", key="gw_bin"); use_lmm = st.checkbox("Modèle mixte (GRM VanRaden)", value=True, key="gw_lmm"); ps = pd.to_numeric(ss.pheno_df[pc], errors="coerce"); nv = int(ps.notna().sum())
                    if nv < 10: st.error(f"❌ {nv} valides")
                    else:
                        st.info(f"✅ {nv} valides")
                        if st.button("🚀 Lancer GWAS", type="primary", key="gwas_btn"):
                            try:
                                gg, sg, _ = work_data(); cv = ss.pca_s[:, :5] if (ss.pca_s is not None and len(ss.pca_s) == gg.shape[0]) else None
                                with st.spinner("GWAS..."):
                                    Kg = grm_vanraden(gg) if (use_lmm and not ib) else None
                                    r = run_gwas(gg, ps.values, Kg, cv, use_lmm=use_lmm and not ib, is_bin=ib); r = annotate_gwas(r, sg); r = postprocess_gwas(r, apply_gc=True)
                                ss.gwas_res = r; ss.lambda_gc = lambda_gc(r["P"].values) if len(r) else np.nan
                            except Exception as e: st.error(f"❌ {e}")
            r = ss.gwas_res
            if r is not None and len(r):
                lam = ss.lambda_gc; st.success(f"✅ {len(r)} SNPs · λGC = {lam:.3f}" if np.isfinite(lam) else f"✅ {len(r)} SNPs")
                c1, c2, c3 = st.columns(3); c1.metric("SNPs FDR<0.05", int((r["FDR"] < 0.05).sum())); c2.metric("Bonferroni", int((r["P"] < 0.05 / len(r)).sum())); c3.metric("λGC", f"{lam:.3f}" if np.isfinite(lam) else "N/A")
                yc_choice = st.radio("Axe -log10 P", ["P", "P_GC"] if "P_GC" in r.columns else ["P"], horizontal=True, key="manh_yc")
                st.plotly_chart(plot_manhattan(r, yc=yc_choice, title="Manhattan", thr=-np.log10(0.05 / len(r))), use_container_width=True, key="plot_gwas_manh_classic"); st.plotly_chart(plot_qq(r["P"].values), use_container_width=True, key="plot_gwas_qq")
                show_cols = [c for c in ["SNP", "CHR", "BP", "MAF", "BETA", "SE", "T", "P", "FDR"] if c in r.columns]; st.dataframe(r[show_cols].head(20), use_container_width=True); st.download_button("⬇ CSV GWAS", r.to_csv(index=False).encode(), "gwas.csv", "text/csv")
            elif r is not None: st.warning("Aucun SNP testable (MAF/missing trop stricts ?).")

    # ---------------- Post-GWAS ----------------
    with tabs[7]:
        if ss.gwas_res is not None and len(ss.gwas_res):
            st.subheader("🎯 Fine-mapping (ABF de Wakefield / PIP)"); wmb = st.slider("Fenêtre autour du SNP principal (± Mb)", 0.1, 5.0, 1.0, 0.1, key="fm_win")
            pip = fine_mapping(ss.gwas_res, window_mb=wmb)
            if pip.empty: st.warning("Pas de SNP exploitable (BETA/SE manquants).")
            else:
                st.caption(f"{len(pip)} SNPs dans la fenêtre (CHR {pip['CHR'].iloc[0]}) · hypothèse : un seul variant causal.")
                st.dataframe(pip[["SNP", "CHR", "BP", "P", "PIP", "CUM_PIP"]].head(15), use_container_width=True); st.plotly_chart(plot_pip(pip), use_container_width=True, key="plot_pip_postgwas")
                st.info(f"Ensemble de crédibilité à 95 % : **{int((pip['CUM_PIP'].shift(fill_value=0) < 0.95).sum())}** SNP(s)")
        else: st.warning("Lancez GWAS d'abord")

    # ---------------- GBLUP ----------------
    with tabs[8]:
        if ss.pheno_df is not None and ss.gt_filt is not None and len(ss.pheno_df) == len(ss.ind_filt):
            cols = [c for c in ss.pheno_df.columns if c not in ("FID", "IID", "Animal")]; pc = st.selectbox("Phénotype GBLUP", cols, key="gb_pc")
            if st.button("▶ GBLUP + validation", key="gb_btn"):
                try:
                    gu, _, iu = work_data(); ps = pd.to_numeric(ss.pheno_df[pc], errors="coerce")
                    with st.spinner("ML + validation croisée..."): df, st_ = run_gblup(gu, ps.values, ids=iu["IID"].values)
                    ss.gebv_df = df; ss.gblup_stats = st_
                except Exception as e: st.error(f"❌ {e}")
            if ss.gebv_df is not None: st.plotly_chart(plot_gblup(ss.gebv_df), use_container_width=True, key="plot_gblup"); st.json(ss.gblup_stats)
        else: st.warning("Chargez les phénotypes (onglet GWAS Classique)")

    # ---------------- Annotation dédiée ----------------
    with tabs[9]:
        gff_f = st.file_uploader("GFF3/GTF", type=["gff", "gff3", "gtf", "gz"], key="ann_ded_gff"); src = st.radio("Source", ["Sélection", "GWAS"], horizontal=True, key="ann_ded_src"); wk = st.slider("Fenêtre kb", 1, 500, 50, key="ann_ded_wk")
        if gff_f:
            try:
                gff = load_gff(gff_f.getvalue()); st.success(f"✅ {len(gff)} gènes"); snps = None
                if src == "Sélection" and ss.win_fst is not None: snps = ss.win_fst.head(20).rename(columns={"MID": "BP", "TOP_SNP": "SNP"})[["SNP", "CHR", "BP"]]
                elif src == "GWAS" and ss.gwas_res is not None: snps = ss.gwas_res.sort_values("P").head(20)[["SNP", "CHR", "BP"]]
                if snps is not None and st.button("🔍 Annoter", key="ann_ded_btn"): ss.ann_df = annotate_genes(snps, gff, wk); st.success("✅")
                if ss.ann_df is not None: st.dataframe(ss.ann_df, use_container_width=True)
            except Exception as e: st.error(f"❌ {e}")

    # ---------------- VEP & NCBI ----------------
    with tabs[10]:
        st.subheader("🧬 Annotation Fonctionnelle (VEP & NCBI)"); sources_snp = []
        if ss.get("pair_res_group") is not None and not ss.pair_res_group.empty: sources_snp.append("Top SNPs Groupe")
        if ss.get("pair_res_race") is not None and not ss.pair_res_race.empty: sources_snp.append("Top SNPs Race")
        if ss.get("gwas_res") is not None and len(ss.gwas_res): sources_snp.append("Top SNPs GWAS")
        if ss.get("win_fst") is not None and not ss.win_fst.empty: sources_snp.append("Top SNPs FST")
        sources_snp.append("Saisie manuelle"); src_snp = st.selectbox("Source des SNPs", sources_snp, key="vep_src"); snps_to_analyze = []
        if src_snp == "Top SNPs Groupe": snps_to_analyze = ss.pair_res_group["SNP"].head(20).tolist()
        elif src_snp == "Top SNPs Race": snps_to_analyze = ss.pair_res_race["SNP"].head(20).tolist()
        elif src_snp == "Top SNPs GWAS": snps_to_analyze = ss.gwas_res.sort_values("P")["SNP"].head(20).tolist()
        elif src_snp == "Top SNPs FST": snps_to_analyze = ss.win_fst["TOP_SNP"].head(20).tolist()
        else:
            txt = st.text_area("Liste de SNPs (1/ligne)", key="vep_manual")
            if txt.strip(): snps_to_analyze = [s.strip() for s in txt.splitlines() if s.strip()]
        if snps_to_analyze:
            st.info(f"📊 {len(snps_to_analyze)} SNPs sélectionnés."); snp_full = ss.snp_filt if ss.snp_filt is not None else ss.snp_df; items = []; approx = 0; ACGT = ("A", "C", "G", "T")
            for s in snps_to_analyze:
                s = str(s)
                if s.lower().startswith("rs"): items.append((s, "id", s)); continue
                row = snp_full[snp_full["SNP"].astype(str) == s] if snp_full is not None else pd.DataFrame()
                if row.empty: continue
                r = row.iloc[0]; a1 = str(r.get("A1", ".")); a2 = str(r.get("A2", "."))
                if a1 not in ACGT or a2 not in ACGT: a1, a2 = "A", "G"; approx += 1
                items.append((s, "region", f"{r['CHR']} {int(r['BP'])} {int(r['BP'])} {a1}/{a2} 1"))
            if approx: st.caption(f"ℹ️ {approx} SNP(s) sans allèles A/C/G/T connus : conséquence positionnelle approximative (allèles A/G fictifs).")
            st.caption("ℹ️ Les coordonnées doivent correspondre à l'assemblage interrogé par Ensembl (ARS-UCD1.2 pour le bovin actuel) ; sinon convertissez d'abord.")
            if st.button("🔬 Lancer VEP", type="primary", key="run_vep_btn"):
                with st.spinner("Interrogation Ensembl VEP..."):
                    vep_res = run_vep_api(items, VEP_SPECIES)
                if vep_res.empty: st.warning("VEP n'a retourné aucun résultat.")
                else: vep_res["LOCALISATION"] = vep_res["CONSEQUENCE"].apply(traduire_consequence_vep); ss.vep_res = vep_res; st.success("✅ VEP terminé.")
            if ss.get("vep_res") is not None and not ss.vep_res.empty:
                st.dataframe(ss.vep_res[["SNP", "GENE", "LOCALISATION", "CONSEQUENCE", "IMPACT", "AMINO_ACID_CHANGE"]], use_container_width=True); st.download_button("⬇ CSV VEP", ss.vep_res.to_csv(index=False).encode(), "vep_results.csv", "text/csv")
            st.markdown("### NCBI Gene"); genes_to_fetch = []
            if ss.get("vep_res") is not None and "GENE" in ss.vep_res.columns: genes_to_fetch = ss.vep_res["GENE"].dropna().unique().tolist()
            elif ss.get("sel_ann") is not None and "GENE" in ss.sel_ann.columns: genes_to_fetch = ss.sel_ann["GENE"].dropna().unique().tolist()
            genes_to_fetch = [g for g in genes_to_fetch if str(g).strip() not in ("N/A", "Intergenic/Non-codant", "Région régulatrice", "Unknown", "nan", "")]
            if genes_to_fetch:
                st.info(f"🧬 {len(genes_to_fetch)} gènes valides.")
                if st.button("🏛️ Récupérer NCBI", type="primary", key="run_ncbi_btn"):
                    with st.spinner("NCBI Datasets..."): ss.ncbi_gene_res = fetch_ncbi_gene_info(genes_to_fetch[:20], TAXON)
                if ss.get("ncbi_gene_res") is not None and not ss.ncbi_gene_res.empty:
                    st.dataframe(ss.ncbi_gene_res, use_container_width=True); st.download_button("⬇ CSV NCBI", ss.ncbi_gene_res.to_csv(index=False).encode(), "ncbi_gene_info.csv", "text/csv")

    # ---------------- Structure 3D ----------------
    with tabs[11]:
        st.subheader("🧬 AlphaFold — Visualisation 3D"); genes_to_structure = []
        for key in ("vep_res", "sel_ann", "ann_df"):
            v = ss.get(key)
            if v is not None and hasattr(v, "columns") and "GENE" in v.columns and not v.empty:
                genes_to_structure = [g for g in v["GENE"].unique() if str(g).strip() not in ("N/A", "Intergenic/Non-codant", "Région régulatrice", "Unknown", "nan", "")]
                if genes_to_structure: break
        if not genes_to_structure: st.warning("⚠️ Lancez d'abord VEP ou Annotation.")
        else:
            c1, c2, c3 = st.columns(3); target_gene = c1.selectbox("Gène", genes_to_structure, key="gene_3d"); view_style = c2.selectbox("Style", ["cartoon", "stick", "sphere", "line"], key="style_3d"); color_scheme = c3.selectbox("Coloration", ["spectrum", "chainHetatm", "ssPyMol"], key="color_3d"); highlight_res = st.text_input("Surligner résidu (ex: 123)", key="hl_res")
            if st.button("🔬 Charger 3D", type="primary", key="btn_load_3d"):
                with st.spinner(f"Recherche pour {target_gene}..."):
                    uid = fetch_uniprot_id(target_gene, TAXON)
                    if uid:
                        pdb = fetch_alphafold_pdb(uid)
                        if pdb: ss.pdb_gene = target_gene; ss.pdb_data = pdb; st.success(f"✅ UniProt : {uid}")
                        else: st.error("❌ AlphaFold introuvable.")
                    else: st.error("❌ Gène non trouvé dans UniProt.")
            if ss.pdb_data and ss.pdb_gene == target_gene: st.info("💡 Cliquer-glisser pour tourner, molette pour zoomer."); render_3d_structure(ss.pdb_data, style=view_style, color_scheme=color_scheme, highlight_res=highlight_res)

    # ---------------- Alignement ----------------
    with tabs[12]:
        st.subheader("🧬 Illustration allélique autour d'un SNP")
        if not HAS_BIO: st.error("⚠️ Biopython requis : `pip install biopython`")
        else:
            cand = [("pair_res_group", ss.get("pair_res_group")), ("pair_res_race", ss.get("pair_res_race")), ("gwas_res", ss.get("gwas_res"))]
            snps_to_align = next((d.head(10) for _, d in cand if d is not None and len(d)), None)
            if snps_to_align is not None:
                st.caption("Séquence de référence Ensembl avec substitution de l'allèle du SNP (illustration, ce n'est PAS un alignement de séquences réelles de groupes).")
                snp_to_choose = st.selectbox("SNP à visualiser :", snps_to_align["SNP"].tolist())
                if st.button("🔍 Générer l'alignement", type="primary", key="aln_btn"):
                    snp_info = snps_to_align[snps_to_align["SNP"] == snp_to_choose].iloc[0]; chrom = str(snp_info["CHR"]); bp = int(snp_info["BP"]); start = max(1, bp - 25); end = bp + 25
                    with st.spinner(f"Téléchargement CHR{chrom}:{start}-{end}..."): seq_ref = fetch_ensembl_sequence(chrom, start, end, VEP_SPECIES)
                    if seq_ref and len(seq_ref) > bp - start:
                        pos = bp - start; ref_b = seq_ref[pos]; row = ss.snp_filt[ss.snp_filt["SNP"].astype(str) == str(snp_to_choose)]
                        al = [a for a in ((str(row.iloc[0].get("A1", ".")), str(row.iloc[0].get("A2", "."))) if not row.empty else ()) if a in ("A", "C", "G", "T") and a != ref_b]
                        alt_b = al[0] if al else ("A" if ref_b != "A" else "G"); sa = list(seq_ref); sa[pos] = alt_b
                        st.markdown(align_and_highlight(seq_ref, "".join(sa), seq_ref, f"Allèle {alt_b}", f"Allèle {ref_b}"), unsafe_allow_html=True)
                    else: st.error("❌ Ensembl n'a pas retourné la séquence.")
            else: st.warning("⚠️ Lancez d'abord un GWAS ou une analyse Groupe/Race.")

    # ---------------- Export ----------------
    with tabs[13]:
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            gu, su, iu = work_data()
            if st.button("📦 Préparer PLINK", key="pl_btn"):
                try: ss._plink = build_plink(gu, iu, su, PREFIX)
                except Exception as e: st.error(f"❌ {e}")
            if ss._plink: st.download_button("⬇ PLINK (.bed/.bim/.fam)", ss._plink, f"{PREFIX}.zip", "application/zip")
            if st.button("📄 Préparer VCF", key="vcf_btn"):
                try: ss._vcf = build_vcf(gu, iu, su, PREFIX).encode()
                except Exception as e: st.error(f"❌ {e}")
            if ss._vcf: st.download_button("⬇ VCF", ss._vcf, f"{PREFIX}.vcf", "text/plain")

    # ---------------- Méta-analyse ----------------
    with tabs[14]:
        st.subheader("🔬 Méta-analyse & Enrichissement"); st.markdown("## 1️⃣ Permutation GWAS")
        if ss.gt_filt is not None:
            gu_perm, su_perm, iu_perm = work_data(); races_perm, _ = get_pairs(iu_perm["FID"].astype(str).values)
            c1, c2 = st.columns(2); ra_p = c1.selectbox("Race A", races_perm, index=0, key="perm_ra"); rb_p = c2.selectbox("Race B", races_perm, index=1 if len(races_perm) > 1 else 0, key="perm_rb")
            c1, c2 = st.columns(2); n_perm = c1.slider("N permutations", 50, 1000, 100, 50, key="perm_n"); maf_p = c2.slider("MAF", 0.0, 0.3, 0.05, key="perm_maf")
            if ra_p != rb_p and st.button("🧪 Permutation test", type="primary", key="perm_btn"):
                try:
                    with st.spinner(f"{n_perm} permutations..."): ss.perm_res = permutation_test_race(gu_perm, su_perm, iu_perm["FID"].astype(str).values, ra_p, rb_p, n_perm=n_perm, maf_min=float(maf_p))
                except Exception as e: st.error(f"❌ {e}")
            if ss.perm_res is not None:
                orig, p_emp, n_obs, perm_counts, p_min = ss.perm_res
                if orig.empty: st.warning("Aucun SNP testé.")
                else:
                    c1, c2, c3 = st.columns(3); c1.metric("SNPs Bonferroni observés", n_obs); c2.metric("p empirique (nb SNPs)", f"{p_emp:.4f}"); c3.metric("p empirique (min P)", f"{p_min:.4f}")
                    st.plotly_chart(px.histogram(x=perm_counts, nbins=20, labels={"x": "Nb SNPs significatifs"}, title="Distribution nulle"), use_container_width=True, key="perm_hist")

        st.divider(); st.markdown("## 2️⃣ Combinaison P-values"); res_r = ss.get("pair_res_race"); res_g = ss.get("pair_res_group")
        if res_r is None or res_g is None or res_r.empty or res_g.empty: st.warning("⚠️ Lancez RACE puis GROUPE.")
        else:
            st.caption("⚠️ Les deux analyses réutilisent les mêmes individus : les p-values ne sont pas indépendantes, la p combinée est indicative (anti-conservative).")
            method = st.radio("Méthode", ["Fisher", "Stouffer pondéré"], horizontal=True, key="meta_meth")
            if st.button("🧪 Combiner", type="primary", key="meta_btn"):
                if method == "Fisher": ss.meta_res = combine_pvalues_fisher(res_r, res_g)
                else: ss.meta_res = combine_stouffer(res_r, res_g, int(res_r["n_A"].max() + res_r["n_B"].max()), int(res_g["n_A"].max() + res_g["n_B"].max()))
            if ss.meta_res is not None:
                fish = ss.meta_res; st.success(f"✅ {len(fish)} SNPs combinés"); st.dataframe(fish.head(30), use_container_width=True)
                st.plotly_chart(px.scatter(fish, x="P_race", y="P_group", hover_data=["SNP"], log_x=True, log_y=True, title="P_race vs P_group"), use_container_width=True, key="fisher_scatter"); st.download_button("⬇ CSV", fish.to_csv(index=False).encode(), "meta_analysis.csv", "text/csv")

        st.divider(); st.markdown("## 3️⃣ Enrichissement fonctionnel"); sources_genes = []
        if ss.get("sel_ann_race") is not None: sources_genes.append("Gènes SNPs Race")
        if ss.get("sel_ann_group") is not None: sources_genes.append("Gènes SNPs Groupe")
        if ss.get("sel_ann") is not None: sources_genes.append("Gènes annotés (sel_ann)")
        if ss.get("vep_res") is not None: sources_genes.append("Gènes VEP")
        sources_genes.append("Liste manuelle")
        c1, c2 = st.columns(2); src_g = c1.selectbox("Source gènes", sources_genes, key="enr_src"); outdir = c2.text_input("Dossier sortie", "", key="enr_outdir"); gmt_file = st.file_uploader("GMT (optionnel)", type=["gmt"], key="enr_gmt"); gene_sets = None
        if gmt_file is not None: gene_sets = parse_gmt(gmt_file.getvalue()); st.success(f"✅ {len(gene_sets)} gene sets")
        genes_to_enrich = None; src_df = {"Gènes SNPs Race": ss.get("sel_ann_race"), "Gènes SNPs Groupe": ss.get("sel_ann_group"), "Gènes annotés (sel_ann)": ss.get("sel_ann"), "Gènes VEP": ss.get("vep_res")}.get(src_g)
        if src_g == "Liste manuelle":
            txt = st.text_area("Liste de gènes (1/ligne)", key="enr_manual")
            if txt.strip(): genes_to_enrich = [g.strip() for g in txt.splitlines() if g.strip()]
        elif src_df is not None and "GENE" in src_df.columns: genes_to_enrich = src_df["GENE"].dropna().unique().tolist()
        if genes_to_enrich: genes_to_enrich = [g for g in genes_to_enrich if str(g).strip() not in ("N/A", "Intergenic/Non-codant", "Région régulatrice", "Unknown", "nan", "")]
        if genes_to_enrich and st.button("🔬 Enrichissement", type="primary", key="enr_btn"):
            with st.spinner("Enrichissement (orthologues humains)..."): ss.enr_res = run_enrichment(genes_to_enrich, gene_sets=gene_sets, organism="Human", outdir=outdir or None)
        if ss.enr_res is not None:
            if ss.enr_res.empty: st.warning("Aucun terme enrichi (≥ 3 gènes requis).")
            else: st.success(f"✅ {len(ss.enr_res)} termes"); st.dataframe(ss.enr_res.head(30), use_container_width=True); st.download_button("⬇ CSV", ss.enr_res.to_csv(index=False).encode(), "enrichment.csv", "text/csv")

        st.divider(); st.markdown("## 4️⃣ Réseau STRING"); genes_string = []
        if ss.get("sel_ann") is not None and "GENE" in ss.sel_ann.columns: genes_string = ss.sel_ann["GENE"].dropna().unique().tolist()
        st.info(f"Gènes disponibles : {len(genes_string)}"); c1, c2 = st.columns(2); score_str = c1.selectbox("Score STRING", [150, 400, 700, 900], index=1, key="string_score"); max_genes = c2.slider("Nb max", 5, 100, 30, key="string_max")
        if st.button("🕸️ Réseau STRING", type="primary", key="string_btn"):
            if len(genes_string) < 2: st.warning("Il faut ≥ 2 gènes.")
            else:
                with st.spinner("STRING..."): ss.string_res = query_string_network(genes_string[:max_genes], species=TAXON, required_score=int(score_str))
        if ss.string_res is not None:
            edges, nodes = ss.string_res
            if edges.empty: st.warning("Aucune interaction (espèce absente de STRING ?).")
            else:
                st.success(f"✅ {len(edges)} interactions"); fig = plot_string_network(edges, nodes)
                if fig: st.plotly_chart(fig, use_container_width=True, key="string_network_plot")
                st.dataframe(edges, use_container_width=True); st.download_button("⬇ CSV", edges.to_csv(index=False).encode(), "string_network.csv", "text/csv")

    # ---------------- ML ----------------
    with tabs[15]:
        st.subheader("🧠 Random Forest")
        if ss.gt_filt is not None and ss.get("pheno_df") is not None and len(ss.pheno_df) == len(ss.ind_filt):
            cols = [c for c in ss.pheno_df.columns if c not in ("FID", "IID", "Animal")]
            if cols:
                pc = st.selectbox("Phénotype", cols, key="ml_pc"); nmax = st.number_input("Nb max de SNPs (plus variables d'abord)", 500, 50000, 5000, 500, key="ml_nmax")
                if st.button("🚀 Entraîner", type="primary", key="ml_btn"):
                    try:
                        gu, sg, _ = work_data(); ps = pd.to_numeric(ss.pheno_df[pc], errors="coerce"); valid = np.isfinite(ps.values)
                        if valid.sum() < 20: st.error("❌ Pas assez d'individus.")
                        else:
                            X = impute_mean(gu[valid]); y = ps.values[valid]; top = np.argsort(-X.var(0))[:int(nmax)]; X = X[:, top]; names = sg["SNP"].values[top]
                            with st.spinner("Random Forest..."):
                                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42); rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
                                ss.rf_res = (float(r2_score(y_te, rf.predict(X_te))), pd.DataFrame({"SNP": names, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False).head(15))
                    except Exception as e: st.error(f"❌ {e}")
                if ss.rf_res is not None:
                    r2_rf, df_imp = ss.rf_res; st.success(f"✅ R² (RF, test 20 %) = {r2_rf:.3f}"); st.info("💡 Comparez avec le r² de validation croisée du GBLUP (métriques différentes : à interpréter avec prudence).")
                    fig = px.bar(df_imp, x="Importance", y="SNP", orientation="h", title="Top 15 SNPs"); fig.update_layout(yaxis={"categoryorder": "total ascending"}); st.plotly_chart(fig, use_container_width=True, key="plot_rf")
        else: st.warning("⚠️ Chargez données + phénotypes.")

    # ---------------- Génotypes / blocs ----------------
    with tabs[16]:
        st.subheader("🧬 Carte des génotypes par chromosome")
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            gu, su, iu = work_data(); chroms = sorted(su["CHR"].astype(str).unique(), key=_chk); sel_chr = st.selectbox("Chromosome", chroms, key="hap_chr"); sub_idx = su[su["CHR"].astype(str) == sel_chr].sort_values("BP").index
            if len(sub_idx) > 0:
                sub_idx = sub_idx[:50]; groups = iu["FID"].astype(str).values; order = np.argsort(groups, kind="stable")
                fig = px.imshow(gu[:, sub_idx][order], labels=dict(x="SNPs (ordre physique)", y="Individus", color="Génotype"), x=su.loc[sub_idx, "SNP"].values, y=groups[order], color_continuous_scale=["#F0F0F0", "#3498db", "#e74c3c"], title=f"Génotypes CHR{sel_chr} (50 premiers SNPs)", aspect="auto")
                st.plotly_chart(fig, use_container_width=True, key="plot_hap"); st.caption("🖱️ Génotypes NON phasés (0/1/2 copies de l'allèle minoritaire) : ce n'est pas une inférence de haplotypes.")
            else: st.warning("Aucun SNP sur ce chromosome.")

    # ---------------- Orthologue ----------------
    with tabs[17]:
        st.subheader("🌍 Gène chevauchant → orthologue humain (Ensembl Compara)")
        if ss.get("snp_filt") is not None:
            su = ss.snp_pruned if ss.snp_pruned is not None else ss.snp_filt; target_snp = st.selectbox("SNP à convertir", su["SNP"].tolist(), key="lift_snp")
            if st.button("🔄 Chercher", type="primary", key="lift_btn"):
                row = su[su["SNP"] == target_snp].iloc[0]
                with st.spinner(f"Recherche pour {row['CHR']}:{int(row['BP'])}..."): ss.lift_res = ortholog_lookup(str(row["CHR"]), int(row["BP"]), VEP_SPECIES)
                if ss.lift_res is None: st.error("❌ Aucun gène (±50 kb) ou erreur réseau.")
            res = ss.get("lift_res")
            if res:
                if not res.get("human_gene"): st.warning(f"Gène {res['gene_src']} trouvé, mais sans orthologue humain.")
                else: st.success(f"✅ {res['gene_src']} → **{res['human_gene']}** (humain, chr{res['chr']}:{res['start']}-{res['end']}, {res.get('assembly', '')}, identité {res.get('perc_id', 'N/A')} %)"); st.info("💡 Interroger le GWAS Catalog pour ce gène humain.")
        else: st.warning("Chargez des données.")

    # ---------------- Stats publication ----------------
    with tabs[18]:
        st.subheader("📐 Statistiques pour publication")
        if ss.gt_filt is None: st.info(NEED_QC)
        else:
            gf, sf, inf = ss.gt_filt, ss.snp_filt, ss.ind_filt; lab_all = inf["FID"].astype(str).values; gp_, sp_, ip_ = work_data(); lab_p = ip_["FID"].astype(str).values
            c1, c2 = st.columns(2); min_n = c1.number_input("N min par population", 3, 50, 5, key="pub_minn"); hwe_thr = c2.number_input("Seuil HWE par population", value=float(hwe_t), min_value=0.0, max_value=1.0, step=1e-6, format="%.0e", key="pub_hwe")
            st.caption("Diversité, allèles privés et HWE : tous les SNPs filtrés QC. FST, Fis/Fst/Fit, AMOVA, Mantel : SNPs élagués LD si disponibles (loci plus indépendants).")

            st.markdown("### 1️⃣ Tableau de diversité complet par population")
            if st.button("▶ Calculer le tableau complet", key="pub_div_btn", type="primary"):
                try:
                    with st.spinner("Ho/He/Fis, Na/Ae/Ar, π, θ_W, Shannon, allèles privés, HWE par population (peut prendre quelques minutes)..."):
                        base = compute_diversity(gf, lab_all); ext, g_used = diversity_extended(gf, lab_all, int(min_n)); priv = private_alleles(gf, lab_all, int(min_n)); pvd = hwe_by_pop(gf, lab_all, int(min_n))
                        hw = pd.DataFrame({"Population": pvd.columns, "SNPs testés (HWE)": pvd.notna().sum().values, "SNPs HWE<seuil": (pvd < hwe_thr).sum().values}); hw["% HWE<seuil"] = (100 * hw["SNPs HWE<seuil"] / hw["SNPs testés (HWE)"].clip(lower=1)).round(2)
                        tbl = base.merge(ext.drop(columns=["N"]), on="Population").merge(priv.drop(columns=["N"]), on="Population").merge(hw, on="Population")
                    ss.pub_div = (tbl, g_used)
                except Exception as e: st.error(f"❌ {e}")
            if ss.get("pub_div") is not None:
                tbl, g_used = ss.pub_div; st.dataframe(tbl, use_container_width=True); st.download_button("⬇ Tableau 1 (CSV)", tbl.to_csv(index=False).encode(), "tableau_diversite_complet.csv", "text/csv")
                st.caption(f"Ar : raréfaction à g = {g_used} gènes (même taille d'échantillon pour toutes les races). π et θ_W : par SNP typé, sans correction par la longueur : sur puce SNP (biais d'ascertainment), comparer les races entre elles, pas à des valeurs par pb de la littérature. "
                           "Les allèles privés dépendent de la taille d'échantillon : à interpréter avec Ar.")

            st.divider(); st.markdown("### 2️⃣ FST pairwise : p-values par permutation et IC95 %")
            c1, c2 = st.columns(2); nperm = c1.slider("Permutations", 100, 2000, 1000, 100, key="pub_nperm"); nboot = c2.slider("Bootstrap loci (IC)", 100, 2000, 1000, 100, key="pub_nboot")
            if st.button("▶ Lancer FST pairwise + permutations", key="pub_fst_btn", type="primary"):
                try:
                    with st.spinner(f"{nperm} permutations par paire..."): ss.fst_perm = fst_pairwise_perm(gp_, lab_p, int(nperm), 42, int(nboot))
                except Exception as e: st.error(f"❌ {e}")
            if ss.get("fst_perm") is not None:
                fp = ss.fst_perm
                if fp.empty: st.warning("Aucune paire exploitable (≥ 2 individus par population).")
                else:
                    st.dataframe(fp, use_container_width=True); st.download_button("⬇ FST pairwise (CSV)", fp.to_csv(index=False).encode(), "fst_pairwise_perm.csv", "text/csv")
                    pops_ = sorted(set(fp["Pop1"]) | set(fp["Pop2"])); Mx = pd.DataFrame(np.nan, index=pops_, columns=pops_)
                    for _, r in fp.iterrows(): Mx.loc[r["Pop1"], r["Pop2"]] = r["FST"]; Mx.loc[r["Pop2"], r["Pop1"]] = r["FST"]
                    st.plotly_chart(px.imshow(Mx, text_auto=".3f", color_continuous_scale="Viridis", title="FST pairwise (W&C 1984)"), use_container_width=True, key="plot_pub_fst")
                    st.caption(f"p minimale atteignable = 1/({int(nperm)}+1). Corrections : Bonferroni et Benjamini-Hochberg sur l'ensemble des {len(fp)} paires.")

            st.divider(); st.markdown("### 3️⃣ Fis / Fst / Fit de Weir & Cockerham par locus")
            if st.button("▶ Calculer Fis / Fst / Fit", key="pub_f_btn", type="primary"):
                try:
                    with st.spinner("W&C + bootstrap..."): ss.fstats = wc_fstats(gp_, lab_p, 1000)
                except Exception as e: st.error(f"❌ {e}")
            if ss.get("fstats") is not None:
                arr, gl = ss.fstats
                if not gl: st.warning("Pas assez de loci valides (≥ 2 populations avec ≥ 2 individus).")
                else:
                    c1, c2, c3 = st.columns(3)
                    for col, k in ((c1, "Fis"), (c2, "Fst"), (c3, "Fit")): col.metric(k, f"{gl[k]:.4f}", delta=f"IC95 [{gl[k + '_CI'][0]:.4f} ; {gl[k + '_CI'][1]:.4f}]", delta_color="off")
                    n_ = min(len(sp_), len(arr["FST"])); dl = pd.DataFrame({"SNP": sp_["SNP"].values[:n_], "CHR": sp_["CHR"].astype(str).values[:n_], "BP": sp_["BP"].values[:n_], "FIS": arr["FIS"][:n_], "FST": arr["FST"][:n_], "FIT": arr["FIT"][:n_]})
                    st.caption(f"{gl['n_loci']} loci · globaux = rapport des sommes · IC par bootstrap des loci (1000) · 1−Fit = (1−Fis)(1−Fst)")
                    st.plotly_chart(px.histogram(dl.dropna(subset=["FST"]), x="FST", nbins=60, title="Distribution des FST par locus"), use_container_width=True, key="plot_pub_fsth")
                    st.plotly_chart(plot_manhattan(dl.dropna(subset=["FST"]), yc="FST", title="FST par locus"), use_container_width=True, key="plot_pub_fstm"); st.download_button("⬇ Fis/Fst/Fit par locus (CSV)", dl.to_csv(index=False).encode(), "fis_fst_fit_par_locus.csv", "text/csv")

            st.divider(); st.markdown("### 4️⃣ AMOVA hiérarchique (groupes › races › individus)")
            c1, c2 = st.columns(2); crit = c1.selectbox("Critère de groupes", list(GROUPES_DEFAULT.keys()), key="pub_amova_crit"); nperm_a = c2.slider("Permutations", 99, 999, 999, 100, key="pub_amova_perm")
            if st.button("▶ Lancer l'AMOVA", key="pub_amova_btn", type="primary"):
                try:
                    ig = apply_group_mapping(ip_, make_group_map(GROUPES_DEFAULT[crit])); keep = (ig["FID_GROUPE"] != "AUTRE").values
                    if keep.sum() < 10: raise ValueError("Moins de 10 individus appartiennent aux groupes du critère")
                    with st.spinner("AMOVA + permutations..."): ss.amova_res = amova(gp_[keep], ig["FID"].values[keep], ig["FID_GROUPE"].values[keep], int(nperm_a))
                    ss.amova_crit = crit
                except Exception as e: st.error(f"❌ {e}")
            if ss.get("amova_res") is not None:
                a = ss.amova_res; st.dataframe(a["table"], use_container_width=True); p = a["p"]; c1, c2, c3 = st.columns(3)
                c1.metric("ΦCT (entre groupes)", f"{a['PhiCT']:.4f}", delta=f"p = {p['PhiCT']:.4f}", delta_color="off"); c2.metric("ΦSC (races dans groupes)", f"{a['PhiSC']:.4f}", delta=f"p = {p['PhiSC']:.4f}", delta_color="off"); c3.metric("ΦST (total)", f"{a['PhiST']:.4f}", delta=f"p = {p['PhiST']:.4f}", delta_color="off")
                st.caption(f"{a['n_groups']} groupes, {a['n_pops']} races, {a['n_perm']} permutations. Avec peu de races, le nombre de permutations distinctes de ΦCT est très limité (p minimale élevée). Distances euclidiennes² de génotypes (manquants imputés par la moyenne).")
                st.download_button("⬇ Tableau AMOVA (CSV)", a["table"].to_csv(index=False).encode(), "amova.csv", "text/csv")

            st.divider(); st.markdown("### 5️⃣ Isolement par la distance (IBD) + test de Mantel")
            cf = st.file_uploader("CSV coordonnées GPS par race (colonnes RACE, LAT, LON)", type=["csv"], key="pub_coords")
            c1, c2 = st.columns(2); tr_ = c1.radio("Distance géographique", ["ln(km)", "km"], horizontal=True, key="pub_ibd_tr"); nperm_m = c2.slider("Permutations Mantel", 99, 9999, 999, 100, key="pub_mantel_perm")
            if cf is not None and st.button("▶ Lancer IBD / Mantel", key="pub_mantel_btn", type="primary"):
                try:
                    co = pd.read_csv(io.BytesIO(cf.getvalue())); co["RACE"] = co["RACE"].astype(str).str.upper(); M, pops_m = fst_pairwise(gp_, lab_p, True); idx_m = {p.upper(): i for i, p in enumerate(pops_m)}
                    co = co[co["RACE"].isin(idx_m)].drop_duplicates("RACE")
                    if len(co) < 4: raise ValueError("Au moins 4 races avec coordonnées sont nécessaires")
                    ii = [idx_m[r] for r in co["RACE"]]; Fm = np.clip(M[np.ix_(ii, ii)], 0, 0.999); Gd = Fm / (1 - Fm); km = haversine_matrix(co["LAT"], co["LON"]); Xd = np.log(km + 1.0) if tr_ == "ln(km)" else km
                    r_, p_ = mantel_test(Gd, Xd, int(nperm_m)); iu = np.triu_indices(len(co), 1); ss.mantel_res = {"r": r_, "p": p_, "x": Xd[iu], "y": Gd[iu], "n": len(co), "lab": tr_}
                except Exception as e: st.error(f"❌ {e}")
            if ss.get("mantel_res") is not None:
                m = ss.mantel_res; st.metric("Corrélation de Mantel (r)", f"{m['r']:.4f}", delta=f"p = {m['p']:.4f} (unilatéral) · {m['n']} races", delta_color="off")
                fg = px.scatter(x=m["x"], y=m["y"], labels={"x": f"Distance géographique ({m['lab']})", "y": "FST/(1−FST)"}, title="Isolement par la distance")
                if len(m["x"]) > 1: z = np.polyfit(m["x"], m["y"], 1); xs = np.linspace(m["x"].min(), m["x"].max(), 50); fg.add_trace(go.Scatter(x=xs, y=np.poly1d(z)(xs), mode="lines", name="régression", line=dict(color="red", dash="dash")))
                st.plotly_chart(fg, use_container_width=True, key="plot_pub_ibd")

if __name__ == "__main__":
    main()
