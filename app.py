"""🐪 Camel Genomics Pipeline v20.5 — Correction de l'erreur Read-Only (Ne)"""
import gzip, io, os, warnings, time, json, hashlib, gc
from collections import Counter
from urllib.parse import quote
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy import stats
from scipy.stats import t as t_dist
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from sklearn.decomposition import NMF, PCA

import requests
import streamlit.components.v1 as components

try: from statsmodels.stats.multitest import multipletests; HAS_STATSMODELS = True
except ImportError: HAS_STATSMODELS = False

warnings.filterwarnings("ignore")

st.set_page_config(page_title="🐪 Camel Genomics v20.5", page_icon="🐪", layout="wide")

# ============================================================
# CONSTANTES
# ============================================================
ARS = {str(i): 100.0 for i in range(1, 37)}
AUTO = [str(i) for i in range(1, 37)]

# ============================================================
# UTILS
# ============================================================
def get_hash(*args):
    h = hashlib.md5()
    for arg in args:
        if isinstance(arg, np.ndarray):
            h.update(f"{arg.shape}_{np.nansum(arg)}".encode())
        elif isinstance(arg, pd.DataFrame):
            h.update(f"{arg.shape}_{pd.util.hash_pandas_object(arg).sum()}".encode())
        else:
            h.update(str(arg).encode())
    return h.hexdigest()

def impute_mean(gt):
    g = gt.astype(np.float32, copy=True)
    cm = np.nanmean(g, axis=0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(g)
    if nm.any(): g[nm] = np.take(cm, np.where(nm)[1])
    return g

def _chk(c):
    s = str(c).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    return (1, sp[s], "") if s in sp else (2, 0, s)

def _rt(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b": return gzip.decompress(raw).decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")

def rec_fid(ind):
    df = ind.copy()
    if df["FID"].nunique() <= 1: df["FID"] = df["IID"].astype(str).str.split("_").str[0]
    return df

def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def maf(g): p = np.nanmean(g, axis=0) / 2.0; return np.minimum(p, 1.0 - p)

# ============================================================
# PARSERS
# ============================================================
def parse_map(b):
    rows = []
    for line in _rt(b).splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        p = line.split()
        if len(p) >= 4 and p[0].upper() not in ("CHR", "CHROM"):
            try:
                cm = float(p[2]) if p[2] not in ("0", ".", "NA") else 0.0
                rows.append({"CHR": str(p[0]), "SNP": str(p[1]), "CM": cm, "BP": int(float(p[3]))})
            except: continue
    if not rows: raise ValueError("Fichier .map vide ou mal formaté.")
    return pd.DataFrame(rows)

def parse_ped(b, n_map):
    lines = [l.strip() for l in _rt(b).splitlines() if l.strip()]
    n_ped = (len(lines[0].split()) - 6) // 2
    n = min(n_map, n_ped)
    fids, iids, rows = [], [], []
    for line in lines:
        p = line.split()
        if len(p) >= 6 + 2*n:
            fids.append(p[0]); iids.append(p[1]); rows.append(p[6:6+2*n])
    ac = [Counter() for _ in range(n)]
    for r in rows:
        for j in range(n):
            if r[2*j] != "0": ac[j].update([r[2*j], r[2*j+1]])
    minor = [min(c, key=c.get) if len(c) >= 2 else None for c in ac]
    gt = np.full((len(iids), n), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        for j in range(n):
            if minor[j] and r[2*j] != "0":
                gt[i, j] = (r[2*j] == minor[j]) + (r[2*j+1] == minor[j])
    return gt, pd.DataFrame({"FID": fids, "IID": iids})

# ============================================================
# ALGORITHMES
# ============================================================
@st.cache_data(show_spinner=False)
def apply_qc(_gt, _ind_df, _snp_df, hash_key, geno, mind, maf_t, keep_autosomes):
    if keep_autosomes:
        mask = _snp_df["CHR"].astype(str).str.upper().isin([a.upper() for a in AUTO])
        _gt = _gt[:, mask]; _snp_df = _snp_df[mask].reset_index(drop=True)
    n0, m0 = _gt.shape
    k = miss_snp(_gt) <= geno; gt = _gt[:, k]; snp_df = _snp_df[k].reset_index(drop=True)
    k = miss_ind(gt) <= mind; gt = gt[k]; ind_df = _ind_df[k].reset_index(drop=True)
    m = maf(gt); k = (m >= maf_t); gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    return gt, ind_df, snp_df, {"init": (n0, m0), "final": (gt.shape[0], gt.shape[1])}

@st.cache_data(show_spinner=False)
def pca_analysis(_gt, hash_key):
    X = impute_mean(_gt); X = X - X.mean(0)
    n = min(10, X.shape[0]-1, X.shape[1])
    p = PCA(n_components=n)
    res = p.fit_transform(X), p.explained_variance_ratio_ * 100.0
    gc.collect()
    return res

@st.cache_data(show_spinner=False)
def ld_decay(_gt, _snp_df, hash_key, max_kb=7500, max_snp=2000):
    n = _gt.shape[1]
    if n < 10: return pd.DataFrame()
    
    if n > max_snp:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, max_snp, replace=False))
        _gt = _gt[:, idx]; _snp_df = _snp_df.iloc[idx].reset_index(drop=True)
        n = max_snp

    X = impute_mean(_gt); X = X - X.mean(0)
    stds = X.std(0); stds[stds == 0] = 1e-9; X = X / stds
    
    distances = []; r2_vals = []
    bp = _snp_df["BP"].values; chrs = _snp_df["CHR"].astype(str).values
    max_bp = max_kb * 1000
    
    for i in range(n):
        j_end = min(i+100, n)
        dists = np.abs(bp[i] - bp[i+1:j_end])
        valid = (chrs[i] == chrs[i+1:j_end]) & (dists <= max_bp)
        if np.any(valid):
            r = np.dot(X[:, i], X[:, i+1:j_end][:, valid]) / X.shape[0]
            distances.extend(dists[valid] / 1000.0)
            r2_vals.extend(r**2)
            
    if not distances: return pd.DataFrame()
    df = pd.DataFrame({"dist_kb": distances, "r2": r2_vals})
    df["bin"] = pd.cut(df["dist_kb"], bins=np.linspace(0, max_kb, 30), labels=False)
    res = df.groupby("bin").agg(dist_kb=("dist_kb", "mean"), r2=("r2", "mean")).dropna().sort_values("dist_kb")
    gc.collect()
    return res

@st.cache_data(show_spinner=False)
def ne_snep(_gt, _snp_df, hash_key):
    ld = ld_decay(_gt, _snp_df, hash_key)
    if ld.empty: return pd.DataFrame()
    
    c = ld["dist_kb"].values.copy() / 100.0 
    r2 = ld["r2"].values.copy()
    
    # Correction : utilisation de np.where au lieu de modifier la vue en place
    r2 = np.where(r2 < 0.01, 0.01, r2)
    r2 = np.where(r2 > 1.0, 1.0, r2)
    
    ne = 1.0 / (4 * c * r2)
    generations = 1.0 / (2 * c)
    
    df = pd.DataFrame({"Generations": generations, "Ne": ne}).dropna()
    res = df[df["Ne"] > 0].sort_values("Generations")
    gc.collect()
    return res

@st.cache_data(show_spinner=False)
def compute_admixture(_gt, hash_key, K=3):
    m = NMF(n_components=K, init="nndsvda", max_iter=300, random_state=42)
    W = m.fit_transform(np.clip(impute_mean(_gt), 0.01, 2.0))
    res = W / W.sum(axis=1, keepdims=True)
    gc.collect()
    return res

@st.cache_data(show_spinner=False)
def reynolds_matrix(_gt, _labels, hash_key):
    pops = np.unique(_labels); K = len(pops)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i+1, K):
            p1 = np.nanmean(_gt[_labels == pops[i]], axis=0) / 2.0
            p2 = np.nanmean(_gt[_labels == pops[j]], axis=0) / 2.0
            num = np.nansum((p1 - p2) ** 2)
            den = np.nansum(p1 * (1 - p2) + p2 * (1 - p1))
            if den > 0: D[i, j] = D[j, i] = num / den
    gc.collect()
    return D, pops

@st.cache_data(show_spinner=False)
def fst_snp_wc(_gt, _labels, hash_key):
    pops = np.unique(_labels); r = len(pops); n_snp = _gt.shape[1]
    fst = np.full(n_snp, np.nan)
    for j in range(n_snp):
        n_i = np.zeros(r); p_i = np.zeros(r); h_i = np.zeros(r)
        for k, p in enumerate(pops):
            g = _gt[_labels == p, j]; g = g[~np.isnan(g)]
            if len(g) == 0: continue
            n_i[k] = len(g); p_i[k] = g.mean() / 2.0; h_i[k] = np.mean(g == 1)
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
        den = a + b + (h_bar / 2.0)
        if den > 1e-9: fst[j] = max(0.0, a / den)
    gc.collect()
    return fst

@st.cache_data(show_spinner=False)
def detect_roh(_gt, _snp_df, hash_key, min_snps=30):
    rohs = []
    for c in _snp_df["CHR"].unique():
        idxs = _snp_df[_snp_df["CHR"] == c].index
        if len(idxs) < min_snps: continue
        for i in range(_gt.shape[0]):
            col = _gt[i, idxs]
            sk = None
            for k in range(len(idxs)):
                if col[k] in (0, 2):
                    if sk is None: sk = k
                elif sk is not None:
                    if k - sk >= min_snps: rohs.append({"IID": i, "CHR": c, "N_SNP": k-sk})
                    sk = None
    gc.collect()
    return pd.DataFrame(rohs)

@st.cache_data(show_spinner=False)
def run_gwas(_gt, _pheno, _covs, hash_key, maf_min=0.05):
    valid = np.isfinite(_pheno)
    Y = _pheno[valid]; gv = _gt[valid]
    X0 = np.ones((len(Y), 1)) if _covs is None else np.column_stack([np.ones(len(Y)), _covs[valid]])
    rows = []
    for j in range(gv.shape[1]):
        g = gv[:, j].astype(float)
        if np.isnan(g).mean() > 0.1: continue
        g = np.where(np.isnan(g), np.nanmean(g), g)
        p = g.mean() / 2.0
        if min(p, 1 - p) < maf_min: continue
        X = np.column_stack([X0, g])
        try: b, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except: continue
        res = Y - X @ b
        s2 = np.sum(res ** 2) / max(len(Y) - X.shape[1], 1)
        try: se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[-1, -1]))
        except: continue
        if se <= 0: continue
        tv = b[-1] / se
        pval = float(2 * t_dist.sf(abs(tv), df=max(len(Y) - X.shape[1], 1)))
        rows.append({"SNP_idx": j, "P": pval})
    df = pd.DataFrame(rows)
    if not df.empty and HAS_STATSMODELS:
        _, fdr, _, _ = multipletests(df["P"].values, method="fdr_bh")
        df["FDR"] = fdr
    gc.collect()
    return df

# ============================================================
# EXPORT & 3D
# ============================================================
def build_ped_map(gt, ind_df, snp_df):
    ped_lines = []
    for i in range(gt.shape[0]):
        line = [str(ind_df.iloc[i]["FID"]), str(ind_df.iloc[i]["IID"]), "0", "0", "0", "-9"]
        for j in range(gt.shape[1]):
            val = gt[i, j]
            if np.isnan(val): line.extend(["0", "0"])
            elif val == 0: line.extend(["A", "A"])
            elif val == 1: line.extend(["A", "G"])
            else: line.extend(["G", "G"])
        ped_lines.append("\t".join(line))
    map_lines = [f"{snp_df.iloc[j]['CHR']}\t{snp_df.iloc[j]['SNP']}\t{snp_df.iloc[j]['CM']}\t{int(snp_df.iloc[j]['BP'])}" for j in range(gt.shape[1])]
    return "\n".join(ped_lines) + "\n", "\n".join(map_lines) + "\n"

def render_3d_structure(pdb_data):
    html = f"""
    <div id="viewport" style="width: 100%; height: 500px;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        let viewer = $3Dmol.createViewer(document.getElementById("viewport"), {{ backgroundColor: 'white' }});
        viewer.addModel({json.dumps(pdb_data)}, "pdb");
        viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
        viewer.zoomTo(); viewer.render();
    </script>"""
    components.html(html, height=520)

def fetch_alphafold(gene_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+organism_id:9838&format=json"
    try:
        r = requests.get(url, timeout=10).json()
        if r.get("results"):
            acc = r["results"][0]["primaryAccession"]
            pdb = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb", timeout=10)
            if pdb.status_code == 200: return pdb.text
    except: pass
    return None

# ============================================================
# INTERFACE
# ============================================================
def main():
    st.title("🐪 Camel Genomics Pipeline v20.5")
    st.caption("Optimisation mémoire, vitesse et correction de bug Ne.")

    if "gt" not in st.session_state: st.session_state.gt = None

    with st.sidebar:
        st.header("📁 1. Données")
        mode = st.radio("Source", ["Démo", "PED/MAP", "Axiom→PED"])
        
        if mode == "Démo":
            if st.button("Générer Démo (150x1500)"):
                rng = np.random.default_rng(42)
                n, m = 150, 1500
                gt = rng.choice([0, 1, 2], size=(n, m), p=[0.5, 0.3, 0.2]).astype(np.float32)
                pops = ["ACM", "AHY", "ANL", "ARG", "ASH", "ATR", "AZA", "AOS"]
                ind_df = pd.DataFrame({"FID": [pops[i%8] for i in range(n)], "IID": [f"Ind_{i}" for i in range(n)]})
                ch = rng.choice(AUTO + ["X", "Y"], m)
                bp = np.array([rng.integers(1, int(ARS.get(c, 100)*1e6)) for c in ch])
                snp_df = pd.DataFrame({"CHR": ch, "SNP": [f"rs{i}" for i in range(m)], "CM": bp/1e6, "BP": bp})
                st.session_state.gt = gt; st.session_state.ind_df = ind_df; st.session_state.snp_df = snp_df
                st.session_state.pheno = gt[:, 0]*0.4 + rng.normal(0, 1, n)
                st.success("✅ Prêt !")
                
        elif mode == "PED/MAP":
            ped = st.file_uploader(".ped ou .txt", type=["ped", "gz", "txt"])
            mp = st.file_uploader(".map ou .txt", type=["map", "gz", "txt"])
            if ped and mp and st.button("Charger"):
                try:
                    snp_df = parse_map(mp.read()); gt, ind_df = parse_ped(ped.read(), len(snp_df))
                    st.session_state.gt = gt; st.session_state.ind_df = rec_fid(ind_df); st.session_state.snp_df = snp_df.iloc[:gt.shape[1]]
                    st.success("✅ Chargé !")
                except Exception as e: st.error(f"Erreur : {e}")

        elif mode == "Axiom→PED":
            af = st.file_uploader("Axiom", type=["txt", "csv", "gz"])
            if af and st.button("🔄 Convertir"):
                try:
                    raw = gzip.decompress(af.read()) if af.name.endswith(".gz") else af.read()
                    lines = [l.rstrip("\r\n") for l in raw.decode("utf-8",errors="replace").splitlines() if l.strip()]
                    start = 1 if (lines[0].startswith("#") or "Sample" in lines[0][:50]) else 0
                    samples, gtypes = [], []
                    for line in lines[start:]:
                        parts = line.split("\t") if "\t" in line else line.split()
                        if len(parts)<2: continue
                        samples.append(parts[0].strip()); alleles=[]
                        for cell in parts[1:]:
                            tk = cell.split()
                            if len(tk)==2: alleles.append((tk[0],tk[1]))
                            elif len(tk)==1 and len(tk[0])==2: alleles.append((tk[0][0],tk[0][1]))
                            else: alleles.append(("0","0"))
                        gtypes.append(alleles)
                    c = Counter([len(r) for r in gtypes]).most_common(1)[0][0]
                    for i,r in enumerate(gtypes): gtypes[i] = r[:c] + [("0","0")]*(c-len(r))
                    mpl = [f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(c)]
                    pdl = ["\t".join([s, s, "0","0","0","-9"] + [a for pr in row for a in pr]) for s, row in zip(samples, gtypes)]
                    st.session_state["_plink_ped"] = "\n".join(pdl)+"\n"
                    st.session_state["_plink_map"] = "\n".join(mpl)+"\n"
                    st.success("✅ Converti !")
                except Exception as e: st.error(f"Erreur : {e}")
            if st.session_state.get("_plink_ped"):
                st.download_button("⬇ .ped", st.session_state["_plink_ped"].encode(), "out.ped")
                st.download_button("⬇ .map", st.session_state["_plink_map"].encode(), "out.map")

        st.divider()
        st.header("⚙️ 2. Seuils QC")
        keep_auto = st.checkbox("🧬 Exclure Chromosomes Sexuels (X, Y, MT)", value=True)
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.1)
        maf_t = st.slider("MAF", 0.0, 0.5, 0.05)
        
        if st.button("Appliquer QC & Filtre", type="primary") and st.session_state.gt is not None:
            try:
                h_key = get_hash(st.session_state.gt, st.session_state.snp_df, geno, maf_t, keep_auto)
                gt, ind, snp, stats = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, h_key, geno, 0.2, maf_t, keep_auto)
                st.session_state.gt_f, st.session_state.ind_f, st.session_state.snp_f = gt, ind, snp
                st.success(f"✅ QC : {gt.shape[0]} ind × {gt.shape[1]} snp")
            except Exception as e: st.error(f"Erreur QC : {e}")

    if st.session_state.gt is None:
        st.info("👈 Utilisez la barre latérale pour générer ou charger des données.")
        return

    gt = getattr(st.session_state, "gt_f", st.session_state.gt)
    ind = getattr(st.session_state, "ind_f", st.session_state.ind_df)
    snp = getattr(st.session_state, "snp_f", st.session_state.snp_df)
    labels = ind["FID"].astype(str).values
    global_hash = get_hash(gt, snp, labels)

    tabs = st.tabs([
        "🧬 ACP", "📉 LD Decay", "📈 Ne (SNeP)", "🎨 Admixture", "🌳 Arbre NJ", 
        "🔍 FST (Sélection)", "🏃 ROH", "🧪 GWAS", "🧬 VEP", "🧬 3D", "📤 Export"
    ])

    with tabs[0]:
        st.subheader("Analyse en Composantes Principales")
        if st.button("Lancer l'ACP"):
            s, v = pca_analysis(gt, global_hash)
            fig = px.scatter(pd.DataFrame({"PC1": s[:, 0], "PC2": s[:, 1], "Pop": labels}),
                             x="PC1", y="PC2", color="Pop", title=f"ACP : PC1 ({v[0]:.1f}%) vs PC2 ({v[1]:.1f}%)")
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
            st.plotly_chart(fig, width='stretch')
            st.session_state.pca_s = s

    with tabs[1]:
        st.subheader("Décroissance du Déséquilibre de Liaison")
        if st.button("Calculer LD Decay"):
            with st.spinner("Calcul optimisé..."):
                ld_df = ld_decay(gt, snp, global_hash)
            if not ld_df.empty:
                fig = px.line(ld_df, x="dist_kb", y="r2", title="LD Decay (r²)")
                st.plotly_chart(fig, width='stretch')
            else: st.warning("Pas assez de SNPs.")

    with tabs[2]:
        st.subheader("Taille Efficace de la Population (Ne)")
        if st.button("Estimer Ne"):
            try:
                with st.spinner("Calcul de Ne..."):
                    ne_df = ne_snep(gt, snp, global_hash)
                if not ne_df.empty:
                    fig = px.line(ne_df, x="Generations", y="Ne", title="Ne à travers le temps")
                    st.plotly_chart(fig, width='stretch')
                else: st.warning("Impossible d'estimer Ne.")
            except Exception as e:
                st.error(f"Erreur lors de l'estimation de Ne : {e}")

    with tabs[3]:
        st.subheader("Admixture (NMF)")
        K = st.slider("Nombre de clusters K", 2, 8, 3)
        if st.button(f"Lancer Admixture (K={K})"):
            Q = compute_admixture(gt, global_hash, K=K)
            df_q = pd.DataFrame(Q, columns=[f"K{i}" for i in range(K)])
            df_q["Pop"] = labels; df_q["x"] = range(len(Q))
            df_melt = df_q.melt(id_vars=["Pop", "x"], var_name="Cluster", value_name="Prop")
            fig = px.bar(df_melt, x="x", y="Prop", color="Cluster", title=f"Admixture K={K}")
            fig.update_layout(barmode='stack', xaxis_visible=False)
            st.plotly_chart(fig, width='stretch')

    with tabs[4]:
        st.subheader("Arbre Phylogénétique (Neighbor-Joining)")
        if st.button("Construire l'arbre (Reynolds)"):
            D, pops = reynolds_matrix(gt, labels, global_hash)
            Z = linkage(squareform(D), method='average')
            dn = dendrogram(Z, labels=pops, no_plot=True)
            
            fig = go.Figure()
            for i, d in enumerate(dn['dcoord']):
                fig.add_trace(go.Scatter(x=d[1], y=d[0], mode='lines', line=dict(color='black', width=2), showlegend=False))
            
            for i, (leaf_x, leaf_label) in enumerate(zip(dn['leaves'], dn['ivl'])):
                fig.add_annotation(x=0, y=i*10+5, text=leaf_label, showarrow=False, xanchor='right', xshift=-10)
                
            fig.update_layout(title="Arbre NJ (Distances de Reynolds)", 
                              xaxis_title="Distance", yaxis=dict(showticklabels=False),
                              height=600)
            st.plotly_chart(fig, width='stretch')
            st.dataframe(pd.DataFrame(D, index=pops, columns=pops), width='stretch')

    with tabs[5]:
        st.subheader("Signatures de Sélection (FST de Weir & Cockerham)")
        if st.button("Calculer FST"):
            fst = fst_snp_wc(gt, labels, global_hash)
            df_fst = snp.copy(); df_fst["FST"] = fst
            df_fst = df_fst.dropna(subset=["FST"])
            if not df_fst.empty:
                df_fst = df_fst.assign(_k=df_fst["CHR"].map(_chk)).sort_values(["_k", "BP"]).reset_index(drop=True)
                fig = go.Figure(go.Scattergl(x=df_fst.index, y=df_fst["FST"], mode='markers', 
                                             marker=dict(size=5, color=df_fst["CHR"].astype('category').cat.codes, colorscale='Viridis')))
                fig.update_layout(title="FST par SNP (Sélection)", xaxis_title="Position génomique", yaxis_title="FST")
                st.plotly_chart(fig, width='stretch')
                st.dataframe(df_fst.sort_values("FST", ascending=False).head(10), width='stretch')

    with tabs[6]:
        st.subheader("Démographie : ROH (Runs of Homozygosity)")
        if st.button("Détecter les ROH"):
            roh = detect_roh(gt, snp, global_hash)
            if not roh.empty:
                st.metric("ROH détectés", len(roh))
                st.dataframe(roh, width='stretch')
            else: st.info("Aucun ROH détecté.")

    with tabs[7]:
        st.subheader("GWAS (OLS + FDR)")
        if hasattr(st.session_state, "pheno"):
            if st.button("Lancer GWAS"):
                covs = st.session_state.pca_s[:, :3] if hasattr(st.session_state, "pca_s") else None
                res = run_gwas(gt, st.session_state.pheno, covs, global_hash)
                if not res.empty:
                    res["SNP"] = snp["SNP"].values[res["SNP_idx"]]
                    res["CHR"] = snp["CHR"].values[res["SNP_idx"]]
                    res["BP"] = snp["BP"].values[res["SNP_idx"]]
                    st.dataframe(res.sort_values("P").head(10), width='stretch')
                else: st.warning("Aucun SNP significatif.")

    with tabs[8]:
        st.subheader("Annotation VEP (Ensembl)")
        snp_input = st.text_input("Variant (ex: 1:100000:A:G)", "1:100000:A:G")
        if st.button("Interroger VEP"):
            try:
                url = "https://rest.ensembl.org/vep/camelus_dromedarius/region"
                r = requests.post(url, headers={"Content-Type": "application/json"}, json={"variants": [snp_input]}, timeout=10)
                if r.status_code == 200:
                    data = r.json()[0]
                    if "transcript_consequences" in data:
                        c = data["transcript_consequences"][0]
                        st.success(f"Gène: {c.get('gene_symbol')} | Impact: {c.get('impact')}")
                    else: st.json(data)
            except Exception as e: st.error(f"Erreur : {e}")

    with tabs[9]:
        st.subheader("Structure 3D (AlphaFold)")
        gene = st.text_input("Nom du gène (ex: CSN2)", "CSN2")
        if st.button("Charger 3D"):
            pdb = fetch_alphafold(gene)
            if pdb: render_3d_structure(pdb)
            else: st.error("Protéine introuvable.")

    with tabs[10]:
        st.subheader("📤 Exporter PED / MAP (Autosomal)")
        if st.button("Générer les fichiers filtrés"):
            ped_str, map_str = build_ped_map(gt, ind, snp)
            st.download_button("⬇ Télécharger .ped", ped_str.encode(), "filtered.ped")
            st.download_button("⬇ Télécharger .map", map_str.encode(), "filtered.map")

if __name__ == "__main__":
    main()
