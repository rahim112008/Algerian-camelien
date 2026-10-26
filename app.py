"""🐪 Camel Genomics Pipeline v14.0 — Édition Scientifique & Complète"""
import gzip, io, os, warnings, zipfile, time, json
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
from scipy.stats import t as t_dist
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import requests
import streamlit.components.v1 as components

# Optionnel : Biopython
try:
    from Bio import Align
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

# Optionnel : Statsmodels pour correction FDR
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings("ignore")

st.set_page_config(page_title="🐪 Camel Genomics v14.0", page_icon="🐪", layout="wide")

# ============================================================
# CONSTANTES GÉNOMIQUES — Camelus dromedarius (2n=74)
# ============================================================
ARS = {
    "1": 124.7, "2": 123.8, "3": 120.7, "4": 68.3, "5": 97.5, "6": 97.5, "7": 91.2, "8": 89.4,
    "9": 86.7, "10": 84.1, "11": 81.5, "12": 78.9, "13": 76.3, "14": 73.8, "15": 71.2, "16": 68.6,
    "17": 66.1, "18": 63.5, "19": 60.9, "20": 58.4, "21": 55.8, "22": 53.2, "23": 50.7, "24": 48.1,
    "25": 45.5, "26": 43.0, "27": 40.4, "28": 37.8, "29": 35.3, "30": 32.7, "31": 30.1, "32": 27.6,
    "33": 25.0, "34": 22.4, "35": 19.9, "36": 17.3, "X": 140.0, "Y": 45.0, "MT": 16.6
}
AUTO = [str(i) for i in range(1, 37)]

GROUPES_DEFAULT = {
    "Production": {"HAUTE_LAITIERE": ["MOJAHEEM", "MAGHATEER"], "VIANDE": ["JANDAWEL", "NABUL"]},
    "Adaptation": {"CHAUD": ["MOJAHEEM", "BIKANERI"], "FROID": ["KAZAKH_BACTRIAN", "ARVANA"]},
    "Sous-espèce": {"DROMADAIRE": ["MOJAHEEM", "MAGHATEER"], "BACTRIANE": ["KAZAKH_BACTRIAN", "ARVANA"]}
}

def make_group_map(group_dict):
    return {r.upper(): grp for grp, races in group_dict.items() for r in races}

# ============================================================
# UTILS & CACHE
# ============================================================
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
    return (1, {"X": 100, "Y": 101, "MT": 102}.get(s, 999), s)

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
        p = line.split()
        if len(p) >= 4:
            rows.append({"CHR": str(p[0]), "SNP": p[1], "CM": float(p[2]) if p[2] not in ("0", ".") else 0.0, "BP": int(float(p[3]))})
    return pd.DataFrame(rows)

def parse_ped(b, n_map):
    lines = [l.strip() for l in _rt(b).splitlines() if l.strip()]
    n_ped = (len(lines[0].split()) - 6) // 2
    n = min(n_map, n_ped)
    fids, iids, rows = [], [], []
    for line in lines:
        p = line.split()
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
# ALGORITHMES SCIENTIFIQUES
# ============================================================
@st.cache_data(show_spinner=False)
def apply_qc(gt, ind_df, snp_df, geno, mind, maf_t):
    n0, m0 = gt.shape
    k = miss_snp(gt) <= geno; gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    k = miss_ind(gt) <= mind; gt = gt[k]; ind_df = ind_df[k].reset_index(drop=True)
    m = maf(gt); k = (m >= maf_t); gt = gt[:, k]; snp_df = snp_df[k].reset_index(drop=True)
    return gt, ind_df, snp_df, {"init": (n0, m0), "final": (gt.shape[0], gt.shape[1])}

@st.cache_data(show_spinner=False)
def pca_analysis(gt):
    X = impute_mean(gt); X = X - X.mean(0)
    n = min(10, X.shape[0]-1, X.shape[1])
    p = PCA(n_components=n)
    return p.fit_transform(X), p.explained_variance_ratio_ * 100.0

@st.cache_data(show_spinner=False)
def fst_snp_wc(gt, labels):
    pops = np.unique(labels); r = len(pops); n_snp = gt.shape[1]
    fst = np.full(n_snp, np.nan)
    for j in range(n_snp):
        n_i = np.zeros(r); p_i = np.zeros(r); h_i = np.zeros(r)
        for k, p in enumerate(pops):
            g = gt[labels == p, j]; g = g[~np.isnan(g)]
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
    return fst

@st.cache_data(show_spinner=False)
def windowed_fst(gt, snp_df, labels, w=50):
    fst = fst_snp_wc(gt, labels)
    res = []
    for c in snp_df["CHR"].unique():
        sub = snp_df[snp_df["CHR"] == c].sort_values("BP").reset_index(drop=True)
        sub["FST"] = fst[sub.index]
        if len(sub) < w: continue
        for i in range(0, len(sub)-w+1, w//2):
            win = sub.iloc[i:i+w]
            res.append({"CHR": str(c), "MID": int(win["BP"].mean()), "MEAN_FST": win["FST"].mean()})
    return pd.DataFrame(res)

@st.cache_data(show_spinner=False)
def detect_roh(gt, snp_df, min_snps=30):
    rohs = []
    for c in snp_df["CHR"].unique():
        idxs = snp_df[snp_df["CHR"] == c].index
        if len(idxs) < min_snps: continue
        for i in range(gt.shape[0]):
            col = gt[i, idxs]
            sk = None
            for k in range(len(idxs)):
                if col[k] in (0, 2):
                    if sk is None: sk = k
                elif sk is not None:
                    if k - sk >= min_snps: rohs.append({"IID": i, "CHR": c, "N_SNP": k-sk})
                    sk = None
    return pd.DataFrame(rohs)

@st.cache_data(show_spinner=False)
def run_gwas(gt, pheno, covs=None, maf_min=0.05):
    valid = np.isfinite(pheno)
    Y = pheno[valid]; gv = gt[valid]
    X0 = np.ones((len(Y), 1)) if covs is None else np.column_stack([np.ones(len(Y)), covs[valid]])
    
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
        rows.append({"SNP_idx": j, "MAF": min(p, 1-p), "BETA": b[-1], "SE": se, "T": tv, "P": pval})
    
    df = pd.DataFrame(rows)
    if not df.empty and HAS_STATSMODELS:
        _, fdr, _, _ = multipletests(df["P"].values, method="fdr_bh")
        df["FDR"] = fdr
    return df

# ============================================================
# VISUALISATIONS
# ============================================================
def plot_pca(pca_s, pca_v, pops):
    fig = px.scatter(pd.DataFrame({"PC1": pca_s[:, 0], "PC2": pca_s[:, 1], "Pop": pops}),
                     x="PC1", y="PC2", color="Pop", title=f"ACP : PC1 ({pca_v[0]:.1f}%) vs PC2 ({pca_v[1]:.1f}%)")
    return fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))

def plot_manhattan(df, yc="P"):
    if "P" in df.columns and yc == "P":
        df["logP"] = -np.log10(df["P"].clip(1e-300, 1)); yc = "logP"
    df = df.assign(_k=df["CHR"].map(_chk)).sort_values(["_k", "BP"]).reset_index(drop=True)
    fig = go.Figure()
    for i, (c, sub) in enumerate(df.groupby("CHR")):
        fig.add_trace(go.Scatter(x=sub.index, y=sub[yc], mode="markers",
                                 marker=dict(size=6, color="#2c3e50" if i%2==0 else "#7f8c8d"),
                                 text=sub.get("SNP", ""), showlegend=False))
    fig.update_layout(title="Manhattan Plot", xaxis_title="Chromosome", yaxis_title="-log10(P) ou Valeur")
    return fig

def render_3d_structure(pdb_data):
    html = f"""
    <div id="viewport" style="width: 100%; height: 500px; position: relative;"></div>
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
# INTERFACE STREAMLIT
# ============================================================
def main():
    st.title("🐪 Camel Genomics Pipeline v14.0")
    st.caption("Analyse complète et rigoureuse pour le Dromadaire (2n=74). Inclut : FST WC84, GWAS, AlphaFold, VEP.")

    if "gt" not in st.session_state: st.session_state.gt = None

    # --- BARRE LATÉRALE ---
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source", ["Démo", "PED/MAP"])
        
        if mode == "Démo":
            if st.button("Générer 150 Ind × 1500 SNP"):
                rng = np.random.default_rng(42)
                n, m = 150, 1500
                gt = rng.choice([0, 1, 2], size=(n, m), p=[0.5, 0.3, 0.2]).astype(np.float32)
                pops = ["MOJAHEEM", "MAGHATEER", "WADAH", "BIKANERI"]
                ind_df = pd.DataFrame({"FID": [pops[i%4] for i in range(n)], "IID": [f"Ind_{i}" for i in range(n)]})
                ch = rng.choice(AUTO, m)
                bp = np.array([rng.integers(1, int(ARS[c]*1e6)) for c in ch])
                snp_df = pd.DataFrame({"CHR": ch, "SNP": [f"rs{i}" for i in range(m)], "BP": bp})
                
                # Phénotype simulé
                y = gt[:, 0]*0.4 + rng.normal(0, 1, n)
                
                st.session_state.gt = gt; st.session_state.ind_df = ind_df; st.session_state.snp_df = snp_df
                st.session_state.pheno = y
                st.success("✅ Prêt !")
                
        elif mode == "PED/MAP":
            ped = st.file_uploader(".ped", type=["ped", "gz"])
            mp = st.file_uploader(".map", type=["map", "gz"])
            if ped and mp and st.button("Charger"):
                snp_df = parse_map(mp.read()); gt, ind_df = parse_ped(ped.read(), len(snp_df))
                st.session_state.gt = gt; st.session_state.ind_df = rec_fid(ind_df); st.session_state.snp_df = snp_df.iloc[:gt.shape[1]]
                st.success("✅ Chargé !")

        st.divider()
        st.header("⚙️ Filtres QC")
        geno = st.slider("Missing SNP", 0.0, 0.5, 0.1)
        maf_t = st.slider("MAF", 0.0, 0.5, 0.05)
        if st.button("Appliquer QC") and st.session_state.gt is not None:
            gt, ind, snp, stats = apply_qc(st.session_state.gt, st.session_state.ind_df, st.session_state.snp_df, geno, 0.2, maf_t)
            st.session_state.gt_f, st.session_state.ind_f, st.session_state.snp_f = gt, ind, snp
            st.success(f"QC : {gt.shape[0]} ind × {gt.shape[1]} snp")

    if st.session_state.gt is None:
        st.info("👈 Utilisez la barre latérale pour générer ou charger des données.")
        return

    # Raccourcis
    gt = getattr(st.session_state, "gt_f", st.session_state.gt)
    ind = getattr(st.session_state, "ind_f", st.session_state.ind_df)
    snp = getattr(st.session_state, "snp_f", st.session_state.snp_df)
    labels = ind["FID"].astype(str).values

    # --- ONGLETS ---
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🧬 Structure", "🎨 Admixture", "🔍 Sélection", "📈 Démographie", "🧪 GWAS", "🔬 Annotation (VEP)", "🧬 Structure 3D"
    ])

    # --- 1. STRUCTURE ---
    with t1:
        st.subheader("Analyse en Composantes Principales")
        if st.button("Lancer l'ACP"):
            try:
                s, v = pca_analysis(gt)
                st.plotly_chart(plot_pca(s, v, labels), use_container_width=True)
                st.session_state.pca_s = s
            except Exception as e: st.error(f"Erreur ACP : {e}")

    # --- 2. ADMIXTURE ---
    with t2:
        st.subheader("Estimation d'ascendance (NMF)")
        K = st.slider("Populations (K)", 2, 8, 3)
        if st.button("Lancer Admixture"):
            try:
                m = NMF(n_components=K, init="nndsvda", max_iter=200)
                Q = m.fit_transform(np.clip(impute_mean(gt), 0, 2))
                Q = Q / Q.sum(1, keepdims=True)
                df_q = pd.DataFrame(Q, columns=[f"K{i}" for i in range(K)])
                df_q["Pop"] = labels
                df_q["x"] = range(len(Q))
                fig = go.Figure()
                for i in range(K):
                    fig.add_trace(go.Bar(x=df_q["x"], y=df_q[f"K{i}"], name=f"K{i+1}"))
                fig.update_layout(barmode="stack", title="Ascendance", xaxis_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e: st.error(f"Erreur NMF : {e}")

    # --- 3. SÉLECTION ---
    with t3:
        st.subheader("Signatures de sélection")
        if st.button("Calculer FST par fenêtre (WC84)"):
            try:
                df_w = windowed_fst(gt, snp, labels)
                st.plotly_chart(plot_manhattan(df_w, yc="MEAN_FST"), use_container_width=True)
                st.dataframe(df_w.sort_values("MEAN_FST", ascending=False).head(10))
            except Exception as e: st.error(f"Erreur FST : {e}")

    # --- 4. DÉMOGRAPHIE ---
    with t4:
        st.subheader("Démographie : ROH")
        if st.button("Détecter les ROH"):
            try:
                df_roh = detect_roh(gt, snp)
                st.metric("ROH détectés", len(df_roh))
                st.dataframe(df_roh.head(20))
            except Exception as e: st.error(f"Erreur ROH : {e}")

    # --- 5. GWAS ---
    with t5:
        st.subheader("GWAS (OLS + FDR)")
        if hasattr(st.session_state, "pheno"):
            if st.button("Lancer GWAS"):
                try:
                    covs = st.session_state.pca_s[:, :3] if hasattr(st.session_state, "pca_s") else None
                    res = run_gwas(gt, st.session_state.pheno, covs)
                    if res.empty: st.warning("Aucun SNP significatif (MAF trop bas).")
                    else:
                        res["SNP"] = snp["SNP"].values[res["SNP_idx"]]
                        res["CHR"] = snp["CHR"].values[res["SNP_idx"]]
                        res["BP"] = snp["BP"].values[res["SNP_idx"]]
                        st.plotly_chart(plot_manhattan(res), use_container_width=True)
                        st.dataframe(res.sort_values("P").head(15))
                except Exception as e: st.error(f"Erreur GWAS : {e}")

    # --- 6. ANNOTATION VEP ---
    with t6:
        st.subheader("Ensembl VEP (Camelus dromedarius)")
        snp_input = st.text_input("Variant (ex: 1:100000:A:G)", "1:100000:A:G")
        if st.button("VEP"):
            try:
                url = "https://rest.ensembl.org/vep/camelus_dromedarius/region"
                r = requests.post(url, headers={"Content-Type": "application/json"}, json={"variants": [snp_input]}, timeout=10)
                if r.status_code == 200:
                    data = r.json()[0]
                    if "transcript_consequences" in data:
                        c = data["transcript_consequences"][0]
                        st.success(f"Gène: {c.get('gene_symbol')} | Impact: {c.get('impact')} | Consequence: {c.get('consequence_terms')}")
                    else: st.json(data)
                else: st.error("Format invalide ou introuvable.")
            except Exception as e: st.error(f"Erreur : {e}")

    # --- 7. STRUCTURE 3D ---
    with t7:
        st.subheader("AlphaFold — Protéines Camelines")
        gene = st.text_input("Nom du gène (ex: CSN2)", "CSN2")
        if st.button("Charger AlphaFold"):
            with st.spinner("Recherche sur UniProt & AlphaFold..."):
                pdb = fetch_alphafold(gene)
                if pdb: render_3d_structure(pdb)
                else: st.error("Protéine introuvable pour Camelus dromedarius.")

if __name__ == "__main__":
    main()
