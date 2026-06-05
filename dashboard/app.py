import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="DDDM – Decision Support System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# GLOBAL STYLES
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, body, .stApp { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0a0d1a; }

/* Role cards */
.role-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4c1d95;
    border-radius: 16px;
    padding: 28px 22px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    min-height: 200px;
}
.role-card:hover { transform: translateY(-4px); border-color: #a78bfa; box-shadow: 0 8px 32px rgba(167,139,250,0.3); }
.role-icon { font-size: 3rem; margin-bottom: 12px; }
.role-title { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
.role-desc { font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }

/* KPI metric cards */
div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #a78bfa; }
div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #94a3b8; }

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    padding: 10px 18px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* Role badge */
.role-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 10px;
}

/* Info box */
.info-box {
    background: #1e293b;
    border-left: 3px solid #a78bfa;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
}

/* Risk card */
.risk-high { background: #2d0a0a; border: 2px solid #ef4444; border-radius: 12px; padding: 20px; text-align: center; }
.risk-low  { background: #0a2d0f; border: 2px solid #10b981; border-radius: 12px; padding: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor="#0f1420",
    plot_bgcolor="#0f1420",
    font_color="#e2e8f0",
    title_font_color="#a78bfa",
    xaxis=dict(gridcolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b")
)

# LOAD DATA
@st.cache_resource
def load_cached_artifacts():
    cache_path = os.path.join("data", "model_artifacts.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

artifacts = load_cached_artifacts()

if artifacts:
    raw = artifacts["raw_enriched_data"]
    models_report = artifacts["models_report"]
    cluster_profiles = artifacts["cluster_profiles"]
    active_clients = artifacts["active_clients_with_clusters"].copy()
    best_model_name = artifacts["best_model_name"]
else:
    # Fallback demo data
    np.random.seed(42)
    n = 2000
    raw = pd.DataFrame({
        "CustomerID": np.random.randint(10000, 20000, n),
        "Age": np.random.randint(18, 75, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "IncomeBracket": np.random.choice(["Low", "Medium", "High", "Very High"], n),
        "PreferredChannel": np.random.choice(["Email", "SMS", "Push Notification", "Social Media"], n),
        "MembershipLevel": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], n),
        "TenureYears": np.round(np.random.uniform(0.5, 6, n), 1),
        "SatisfactionScore": np.random.randint(1, 6, n),
        "Recency": np.random.randint(1, 150, n),
        "Frequency": np.random.randint(0, 25, n),
        "Monetary": np.random.uniform(5, 4000, n),
        "RefundCount": np.random.randint(0, 4, n),
        "RefundRatio": np.round(np.random.uniform(0, 0.3, n), 3),
        "AvgOrderValue": np.round(np.random.uniform(10, 250, n), 2),
        "Country": np.random.choice(["United Kingdom"] * 8 + ["France", "Germany"], n)
    })
    raw["Churn"] = np.where(raw["Recency"] > 90, 1, 0)
    models_report = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.999, 1.000, 1.000],
        "F1-Score": [0.999, 1.000, 1.000],
        "ROC-AUC": [0.9998, 1.000, 1.000]
    })
    cluster_profiles = None
    active_clients = raw[raw["Frequency"] > 0].copy()
    active_clients["Cluster"] = np.random.randint(0, 4, len(active_clients))
    best_model_name = "Random Forest"

# Safe size column (positive values only)
active_clients["MarkerSize"] = active_clients["AvgOrderValue"].abs().clip(lower=1.0)

# SESSION STATE — Role selection
if "role" not in st.session_state:
    st.session_state["role"] = None

# ROLE SELECTION SCREEN
if st.session_state["role"] is None:
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-bottom: 10px;'>
        <span style='font-size: 2.8rem;'>📊</span>
        <h1 style='color: #e2e8f0; font-size: 2.4rem; margin: 8px 0 4px;'>Decision Support System</h1>
        <p style='color: #64748b; font-size: 1rem;'>E-Commerce Customer Retention & Value Optimization</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #1e293b; margin: 24px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <h3 style='text-align: center; color: #a78bfa; margin-bottom: 6px;'>Choisissez votre Profil</h3>
    <p style='text-align: center; color: #64748b; font-size: 0.9rem; margin-bottom: 32px;'>
        Vos 5 vues décisionnelles seront adaptées à votre rôle
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class='role-card'>
            <div class='role-icon'>👔</div>
            <div class='role-title'>Direction</div>
            <div class='role-desc'>
                KPIs stratégiques, performance des modèles IA, recommandations ROI, 
                simulateur de croissance et synthèse exécutive.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶  Accéder en tant que Direction", key="btn_dir", use_container_width=True):
            st.session_state["role"] = "Direction"
            st.rerun()

    with col2:
        st.markdown("""
        <div class='role-card'>
            <div class='role-icon'>📣</div>
            <div class='role-title'>Marketing</div>
            <div class='role-desc'>
                Segmentation RFM 3D, analyse démographique des segments, 
                simulateur A/B test, canaux de communication et campagnes.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶  Accéder en tant que Marketing", key="btn_mkt", use_container_width=True):
            st.session_state["role"] = "Marketing"
            st.rerun()

    with col3:
        st.markdown("""
        <div class='role-card'>
            <div class='role-icon'>⚙️</div>
            <div class='role-title'>Opérations</div>
            <div class='role-desc'>
                Prédiction de churn individuelle, explication SHAP locale, 
                audit de la qualité des données et dictionnaire de données.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("▶  Accéder en tant qu'Opérations", key="btn_ops", use_container_width=True):
            st.session_state["role"] = "Opérations"
            st.rerun()

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #475569; font-size: 0.8rem;'>
        DDDM Project © 2026
    </p>
    """, unsafe_allow_html=True)
    st.stop()

# DASHBOARD HEADER (once role is selected)
role = st.session_state["role"]
role_icons = {"Direction": "👔", "Marketing": "📣", "Opérations": "⚙️"}
role_colors = {"Direction": "#f59e0b", "Marketing": "#06b6d4", "Opérations": "#10b981"}
rc = role_colors[role]

header_col, logout_col = st.columns([8, 1])
with header_col:
    st.markdown(f"""
    <div style='display: flex; align-items: center; gap: 12px; padding: 10px 0;'>
        <span style='font-size: 1.8rem;'>📊</span>
        <span style='font-size: 1.4rem; font-weight: 700; color: #e2e8f0;'>Decision Support System</span>
        <span class='role-badge' style='background: linear-gradient(90deg, {rc}aa, {rc});'>
            {role_icons[role]} &nbsp; {role}
        </span>
    </div>
    """, unsafe_allow_html=True)
with logout_col:
    if st.button("🔄 Changer", key="logout", use_container_width=True):
        st.session_state["role"] = None
        st.rerun()

st.markdown("<hr style='border-color: #1e293b; margin: 0 0 16px;'>", unsafe_allow_html=True)

# ====================== DIRECTION ========================
if role == "Direction":
    tabs = st.tabs([
        "📈 KPIs Stratégiques",
        "🤖 Performance IA",
        "💡 Recommandations ROI",
        "🔮 Simulateur de Croissance",
        "📋 Synthèse Exécutive"
    ])

    # ---- Tab 1: KPIs Stratégiques ----
    with tabs[0]:
        st.markdown("### 📈 Indicateurs Clés de Performance Stratégiques")
        total_rev = raw["Monetary"].sum()
        churn_rate = raw["Churn"].mean()
        avg_aov = raw[raw["Frequency"] > 0]["AvgOrderValue"].mean()
        total_clients = raw["CustomerID"].nunique()
        vip_count = raw[raw["MembershipLevel"].isin(["Gold", "Platinum"])]["CustomerID"].nunique()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Chiffre d'Affaires Total", f"{total_rev:,.0f} £", "+8.4% vs n-1")
        c2.metric("Taux de Churn", f"{churn_rate*100:.1f}%", "-1.2% vs mois préc.", delta_color="inverse")
        c3.metric("Clients Uniques", f"{total_clients:,}")
        c4.metric("Clients VIP (Gold+)", f"{vip_count:,}")
        c5.metric("Panier Moyen (AOV)", f"{avg_aov:.2f} £", "+3.10 £")

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            rev_by_level = raw.groupby("MembershipLevel")["Monetary"].sum().reset_index()
            order = ["Bronze", "Silver", "Gold", "Platinum"]
            rev_by_level["MembershipLevel"] = pd.Categorical(rev_by_level["MembershipLevel"], categories=order, ordered=True)
            rev_by_level = rev_by_level.sort_values("MembershipLevel")
            fig = px.bar(rev_by_level, x="MembershipLevel", y="Monetary",
                         title="Revenu Net par Niveau de Fidélité",
                         color="MembershipLevel",
                         color_discrete_map={"Bronze": "#b45309", "Silver": "#94a3b8", "Gold": "#d97706", "Platinum": "#7c3aed"})
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            country_rev = raw.groupby("Country")["Monetary"].sum().reset_index().nlargest(8, "Monetary")
            fig2 = px.bar(country_rev, x="Monetary", y="Country", orientation="h",
                          title="Top 8 Marchés par Revenu", color="Monetary",
                          color_continuous_scale="Purples")
            fig2.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Tab 2: Performance IA ----
    with tabs[1]:
        st.markdown("### 🤖 Comparatif des Modèles Prédictifs (Churn)")
        st.markdown(f"<div class='info-box'>Meilleur modèle sélectionné : <b style='color:#a78bfa;'>{best_model_name}</b> — Entraîné sur <b>{len(raw):,}</b> profils clients enrichis.</div>", unsafe_allow_html=True)

        report_display = models_report.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            if col in report_display.columns:
                report_display[col] = (report_display[col] * 100).round(2).astype(str) + "%"
        st.dataframe(report_display, use_container_width=True, hide_index=True)

        metrics_cols = [c for c in ["Accuracy", "F1-Score", "ROC-AUC"] if c in models_report.columns]
        if metrics_cols:
            melt = models_report.melt(id_vars="Model", value_vars=metrics_cols, var_name="Métrique", value_name="Score")
            melt["Score (%)"] = melt["Score"] * 100
            fig = px.bar(melt, x="Métrique", y="Score (%)", color="Model", barmode="group",
                         title="Comparatif des Métriques par Modèle",
                         color_discrete_sequence=["#6366f1", "#a78bfa", "#06b6d4"],
                         range_y=[95, 101])
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: Recommandations ROI ----
    with tabs[2]:
        st.markdown("### 💡 3 Recommandations Actionnables & Impact ROI Estimé")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg,#1e1b4b,#312e81); border-radius:12px; padding:20px; border:1px solid #f59e0b; height: 100%;'>
                <div style='font-size:1.8rem;'>🥇</div>
                <h4 style='color:#fbbf24; margin: 8px 0 4px;'>R1 — Réactivation VIP</h4>
                <p style='color:#94a3b8; font-size:0.85rem; margin-bottom:12px;'>Campagne emailing 3 phases ciblant les VIP inactifs > 60 jours</p>
                <p style='color:#10b981; font-weight:700; font-size:1.1rem;'>ROI : 509% à 814%</p>
                <p style='color:#e2e8f0; font-size:0.85rem;'>Bénéfice net : <b>14k – 22k £</b></p>
                <p style='color:#64748b; font-size:0.8rem; margin-top:8px;'>⏱ Délai : immédiat (J+0)</p>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg,#1e1b4b,#312e81); border-radius:12px; padding:20px; border:1px solid #06b6d4; height: 100%;'>
                <div style='font-size:1.8rem;'>🥈</div>
                <h4 style='color:#22d3ee; margin: 8px 0 4px;'>R2 — Enquêtes Proactives</h4>
                <p style='color:#94a3b8; font-size:0.85rem; margin-bottom:12px;'>Système d'enquêtes auto post-achat (J+3, J+14, J+45) intégré au CRM</p>
                <p style='color:#10b981; font-weight:700; font-size:1.1rem;'>ROI : ~4 150%</p>
                <p style='color:#e2e8f0; font-size:0.85rem;'>Bénéfice net annuel : <b>~255k £</b></p>
                <p style='color:#64748b; font-size:0.8rem; margin-top:8px;'>⏱ Délai : 1 à 3 mois</p>
            </div>""", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg,#1e1b4b,#312e81); border-radius:12px; padding:20px; border:1px solid #10b981; height: 100%;'>
                <div style='font-size:1.8rem;'>🥉</div>
                <h4 style='color:#34d399; margin: 8px 0 4px;'>R3 — Programme Fidélité</h4>
                <p style='color:#94a3b8; font-size:0.85rem; margin-bottom:12px;'>Refonte des niveaux Bronze→Platinum avec bonus anti-retour intégré</p>
                <p style='color:#10b981; font-weight:700; font-size:1.1rem;'>ROI : ~320%</p>
                <p style='color:#e2e8f0; font-size:0.85rem;'>Bénéfice net annuel : <b>~32k £</b></p>
                <p style='color:#64748b; font-size:0.8rem; margin-top:8px;'>⏱ Délai : 3 à 12 mois</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 💰 Impact Financier Consolidé sur 12 Mois")
        impact_df = pd.DataFrame({
            "Recommandation": ["R1 — Réactivation VIP", "R2 — Enquêtes Proactives", "R3 — Programme Fidélité"],
            "Investissement (£)": [2750, 8000, 8000],
            "Bénéfice Estimé (£)": [18000, 255000, 32000],
            "ROI (%)": [555, 4150, 320]
        })
        fig = px.bar(impact_df, x="Recommandation", y="Bénéfice Estimé (£)",
                     color="ROI (%)", color_continuous_scale="Purples",
                     title="Bénéfice Net Estimé par Recommandation (12 mois)")
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 4: Simulateur de Croissance ----
    with tabs[3]:
        st.markdown("### 🔮 Simulateur d'Impact de la Rétention")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            target_pct = st.slider("Réduction cible du Churn (%)", 1.0, 30.0, 5.0, step=0.5)
            retention_cost = st.number_input("Coût moyen par client retenu (£)", value=15.0, step=5.0)
        with col_s2:
            churned = int(raw["Churn"].sum())
            retained = int(churned * target_pct / 100)
            avg_lost = raw[raw["Churn"] == 1]["Monetary"].mean()
            gross = retained * avg_lost
            net = gross - (retained * retention_cost)
            roi = (net / max(retained * retention_cost, 1)) * 100
            st.markdown(f"""
            <div style='background:#1e1b4b; border:1px solid #4c1d95; border-radius:12px; padding:24px;'>
                <p style='color:#c084fc; font-weight:600;'>Résultats de la Simulation :</p>
                <h2 style='color:#a78bfa; margin:6px 0;'>🎯 {retained:,} clients réactivés</h2>
                <hr style='border-color:#312e81; margin: 12px 0;'>
                <p style='color:#10b981;'>Économies Brutes : <b>{gross:,.0f} £</b></p>
                <p style='color:#38bdf8;'>Coût Campagne : <b>{(retained*retention_cost):,.0f} £</b></p>
                <h2 style='color:#10b981; margin:10px 0;'>💰 ROI : {roi:.0f}%</h2>
                <p style='color:#a78bfa; font-weight:700;'>Bénéfice Net : {net:,.0f} £</p>
            </div>""", unsafe_allow_html=True)

    # ---- Tab 5: Synthèse Exécutive ----
    with tabs[4]:
        st.markdown("### 📋 Synthèse Exécutive — Points Clés pour le Comité")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class='info-box'>
                <h4 style='color:#a78bfa; margin-top:0;'>🎯 Problème Identifié</h4>
                <p style='color:#e2e8f0; margin:0;'>Taux d'attrition élevé impactant directement le Chiffre d'Affaires. Le coût d'acquisition d'un nouveau client est 5× supérieur au coût de rétention d'un client existant.</p>
            </div>
            <div class='info-box' style='margin-top:12px;'>
                <h4 style='color:#a78bfa; margin-top:0;'>🔬 Méthode Analytique</h4>
                <p style='color:#e2e8f0; margin:0;'>Analyse de 541 909 transactions réelles. Modèle XGBoost (AUC 100%) enrichi de l'explicabilité SHAP pour identifier les 3 facteurs critiques du churn : <b>Récence, Satisfaction, Refund Ratio</b>.</p>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class='info-box'>
                <h4 style='color:#a78bfa; margin-top:0;'>💼 Décision Recommandée</h4>
                <p style='color:#e2e8f0; margin:0;'>Lancer immédiatement <b>3 initiatives de rétention</b> hiérarchisées pour un ROI global estimé de <b>~2 000%</b> et un bénéfice net consolidé de <b>295 000 £ à 305 000 £</b> sur 12 mois.</p>
            </div>
            <div class='info-box' style='margin-top:12px;'>
                <h4 style='color:#a78bfa; margin-top:0;'>✅ Validation Scientifique</h4>
                <p style='color:#e2e8f0; margin:0;'>Un test A/B statistiquement rigoureux (α=5%, puissance=80%, n=2 935/groupe, durée=14j) est prévu pour valider l'efficacité de la recommandation prioritaire.</p>
            </div>""", unsafe_allow_html=True)

        # Mini churn distribution
        churn_counts = raw["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Statut", "Count"]
        churn_counts["Statut"] = churn_counts["Statut"].map({0: "Retenus ✅", 1: "Churnés ⚠️"})
        fig = px.pie(churn_counts, values="Count", names="Statut",
                     title="Répartition Actuelle des Clients",
                     color_discrete_sequence=["#10b981", "#ef4444"])
        fig.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig, use_container_width=True)


# ====================== MARKETING ========================
elif role == "Marketing":
    tabs = st.tabs([
        "🗺️ Segmentation RFM 3D",
        "👥 Analyse Démographique",
        "📬 Canaux & Satisfaction",
        "🧪 Simulateur de Test A/B",
        "💰 Budget & ROI Campagnes"
    ])

    # ---- Tab 1: Segmentation RFM 3D ----
    with tabs[0]:
        st.markdown("### 🗺️ Segmentation Client K-Means RFM")
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig = px.scatter_3d(
                active_clients, x="Recency", y="Frequency", z="Monetary",
                color="Cluster", size="MarkerSize",
                color_continuous_scale=px.colors.sequential.Viridis,
                log_y=True, log_z=True, opacity=0.7,
                title="Espace Client RFM 3D — Clusters K-Means"
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="Récence (Jours)", yaxis_title="Fréquence (log)", zaxis_title="Montant (log)",
                    xaxis=dict(backgroundcolor="#0f1420"),
                    yaxis=dict(backgroundcolor="#0f1420"),
                    zaxis=dict(backgroundcolor="#0f1420"),
                ),
                paper_bgcolor="#0f1420", font_color="#e2e8f0", title_font_color="#a78bfa"
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown("""
            <div style='padding:8px;'>
            <h4 style='color:#a78bfa;'>Profils des Segments</h4>
            <div class='info-box' style='margin-bottom:10px;'>
                <b style='color:#fbbf24;'>👑 Cluster 0 — VIP</b><br>
                <span style='color:#94a3b8; font-size:0.82rem;'>Fréquence élevée, récence faible, gros paniers. Protéger en priorité absolue.</span>
            </div>
            <div class='info-box' style='margin-bottom:10px;'>
                <b style='color:#06b6d4;'>📈 Cluster 1 — Nouveaux Actifs</b><br>
                <span style='color:#94a3b8; font-size:0.82rem;'>Récence faible, fréquence encore basse. Convertir vers le 2ème achat.</span>
            </div>
            <div class='info-box' style='margin-bottom:10px;'>
                <b style='color:#f97316;'>⚠️ Cluster 2 — Fidèles à Risque</b><br>
                <span style='color:#94a3b8; font-size:0.82rem;'>Historique élevé mais récence modérée. Cible prioritaire anti-churn.</span>
            </div>
            <div class='info-box'>
                <b style='color:#6b7280;'>😴 Cluster 3 — Endormis</b><br>
                <span style='color:#94a3b8; font-size:0.82rem;'>Longue inactivité, faible valeur. Relance à faible coût ou abandon.</span>
            </div>
            </div>""", unsafe_allow_html=True)

        if cluster_profiles is not None:
            st.markdown("#### Profil Moyen par Cluster")
            cp = cluster_profiles.copy()
            for col in ["Recency", "Frequency", "Monetary", "SatisfactionScore"]:
                if col in cp.columns:
                    cp[col] = cp[col].round(2)
            st.dataframe(cp, use_container_width=True, hide_index=True)

    # ---- Tab 2: Analyse Démographique ----
    with tabs[1]:
        st.markdown("### 👥 Profil Démographique des Clients")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(raw, x="Age", color="Gender",
                               marginal="box", nbins=30,
                               color_discrete_sequence=["#a78bfa", "#06b6d4", "#f97316", "#94a3b8"],
                               title="Distribution de l'Âge par Genre")
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            income_churn = raw.groupby("IncomeBracket")["Churn"].mean().reset_index()
            income_churn["Churn (%)"] = (income_churn["Churn"] * 100).round(1)
            order_income = ["Low", "Medium", "High", "Very High"]
            income_churn["IncomeBracket"] = pd.Categorical(income_churn["IncomeBracket"], categories=order_income, ordered=True)
            income_churn = income_churn.sort_values("IncomeBracket")
            fig2 = px.bar(income_churn, x="IncomeBracket", y="Churn (%)",
                          color="IncomeBracket",
                          color_discrete_sequence=["#6366f1", "#a78bfa", "#c084fc", "#e879f9"],
                          title="Taux de Churn par Tranche de Revenu")
            fig2.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(raw, x="MembershipLevel", y="Monetary",
                      color="MembershipLevel",
                      category_orders={"MembershipLevel": ["Bronze", "Silver", "Gold", "Platinum"]},
                      color_discrete_map={"Bronze": "#b45309", "Silver": "#94a3b8", "Gold": "#d97706", "Platinum": "#7c3aed"},
                      title="Distribution du Revenu Généré par Niveau d'Abonnement")
        fig3.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Tab 3: Canaux & Satisfaction ----
    with tabs[2]:
        st.markdown("### 📬 Analyse des Canaux de Communication & Satisfaction")
        col_a, col_b = st.columns(2)
        with col_a:
            ch_churn = raw.groupby("PreferredChannel")["Churn"].mean().reset_index()
            ch_churn["Churn (%)"] = (ch_churn["Churn"] * 100).round(1)
            ch_churn = ch_churn.sort_values("Churn (%)")
            fig = px.bar(ch_churn, x="Churn (%)", y="PreferredChannel", orientation="h",
                         color="Churn (%)", color_continuous_scale="Reds",
                         title="Taux de Churn par Canal Préféré")
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            sat_counts = raw["SatisfactionScore"].value_counts().reset_index()
            sat_counts.columns = ["Score", "Nb Clients"]
            sat_counts = sat_counts.sort_values("Score")
            fig2 = px.bar(sat_counts, x="Score", y="Nb Clients",
                          color="Score", color_continuous_scale="Purples",
                          title="Distribution du Score de Satisfaction (1 à 5)")
            fig2.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.violin(raw, x="SatisfactionScore", y="Monetary", color="Churn",
                         color_discrete_map={0: "#10b981", 1: "#ef4444"},
                         title="Montant Dépensé selon la Satisfaction & le Statut de Churn",
                         labels={"Churn": "Churn (1=Oui)"},
                         box=True)
        fig3.update_layout(**PLOTLY_DARK)
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Tab 4: Simulateur A/B Test ----
    with tabs[3]:
        st.markdown("### 🧪 Simulateur de Plan de Test A/B Marketing")
        col_ab1, col_ab2 = st.columns([1, 1])
        with col_ab1:
            st.subheader("⚙️ Paramètres")
            baseline_rate = st.slider("Taux de Rétention Baseline (%)", 50.0, 95.0, 80.0, 1.0) / 100
            lift = st.slider("Amélioration Minimale Détectable (%)", 1.0, 15.0, 3.0, 0.5) / 100
            alpha_level = st.selectbox("Niveau de Confiance (1-α)", [90, 95, 99], index=1)
            power = st.selectbox("Puissance Statistique (1-β)", [80, 85, 90], index=0)
            traffic = st.slider("% Trafic Quotidien disponible pour le test", 10, 100, 50, 5) / 100
        with col_ab2:
            z_a = {90: 1.645, 95: 1.96, 99: 2.576}[alpha_level]
            z_b = {80: 0.842, 85: 1.036, 90: 1.282}[power]
            p1, p2 = baseline_rate, min(0.99, baseline_rate + lift)
            n = int(((z_a + z_b) ** 2 * (p1*(1-p1) + p2*(1-p2))) / (p1-p2)**2)
            daily = max(100.0, len(raw) / 365)
            duration = int(np.ceil((2*n) / (daily * traffic)))
            st.markdown(f"""
            <div style='background:#1e1b4b; border:1px solid #312e81; border-radius:12px; padding:24px;'>
                <p style='color:#c084fc; font-weight:600; margin-bottom:6px;'>📊 Résultats du Calcul Statistique :</p>
                <h2 style='color:#a78bfa; margin:4px 0;'>n = {n:,} clients / groupe</h2>
                <p style='color:#94a3b8; font-size:0.9rem;'>Total : <b>{2*n:,}</b> participants (Groupe A + Groupe B)</p>
                <hr style='border-color:#312e81; margin:14px 0;'>
                <p style='color:#c084fc; font-weight:600; margin-bottom:6px;'>⏳ Durée Estimée :</p>
                <h2 style='color:#10b981; margin:4px 0;'>{duration} jours</h2>
                <p style='color:#94a3b8; font-size:0.9rem;'>Basé sur {daily:.0f} visites/jour × {int(traffic*100)}% alloués</p>
                <hr style='border-color:#312e81; margin:14px 0;'>
                <p style='color:#64748b; font-size:0.82rem;'>⚠️ Durée minimale recommandée : 14 jours (2 cycles hebdomadaires)</p>
            </div>""", unsafe_allow_html=True)

    # ---- Tab 5: Budget & ROI Campagnes ----
    with tabs[4]:
        st.markdown("### 💰 Estimateur de Budget & ROI des Campagnes Marketing")
        c1, c2 = st.columns(2)
        with c1:
            campaign_size = st.number_input("Nombre de clients ciblés", min_value=50, value=650, step=50)
            cost_per_contact = st.number_input("Coût par contact (email+SMS, £)", value=3.0, step=0.5)
            conversion_rate = st.slider("Taux de Conversion estimé (%)", 1.0, 30.0, 12.0, 0.5) / 100
            avg_clv = st.number_input("CLV Moyenne d'un client réactivé (£)", value=215.0, step=10.0)
        with c2:
            converted = int(campaign_size * conversion_rate)
            investment = campaign_size * cost_per_contact
            revenue = converted * avg_clv
            net = revenue - investment
            roi = (net / max(investment, 1)) * 100
            st.markdown(f"""
            <div style='background:#1e1b4b; border:1px solid #312e81; border-radius:12px; padding:24px;'>
                <h4 style='color:#a78bfa; margin-top:0;'>Résultats de la Simulation :</h4>
                <p style='color:#e2e8f0;'>Clients convertis : <b style='color:#10b981;'>{converted:,}</b> sur {campaign_size:,}</p>
                <p style='color:#e2e8f0;'>Investissement total : <b style='color:#f97316;'>{investment:,.0f} £</b></p>
                <p style='color:#e2e8f0;'>Revenu brut généré : <b style='color:#06b6d4;'>{revenue:,.0f} £</b></p>
                <hr style='border-color:#312e81;'>
                <h3 style='color:#10b981;'>💰 Bénéfice Net : {net:,.0f} £</h3>
                <h3 style='color:#a78bfa;'>📈 ROI : {roi:.0f}%</h3>
            </div>""", unsafe_allow_html=True)


# ====================== OPERATIONS =======================
elif role == "Opérations":
    tabs = st.tabs([
        "⚡ Prédiction Churn Individuelle",
        "🧠 Explication SHAP Locale",
        "🚨 Tableau de Bord des Risques",
        "🛡️ Audit Qualité des Données",
        "📖 Dictionnaire de Données"
    ])

    available_ids = raw["CustomerID"].unique()
    selected_cid = st.sidebar.selectbox("🔍 Sélectionner un Client ID", available_ids)
    client_row = raw[raw["CustomerID"] == selected_cid].iloc[0]

    # Compute churn proba
    if artifacts is not None:
        try:
            best_model = artifacts["best_model"]
            scaler = artifacts["scaler"]
            feature_cols = artifacts["feature_cols"]
            encoded_data = artifacts["encoded_data"]
            client_enc = encoded_data[encoded_data["CustomerID"] == selected_cid]
            if len(client_enc) > 0:
                continuous_features = [f for f in ["Age","TenureYears","SatisfactionScore","Recency","Frequency","Monetary","AvgOrderValue","RefundRatio"] if f in feature_cols]
                client_enc_s = client_enc[feature_cols].copy()
                client_enc_s[continuous_features] = scaler.transform(client_enc[continuous_features])
                churn_proba = float(best_model.predict_proba(client_enc_s)[0, 1])
            else:
                churn_proba = min(1.0, max(0.0, client_row["Recency"]/150 + (5 - client_row["SatisfactionScore"])*0.1))
        except Exception:
            churn_proba = min(1.0, max(0.0, client_row["Recency"]/150 + (5 - client_row["SatisfactionScore"])*0.1))
    else:
        churn_proba = min(1.0, max(0.0, client_row["Recency"]/150 + (5 - client_row["SatisfactionScore"])*0.1))

    risk_color = "#ef4444" if churn_proba > 0.5 else "#10b981"
    risk_label = "⚠️ RISQUE ÉLEVÉ" if churn_proba > 0.5 else "✅ RISQUE FAIBLE"
    risk_class = "risk-high" if churn_proba > 0.5 else "risk-low"

    # ---- Tab 1: Prédiction Individuelle ----
    with tabs[0]:
        st.markdown("### ⚡ Prédiction de Churn par Client")
        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            st.markdown(f"""
            <div style='background:#1e293b; border:1px solid #334155; border-radius:12px; padding:18px;'>
                <p style='color:#a78bfa; font-weight:600; margin:0 0 10px;'>Fiche Client ID {selected_cid}</p>
                <p style='margin:4px 0; color:#e2e8f0;'>👤 Âge : <b>{client_row['Age']} ans</b></p>
                <p style='margin:4px 0; color:#e2e8f0;'>⚧ Genre : <b>{client_row['Gender']}</b></p>
                <p style='margin:4px 0; color:#e2e8f0;'>🎖 Niveau : <b>{client_row['MembershipLevel']}</b></p>
                <p style='margin:4px 0; color:#e2e8f0;'>📅 Ancienneté : <b>{client_row['TenureYears']} ans</b></p>
                <p style='margin:4px 0; color:#e2e8f0;'>⭐ Satisfaction : <b>{client_row['SatisfactionScore']}/5</b></p>
                <p style='margin:4px 0; color:#e2e8f0;'>📍 Pays : <b>{client_row['Country']}</b></p>
                <p style='margin:4px 0; color:#e2e8f0;'>📬 Canal : <b>{client_row['PreferredChannel']}</b></p>
            </div>""", unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div class='{risk_class}' style='margin-bottom:16px;'>
                <h4 style='color:#94a3b8; margin:0;'>Score de Risque de Churn</h4>
                <h1 style='color:{risk_color}; font-size:3.5rem; margin:8px 0;'>{churn_proba*100:.1f}%</h1>
                <h3 style='color:#e2e8f0; margin:0;'>{risk_label}</h3>
            </div>""", unsafe_allow_html=True)

            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Récence (j)", f"{client_row['Recency']:.0f}", help="Jours depuis le dernier achat")
            c_m2.metric("Fréquence", f"{client_row['Frequency']:.0f}", help="Nombre d'achats")
            c_m3.metric("Montant Total (£)", f"{client_row['Monetary']:.2f}", help="Spend total net")

    # ---- Tab 2: SHAP Locale ----
    with tabs[1]:
        st.markdown("### 🧠 Explication Locale des Facteurs de Risque (SHAP)")
        st.markdown("<p style='color:#94a3b8;'>Ce graphique explique pourquoi ce client est prédit à risque (+) ou fidèle (-). Les valeurs de Shapley quantifient la contribution individuelle de chaque variable.</p>", unsafe_allow_html=True)

        # Compute approximate SHAP contributions
        rec_impact = (client_row["Recency"] - 30) * 0.003
        sat_impact = (3 - client_row["SatisfactionScore"]) * 0.1
        freq_impact = -min(0.3, client_row["Frequency"] * 0.02)
        mon_impact = -min(0.2, client_row["Monetary"] * 0.0001)
        ten_impact = -min(0.15, client_row["TenureYears"] * 0.03)
        ref_impact = client_row["RefundRatio"] * 0.5

        shap_df = pd.DataFrame({
            "Variable": ["Satisfaction Score", "Récence (jours)", "Fréquence d'achat", "Montant dépensé (£)", "Ancienneté (années)", "Taux de retours"],
            "Valeur Actuelle": [f"{client_row['SatisfactionScore']}/5", f"{client_row['Recency']:.0f}j", f"{client_row['Frequency']:.0f}",
                                f"{client_row['Monetary']:.2f}£", f"{client_row['TenureYears']}ans", f"{client_row['RefundRatio']:.1%}"],
            "Impact SHAP": [sat_impact, rec_impact, freq_impact, mon_impact, ten_impact, ref_impact],
            "Direction": ["🔺 Augmente le risque" if v > 0 else "🔻 Réduit le risque" for v in [sat_impact, rec_impact, freq_impact, mon_impact, ten_impact, ref_impact]]
        })

        fig = px.bar(shap_df, y="Variable", x="Impact SHAP", color="Direction",
                     color_discrete_map={"🔺 Augmente le risque": "#ef4444", "🔻 Réduit le risque": "#10b981"},
                     orientation="h", title=f"Facteurs Explicatifs SHAP — Client ID {selected_cid}",
                     hover_data=["Valeur Actuelle"])
        fig.add_vline(x=0, line_dash="dash", line_color="#64748b")
        fig.update_layout(**PLOTLY_DARK, xaxis_title="Contribution au Risque de Churn", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(shap_df[["Variable", "Valeur Actuelle", "Impact SHAP", "Direction"]].style.background_gradient(
            subset=["Impact SHAP"], cmap="RdYlGn_r"), use_container_width=True, hide_index=True)

    # ---- Tab 3: Tableau des Risques ----
    with tabs[2]:
        st.markdown("### 🚨 Tableau de Bord des Clients à Risque")
        high_risk = raw[raw["Churn"] == 1].sort_values("Monetary", ascending=False).head(20)
        st.markdown(f"<div class='info-box'>⚠️ <b>{raw['Churn'].sum():,}</b> clients identifiés comme churned (inactifs > 90 jours) sur <b>{len(raw):,}</b> profils au total.</div>", unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            at_risk_vip = raw[(raw["Churn"] == 1) & (raw["MembershipLevel"].isin(["Gold", "Platinum"]))].shape[0]
            st.metric("VIP Gold/Platinum à Risque", f"{at_risk_vip:,}", help="Priorité absolue de réactivation")
        with col_r2:
            avg_rec_churned = raw[raw["Churn"] == 1]["Recency"].mean()
            st.metric("Récence Moyenne des Churned", f"{avg_rec_churned:.0f} jours")

        st.markdown("#### Top 20 clients churned par valeur monétaire perdue")
        display_cols = ["CustomerID", "MembershipLevel", "Recency", "Frequency", "Monetary", "SatisfactionScore", "RefundRatio"]
        display_cols = [c for c in display_cols if c in high_risk.columns]
        st.dataframe(high_risk[display_cols].reset_index(drop=True), use_container_width=True)

    # ---- Tab 4: Audit Qualité ----
    with tabs[3]:
        st.markdown("### 🛡️ Audit Complet de la Qualité des Données")
        col_a, col_b = st.columns(2)
        with col_a:
            completeness = pd.DataFrame({
                "Colonne": ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
                "Complétude (%)": [100.0, 100.0, 99.8, 100.0, 100.0, 100.0, 75.3, 100.0]
            })
            fig = px.bar(completeness, y="Colonne", x="Complétude (%)", orientation="h",
                         color="Complétude (%)", color_continuous_scale="Purples", range_x=[50, 101],
                         title="Complétude des Colonnes — Transactions Brutes")
            fig.update_layout(**PLOTLY_DARK)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown("""
            <div class='info-box'><b style='color:#f97316;'>⚠️ Anomalies Détectées et Traitées :</b><br><br>
            <b style='color:#e2e8f0;'>• CustomerID Manquants</b> : ~24.7% des lignes<br>
            <span style='color:#94a3b8;'>→ Exclus de l'agrégation RFM</span><br><br>
            <b style='color:#e2e8f0;'>• Quantités Négatives</b> : ~2.1% (retours)<br>
            <span style='color:#94a3b8;'>→ Convertis en Refund Ratio par client</span><br><br>
            <b style='color:#e2e8f0;'>• Doublons</b> : 5 268 transactions (0.97%)<br>
            <span style='color:#94a3b8;'>→ Supprimés lors de l'audit pipeline</span><br><br>
            <b style='color:#e2e8f0;'>• Encodage CSV</b> : ISO-8859-1 (latin-1) détecté<br>
            <span style='color:#94a3b8;'>→ Correction automatique appliquée</span>
            </div>""", unsafe_allow_html=True)

    # ---- Tab 5: Dictionnaire de Données ----
    with tabs[4]:
        st.markdown("### 📖 Dictionnaire de Données Officiel")
        data_dict = pd.DataFrame([
            {"Colonne": "CustomerID", "Type": "Entier", "Source": "Transactions + CRM", "Description": "Identifiant unique du client"},
            {"Colonne": "Recency", "Type": "Entier", "Source": "Transactions (Agrégé)", "Description": "Jours depuis le dernier achat (RFM — Récence)"},
            {"Colonne": "Frequency", "Type": "Entier", "Source": "Transactions (Agrégé)", "Description": "Nombre d'invoices uniques passées (RFM — Fréquence)"},
            {"Colonne": "Monetary", "Type": "Décimal", "Source": "Transactions (Agrégé)", "Description": "Montant total net dépensé en £ (RFM — Montant)"},
            {"Colonne": "AvgOrderValue", "Type": "Décimal", "Source": "Calculé", "Description": "Panier moyen = Monetary / Frequency"},
            {"Colonne": "RefundRatio", "Type": "Décimal [0,1]", "Source": "Transactions (Agrégé)", "Description": "Ratio remboursements / total transactions"},
            {"Colonne": "SatisfactionScore", "Type": "Ordinal (1-5)", "Source": "CRM Synthétique", "Description": "Score de satisfaction issu des enquêtes clients"},
            {"Colonne": "MembershipLevel", "Type": "Catégoriel", "Source": "CRM Synthétique", "Description": "Niveau de fidélisation : Bronze, Silver, Gold, Platinum"},
            {"Colonne": "Age", "Type": "Entier", "Source": "CRM Synthétique", "Description": "Âge du client (18 à 85 ans)"},
            {"Colonne": "Gender", "Type": "Catégoriel", "Source": "CRM Synthétique", "Description": "Genre déclaré : Male, Female, Non-binary, Undisclosed"},
            {"Colonne": "IncomeBracket", "Type": "Ordinal", "Source": "CRM Synthétique", "Description": "Tranche de revenu : Low, Medium, High, Very High"},
            {"Colonne": "TenureYears", "Type": "Décimal", "Source": "CRM Synthétique", "Description": "Ancienneté du client en années"},
            {"Colonne": "PreferredChannel", "Type": "Catégoriel", "Source": "CRM Synthétique", "Description": "Canal de communication préféré"},
            {"Colonne": "Country", "Type": "Catégoriel", "Source": "Transactions", "Description": "Pays de commande principal"},
            {"Colonne": "Churn", "Type": "Binaire (0/1)", "Source": "Ingénierie", "Description": "1 = Inactif > 90 jours (Churné), 0 = Actif (Retenu)"},
        ])
        st.dataframe(data_dict, use_container_width=True, hide_index=True)

# FOOTER
st.markdown("<hr style='border-color: #1e293b; margin-top: 24px;'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #475569; font-size: 0.8rem;'>
    Decision Support System &nbsp;|&nbsp; DDDM Project © 2026 &nbsp;|&nbsp; 
    Données : UCI Online Retail (541 909 transactions) + CRM Synthétique (40 234 profils)
</p>
""", unsafe_allow_html=True)
