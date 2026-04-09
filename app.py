# app.py
# EXÉCUTION : streamlit run app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Big Four — Analyse Employés",
    page_icon="📊",
    layout="wide"
)

COLORS = {
    "Deloitte": "#86BC25",
    "PwC":      "#D04A02",
    "EY":       "#2E2D62",
    "KPMG":     "#00338D"
}

CABINETS = ["Deloitte", "PwC", "EY", "KPMG"]

# ─────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("reviews_clean.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df

df = load_data()

# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────
@st.cache_data
def get_stopwords():
    import nltk
    stop_en = set(stopwords.words("english"))
    stop_fr = set(stopwords.words("french"))
    return stop_en | stop_fr | {
        "work", "company", "place", "good", "great", "working",
        "people", "really", "make", "get", "lot", "also", "one",
        "well", "can", "time", "much", "many", "job", "firm",
        "big", "four", "deloitte", "pwc", "ey", "kpmg",
        "na", "n", "none", "nothing", "experience"
    }

STOPWORDS = get_stopwords()

# ─────────────────────────────────────────────
# TFIDF FUNCTION
# ─────────────────────────────────────────────
@st.cache_data
def get_tfidf_words(texts_series, n=10):
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=5
    )
    try:
        matrix = vectorizer.fit_transform(texts_series.fillna(""))
        scores = matrix.mean(axis=0).A1
        words  = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:n]
        return [(words[i], round(float(scores[i]), 4)) for i in top_idx]
    except:
        return []

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📊 Big Four — Analyse des avis employés")
st.markdown("**Persona** : Étudiant M2 cherchant son premier employeur dans le conseil")
st.markdown("**Source** : Glassdoor · 217 462 avis · 2008–2024")
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR — FILTRES
# ─────────────────────────────────────────────
st.sidebar.title("🔧 Filtres")

cabinets_selected = st.sidebar.multiselect(
    "Cabinets",
    options=CABINETS,
    default=CABINETS
)

years = sorted(df["year"].dropna().unique().astype(int).tolist())
year_range = st.sidebar.slider(
    "Période",
    min_value=min(years),
    max_value=max(years),
    value=(2015, 2024)
)

seniority_options = ["Intern", "Junior", "Mid-level", "Senior", "Executive"]
seniority_selected = st.sidebar.multiselect(
    "Séniorité",
    options=seniority_options,
    default=seniority_options
)

# Application des filtres
mask = (
    df["cabinet"].isin(cabinets_selected) &
    df["year"].between(year_range[0], year_range[1]) &
    df["seniority"].isin(seniority_selected)
)
dff = df[mask].copy()

st.sidebar.markdown(f"**{len(dff):,} avis** sélectionnés")

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.subheader("🏆 Vue d'ensemble")
cols = st.columns(len(cabinets_selected))

for i, cab in enumerate(cabinets_selected):
    sub = dff[dff["cabinet"] == cab]
    avg = sub["rating"].mean()
    n   = len(sub)
    rec = dff_rec = sub[sub["recommends"] != "unknown"]
    taux = (rec["recommends"] == "positive").sum() / len(rec) * 100 if len(rec) > 0 else 0

    with cols[i]:
        color = COLORS.get(cab, "#333")
        st.markdown(f"""
        <div style="background:{color}15; border-left:4px solid {color};
                    padding:12px; border-radius:6px; margin-bottom:8px">
            <h4 style="margin:0;color:{color}">{cab}</h4>
            <h2 style="margin:4px 0">⭐ {avg:.2f}</h2>
            <p style="margin:0;font-size:13px">{n:,} avis · {taux:.0f}% recommandent</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# ONGLETS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Notes & Tendances",
    "💬 Ce qu'ils disent",
    "🗺️ Thèmes clés",
    "👔 Profils"
])

# ══════════════════════════════════════════════
# TAB 1 — NOTES & TENDANCES
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        avg_rating = dff.groupby("cabinet")["rating"].mean().round(2).reset_index()
        avg_rating = avg_rating.sort_values("rating", ascending=False)

        fig1 = px.bar(
            avg_rating,
            x="cabinet", y="rating",
            color="cabinet",
            color_discrete_map=COLORS,
            text="rating",
            title="Note moyenne par cabinet",
            range_y=[3.0, 4.5],
            labels={"rating": "Note moyenne", "cabinet": ""}
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, plot_bgcolor="white", height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        rating_dist = dff.groupby(["cabinet", "rating"]).size().reset_index(name="count")
        rating_dist["pct"] = rating_dist.groupby("cabinet")["count"].transform(
            lambda x: x / x.sum() * 100
        ).round(1)

        fig2 = px.bar(
            rating_dist,
            x="rating", y="pct",
            color="cabinet",
            barmode="group",
            color_discrete_map=COLORS,
            title="Distribution des notes (%)",
            labels={"pct": "%", "rating": "Note", "cabinet": ""}
        )
        fig2.update_layout(plot_bgcolor="white", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Évolution temporelle
    evolution = dff.groupby(["year", "cabinet"])["rating"].mean().round(2).reset_index()
    fig3 = px.line(
        evolution,
        x="year", y="rating",
        color="cabinet",
        color_discrete_map=COLORS,
        markers=True,
        title="Évolution de la note moyenne par année",
        labels={"rating": "Note moyenne", "year": "Année", "cabinet": ""}
    )
    fig3.update_layout(plot_bgcolor="white", hovermode="x unified", height=380)
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — CE QU'ILS DISENT
# ══════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💚 Ce qu'ils aiment")
        cabinet_choice = st.selectbox("Cabinet", cabinets_selected, key="pros_cab")
        sub = dff[dff["cabinet"] == cabinet_choice]
        top_pros = get_tfidf_words(sub["pros_clean"], n=12)
        if top_pros:
            words  = [w for w, s in top_pros][::-1]
            scores = [s for w, s in top_pros][::-1]
            fig4 = go.Figure(go.Bar(
                x=scores, y=words,
                orientation="h",
                marker_color=COLORS.get(cabinet_choice, "#86BC25")
            ))
            fig4.update_layout(
                title=f"Top mots — Avantages ({cabinet_choice})",
                plot_bgcolor="white", height=400,
                xaxis_title="Score TF-IDF",
                margin=dict(l=150)
            )
            st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("### 🔴 Ce qu'ils reprochent")
        cabinet_choice2 = st.selectbox("Cabinet", cabinets_selected, key="cons_cab")
        sub2 = dff[dff["cabinet"] == cabinet_choice2]
        top_cons = get_tfidf_words(sub2["cons_clean"], n=12)
        if top_cons:
            words  = [w for w, s in top_cons][::-1]
            scores = [s for w, s in top_cons][::-1]
            fig5 = go.Figure(go.Bar(
                x=scores, y=words,
                orientation="h",
                marker_color="#E74C3C"
            ))
            fig5.update_layout(
                title=f"Top mots — Inconvénients ({cabinet_choice2})",
                plot_bgcolor="white", height=400,
                xaxis_title="Score TF-IDF",
                margin=dict(l=150)
            )
            st.plotly_chart(fig5, use_container_width=True)

    # Taux de recommandation
    st.divider()
    rec = dff[dff["recommends"] != "unknown"].copy()
    if len(rec) > 0:
        rec_rate = rec.groupby("cabinet").apply(
            lambda x: (x["recommends"] == "positive").sum() / len(x) * 100,
            include_groups=False
        ).round(1).reset_index()
        rec_rate.columns = ["cabinet", "taux"]

        fig8 = px.bar(
            rec_rate.sort_values("taux", ascending=False),
            x="cabinet", y="taux",
            color="cabinet",
            color_discrete_map=COLORS,
            text="taux",
            title="👍 Taux de recommandation (%)",
            range_y=[0, 100],
            labels={"taux": "% Recommandation", "cabinet": ""}
        )
        fig8.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig8.update_layout(showlegend=False, plot_bgcolor="white", height=350)
        st.plotly_chart(fig8, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — HEATMAP THÈMES
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🗺️ Forces et faiblesses par thème")
    st.caption("Score = fréquence dans les avantages − fréquence dans les inconvénients. Vert = point fort, Rouge = point faible.")

    THEMES = {
        "Salaire":      ["salary", "pay", "compensation", "wage", "bonus"],
        "WLB":          ["work life", "work-life", "balance", "hours", "overtime", "burnout"],
        "Culture":      ["culture", "diversity", "inclusive", "environment", "atmosphere"],
        "Évolution":    ["growth", "promotion", "career", "learning", "development", "training"],
        "Management":   ["management", "manager", "leadership", "leader", "support"],
        "Opportunités": ["opportunity", "project", "client", "international", "exposure", "network"]
    }

    def theme_score(text, keywords):
        if pd.isna(text): return 0
        text = str(text).lower()
        return sum(1 for k in keywords if k in text)

heatmap_data = []
for cab in cabinets_selected:
    sub = dff[dff["cabinet"] == cab]
    row = {"cabinet": cab}
    for theme, keywords in THEMES.items():
        pros_s = sub["pros_clean"].apply(lambda x: theme_score(x, keywords)).mean()
        cons_s = sub["cons_clean"].apply(lambda x: theme_score(x, keywords)).mean()
        total  = pros_s + cons_s
        # Normalisation -1 à +1
        score  = (pros_s - cons_s) / total if total > 0 else 0
        row[theme] = round(score, 3)
    heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index("cabinet")

    fig6 = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1, 
        text=heatmap_df.values.round(3),
        texttemplate="%{text}",
        colorbar_title="Score"
    ))
    fig6.update_layout(
        height=350,
        xaxis_title="Thème",
        yaxis_title="Cabinet"
    )
    st.plotly_chart(fig6, use_container_width=True)

    # Insight automatique
    st.markdown("#### 💡 Lecture rapide")
    for cab in cabinets_selected:
        if cab in heatmap_df.index:
            row = heatmap_df.loc[cab]
            best  = row.idxmax()
            worst = row.idxmin()
            st.markdown(f"**{cab}** → ✅ Point fort : **{best}** · ⚠️ Point faible : **{worst}**")

# ══════════════════════════════════════════════
# TAB 4 — PROFILS
# ══════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        seniority_order = ["Intern", "Junior", "Mid-level", "Senior", "Executive"]
        sen_rating = dff.groupby(["cabinet", "seniority"])["rating"].mean().round(2).reset_index()
        fig7 = px.line(
            sen_rating,
            x="seniority", y="rating",
            color="cabinet",
            color_discrete_map=COLORS,
            markers=True,
            category_orders={"seniority": seniority_order},
            title="Note par niveau de séniorité",
            labels={"rating": "Note", "seniority": "", "cabinet": ""}
        )
        fig7.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        vol_year = dff.groupby(["year", "cabinet"]).size().reset_index(name="count")
        fig9 = px.bar(
            vol_year,
            x="year", y="count",
            color="cabinet",
            color_discrete_map=COLORS,
            barmode="stack",
            title="Volume d'avis par année",
            labels={"count": "Nombre d'avis", "year": "Année", "cabinet": ""}
        )
        fig9.update_layout(plot_bgcolor="white", height=380)
        st.plotly_chart(fig9, use_container_width=True)

    # Répartition séniorité par cabinet
    sen_dist = dff.groupby(["cabinet", "seniority"]).size().reset_index(name="count")
    sen_dist["pct"] = sen_dist.groupby("cabinet")["count"].transform(
        lambda x: x / x.sum() * 100
    ).round(1)

    fig10 = px.bar(
        sen_dist,
        x="cabinet", y="pct",
        color="seniority",
        barmode="stack",
        title="Répartition des profils par cabinet (%)",
        labels={"pct": "%", "cabinet": "", "seniority": "Séniorité"},
        category_orders={"seniority": seniority_order}
    )
    fig10.update_layout(plot_bgcolor="white", height=350)
    st.plotly_chart(fig10, use_container_width=True)