# 03_eda_nlp.py
# OBJECTIF : EDA + NLP → insights actionnables
# EXÉCUTION : python3 03_eda_nlp.py

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import os

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
df = pd.read_csv("reviews_clean.csv")
print(f"✅ Chargé : {df.shape[0]} reviews")

CABINETS = ["Deloitte", "PwC", "EY", "KPMG"]
COLORS   = {
    "Deloitte": "#86BC25",
    "PwC":      "#D04A02",
    "EY":       "#FFE600",
    "KPMG":     "#00338D"
}

# ─────────────────────────────────────────────
# STOPWORDS
# ─────────────────────────────────────────────
stop_en = set(stopwords.words("english"))
stop_fr = set(stopwords.words("french"))
STOPWORDS = stop_en | stop_fr | {
    "work", "company", "place", "good", "great", "working",
    "people", "really", "make", "get", "lot", "also", "one",
    "well", "can", "time", "much", "many", "job", "firm",
    "big", "four", "deloitte", "pwc", "ey", "kpmg",
    "na", "n", "none", "nothing", "experience"
}

def tokenize(text):
    if pd.isna(text) or text == "":
        return []
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", str(text).lower())
    return [t for t in tokens if t not in STOPWORDS]

# ─────────────────────────────────────────────
# 1. NOTE MOYENNE PAR CABINET (bar chart)
# ─────────────────────────────────────────────
avg_rating = df.groupby("cabinet")["rating"].mean().round(2).reset_index()
avg_rating = avg_rating.sort_values("rating", ascending=False)

fig1 = px.bar(
    avg_rating,
    x="cabinet", y="rating",
    color="cabinet",
    color_discrete_map=COLORS,
    text="rating",
    title="⭐ Note moyenne par cabinet",
    labels={"rating": "Note moyenne", "cabinet": "Cabinet"},
    range_y=[3.5, 4.2]
)
fig1.update_traces(textposition="outside", textfont_size=14)
fig1.update_layout(showlegend=False, plot_bgcolor="white")
fig1.write_html("outputs/01_avg_rating.html")
print("✅ 01_avg_rating.html")

# ─────────────────────────────────────────────
# 2. DISTRIBUTION DES NOTES (histogramme groupé)
# ─────────────────────────────────────────────
rating_dist = df.groupby(["cabinet", "rating"]).size().reset_index(name="count")
rating_dist["pct"] = rating_dist.groupby("cabinet")["count"].transform(lambda x: x / x.sum() * 100).round(1)

fig2 = px.bar(
    rating_dist,
    x="rating", y="pct",
    color="cabinet",
    barmode="group",
    color_discrete_map=COLORS,
    title="📊 Distribution des notes par cabinet (%)",
    labels={"pct": "% des avis", "rating": "Note", "cabinet": "Cabinet"},
    text="pct"
)
fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig2.update_layout(plot_bgcolor="white")
fig2.write_html("outputs/02_rating_distribution.html")
print("✅ 02_rating_distribution.html")

# ─────────────────────────────────────────────
# 3. ÉVOLUTION TEMPORELLE (line chart)
# ─────────────────────────────────────────────
df_time = df[df["year"] >= 2015].copy()
evolution = df_time.groupby(["year", "cabinet"])["rating"].mean().round(2).reset_index()

fig3 = px.line(
    evolution,
    x="year", y="rating",
    color="cabinet",
    color_discrete_map=COLORS,
    markers=True,
    title="📈 Évolution de la note moyenne (2015-2024)",
    labels={"rating": "Note moyenne", "year": "Année", "cabinet": "Cabinet"}
)
fig3.update_layout(plot_bgcolor="white", hovermode="x unified")
fig3.write_html("outputs/03_evolution.html")
print("✅ 03_evolution.html")

# ─────────────────────────────────────────────
# 4. TOP MOTS PROS PAR CABINET (TF-IDF)
# ─────────────────────────────────────────────
def get_top_tfidf_words(texts, n=15):
    """Mots les plus spécifiques via TF-IDF"""
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=5
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts.fillna(""))
        scores = tfidf_matrix.mean(axis=0).A1
        words  = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:n]
        return [(words[i], round(scores[i], 4)) for i in top_idx]
    except:
        return []

print("\n🔍 Calcul TF-IDF pros/cons par cabinet...")
tfidf_pros = {}
tfidf_cons = {}

for cab in CABINETS:
    sub = df[df["cabinet"] == cab]
    tfidf_pros[cab] = get_top_tfidf_words(sub["pros_clean"])
    tfidf_cons[cab] = get_top_tfidf_words(sub["cons_clean"])
    print(f"   ✅ {cab} : {len(tfidf_pros[cab])} mots pros, {len(tfidf_cons[cab])} mots cons")

# Graphique top mots pros
fig4 = make_subplots(
    rows=2, cols=2,
    subplot_titles=CABINETS,
    shared_xaxes=False
)
positions = [(1,1),(1,2),(2,1),(2,2)]
for idx, cab in enumerate(CABINETS):
    r, c = positions[idx]
    words_scores = tfidf_pros[cab][:10]
    words  = [w for w, s in words_scores][::-1]
    scores = [s for w, s in words_scores][::-1]
    fig4.add_trace(
        go.Bar(
            x=scores, y=words,
            orientation="h",
            marker_color=COLORS[cab],
            name=cab, showlegend=False
        ),
        row=r, col=c
    )
fig4.update_layout(
    title_text="💚 Top mots — AVANTAGES par cabinet",
    height=700,
    plot_bgcolor="white"
)
fig4.write_html("outputs/04_top_pros.html")
print("✅ 04_top_pros.html")

# Graphique top mots cons
fig5 = make_subplots(
    rows=2, cols=2,
    subplot_titles=CABINETS,
    shared_xaxes=False
)
for idx, cab in enumerate(CABINETS):
    r, c = positions[idx]
    words_scores = tfidf_cons[cab][:10]
    words  = [w for w, s in words_scores][::-1]
    scores = [s for w, s in words_scores][::-1]
    fig5.add_trace(
        go.Bar(
            x=scores, y=words,
            orientation="h",
            marker_color=COLORS[cab],
            name=cab, showlegend=False
        ),
        row=r, col=c
    )
fig5.update_layout(
    title_text="🔴 Top mots — INCONVÉNIENTS par cabinet",
    height=700,
    plot_bgcolor="white"
)
fig5.write_html("outputs/05_top_cons.html")
print("✅ 05_top_cons.html")

# ─────────────────────────────────────────────
# 5. HEATMAP THÈMES CLÉS
# ─────────────────────────────────────────────
THEMES = {
    "Salaire":      ["salary", "pay", "compensation", "wage", "remuneration", "bonus"],
    "WLB":          ["work life", "work-life", "balance", "hours", "overtime", "burnout"],
    "Culture":      ["culture", "diversity", "inclusive", "environment", "atmosphere", "toxic"],
    "Évolution":    ["growth", "promotion", "career", "learning", "development", "training"],
    "Management":   ["management", "manager", "leadership", "leader", "senior", "support"],
    "Opportunités": ["opportunity", "project", "client", "international", "exposure", "network"]
}

def theme_score(text, keywords):
    if pd.isna(text):
        return 0
    text = str(text).lower()
    return sum(1 for k in keywords if k in text)

# Score moyen par thème et cabinet
heatmap_data = []
for cab in CABINETS:
    sub = df[df["cabinet"] == cab]
    row = {"cabinet": cab}
    for theme, keywords in THEMES.items():
        # Score pros (positif) - cons (négatif)
        pros_score = sub["pros_clean"].apply(lambda x: theme_score(x, keywords)).mean()
        cons_score = sub["cons_clean"].apply(lambda x: theme_score(x, keywords)).mean()
        row[theme] = round(pros_score - cons_score, 4)
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data).set_index("cabinet")

fig6 = go.Figure(data=go.Heatmap(
    z=heatmap_df.values,
    x=heatmap_df.columns.tolist(),
    y=heatmap_df.index.tolist(),
    colorscale="RdYlGn",
    text=heatmap_df.values.round(3),
    texttemplate="%{text}",
    colorbar_title="Score\n(pros-cons)"
))
fig6.update_layout(
    title="🗺️ Heatmap : Forces vs Faiblesses par thème",
    xaxis_title="Thème",
    yaxis_title="Cabinet",
    height=400
)
fig6.write_html("outputs/06_heatmap_themes.html")
print("✅ 06_heatmap_themes.html")

# ─────────────────────────────────────────────
# 6. NOTE PAR SÉNIORITÉ ET CABINET
# ─────────────────────────────────────────────
seniority_order = ["Intern", "Junior", "Mid-level", "Senior", "Executive"]
seniority_rating = df.groupby(["cabinet", "seniority"])["rating"].mean().round(2).reset_index()

fig7 = px.line(
    seniority_rating,
    x="seniority", y="rating",
    color="cabinet",
    color_discrete_map=COLORS,
    markers=True,
    category_orders={"seniority": seniority_order},
    title="👔 Note moyenne par niveau de séniorité",
    labels={"rating": "Note moyenne", "seniority": "Séniorité", "cabinet": "Cabinet"}
)
fig7.update_layout(plot_bgcolor="white", hovermode="x unified")
fig7.write_html("outputs/07_seniority_rating.html")
print("✅ 07_seniority_rating.html")

# ─────────────────────────────────────────────
# 7. TAUX DE RECOMMANDATION PAR CABINET
# ─────────────────────────────────────────────
rec = df[df["recommends"] != "unknown"].copy()
rec_rate = rec.groupby("cabinet").apply(
    lambda x: (x["recommends"] == "positive").sum() / len(x) * 100
).round(1).reset_index()
rec_rate.columns = ["cabinet", "taux_recommandation"]

fig8 = px.bar(
    rec_rate.sort_values("taux_recommandation", ascending=False),
    x="cabinet", y="taux_recommandation",
    color="cabinet",
    color_discrete_map=COLORS,
    text="taux_recommandation",
    title="👍 Taux de recommandation par cabinet (%)",
    labels={"taux_recommandation": "% Recommandation", "cabinet": "Cabinet"},
    range_y=[0, 100]
)
fig8.update_traces(texttemplate="%{text}%", textposition="outside")
fig8.update_layout(showlegend=False, plot_bgcolor="white")
fig8.write_html("outputs/08_recommandation.html")
print("✅ 08_recommandation.html")

# ─────────────────────────────────────────────
# RÉSUMÉ INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("📋 INSIGHTS CLÉS")
print("="*50)

print(f"\n⭐ Note moyenne par cabinet :")
for _, row in avg_rating.iterrows():
    print(f"   {row['cabinet']:10} : {row['rating']}")

print(f"\n👍 Taux de recommandation :")
for _, row in rec_rate.sort_values("taux_recommandation", ascending=False).iterrows():
    print(f"   {row['cabinet']:10} : {row['taux_recommandation']}%")

print(f"\n🔍 Top 5 mots pros par cabinet :")
for cab in CABINETS:
    top5 = [w for w, s in tfidf_pros[cab][:5]]
    print(f"   {cab:10} : {', '.join(top5)}")

print(f"\n🔴 Top 5 mots cons par cabinet :")
for cab in CABINETS:
    top5 = [w for w, s in tfidf_cons[cab][:5]]
    print(f"   {cab:10} : {', '.join(top5)}")

print("\n✅ Tous les graphiques sauvegardés dans outputs/")