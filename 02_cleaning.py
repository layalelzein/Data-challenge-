# 02_cleaning.py
# EXÉCUTION : python3 02_cleaning.py

import pandas as pd
import re
from nltk.corpus import stopwords

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
df = pd.read_csv("reviews_raw.csv")
print(f"✅ Chargé : {df.shape[0]} reviews, {df.shape[1]} colonnes")

# ─────────────────────────────────────────────
# 1. SUPPRESSION DOUBLONS
# ─────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates(subset=["review_id", "cabinet"])
print(f"🧹 Doublons supprimés : {before - len(df)} | Restant : {len(df)}")

# ─────────────────────────────────────────────
# 2. RATING → numérique (déjà float, sécurité)
# ─────────────────────────────────────────────
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])
print(f"⭐ Ratings valides : {len(df)}")

# ─────────────────────────────────────────────
# 3. DATE → datetime + year + month
# ─────────────────────────────────────────────
mois_fr = {
    "janv": "Jan", "févr": "Feb", "mars": "Mar", "avr": "Apr",
    "mai": "May", "juin": "Jun", "juil": "Jul", "août": "Aug",
    "sept": "Sep", "oct": "Oct", "nov": "Nov", "déc": "Dec"
}

def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    s = str(date_str).strip()
    for fr, en in mois_fr.items():
        s = re.sub(fr + r"\.?", en, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    try:
        return pd.to_datetime(s, dayfirst=True)
    except:
        return pd.NaT

df["date"]  = df["date"].apply(parse_date)
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
print(f"📅 Dates parsées : {df['date'].notna().sum()} / {len(df)}")
print(f"📅 Années disponibles : {sorted(df['year'].dropna().unique().astype(int).tolist())}")

# ─────────────────────────────────────────────
# 4. NETTOYAGE PROS / CONS
# ─────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text)
    # Supprimer préfixes Glassdoor
    text = re.sub(r"^(Avantages|Inconvénients)\s*", "", text, flags=re.IGNORECASE)
    # Supprimer HTML résiduel
    text = re.sub(r"<[^>]+>", " ", text)
    # Nettoyer espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["pros_clean"] = df["pros"].apply(clean_text)
df["cons_clean"] = df["cons"].apply(clean_text)

# ─────────────────────────────────────────────
# 5. JOB TITLE → nettoyé
# ─────────────────────────────────────────────
def clean_job_title(info):
    if pd.isna(info):
        return "Unknown"
    info = str(info)
    for noise in ["Employé anonyme", "Stagiaire anonyme", "anonyme"]:
        info = info.replace(noise, "Unknown")
    info = re.sub(r"\s+", " ", info).strip()
    return info if info else "Unknown"

df["job_title"] = df["employee_info"].apply(clean_job_title)

# ─────────────────────────────────────────────
# 6. SÉNIORITÉ → catégorie utile pour analyse
# ─────────────────────────────────────────────
def categorize_seniority(title):
    title = str(title).lower()
    if any(x in title for x in ["intern", "stagiaire", "stage"]):
        return "Intern"
    elif any(x in title for x in ["analyst", "associate", "assistant", "junior", "consultant i", "consultant ii"]):
        return "Junior"
    elif any(x in title for x in ["senior", "manager", "lead", "principal"]):
        return "Senior"
    elif any(x in title for x in ["director", "partner", "vp ", "vice president", "chief", "head"]):
        return "Executive"
    else:
        return "Mid-level"

df["seniority"] = df["job_title"].apply(categorize_seniority)

# ─────────────────────────────────────────────
# 7. TEXTE COMBINÉ POUR NLP
# ─────────────────────────────────────────────
df["full_text"] = (df["pros_clean"] + " " + df["cons_clean"]).str.strip()

# ─────────────────────────────────────────────
# 8. NORMALISATION CABINET
# ─────────────────────────────────────────────
cabinet_map = {"Deloitte": "Deloitte", "Pwc": "PwC", "Ey": "EY", "Kpmg": "KPMG"}
df["cabinet"] = df["cabinet"].str.strip().str.title()
df["cabinet"] = df["cabinet"].map(cabinet_map).fillna(df["cabinet"])
print(f"\n🏢 Reviews par cabinet :")
print(df["cabinet"].value_counts().to_string())

# ─────────────────────────────────────────────
# 9. COLONNES FINALES
# ─────────────────────────────────────────────
df_clean = df[[
    "cabinet", "review_id", "rating", "date", "year", "month",
    "title", "job_title", "seniority", "status",
    "recommends", "ceo_approval", "business_outlook",
    "pros_clean", "cons_clean", "full_text"
]].reset_index(drop=True)

# ─────────────────────────────────────────────
# 10. STATS FINALES
# ─────────────────────────────────────────────
print(f"\n📊 Shape finale : {df_clean.shape}")
print(f"\n⭐ Note moyenne par cabinet :")
print(df_clean.groupby("cabinet")["rating"].mean().round(2).sort_values(ascending=False).to_string())
print(f"\n👔 Répartition séniorité :")
print(df_clean["seniority"].value_counts().to_string())
print(f"\n📝 Pros vides : {(df_clean['pros_clean'] == '').sum()}")
print(f"📝 Cons vides : {(df_clean['cons_clean'] == '').sum()}")

# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────
df_clean.to_csv("reviews_clean.csv", index=False, encoding="utf-8-sig")
print("\n💾 Sauvegardé → reviews_clean.csv ✅")