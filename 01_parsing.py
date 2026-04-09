# 01_parsing.py
# OBJECTIF : Parser les fichiers HTML des reviews Big Four → DataFrame propre
# EXÉCUTION : python3 01_parsing.py

import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
CABINETS = ["Deloitte", "PwC", "EY", "KPMG"]

# ─────────────────────────────────────────────
# FONCTION PRINCIPALE DE PARSING
# ─────────────────────────────────────────────
def parse_review_file(filepath: Path, cabinet: str) -> list:

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    reviews = []

    for li in soup.find_all("li", id=re.compile(r"^empReview_")):
        review = {"cabinet": cabinet}

        # ── ID ───────────────────────────────────────────────────────
        review["review_id"] = li.get("id", "").replace("empReview_", "")

        # ── RATING GLOBAL ────────────────────────────────────────────
        rating_span = li.find(
            "span",
            class_="review-details__review-details-module__overallRating"
        )
        if rating_span:
            raw = rating_span.get_text(strip=True).replace(",", ".")
            try:
                review["rating"] = float(raw)
            except:
                review["rating"] = None
        else:
            review["rating"] = None

        # ── DATE ─────────────────────────────────────────────────────
        date_tag = li.find("time")
        if date_tag:
            review["date"] = date_tag.get("datetime") or date_tag.get_text(strip=True)
        else:
            text = li.get_text()
            date_match = re.search(r"\d{1,2}\s+\w+\.?\s+\d{4}", text)
            review["date"] = date_match.group(0) if date_match else None

        # ── TITRE DE LA REVIEW ───────────────────────────────────────
        title_div = li.find(
            "div",
            class_="review-details__review-details-module__titleHeadline"
        )
        if title_div:
            review["title"] = title_div.get_text(strip=True)
        else:
            # fallback regex sur classe partielle
            title_div = li.find("div", class_=re.compile(r"titleHeadline"))
            review["title"] = title_div.get_text(strip=True) if title_div else None

        # ── POSTE / EMPLOYEE INFO ────────────────────────────────────
        employee_div = li.find(
            "div",
            class_="review-details__review-details-module__employeeContainer"
        )
        if not employee_div:
            employee_div = li.find("div", class_=re.compile(r"employeeContainer"))
        review["employee_info"] = employee_div.get_text(separator=" ", strip=True) if employee_div else None

        # ── STATUT EMPLOYÉ ───────────────────────────────────────────
        emp_info = review.get("employee_info", "") or ""
        if "actuel" in emp_info.lower():
            review["status"] = "current"
        elif "ancien" in emp_info.lower() or "ex-" in emp_info.lower():
            review["status"] = "former"
        else:
            review["status"] = "unknown"

        # ── ICÔNES : RECOMMANDE / PDG / PERSPECTIVE ──────────────────
        icon_container = li.find("div", class_=re.compile(r"iconContainer"))
        if icon_container:
            review["recommends"]        = _parse_icon(icon_container, "Recommand")
            review["ceo_approval"]      = _parse_icon(icon_container, "PDG")
            review["business_outlook"]  = _parse_icon(icon_container, "commerciale")
        else:
            review["recommends"]        = None
            review["ceo_approval"]      = None
            review["business_outlook"]  = None

        # ── PROS ─────────────────────────────────────────────────────
        pros_div = li.find(
            "div",
            class_="review-details__review-details-module__pro"
        )
        if not pros_div:
            pros_div = li.find("div", class_=re.compile(r"module__pro\b"))
        review["pros"] = pros_div.get_text(separator=" ", strip=True) if pros_div else None

        # ── CONS ─────────────────────────────────────────────────────
        cons_div = li.find(
            "div",
            class_="review-details__review-details-module__con"
        )
        if not cons_div:
            cons_div = li.find("div", class_=re.compile(r"module__con\b"))
        review["cons"] = cons_div.get_text(separator=" ", strip=True) if cons_div else None

        reviews.append(review)

    return reviews


# ─────────────────────────────────────────────
# PARSING ICÔNES
# ─────────────────────────────────────────────
def _parse_icon(container, keyword: str):
    """
    Détecte positif/négatif/neutre pour chaque icône.
    Stratégie : aria-label > data-* > classes CSS > fallback unknown
    """
    for elem in container.find_all(True):
        text = elem.get_text(strip=True)
        if keyword.lower() in text.lower():
            parent = elem.parent
            if parent:
                for sibling in parent.find_all(True):
                    aria      = sibling.get("aria-label", "").lower()
                    data_test = sibling.get("data-test", "").lower()
                    class_str = " ".join(sibling.get("class", [])).lower()
                    combined  = aria + " " + data_test + " " + class_str

                    if any(x in combined for x in ["negative", "notok", "false", "no-", "bad"]):
                        return "negative"
                    elif any(x in combined for x in ["neutral", "ok", "mixed", "minus"]):
                        return "neutral"
                    elif any(x in combined for x in ["positive", "yes", "true", "good"]):
                        return "positive"
    return "unknown"


# ─────────────────────────────────────────────
# BOUCLE SUR TOUS LES FICHIERS
# ─────────────────────────────────────────────
def parse_all_files() -> pd.DataFrame:
    all_reviews = []

    for cabinet in CABINETS:
        cabinet_path = DATA_DIR / cabinet
        if not cabinet_path.exists():
            print(f"⚠️  Dossier introuvable : {cabinet_path}")
            continue

        html_files = list(cabinet_path.glob("*.html")) + list(cabinet_path.glob("*.htm"))
        print(f"📁 {cabinet} : {len(html_files)} fichier(s) trouvé(s)")

        for filepath in html_files:
            try:
                reviews = parse_review_file(filepath, cabinet)
                all_reviews.extend(reviews)
                print(f"   ✅ {filepath.name} → {len(reviews)} reviews")
            except Exception as e:
                print(f"   ❌ Erreur sur {filepath.name} : {e}")

    df = pd.DataFrame(all_reviews)
    print(f"\n✅ Total reviews parsées : {len(df)}")
    return df


# ─────────────────────────────────────────────
# EXÉCUTION
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = parse_all_files()

    print("\n📊 Aperçu :")
    print(df.head(3).to_string())
    print(f"\n📐 Shape : {df.shape}")
    print(f"📋 Colonnes : {list(df.columns)}")

    # Vérification ratings
    print(f"\n⭐ Distribution ratings :")
    print(df["rating"].value_counts().sort_index())
    print(f"❓ Ratings manquants : {df['rating'].isna().sum()}")

    # Vérification icônes
    print(f"\n👍 Recommends :")
    print(df["recommends"].value_counts())

    df.to_csv("reviews_raw.csv", index=False, encoding="utf-8-sig")
    print("\n💾 Sauvegardé → reviews_raw.csv")