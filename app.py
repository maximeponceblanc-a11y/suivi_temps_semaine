"""
Suivi d'activité hebdomadaire — Dashboard Streamlit
=====================================================

Lancer avec :  streamlit run app.py

Le fichier Excel source doit contenir au minimum les feuilles :
  - temps_reel_operateur  (pointages des opérateurs)
  - ordres_fabrication    (ordres de fabrication / dossiers)
"""

import io
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Suivi d'activité hebdomadaire",
    page_icon="📊",
    layout="wide",
)

MS_PER_HOUR = 3_600_000  # les durées "temps_devis" / "temps_operateurs" / "duree"
                         # sont stockées en millisecondes dans le fichier source

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Lecture du fichier Excel…")
def load_data(file_bytes: bytes):
    """Charge et prépare les feuilles nécessaires du fichier Excel."""
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")

    # --- Feuille des pointages -------------------------------------------------
    df_pointages = pd.read_excel(xls, sheet_name="temps_reel_operateur")
    df_pointages["heure_debut"] = pd.to_datetime(df_pointages["heure_debut"])
    df_pointages["heure_fin"] = pd.to_datetime(df_pointages["heure_fin"])

    # La colonne "Durée h" fournie dans le fichier n'est pas fiable (valeurs
    # décalées / manquantes sur une grande partie des lignes). On recalcule
    # donc la durée en heures à partir de la colonne "duree" (en millisecondes),
    # qui elle correspond bien à l'écart heure_fin - heure_debut.
    df_pointages["Durée h"] = df_pointages["duree"] / MS_PER_HOUR

    iso = df_pointages["heure_debut"].dt.isocalendar()
    df_pointages["iso_year"] = iso["year"]
    df_pointages["iso_week"] = iso["week"]

    # --- Feuille des ordres de fabrication --------------------------------------
    df_of = pd.read_excel(xls, sheet_name="ordres_fabrication")
    df_of["date_cloture"] = pd.to_datetime(df_of["date_cloture"])
    df_of["temps_devis_h"] = df_of["temps_devis"] / MS_PER_HOUR
    df_of["temps_operateurs_h"] = df_of["temps_operateurs"] / MS_PER_HOUR

    iso_of = df_of["date_cloture"].dt.isocalendar()
    df_of["iso_year"] = iso_of["year"]
    df_of["iso_week"] = iso_of["week"]

    return df_pointages, df_of


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def week_bounds(iso_year: int, iso_week: int) -> tuple[date, date]:
    """Retourne (lundi, dimanche) de la semaine ISO demandée."""
    monday = date.fromisocalendar(int(iso_year), int(iso_week), 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


def build_week_options(df_pointages: pd.DataFrame, df_of: pd.DataFrame) -> pd.DataFrame:
    """Construit la liste des semaines disponibles à partir des deux feuilles."""
    weeks_a = df_pointages.dropna(subset=["iso_year", "iso_week"])[["iso_year", "iso_week"]]
    weeks_b = df_of.dropna(subset=["iso_year", "iso_week"])[["iso_year", "iso_week"]]
    weeks = pd.concat([weeks_a, weeks_b], ignore_index=True).drop_duplicates()
    weeks = weeks.sort_values(["iso_year", "iso_week"], ascending=[False, False])

    labels = []
    for _, row in weeks.iterrows():
        monday, sunday = week_bounds(row["iso_year"], row["iso_week"])
        labels.append(
            f"Semaine {int(row['iso_week']):02d} — {int(row['iso_year'])} "
            f"({monday.strftime('%d/%m/%Y')} au {sunday.strftime('%d/%m/%Y')})"
        )
    weeks = weeks.copy()
    weeks["label"] = labels
    return weeks.reset_index(drop=True)


def pie_top_n(df: pd.DataFrame, group_col: str, value_col: str, n: int = 12):
    """Regroupe les catégories au-delà du top N dans 'Autres' pour un camembert lisible."""
    agg = df.groupby(group_col, dropna=False)[value_col].sum().sort_values(ascending=False)
    if len(agg) > n:
        top = agg.iloc[:n]
        other = pd.Series({"Autres": agg.iloc[n:].sum()})
        agg = pd.concat([top, other])
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Interface — chargement du fichier
# ---------------------------------------------------------------------------

st.title("📊 Suivi d'activité hebdomadaire")

with st.sidebar:
    st.header("📁 Données")
    uploaded_file = st.file_uploader(
        "Charger le fichier de données (.xlsx)",
        type=["xlsx"],
        help="Fichier contenant les feuilles 'temps_reel_operateur' et 'ordres_fabrication'.",
    )

if uploaded_file is None:
    st.info("👈 Chargez votre fichier Excel de données dans la barre latérale pour démarrer.")
    st.stop()

try:
    df_pointages, df_of = load_data(uploaded_file.getvalue())
except Exception as e:
    st.error(f"Impossible de lire le fichier : {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sélecteur de semaine
# ---------------------------------------------------------------------------

weeks_df = build_week_options(df_pointages, df_of)
if weeks_df.empty:
    st.warning("Aucune semaine exploitable n'a été trouvée dans le fichier.")
    st.stop()

with st.sidebar:
    st.header("🗓️ Semaine")
    selected_label = st.selectbox("Sélectionner une semaine", weeks_df["label"])

selected_row = weeks_df.loc[weeks_df["label"] == selected_label].iloc[0]
sel_year, sel_week = int(selected_row["iso_year"]), int(selected_row["iso_week"])
monday, sunday = week_bounds(sel_year, sel_week)

st.caption(f"Semaine sélectionnée : **du {monday.strftime('%d/%m/%Y')} au {sunday.strftime('%d/%m/%Y')}**")

# ---------------------------------------------------------------------------
# Filtrage des données de la semaine
# ---------------------------------------------------------------------------

mask_pointages = (df_pointages["iso_year"] == sel_year) & (df_pointages["iso_week"] == sel_week)
pointages_semaine = df_pointages.loc[mask_pointages].copy()

mask_of = (df_of["iso_year"] == sel_year) & (df_of["iso_week"] == sel_week)
of_semaine = df_of.loc[mask_of].copy()

# ---------------------------------------------------------------------------
# Indicateurs — activité des opérateurs
# ---------------------------------------------------------------------------

heures_attribuees = pointages_semaine["Durée h"].sum()
nb_operateurs = pointages_semaine["id_operateur"].nunique()
taux_attribution = (heures_attribuees / (35 * nb_operateurs)) if nb_operateurs > 0 else 0

st.subheader("👷 Activité des opérateurs")
c1, c2, c3 = st.columns(3)
c1.metric("Nombre d'heures attribuées", f"{heures_attribuees:.1f} h")
c2.metric("Opérateurs ayant pointé", f"{nb_operateurs}")
c3.metric("Taux d'attribution", f"{taux_attribution:.0%}", help="Heures attribuées / (35 × nb opérateurs)")

col_pie1, col_pie2 = st.columns(2)

with col_pie1:
    st.markdown("**Répartition des heures par dossier**")
    if not pointages_semaine.empty:
        data1 = pie_top_n(pointages_semaine, "ordre_fabrication", "Durée h")
        fig1 = px.pie(data1, names="ordre_fabrication", values="Durée h", hole=0.35)
        fig1.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig1, use_container_width=True, key="pie_heures_par_dossier")
    else:
        st.info("Aucun pointage sur cette semaine.")

with col_pie2:
    st.markdown("**Répartition de la durée par ordre de fabrication**")
    if not pointages_semaine.empty:
        data2 = pie_top_n(pointages_semaine, "ordre_fabrication", "Durée h")
        fig2 = px.pie(data2, names="ordre_fabrication", values="Durée h", hole=0.35)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True, key="pie_duree_par_of")
    else:
        st.info("Aucun pointage sur cette semaine.")

st.caption(
    "ℹ️ Dans le fichier source, le champ « ordre de fabrication » des pointages correspond "
    "au numéro de dossier : les deux camemberts ci-dessus reflètent donc la même répartition."
)

st.markdown("**Détail des pointages de la semaine**")
if not pointages_semaine.empty:
    table_pointages = pointages_semaine[
        ["id", "created_at", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
    ].sort_values("heure_debut", ascending=False).reset_index(drop=True)
    table_pointages["Durée h"] = table_pointages["Durée h"].round(2)
    st.dataframe(table_pointages, use_container_width=True, hide_index=True)
else:
    st.info("Aucun pointage sur cette semaine.")

# ---------------------------------------------------------------------------
# Indicateurs — dossiers clôturés dans la semaine
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📦 Dossiers clôturés cette semaine")

heures_livrees = of_semaine["temps_operateurs_h"].sum()
heures_theoriques = of_semaine["temps_devis_h"].sum()
ratio_temps = (heures_livrees / heures_theoriques) if heures_theoriques > 0 else 0

d1, d2, d3 = st.columns(3)
d1.metric("Heures livrées cette semaine", f"{heures_livrees:.1f} h")
d2.metric("Heures théoriques livrées cette semaine", f"{heures_theoriques:.1f} h")
d3.metric("Ratio temps (livré / théorique)", f"{ratio_temps:.0%}")

st.markdown("**Ordres de fabrication clôturés dans la semaine**")
if not of_semaine.empty:
    table_of = of_semaine[
        [
            "id", "created_at", "numero_devis", "numero_dossier", "client",
            "reference", "operation", "temps_devis_h", "temps_operateurs_h",
        ]
    ].rename(
        columns={
            "temps_devis_h": "temps_devis (h)",
            "temps_operateurs_h": "temps_operateurs (h)",
        }
    ).sort_values("created_at", ascending=False).reset_index(drop=True)
    table_of["temps_devis (h)"] = table_of["temps_devis (h)"].round(2)
    table_of["temps_operateurs (h)"] = table_of["temps_operateurs (h)"].round(2)
    st.dataframe(table_of, use_container_width=True, hide_index=True)
else:
    st.info("Aucun dossier clôturé sur cette semaine.")
