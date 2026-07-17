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

    # --- Feuille des employés (pour afficher le nom des opérateurs) -------------
    try:
        df_emp = pd.read_excel(xls, sheet_name="employes")
        noms = (df_emp["prenom"].fillna("") + " " + df_emp["nom"].fillna("")).str.strip()
        operateur_map = dict(zip(df_emp["id"].astype(str), noms))
    except Exception:
        operateur_map = {}

    df_pointages["operateur"] = (
        df_pointages["id_operateur"].astype(str).map(operateur_map).fillna(df_pointages["id_operateur"].astype(str))
    )

    # --- Jointure dossier -> client (via numero_dossier de ordres_fabrication) --
    client_map = (
        df_of.dropna(subset=["numero_dossier"])
        .drop_duplicates(subset="numero_dossier")
        .set_index("numero_dossier")["client"]
    )
    df_pointages["client"] = df_pointages["ordre_fabrication"].map(client_map)
    df_pointages["dossier_client"] = (
        df_pointages["ordre_fabrication"].astype(str)
        + " – "
        + df_pointages["client"].fillna("Client inconnu")
    )

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
    agg.index.name = group_col
    agg.name = value_col
    if len(agg) > n:
        top = agg.iloc[:n]
        other = pd.Series({"Autres": agg.iloc[n:].sum()}, name=value_col)
        other.index.name = group_col
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
temps_theorique = nb_operateurs * 35
taux_attribution = (heures_attribuees / temps_theorique) if temps_theorique > 0 else 0

st.subheader("👷 Activité des opérateurs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Nombre d'heures attribuées", f"{heures_attribuees:.1f} h")
c2.metric("Opérateurs ayant pointé", f"{nb_operateurs}")
c3.metric("Temps travaillé théorique", f"{temps_theorique:.0f} h", help="Nombre d'opérateurs ayant pointé × 35 h")
c4.metric("Taux d'attribution", f"{taux_attribution:.0%}", help="Heures attribuées / (35 × nb opérateurs)")

col_pie1, col_pie2 = st.columns(2)

with col_pie1:
    st.markdown("**Répartition des heures par dossier**")
    if not pointages_semaine.empty:
        data1 = pie_top_n(pointages_semaine, "dossier_client", "Durée h")
        fig1 = px.pie(data1, names="dossier_client", values="Durée h", hole=0.35)
        fig1.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig1, use_container_width=True, key="pie_heures_par_dossier")
    else:
        st.info("Aucun pointage sur cette semaine.")

with col_pie2:
    st.markdown("**Répartition du nombre d'heures par opérateur**")
    if not pointages_semaine.empty:
        data2 = pie_top_n(pointages_semaine, "id_operateur", "Durée h")
        fig2 = px.pie(data2, names="id_operateur", values="Durée h", hole=0.35)
        fig2.update_traces(
            textposition="inside",
            textinfo="label+percent",
            texttemplate="%{label}<br>%{value:.1f} h (%{percent})",
        )
        st.plotly_chart(fig2, use_container_width=True, key="pie_heures_par_operateur")
    else:
        st.info("Aucun pointage sur cette semaine.")

st.caption(
    "ℹ️ Dans le fichier source, le champ « ordre de fabrication » des pointages correspond "
    "au numéro de dossier."
)

st.markdown("**Détail des pointages de la semaine**")
if not pointages_semaine.empty:
    table_pointages = pointages_semaine[
        ["id", "created_at", "id_operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
    ].sort_values("heure_debut", ascending=False).reset_index(drop=True)
    table_pointages["Durée h"] = table_pointages["Durée h"].round(2)
    st.dataframe(table_pointages, use_container_width=True, hide_index=True)
else:
    st.info("Aucun pointage sur cette semaine.")

# ---------------------------------------------------------------------------
# Indicateurs — dossiers clôturés dans la semaine
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🎯 Mise en forme")
    seuil_ecart = st.slider(
        "Seuil de tolérance de l'écart (%)",
        min_value=5, max_value=100, value=20, step=5,
        help="Écart = (temps_devis − temps_opérateurs) / temps_devis. "
             "Au-delà de ce seuil (en valeur absolue), la ligne est colorée.",
    )

st.divider()
st.subheader("📦 Dossiers clôturés cette semaine")

heures_livrees = of_semaine["temps_operateurs_h"].sum()
heures_theoriques = of_semaine["temps_devis_h"].sum()
ratio_temps = (heures_livrees / heures_theoriques) if heures_theoriques > 0 else 0
nb_dossiers_clotures = of_semaine["numero_dossier"].nunique()

d1, d2, d3, d4 = st.columns(4)
d1.metric("Nombre de dossiers clôturés", f"{nb_dossiers_clotures}")
d2.metric("Heures livrées cette semaine", f"{heures_livrees:.1f} h")
d3.metric("Heures théoriques livrées cette semaine", f"{heures_theoriques:.1f} h")
d4.metric("Ratio temps (livré / théorique)", f"{ratio_temps:.0%}")

st.markdown("**Temps par poste**")
if not of_semaine.empty:
    par_poste = (
        of_semaine.groupby("poste")[["temps_operateurs_h", "temps_devis_h"]]
        .sum()
        .rename(columns={"temps_operateurs_h": "Temps de production", "temps_devis_h": "Temps devis"})
        .sort_values("Temps devis", ascending=True)
        .reset_index()
    )
    fig_poste = px.bar(
        par_poste,
        y="poste",
        x=["Temps de production", "Temps devis"],
        orientation="h",
        barmode="group",
        labels={"value": "Heures", "poste": "", "variable": ""},
    )
    fig_poste.update_layout(legend_title_text="")
    st.plotly_chart(fig_poste, use_container_width=True, key="bar_temps_par_poste")
else:
    st.info("Aucun dossier clôturé sur cette semaine.")

st.markdown("**Temps par dossier clôturé**")
if not of_semaine.empty:
    of_semaine_label = of_semaine.copy()
    of_semaine_label["dossier_client"] = (
        of_semaine_label["numero_dossier"].astype(str) + " – " + of_semaine_label["client"].fillna("Client inconnu")
    )
    par_dossier = (
        of_semaine_label.groupby("dossier_client")[["temps_operateurs_h", "temps_devis_h"]]
        .sum()
        .rename(columns={"temps_operateurs_h": "Temps de production", "temps_devis_h": "Temps devis"})
        .sort_values("Temps devis", ascending=True)
        .reset_index()
    )
    fig_dossier = px.bar(
        par_dossier,
        y="dossier_client",
        x=["Temps de production", "Temps devis"],
        orientation="h",
        barmode="group",
        labels={"value": "Heures", "dossier_client": "", "variable": ""},
    )
    fig_dossier.update_layout(legend_title_text="", height=max(300, 40 * len(par_dossier)))
    st.plotly_chart(fig_dossier, use_container_width=True, key="bar_temps_par_dossier")
else:
    st.info("Aucun dossier clôturé sur cette semaine.")

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

    # Delta = temps_devis - temps_operateurs : positif si le dossier a été réalisé
    # plus vite que prévu, négatif s'il a pris plus de temps que le devis.
    table_of["delta (h)"] = (table_of["temps_devis (h)"] - table_of["temps_operateurs (h)"]).round(2)

    # Ratio = (temps_devis - temps_operateurs) / temps_devis, en % : positif si le
    # dossier a été réalisé plus vite que prévu, négatif s'il a pris plus de temps.
    table_of["ratio (devis-opérateurs)/devis (%)"] = (
        table_of["delta (h)"] / table_of["temps_devis (h)"].replace(0, float("nan")) * 100
    ).round(1)

    # Écart relatif utilisé pour la mise en forme conditionnelle (identique au ratio ci-dessus).
    ecart_pct = table_of["ratio (devis-opérateurs)/devis (%)"]

    def highlight_row(row):
        pct = ecart_pct.loc[row.name]
        if pd.isna(pct):
            return [""] * len(row)
        if pct < -seuil_ecart:
            # écart trop négatif : le dossier a pris beaucoup plus de temps que prévu
            color = "background-color: #f8d7da"
        elif pct > seuil_ecart:
            # écart trop positif : le dossier a été réalisé beaucoup plus vite que prévu
            color = "background-color: #ffe5b4"
        else:
            # écart nul ou faible
            color = "background-color: #d4edda"
        return [color] * len(row)

    styled_table_of = (
        table_of.style.apply(highlight_row, axis=1)
        .format(
            {
                "temps_devis (h)": "{:.2f}",
                "temps_operateurs (h)": "{:.2f}",
                "delta (h)": "{:.2f}",
                "ratio (devis-opérateurs)/devis (%)": "{:.0f}%",
            },
            na_rep="–",
        )
    )
    st.dataframe(styled_table_of, use_container_width=True, hide_index=True)
    st.caption(
        "🟥 Écart trop négatif (dossier plus long que prévu) · "
        "🟧 Écart trop positif (dossier plus rapide que prévu) · "
        "🟩 Écart nul ou faible, dans la tolérance."
    )
else:
    st.info("Aucun dossier clôturé sur cette semaine.")
