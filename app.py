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

import numpy as np
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
# Palette de couleurs commune à tous les graphiques "devisé vs réalisé"
# ---------------------------------------------------------------------------
COULEUR_REALISE = "#0068C9"  # bleu foncé
COULEUR_DEVISE = "#83C9FF"   # bleu clair
COLOR_MAP_DEVIS_REALISE = {
    "Temps réalisé": COULEUR_REALISE,
    "Temps devisé": COULEUR_DEVISE,
}

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
    df_pointages["cal_year"] = df_pointages["heure_debut"].dt.year
    df_pointages["cal_month"] = df_pointages["heure_debut"].dt.month

    # --- Feuille des ordres de fabrication --------------------------------------
    df_of = pd.read_excel(xls, sheet_name="ordres_fabrication")
    df_of["date_cloture"] = pd.to_datetime(df_of["date_cloture"])
    df_of["temps_devis_h"] = df_of["temps_devis"] / MS_PER_HOUR
    df_of["temps_operateurs_h"] = df_of["temps_operateurs"] / MS_PER_HOUR

    iso_of = df_of["date_cloture"].dt.isocalendar()
    df_of["iso_year"] = iso_of["year"]
    df_of["iso_week"] = iso_of["week"]
    df_of["cal_year"] = df_of["date_cloture"].dt.year
    df_of["cal_month"] = df_of["date_cloture"].dt.month

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


NOMS_MOIS = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]


def month_bounds(year: int, month: int) -> tuple[date, date]:
    """Retourne (1er jour, dernier jour) du mois calendaire demandé."""
    first_day = date(int(year), int(month), 1)
    if month == 12:
        next_month_first_day = date(int(year) + 1, 1, 1)
    else:
        next_month_first_day = date(int(year), int(month) + 1, 1)
    last_day = next_month_first_day - timedelta(days=1)
    return first_day, last_day


def nb_jours_ouvres(premier_jour: date, dernier_jour: date) -> int:
    """Retourne le nombre de jours ouvrés (lundi-vendredi) entre deux dates incluses.

    Basé sur np.busday_count, qui exclut uniquement les samedis/dimanches
    (les jours fériés ne sont pas pris en compte, faute de calendrier fourni).
    """
    # np.busday_count exclut la borne de fin -> on l'étend d'un jour
    return int(np.busday_count(premier_jour, dernier_jour + timedelta(days=1)))


def build_month_options(df_pointages: pd.DataFrame, df_of: pd.DataFrame) -> pd.DataFrame:
    """Construit la liste des mois disponibles à partir des deux feuilles."""
    months_a = df_pointages.dropna(subset=["cal_year", "cal_month"])[["cal_year", "cal_month"]]
    months_b = df_of.dropna(subset=["cal_year", "cal_month"])[["cal_year", "cal_month"]]
    months = pd.concat([months_a, months_b], ignore_index=True).drop_duplicates()
    months = months.sort_values(["cal_year", "cal_month"], ascending=[False, False])

    labels = []
    for _, row in months.iterrows():
        labels.append(
            f"{NOMS_MOIS[int(row['cal_month']) - 1]} {int(row['cal_year'])}"
        )
    months = months.copy()
    months["label"] = labels
    return months.reset_index(drop=True)


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
    if st.button("🔄 Vider le cache et recharger", help="À utiliser si vous venez de mettre à jour le fichier source et que les chiffres semblent obsolètes."):
        st.cache_data.clear()
        st.rerun()

if uploaded_file is None:
    st.info("👈 Chargez votre fichier Excel de données dans la barre latérale pour démarrer.")
    st.stop()

try:
    df_pointages, df_of = load_data(uploaded_file.getvalue())
except Exception as e:
    st.error(f"Impossible de lire le fichier : {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Onglets — Suivi hebdomadaire / Bilan annuel
# ---------------------------------------------------------------------------

tab_hebdo, tab_mensuel, tab_annuel = st.tabs(
    ["📊 Suivi hebdomadaire", "📆 Suivi mensuel", "📅 Bilan annuel"]
)

with tab_hebdo:

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
    st.caption(
        f"🕒 Dernière date de clôture présente dans le fichier chargé : "
        f"**{df_of['date_cloture'].max().strftime('%d/%m/%Y') if df_of['date_cloture'].notna().any() else 'aucune'}** "
        f"— si cette date vous semble ancienne, cliquez sur « Vider le cache et recharger »."
    )

    # ---------------------------------------------------------------------------
    # Filtrage des données de la semaine
    # ---------------------------------------------------------------------------

    mask_pointages = (df_pointages["iso_year"] == sel_year) & (df_pointages["iso_week"] == sel_week)
    pointages_semaine = df_pointages.loc[mask_pointages].copy()

    mask_of = (
        (df_of["iso_year"] == sel_year)
        & (df_of["iso_week"] == sel_week)
        & (df_of["statut_production"] == "Clos")  # ne garder que les dossiers réellement clôturés
    )
    of_semaine = df_of.loc[mask_of].copy()

    # ---------------------------------------------------------------------------
    # Indicateurs — activité des opérateurs
    # ---------------------------------------------------------------------------

    heures_attribuees = pointages_semaine["Durée h"].sum()
    nb_operateurs = pointages_semaine["id_operateur"].nunique()
    temps_theorique = nb_operateurs * 39
    taux_attribution = (heures_attribuees / temps_theorique) if temps_theorique > 0 else 0

    st.subheader("👷 Activité des opérateurs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre d'heures attribuées", f"{heures_attribuees:.1f} h")
    c2.metric("Opérateurs ayant pointé", f"{nb_operateurs}")
    c3.metric("Temps travaillé théorique", f"{temps_theorique:.0f} h", help="Nombre d'opérateurs ayant pointé × 39 h")
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
            help="Écart = (temps_opérateurs − temps_devis) / temps_devis. "
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
            .rename(columns={"temps_operateurs_h": "Temps réalisé", "temps_devis_h": "Temps devisé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        fig_poste = px.bar(
            par_poste,
            y="poste",
            x=["Temps réalisé", "Temps devisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "poste": "", "variable": ""},
            color_discrete_map=COLOR_MAP_DEVIS_REALISE,
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
            .rename(columns={"temps_operateurs_h": "Temps réalisé", "temps_devis_h": "Temps devisé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        fig_dossier = px.bar(
            par_dossier,
            y="dossier_client",
            x=["Temps réalisé", "Temps devisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "dossier_client": "", "variable": ""},
            color_discrete_map=COLOR_MAP_DEVIS_REALISE,
        )
        fig_dossier.update_layout(legend_title_text="", height=max(300, 40 * len(par_dossier)))
        st.plotly_chart(fig_dossier, use_container_width=True, key="bar_temps_par_dossier")
    else:
        st.info("Aucun dossier clôturé sur cette semaine.")

    st.markdown("**Ordres de fabrication clôturés dans la semaine**")
    if not of_semaine.empty:
        # -----------------------------------------------------------------------
        # Ajout du filtre sur le numéro de dossier
        # -----------------------------------------------------------------------
        dossiers_disponibles = sorted(
            of_semaine["numero_dossier"].dropna().astype(str).unique().tolist()
        )

        f_col1, f_col2 = st.columns([2, 1])
        with f_col1:
            selected_dossiers = st.multiselect(
                "🔎 Filtrer par numéro de dossier",
                options=dossiers_disponibles,
                default=[],
                placeholder="Sélectionnez un ou plusieurs dossiers (laisser vide pour tout voir)",
            )

        # Filtrage du dataframe selon la sélection
        if selected_dossiers:
            of_semaine_filtered = of_semaine[
                of_semaine["numero_dossier"].astype(str).isin(selected_dossiers)
            ].copy()
        else:
            of_semaine_filtered = of_semaine.copy()

        # -----------------------------------------------------------------------
        # Construction du tableau avec les données filtrées
        # -----------------------------------------------------------------------
        if not of_semaine_filtered.empty:
            table_of = of_semaine_filtered[
                [
                    "id",
                    "created_at",
                    "numero_devis",
                    "numero_dossier",
                    "client",
                    "reference",
                    "operation",
                    "temps_devis_h",
                    "temps_operateurs_h",
                ]
            ].rename(
                columns={
                    "temps_devis_h": "temps_devis (h)",
                    "temps_operateurs_h": "temps_operateurs (h)",
                }
            ).sort_values("created_at", ascending=False).reset_index(drop=True)

            table_of["temps_devis (h)"] = table_of["temps_devis (h)"].round(2)
            table_of["temps_operateurs (h)"] = table_of["temps_operateurs (h)"].round(2)

            # Delta = temps_operateurs - temps_devis (positif si heures en trop)
            table_of["delta (h)"] = (
                table_of["temps_operateurs (h)"] - table_of["temps_devis (h)"]
            ).round(2)

            # Ratio = (temps_operateurs - temps_devis) / temps_devis
            table_of["ratio (opérateurs-devis)/devis (%)"] = (
                table_of["delta (h)"]
                / table_of["temps_devis (h)"].replace(0, float("nan"))
                * 100
            ).round(1)

            ecart_pct = table_of["ratio (opérateurs-devis)/devis (%)"]

            def highlight_row(row):
                pct = ecart_pct.loc[row.name]
            
                # Opération non pointée (Gris)
                if row["temps_operateurs (h)"] == 0:
                    return ["background-color: #e2e3e5"] * len(row)
                
                if pd.isna(pct):
                    return [""] * len(row)
                if pct > seuil_ecart:
                    color = "background-color: #f8d7da"  # Rouge : plus long que prévu (dépassement)
                elif pct < -seuil_ecart:
                    color = "background-color: #ffe5b4"  # Orange : plus rapide que prévu (économie)
                else:
                    color = "background-color: #d4edda"  # Vert : dans la tolérance
                return [color] * len(row)

            styled_table_of = (
                table_of.style.apply(highlight_row, axis=1)
                .format(
                    {
                        "temps_devis (h)": "{:.2f}",
                        "temps_operateurs (h)": "{:.2f}",
                        "delta (h)": "{:.2f}",
                        "ratio (opérateurs-devis)/devis (%)": "{:.0f}%",
                    },
                    na_rep="–",
                )
            )
            st.dataframe(styled_table_of, use_container_width=True, hide_index=True)
            st.caption(
                "⬜ Opération non pointée (0h) · "
                "🟥 Écart trop positif (dépassement d'heures) · "
                "🟧 Écart trop négatif (dossier plus rapide que prévu) · "
                "🟩 Écart nul ou faible, dans la tolérance."
            )
        else:
            st.info("Aucun dossier ne correspond aux critères de recherche sélectionnés.")
    else:
        st.info("Aucun dossier clôturé sur cette semaine.")


    # ---------------------------------------------------------------------------
    # Recherche et Analyse d'un dossier spécifique (Global)
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("🔍 Analyse d'un dossier spécifique")
    st.markdown("Cette section permet de consulter les détails d'un dossier indépendamment de la semaine sélectionnée (dossiers en cours, anciens, etc.).")

    # 1. Récupérer la liste de tous les dossiers (sans le filtre de la semaine)
    tous_les_dossiers = sorted(df_of["numero_dossier"].dropna().astype(str).unique().tolist())

    # 2. Sélecteur de dossier
    dossier_choisi = st.selectbox(
        "Sélectionnez ou tapez le numéro d'un dossier :",
        options=[""] + tous_les_dossiers,
        format_func=lambda x: "Sélectionnez un dossier..." if x == "" else x,
        help="Vous pouvez taper directement le numéro pour le trouver plus vite."
    )

    if dossier_choisi != "":
        # 3. Filtrer les données globales pour ce dossier spécifique
        spec_of = df_of[df_of["numero_dossier"].astype(str) == dossier_choisi].copy()

        if not spec_of.empty:
            # En-tête du dossier
            client_nom = spec_of["client"].iloc[0]
            st.subheader(f"Dossier : {dossier_choisi} — {client_nom if pd.notna(client_nom) else 'Client inconnu'}")

            # 4. Calcul des indicateurs globaux du dossier
            devis_total = spec_of["temps_devis_h"].sum()
            realise_total = spec_of["temps_operateurs_h"].sum()
            ecart_total = realise_total - devis_total

            c_spec1, c_spec2, c_spec3 = st.columns(3)
            c_spec1.metric("Temps devisé (Total)", f"{devis_total:.1f} h")
            c_spec2.metric("Temps réalisé (Total)", f"{realise_total:.1f} h")
        
            # Coloration de l'écart : Rouge si on dépasse le devis (>0), Vert si en dessous (<=0)
            delta_color = "inverse" if ecart_total > 0 else "normal"
            c_spec3.metric("Écart (Réalisé - Devis)", f"{ecart_total:.1f} h", delta_color=delta_color)

            # 5. Graphique : Temps par poste
            st.markdown("**Temps par poste**")
            par_poste_spec = (
                spec_of.groupby("poste", dropna=False)[["temps_operateurs_h", "temps_devis_h"]]
                .sum()
                .rename(columns={"temps_operateurs_h": "Temps réalisé", "temps_devis_h": "Temps devisé"})
                .sort_values("Temps devisé", ascending=True)
                .reset_index()
            )
            par_poste_spec["poste"] = par_poste_spec["poste"].fillna("Non défini")

            fig_poste_spec = px.bar(
                par_poste_spec,
                y="poste",
                x=["Temps réalisé", "Temps devisé"],
                orientation="h",
                barmode="group",
                labels={"value": "Heures", "poste": "Poste", "variable": ""},
                color_discrete_map=COLOR_MAP_DEVIS_REALISE,
            )
            fig_poste_spec.update_layout(legend_title_text="")
            st.plotly_chart(fig_poste_spec, use_container_width=True, key=f"bar_spec_{dossier_choisi}")

            # 6. Tableau détaillé par opérations du dossier
            st.markdown("**Détail des opérations (Ordres de Fabrication) de ce dossier**")
        
            table_spec_of = spec_of[
                [
                    "id",
                    "created_at",
                    "numero_devis",
                    "client",
                    "reference",
                    "operation",
                    "temps_devis_h",
                    "temps_operateurs_h",
                ]
            ].rename(
                columns={
                    "temps_devis_h": "temps_devis (h)",
                    "temps_operateurs_h": "temps_operateurs (h)",
                }
            ).sort_values("created_at", ascending=False).reset_index(drop=True)

            table_spec_of["temps_devis (h)"] = table_spec_of["temps_devis (h)"].round(2)
            table_spec_of["temps_operateurs (h)"] = table_spec_of["temps_operateurs (h)"].round(2)

            # Calculs Delta et Ratio (Inversés pour refléter le réalisé par rapport au devis)
            table_spec_of["delta (h)"] = (
                table_spec_of["temps_operateurs (h)"] - table_spec_of["temps_devis (h)"]
            ).round(2)

            table_spec_of["ratio (opérateurs-devis)/devis (%)"] = (
                table_spec_of["delta (h)"]
                / table_spec_of["temps_devis (h)"].replace(0, float("nan"))
                * 100
            ).round(1)

            ecart_pct_spec = table_spec_of["ratio (opérateurs-devis)/devis (%)"]

            def highlight_row_spec(row):
                pct = ecart_pct_spec.loc[row.name]
            
                # Opération non pointée (Gris)
                if row["temps_operateurs (h)"] == 0:
                    return ["background-color: #e2e3e5"] * len(row)
                
                if pd.isna(pct):
                    return [""] * len(row)
                if pct > seuil_ecart:
                    color = "background-color: #f8d7da"  # Rouge : dépassement
                elif pct < -seuil_ecart:
                    color = "background-color: #ffe5b4"  # Orange : sous le devis
                else:
                    color = "background-color: #d4edda"  # Vert : dans la tolérance
                return [color] * len(row)

            styled_table_spec_of = (
                table_spec_of.style.apply(highlight_row_spec, axis=1)
                .format(
                    {
                        "temps_devis (h)": "{:.2f}",
                        "temps_operateurs (h)": "{:.2f}",
                        "delta (h)": "{:.2f}",
                        "ratio (opérateurs-devis)/devis (%)": "{:.0f}%",
                    },
                    na_rep="–",
                )
            )
            st.dataframe(styled_table_spec_of, use_container_width=True, hide_index=True)
            st.caption(
                "⬜ Opération non pointée (0h) · "
                "🟥 Écart trop positif (dépassement d'heures) · "
                "🟧 Écart trop négatif (dossier plus rapide que prévu) · "
                "🟩 Écart nul ou faible, dans la tolérance."
            )

        else:
            st.warning("Ce dossier est introuvable dans la liste globale des ordres de fabrication.")

# ---------------------------------------------------------------------------
# Onglet — Suivi mensuel
# ---------------------------------------------------------------------------

with tab_mensuel:

    # ---------------------------------------------------------------------------
    # Sélecteur de mois
    # ---------------------------------------------------------------------------

    months_df = build_month_options(df_pointages, df_of)
    if months_df.empty:
        st.warning("Aucun mois exploitable n'a été trouvé dans le fichier.")
        st.stop()

    with st.sidebar:
        st.header("🗓️ Mois")
        selected_label_mois = st.selectbox(
            "Sélectionner un mois", months_df["label"], key="select_mois"
        )

    selected_row_mois = months_df.loc[months_df["label"] == selected_label_mois].iloc[0]
    sel_year_m, sel_month = int(selected_row_mois["cal_year"]), int(selected_row_mois["cal_month"])
    premier_jour, dernier_jour = month_bounds(sel_year_m, sel_month)

    st.caption(f"Mois sélectionné : **du {premier_jour.strftime('%d/%m/%Y')} au {dernier_jour.strftime('%d/%m/%Y')}**")
    st.caption(
        f"🕒 Dernière date de clôture présente dans le fichier chargé : "
        f"**{df_of['date_cloture'].max().strftime('%d/%m/%Y') if df_of['date_cloture'].notna().any() else 'aucune'}** "
        f"— si cette date vous semble ancienne, cliquez sur « Vider le cache et recharger »."
    )

    # ---------------------------------------------------------------------------
    # Filtrage des données du mois
    # ---------------------------------------------------------------------------

    mask_pointages_m = (df_pointages["cal_year"] == sel_year_m) & (df_pointages["cal_month"] == sel_month)
    pointages_mois = df_pointages.loc[mask_pointages_m].copy()

    mask_of_m = (
        (df_of["cal_year"] == sel_year_m)
        & (df_of["cal_month"] == sel_month)
        & (df_of["statut_production"] == "Clos")  # ne garder que les dossiers réellement clôturés
    )
    of_mois = df_of.loc[mask_of_m].copy()

    # ---------------------------------------------------------------------------
    # Indicateurs — activité des opérateurs
    # ---------------------------------------------------------------------------

    heures_attribuees_m = pointages_mois["Durée h"].sum()
    nb_operateurs_m = pointages_mois["id_operateur"].nunique()
    jours_ouvres_m = nb_jours_ouvres(premier_jour, dernier_jour)
    temps_theorique_m = jours_ouvres_m * nb_operateurs_m * 7.8
    taux_attribution_m = (heures_attribuees_m / temps_theorique_m) if temps_theorique_m > 0 else 0

    st.subheader("👷 Activité des opérateurs")
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Nombre d'heures attribuées", f"{heures_attribuees_m:.1f} h")
    cm2.metric("Opérateurs ayant pointé", f"{nb_operateurs_m}")
    cm3.metric(
        "Temps travaillé théorique",
        f"{temps_theorique_m:.0f} h",
        help=f"{jours_ouvres_m} jours ouvrés × {nb_operateurs_m} opérateur(s) × 7,8 h",
    )
    cm4.metric(
        "Taux d'attribution",
        f"{taux_attribution_m:.0%}",
        help="Heures attribuées / (jours ouvrés × nb opérateurs × 7,8 h)",
    )

    col_pie1_m, col_pie2_m = st.columns(2)

    with col_pie1_m:
        st.markdown("**Répartition des heures par dossier**")
        if not pointages_mois.empty:
            data1_m = pie_top_n(pointages_mois, "dossier_client", "Durée h")
            fig1_m = px.pie(data1_m, names="dossier_client", values="Durée h", hole=0.35)
            fig1_m.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig1_m, use_container_width=True, key="pie_heures_par_dossier_mensuel")
        else:
            st.info("Aucun pointage sur ce mois.")

    with col_pie2_m:
        st.markdown("**Répartition du nombre d'heures par opérateur**")
        if not pointages_mois.empty:
            data2_m = pie_top_n(pointages_mois, "id_operateur", "Durée h")
            fig2_m = px.pie(data2_m, names="id_operateur", values="Durée h", hole=0.35)
            fig2_m.update_traces(
                textposition="inside",
                textinfo="label+percent",
                texttemplate="%{label}<br>%{value:.1f} h (%{percent})",
            )
            st.plotly_chart(fig2_m, use_container_width=True, key="pie_heures_par_operateur_mensuel")
        else:
            st.info("Aucun pointage sur ce mois.")

    st.caption(
        "ℹ️ Dans le fichier source, le champ « ordre de fabrication » des pointages correspond "
        "au numéro de dossier."
    )

    st.markdown("**Détail des pointages du mois**")
    if not pointages_mois.empty:
        table_pointages_m = pointages_mois[
            ["id", "created_at", "id_operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
        ].sort_values("heure_debut", ascending=False).reset_index(drop=True)
        table_pointages_m["Durée h"] = table_pointages_m["Durée h"].round(2)
        st.dataframe(table_pointages_m, use_container_width=True, hide_index=True, key="df_pointages_mensuel")
    else:
        st.info("Aucun pointage sur ce mois.")

    # ---------------------------------------------------------------------------
    # Indicateurs — dossiers clôturés dans le mois
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("📦 Dossiers clôturés ce mois")

    heures_livrees_m = of_mois["temps_operateurs_h"].sum()
    heures_theoriques_m = of_mois["temps_devis_h"].sum()
    ratio_temps_m = (heures_livrees_m / heures_theoriques_m) if heures_theoriques_m > 0 else 0
    nb_dossiers_clotures_m = of_mois["numero_dossier"].nunique()

    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Nombre de dossiers clôturés", f"{nb_dossiers_clotures_m}")
    dm2.metric("Heures livrées ce mois", f"{heures_livrees_m:.1f} h")
    dm3.metric("Heures théoriques livrées ce mois", f"{heures_theoriques_m:.1f} h")
    dm4.metric("Ratio temps (livré / théorique)", f"{ratio_temps_m:.0%}")

    st.markdown("**Temps par poste**")
    if not of_mois.empty:
        par_poste_m = (
            of_mois.groupby("poste")[["temps_operateurs_h", "temps_devis_h"]]
            .sum()
            .rename(columns={"temps_operateurs_h": "Temps réalisé", "temps_devis_h": "Temps devisé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        fig_poste_m = px.bar(
            par_poste_m,
            y="poste",
            x=["Temps réalisé", "Temps devisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "poste": "", "variable": ""},
            color_discrete_map=COLOR_MAP_DEVIS_REALISE,
        )
        fig_poste_m.update_layout(legend_title_text="")
        st.plotly_chart(fig_poste_m, use_container_width=True, key="bar_temps_par_poste_mensuel")
    else:
        st.info("Aucun dossier clôturé sur ce mois.")

    st.markdown("**Temps par dossier clôturé**")
    if not of_mois.empty:
        of_mois_label = of_mois.copy()
        of_mois_label["dossier_client"] = (
            of_mois_label["numero_dossier"].astype(str) + " – " + of_mois_label["client"].fillna("Client inconnu")
        )
        par_dossier_m = (
            of_mois_label.groupby("dossier_client")[["temps_operateurs_h", "temps_devis_h"]]
            .sum()
            .rename(columns={"temps_operateurs_h": "Temps réalisé", "temps_devis_h": "Temps devisé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        fig_dossier_m = px.bar(
            par_dossier_m,
            y="dossier_client",
            x=["Temps réalisé", "Temps devisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "dossier_client": "", "variable": ""},
            color_discrete_map=COLOR_MAP_DEVIS_REALISE,
        )
        fig_dossier_m.update_layout(legend_title_text="", height=max(300, 40 * len(par_dossier_m)))
        st.plotly_chart(fig_dossier_m, use_container_width=True, key="bar_temps_par_dossier_mensuel")
    else:
        st.info("Aucun dossier clôturé sur ce mois.")

    st.markdown("**Ordres de fabrication clôturés dans le mois**")
    if not of_mois.empty:
        # -----------------------------------------------------------------------
        # Ajout du filtre sur le numéro de dossier
        # -----------------------------------------------------------------------
        dossiers_disponibles_m = sorted(
            of_mois["numero_dossier"].dropna().astype(str).unique().tolist()
        )

        fm_col1, fm_col2 = st.columns([2, 1])
        with fm_col1:
            selected_dossiers_m = st.multiselect(
                "🔎 Filtrer par numéro de dossier",
                options=dossiers_disponibles_m,
                default=[],
                placeholder="Sélectionnez un ou plusieurs dossiers (laisser vide pour tout voir)",
                key="filtre_dossiers_mensuel",
            )

        # Filtrage du dataframe selon la sélection
        if selected_dossiers_m:
            of_mois_filtered = of_mois[
                of_mois["numero_dossier"].astype(str).isin(selected_dossiers_m)
            ].copy()
        else:
            of_mois_filtered = of_mois.copy()

        # -----------------------------------------------------------------------
        # Construction du tableau avec les données filtrées
        # -----------------------------------------------------------------------
        if not of_mois_filtered.empty:
            table_of_m = of_mois_filtered[
                [
                    "id",
                    "created_at",
                    "numero_devis",
                    "numero_dossier",
                    "client",
                    "reference",
                    "operation",
                    "temps_devis_h",
                    "temps_operateurs_h",
                ]
            ].rename(
                columns={
                    "temps_devis_h": "temps_devis (h)",
                    "temps_operateurs_h": "temps_operateurs (h)",
                }
            ).sort_values("created_at", ascending=False).reset_index(drop=True)

            table_of_m["temps_devis (h)"] = table_of_m["temps_devis (h)"].round(2)
            table_of_m["temps_operateurs (h)"] = table_of_m["temps_operateurs (h)"].round(2)

            # Delta = temps_operateurs - temps_devis (positif si heures en trop)
            table_of_m["delta (h)"] = (
                table_of_m["temps_operateurs (h)"] - table_of_m["temps_devis (h)"]
            ).round(2)

            # Ratio = (temps_operateurs - temps_devis) / temps_devis
            table_of_m["ratio (opérateurs-devis)/devis (%)"] = (
                table_of_m["delta (h)"]
                / table_of_m["temps_devis (h)"].replace(0, float("nan"))
                * 100
            ).round(1)

            ecart_pct_m = table_of_m["ratio (opérateurs-devis)/devis (%)"]

            def highlight_row_mensuel(row):
                pct = ecart_pct_m.loc[row.name]

                # Opération non pointée (Gris)
                if row["temps_operateurs (h)"] == 0:
                    return ["background-color: #e2e3e5"] * len(row)

                if pd.isna(pct):
                    return [""] * len(row)
                if pct > seuil_ecart:
                    color = "background-color: #f8d7da"  # Rouge : plus long que prévu (dépassement)
                elif pct < -seuil_ecart:
                    color = "background-color: #ffe5b4"  # Orange : plus rapide que prévu (économie)
                else:
                    color = "background-color: #d4edda"  # Vert : dans la tolérance
                return [color] * len(row)

            styled_table_of_m = (
                table_of_m.style.apply(highlight_row_mensuel, axis=1)
                .format(
                    {
                        "temps_devis (h)": "{:.2f}",
                        "temps_operateurs (h)": "{:.2f}",
                        "delta (h)": "{:.2f}",
                        "ratio (opérateurs-devis)/devis (%)": "{:.0f}%",
                    },
                    na_rep="–",
                )
            )
            st.dataframe(styled_table_of_m, use_container_width=True, hide_index=True, key="df_of_mensuel")
            st.caption(
                "⬜ Opération non pointée (0h) · "
                "🟥 Écart trop positif (dépassement d'heures) · "
                "🟧 Écart trop négatif (dossier plus rapide que prévu) · "
                "🟩 Écart nul ou faible, dans la tolérance."
            )
        else:
            st.info("Aucun dossier ne correspond aux critères de recherche sélectionnés.")
    else:
        st.info("Aucun dossier clôturé sur ce mois.")


    # ---------------------------------------------------------------------------
    # Recherche et Analyse d'un dossier spécifique (Global)
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("🔍 Analyse d'un dossier spécifique")
    st.markdown("Cette section permet de consulter les détails d'un dossier indépendamment du mois sélectionné (dossiers en cours, anciens, etc.).")

    # 1. Récupérer la liste de tous les dossiers (sans le filtre du mois)
    tous_les_dossiers_m = sorted(df_of["numero_dossier"].dropna().astype(str).unique().tolist())

    # 2. Sélecteur de dossier
    dossier_choisi_m = st.selectbox(
        "Sélectionnez ou tapez le numéro d'un dossier :",
        options=[""] + tous_les_dossiers_m,
        format_func=lambda x: "Sélectionnez un dossier..." if x == "" else x,
        help="Vous pouvez taper directement le numéro pour le trouver plus vite.",
        key="select_dossier_mensuel",
    )

    if dossier_choisi_m != "":
        # 3. Filtrer les données globales pour ce dossier spécifique
        spec_of_m = df_of[df_of["numero_dossier"].astype(str) == dossier_choisi_m].copy()

        if not spec_of_m.empty:
            # En-tête du dossier
            client_nom_m = spec_of_m["client"].iloc[0]
            st.subheader(f"Dossier : {dossier_choisi_m} — {client_nom_m if pd.notna(client_nom_m) else 'Client inconnu'}")

            # 4. Calcul des indicateurs globaux du dossier
            devis_total_m = spec_of_m["temps_devis_h"].sum()
            realise_total_m = spec_of_m["temps_operateurs_h"].sum()
            ecart_total_m = realise_total_m - devis_total_m

            c_spec1_m, c_spec2_m, c_spec3_m = st.columns(3)
            c_spec1_m.metric("Temps devisé (Total)", f"{devis_total_m:.1f} h")
            c_spec2_m.metric("Temps réalisé (Total)", f"{realise_total_m:.1f} h")

            # Coloration de l'écart : Rouge si on dépasse le devis (>0), Vert si en dessous (<=0)
            delta_color_m = "inverse" if ecart_total_m > 0 else "normal"
            c_spec3_m.metric("Écart (Réalisé - Devis)", f"{ecart_total_m:.1f} h", delta_color=delta_color_m)

            # 5. Graphique : Temps par poste
            st.markdown("**Temps par poste**")
            par_poste_spec_m = (
                spec_of_m.groupby("poste", dropna=False)[["temps_operateurs_h", "temps_devis_h"]]
                .sum()
                .rename(columns={"temps_operateurs_h": "Temps réalisé", "temps_devis_h": "Temps devisé"})
                .sort_values("Temps devisé", ascending=True)
                .reset_index()
            )
            par_poste_spec_m["poste"] = par_poste_spec_m["poste"].fillna("Non défini")

            fig_poste_spec_m = px.bar(
                par_poste_spec_m,
                y="poste",
                x=["Temps réalisé", "Temps devisé"],
                orientation="h",
                barmode="group",
                labels={"value": "Heures", "poste": "Poste", "variable": ""},
                color_discrete_map=COLOR_MAP_DEVIS_REALISE,
            )
            fig_poste_spec_m.update_layout(legend_title_text="")
            st.plotly_chart(fig_poste_spec_m, use_container_width=True, key=f"bar_spec_mensuel_{dossier_choisi_m}")

            # 6. Tableau détaillé par opérations du dossier
            st.markdown("**Détail des opérations (Ordres de Fabrication) de ce dossier**")

            table_spec_of_m = spec_of_m[
                [
                    "id",
                    "created_at",
                    "numero_devis",
                    "client",
                    "reference",
                    "operation",
                    "temps_devis_h",
                    "temps_operateurs_h",
                ]
            ].rename(
                columns={
                    "temps_devis_h": "temps_devis (h)",
                    "temps_operateurs_h": "temps_operateurs (h)",
                }
            ).sort_values("created_at", ascending=False).reset_index(drop=True)

            table_spec_of_m["temps_devis (h)"] = table_spec_of_m["temps_devis (h)"].round(2)
            table_spec_of_m["temps_operateurs (h)"] = table_spec_of_m["temps_operateurs (h)"].round(2)

            # Calculs Delta et Ratio (Inversés pour refléter le réalisé par rapport au devis)
            table_spec_of_m["delta (h)"] = (
                table_spec_of_m["temps_operateurs (h)"] - table_spec_of_m["temps_devis (h)"]
            ).round(2)

            table_spec_of_m["ratio (opérateurs-devis)/devis (%)"] = (
                table_spec_of_m["delta (h)"]
                / table_spec_of_m["temps_devis (h)"].replace(0, float("nan"))
                * 100
            ).round(1)

            ecart_pct_spec_m = table_spec_of_m["ratio (opérateurs-devis)/devis (%)"]

            def highlight_row_spec_mensuel(row):
                pct = ecart_pct_spec_m.loc[row.name]

                # Opération non pointée (Gris)
                if row["temps_operateurs (h)"] == 0:
                    return ["background-color: #e2e3e5"] * len(row)

                if pd.isna(pct):
                    return [""] * len(row)
                if pct > seuil_ecart:
                    color = "background-color: #f8d7da"  # Rouge : dépassement
                elif pct < -seuil_ecart:
                    color = "background-color: #ffe5b4"  # Orange : sous le devis
                else:
                    color = "background-color: #d4edda"  # Vert : dans la tolérance
                return [color] * len(row)

            styled_table_spec_of_m = (
                table_spec_of_m.style.apply(highlight_row_spec_mensuel, axis=1)
                .format(
                    {
                        "temps_devis (h)": "{:.2f}",
                        "temps_operateurs (h)": "{:.2f}",
                        "delta (h)": "{:.2f}",
                        "ratio (opérateurs-devis)/devis (%)": "{:.0f}%",
                    },
                    na_rep="–",
                )
            )
            st.dataframe(styled_table_spec_of_m, use_container_width=True, hide_index=True, key="df_spec_mensuel")
            st.caption(
                "⬜ Opération non pointée (0h) · "
                "🟥 Écart trop positif (dépassement d'heures) · "
                "🟧 Écart trop négatif (dossier plus rapide que prévu) · "
                "🟩 Écart nul ou faible, dans la tolérance."
            )

        else:
            st.warning("Ce dossier est introuvable dans la liste globale des ordres de fabrication.")

# ---------------------------------------------------------------------------
# Onglet — Bilan annuel
# ---------------------------------------------------------------------------

with tab_annuel:
    st.header("📅 Bilan annuel")

    # -- Sélecteur d'année -------------------------------------------------
    annees_pointages = df_pointages["iso_year"].dropna().astype(int).unique().tolist()
    annees_of = df_of["date_cloture"].dropna().dt.year.astype(int).unique().tolist()
    annees_disponibles = sorted(set(annees_pointages) | set(annees_of), reverse=True)

    if not annees_disponibles:
        st.warning("Aucune année exploitable n'a été trouvée dans le fichier.")
        st.stop()

    annee_defaut = 2026 if 2026 in annees_disponibles else annees_disponibles[0]
    annee_choisie = st.selectbox(
        "Année",
        options=annees_disponibles,
        index=annees_disponibles.index(annee_defaut),
    )

    st.caption(
        "ℹ️ Les indicateurs « devisé » / « attribué » et les graphiques par poste / par opération "
        "portent sur les dossiers **clôturés** durant l'année sélectionnée (statut « Clos »), "
        "sur la base de la date de clôture. Les graphiques hebdomadaires « heures attribuées » "
        "et « taux d'attribution » portent quant à eux sur l'ensemble des pointages de l'année, "
        "quel que soit le statut du dossier."
    )

    # -- Données filtrées sur l'année ---------------------------------------
    pointages_annee = df_pointages.loc[df_pointages["iso_year"] == annee_choisie].copy()

    of_annee = df_of.loc[
        (df_of["date_cloture"].dt.year == annee_choisie)
        & (df_of["statut_production"] == "Clos")
    ].copy()

    # -----------------------------------------------------------------------
    # Graphiques hebdomadaires : heures attribuées & taux d'attribution
    # -----------------------------------------------------------------------
    st.subheader("🗓️ Évolution hebdomadaire")

    if not pointages_annee.empty:
        weekly = (
            pointages_annee.groupby("iso_week")
            .agg(
                heures_attribuees=("Durée h", "sum"),
                nb_operateurs=("id_operateur", "nunique"),
            )
            .reset_index()
            .sort_values("iso_week")
        )
        weekly["temps_theorique"] = weekly["nb_operateurs"] * 39
        weekly["taux_attribution"] = (
            weekly["heures_attribuees"] / weekly["temps_theorique"].replace(0, float("nan"))
        )
        weekly["semaine"] = weekly["iso_week"].apply(lambda w: f"S{int(w):02d}")

        moyenne_heures_semaine = weekly["heures_attribuees"].mean()
        moyenne_taux_semaine = weekly["taux_attribution"].mean()

        # Temps travaillé théorique = somme, semaine par semaine, du temps théorique
        # calculé exactement comme dans l'onglet "Suivi hebdomadaire"
        # (nb d'opérateurs distincts ayant pointé cette semaine-là x 39h),
        # additionné sur toutes les semaines de l'année où au moins un pointage existe.
        nb_semaines_ytd = len(weekly)
        moyenne_operateurs_semaine = weekly["nb_operateurs"].mean()
        temps_travaille_theorique_annuel = weekly["temps_theorique"].sum()

        # Nombre total d'heures attribuées = somme de tous les pointages de l'année
        # (indépendamment du statut des dossiers), déjà utilisée pour le graphique
        # "Nombre d'heures attribuées par semaine" ci-dessous.
        heures_attribuees_total_annee = weekly["heures_attribuees"].sum()

        # Taux d'attribution (année) = heures attribuées (pointées) / temps théorique,
        # calculé sur les TOTAUX annuels (et non comme une moyenne des taux hebdomadaires).
        # Par construction : Temps travaillé théorique x Taux d'attribution (année)
        #                    = Nombre total d'heures attribuées.
        taux_attribution_annuel = (
            heures_attribuees_total_annee / temps_travaille_theorique_annuel
            if temps_travaille_theorique_annuel > 0
            else float("nan")
        )

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown("**Nombre d'heures attribuées par semaine**")
            fig_heures_semaine = px.bar(
                weekly,
                x="semaine",
                y="heures_attribuees",
                labels={"semaine": "", "heures_attribuees": "Heures attribuées"},
            )
            fig_heures_semaine.add_hline(
                y=moyenne_heures_semaine,
                line_dash="dash",
                line_color="firebrick",
                annotation_text=f"Moyenne : {moyenne_heures_semaine:.1f} h",
                annotation_position="top left",
            )
            st.plotly_chart(fig_heures_semaine, use_container_width=True, key="bar_heures_par_semaine_annuel")

        with col_a2:
            st.markdown("**Taux d'attribution par semaine**")
            st.caption("Heures attribuées / temps théorique de travail opérateur (nb opérateurs × 39 h)")
            fig_taux_semaine = px.line(
                weekly,
                x="semaine",
                y="taux_attribution",
                markers=True,
                labels={"semaine": "", "taux_attribution": "Taux d'attribution"},
            )
            fig_taux_semaine.update_yaxes(tickformat=".0%")
            fig_taux_semaine.add_hline(
                y=moyenne_taux_semaine,
                line_dash="dash",
                line_color="firebrick",
                annotation_text=f"Moyenne : {moyenne_taux_semaine:.0%}",
                annotation_position="top left",
            )
            st.plotly_chart(fig_taux_semaine, use_container_width=True, key="line_taux_par_semaine_annuel")
            st.caption(
                f"ℹ️ La ligne « Moyenne » ({moyenne_taux_semaine:.0%}) est la moyenne simple "
                f"des taux hebdomadaires (chaque semaine compte pour 1, peu importe son volume "
                f"d'heures). Le « Taux d'attribution (année) » affiché plus bas "
                f"({taux_attribution_annuel:.0%}) est légèrement différent : c'est le total des "
                "heures attribuées sur l'année divisé par le total du temps théorique — chaque "
                "semaine y pèse en fonction de son volume réel d'heures, pas de façon égale. "
                "Les deux sont corrects, ils répondent juste à une pondération différente."
            )
    else:
        st.info("Aucun pointage sur l'année sélectionnée.")
        temps_travaille_theorique_annuel = float("nan")
        heures_attribuees_total_annee = float("nan")
        taux_attribution_annuel = float("nan")
        moyenne_taux_semaine = float("nan")
        nb_semaines_ytd = 0
        moyenne_operateurs_semaine = float("nan")

    # -----------------------------------------------------------------------
    # Indicateurs généraux de l'année
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader(f"📈 Indicateurs généraux — {annee_choisie}")

    if not of_annee.empty:
        par_dossier_annee = (
            of_annee.groupby("numero_dossier")[["temps_devis_h", "temps_operateurs_h"]]
            .sum()
            .reset_index()
        )
        nb_dossiers_annee = par_dossier_annee["numero_dossier"].nunique()

        devis_total_annee = par_dossier_annee["temps_devis_h"].sum()
        travaille_total_annee = par_dossier_annee["temps_operateurs_h"].sum()

        devis_moyen_dossier = devis_total_annee / nb_dossiers_annee if nb_dossiers_annee else 0
        travaille_moyen_dossier = travaille_total_annee / nb_dossiers_annee if nb_dossiers_annee else 0

        # Écart relatif entre heures travaillées et heures devisées, calculé sur les
        # TOTAUX (et non comme une moyenne de ratios par dossier), afin que :
        # Nombre total d'heures travaillées / Nombre total d'heures devisées
        # = 1 + écart relatif, exactement.
        ecart_relatif_devis_travaille = (
            (travaille_total_annee - devis_total_annee) / devis_total_annee
            if devis_total_annee > 0
            else float("nan")
        )

        st.markdown("**Les indicateurs clés**")

        # Ligne 1 : temps théorique, heures attribuées, taux d'attribution
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric(
            "Temps travaillé théorique",
            f"{temps_travaille_theorique_annuel:.0f} h" if pd.notna(temps_travaille_theorique_annuel) else "–",
            help="Somme, semaine par semaine, du nombre d'opérateurs distincts ayant "
                 "pointé cette semaine-là × 39 h (calcul identique à celui de l'onglet "
                 f"« Suivi hebdomadaire »). Sur les {nb_semaines_ytd} semaines de l'année "
                 f"comportant au moins un pointage, cela représente en moyenne "
                 f"{moyenne_operateurs_semaine:.1f} opérateur(s) actif(s) par semaine. "
                 "C'est le volume d'heures que l'effectif aurait dû produire en théorie "
                 "sur la période (base 39 h/semaine/opérateur).",
        )
        r1c2.metric(
            "Nombre total d'heures attribuées",
            f"{heures_attribuees_total_annee:.1f} h" if pd.notna(heures_attribuees_total_annee) else "–",
            help="Somme de tous les pointages opérateurs de l'année (feuille "
                 "« temps_reel_operateur »), tous dossiers confondus, indépendamment de "
                 "leur statut (clos ou non). C'est le numérateur du taux d'attribution "
                 "et la même donnée que le graphique « Nombre d'heures attribuées par "
                 "semaine » ci-dessus (mais cumulée sur l'année entière).",
        )
        r1c3.metric(
            "Taux d'attribution (année)",
            f"{taux_attribution_annuel:.0%}" if pd.notna(taux_attribution_annuel) else "–",
            help="Nombre total d'heures attribuées ÷ Temps travaillé théorique. "
                 "Répond à : sur le temps que les opérateurs auraient dû travailler, "
                 "quelle part a effectivement été attribuée (pointée) à un dossier ? "
                 "Par construction : Temps travaillé théorique × Taux d'attribution "
                 "(année) = Nombre total d'heures attribuées.",
        )

        # Ligne 2 : dossiers clôturés, devisé, travaillé
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric(
            "Nombre de dossiers clôturés",
            f"{nb_dossiers_annee}",
            help=f"Nombre de dossiers avec le statut « Clos » et une date de clôture en {annee_choisie}.",
        )
        r2c2.metric(
            "Nombre total d'heures devisées",
            f"{devis_total_annee:.1f} h",
            help="Somme des heures devisées (temps prévu au devis) sur l'ensemble "
                 f"des dossiers clôturés en {annee_choisie}.",
        )
        r2c3.metric(
            "Nombre total d'heures travaillées",
            f"{travaille_total_annee:.1f} h",
            help="Somme des heures réellement travaillées, telles qu'enregistrées sur "
                 f"les ordres de fabrication des dossiers clôturés en {annee_choisie}. "
                 "⚠️ Cette donnée porte uniquement sur les dossiers clôturés cette année : "
                 "elle diffère donc du « Nombre total d'heures attribuées » ci-dessus, qui "
                 "porte sur tous les pointages de l'année, y compris ceux liés à des "
                 "dossiers pas encore clôturés.",
        )

        # Ligne 3 : écart relatif, moyennes par dossier
        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric(
            "Écart relatif travaillé / devisé",
            f"{ecart_relatif_devis_travaille:+.1%}" if pd.notna(ecart_relatif_devis_travaille) else "–",
            help="(Nombre total d'heures travaillées − Nombre total d'heures devisées) "
                 "÷ Nombre total d'heures devisées, calculé sur l'ensemble des dossiers "
                 "clôturés dans l'année. Répond à : en général, dépasse-t-on le devis ou "
                 "le dossier est-il réalisé plus vite que prévu ? "
                 "(positif = dépassement global du devis, négatif = gain de temps global). "
                 "Par construction : Nombre total d'heures travaillées ÷ Nombre total "
                 "d'heures devisées = 1 + Écart relatif.",
        )
        r3c2.metric(
            "Heures devisées moy. / dossier",
            f"{devis_moyen_dossier:.1f} h",
            help="Nombre total d'heures devisées ÷ nombre de dossiers clôturés.",
        )
        r3c3.metric(
            "Heures travaillées moy. / dossier",
            f"{travaille_moyen_dossier:.1f} h",
            help="Nombre total d'heures travaillées ÷ nombre de dossiers clôturés.",
        )

        # -------------------------------------------------------------------
        # Temps par poste
        # -------------------------------------------------------------------
        st.markdown("**Temps totaux par poste — devisé vs pointé**")
        par_poste_annee = (
            of_annee.groupby("poste", dropna=False)[["temps_devis_h", "temps_operateurs_h"]]
            .sum()
            .rename(columns={"temps_devis_h": "Temps devisé", "temps_operateurs_h": "Temps réalisé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        par_poste_annee["poste"] = par_poste_annee["poste"].fillna("Non défini")

        fig_poste_annee = px.bar(
            par_poste_annee,
            y="poste",
            x=["Temps réalisé", "Temps devisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "poste": "", "variable": ""},
            height=max(300, 35 * len(par_poste_annee)),
            color_discrete_map=COLOR_MAP_DEVIS_REALISE,
        )
        fig_poste_annee.update_layout(legend_title_text="")
        st.plotly_chart(fig_poste_annee, use_container_width=True, key="bar_temps_par_poste_annuel")

        # -------------------------------------------------------------------
        # Temps par opération
        # -------------------------------------------------------------------
        st.markdown("**Temps totaux par opération — devisé vs réalisé**")
        par_operation_annee = (
            of_annee.groupby("operation", dropna=False)[["temps_devis_h", "temps_operateurs_h"]]
            .sum()
            .rename(columns={"temps_devis_h": "Temps devisé", "temps_operateurs_h": "Temps réalisé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        par_operation_annee["operation"] = par_operation_annee["operation"].fillna("Non défini")

        fig_operation_annee = px.bar(
            par_operation_annee,
            y="operation",
            x=["Temps réalisé", "Temps devisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "operation": "", "variable": ""},
            height=max(300, 35 * len(par_operation_annee)),
            color_discrete_map=COLOR_MAP_DEVIS_REALISE,
        )
        fig_operation_annee.update_layout(legend_title_text="")
        st.plotly_chart(fig_operation_annee, use_container_width=True, key="bar_temps_par_operation_annuel")
    else:
        st.info("Aucun dossier clôturé sur l'année sélectionnée.")
