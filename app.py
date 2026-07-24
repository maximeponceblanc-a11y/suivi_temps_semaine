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

tab_hebdo, tab_annuel = st.tabs(["📊 Suivi hebdomadaire", "📅 Bilan annuel"])

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
                .rename(columns={"temps_operateurs_h": "Temps de production", "temps_devis_h": "Temps devis"})
                .sort_values("Temps devis", ascending=True)
                .reset_index()
            )
            par_poste_spec["poste"] = par_poste_spec["poste"].fillna("Non défini")

            fig_poste_spec = px.bar(
                par_poste_spec,
                y="poste",
                x=["Temps de production", "Temps devis"],
                orientation="h",
                barmode="group",
                labels={"value": "Heures", "poste": "Poste", "variable": "Type de temps"}
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
    else:
        st.info("Aucun pointage sur l'année sélectionnée.")
        temps_travaille_theorique_annuel = float("nan")
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

        # Ratio travaillé/devisé par dossier : répond à « passe-t-on plus de temps
        # que prévu sur un dossier ? » — à ne pas confondre avec le taux d'attribution.
        ratios_dossier = (
            par_dossier_annee["temps_operateurs_h"]
            / par_dossier_annee["temps_devis_h"].replace(0, float("nan"))
        )
        ratio_travaille_devis_moyen_dossier = ratios_dossier.mean()

        # Taux d'attribution (année) = heures devisées (= heures attribuées à un dossier
        # au moment du devis) / temps travaillé théorique (nb opérateurs x 39h, cumulé
        # semaine par semaine sur l'année). Par construction :
        # devis_total_annee / temps_travaille_theorique_annuel == taux_attribution_annuel
        taux_attribution_annuel = (
            devis_total_annee / temps_travaille_theorique_annuel
            if pd.notna(temps_travaille_theorique_annuel) and temps_travaille_theorique_annuel > 0
            else float("nan")
        )

        st.markdown("**Les deux indicateurs clés**")

        # Ligne 1 : temps théorique & taux d'attribution
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
            "Taux d'attribution (année)",
            f"{taux_attribution_annuel:.0%}" if pd.notna(taux_attribution_annuel) else "–",
            help="Nombre total d'heures devisées (= heures attribuées à un dossier lors "
                 "du devis) ÷ Temps travaillé théorique (indicateur ci-contre). "
                 "Répond à : sur le temps que les opérateurs auraient dû travailler, "
                 "quelle part a été attribuée à un dossier client ? "
                 "Par construction : Nombre total d'heures devisées ÷ Temps travaillé "
                 "théorique = Taux d'attribution (année).",
        )

        # Ligne 2 : volumes d'heures totaux + ratio travaillé/devisé
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric(
            "Nombre total d'heures devisées",
            f"{devis_total_annee:.1f} h",
            help="Somme des heures devisées (temps prévu au devis) sur l'ensemble "
                 f"des dossiers clôturés en {annee_choisie}.",
        )
        r2c2.metric(
            "Nombre total d'heures travaillées",
            f"{travaille_total_annee:.1f} h",
            help="Somme des heures réellement travaillées (pointées par les opérateurs) "
                 f"sur l'ensemble des dossiers clôturés en {annee_choisie}.",
        )
        r2c3.metric(
            "Ratio travaillé / devisé moyen par dossier",
            f"{ratio_travaille_devis_moyen_dossier:.0%}" if pd.notna(ratio_travaille_devis_moyen_dossier) else "–",
            help="Moyenne, sur tous les dossiers clôturés dans l'année, du ratio "
                 "(heures travaillées ÷ heures devisées) de chaque dossier. "
                 "Répond à : passe-t-on en général plus de temps que prévu sur un dossier ? "
                 "(>100 % = dépassement du devis, <100 % = dossier réalisé plus vite que prévu).",
        )

        # Ligne 3 : moyennes par dossier
        r3c1, r3c2, r3c3 = st.columns(3)
        r3c1.metric(
            "Heures devisées moy. / dossier",
            f"{devis_moyen_dossier:.1f} h",
            help="Nombre total d'heures devisées ÷ nombre de dossiers clôturés.",
        )
        r3c2.metric(
            "Heures travaillées moy. / dossier",
            f"{travaille_moyen_dossier:.1f} h",
            help="Nombre total d'heures travaillées ÷ nombre de dossiers clôturés.",
        )

        # Ligne 4 : nombre de dossiers
        r4c1, r4c2, r4c3 = st.columns(3)
        r4c1.metric(
            "Nombre de dossiers clôturés",
            f"{nb_dossiers_annee}",
            help=f"Nombre de dossiers avec le statut « Clos » et une date de clôture en {annee_choisie}.",
        )

        # -------------------------------------------------------------------
        # Temps par poste
        # -------------------------------------------------------------------
        st.markdown("**Temps totaux par poste — devisé vs pointé**")
        par_poste_annee = (
            of_annee.groupby("poste", dropna=False)[["temps_devis_h", "temps_operateurs_h"]]
            .sum()
            .rename(columns={"temps_devis_h": "Temps devisé", "temps_operateurs_h": "Temps pointé"})
            .sort_values("Temps devisé", ascending=True)
            .reset_index()
        )
        par_poste_annee["poste"] = par_poste_annee["poste"].fillna("Non défini")

        fig_poste_annee = px.bar(
            par_poste_annee,
            y="poste",
            x=["Temps devisé", "Temps pointé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "poste": "", "variable": ""},
            height=max(300, 35 * len(par_poste_annee)),
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
            x=["Temps devisé", "Temps réalisé"],
            orientation="h",
            barmode="group",
            labels={"value": "Heures", "operation": "", "variable": ""},
            height=max(300, 35 * len(par_operation_annee)),
        )
        fig_operation_annee.update_layout(legend_title_text="")
        st.plotly_chart(fig_operation_annee, use_container_width=True, key="bar_temps_par_operation_annuel")
    else:
        st.info("Aucun dossier clôturé sur l'année sélectionnée.")
