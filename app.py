"""
Suivi d'activité hebdomadaire — Dashboard Streamlit
=====================================================

Lancer avec :  streamlit run app.py

Les données sont lues directement dans Supabase, tables :
  - temps_reel_operateur  (pointages des opérateurs)
  - ordres_fabrication    (ordres de fabrication / dossiers)
  - employes              (nom / prénom des opérateurs)

Identifiants à renseigner dans .streamlit/secrets.toml (en local) ou dans
Settings > Secrets (Streamlit Community Cloud) :
  SUPABASE_URL = "https://xxxx.supabase.co"
  SUPABASE_KEY = "..."
"""

import json
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from supabase import create_client

st.set_page_config(
    page_title="Suivi d'activité hebdomadaire",
    page_icon="📊",
    layout="wide",
)

MS_PER_HOUR = 3_600_000  # les durées "temps_devis" / "temps_operateurs" / "duree"
                         # sont stockées en millisecondes dans la base source
TABLE_POINTAGES = "temps_reel_operateur"
TABLE_OF = "ordres_fabrication"
TABLE_EMPLOYES = "employes"
FUSEAU_HORAIRE = "Europe/Paris"
# Les heures de pointage sont enregistrées dans Supabase avec l'heure de Paris
# telle quelle, mais étiquetées "+00" (UTC). Un pointage à 7h00 apparaît donc
# "07:00:00+00" dans Supabase. Les reconvertir en heure de Paris ajoutait à tort
# +2 h en été / +1 h en hiver. True = on lit l'heure stockée telle quelle.
# (Passer à False si la source enregistre un jour de vraies heures UTC.)
HEURES_DEJA_EN_HEURE_LOCALE = True
DUREE_CACHE_S = 600  # les données sont relues dans Supabase au plus toutes les 10 min

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

@st.cache_resource
def get_supabase_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def fetch_table(table: str, page_size: int = 1000) -> pd.DataFrame:
    """Lit une table Supabase entière.

    L'API Supabase renvoie au maximum 1000 lignes par requête : on pagine donc
    (tri par id pour que les pages soient stables).
    """
    client = get_supabase_client()
    rows, start = [], 0
    while True:
        batch = (
            client.table(table)
            .select("*")
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    if not rows:
        # Cas typique : Row Level Security activée sans politique de lecture
        # pour la clé utilisée -> Supabase renvoie 0 ligne sans erreur.
        raise ValueError(
            f"la table « {table} » est vide ou illisible avec la clé fournie "
            "(vérifier les politiques RLS de Supabase)."
        )
    return pd.DataFrame(rows)


def to_local_datetime(serie: pd.Series) -> pd.Series:
    """Convertit une colonne date/heure Supabase en heure locale sans fuseau.

    Voir HEURES_DEJA_EN_HEURE_LOCALE : soit l'heure stockée est déjà l'heure
    de Paris (on retire juste l'étiquette de fuseau), soit c'est une vraie
    heure UTC (on la convertit en heure de Paris).
    """
    serie = pd.to_datetime(serie, errors="coerce", utc=True, format="ISO8601")
    if HEURES_DEJA_EN_HEURE_LOCALE:
        return serie.dt.tz_localize(None)
    return serie.dt.tz_convert(FUSEAU_HORAIRE).dt.tz_localize(None)


@st.cache_data(ttl=DUREE_CACHE_S, show_spinner="Lecture des données Supabase…")
def load_data():
    """Charge et prépare les tables nécessaires depuis Supabase."""
    # --- Table des pointages ---------------------------------------------------
    df_pointages = fetch_table(TABLE_POINTAGES)
    for col in ["created_at", "heure_debut", "heure_fin"]:
        df_pointages[col] = to_local_datetime(df_pointages[col])
    df_pointages["duree"] = pd.to_numeric(df_pointages["duree"], errors="coerce")

    # La colonne "Durée h" fournie dans la base n'est pas fiable (valeurs
    # décalées / manquantes sur une grande partie des lignes). On recalcule
    # donc la durée en heures à partir de la colonne "duree" (en millisecondes),
    # qui elle correspond bien à l'écart heure_fin - heure_debut.
    df_pointages["Durée h"] = df_pointages["duree"] / MS_PER_HOUR

    iso = df_pointages["heure_debut"].dt.isocalendar()
    df_pointages["iso_year"] = iso["year"]
    df_pointages["iso_week"] = iso["week"]
    df_pointages["cal_year"] = df_pointages["heure_debut"].dt.year
    df_pointages["cal_month"] = df_pointages["heure_debut"].dt.month

    # --- Table des ordres de fabrication ---------------------------------------
    df_of = fetch_table(TABLE_OF)
    for col in ["created_at", "date_cloture"]:
        df_of[col] = to_local_datetime(df_of[col])
    for col in ["temps_devis", "temps_operateurs"]:
        df_of[col] = pd.to_numeric(df_of[col], errors="coerce")
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

    # --- Jointure opérateur -> "Prénom NOM" (via la table employes) -------------
    # id_operateur de temps_reel_operateur = id de la table employes.
    # On garde id_operateur pour les comptages (nunique) et on ajoute une
    # colonne "operateur" lisible pour l'affichage.
    df_pointages["operateur"] = build_operateur_labels(df_pointages["id_operateur"])

    return df_pointages, df_of


def _normalise_id(serie: pd.Series) -> pd.Series:
    """Ramène un identifiant (int, float, texte) à une clé texte comparable."""
    return pd.to_numeric(serie, errors="coerce").astype("Int64").astype(str)


def build_operateur_labels(ids_operateur: pd.Series) -> pd.Series:
    """Renvoie "Prénom NOM" pour chaque id_operateur (repli : "Opérateur #id")."""
    cles = _normalise_id(ids_operateur)
    repli = ("Opérateur #" + cles).where(cles != "<NA>", "Opérateur inconnu")
    try:
        df_emp = fetch_table(TABLE_EMPLOYES)
    except Exception as e:  # table illisible -> on garde les id, sans bloquer le dashboard
        st.warning(f"Noms des opérateurs indisponibles ({e}). Affichage des identifiants.")
        return repli.fillna("Opérateur inconnu")

    df_emp = df_emp.copy()
    df_emp["cle"] = _normalise_id(df_emp["id"])
    df_emp["libelle"] = (
        df_emp.get("prenom", pd.Series("", index=df_emp.index)).fillna("").astype(str).str.strip()
        + " "
        + df_emp.get("nom", pd.Series("", index=df_emp.index)).fillna("").astype(str).str.strip().str.upper()
    ).str.strip()
    df_emp.loc[df_emp["libelle"] == "", "libelle"] = "Opérateur #" + df_emp["cle"]

    # Homonymes éventuels : on ajoute l'id pour les distinguer dans les graphiques
    doublons = df_emp["libelle"].duplicated(keep=False)
    df_emp.loc[doublons, "libelle"] = df_emp.loc[doublons, "libelle"] + " (#" + df_emp.loc[doublons, "cle"] + ")"

    noms = cles.map(df_emp.drop_duplicates("cle").set_index("cle")["libelle"])
    return noms.fillna(repli).fillna("Opérateur inconnu")


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


TAUX_HORAIRE_JOUR = 7.8  # heures théoriques par opérateur et par jour ouvré travaillé


def temps_theorique_semaine(pointages_semaine: pd.DataFrame, monday: date, sunday: date,
                             taux_horaire: float = TAUX_HORAIRE_JOUR) -> float:
    """Temps travaillé théorique d'une semaine ISO complète.

    = (nb d'opérateurs distincts ayant pointé au moins une fois durant la semaine)
      x (jours ouvrés de la semaine) x taux_horaire.

    Choix retenu après plusieurs itérations : le pointage jour par jour s'est révélé
    trop peu fiable (oublis de pointage isolés, sur-pointage compensatoire le
    lendemain) pour servir de base à un calcul journalier — cf. échanges précédents.
    À l'échelle de la semaine en revanche, un opérateur qui a travaillé pointe
    presque toujours au moins une fois ; on suppose donc que tout opérateur ayant
    pointé au moins une fois durant la semaine a été présent tous les jours ouvrés
    de cette semaine.

    Limite assumée : surestime le temps théorique des opérateurs réellement à temps
    partiel ou n'ayant travaillé qu'une partie de la semaine (congé, arrêt, entrée/
    sortie en cours de semaine).
    """
    nb_operateurs = pointages_semaine["id_operateur"].nunique()
    jours_ouvres_semaine = nb_jours_ouvres(monday, sunday)
    return nb_operateurs * jours_ouvres_semaine * taux_horaire


def temps_theorique_mois_via_semaines(df_pointages: pd.DataFrame, premier_jour: date, dernier_jour: date,
                                       taux_horaire: float = TAUX_HORAIRE_JOUR) -> float:
    """Temps travaillé théorique d'un mois calendaire, calculé en sommant les semaines.

    Pour chaque semaine ISO chevauchant le mois : le nombre d'opérateurs distincts est
    calculé sur la semaine ISO COMPLÈTE (y compris les jours qui débordent sur le mois
    voisin), afin de ne pas sous-compter un opérateur présent toute la semaine mais
    dont les pointages tombent en partie sur l'autre mois. En revanche, seuls les
    jours ouvrés de cette semaine qui appartiennent réellement au mois sélectionné
    sont comptés dans le volume d'heures — c'est ce qui règle le problème des
    semaines à cheval sur deux mois : chaque jour ouvré n'est valorisé que dans le
    mois auquel il appartient, avec l'effectif de sa semaine complète.
    """
    total = 0.0
    lundi = premier_jour - timedelta(days=premier_jour.weekday())
    while lundi <= dernier_jour:
        dimanche = lundi + timedelta(days=6)
        iso_year, iso_week, _ = lundi.isocalendar()
        mask_semaine = (df_pointages["iso_year"] == iso_year) & (df_pointages["iso_week"] == iso_week)
        nb_operateurs_semaine = df_pointages.loc[mask_semaine, "id_operateur"].nunique()

        debut_intersection = max(lundi, premier_jour)
        fin_intersection = min(dimanche, dernier_jour)
        jours_ouvres_dans_mois = (
            nb_jours_ouvres(debut_intersection, fin_intersection) if debut_intersection <= fin_intersection else 0
        )

        total += nb_operateurs_semaine * jours_ouvres_dans_mois * taux_horaire
        lundi += timedelta(days=7)

    return total


def temps_theorique_par_groupe_semaine(df_pointages: pd.DataFrame, group_col: str,
                                        taux_horaire: float = TAUX_HORAIRE_JOUR) -> pd.Series:
    """Calcule le temps théorique semaine par semaine (nb opérateurs distincts de la
    semaine ISO x jours ouvrés de la semaine x taux_horaire), puis regroupe le résultat
    par `group_col` (ex. iso_week directement, ou cal_month pour un total mensuel basé
    sur des semaines ISO complètes non ajustées aux frontières du mois)."""
    cols = list(dict.fromkeys(["iso_year", "iso_week", group_col]))  # évite les doublons de colonnes
    semaines = (
        df_pointages[cols]
        .dropna(subset=["iso_year", "iso_week"])
        .drop_duplicates(subset=["iso_year", "iso_week"])
    )
    valeurs = []
    for _, row in semaines.iterrows():
        monday, sunday = week_bounds(row["iso_year"], row["iso_week"])
        mask = (df_pointages["iso_year"] == row["iso_year"]) & (df_pointages["iso_week"] == row["iso_week"])
        valeurs.append(temps_theorique_semaine(df_pointages.loc[mask], monday, sunday, taux_horaire))
    semaines = semaines.assign(temps_theorique=valeurs)
    return semaines.groupby(group_col)["temps_theorique"].sum()


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
# Affichage des tableaux : AgGrid (tri + filtre directement dans les en-têtes)
# ---------------------------------------------------------------------------

# Colonnes numériques à filtrer comme du texte (identifiants, numéros…)
_COLONNES_IDENTIFIANTS = re.compile(r"(numero|ordre_fabrication|reference|dossier)", re.IGNORECASE)
# Ligne de filtres affichée sous les en-têtes (en plus de l'icône de filtre de chaque colonne)
LIGNE_DE_FILTRES = True

_TRADUCTION_AGGRID = {
    # Filtres
    "filterOoo": "Filtrer…", "searchOoo": "Rechercher…", "equals": "Égal à",
    "notEqual": "Différent de", "blank": "Vide", "notBlank": "Non vide",
    "empty": "Choisir…", "lessThan": "Inférieur à", "greaterThan": "Supérieur à",
    "lessThanOrEqual": "Inférieur ou égal à", "greaterThanOrEqual": "Supérieur ou égal à",
    "inRange": "Entre", "inRangeStart": "De", "inRangeEnd": "À",
    "contains": "Contient", "notContains": "Ne contient pas",
    "startsWith": "Commence par", "endsWith": "Se termine par",
    "before": "Avant", "after": "Après", "dateFormatOoo": "jj/mm/aaaa",
    "andCondition": "ET", "orCondition": "OU",
    "applyFilter": "Appliquer", "resetFilter": "Réinitialiser",
    "clearFilter": "Effacer", "cancelFilter": "Annuler",
    # Menu de colonne / divers
    "sortAscending": "Tri croissant", "sortDescending": "Tri décroissant",
    "sortUnSort": "Annuler le tri", "pinColumn": "Épingler la colonne",
    "pinLeft": "Épingler à gauche", "pinRight": "Épingler à droite",
    "noPin": "Ne pas épingler", "autosizeThisColumn": "Ajuster cette colonne",
    "autosizeAllColumns": "Ajuster toutes les colonnes", "resetColumns": "Réinitialiser les colonnes",
    "columnFilter": "Filtre de colonne", "columnChooser": "Choisir les colonnes",
    "noRowsToShow": "Aucune ligne à afficher", "loadingOoo": "Chargement…",
    "page": "Page", "of": "sur", "to": "à", "more": "plus",
}

# Formatage des dates : jj/mm/aaaa hh:mm
_JS_FORMAT_DATE = JsCode("""
function(p) {
    if (!p.value) return '';
    const d = new Date(p.value);
    if (isNaN(d)) return p.value;
    const z = n => String(n).padStart(2, '0');
    return z(d.getDate()) + '/' + z(d.getMonth() + 1) + '/' + d.getFullYear()
         + ' ' + z(d.getHours()) + ':' + z(d.getMinutes());
}""")

# Filtre de date : compare uniquement le jour (sans l'heure)
_JS_COMPARATEUR_DATE = JsCode("""
function(filtre, valeur) {
    if (!valeur) return -1;
    const d = new Date(valeur);
    const jour = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    if (jour < filtre) return -1;
    if (jour > filtre) return 1;
    return 0;
}""")

# Couleur de fond de ligne (colonne cachée "_couleur")
_JS_STYLE_LIGNE = JsCode("""
function(p) {
    if (p.data && p.data._couleur) return { background: p.data._couleur };
    return null;
}""")


def _js_format_nombre(decimales: int, suffixe: str = "") -> JsCode:
    return JsCode(f"""
function(p) {{
    if (p.value === null || p.value === undefined || isNaN(p.value)) return '–';
    return Number(p.value).toFixed({decimales}) + '{suffixe}';
}}""")


def afficher_tableau(df: pd.DataFrame, key: str, style_ligne=None, hauteur_max: int = 450) -> None:
    """Affiche un tableau triable et filtrable depuis les en-têtes de colonnes.

    - Texte / identifiants : filtre « contient », « commence par »…
    - Nombres (durées, écarts) : filtre « supérieur à », « entre »…
    - Dates : filtre « avant », « après », « entre »…
    `style_ligne` : fonction (row -> liste de styles CSS) reprise des anciennes
    mises en couleur ; seule la couleur de fond est conservée.
    """
    df = df.copy()

    if style_ligne is not None:
        df["_couleur"] = df.apply(
            lambda r: (style_ligne(r) or [""])[0].replace("background-color:", "").strip() or None,
            axis=1,
        )

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        sortable=True, resizable=True, filter=True,
        floatingFilter=LIGNE_DE_FILTRES, suppressHeaderMenuButton=False,
        minWidth=90,
    )

    for col in df.columns:
        serie = df[col]
        if col == "_couleur":
            gb.configure_column(col, hide=True, filter=False)
        elif pd.api.types.is_datetime64_any_dtype(serie):
            df[col] = serie.dt.strftime("%Y-%m-%dT%H:%M:%S").where(serie.notna(), None)
            gb.configure_column(
                col, filter="agDateColumnFilter", valueFormatter=_JS_FORMAT_DATE,
                filterParams={"comparator": _JS_COMPARATEUR_DATE, "inRangeInclusive": True,
                              "buttons": ["reset"]},
            )
        elif pd.api.types.is_numeric_dtype(serie) and _COLONNES_IDENTIFIANTS.search(col):
            # Numéros de dossier / devis : recherche texte, tri numérique conservé
            nom_js = json.dumps(col)
            gb.configure_column(
                col, filter="agTextColumnFilter",
                filterValueGetter=JsCode(
                    f"function(p) {{ const v = p.data[{nom_js}]; return v == null ? null : String(v); }}"
                ),
                filterParams={"buttons": ["reset"]},
            )
        elif pd.api.types.is_numeric_dtype(serie) and col != "id":
            if "%" in col:
                fmt = _js_format_nombre(0, "%")
            else:
                fmt = _js_format_nombre(2)
            gb.configure_column(
                col, type=["numericColumn"], filter="agNumberColumnFilter",
                valueFormatter=fmt, filterParams={"buttons": ["reset"]},
            )
        elif pd.api.types.is_numeric_dtype(serie):
            gb.configure_column(col, filter="agNumberColumnFilter", filterParams={"buttons": ["reset"]})
        else:
            df[col] = serie.astype("string").where(serie.notna(), None)
            gb.configure_column(col, filter="agTextColumnFilter", filterParams={"buttons": ["reset"]})

    gb.configure_grid_options(
        localeText=_TRADUCTION_AGGRID,
        getRowStyle=_JS_STYLE_LIGNE,
        autoSizeStrategy={"type": "fitGridWidth", "defaultMinWidth": 90},
        enableCellTextSelection=True,
        animateRows=False,
    )

    entete = 2 * 42 if LIGNE_DE_FILTRES else 42
    hauteur = min(hauteur_max, entete + 30 * max(len(df), 1) + 12)

    AgGrid(
        df,
        gridOptions=gb.build(),
        height=hauteur,
        theme="streamlit",
        allow_unsafe_jscode=True,
        update_on=[],               # filtrer / trier ne relance pas le script
        show_toolbar=False,
        show_search=False,
        show_download_button=False,
        key=key,
    )


# ---------------------------------------------------------------------------
# Interface — chargement des données
# ---------------------------------------------------------------------------

st.title("📊 Suivi d'activité hebdomadaire")

with st.sidebar:
    st.header("📁 Données")
    st.caption(f"Source : Supabase — actualisation automatique toutes les {DUREE_CACHE_S // 60} min.")
    if st.button("🔄 Actualiser maintenant", help="Relit immédiatement les données dans Supabase."):
        st.cache_data.clear()
        st.rerun()

try:
    df_pointages, df_of = load_data()
except Exception as e:
    st.error(f"Impossible de lire les données Supabase : {e}")
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
        st.warning("Aucune semaine exploitable n'a été trouvée dans la base.")
        st.stop()

    with st.sidebar:
        st.header("🗓️ Semaine")
        selected_label = st.selectbox("Sélectionner une semaine", weeks_df["label"])

    selected_row = weeks_df.loc[weeks_df["label"] == selected_label].iloc[0]
    sel_year, sel_week = int(selected_row["iso_year"]), int(selected_row["iso_week"])
    monday, sunday = week_bounds(sel_year, sel_week)

    st.caption(f"Semaine sélectionnée : **du {monday.strftime('%d/%m/%Y')} au {sunday.strftime('%d/%m/%Y')}**")
    st.caption(
        f"🕒 Dernière date de clôture présente dans la base : "
        f"**{df_of['date_cloture'].max().strftime('%d/%m/%Y') if df_of['date_cloture'].notna().any() else 'aucune'}** "
        f"— pour forcer la mise à jour, cliquez sur « Actualiser maintenant »."
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
    jours_ouvres_semaine_sel = nb_jours_ouvres(monday, sunday)
    temps_theorique = temps_theorique_semaine(pointages_semaine, monday, sunday)
    taux_attribution = (heures_attribuees / temps_theorique) if temps_theorique > 0 else 0

    st.subheader("👷 Activité des opérateurs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre d'heures attribuées", f"{heures_attribuees:.1f} h")
    c2.metric("Opérateurs ayant pointé", f"{nb_operateurs}")
    c3.metric(
        "Temps travaillé théorique",
        f"{temps_theorique:.0f} h",
        help=(
            f"{nb_operateurs} opérateur(s) ayant pointé au moins une fois cette semaine × "
            f"{jours_ouvres_semaine_sel} jours ouvrés × 7,8 h. On suppose qu'un opérateur "
            "ayant pointé au moins une fois dans la semaine y a travaillé toute la semaine."
        ),
    )
    c4.metric(
        "Taux d'attribution",
        f"{taux_attribution:.0%}",
        help="Heures attribuées / temps théorique (nb opérateurs de la semaine × jours ouvrés × 7,8 h)",
    )

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
            data2 = pie_top_n(pointages_semaine, "operateur", "Durée h")
            fig2 = px.pie(data2, names="operateur", values="Durée h", hole=0.35)
            fig2.update_traces(
                textposition="inside",
                textinfo="label+percent",
                texttemplate="%{label}<br>%{value:.1f} h (%{percent})",
            )
            st.plotly_chart(fig2, use_container_width=True, key="pie_heures_par_operateur")
        else:
            st.info("Aucun pointage sur cette semaine.")

    st.caption(
        "ℹ️ Dans la base source, le champ « ordre de fabrication » des pointages correspond "
        "au numéro de dossier."
    )

    st.markdown("**Détail des pointages de la semaine**")
    if not pointages_semaine.empty:
        table_pointages = pointages_semaine[
            ["id", "created_at", "operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
        ].sort_values("heure_debut", ascending=False).reset_index(drop=True)
        table_pointages["Durée h"] = table_pointages["Durée h"].round(2)
        afficher_tableau(table_pointages, key="grille_pointages_hebdo")
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

            afficher_tableau(table_of, key="grille_table_of", style_ligne=highlight_row)
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

            afficher_tableau(table_spec_of, key="grille_table_spec_of", style_ligne=highlight_row_spec)
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
        st.warning("Aucun mois exploitable n'a été trouvé dans la base.")
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
        f"🕒 Dernière date de clôture présente dans la base : "
        f"**{df_of['date_cloture'].max().strftime('%d/%m/%Y') if df_of['date_cloture'].notna().any() else 'aucune'}** "
        f"— pour forcer la mise à jour, cliquez sur « Actualiser maintenant »."
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
    temps_theorique_m = temps_theorique_mois_via_semaines(df_pointages, premier_jour, dernier_jour)
    taux_attribution_m = (heures_attribuees_m / temps_theorique_m) if temps_theorique_m > 0 else 0

    st.subheader("👷 Activité des opérateurs")
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Nombre d'heures attribuées", f"{heures_attribuees_m:.1f} h")
    cm2.metric("Opérateurs ayant pointé", f"{nb_operateurs_m}")
    cm3.metric(
        "Temps travaillé théorique",
        f"{temps_theorique_m:.0f} h",
        help=(
            "Somme, semaine par semaine, du temps théorique de chaque semaine ISO "
            "chevauchant le mois (nb d'opérateurs distincts de la semaine complète × "
            "jours ouvrés de cette semaine tombant dans le mois × 7,8 h). Pour les "
            "semaines à cheval sur deux mois, l'effectif est calculé sur la semaine "
            "entière mais seuls les jours ouvrés du mois sélectionné sont valorisés — "
            f"le mois compte au total {jours_ouvres_m} jours ouvrés."
        ),
    )
    cm4.metric(
        "Taux d'attribution",
        f"{taux_attribution_m:.0%}",
        help="Heures attribuées / temps théorique (somme, semaine par semaine, du nb d'opérateurs × jours ouvrés × 7,8 h)",
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
            data2_m = pie_top_n(pointages_mois, "operateur", "Durée h")
            fig2_m = px.pie(data2_m, names="operateur", values="Durée h", hole=0.35)
            fig2_m.update_traces(
                textposition="inside",
                textinfo="label+percent",
                texttemplate="%{label}<br>%{value:.1f} h (%{percent})",
            )
            st.plotly_chart(fig2_m, use_container_width=True, key="pie_heures_par_operateur_mensuel")
        else:
            st.info("Aucun pointage sur ce mois.")

    st.caption(
        "ℹ️ Dans la base source, le champ « ordre de fabrication » des pointages correspond "
        "au numéro de dossier."
    )

    st.markdown("**Détail des pointages du mois**")
    if not pointages_mois.empty:
        table_pointages_m = pointages_mois[
            ["id", "created_at", "operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
        ].sort_values("heure_debut", ascending=False).reset_index(drop=True)
        table_pointages_m["Durée h"] = table_pointages_m["Durée h"].round(2)
        afficher_tableau(table_pointages_m, key="grille_pointages_mensuel")
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

            afficher_tableau(table_of_m, key="grille_table_of_m", style_ligne=highlight_row_mensuel)
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

            afficher_tableau(table_spec_of_m, key="grille_table_spec_of_m", style_ligne=highlight_row_spec_mensuel)
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
        st.warning("Aucune année exploitable n'a été trouvée dans la base.")
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
    # Calculs hebdomadaires (utilisés à la fois par "Indicateurs généraux"
    # ci-dessous et par le graphique "Évolution hebdomadaire" plus bas)
    # -----------------------------------------------------------------------
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
        weekly["temps_theorique"] = weekly["iso_week"].map(
            temps_theorique_par_groupe_semaine(pointages_annee, "iso_week")
        )
        weekly["taux_attribution"] = (
            weekly["heures_attribuees"] / weekly["temps_theorique"].replace(0, float("nan"))
        )
        weekly["semaine"] = weekly["iso_week"].apply(lambda w: f"S{int(w):02d}")

        moyenne_heures_semaine = weekly["heures_attribuees"].mean()
        moyenne_taux_semaine = weekly["taux_attribution"].mean()

        # Temps travaillé théorique = somme, semaine par semaine, du temps théorique
        # calculé exactement comme dans l'onglet "Suivi hebdomadaire" (nb d'opérateurs
        # distincts ayant pointé au moins une fois durant la semaine x jours ouvrés de
        # la semaine x 7,8h), additionné sur toutes les semaines de l'année où au
        # moins un pointage existe.
        nb_semaines_ytd = len(weekly)
        moyenne_operateurs_semaine = weekly["nb_operateurs"].mean()
        temps_travaille_theorique_annuel = weekly["temps_theorique"].sum()

        # Nombre total d'heures attribuées = somme de tous les pointages de l'année
        # (indépendamment du statut des dossiers), déjà utilisée pour le graphique
        # "Nombre d'heures attribuées par semaine" plus bas.
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
    else:
        weekly = pd.DataFrame()
        temps_travaille_theorique_annuel = float("nan")
        heures_attribuees_total_annee = float("nan")
        taux_attribution_annuel = float("nan")
        moyenne_heures_semaine = float("nan")
        moyenne_taux_semaine = float("nan")
        nb_semaines_ytd = 0
        moyenne_operateurs_semaine = float("nan")

    # -----------------------------------------------------------------------
    # Indicateurs généraux de l'année
    # -----------------------------------------------------------------------
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

        # "Nombre total d'heures travaillées et livrées" = heures réellement
        # travaillées sur les dossiers déjà clôturés (livrés) cette année.
        heures_travaillees_livrees_annee = travaille_total_annee

        # "Nombre total d'heures travaillées et non-livrées" = Nombre total d'heures
        # attribuées (tous pointages de l'année, tous dossiers confondus) − Nombre
        # total d'heures travaillées et livrées. Cela correspond à la somme des
        # pointages de la feuille "temps_reel_operateur" qui sont rattachés à un
        # dossier dont le statut n'est pas encore "Clos".
        heures_travaillees_non_livrees_annee = (
            heures_attribuees_total_annee - heures_travaillees_livrees_annee
            if pd.notna(heures_attribuees_total_annee)
            else float("nan")
        )

        # Écart relatif entre heures travaillées (et livrées) et heures devisées,
        # calculé sur les TOTAUX (et non comme une moyenne de ratios par dossier),
        # afin que : Nombre total d'heures travaillées et livrées / Nombre total
        # d'heures devisées = 1 + écart relatif, exactement.
        ecart_relatif_devis_travaille = (
            (heures_travaillees_livrees_annee - devis_total_annee) / devis_total_annee
            if devis_total_annee > 0
            else float("nan")
        )

        st.markdown("**Les indicateurs clés**")
        st.caption(
            "À lire comme une formule : chaque encadré est un indicateur, les signes "
            "×, = et + montrent comment ils se déduisent les uns des autres."
        )

        def _kpi_box(label: str, value: str, help_text: str) -> None:
            """Affiche un indicateur dans un encadré, façon 'bloc de formule'."""
            with st.container(border=True):
                st.metric(label, value, help=help_text)

        def _operateur(symbole: str) -> None:
            """Affiche un opérateur (×, =, +) centré et aligné avec les encadrés."""
            st.markdown(
                "<div style='text-align:center; font-size:2.1rem; font-weight:600; "
                f"padding-top:2.6rem;'>{symbole}</div>",
                unsafe_allow_html=True,
            )

        # Grille commune à toutes les lignes : 4 colonnes de contenu (ColA, ColB,
        # ColC, ColD) séparées par des opérateurs. La ColC est la colonne pivot :
        # elle contient "Nombre total d'heures attribuées" (ligne 1), "Nombre total
        # d'heures travaillées et livrées" (ligne 2) et "Nombre total d'heures
        # devisées" (ligne 3), alignés à la verticale pour que ces indicateurs à
        # comparer entre eux tombent dans la même colonne.
        GRID = [3, 0.6, 3, 0.6, 3, 0.6, 3]

        # -- Ligne 1 : Temps théorique × Taux d'attribution = Heures attribuées --
        l1a, l1op1, l1b, l1op2, l1c, l1op3, l1d = st.columns(GRID)
        with l1a:
            _kpi_box(
                "Temps travaillé théorique",
                f"{temps_travaille_theorique_annuel:.0f} h" if pd.notna(temps_travaille_theorique_annuel) else "–",
                help_text="Somme, semaine par semaine, du temps théorique de chaque semaine "
                          "(nb d'opérateurs distincts ayant pointé au moins une fois durant la "
                          "semaine × jours ouvrés de la semaine × 7,8 h — calcul identique à "
                          "celui de l'onglet « Suivi hebdomadaire »). Sur "
                          f"les {nb_semaines_ytd} semaines de l'année comportant au moins un "
                          f"pointage, cela représente en moyenne {moyenne_operateurs_semaine:.1f} "
                          "opérateur(s) actif(s) par semaine. C'est le volume d'heures que "
                          "l'effectif aurait dû produire en théorie sur la période, en supposant "
                          "qu'un opérateur ayant pointé au moins une fois dans une semaine y a "
                          "travaillé toute la semaine.",
            )
        with l1op1:
            _operateur("×")
        with l1b:
            _kpi_box(
                "Taux d'attribution (année)",
                f"{taux_attribution_annuel:.0%}" if pd.notna(taux_attribution_annuel) else "–",
                help_text="Part du temps théorique qui a effectivement été attribuée "
                          "(pointée) à un dossier, quel que soit son statut.",
            )
        with l1op2:
            _operateur("=")
        with l1c:
            _kpi_box(
                "Nombre total d'heures attribuées",
                f"{heures_attribuees_total_annee:.1f} h" if pd.notna(heures_attribuees_total_annee) else "–",
                help_text="Somme de tous les pointages opérateurs de l'année (feuille "
                          "« temps_reel_operateur »), tous dossiers confondus, indépendamment "
                          "de leur statut (clos ou non).",
            )
        # l1op3 et l1d restent vides sur cette ligne

        # -- Ligne 2 : Heures attribuées = Heures livrées + Heures non-livrées --
        l2a, l2op1, l2b, l2op2, l2c, l2op3, l2d = st.columns(GRID)
        # l2a, l2op1 et l2b restent vides : le résultat de la ligne 1 (colonne C)
        # devient l'opérande de gauche de cette ligne.
        with l2op2:
            _operateur("=")
        with l2c:
            _kpi_box(
                "Nombre total d'heures travaillées et livrées",
                f"{heures_travaillees_livrees_annee:.1f} h",
                help_text="Somme des heures réellement travaillées, telles qu'enregistrées "
                          "sur les ordres de fabrication des dossiers déjà clôturés (livrés) "
                          f"en {annee_choisie}.",
            )
        with l2op3:
            _operateur("+")
        with l2d:
            _kpi_box(
                "Nombre total d'heures travaillées et non-livrées",
                f"{heures_travaillees_non_livrees_annee:.1f} h"
                if pd.notna(heures_travaillees_non_livrees_annee) else "–",
                help_text="Nombre total d'heures attribuées − Nombre total d'heures "
                          "travaillées et livrées. Correspond à la somme des pointages "
                          "attribués à un dossier dont le statut n'est pas encore « Clos » "
                          "(travail en cours, pas encore livré).",
            )

        # -- Écart (balance) entre heures livrées et heures devisées --
        # Positionné dans la même colonne (ColC) que "livrées" au-dessus et
        # "devisées" en dessous, pour bien montrer les deux indicateurs comparés.
        e_a, e_op1, e_b, e_op2, e_c, e_op3, e_d = st.columns(GRID)
        with e_c:
            ecart_txt = (
                f"{ecart_relatif_devis_travaille:+.1%}"
                if pd.notna(ecart_relatif_devis_travaille) else "–"
            )
            st.markdown(
                "<div style='text-align:center; line-height:1.9;'>"
                "<span style='font-size:1.7rem;'>⚖️</span><br>"
                "<span style='font-size:1.3rem;'>↕️</span><br>"
                "Écart relatif travaillé / devisé<br>"
                f"<span style='font-size:1.4rem; font-weight:600;'>{ecart_txt}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "ℹ️ Compare le total des heures travaillées et livrées (ci-dessus) au total "
            "des heures devisées (ci-dessous) : (travaillé et livré − devisé) ÷ devisé. "
            "Positif = dépassement du devis, négatif = gain de temps."
        )

        # -- Ligne 3 : Dossiers clôturés × Heures devisées moy. = Heures devisées --
        l3a, l3op1, l3b, l3op2, l3c, l3op3, l3d = st.columns(GRID)
        with l3a:
            _kpi_box(
                "Nombre de dossiers clôturés",
                f"{nb_dossiers_annee}",
                help_text=f"Nombre de dossiers avec le statut « Clos » et une date de "
                          f"clôture en {annee_choisie}.",
            )
        with l3op1:
            _operateur("×")
        with l3b:
            _kpi_box(
                "Heures devisées moy. / dossier",
                f"{devis_moyen_dossier:.1f} h",
                help_text="Nombre total d'heures devisées ÷ nombre de dossiers clôturés.",
            )
        with l3op2:
            _operateur("=")
        with l3c:
            _kpi_box(
                "Nombre total d'heures devisées",
                f"{devis_total_annee:.1f} h",
                help_text="Somme des heures devisées (temps prévu au devis) sur l'ensemble "
                          f"des dossiers clôturés en {annee_choisie}.",
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

    # -----------------------------------------------------------------------
    # Graphiques hebdomadaires : heures attribuées & taux d'attribution
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🗓️ Évolution hebdomadaire")

    if not pointages_annee.empty:
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
            st.caption(
                "Heures attribuées / temps théorique de travail opérateur "
                "(nb d'opérateurs distincts de la semaine × jours ouvrés × 7,8 h)"
            )
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
                f"d'heures). Le « Taux d'attribution (année) » affiché plus haut "
                f"({taux_attribution_annuel:.0%}) est légèrement différent : c'est le total des "
                "heures attribuées sur l'année divisé par le total du temps théorique — chaque "
                "semaine y pèse en fonction de son volume réel d'heures, pas de façon égale. "
                "Les deux sont corrects, ils répondent juste à une pondération différente."
            )
    else:
        st.info("Aucun pointage sur l'année sélectionnée.")

    # -----------------------------------------------------------------------
    # Graphiques mensuels : heures attribuées & taux d'attribution
    # -----------------------------------------------------------------------
    st.subheader("🗓️ Évolution mensuelle")

    if not pointages_annee.empty:
        monthly = (
            pointages_annee.groupby("cal_month")
            .agg(
                heures_attribuees=("Durée h", "sum"),
                nb_operateurs=("id_operateur", "nunique"),
            )
            .reset_index()
            .sort_values("cal_month")
        )
        monthly["jours_ouvres"] = monthly["cal_month"].apply(
            lambda m: nb_jours_ouvres(*month_bounds(annee_choisie, int(m)))
        )
        monthly["temps_theorique"] = monthly["cal_month"].apply(
            lambda m: temps_theorique_mois_via_semaines(df_pointages, *month_bounds(annee_choisie, int(m)))
        )
        monthly["taux_attribution"] = (
            monthly["heures_attribuees"] / monthly["temps_theorique"].replace(0, float("nan"))
        )
        monthly["mois"] = monthly["cal_month"].apply(lambda m: NOMS_MOIS[int(m) - 1])

        moyenne_heures_mois = monthly["heures_attribuees"].mean()
        moyenne_taux_mois = monthly["taux_attribution"].mean()

        col_am1, col_am2 = st.columns(2)

        with col_am1:
            st.markdown("**Nombre d'heures attribuées par mois**")
            fig_heures_mois = px.bar(
                monthly,
                x="mois",
                y="heures_attribuees",
                labels={"mois": "", "heures_attribuees": "Heures attribuées"},
            )
            fig_heures_mois.add_hline(
                y=moyenne_heures_mois,
                line_dash="dash",
                line_color="firebrick",
                annotation_text=f"Moyenne : {moyenne_heures_mois:.1f} h",
                annotation_position="top left",
            )
            st.plotly_chart(fig_heures_mois, use_container_width=True, key="bar_heures_par_mois_annuel")

        with col_am2:
            st.markdown("**Taux d'attribution par mois**")
            st.caption(
                "Heures attribuées / temps théorique de travail opérateur "
                "(somme, semaine par semaine, du nb d'opérateurs × jours ouvrés × 7,8 h)"
            )
            fig_taux_mois = px.line(
                monthly,
                x="mois",
                y="taux_attribution",
                markers=True,
                labels={"mois": "", "taux_attribution": "Taux d'attribution"},
            )
            fig_taux_mois.update_yaxes(tickformat=".0%")
            fig_taux_mois.add_hline(
                y=moyenne_taux_mois,
                line_dash="dash",
                line_color="firebrick",
                annotation_text=f"Moyenne : {moyenne_taux_mois:.0%}",
                annotation_position="top left",
            )
            st.plotly_chart(fig_taux_mois, use_container_width=True, key="line_taux_par_mois_annuel")
            st.caption(
                f"ℹ️ La ligne « Moyenne » ({moyenne_taux_mois:.0%}) est la moyenne simple "
                "des taux mensuels (chaque mois compte pour 1, peu importe son volume "
                "d'heures)."
            )
    else:
        st.info("Aucun pointage sur l'année sélectionnée.")
