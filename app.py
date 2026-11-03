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

import base64
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from supabase import create_client

_FAVICON = Path(__file__).parent / "assets" / "favicon_lbfi.png"
st.set_page_config(
    page_title="LBFI — Suivi des temps de production",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "🏭",
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
COULEUR_REALISE = "#94C000"  # vert LBFI
COULEUR_DEVISE = "#BDBDB8"   # gris clair
COLOR_MAP_DEVIS_REALISE = {
    "Temps réalisé": COULEUR_REALISE,
    "Temps devisé": COULEUR_DEVISE,
}

# Style par défaut de tous les graphiques : fond blanc + palette harmonisée au logo
PALETTE_LBFI = [
    "#94C000", "#575756", "#2E86AB", "#F4A259", "#8E6C8A",
    "#5BC0BE", "#E4572E", "#B8B8B3", "#3D5A80", "#C9A227",
]
_modele = go.layout.Template(pio.templates["plotly_white"])
_modele.layout.colorway = PALETTE_LBFI          # utilisé aussi par les camemberts
_modele.layout.font = dict(family="Source Sans Pro, sans-serif", color="#2B2B2A")
_modele.layout.separators = ", "                # format français : virgule décimale, espace des milliers
pio.templates["lbfi"] = _modele
px.defaults.template = "lbfi"
px.defaults.color_discrete_sequence = PALETTE_LBFI

# Graphiques sans la barre d'outils Plotly (zoom, capture…) qui encombre au survol
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}

# ---------------------------------------------------------------------------
# Formatage des nombres (format français)
# ---------------------------------------------------------------------------
_ESPACE_FINE = " "  # espace fine insécable (séparateur de milliers)


def fr_nombre(valeur, decimales: int = 1) -> str:
    """12066.34 -> "12 066,3" ; NaN -> "–"."""
    if valeur is None or pd.isna(valeur):
        return "–"
    texte = f"{valeur:,.{decimales}f}"
    return texte.replace(",", _ESPACE_FINE).replace(".", ",")


def fr_heures(valeur, decimales: int = 1) -> str:
    """Heures au format français ; au-delà de 1 000 h, sans décimale pour rester lisible."""
    if valeur is not None and pd.notna(valeur) and abs(valeur) >= 1000:
        decimales = 0
    texte = fr_nombre(valeur, decimales)
    return texte if texte == "–" else f"{texte} h"


def fr_pct(ratio, decimales: int = 0, signe: bool = False) -> str:
    """0.113 -> "11 %" (signe=True -> "+11 %")."""
    if ratio is None or pd.isna(ratio):
        return "–"
    texte = fr_nombre(ratio * 100, decimales)
    if signe and ratio > 0:
        texte = "+" + texte
    return f"{texte}{_ESPACE_FINE}%"

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

    # --- Pointages suspects (calculés sur l'ensemble des pointages, pour que les
    # chevauchements soient détectés même entre deux semaines / deux jours) -----
    instant_lecture = maintenant_local()
    df_pointages["motifs_suspects"] = marquer_pointages_suspects(
        df_pointages, instant_lecture.normalize(), instant_lecture
    )

    return df_pointages, df_of, maintenant_local()


def _normalise_id(serie: pd.Series) -> pd.Series:
    """Ramène un identifiant (int, float, texte) à une clé texte comparable."""
    return pd.to_numeric(serie, errors="coerce").astype("Int64").astype(str)


@st.cache_data(ttl=DUREE_CACHE_S, show_spinner=False)
def load_employes() -> pd.DataFrame:
    """Table employes + colonnes "cle" (id normalisé) et "libelle" ("Prénom NOM").

    Renvoie un DataFrame vide si la table est illisible (le dashboard affiche
    alors les identifiants).
    """
    try:
        df_emp = fetch_table(TABLE_EMPLOYES).copy()
    except Exception:
        return pd.DataFrame(columns=["cle", "libelle"])

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
    return df_emp.drop_duplicates("cle")


def build_operateur_labels(ids_operateur: pd.Series) -> pd.Series:
    """Renvoie "Prénom NOM" pour chaque id_operateur (repli : "Opérateur #id")."""
    cles = _normalise_id(ids_operateur)
    repli = ("Opérateur #" + cles).where(cles != "<NA>", "Opérateur inconnu")
    df_emp = load_employes()
    if df_emp.empty:
        st.warning("Noms des opérateurs indisponibles (table employes illisible). Affichage des identifiants.")
        return repli.fillna("Opérateur inconnu")
    noms = cles.map(df_emp.set_index("cle")["libelle"])
    return noms.fillna(repli).fillna("Opérateur inconnu")


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def week_bounds(iso_year: int, iso_week: int) -> tuple[date, date]:
    """Retourne (lundi, dimanche) de la semaine ISO demandée."""
    monday = date.fromisocalendar(int(iso_year), int(iso_week), 1)
    sunday = monday + timedelta(days=6)
    return monday, sunday


JOURS_SEMAINE = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

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


# Le lundi de Pentecôte est souvent travaillé (journée de solidarité) : passer à
# True si l'atelier est fermé ce jour-là.
LUNDI_DE_PENTECOTE_CHOME = False


def _paques(annee: int) -> date:
    """Dimanche de Pâques (algorithme de Meeus / Jones / Butcher)."""
    a, b, c = annee % 19, annee // 100, annee % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mois = (h + l - 7 * m + 114) // 31
    jour = (h + l - 7 * m + 114) % 31 + 1
    return date(annee, mois, jour)


def jours_feries(annee: int) -> list[date]:
    """Jours fériés légaux en France métropolitaine pour une année."""
    paques = _paques(annee)
    feries = [
        date(annee, 1, 1), paques + timedelta(days=1), date(annee, 5, 1), date(annee, 5, 8),
        paques + timedelta(days=39), date(annee, 7, 14), date(annee, 8, 15),
        date(annee, 11, 1), date(annee, 11, 11), date(annee, 12, 25),
    ]
    if LUNDI_DE_PENTECOTE_CHOME:
        feries.append(paques + timedelta(days=50))
    return feries


def est_jour_ouvre(jour: date) -> bool:
    return jour.weekday() < 5 and jour not in jours_feries(jour.year)


def nb_jours_ouvres(premier_jour: date, dernier_jour: date) -> int:
    """Nombre de jours ouvrés (lundi–vendredi hors jours fériés) entre deux dates incluses."""
    if premier_jour > dernier_jour:
        return 0
    feries = [f for a in range(premier_jour.year, dernier_jour.year + 1) for f in jours_feries(a)]
    # np.busday_count exclut la borne de fin -> on l'étend d'un jour
    return int(np.busday_count(premier_jour, dernier_jour + timedelta(days=1), holidays=feries))


TAUX_HORAIRE_JOUR = 7.8  # heures théoriques par opérateur et par jour ouvré travaillé


def maintenant_local() -> pd.Timestamp:
    """Date et heure actuelles à Paris, sans fuseau (comme les heures de pointage)."""
    return pd.Timestamp.now(tz=FUSEAU_HORAIRE).tz_localize(None)


def fraction_journee_ecoulee(instant: pd.Timestamp) -> float:
    """Part (0 à 1) des horaires d'ouverture du jour déjà écoulée à `instant`.

    Lundi–vendredi 5h00–16h30 : à 8h45, 3,75 h sur 11,5 h -> 0,33. Hors jours
    ouvrés (samedi, dimanche), renvoie 0 : ces jours ne comptent pas dans le
    temps théorique.
    """
    if not est_jour_ouvre(instant.date()):
        return 0.0
    ouverture, fermeture = HORAIRES_ATELIER[instant.weekday()]
    heure = instant.hour + instant.minute / 60
    return float(min(max((heure - ouverture) / (fermeture - ouverture), 0.0), 1.0))


def jours_ouvres_theoriques(premier_jour: date, dernier_jour: date,
                            instant: pd.Timestamp | None = None) -> float:
    """Jours ouvrés (lun.–ven.) entre deux dates incluses, limités à ce qui est écoulé.

    Pour une période passée, identique à nb_jours_ouvres. Pour la période en cours,
    les jours à venir ne comptent pas et le jour courant compte au prorata des
    horaires d'ouverture déjà écoulés : mardi à 8h45, une semaine vaut 1 + 0,33 jour
    (et non 5). C'est ce qui évite d'afficher un taux d'attribution de 11 % un
    mardi matin.
    """
    instant = maintenant_local() if instant is None else instant
    aujourdhui = instant.date()
    if premier_jour > dernier_jour or premier_jour > aujourdhui:
        return 0.0
    if dernier_jour < aujourdhui:
        return float(nb_jours_ouvres(premier_jour, dernier_jour))
    jours_passes = nb_jours_ouvres(premier_jour, aujourdhui - timedelta(days=1)) if premier_jour < aujourdhui else 0
    return jours_passes + fraction_journee_ecoulee(instant)


def periode_en_cours(premier_jour: date, dernier_jour: date) -> bool:
    """True si la période contient la date du jour."""
    return premier_jour <= maintenant_local().date() <= dernier_jour


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

    Semaine en cours : seuls les jours ouvrés déjà écoulés comptent (voir
    jours_ouvres_theoriques).
    """
    nb_operateurs = pointages_semaine["id_operateur"].nunique()
    jours_ouvres_semaine = jours_ouvres_theoriques(monday, sunday)
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
            jours_ouvres_theoriques(debut_intersection, fin_intersection)
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
        premier, dernier = month_bounds(row["cal_year"], row["cal_month"])
        labels.append(
            f"{NOMS_MOIS[int(row['cal_month']) - 1]} {int(row['cal_year'])}"
            + (" · en cours" if periode_en_cours(premier, dernier) else "")
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
            + (" · en cours" if periode_en_cours(monday, sunday) else "")
        )
    weeks = weeks.copy()
    weeks["label"] = labels
    return weeks.reset_index(drop=True)


COULEUR_EN_COURS = "#D6E7A3"  # vert pâle : période en cours (incomplète)


def marquer_periode_en_cours_barres(fig: go.Figure, donnees: pd.DataFrame, col_x: str) -> None:
    """Barre de la période en cours en vert pâle, avec la mention « en cours »."""
    fig.update_traces(marker_color=[COULEUR_EN_COURS if ec else COULEUR_REALISE for ec in donnees["en_cours"]])
    for x in donnees.loc[donnees["en_cours"], col_x]:
        fig.add_annotation(x=x, y=1, yref="paper", text="en cours", showarrow=False,
                           font=dict(size=10, color=GRAPHITE_LBFI))


def courbe_taux_avec_periode_en_cours(donnees: pd.DataFrame, col_x: str) -> go.Figure:
    """Courbe du taux d'attribution des périodes terminées ; la période en cours,
    incomplète, est un point creux à part, non relié à la courbe."""
    terminees = donnees.loc[~donnees["en_cours"]]
    en_cours = donnees.loc[donnees["en_cours"]]
    fig = px.line(
        terminees, x=col_x, y="taux_attribution", markers=True,
        labels={col_x: "", "taux_attribution": "Taux d'attribution"},
        category_orders={col_x: donnees[col_x].tolist()},
    )
    if not en_cours.empty:
        fig.add_scatter(
            x=en_cours[col_x], y=en_cours["taux_attribution"], mode="markers+text",
            marker=dict(symbol="circle-open", size=10, color=COULEUR_REALISE, line=dict(width=2)),
            text=["en cours"] * len(en_cours), textposition="top center",
            textfont=dict(size=10, color=GRAPHITE_LBFI), showlegend=False,
            hovertemplate="%{x} (en cours) : %{y:.0%}<extra></extra>",
        )
    fig.update_xaxes(categoryorder="array", categoryarray=donnees[col_x].tolist())
    fig.update_yaxes(tickformat=".0%")
    return fig


# 20 couleurs distinctes : au-delà de 10 dossiers, deux dossiers ne partagent plus la même teinte
PALETTE_ETENDUE = PALETTE_LBFI + [
    "#B5D46A", "#9A9A98", "#7FB8D4", "#F7C59F", "#B9A0B6",
    "#9ED9D7", "#F09A7F", "#6F87A8", "#E3CC7A", "#4F6B00",
]


def couleurs_par_categorie(categories) -> dict:
    """Une couleur fixe par catégorie (ordre d'apparition), réutilisée par plusieurs graphiques."""
    uniques = list(dict.fromkeys(c for c in categories if pd.notna(c)))
    return {c: PALETTE_ETENDUE[i % len(PALETTE_ETENDUE)] for i, c in enumerate(uniques)}


def graphique_repartition(df: pd.DataFrame, col: str, key: str, n: int = 12) -> None:
    """Heures pointées par `col` en barres horizontales triées (top n + « Autres »).

    Remplace les camemberts : des longueurs se comparent bien mieux que des angles,
    et les libellés restent lisibles même avec beaucoup de catégories.
    """
    data = pie_top_n(df, col, "Durée h", n=n)
    total = data["Durée h"].sum()
    data = data.iloc[::-1]  # Plotly trace de bas en haut : la plus grosse catégorie en haut
    textes = [f"{fr_heures(v)} ({fr_pct(v / total) if total else '–'})" for v in data["Durée h"]]
    fig = go.Figure(go.Bar(
        y=data[col].astype(str), x=data["Durée h"], orientation="h", text=textes,
        textposition="outside", cliponaxis=False,
        marker_color=[COULEUR_DEVISE if c == "Autres" else COULEUR_REALISE for c in data[col]],
        hovertemplate="%{y} : %{x:.1f} h<extra></extra>",
    ))
    fig.update_layout(height=max(200, 60 + 28 * len(data)), margin=dict(l=10, r=100, t=10, b=10),
                      xaxis_title="Heures pointées", yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)


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
# Pointages suspects (onglets « Suivi journalier » et « Suivi hebdomadaire »)
# ---------------------------------------------------------------------------

# Horaires d'ouverture de l'atelier : jour (0 = lundi … 6 = dimanche) ->
# (ouverture, fermeture) en heures décimales. Jour absent = atelier fermé.
HORAIRES_ATELIER = {
    0: (5.0, 16.5),   # lundi    5h00 – 16h30
    1: (5.0, 16.5),   # mardi
    2: (5.0, 16.5),   # mercredi
    3: (5.0, 16.5),   # jeudi
    4: (5.0, 16.5),   # vendredi
    5: (6.0, 12.0),   # samedi   6h00 – 12h00
}                     # dimanche : fermé
HORAIRES_ATELIER_TEXTE = "lun.–ven. 5h00–16h30, sam. 6h00–12h00, fermé le dimanche"
TOLERANCE_HORAIRES_MIN = 0           # marge tolérée autour des horaires d'ouverture (minutes)
SEUIL_POINTAGE_TROP_LONG_H = 12      # un pointage de plus de 12 h est suspect

# Opérateurs qui conduisent plusieurs machines en parallèle (machines numériques qui
# tournent seules) : pour eux, deux opérations qui se chevauchent sont normales et ne
# sont pas signalées. Noms comparés sans tenir compte des accents, majuscules et tirets.
OPERATEURS_TRAVAIL_EN_PARALLELE = ["Ana-Paula LARREY"]


def _nom_normalise(nom) -> str:
    """"Ana-Paula LARREY" -> "ana paula larrey" (sans accents, tirets ni majuscules)."""
    import unicodedata
    texte = unicodedata.normalize("NFKD", str(nom)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]+", " ", texte.lower()).strip()


_NOMS_TRAVAIL_EN_PARALLELE = [_nom_normalise(n) for n in OPERATEURS_TRAVAIL_EN_PARALLELE]


def travaille_en_parallele(libelle_operateur) -> bool:
    """True si l'opérateur fait partie de OPERATEURS_TRAVAIL_EN_PARALLELE."""
    nom = _nom_normalise(libelle_operateur)
    return any(n and n in nom for n in _NOMS_TRAVAIL_EN_PARALLELE)


TOLERANCE_FUTUR_MIN = 5  # marge (minutes) avant de considérer qu'une heure est « dans le futur »


def marquer_pointages_suspects(df: pd.DataFrame, aujourdhui: pd.Timestamp,
                               maintenant: pd.Timestamp | None = None) -> pd.Series:
    """Renvoie, pour chaque pointage, la liste des motifs de suspicion (texte, "" si aucun).

    Un pointage est suspect s'il :
      - dure plus de SEUIL_POINTAGE_TROP_LONG_H heures ;
      - finit avant d'avoir commencé (durée négative) ;
      - commence ou finit dans le futur (heure saisie à l'avance ou erreur de saisie) ;
      - s'étale sur plusieurs jours (ou n'est toujours pas clôturé un jour suivant) ;
      - chevauche un autre pointage du même opérateur (sauf OPERATEURS_TRAVAIL_EN_PARALLELE) ;
      - a lieu, même en partie, en dehors des horaires d'ouverture de l'atelier.
    """
    debut = df["heure_debut"]
    fin = df["heure_fin"]
    # Fin "effective" : heure de fin, sinon début + durée enregistrée (si > 0)
    duree_ms = df["duree"].where(df["duree"] > 0)
    fin_eff = fin.fillna(debut + pd.to_timedelta(duree_ms, unit="ms"))
    jour_debut = debut.dt.normalize()

    regles = []

    # 1. Pointage trop long
    duree_h = pd.concat(
        [df["Durée h"], (fin_eff - debut).dt.total_seconds() / 3600], axis=1
    ).max(axis=1)
    regles.append((duree_h > SEUIL_POINTAGE_TROP_LONG_H, f"Durée > {SEUIL_POINTAGE_TROP_LONG_H} h"))

    # 1 bis. Fin avant le début : durée négative, qui retire des heures aux totaux
    regles.append((fin_eff.notna() & (fin_eff < debut), "Fin avant le début (durée négative)"))

    # 1 ter. Heure dans le futur : saisie anticipée ou erreur, qui gonfle les heures du jour
    maintenant = maintenant_local() if maintenant is None else maintenant
    limite_futur = maintenant + pd.Timedelta(minutes=TOLERANCE_FUTUR_MIN)
    regles.append((debut > limite_futur, "Début dans le futur"))
    regles.append(((debut <= limite_futur) & fin_eff.notna() & (fin_eff > limite_futur), "Fin dans le futur"))

    # 2. Pointage sur plusieurs jours
    regles.append((fin_eff.notna() & (fin_eff.dt.normalize() > jour_debut), "S'étale sur plusieurs jours"))
    regles.append((fin_eff.isna() & (jour_debut < aujourdhui),
                   "Non clôturé depuis un jour précédent"))

    # 3. Chevauchement avec un autre pointage du même opérateur : on nomme le ou
    # les pointages en conflit pour que l'utilisateur puisse vérifier.
    texte_chevauche = _textes_chevauchements(df, fin_eff)
    regles.append((texte_chevauche != "", texte_chevauche))

    # 4. En dehors des horaires d'ouverture de l'atelier
    jour_semaine = debut.dt.weekday
    tolerance = pd.Timedelta(minutes=TOLERANCE_HORAIRES_MIN)
    ouverture = jour_debut + pd.to_timedelta(
        jour_semaine.map({j: h[0] for j, h in HORAIRES_ATELIER.items()}), unit="h") - tolerance
    fermeture = jour_debut + pd.to_timedelta(
        jour_semaine.map({j: h[1] for j, h in HORAIRES_ATELIER.items()}), unit="h") + tolerance
    atelier_ferme = debut.notna() & ouverture.isna()
    regles.append((atelier_ferme, "Atelier fermé ce jour-là"))
    regles.append((~atelier_ferme & (debut < ouverture), "Début avant l'ouverture"))
    regles.append((~atelier_ferme & (debut >= fermeture), "Début après la fermeture"))
    regles.append((~atelier_ferme & (debut < fermeture) & (fin_eff > fermeture), "Fin après la fermeture"))

    motifs = pd.Series("", index=df.index, dtype=object)
    for masque, texte in regles:
        m = masque.fillna(False).astype(bool)
        if isinstance(texte, pd.Series):  # texte propre à chaque pointage
            texte = texte[m]
        motifs[m] = np.where(motifs[m] == "", texte, motifs[m] + " · " + texte)
    return motifs


MOTIF_CHEVAUCHEMENT = "Chevauche"  # début commun des motifs de chevauchement


def _textes_chevauchements(df: pd.DataFrame, fin_eff: pd.Series) -> pd.Series:
    """Pour chaque pointage, "Chevauche n°7529 (09:00–10:30)" ou "" s'il n'y a pas de conflit.

    Balayage chronologique par opérateur : on garde les pointages encore « ouverts »
    (fin > début du pointage examiné) ; chacun d'eux est en conflit avec lui.
    Les opérateurs de OPERATEURS_TRAVAIL_EN_PARALLELE ne sont pas contrôlés.
    """
    partenaires: dict = {i: [] for i in df.index}
    valides = df["heure_debut"].notna() & df["id_operateur"].notna()
    if "operateur" in df.columns:
        valides &= ~df["operateur"].map(travaille_en_parallele).astype(bool)
    donnees = df.loc[valides, ["id_operateur", "heure_debut"]].assign(_fin=fin_eff[valides])
    for _, groupe in donnees.sort_values("heure_debut").groupby("id_operateur", sort=False):
        actifs = []  # (fin, index)
        for idx, debut, fin in zip(groupe.index, groupe["heure_debut"], groupe["_fin"]):
            actifs = [(f, j) for f, j in actifs if f > debut]
            for _, j in actifs:
                partenaires[idx].append(j)
                partenaires[j].append(idx)
            if pd.notna(fin):
                actifs.append((fin, idx))

    def libelle(j, jour_ref) -> str:
        debut, fin = df.at[j, "heure_debut"], fin_eff.at[j]
        fmt = "%H:%M" if debut.date() == jour_ref else "%d/%m %H:%M"
        fin_txt = fin.strftime("%H:%M") if pd.notna(fin) else "…"
        return f"n°{df.at[j, 'id']} ({debut.strftime(fmt)}–{fin_txt})"

    textes = pd.Series("", index=df.index, dtype=object)
    for idx, liste in partenaires.items():
        if liste:
            jour_ref = df.at[idx, "heure_debut"].date()
            noms = ", ".join(libelle(j, jour_ref) for j in sorted(set(liste), key=lambda j: df.at[j, "heure_debut"]))
            textes.at[idx] = f"{MOTIF_CHEVAUCHEMENT} {noms}"
    return textes


# ---------------------------------------------------------------------------
# Affichage des tableaux : AgGrid (tri + filtre directement dans les en-têtes)
# ---------------------------------------------------------------------------

# Colonnes numériques à filtrer comme du texte (identifiants, numéros…)
_COLONNES_IDENTIFIANTS = re.compile(r"(numero|ordre_fabrication|reference|dossier)", re.IGNORECASE)
# Ligne de filtres affichée sous les en-têtes (en plus de l'icône de filtre de chaque colonne)
LIGNE_DE_FILTRES = True
SEUIL_LIGNES_FILTRES = 15  # en dessous, pas de ligne de filtres (tableau court, lisible d'un coup d'œil)

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
    """Nombre au format français dans les tableaux : 1 234,56."""
    return JsCode(f"""
function(p) {{
    if (p.value === null || p.value === undefined || isNaN(p.value)) return '–';
    return Number(p.value).toLocaleString('fr-FR', {{minimumFractionDigits: {decimales},
                                                    maximumFractionDigits: {decimales}}}) + '{suffixe}';
}}""")


# Écart en % avec flèche et signe : « ▲ +82 % » (dépassement), « ▼ −71 % » (sous le devis)
_JS_FORMAT_ECART_PCT = JsCode("""
function(p) {
    if (p.value === null || p.value === undefined || isNaN(p.value)) return '–';
    const v = Number(p.value);
    const txt = Math.abs(v).toLocaleString('fr-FR', {maximumFractionDigits: 0}) + '\u202f%';
    if (v > 0) return '▲ +' + txt;
    if (v < 0) return '▼ −' + txt;
    return '= 0\u202f%';
}""")


# Libellés affichés dans les en-têtes des tableaux. Les noms techniques restent
# les noms de champ (utilisés par le code, ex. la ligne sélectionnée) ; seul
# l'en-tête affiché change.
LIBELLES_COLONNES = {
    "id": "N°",
    "created_at": "Créé le",
    "operateur": "Opérateur",
    "operation": "Opération",
    "ordre_fabrication": "Dossier",
    "numero_dossier": "Dossier",
    "numero_devis": "Devis",
    "client": "Client",
    "reference": "Référence",
    "heure_debut": "Début",
    "heure_fin": "Fin",
    "Durée h": "Durée (h)",
    "temps_devis (h)": "Devisé (h)",
    "temps_operateurs (h)": "Réalisé (h)",
    "delta (h)": "Écart (h)",
    "ratio (opérateurs-devis)/devis (%)": "Écart (%)",
    "statut_production": "Statut",
}


def afficher_tableau(df: pd.DataFrame, key: str, style_ligne=None, hauteur_max: int = 450,
                     selection: bool = False):
    """Affiche un tableau triable et filtrable depuis les en-têtes de colonnes.

    - Texte / identifiants : filtre « contient », « commence par »…
    - Nombres (durées, écarts) : filtre « supérieur à », « entre »…
    - Dates : filtre « avant », « après », « entre »…
    `style_ligne` : fonction (row -> liste de styles CSS) reprise des anciennes
    mises en couleur ; seule la couleur de fond est conservée.
    `selection=True` : un clic sélectionne une ligne ; la fonction renvoie alors
    la ligne sélectionnée (dict) ou None.
    """
    df = df.copy()

    if style_ligne is not None:
        df["_couleur"] = df.apply(
            lambda r: (style_ligne(r) or [""])[0].replace("background-color:", "").strip() or None,
            axis=1,
        )

    # Ligne de filtres seulement si le tableau est assez long pour qu'on ait besoin de filtrer
    # (le filtre reste accessible par l'icône de chaque en-tête)
    ligne_de_filtres = LIGNE_DE_FILTRES and len(df) > SEUIL_LIGNES_FILTRES
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        sortable=True, resizable=True, filter=True,
        floatingFilter=ligne_de_filtres, suppressHeaderMenuButton=False,
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
            if col == "ratio (opérateurs-devis)/devis (%)":
                fmt = _JS_FORMAT_ECART_PCT
            elif "%" in col:
                fmt = _js_format_nombre(0, "\u202f%")
            elif pd.api.types.is_integer_dtype(serie):
                fmt = _js_format_nombre(0)
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
            extra = {"minWidth": 320, "wrapText": True, "autoHeight": True} if col == "Alertes" else {}
            gb.configure_column(col, filter="agTextColumnFilter", filterParams={"buttons": ["reset"]}, **extra)

    # En-têtes lisibles (après la configuration ci-dessus : chaque appel à
    # configure_column réécrit l'en-tête s'il n'est pas fourni)
    for col in df.columns:
        if col != "_couleur":
            libelle = LIBELLES_COLONNES.get(col, col)
            gb.configure_column(col, header_name=libelle, headerTooltip=libelle)

    gb.configure_grid_options(
        localeText=_TRADUCTION_AGGRID,
        getRowStyle=_JS_STYLE_LIGNE,
        autoSizeStrategy={"type": "fitGridWidth", "defaultMinWidth": 90},
        enableCellTextSelection=True,
        animateRows=False,
    )

    if selection:
        gb.configure_selection(selection_mode="single", use_checkbox=False)

    entete = 2 * 42 if ligne_de_filtres else 42
    hauteur = min(hauteur_max, entete + 30 * max(len(df), 1) + 12)

    reponse = AgGrid(
        df,
        gridOptions=gb.build(),
        height=hauteur,
        theme="streamlit",
        allow_unsafe_jscode=True,
        # filtrer / trier ne relance pas le script ; sélectionner une ligne, si
        # la sélection est activée, oui (pour afficher le bouton de détail)
        update_on=["selectionChanged"] if selection else [],
        custom_css={
            ".ag-row-selected": {"outline": "2px solid #0068C9 !important",
                                 "outline-offset": "-2px", "font-weight": "600",
                                 "--ag-selected-row-background-color": "transparent"},
            ".ag-row-selected::before": {"background-color": "transparent !important"},
            ".ag-row": {"cursor": "pointer" if selection else "default"},
        },
        show_toolbar=False,
        show_search=False,
        show_download_button=False,
        key=key,
    )
    if not selection:
        return None
    lignes = reponse.selected_rows
    if lignes is None or len(lignes) == 0:
        return None
    if isinstance(lignes, pd.DataFrame):
        ligne = lignes.iloc[0].to_dict()
    else:
        ligne = dict(lignes[0])
    ligne.pop("_couleur", None)
    return ligne


CRITERES_SUSPECTS_TEXTE = (
    f"Critères : pointage de plus de {SEUIL_POINTAGE_TROP_LONG_H} h, fin avant le début (durée négative), "
    "début ou fin dans le futur (heure saisie à l'avance), "
    "pointage qui s'étale sur "
    "plusieurs jours (ou toujours ouvert un jour suivant), chevauchement avec un autre pointage "
    f"du même opérateur (sauf {', '.join(OPERATEURS_TRAVAIL_EN_PARALLELE)}, qui conduit plusieurs machines "
    "en parallèle), ou pointage (même en partie) en dehors des horaires d'ouverture de "
    f"l'atelier ({HORAIRES_ATELIER_TEXTE})."
)


def alerte_pointages_suspects(pointages: pd.DataFrame, periode: str, ancre: str) -> None:
    """Bandeau d'alerte en haut de page, avec lien vers la liste en bas de page."""
    nb = int((pointages["motifs_suspects"] != "").sum()) if not pointages.empty else 0
    if nb:
        pluriel = "s" if nb > 1 else ""
        st.error(
            f"⚠️ Attention, {periode} comporte **{nb} pointage{pluriel} suspect{pluriel}**. "
            f"[⇒ Voir les pointages suspects](#{ancre})"
        )


def section_pointages_suspects(pointages: pd.DataFrame, periode: str, ancre: str, key: str) -> None:
    """Liste des pointages suspects, affichée en bas de page."""
    st.divider()
    st.subheader("🚨 Pointages suspects", anchor=ancre)
    st.caption(CRITERES_SUSPECTS_TEXTE)

    suspects = pointages.loc[pointages["motifs_suspects"] != ""] if not pointages.empty else pointages
    if suspects.empty:
        st.success(f"Aucun pointage suspect sur {periode}.")
        return

    # Nombre de pointages par motif
    nb_par_motif = (
        suspects["motifs_suspects"].str.split(" · ").explode()
        .where(lambda m: ~m.str.startswith(MOTIF_CHEVAUCHEMENT), "Chevauche un autre pointage")
        .value_counts()
    )
    st.markdown(" · ".join(f"**{n}** {motif.lower()}" for motif, n in nb_par_motif.items()))

    table = suspects[
        ["id", "operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin",
         "Durée h", "motifs_suspects"]
    ].rename(columns={"motifs_suspects": "Alertes"}).sort_values(["operateur", "heure_debut"])
    table = table.reset_index(drop=True)
    table["Durée h"] = table["Durée h"].round(2)
    afficher_tableau(table, key=key,
                     style_ligne=lambda r: ["background-color: #f8d7da"] * len(r))


# ---------------------------------------------------------------------------
# Détail des pointages d'une opération (fenêtre ouverte depuis
# « Analyse d'un dossier spécifique »)
# ---------------------------------------------------------------------------

# Les pointages suspects sont détectés par marquer_pointages_suspects (une seule règle
# pour toute l'application, calculée au chargement dans la colonne "motifs_suspects").


def _cle_texte(serie: pd.Series) -> pd.Series:
    """Clé de comparaison souple : texte, sans espaces superflus, sans ".0", en minuscules."""
    return (
        serie.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True).str.lower()
    )


@st.dialog("🔎 Détail des pointages", width="large")
def fenetre_detail_pointages(numero_dossier: str, operation: str, devis_h: float, realise_h: float) -> None:
    st.markdown(f"**Dossier {numero_dossier}** — opération **{operation}**")

    du_dossier = df_pointages[_cle_texte(df_pointages["ordre_fabrication"]) == _cle_texte(pd.Series([numero_dossier])).iloc[0]]
    pts = du_dossier[_cle_texte(du_dossier["operation"]) == _cle_texte(pd.Series([operation])).iloc[0]].copy()

    if pts.empty:
        st.warning("Aucun pointage trouvé pour cette opération dans « temps_reel_operateur ».")
        if not du_dossier.empty:
            autres = (du_dossier.groupby("operation")["Durée h"].sum().sort_values(ascending=False).round(2)
                      .rename("Heures pointées").reset_index())
            st.markdown("Opérations pointées sur ce dossier (le libellé diffère peut-être légèrement) :")
            st.dataframe(autres, hide_index=True, use_container_width=True)
        return

    somme_pts = pts["Durée h"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Devisé", fr_heures(devis_h, 2))
    c2.metric("Réalisé (OF)", fr_heures(realise_h, 2),
              help="Temps inscrit sur l'ordre de fabrication (table ordres_fabrication), affiché dans le tableau.")
    c3.metric("Pointé", fr_heures(somme_pts, 2),
              help="Somme des pointages des opérateurs (table temps_reel_operateur) pour ce dossier et cette opération.")
    c4.metric("Pointages / op.", f"{len(pts)} / {pts['id_operateur'].nunique()}")

    if pd.notna(realise_h) and abs(somme_pts - realise_h) > 0.05:
        st.info(
            f"La somme des pointages ({fr_heures(somme_pts, 2)}) diffère du temps réalisé inscrit sur l'OF "
            f"({fr_heures(realise_h, 2)}) : l'OF n'est peut-être pas à jour, ou des pointages portent un "
            "libellé d'opération différent."
        )

    # Même règle que la section « Pointages suspects » des onglets journalier / hebdomadaire
    pts["Alertes"] = pts["motifs_suspects"]
    est_suspect = pts["Alertes"] != ""
    nb_suspects = int(est_suspect.sum())
    if nb_suspects:
        heures_suspectes = pts.loc[est_suspect, "Durée h"].sum()
        part = fr_pct(heures_suspectes / somme_pts) if somme_pts else "–"
        st.error(f"⚠️ {nb_suspects} pointage(s) suspect(s), soit {fr_heures(heures_suspectes, 2)} "
                 f"({part} du total pointé). Ils sont surlignés en rouge ci-dessous.")
        st.caption(CRITERES_SUSPECTS_TEXTE)
    else:
        st.success("Aucune anomalie détectée : les heures semblent correspondre à du temps réel de production.")

    # Heures par jour et par opérateur : un pic isolé saute aux yeux
    st.markdown("**Heures pointées par jour**")
    # Axe en catégories (un jour = une barre) : avec un axe de dates, un seul jour
    # pointé donnait une barre étalée sur plusieurs jours et des dates répétées.
    par_jour = (pts.assign(jour=pts["heure_debut"].dt.normalize())
                .groupby(["jour", "operateur"], as_index=False)["Durée h"].sum()
                .sort_values("jour"))
    par_jour["jour_txt"] = par_jour["jour"].dt.strftime("%d/%m/%Y")
    fig = px.bar(par_jour, x="jour_txt", y="Durée h", color="operateur",
                 category_orders={"jour_txt": list(dict.fromkeys(par_jour["jour_txt"]))},
                 labels={"jour_txt": "", "Durée h": "Heures", "operateur": "Opérateur"})
    fig.update_xaxes(type="category")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), bargap=0.3)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"bar_detail_{numero_dossier}_{operation}")

    st.markdown("**Pointages**")
    table = pts.sort_values("heure_debut")[
        ["id", "operateur", "heure_debut", "heure_fin", "Durée h", "Alertes"]
    ].reset_index(drop=True)
    table["Durée h"] = table["Durée h"].round(2)
    afficher_tableau(
        table, key=f"grille_detail_{numero_dossier}_{operation}",
        style_ligne=lambda r: ["background-color: #f8d7da" if r["Alertes"] else ""] * len(r),
    )

    st.markdown("**Total par opérateur**")
    par_op = (pts.groupby("operateur")
              .agg(heures=("Durée h", "sum"), pointages=("id", "count"),
                   premier=("heure_debut", "min"), dernier=("heure_debut", "max"))
              .sort_values("heures", ascending=False).reset_index())
    par_op["heures"] = par_op["heures"].map(lambda h: fr_nombre(h, 2))
    par_op["premier"] = par_op["premier"].dt.strftime("%d/%m/%Y")
    par_op["dernier"] = par_op["dernier"].dt.strftime("%d/%m/%Y")
    st.dataframe(
        par_op.rename(columns={"operateur": "Opérateur", "heures": "Heures", "pointages": "Pointages",
                               "premier": "Premier jour", "dernier": "Dernier jour"}),
        hide_index=True, use_container_width=True,
    )


def bouton_detail_pointages(ligne: dict | None, numero_dossier: str | None, key: str) -> None:
    """Sous un tableau d'opérations : ouvre le détail des pointages de la ligne cliquée.

    La fenêtre s'ouvre dès qu'on clique sur une nouvelle ligne ; le bouton permet de
    la rouvrir pour la même ligne après l'avoir fermée.
    `numero_dossier=None` : le numéro est lu dans la colonne "numero_dossier" de la ligne.
    """
    if ligne is None:
        st.caption("🔎 Cliquez sur une ligne du tableau pour afficher le détail de ses pointages.")
        return
    operation = ligne.get("operation")
    if numero_dossier is None:
        numero_dossier = ligne.get("numero_dossier")
        libelle = f"dossier {numero_dossier} · {operation}"
    else:
        libelle = str(operation)

    def _ouvrir() -> None:
        fenetre_detail_pointages(
            str(numero_dossier), str(operation),
            pd.to_numeric(ligne.get("temps_devis (h)"), errors="coerce"),
            pd.to_numeric(ligne.get("temps_operateurs (h)"), errors="coerce"),
        )

    # Ouverture automatique, une seule fois par nouvelle ligne cliquée
    cle_ouverte = f"_detail_ouvert_{key}"
    identifiant = f"{numero_dossier}|{operation}"
    nouvelle_ligne = st.session_state.get(cle_ouverte) != identifiant
    st.session_state[cle_ouverte] = identifiant
    if st.button(f"🔎 Revoir le détail des pointages — {libelle}", key=key) or nouvelle_ligne:
        _ouvrir()


# ---------------------------------------------------------------------------
# Interface — chargement des données
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Identité visuelle LBFI (logo, bandeaux photo, couleurs)
# ---------------------------------------------------------------------------
DOSSIER_ASSETS = Path(__file__).parent / "assets"
VERT_LBFI = "#94C000"        # vert du logo
VERT_LBFI_FONCE = "#5E7C00"  # vert lisible pour du texte / boutons
GRAPHITE_LBFI = "#575756"    # gris du logo


@st.cache_data(show_spinner=False)
def _image_base64(nom: str) -> str | None:
    chemin = DOSSIER_ASSETS / nom
    if not chemin.exists():
        return None
    mime = "image/png" if chemin.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(chemin.read_bytes()).decode()


def injecter_style() -> None:
    st.markdown(f"""
<style>
  /* Pas de barre latérale : tous les réglages sont dans les pages */
  [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] {{ display: none !important; }}
  /* La barre d'outils Streamlit (Fork, GitHub, ⋮) fait 60 px de haut : on laisse la place */
  .block-container {{ padding-top: 4.4rem; }}

  /* En-tête compact */
  .lbfi-entete {{ display: flex; align-items: center; gap: 14px; }}
  .lbfi-entete img {{ height: 34px; display: block; }}
  .lbfi-entete .lbfi-titre {{ font-size: 1.25rem; font-weight: 700; color: #2B2B2A; line-height: 1.2; }}
  .lbfi-entete .lbfi-trait {{ width: 4px; align-self: stretch; background: {VERT_LBFI}; border-radius: 2px; }}
  .lbfi-fraicheur {{ text-align: right; color: {GRAPHITE_LBFI}; font-size: .85rem; padding-top: .55rem; }}

  /* Menu des pages, présenté comme des onglets */
  /* (Streamlit met déjà la page courante en gras sur fond grisé) */
  [data-testid="stPageLink-NavLink"] {{ justify-content: center; border-radius: 8px; }}
  .st-key-menu_pages {{ border-bottom: 1px solid #E4E5E0; padding-bottom: .3rem; margin-bottom: .6rem; }}

  /* Titre de page */
  .lbfi-page h2 {{ margin: .2rem 0 0 0; padding: 0; font-size: 1.55rem; }}
  .lbfi-page p {{ color: {GRAPHITE_LBFI}; margin: .1rem 0 .6rem 0; font-size: .92rem; }}

  /* Indicateurs en cartes */
  [data-testid="stMetric"] {{
    background: #fff; border: 1px solid #E4E5E0; border-left: 4px solid {VERT_LBFI};
    border-radius: 10px; padding: 12px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }}
  [data-testid="stMetricLabel"] p {{ color: {GRAPHITE_LBFI}; font-weight: 600; }}
  /* Libellés et valeurs jamais tronqués (« Opérateurs ayant poi… », « 12066… ») :
     le libellé passe sur deux lignes si besoin. */
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
  }}
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
    overflow: visible !important; text-overflow: clip !important;
  }}

  /* Titres de section */
  h3 {{ color: #2B2B2A; }}

  /* Ancres "Pointages suspects" : laisser un peu d'air au-dessus après le saut */
  [id^="pointages-suspects"] {{ scroll-margin-top: 4rem; }}
</style>
""", unsafe_allow_html=True)


def en_tete_compacte() -> None:
    """Une seule bande : logo, titre, heure des données, Actualiser, puis le menu des pages."""
    logo = _image_base64("logo_lbfi.png")
    logo_html = f'<img src="{logo}" alt="LBFI International">' if logo else ""
    col_titre, col_fraicheur, col_lexique, col_bouton = st.columns([5.2, 3, 1.6, 1.5], vertical_alignment="center")
    col_titre.markdown(
        f'<div class="lbfi-entete">{logo_html}<div class="lbfi-trait"></div>'
        '<div class="lbfi-titre">Suivi des temps de production</div></div>',
        unsafe_allow_html=True,
    )
    col_fraicheur.markdown(
        f'<div class="lbfi-fraicheur">Données lues le {DONNEES_LUES_LE.strftime("%d/%m à %H:%M")}'
        f" · relues toutes les {DUREE_CACHE_S // 60} min</div>",
        unsafe_allow_html=True,
    )
    with col_lexique.popover("📖 Lexique", use_container_width=True):
        st.markdown(LEXIQUE)
    if col_bouton.button("↻ Actualiser", help="Relit immédiatement les données dans Supabase.",
                         use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    try:
        menu = st.container(key="menu_pages")
    except TypeError:  # Streamlit < 1.42 : pas de clé de conteneur (le menu reste fonctionnel)
        menu = st.container()
    with menu:
        colonnes = st.columns(len(PAGES))
        badges = _badges_suspects()
        for colonne, page in zip(colonnes, PAGES):
            nb = badges.get(page.url_path, 0)
            libelle = f"{page.title} · {nb} ⚠" if nb else page.title
            colonne.page_link(page, label=libelle, use_container_width=True,
                              help=f"{nb} pointage(s) suspect(s) sur la période affichée" if nb else None)


LEXIQUE = """
**Heures pointées** — somme des pointages des opérateurs (début → fin), quel que soit le dossier.

**Temps théorique** — opérateurs ayant pointé dans la semaine × jours ouvrés (hors fériés ;
jours écoulés seulement pour la période en cours) × 7,8 h.

**Taux d'attribution** — heures pointées ÷ temps théorique.

**Heures devisées** — temps prévu au devis pour une opération ou un dossier.

**Heures réalisées** — temps inscrit sur l'ordre de fabrication (OF) une fois l'opération faite.

**Écart** — réalisé − devisé, en heures ou en % du devisé. Positif = dépassement.

**Tolérance** — écart (en %) en deçà duquel une opération est considérée conforme (⚙ au-dessus des tableaux).

**Dossier clôturé** — toutes ses opérations sont au statut « Clos ».

**Pointage suspect** — plus de 12 h, fin avant le début, heure dans le futur, sur plusieurs jours, chevauchement
avec un autre pointage du même opérateur, ou hors horaires d'ouverture.
"""


def _badges_suspects() -> dict:
    """Nombre de pointages suspects de la journée et de la semaine affichées (pour le menu)."""
    suspects = df_pointages.loc[df_pointages["motifs_suspects"] != ""]
    if suspects.empty:
        return {}
    aujourdhui = maintenant_local().date()
    jour = st.session_state.get("periode_jour")
    if jour is None:
        try:
            jour = date.fromisoformat(st.query_params.get("jour", ""))
        except ValueError:
            jour = aujourdhui
    semaine = st.session_state.get("periode_semaine") or st.query_params.get("semaine", "")
    if not re.fullmatch(r"\d{4}-S\d{2}", semaine):
        semaine = id_semaine(*aujourdhui.isocalendar()[:2])
    annee, num = (int(x) for x in semaine.split("-S"))
    return {
        "": int((suspects["heure_debut"].dt.date == jour).sum()),  # page Journée (racine du site)
        "semaine": int(((suspects["iso_year"] == annee) & (suspects["iso_week"] == num)).sum()),
    }


def titre_page(titre: str, sous_titre: str = "") -> None:
    """Titre de page sobre (remplace les bandeaux photo)."""
    st.markdown(
        f'<div class="lbfi-page"><h2>{titre}</h2>{f"<p>{sous_titre}</p>" if sous_titre else ""}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sélecteur de période commun aux pages Semaine / Mois / Année
# ---------------------------------------------------------------------------
# La période choisie est gardée en session (elle survit au changement de page)
# et recopiée dans l'URL (?semaine=2026-S44) pour pouvoir partager le lien.

NOMS_MOIS_COURTS = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
                    "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def _valeur_initiale(nom: str, options: list, defaut):
    depuis_url = st.query_params.get(nom)
    if depuis_url in options:
        return depuis_url
    return defaut if defaut in options else options[-1]


def selecteur_periode(nom: str, options_croissantes: list, format_option, valeur_actuelle,
                      libelle_actuel: str, libelles: tuple = ("◀ Précédent", "Suivant ▶")):
    """◀ · liste · ▶ · bouton « maintenant », comme sur la page Journée.

    `options_croissantes` : identifiants texte du plus ancien au plus récent
    (ex. "2026-S44"). Renvoie l'identifiant choisi.
    """
    cle, cle_widget = f"periode_{nom}", f"_widget_{nom}"
    if st.session_state.get(cle) not in options_croissantes:
        st.session_state[cle] = _valeur_initiale(nom, options_croissantes, valeur_actuelle)
    st.session_state[cle_widget] = st.session_state[cle]
    position = options_croissantes.index(st.session_state[cle])

    def _decaler(pas: int) -> None:
        i = options_croissantes.index(st.session_state[cle]) + pas
        if 0 <= i < len(options_croissantes):
            st.session_state[cle] = options_croissantes[i]

    def _choisir() -> None:
        st.session_state[cle] = st.session_state[cle_widget]

    def _revenir_a_maintenant() -> None:
        st.session_state[cle] = valeur_actuelle if valeur_actuelle in options_croissantes else options_croissantes[-1]

    c1, c2, c3, c4 = st.columns([1, 2.4, 1, 1])
    c1.button(libelles[0], key=f"prec_{nom}", on_click=_decaler, args=(-1,),
              disabled=position == 0, use_container_width=True)
    c2.selectbox(nom, options=options_croissantes[::-1], format_func=format_option, key=cle_widget,
                 on_change=_choisir, label_visibility="collapsed")
    c3.button(libelles[1], key=f"suiv_{nom}", on_click=_decaler, args=(1,),
              disabled=position == len(options_croissantes) - 1, use_container_width=True)
    c4.button(libelle_actuel, key=f"maintenant_{nom}", on_click=_revenir_a_maintenant,
              disabled=st.session_state[cle] == valeur_actuelle, use_container_width=True)

    st.query_params[nom] = st.session_state[cle]
    return st.session_state[cle]


def id_semaine(iso_year: int, iso_week: int) -> str:
    return f"{int(iso_year)}-S{int(iso_week):02d}"


def libelle_semaine(identifiant: str) -> str:
    annee, semaine = identifiant.split("-S")
    lundi, dimanche = week_bounds(int(annee), int(semaine))
    if lundi.month == dimanche.month:
        plage = f"{lundi.day}–{dimanche.day} {NOMS_MOIS_COURTS[dimanche.month - 1]} {dimanche.year}"
    else:
        plage = (f"{lundi.day} {NOMS_MOIS_COURTS[lundi.month - 1]} – "
                 f"{dimanche.day} {NOMS_MOIS_COURTS[dimanche.month - 1]} {dimanche.year}")
    return f"S{int(semaine):02d} · {plage}" + (" · en cours" if periode_en_cours(lundi, dimanche) else "")


def id_mois(annee: int, mois: int) -> str:
    return f"{int(annee)}-{int(mois):02d}"


def libelle_mois(identifiant: str) -> str:
    annee, mois = (int(x) for x in identifiant.split("-"))
    return f"{NOMS_MOIS[mois - 1]} {annee}" + (" · en cours" if periode_en_cours(*month_bounds(annee, mois)) else "")


# ---------------------------------------------------------------------------
# Seuil de tolérance de l'écart devisé / réalisé
# ---------------------------------------------------------------------------
TOLERANCE_PAR_DEFAUT = 20
COULEUR_LIGNE_NON_POINTEE = "#e2e3e5"
COULEUR_LIGNE_DEPASSEMENT = "#f8d7da"
COULEUR_LIGNE_SOUS_DEVIS = "#d6e9f8"   # bleu : sous le devis (l'orange se lisait comme un avertissement)
COULEUR_LIGNE_TOLERANCE = "#d4edda"


def reglage_tolerance(cle: str) -> int:
    """Bouton ⚙ qui affiche la valeur active et ouvre le curseur (st.popover)."""
    seuil_tolerance_actuel()  # initialise la valeur (session ou URL)
    cle_widget = f"_widget_tolerance_{cle}"
    st.session_state[cle_widget] = st.session_state["tolerance_ecart"]

    def _maj() -> None:
        st.session_state["tolerance_ecart"] = st.session_state[cle_widget]

    with st.popover(f"⚙ Tolérance ±{st.session_state['tolerance_ecart']} %", use_container_width=True):
        st.slider(
            "Seuil de tolérance de l'écart (%)", min_value=5, max_value=100, step=5,
            key=cle_widget, on_change=_maj,
            help="Écart = (réalisé − devisé) ÷ devisé. Au-delà de ce seuil (en valeur absolue), "
                 "la ligne est colorée en rouge (dépassement) ou en bleu (sous le devis).",
        )
    valeur = st.session_state["tolerance_ecart"]
    if valeur != TOLERANCE_PAR_DEFAUT:
        st.query_params["tolerance"] = str(valeur)
    elif "tolerance" in st.query_params:
        del st.query_params["tolerance"]
    return valeur


def seuil_tolerance_actuel() -> int:
    """Seuil de tolérance en vigueur (réglé par le bouton ⚙ au-dessus des tableaux d'OF)."""
    if "tolerance_ecart" not in st.session_state:
        depuis_url = st.query_params.get("tolerance", "")
        st.session_state["tolerance_ecart"] = (
            int(depuis_url) if depuis_url.isdigit() and 5 <= int(depuis_url) <= 100 else TOLERANCE_PAR_DEFAUT
        )
    return st.session_state["tolerance_ecart"]


COULEUR_BARRE_DEPASSEMENT = "#D64545"
COULEUR_BARRE_SOUS_DEVIS = "#2E86AB"
COULEUR_BARRE_TOLERANCE = COULEUR_REALISE
LIBELLES_CATEGORIES = {"poste": "postes", "dossier_client": "dossiers", "operation": "opérations"}


def graphique_devis_realise(of: pd.DataFrame, col: str, key: str, top: int | None = None,
                            libelle_vide: str = "Non défini") -> None:
    """Devisé (barre grise large) et réalisé (barre fine colorée selon l'écart) par `col`.

    Rouge : dépassement au-delà du seuil de tolérance ; bleu : sous le devis au-delà
    du seuil ; vert : dans la tolérance. Trié du plus gros dépassement (en heures, en
    haut) au plus gros gain. `top` : n'affiche que les `top` plus gros écarts (en
    valeur absolue) et propose le reste dans un tableau.
    """
    agg = (
        of.assign(**{col: of[col].fillna(libelle_vide).astype(str)})
        .groupby(col)[["temps_devis_h", "temps_operateurs_h"]].sum()
        .rename(columns={"temps_devis_h": "devise", "temps_operateurs_h": "realise"})
    )
    agg["ecart_h"] = agg["realise"] - agg["devise"]
    agg["ecart_pct"] = agg["ecart_h"] / agg["devise"].replace(0, np.nan)
    reste = None
    if top and len(agg) > top:
        ordre_abs = agg["ecart_h"].abs().sort_values(ascending=False).index
        reste = agg.loc[ordre_abs[top:]]
        agg = agg.loc[ordre_abs[:top]]
    agg = agg.sort_values("ecart_h")  # Plotly trace de bas en haut : le plus gros dépassement finit en haut

    seuil_pct = seuil_tolerance_actuel()
    seuil = seuil_pct / 100
    couleurs = [
        COULEUR_BARRE_DEPASSEMENT if (pd.notna(p) and p > seuil)
        else COULEUR_BARRE_SOUS_DEVIS if (pd.notna(p) and p < -seuil)
        else COULEUR_BARRE_TOLERANCE
        for p in agg["ecart_pct"]
    ]
    etiquettes = [
        f"{fr_heures(r)} ({fr_pct(p, 0, signe=True)})" if pd.notna(p) else fr_heures(r)
        for r, p in zip(agg["realise"], agg["ecart_pct"])
    ]
    fig = go.Figure()
    fig.add_bar(y=agg.index, x=agg["devise"], orientation="h", name="Devisé", width=0.8,
                marker_color=COULEUR_DEVISE,
                hovertemplate="%{y}<br>Devisé : %{x:.1f} h<extra></extra>")
    fig.add_bar(y=agg.index, x=agg["realise"], orientation="h", name="Réalisé", width=0.38,
                marker_color=couleurs,
                hovertemplate="%{y}<br>Réalisé : %{x:.1f} h<extra></extra>")
    # Étiquette placée après la plus longue des deux barres (jamais par-dessus la barre grise)
    fig.add_scatter(y=agg.index, x=agg[["devise", "realise"]].max(axis=1), mode="text",
                    text=etiquettes, textposition="middle right", cliponaxis=False,
                    textfont=dict(size=12, color="#2B2B2A"), hoverinfo="skip")
    fig.update_layout(
        barmode="overlay", height=max(220, 70 + 34 * len(agg)),
        margin=dict(l=10, r=140, t=10, b=10), showlegend=False,
        xaxis_title="Heures", yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=key)
    st.caption(
        f"Barre grise : devisé · barre fine : réalisé, :red[**■ dépassement de plus de {seuil_pct} %**] · "
        f":blue[**■ sous le devis de plus de {seuil_pct} %**] · :green[**■ dans la tolérance**]. "
        "Plus gros dépassement en haut."
    )
    if reste is not None:
        nom = LIBELLES_CATEGORIES.get(col, col)
        if st.toggle(f"Afficher les {len(reste)} autres {nom} (écarts plus faibles)", key=f"{key}_voir_reste"):
            tableau = reste.sort_values("ecart_h", ascending=False).reset_index().rename(columns={
                col: "Libellé", "devise": "Devisé (h)", "realise": "Réalisé (h)",
                "ecart_h": "Écart (h)", "ecart_pct": "Écart (%)",
            })
            tableau["Écart (%)"] = (tableau["Écart (%)"] * 100).round(0)
            for c in ["Devisé (h)", "Réalisé (h)", "Écart (h)"]:
                tableau[c] = tableau[c].round(2)
            afficher_tableau(tableau, key=f"{key}_reste")


def legende_ecarts(cle: str) -> int:
    """Légende des couleurs au-dessus du tableau des OF, avec le réglage du seuil à côté."""
    col_legende, col_reglage = st.columns([5, 1.4], vertical_alignment="center")
    with col_reglage:
        seuil = reglage_tolerance(cle)
    col_legende.caption(
        f"⬜ Opération non pointée · 🟥 Dépassement de plus de {seuil} % · "
        f"🟦 Sous le devis de plus de {seuil} % · 🟩 Dans la tolérance (±{seuil} %)"
    )
    return seuil


def couleur_ecart(realise_h, ecart_pct, seuil) -> str:
    """Couleur de fond d'une ligne d'OF selon l'écart réalisé / devisé."""
    if realise_h == 0:
        return f"background-color: {COULEUR_LIGNE_NON_POINTEE}"
    if pd.isna(ecart_pct):
        return ""
    if ecart_pct > seuil:
        return f"background-color: {COULEUR_LIGNE_DEPASSEMENT}"
    if ecart_pct < -seuil:
        return f"background-color: {COULEUR_LIGNE_SOUS_DEVIS}"
    return f"background-color: {COULEUR_LIGNE_TOLERANCE}"


def message_aucune_cloture(periode: str) -> str:
    """Message unique quand aucun dossier n'a été clôturé sur la période."""
    derniere = df_of.loc[df_of["statut_production"] == "Clos", "date_cloture"].max()
    suite = f" Dernière clôture enregistrée : {derniere.strftime('%d/%m/%Y')}." if pd.notna(derniere) else ""
    return f"Aucun dossier clôturé {periode}.{suite}"


# ---------------------------------------------------------------------------
# Pages — Journée / Semaine / Mois / Année / Dossiers
# (chaque page est une fonction ; seule la page affichée est calculée)
# ---------------------------------------------------------------------------

def page_jour():
    titre_page("Suivi journalier", "Qui a pointé quoi, et quand, sur une journée")

    # ---------------------------------------------------------------------------
    # Sélecteur de jour
    # ---------------------------------------------------------------------------
    maintenant = maintenant_local()
    aujourdhui = maintenant.date()
    jours_avec_pointages = sorted(df_pointages["heure_debut"].dropna().dt.date.unique())

    # Jour choisi gardé en session (il survit au changement de page) et dans l'URL (?jour=2026-11-03)
    if "periode_jour" not in st.session_state:
        try:
            st.session_state["periode_jour"] = date.fromisoformat(st.query_params.get("jour", ""))
        except ValueError:
            st.session_state["periode_jour"] = (
                aujourdhui if (not jours_avec_pointages or aujourdhui in jours_avec_pointages)
                else jours_avec_pointages[-1]
            )
    st.session_state["_widget_jour"] = st.session_state["periode_jour"]

    def _aller_au_jour(sens: int) -> None:
        """Passe au jour précédent / suivant ayant des pointages."""
        courant = st.session_state["periode_jour"]
        if sens < 0:
            candidats = [j for j in jours_avec_pointages if j < courant]
            if candidats:
                st.session_state["periode_jour"] = candidats[-1]
        else:
            candidats = [j for j in jours_avec_pointages if j > courant]
            if candidats:
                st.session_state["periode_jour"] = candidats[0]

    def _choisir_jour() -> None:
        st.session_state["periode_jour"] = st.session_state["_widget_jour"]

    def _aller_a_aujourdhui() -> None:
        st.session_state["periode_jour"] = aujourdhui

    nav1, nav2, nav3, nav4 = st.columns([1, 2.4, 1, 1])
    nav1.button("◀ Jour préc.", on_click=_aller_au_jour, args=(-1,), use_container_width=True,
                help="Jour précédent ayant des pointages")
    with nav2:
        st.date_input("Jour affiché", key="_widget_jour", on_change=_choisir_jour,
                      format="DD/MM/YYYY", label_visibility="collapsed")
    nav3.button("Jour suiv. ▶", on_click=_aller_au_jour, args=(1,), use_container_width=True,
                help="Jour suivant ayant des pointages")
    nav4.button("Aujourd'hui", on_click=_aller_a_aujourdhui, use_container_width=True,
                disabled=st.session_state["periode_jour"] == aujourdhui)

    jour = st.session_state["periode_jour"]
    st.query_params["jour"] = jour.isoformat()
    est_aujourdhui = jour == aujourdhui
    st.caption(
        f"Journée du **{JOURS_SEMAINE[jour.weekday()]} {jour.strftime('%d/%m/%Y')}**"
        + (" — en cours (les pointages non terminés sont prolongés jusqu'à maintenant)" if est_aujourdhui else "")
    )

    pointages_jour = df_pointages.loc[df_pointages["heure_debut"].dt.date == jour].copy()

    # Alerte "pointages suspects" en haut de page (lien vers la liste en bas de page)
    alerte_pointages_suspects(pointages_jour, "la journée sélectionnée", "pointages-suspects-jour")

    if pointages_jour.empty:
        st.info("Aucun pointage ce jour-là. Utilisez ◀ / ▶ pour passer au jour précédent / suivant ayant des pointages.")
    else:
        # --- Fin "affichée" des pointages non terminés ------------------------
        en_cours = pointages_jour["heure_fin"].isna()
        fin_estimee = pointages_jour["heure_debut"] + pd.to_timedelta(pointages_jour["duree"], unit="ms")
        fin_par_defaut = maintenant if est_aujourdhui else pointages_jour["heure_debut"] + pd.Timedelta(minutes=15)
        pointages_jour["fin_affichee"] = (
            pointages_jour["heure_fin"].fillna(fin_estimee.where(fin_estimee > pointages_jour["heure_debut"]))
        )
        pointages_jour.loc[pointages_jour["fin_affichee"].isna(), "fin_affichee"] = fin_par_defaut
        pointages_jour["fin_affichee"] = pointages_jour[["fin_affichee", "heure_debut"]].max(axis=1)
        pointages_jour["Statut"] = np.where(en_cours, "En cours / non clôturé", "Terminé")
        # Durée affichée : durée enregistrée, sinon durée jusqu'à la fin affichée
        pointages_jour["Durée affichée h"] = pointages_jour["Durée h"].where(
            ~en_cours | pointages_jour["Durée h"].gt(0),
            (pointages_jour["fin_affichee"] - pointages_jour["heure_debut"]).dt.total_seconds() / 3600,
        ).clip(lower=0)

        # --- Indicateurs ------------------------------------------------------
        nb_op_jour = pointages_jour["id_operateur"].nunique()
        heures_jour = pointages_jour["Durée affichée h"].sum()
        nb_en_cours = int(en_cours.sum())

        st.subheader("👷 Activité de la journée")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        # Moyenne par opérateur : en cours de journée, on la compare aux heures
        # théoriques déjà écoulées (7,8 h au prorata des horaires d'ouverture), et
        # non aux 7,8 h d'une journée complète.
        moyenne_op = heures_jour / nb_op_jour if nb_op_jour else float("nan")
        heures_possibles = TAUX_HORAIRE_JOUR * (
            fraction_journee_ecoulee(maintenant) if est_aujourdhui else (1.0 if jour.weekday() < 5 else 0.0)
        )
        if est_aujourdhui and 0 < heures_possibles < TAUX_HORAIRE_JOUR:
            valeur_moyenne = fr_heures(moyenne_op)
            sous_texte_moyenne = f"sur {fr_heures(heures_possibles)} théoriques écoulées"
            aide_moyenne = (
                f"Heures pointées par opérateur depuis le début de la journée, comparées aux "
                f"{fr_heures(heures_possibles)} théoriques déjà écoulées "
                f"({fr_heures(TAUX_HORAIRE_JOUR)} sur une journée complète, au prorata des horaires "
                "d'ouverture 5h00–16h30)."
            )
        else:
            valeur_moyenne = fr_heures(moyenne_op)
            sous_texte_moyenne = f"sur {fr_heures(TAUX_HORAIRE_JOUR)} théoriques" if heures_possibles else ""
            aide_moyenne = f"Heures pointées par opérateur, à comparer aux {fr_heures(TAUX_HORAIRE_JOUR)} théoriques d'une journée."

        k1.metric("Opérateurs", f"{nb_op_jour}", help="Opérateurs ayant pointé au moins une fois ce jour-là.")
        k2.metric("Heures pointées", fr_heures(heures_jour),
                  help="Somme des durées des pointages de la journée (pointages en cours inclus jusqu'à maintenant).")
        k3.metric("Moy. / opérateur", valeur_moyenne, help=aide_moyenne)
        if sous_texte_moyenne:
            k3.caption(sous_texte_moyenne)
        k4.metric("Dossiers", f"{pointages_jour['ordre_fabrication'].nunique()}",
                  help="Dossiers sur lesquels au moins un pointage a été fait ce jour-là.")
        k5.metric("Non clôturés", f"{nb_en_cours}",
                  help="Pointages sans heure de fin (en cours, ou oubli de clôture).")

        # --- Opérateurs actifs n'ayant pas pointé ----------------------------
        df_emp = load_employes()
        if not df_emp.empty and {"fonction", "actif"}.issubset(df_emp.columns):
            actif = df_emp["actif"].astype(str).str.lower().isin(["true", "1", "t", "oui", "yes"])
            operateurs_actifs = df_emp.loc[
                actif & df_emp["fonction"].astype(str).str.lower().str.startswith("operat")
            ]
            ids_jour = set(_normalise_id(pointages_jour["id_operateur"]))
            absents = operateurs_actifs.loc[~operateurs_actifs["cle"].isin(ids_jour), "libelle"].sort_values()
            k6.metric("Sans pointage", f"{len(absents)}",
                      help="Opérateurs actifs (table employes) qui n'ont aucun pointage ce jour-là : "
                           "absence, congé… ou oubli de pointage. Liste juste en dessous.")
            if len(absents):
                with st.expander(f"Voir les {len(absents)} opérateur(s) sans pointage"):
                    st.write(" · ".join(absents))
        else:
            k6.metric("Sans pointage", "–", help="Liste des employés indisponible.")

        # --- Chronologie (Gantt) ---------------------------------------------
        st.markdown("**Chronologie des pointages**")
        couleur_par = st.radio(
            "Couleur par", ["Dossier", "Opération"], horizontal=True, key="gantt_couleur_jour",
        )
        col_couleur = "dossier_client" if couleur_par == "Dossier" else "operation"

        ordre_ops = (
            pointages_jour.groupby("operateur")["heure_debut"].min().sort_values().index.tolist()
        )
        donnees_gantt = pointages_jour.assign(
            debut_txt=pointages_jour["heure_debut"].dt.strftime("%H:%M"),
            fin_txt=np.where(en_cours, "en cours", pointages_jour["fin_affichee"].dt.strftime("%H:%M")),
            duree_txt=pointages_jour["Durée affichée h"].map(lambda h: f"{int(h)} h {round((h % 1) * 60):02d}" if pd.notna(h) else "–"),
        )
        carte_couleurs = couleurs_par_categorie(
            pointages_jour.sort_values("heure_debut")[col_couleur]
        )
        fig_gantt = px.timeline(
            donnees_gantt,
            x_start="heure_debut", x_end="fin_affichee", y="operateur",
            color=col_couleur, color_discrete_map=carte_couleurs,
            pattern_shape="Statut",
            pattern_shape_map={"Terminé": "", "En cours / non clôturé": "/"},
            category_orders={"operateur": ordre_ops},
            custom_data=["operation", "dossier_client", "debut_txt", "fin_txt", "duree_txt"],
        )
        # Une seule entrée de légende par dossier/opération (le hachurage suffit pour le statut)
        deja_en_legende = set()
        for trace in fig_gantt.data:
            nom = trace.name.rsplit(", ", 1)[0] if trace.name else trace.name
            trace.name, trace.legendgroup = nom, nom
            trace.showlegend = nom not in deja_en_legende
            deja_en_legende.add(nom)
        fig_gantt.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>%{customdata[2]} → %{customdata[3]} (%{customdata[4]})"
                "<br>Opération : %{customdata[0]}<br>Dossier : %{customdata[1]}<extra></extra>"
            ),
            marker_line_width=0.5, marker_line_color="white",
        )
        debut_axe = min(pointages_jour["heure_debut"].min().floor("h"), pd.Timestamp(jour) + pd.Timedelta(hours=7))
        fin_axe = max(pointages_jour["fin_affichee"].max().ceil("h"), pd.Timestamp(jour) + pd.Timedelta(hours=17))
        fig_gantt.update_xaxes(range=[debut_axe, fin_axe], tickformat="%H:%M", dtick=3_600_000,
                               showgrid=True, title=None)
        fig_gantt.update_yaxes(title=None, autorange="reversed")

        # Heures de fermeture de l'atelier grisées (les pointages qui y tombent sont suspects)
        if jour.weekday() in HORAIRES_ATELIER:  # mêmes horaires que la règle « hors horaires »
            ouverture_h, fermeture_h = HORAIRES_ATELIER[jour.weekday()]
            zones_fermees = [(debut_axe, pd.Timestamp(jour) + pd.Timedelta(hours=ouverture_h)),
                             (pd.Timestamp(jour) + pd.Timedelta(hours=fermeture_h), fin_axe)]
        else:
            zones_fermees = [(debut_axe, fin_axe)]
        for x0, x1 in zones_fermees:
            if x0 < x1:
                fig_gantt.add_vrect(x0=x0, x1=x1, fillcolor="#9A9A98", opacity=0.13,
                                    line_width=0, layer="below")

        # Pointages suspects entourés de rouge
        suspects_jour = donnees_gantt.loc[donnees_gantt["motifs_suspects"] != ""]
        for _, ligne_s in suspects_jour.iterrows():
            position = ordre_ops.index(ligne_s["operateur"])
            fig_gantt.add_shape(
                type="rect", x0=ligne_s["heure_debut"], x1=ligne_s["fin_affichee"],
                y0=position - 0.42, y1=position + 0.42, line=dict(color="#D64545", width=2.5),
                fillcolor="rgba(0,0,0,0)",
            )

        if est_aujourdhui and debut_axe <= maintenant <= fin_axe:
            fig_gantt.add_vline(x=maintenant, line_dash="dot", line_color="#d62728")
            fig_gantt.add_annotation(x=maintenant, y=1, yref="paper", yanchor="bottom", text="maintenant",
                                     showarrow=False, bgcolor="white", font=dict(color="#d62728", size=11))
        fig_gantt.update_layout(
            height=120 + 38 * len(ordre_ops),
            legend_title_text=couleur_par,
            margin=dict(l=10, r=10, t=30, b=10),
            bargap=0.25,
        )
        st.plotly_chart(fig_gantt, use_container_width=True, config=PLOTLY_CONFIG, key="gantt_jour")
        st.caption("Zones grisées : atelier fermé · contour rouge : pointage suspect · blocs hachurés : "
                   "pointages sans heure de fin (en cours ou oubli de clôture) · espaces vides : temps non pointé.")

        # --- Récapitulatif par opérateur -------------------------------------
        st.markdown("**Récapitulatif par opérateur**")
        recap = (
            pointages_jour.sort_values("heure_debut")
            .groupby("operateur")
            .agg(
                arrivee=("heure_debut", "min"),
                depart=("fin_affichee", "max"),
                heures_pointees=("Durée affichée h", "sum"),
                nb_pointages=("id", "count"),
                en_cours=("Statut", lambda x: int((x != "Terminé").sum())),
                dossiers=("ordre_fabrication", lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))),
            )
            .reset_index()
        )
        recap["amplitude_h"] = (recap["depart"] - recap["arrivee"]).dt.total_seconds() / 3600
        recap["non_pointe_h"] = (recap["amplitude_h"] - recap["heures_pointees"]).clip(lower=0)
        recap = recap.sort_values("arrivee")
        recap["arrivee"] = recap["arrivee"].dt.strftime("%H:%M")
        recap["depart"] = recap["depart"].dt.strftime("%H:%M")
        recap = recap.rename(columns={
            "operateur": "Opérateur",
            "arrivee": "Premier pointage",
            "depart": "Dernière fin",
            "amplitude_h": "Amplitude (h)",
            "heures_pointees": "Heures pointées (h)",
            "non_pointe_h": "Trous entre pointages (h)",
            "nb_pointages": "Nb pointages",
            "en_cours": "Non clôturés",
            "dossiers": "Dossiers",
        })[["Opérateur", "Premier pointage", "Dernière fin", "Amplitude (h)", "Heures pointées (h)",
            "Trous entre pointages (h)", "Nb pointages", "Non clôturés", "Dossiers"]]

        recap.insert(1, "État", np.select(
            [recap["Non clôturés"] > 0, recap["Trous entre pointages (h)"] > 0.5],
            ["À clôturer", "Temps non pointé"], default="OK",
        ))

        def _style_recap(row):
            if row["Non clôturés"] > 0:
                return ["background-color: #ffe5b4"] * len(row)   # orange : pointage à clôturer
            if row["Trous entre pointages (h)"] > 0.5:
                return ["background-color: #fff3cd"] * len(row)   # jaune : temps non pointé
            return [""] * len(row)

        afficher_tableau(recap, key="grille_recap_jour", style_ligne=_style_recap)
        st.caption("État « À clôturer » (orange) : au moins un pointage sans heure de fin · "
                   "« Temps non pointé » (jaune) : plus de 30 min non pointées entre le premier et le "
                   "dernier pointage (pause déjeuner comprise).")

        # --- Heures par dossier ----------------------------------------------
        st.markdown("**Heures pointées par dossier**")
        par_dossier = (
            pointages_jour.groupby("dossier_client")["Durée affichée h"].sum()
            .sort_values(ascending=True).reset_index()
        )
        # Mêmes couleurs que la chronologie (colorée par dossier)
        couleurs_dossiers = couleurs_par_categorie(pointages_jour.sort_values("heure_debut")["dossier_client"])
        fig_dossiers = px.bar(par_dossier, x="Durée affichée h", y="dossier_client", orientation="h",
                              text_auto=".1f", color="dossier_client", color_discrete_map=couleurs_dossiers)
        fig_dossiers.update_layout(height=80 + 28 * len(par_dossier), xaxis_title="Heures",
                                   yaxis_title=None, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_dossiers, use_container_width=True, config=PLOTLY_CONFIG, key="bar_dossiers_jour")

        # --- Détail -------------------------------------------------------------
        table_jour = pointages_jour[
            ["id", "operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h", "Statut"]
        ].sort_values(["heure_debut"]).reset_index(drop=True)
        table_jour["Durée h"] = table_jour["Durée h"].round(2)
        # Le jour est déjà choisi : on n'affiche que l'heure
        table_jour["heure_debut"] = table_jour["heure_debut"].dt.strftime("%H:%M")
        table_jour["heure_fin"] = table_jour["heure_fin"].dt.strftime("%H:%M").fillna("en cours")
        # Interrupteur plutôt que volet replié : un tableau créé dans un volet fermé
        # se dimensionne à 0 px de large et reste vide à l'ouverture.
        if st.toggle(f"Afficher le détail des {len(table_jour)} pointages de la journée", key="voir_detail_jour"):
            afficher_tableau(table_jour, key="grille_pointages_jour")

        # --- Pointages suspects (bas de page) -----------------------------------
        section_pointages_suspects(pointages_jour, "cette journée", "pointages-suspects-jour",
                                   key="grille_suspects_jour")


def page_semaine():
    titre_page("Suivi hebdomadaire", "Heures pointées et ordres de fabrication clôturés dans la semaine")

    # ---------------------------------------------------------------------------
    # Sélecteur de semaine
    # ---------------------------------------------------------------------------

    weeks_df = build_week_options(df_pointages, df_of)
    if weeks_df.empty:
        st.warning("Aucune semaine exploitable n'a été trouvée dans la base.")
        st.stop()

    annee_courante, semaine_courante, _ = maintenant_local().date().isocalendar()
    semaine_actuelle = id_semaine(annee_courante, semaine_courante)
    options_semaines = sorted(
        {id_semaine(a, s) for a, s in zip(weeks_df["iso_year"], weeks_df["iso_week"])} | {semaine_actuelle}
    )
    semaine_choisie = selecteur_periode(
        "semaine", options_semaines, libelle_semaine, semaine_actuelle, "Cette semaine",
        libelles=("◀ Semaine préc.", "Semaine suiv. ▶"),
    )
    sel_year, sel_week = (int(x) for x in semaine_choisie.split("-S"))
    monday, sunday = week_bounds(sel_year, sel_week)

    # ---------------------------------------------------------------------------
    # Filtrage des données de la semaine
    # ---------------------------------------------------------------------------

    mask_pointages = (df_pointages["iso_year"] == sel_year) & (df_pointages["iso_week"] == sel_week)
    pointages_semaine = df_pointages.loc[mask_pointages].copy()

    # Alerte "pointages suspects" en haut de page (lien vers la liste en bas de page)
    alerte_pointages_suspects(pointages_semaine, "la semaine sélectionnée", "pointages-suspects-semaine")

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
    semaine_en_cours = periode_en_cours(monday, sunday)
    jours_ouvres_semaine_sel = jours_ouvres_theoriques(monday, sunday)
    temps_theorique = temps_theorique_semaine(pointages_semaine, monday, sunday)
    taux_attribution = (heures_attribuees / temps_theorique) if temps_theorique > 0 else float("nan")

    st.subheader("👷 Activité des opérateurs")
    if semaine_en_cours:
        st.caption(
            f"⏳ Semaine en cours : le temps théorique ne compte que les jours ouvrés déjà écoulés "
            f"({fr_nombre(jours_ouvres_semaine_sel)} j sur {nb_jours_ouvres(monday, sunday)})."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Heures pointées", fr_heures(heures_attribuees),
              help="Somme des pointages des opérateurs sur la semaine, tous dossiers confondus.")
    c2.metric("Opérateurs", f"{nb_operateurs}", help="Opérateurs ayant pointé au moins une fois dans la semaine.")
    c3.metric(
        "Temps théorique",
        fr_heures(temps_theorique, 0),
        help=(
            f"{nb_operateurs} opérateur(s) ayant pointé au moins une fois cette semaine × "
            f"{fr_nombre(jours_ouvres_semaine_sel)} jours ouvrés"
            + (" écoulés" if semaine_en_cours else "")
            + f" × {fr_heures(TAUX_HORAIRE_JOUR)}. On suppose qu'un opérateur "
            "ayant pointé au moins une fois dans la semaine y a travaillé toute la semaine."
        ),
    )
    c4.metric(
        "Taux d'attribution",
        fr_pct(taux_attribution),
        help="Heures pointées ÷ temps théorique.",
    )

    if pointages_semaine.empty:
        st.info("Aucun pointage sur cette semaine.")
    else:
        col_pie1, col_pie2 = st.columns(2)

        with col_pie1:
            st.markdown("**Heures pointées par dossier**")
            graphique_repartition(pointages_semaine, "dossier_client", key="bar_heures_par_dossier")

        with col_pie2:
            st.markdown("**Heures pointées par opérateur**")
            graphique_repartition(pointages_semaine, "operateur", key="bar_heures_par_operateur")

        if not pointages_semaine.empty:
            table_pointages = pointages_semaine[
                ["id", "operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
            ].sort_values("heure_debut").reset_index(drop=True)
            table_pointages["Durée h"] = table_pointages["Durée h"].round(2)
            if st.toggle(f"Afficher le détail des {len(table_pointages)} pointages de la semaine",
                         key="voir_detail_semaine"):
                afficher_tableau(table_pointages, key="grille_pointages_hebdo")
        else:
            st.info("Aucun pointage sur cette semaine.")

    # ---------------------------------------------------------------------------
    # Indicateurs — dossiers clôturés dans la semaine
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("📦 Dossiers clôturés cette semaine")

    if of_semaine.empty:
        st.info(message_aucune_cloture("cette semaine"))
    else:
        heures_livrees = of_semaine["temps_operateurs_h"].sum()
        heures_theoriques = of_semaine["temps_devis_h"].sum()
        ratio_temps = (heures_livrees / heures_theoriques) if heures_theoriques > 0 else 0
        nb_dossiers_clotures = of_semaine["numero_dossier"].nunique()

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Dossiers clôturés", f"{nb_dossiers_clotures}")
        d2.metric("Heures réalisées", fr_heures(heures_livrees),
                  help="Temps inscrit sur les ordres de fabrication des dossiers clôturés cette semaine.")
        d3.metric("Heures devisées", fr_heures(heures_theoriques),
                  help="Temps prévu au devis pour ces mêmes dossiers.")
        d4.metric("Réalisé / devisé", fr_pct(ratio_temps),
                  help="Heures réalisées ÷ heures devisées. Au-dessus de 100 % : dépassement du devis.")

        st.markdown("**Temps par poste**")
        if not of_semaine.empty:
            graphique_devis_realise(of_semaine, "poste", key="bar_temps_par_poste")
        else:
            st.info("Aucun dossier clôturé sur cette semaine.")

        st.markdown("**Temps par dossier clôturé**")
        if not of_semaine.empty:
            of_semaine_label = of_semaine.copy()
            of_semaine_label["dossier_client"] = (
                of_semaine_label["numero_dossier"].astype(str) + " – " + of_semaine_label["client"].fillna("Client inconnu")
            )
            graphique_devis_realise(of_semaine_label, "dossier_client", key="bar_temps_par_dossier")
        else:
            st.info("Aucun dossier clôturé sur cette semaine.")

        st.markdown("**Ordres de fabrication clôturés dans la semaine**")
        if not of_semaine.empty:
            of_semaine_filtered = of_semaine.copy()

            # -----------------------------------------------------------------------
            # Construction du tableau avec les données filtrées
            # -----------------------------------------------------------------------
            if not of_semaine_filtered.empty:
                table_of = of_semaine_filtered[
                    ["numero_dossier", "client", "operation", "temps_devis_h", "temps_operateurs_h"]
                ].rename(
                    columns={
                        "temps_devis_h": "temps_devis (h)",
                        "temps_operateurs_h": "temps_operateurs (h)",
                    }
                ).reset_index(drop=True)

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

                # Plus gros dépassement (en heures) en haut du tableau
                table_of = table_of.sort_values("delta (h)", ascending=False).reset_index(drop=True)
                ecart_pct = table_of["ratio (opérateurs-devis)/devis (%)"]

                def highlight_row(row):
                    return [couleur_ecart(row["temps_operateurs (h)"], ecart_pct.loc[row.name], seuil_ecart)] * len(row)

                seuil_ecart = legende_ecarts("grille_table_of")
                ligne_of = afficher_tableau(table_of, key="grille_table_of",
                                            style_ligne=highlight_row, selection=True)
                bouton_detail_pointages(ligne_of, None, key="btn_detail_of_hebdo")
            else:
                st.info("Aucun dossier ne correspond aux critères de recherche sélectionnés.")
        else:
            st.info("Aucun dossier clôturé sur cette semaine.")


    # ---------------------------------------------------------------------------
    # Pointages suspects de la semaine (bas de page)
    # ---------------------------------------------------------------------------
    section_pointages_suspects(pointages_semaine, "cette semaine", "pointages-suspects-semaine",
                               key="grille_suspects_semaine")

# ---------------------------------------------------------------------------
# Page — Suivi mensuel
# ---------------------------------------------------------------------------

def page_mois():
    titre_page("Suivi mensuel", "Heures pointées et ordres de fabrication clôturés dans le mois")

    # ---------------------------------------------------------------------------
    # Sélecteur de mois
    # ---------------------------------------------------------------------------

    months_df = build_month_options(df_pointages, df_of)
    if months_df.empty:
        st.warning("Aucun mois exploitable n'a été trouvé dans la base.")
        st.stop()

    aujourdhui_m = maintenant_local().date()
    mois_actuel = id_mois(aujourdhui_m.year, aujourdhui_m.month)
    options_mois = sorted(
        {id_mois(a, m) for a, m in zip(months_df["cal_year"], months_df["cal_month"])} | {mois_actuel}
    )
    mois_choisi = selecteur_periode(
        "mois", options_mois, libelle_mois, mois_actuel, "Ce mois-ci",
        libelles=("◀ Mois préc.", "Mois suiv. ▶"),
    )
    sel_year_m, sel_month = (int(x) for x in mois_choisi.split("-"))
    premier_jour, dernier_jour = month_bounds(sel_year_m, sel_month)

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
    mois_en_cours = periode_en_cours(premier_jour, dernier_jour)
    temps_theorique_m = temps_theorique_mois_via_semaines(df_pointages, premier_jour, dernier_jour)
    taux_attribution_m = (heures_attribuees_m / temps_theorique_m) if temps_theorique_m > 0 else float("nan")

    st.subheader("👷 Activité des opérateurs")
    if mois_en_cours:
        st.caption(
            f"⏳ Mois en cours : le temps théorique ne compte que les jours ouvrés déjà écoulés "
            f"({fr_nombre(jours_ouvres_theoriques(premier_jour, dernier_jour))} j sur {jours_ouvres_m})."
        )
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Heures pointées", fr_heures(heures_attribuees_m),
               help="Somme des pointages des opérateurs sur le mois, tous dossiers confondus.")
    cm2.metric("Opérateurs", f"{nb_operateurs_m}", help="Opérateurs ayant pointé au moins une fois dans le mois.")
    cm3.metric(
        "Temps théorique",
        fr_heures(temps_theorique_m, 0),
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
        fr_pct(taux_attribution_m),
        help="Heures pointées ÷ temps théorique (somme, semaine par semaine, du nb d'opérateurs × jours ouvrés écoulés × 7,8 h).",
    )

    if pointages_mois.empty:
        st.info("Aucun pointage sur ce mois.")
    else:
        col_pie1_m, col_pie2_m = st.columns(2)

        with col_pie1_m:
            st.markdown("**Heures pointées par dossier**")
            graphique_repartition(pointages_mois, "dossier_client", key="bar_heures_par_dossier_mensuel")

        with col_pie2_m:
            st.markdown("**Heures pointées par opérateur**")
            graphique_repartition(pointages_mois, "operateur", key="bar_heures_par_operateur_mensuel")

        if not pointages_mois.empty:
            table_pointages_m = pointages_mois[
                ["id", "operateur", "operation", "ordre_fabrication", "heure_debut", "heure_fin", "Durée h"]
            ].sort_values("heure_debut").reset_index(drop=True)
            table_pointages_m["Durée h"] = table_pointages_m["Durée h"].round(2)
            if st.toggle(f"Afficher le détail des {len(table_pointages_m)} pointages du mois",
                         key="voir_detail_mois"):
                afficher_tableau(table_pointages_m, key="grille_pointages_mensuel")
        else:
            st.info("Aucun pointage sur ce mois.")

    # ---------------------------------------------------------------------------
    # Indicateurs — dossiers clôturés dans le mois
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("📦 Dossiers clôturés ce mois")

    if of_mois.empty:
        st.info(message_aucune_cloture("ce mois"))
    else:
        heures_livrees_m = of_mois["temps_operateurs_h"].sum()
        heures_theoriques_m = of_mois["temps_devis_h"].sum()
        ratio_temps_m = (heures_livrees_m / heures_theoriques_m) if heures_theoriques_m > 0 else 0
        nb_dossiers_clotures_m = of_mois["numero_dossier"].nunique()

        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric("Dossiers clôturés", f"{nb_dossiers_clotures_m}")
        dm2.metric("Heures réalisées", fr_heures(heures_livrees_m),
                   help="Temps inscrit sur les ordres de fabrication des dossiers clôturés ce mois.")
        dm3.metric("Heures devisées", fr_heures(heures_theoriques_m),
                   help="Temps prévu au devis pour ces mêmes dossiers.")
        dm4.metric("Réalisé / devisé", fr_pct(ratio_temps_m),
                   help="Heures réalisées ÷ heures devisées. Au-dessus de 100 % : dépassement du devis.")

        st.markdown("**Temps par poste**")
        if not of_mois.empty:
            graphique_devis_realise(of_mois, "poste", key="bar_temps_par_poste_mensuel")
        else:
            st.info("Aucun dossier clôturé sur ce mois.")

        st.markdown("**Temps par dossier clôturé**")
        if not of_mois.empty:
            of_mois_label = of_mois.copy()
            of_mois_label["dossier_client"] = (
                of_mois_label["numero_dossier"].astype(str) + " – " + of_mois_label["client"].fillna("Client inconnu")
            )
            graphique_devis_realise(of_mois_label, "dossier_client", key="bar_temps_par_dossier_mensuel")
        else:
            st.info("Aucun dossier clôturé sur ce mois.")

        st.markdown("**Ordres de fabrication clôturés dans le mois**")
        if not of_mois.empty:
            of_mois_filtered = of_mois.copy()

            # -----------------------------------------------------------------------
            # Construction du tableau avec les données filtrées
            # -----------------------------------------------------------------------
            if not of_mois_filtered.empty:
                table_of_m = of_mois_filtered[
                    ["numero_dossier", "client", "operation", "temps_devis_h", "temps_operateurs_h"]
                ].rename(
                    columns={
                        "temps_devis_h": "temps_devis (h)",
                        "temps_operateurs_h": "temps_operateurs (h)",
                    }
                ).reset_index(drop=True)

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

                # Plus gros dépassement (en heures) en haut du tableau
                table_of_m = table_of_m.sort_values("delta (h)", ascending=False).reset_index(drop=True)
                ecart_pct_m = table_of_m["ratio (opérateurs-devis)/devis (%)"]

                def highlight_row_mensuel(row):
                    return [couleur_ecart(row["temps_operateurs (h)"], ecart_pct_m.loc[row.name], seuil_ecart)] * len(row)

                seuil_ecart = legende_ecarts("grille_table_of_m")
                ligne_of_m = afficher_tableau(table_of_m, key="grille_table_of_m",
                                              style_ligne=highlight_row_mensuel, selection=True)
                bouton_detail_pointages(ligne_of_m, None, key="btn_detail_of_mensuel")
            else:
                st.info("Aucun dossier ne correspond aux critères de recherche sélectionnés.")
        else:
            st.info("Aucun dossier clôturé sur ce mois.")


# ---------------------------------------------------------------------------
# Page — Dossiers
# ---------------------------------------------------------------------------

def page_dossiers():
    titre_page("Dossiers", "Devisé et réalisé d'un dossier, toutes périodes confondues (en cours, clôturé, ancien)")

    tous_les_dossiers = sorted(df_of["numero_dossier"].dropna().astype(str).unique().tolist())

    # Dossier choisi gardé en session et dans l'URL (?dossier=BC2600738) pour partager le lien
    if st.session_state.get("dossier_choisi") not in tous_les_dossiers:
        depuis_url = st.query_params.get("dossier", "")
        st.session_state["dossier_choisi"] = depuis_url if depuis_url in tous_les_dossiers else ""
    st.session_state["_widget_dossier"] = st.session_state["dossier_choisi"]

    def _choisir_dossier() -> None:
        st.session_state["dossier_choisi"] = st.session_state["_widget_dossier"]

    dossier_choisi = st.selectbox(
        "Numéro de dossier",
        options=[""] + tous_les_dossiers,
        format_func=lambda x: "Tapez ou choisissez un numéro de dossier…" if x == "" else x,
        key="_widget_dossier", on_change=_choisir_dossier,
        help="Vous pouvez taper directement le numéro pour le trouver plus vite.",
    )
    if dossier_choisi:
        st.query_params["dossier"] = dossier_choisi
    elif "dossier" in st.query_params:
        del st.query_params["dossier"]

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
            c_spec1.metric("Heures devisées", fr_heures(devis_total))
            c_spec2.metric("Heures réalisées", fr_heures(realise_total))

            # Écart en % sous la valeur : rouge si dépassement du devis, vert sinon
            ecart_pct_total = ecart_total / devis_total if devis_total else float("nan")
            c_spec3.metric(
                "Écart réalisé − devisé",
                fr_heures(ecart_total),
                delta=fr_pct(ecart_pct_total, 0, signe=True) if pd.notna(ecart_pct_total) else None,
                delta_color="inverse",
                help="Positif (rouge) : plus d'heures réalisées que devisées. Négatif (vert) : gain de temps.",
            )

            # 5. Graphique : Temps par poste
            st.markdown("**Temps par poste**")
            graphique_devis_realise(spec_of, "poste", key=f"bar_spec_{dossier_choisi}")

            # 6. Tableau détaillé par opérations du dossier
            st.markdown("**Détail des opérations (Ordres de Fabrication) de ce dossier**")

            table_spec_of = spec_of[
                ["reference", "operation", "statut_production", "temps_devis_h", "temps_operateurs_h"]
            ].rename(
                columns={
                    "temps_devis_h": "temps_devis (h)",
                    "temps_operateurs_h": "temps_operateurs (h)",
                }
            ).reset_index(drop=True)

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

            # Plus gros dépassement (en heures) en haut du tableau
            table_spec_of = table_spec_of.sort_values("delta (h)", ascending=False).reset_index(drop=True)
            ecart_pct_spec = table_spec_of["ratio (opérateurs-devis)/devis (%)"]

            def highlight_row_spec(row):
                return [couleur_ecart(row["temps_operateurs (h)"], ecart_pct_spec.loc[row.name], seuil_ecart)] * len(row)

            seuil_ecart = legende_ecarts("grille_table_spec_of")
            ligne_choisie = afficher_tableau(table_spec_of, key="grille_table_spec_of",
                                             style_ligne=highlight_row_spec, selection=True)
            bouton_detail_pointages(ligne_choisie, dossier_choisi, key="btn_detail_hebdo")

        else:
            st.warning("Ce dossier est introuvable dans la liste globale des ordres de fabrication.")


# ---------------------------------------------------------------------------
# Page — Bilan annuel
# ---------------------------------------------------------------------------

def page_annee():
    titre_page("Bilan annuel", "Devisé, réalisé et évolution sur l'année")

    # -- Sélecteur d'année -------------------------------------------------
    annees_pointages = df_pointages["iso_year"].dropna().astype(int).unique().tolist()
    annees_of = df_of["date_cloture"].dropna().dt.year.astype(int).unique().tolist()
    annees_disponibles = sorted(set(annees_pointages) | set(annees_of), reverse=True)

    if not annees_disponibles:
        st.warning("Aucune année exploitable n'a été trouvée dans la base.")
        st.stop()

    annee_actuelle = str(maintenant_local().year)
    options_annees = sorted({str(a) for a in annees_disponibles} | {annee_actuelle})
    annee_choisie = int(selecteur_periode(
        "annee", options_annees, lambda a: a + (" · en cours" if a == annee_actuelle else ""),
        annee_actuelle, "Cette année", libelles=("◀ Année préc.", "Année suiv. ▶"),
    ))

    with st.expander("ℹ️ Comment lire cette page"):
        st.markdown(
            "- **Heures pointées** : somme des pointages des opérateurs (table *temps_reel_operateur*) "
            "sur l'année, tous dossiers confondus.\n"
            "- **Temps théorique** : semaine par semaine, nombre d'opérateurs ayant pointé × jours "
            "ouvrés (hors jours fériés, jours écoulés seulement pour la semaine en cours) × 7,8 h.\n"
            "- **Pointées sur dossiers clôturés / en cours** : les mêmes heures pointées, réparties "
            "selon que toutes les opérations du dossier sont « Clos » ou non.\n"
            "- **Heures devisées / réalisées** : temps prévu au devis et temps inscrit sur les ordres de "
            "fabrication (table *ordres_fabrication*) des dossiers **clôturés dans l'année** "
            "(date de clôture). Les graphiques par poste et par opération portent sur ces mêmes dossiers."
        )

    # -- Données filtrées sur l'année ---------------------------------------
    pointages_annee = df_pointages.loc[df_pointages["iso_year"] == annee_choisie].copy()

    of_annee = df_of.loc[
        (df_of["date_cloture"].dt.year == annee_choisie)
        & (df_of["statut_production"] == "Clos")
    ].copy()

    # -----------------------------------------------------------------------
    # Calculs hebdomadaires (indicateurs généraux + évolution hebdomadaire)
    # -----------------------------------------------------------------------
    if not pointages_annee.empty:
        weekly = (
            pointages_annee.groupby("iso_week")
            .agg(heures_pointees=("Durée h", "sum"), nb_operateurs=("id_operateur", "nunique"))
            .reset_index()
            .sort_values("iso_week")
        )
        weekly["temps_theorique"] = weekly["iso_week"].map(
            temps_theorique_par_groupe_semaine(pointages_annee, "iso_week")
        )
        weekly["taux_attribution"] = weekly["heures_pointees"] / weekly["temps_theorique"].replace(0, np.nan)
        weekly["semaine"] = weekly["iso_week"].apply(lambda w: f"S{int(w):02d}")
        # La semaine en cours est incomplète : affichée à part et exclue des moyennes
        weekly["en_cours"] = weekly["iso_week"].apply(
            lambda w: periode_en_cours(*week_bounds(annee_choisie, int(w)))
        )
        semaines_completes = weekly.loc[~weekly["en_cours"]]

        temps_theorique_annuel = weekly["temps_theorique"].sum()
        heures_pointees_annee = weekly["heures_pointees"].sum()
        # Taux de l'année = total pointé ÷ total théorique (chaque semaine pèse selon son volume)
        taux_attribution_annuel = (
            heures_pointees_annee / temps_theorique_annuel if temps_theorique_annuel > 0 else np.nan
        )
        nb_semaines = len(weekly)
        moyenne_operateurs_semaine = weekly["nb_operateurs"].mean()

        # Heures pointées réparties selon le statut du dossier : un dossier est
        # « clôturé » si toutes ses opérations sont au statut « Clos ». Même population
        # (les pointages de l'année) des deux côtés, donc la somme tombe juste.
        statut_dossiers = (
            df_of.dropna(subset=["numero_dossier"])
            .assign(clos=lambda d: d["statut_production"] == "Clos")
            .groupby(_cle_texte(df_of.dropna(subset=["numero_dossier"])["numero_dossier"]))["clos"].all()
        )
        dossier_clos = _cle_texte(pointages_annee["ordre_fabrication"]).map(statut_dossiers).fillna(False).astype(bool)
        heures_dossiers_clos = pointages_annee.loc[dossier_clos, "Durée h"].sum()
        heures_dossiers_en_cours = pointages_annee.loc[~dossier_clos, "Durée h"].sum()
    else:
        weekly = pd.DataFrame()
        semaines_completes = pd.DataFrame()
        temps_theorique_annuel = heures_pointees_annee = taux_attribution_annuel = np.nan
        heures_dossiers_clos = heures_dossiers_en_cours = np.nan
        nb_semaines, moyenne_operateurs_semaine = 0, np.nan

    # -----------------------------------------------------------------------
    # Indicateurs généraux de l'année
    # -----------------------------------------------------------------------
    st.subheader(f"📈 Indicateurs généraux — {annee_choisie}")
    st.caption("À lire comme une formule : les signes ×, = et + montrent comment les indicateurs "
               "se déduisent les uns des autres.")

    def _kpi_box(label: str, value: str, help_text: str, **kwargs) -> None:
        """Indicateur dans un encadré, façon « bloc de formule »."""
        with st.container(border=True):
            st.metric(label, value, help=help_text, **kwargs)

    def _operateur(symbole: str) -> None:
        """Opérateur (×, =, +) centré et aligné avec les encadrés."""
        st.markdown(
            "<div style='text-align:center; font-size:2.1rem; font-weight:600; "
            f"padding-top:2.6rem;'>{symbole}</div>",
            unsafe_allow_html=True,
        )

    GRID = [3, 0.6, 3, 0.6, 3, 0.6, 3]

    # -- Ligne 1 : Temps théorique × Taux d'attribution = Heures pointées --
    l1a, l1op1, l1b, l1op2, l1c, _, _ = st.columns(GRID)
    with l1a:
        _kpi_box(
            "Temps théorique", fr_heures(temps_theorique_annuel, 0),
            help_text=f"Somme, semaine par semaine, du nombre d'opérateurs ayant pointé × jours ouvrés "
                      f"(hors fériés) × {fr_heures(TAUX_HORAIRE_JOUR)}. Sur les {nb_semaines} semaines "
                      f"comportant au moins un pointage, en moyenne {fr_nombre(moyenne_operateurs_semaine)} "
                      "opérateur(s) par semaine.",
        )
    with l1op1:
        _operateur("×")
    with l1b:
        _kpi_box("Taux d'attribution", fr_pct(taux_attribution_annuel),
                 help_text="Part du temps théorique effectivement pointée sur un dossier.")
    with l1op2:
        _operateur("=")
    with l1c:
        _kpi_box("Heures pointées", fr_heures(heures_pointees_annee),
                 help_text="Somme de tous les pointages de l'année, tous dossiers confondus.")

    # -- Ligne 2 : Heures pointées = sur dossiers clôturés + sur dossiers en cours --
    _, _, _, l2op2, l2c, l2op3, l2d = st.columns(GRID)
    with l2op2:
        _operateur("=")
    with l2c:
        _kpi_box("Pointées · dossiers clôturés", fr_heures(heures_dossiers_clos),
                 help_text="Heures pointées cette année sur des dossiers dont toutes les opérations "
                           "sont au statut « Clos » (travail livré).")
    with l2op3:
        _operateur("+")
    with l2d:
        _kpi_box("Pointées · dossiers en cours", fr_heures(heures_dossiers_en_cours),
                 help_text="Heures pointées cette année sur des dossiers pas encore clôturés "
                           "(ou absents de la table des ordres de fabrication) : travail pas encore livré.")

    if of_annee.empty:
        st.info(f"Aucun dossier clôturé en {annee_choisie}.")
    else:
        par_dossier_annee = of_annee.groupby("numero_dossier")[["temps_devis_h", "temps_operateurs_h"]].sum()
        nb_dossiers_annee = len(par_dossier_annee)
        devis_total_annee = par_dossier_annee["temps_devis_h"].sum()
        realise_total_annee = par_dossier_annee["temps_operateurs_h"].sum()
        devis_moyen_dossier = devis_total_annee / nb_dossiers_annee if nb_dossiers_annee else np.nan
        ecart_relatif = (realise_total_annee - devis_total_annee) / devis_total_annee if devis_total_annee else np.nan

        # -- Ligne 3 : Dossiers clôturés × Devisé moyen = Heures devisées ⚖ Heures réalisées --
        st.markdown(f"**Dossiers clôturés en {annee_choisie} : devisé et réalisé**")
        l3a, l3op1, l3b, l3op2, l3c, l3op3, l3d = st.columns(GRID)
        with l3a:
            _kpi_box("Dossiers clôturés", f"{nb_dossiers_annee}",
                     help_text=f"Dossiers ayant au moins une opération clôturée (statut « Clos ») en {annee_choisie}.")
        with l3op1:
            _operateur("×")
        with l3b:
            _kpi_box("Devisé moyen / dossier", fr_heures(devis_moyen_dossier),
                     help_text="Heures devisées ÷ nombre de dossiers clôturés.")
        with l3op2:
            _operateur("=")
        with l3c:
            _kpi_box("Heures devisées", fr_heures(devis_total_annee),
                     help_text=f"Temps prévu au devis pour les dossiers clôturés en {annee_choisie}.")
        with l3op3:
            _operateur("⚖")
        with l3d:
            _kpi_box("Heures réalisées", fr_heures(realise_total_annee),
                     help_text="Temps inscrit sur les ordres de fabrication de ces mêmes dossiers. "
                               "Sous la valeur : écart avec le devisé (rouge = dépassement).",
                     delta=fr_pct(ecart_relatif, 1, signe=True) if pd.notna(ecart_relatif) else None,
                     delta_color="inverse")

        st.markdown("**Par poste — devisé vs réalisé**")
        graphique_devis_realise(of_annee, "poste", key="bar_temps_par_poste_annuel")

        st.markdown("**Par opération — devisé vs réalisé (15 plus gros écarts)**")
        graphique_devis_realise(of_annee, "operation", key="bar_temps_par_operation_annuel", top=15)

    # -----------------------------------------------------------------------
    # Évolution hebdomadaire
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🗓️ Évolution hebdomadaire")

    if weekly.empty:
        st.info("Aucun pointage sur l'année sélectionnée.")
    else:
        moyenne_heures_semaine = semaines_completes["heures_pointees"].mean()
        moyenne_taux_semaine = _taux_pondere(semaines_completes)
        _graphiques_evolution(weekly, "semaine", moyenne_heures_semaine, moyenne_taux_semaine,
                              "la semaine en cours", "annuel_semaine")

    # -----------------------------------------------------------------------
    # Évolution mensuelle
    # -----------------------------------------------------------------------
    st.subheader("🗓️ Évolution mensuelle")

    if pointages_annee.empty:
        st.info("Aucun pointage sur l'année sélectionnée.")
    else:
        monthly = (
            pointages_annee.groupby("cal_month")
            .agg(heures_pointees=("Durée h", "sum"))
            .reset_index()
            .sort_values("cal_month")
        )
        monthly["temps_theorique"] = monthly["cal_month"].apply(
            lambda m: temps_theorique_mois_via_semaines(df_pointages, *month_bounds(annee_choisie, int(m)))
        )
        monthly["taux_attribution"] = monthly["heures_pointees"] / monthly["temps_theorique"].replace(0, np.nan)
        monthly["mois"] = monthly["cal_month"].apply(lambda m: NOMS_MOIS[int(m) - 1])
        monthly["en_cours"] = monthly["cal_month"].apply(
            lambda m: periode_en_cours(*month_bounds(annee_choisie, int(m)))
        )
        mois_complets = monthly.loc[~monthly["en_cours"]]
        _graphiques_evolution(monthly, "mois", mois_complets["heures_pointees"].mean(),
                              _taux_pondere(mois_complets), "le mois en cours", "annuel_mois")


def _taux_pondere(periodes: pd.DataFrame) -> float:
    """Taux d'attribution d'un ensemble de périodes : total pointé ÷ total théorique."""
    theorique = periodes["temps_theorique"].sum() if not periodes.empty else 0
    return periodes["heures_pointees"].sum() / theorique if theorique > 0 else np.nan


def _graphiques_evolution(donnees: pd.DataFrame, col_x: str, moyenne_heures: float, moyenne_taux: float,
                          libelle_en_cours: str, cle: str) -> None:
    """Heures pointées (barres) et taux d'attribution (courbe) par semaine ou par mois."""
    unite = "semaine" if col_x == "semaine" else "mois"
    col_g, col_d = st.columns(2)
    with col_g:
        st.markdown(f"**Heures pointées par {unite}**")
        fig_h = px.bar(donnees, x=col_x, y="heures_pointees",
                       labels={col_x: "", "heures_pointees": "Heures pointées"})
        marquer_periode_en_cours_barres(fig_h, donnees, col_x)
        fig_h.add_hline(y=moyenne_heures, line_dash="dash", line_color="firebrick",
                        annotation_text=f"Moyenne : {fr_heures(moyenne_heures)}",
                        annotation_position="top left", annotation_bgcolor="rgba(255,255,255,0.85)")
        st.plotly_chart(fig_h, use_container_width=True, config=PLOTLY_CONFIG, key=f"bar_heures_{cle}")
    with col_d:
        st.markdown(f"**Taux d'attribution par {unite}**")
        fig_t = courbe_taux_avec_periode_en_cours(donnees, col_x)
        fig_t.add_hline(y=moyenne_taux, line_dash="dash", line_color="firebrick",
                        annotation_text=f"Moyenne : {fr_pct(moyenne_taux)}",
                        annotation_position="top left", annotation_bgcolor="rgba(255,255,255,0.85)")
        st.plotly_chart(fig_t, use_container_width=True, config=PLOTLY_CONFIG, key=f"line_taux_{cle}")
    st.caption(f"Lignes pointillées : moyenne des {'semaines terminées' if unite == 'semaine' else 'mois terminés'} "
               f"(le taux est pondéré par le temps théorique, comme le taux de l'année). "
               f"En clair : {libelle_en_cours}, incomplet(e), exclu(e) des moyennes.")


# ---------------------------------------------------------------------------
# Démarrage : données, en-tête commun, puis la page choisie
# ---------------------------------------------------------------------------

injecter_style()

try:
    df_pointages, df_of, DONNEES_LUES_LE = load_data()
except Exception as e:
    st.error(f"Impossible de lire les données Supabase : {e}")
    st.stop()

PAGES = [
    st.Page(page_jour, title="Journée", icon="🕐", default=True),  # page d'accueil, à la racine du site
    st.Page(page_semaine, title="Semaine", icon="📊", url_path="semaine"),
    st.Page(page_mois, title="Mois", icon="📆", url_path="mois"),
    st.Page(page_annee, title="Année", icon="📅", url_path="annee"),
    st.Page(page_dossiers, title="Dossiers", icon="🔍", url_path="dossiers"),
]
page_courante = st.navigation(PAGES, position="hidden")  # menu dessiné par en_tete_compacte()
en_tete_compacte()
page_courante.run()
