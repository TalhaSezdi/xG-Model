"""
Fetch team logo URLs from Wikipedia REST API (thumbnail endpoint).
Saves results to data/team_logos.json.

Run once: python scripts/fetch_logos.py
Wikipedia thumbnails are stable CDN-hosted images, no API key needed.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "shots_features.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "team_logos.json"

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

# Wikipedia article title overrides (when team name != Wikipedia article title)
WIKI_OVERRIDES = {
    "Atletico Madrid": "Atletico_de_Madrid",
    "Atletico de Madrid": "Atletico_de_Madrid",
    "Paris Saint-Germain": "Paris_Saint-Germain_FC",
    "AC Milan": "A.C._Milan",
    "Inter Milan": "FC_Internazionale_Milano",
    "AS Roma": "AS_Roma",
    "AS Monaco": "AS_Monaco_FC",
    "Barcelona": "FC_Barcelona",
    "Real Madrid": "Real_Madrid_CF",
    "Arsenal": "Arsenal_F.C.",
    "Liverpool": "Liverpool_F.C.",
    "Chelsea": "Chelsea_F.C.",
    "Manchester City": "Manchester_City_F.C.",
    "Manchester United": "Manchester_United_F.C.",
    "Tottenham Hotspur": "Tottenham_Hotspur_F.C.",
    "Everton": "Everton_F.C.",
    "Leicester City": "Leicester_City_F.C.",
    "West Ham United": "West_Ham_United_F.C.",
    "Aston Villa": "Aston_Villa_F.C.",
    "Newcastle United": "Newcastle_United_F.C.",
    "Crystal Palace": "Crystal_Palace_F.C.",
    "AFC Bournemouth": "AFC_Bournemouth",
    "Southampton": "Southampton_F.C.",
    "Wolverhampton Wanderers": "Wolverhampton_Wanderers_F.C.",
    "Leeds United": "Leeds_United_A.F.C.",
    "Fulham": "Fulham_F.C.",
    "Sunderland": "Sunderland_A.F.C.",
    "Swansea City": "Swansea_City_A.F.C.",
    "Bolton Wanderers": "Bolton_Wanderers_F.C.",
    "Blackburn Rovers": "Blackburn_Rovers_F.C.",
    "Birmingham City": "Birmingham_City_F.C.",
    "Norwich City": "Norwich_City_F.C.",
    "Charlton Athletic": "Charlton_Athletic_F.C.",
    "Middlesbrough": "Middlesbrough_F.C.",
    "Stoke City": "Stoke_City_F.C.",
    "Watford": "Watford_F.C.",
    "Juventus": "Juventus_F.C.",
    "Napoli": "S.S.C._Napoli",
    "Lazio": "S.S._Lazio",
    "Fiorentina": "ACF_Fiorentina",
    "Atalanta": "Atalanta_B.C.",
    "Sampdoria": "U.C._Sampdoria",
    "Udinese": "Udinese_Calcio",
    "Genoa": "Genoa_C.F.C.",
    "Torino": "Torino_F.C.",
    "Bologna": "Bologna_F.C._1909",
    "Sassuolo": "U.S._Sassuolo_Calcio",
    "Empoli": "Empoli_F.C.",
    "Frosinone": "Frosinone_Calcio",
    "Chievo": "Hellas_Verona_F.C.",
    "Hellas Verona": "Hellas_Verona_F.C.",
    "Palermo": "Palermo_F.C.",
    "Carpi": "Carpi_F.C._1909",
    "Bayern Munich": "FC_Bayern_Munich",
    "Borussia Dortmund": "Borussia_Dortmund",
    "Bayer Leverkusen": "Bayer_04_Leverkusen",
    "Borussia Monchengladbach": "Borussia_Monchengladbach",
    "VfB Stuttgart": "VfB_Stuttgart",
    "Schalke 04": "FC_Schalke_04",
    "Wolfsburg": "VfL_Wolfsburg",
    "Augsburg": "FC_Augsburg",
    "Hoffenheim": "TSG_1899_Hoffenheim",
    "Werder Bremen": "SV_Werder_Bremen",
    "Eintracht Frankfurt": "Eintracht_Frankfurt",
    "RB Leipzig": "RB_Leipzig",
    "Hertha Berlin": "Hertha_BSC",
    "Freiburg": "SC_Freiburg",
    "FSV Mainz 05": "1._FSV_Mainz_05",
    "Darmstadt 98": "SV_Darmstadt_98",
    "Bochum": "VfL_Bochum",
    "Hamburger SV": "Hamburger_SV",
    "Hannover 96": "Hannover_96",
    "FC Heidenheim": "1._FC_Heidenheim_1846",
    "Union Berlin": "1._FC_Union_Berlin",
    "FC Koln": "1._FC_Koln",
    "Real Sociedad": "Real_Sociedad",
    "Athletic Club": "Athletic_Club",
    "Sevilla": "Sevilla_FC",
    "Valencia": "Valencia_CF",
    "Real Betis": "Real_Betis",
    "Villarreal": "Villarreal_CF",
    "Celta Vigo": "RC_Celta_de_Vigo",
    "Espanyol": "RCD_Espanyol",
    "Getafe": "Getafe_CF",
    "Osasuna": "CA_Osasuna",
    "Girona": "Girona_FC",
    "Rayo Vallecano": "Rayo_Vallecano",
    "Mallorca": "RCD_Mallorca",
    "Real Valladolid": "Real_Valladolid",
    "Deportivo Alaves": "Deportivo_Alaves",
    "Atletico de Madrid": "Atletico_de_Madrid",
    "Almeria": "UD_Almeria",
    "Cadiz": "Cadiz_CF",
    "Elche": "Elche_CF",
    "Huesca": "SD_Huesca",
    "Las Palmas": "UD_Las_Palmas",
    "Leganes": "CD_Leganes",
    "Real Zaragoza": "Real_Zaragoza",
    "Lyon": "Olympique_Lyonnais",
    "Marseille": "Olympique_de_Marseille",
    "Olympique de Marseille": "Olympique_de_Marseille",
    "Lille": "LOSC_Lille",
    "Nantes": "FC_Nantes",
    "Rennes": "Stade_Rennais_F.C.",
    "Strasbourg": "RC_Strasbourg_Alsace",
    "Montpellier": "Montpellier_HSC",
    "Bordeaux": "FC_Girondins_de_Bordeaux",
    "Lens": "RC_Lens",
    "Angers": "SCO_Angers",
    "Lorient": "FC_Lorient",
    "Troyes": "ESTAC_Troyes",
    "Toulouse": "Toulouse_FC",
    "OGC Nice": "OGC_Nice",
    "Ajax": "AFC_Ajax",
    "River Plate": "Club_Atletico_River_Plate",
    "Boca Juniors": "Club_Atletico_Boca_Juniors",
    "Inter Miami": "Inter_Miami_CF",
    "Seattle Sounders": "Seattle_Sounders_FC",
    "New York Red Bulls": "New_York_Red_Bulls",
    "LAFC": "Los_Angeles_FC",
    "Toronto FC": "Toronto_FC",
    "FC Porto": "FC_Porto",
    "Panathinaikos": "Panathinaikos_F.C.",
    "West Bromwich Albion": "West_Bromwich_Albion_F.C.",
    "Portsmouth": "Portsmouth_F.C.",
    "Stade Malherbe Caen": "SM_Caen",
    "Caen": "SM_Caen",
    "Guingamp": "En_Avant_de_Guingamp",
    "Stade de Reims": "Stade_de_Reims",
    "Stade Brestois": "Stade_Brestois_29",
    "Clermont Foot": "Clermont_Foot_Auvergne_63",
    "Auxerre": "AJ_Auxerre",
    "AC Ajaccio": "AC_Ajaccio",
    "Bastia": "SC_Bastia",
    "Saint-Etienne": "AS_Saint-Etienne",
    # National teams
    "Spain": "Spain_national_football_team",
    "France": "France_national_football_team",
    "Germany": "Germany_national_football_team",
    "England": "England_national_football_team",
    "Brazil": "Brazil_national_football_team",
    "Argentina": "Argentina_national_football_team",
    "Portugal": "Portugal_national_football_team",
    "Italy": "Italy_national_football_team",
    "Belgium": "Belgium_national_football_team",
    "Netherlands": "Netherlands_national_football_team",
    "Croatia": "Croatia_national_football_team",
    "Uruguay": "Uruguay_national_football_team",
    "Colombia": "Colombia_national_football_team",
    "Denmark": "Denmark_national_football_team",
    "Poland": "Poland_national_football_team",
    "Sweden": "Sweden_national_football_team",
    "Switzerland": "Switzerland_national_football_team",
    "Mexico": "Mexico_national_football_team",
    "Japan": "Japan_national_football_team",
    "South Korea": "South_Korea_national_football_team",
    "Australia": "Australia_national_football_team",
    "Morocco": "Morocco_national_football_team",
    "Senegal": "Senegal_national_football_team",
    "Nigeria": "Nigeria_national_football_team",
    "Ghana": "Ghana_national_football_team",
    "Egypt": "Egypt_national_football_team",
    "Turkey": "Turkey_national_football_team",
    "Austria": "Austria_national_football_team",
    "Serbia": "Serbia_national_football_team",
    "Ukraine": "Ukraine_national_football_team",
    "Russia": "Russia_national_football_team",
    "Czech Republic": "Czech_Republic_national_football_team",
    "Hungary": "Hungary_national_football_team",
    "Romania": "Romania_national_football_team",
    "Scotland": "Scotland_national_football_team",
    "Wales": "Wales_national_football_team",
    "Iran": "Iran_national_football_team",
    "Canada": "Canada_national_soccer_team",
    "United States": "United_States_men%27s_national_soccer_team",
    "Chile": "Chile_national_football_team",
    "Ecuador": "Ecuador_national_football_team",
    "Peru": "Peru_national_football_team",
    "Paraguay": "Paraguay_national_football_team",
    "Bolivia": "Bolivia_national_football_team",
    "Venezuela": "Venezuela_national_football_team",
    "Costa Rica": "Costa_Rica_national_football_team",
    "Panama": "Panama_national_football_team",
    "Algeria": "Algeria_national_football_team",
    "Tunisia": "Tunisia_national_football_team",
    "Cameroon": "Cameroon_national_football_team",
    "Mali": "Mali_national_football_team",
    "Congo DR": "DR_Congo_national_football_team",
    "Angola": "Angola_national_football_team",
    "Burkina Faso": "Burkina_Faso_national_football_team",
    "Guinea": "Guinea_national_football_team",
    "Senegal": "Senegal_national_football_team",
    "Ivory Coast": "Ivory_Coast_national_football_team",
    "Saudi Arabia": "Saudi_Arabia_national_football_team",
    "Qatar": "Qatar_national_football_team",
    "Iceland": "Iceland_national_football_team",
    "Slovakia": "Slovakia_national_football_team",
    "Slovenia": "Slovenia_national_football_team",
    "Albania": "Albania_national_football_team",
    "Finland": "Finland_national_football_team",
    "North Macedonia": "North_Macedonia_national_football_team",
    "Georgia": "Georgia_national_football_team",
    "Namibia": "Namibia_national_football_team",
    "South Africa": "South_Africa_national_football_team",
    "Zambia": "Zambia_national_football_team",
    "Jamaica": "Jamaica_national_football_team",
    "Tanzania": "Tanzania_national_football_team",
    "Mozambique": "Mozambique_national_football_team",
}


def get_wiki_thumbnail(article_title: str) -> str | None:
    """Fetch the thumbnail URL for a Wikipedia article."""
    url = WIKI_API.format(article_title.replace(" ", "_"))
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "xG-model-dashboard/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            thumb = data.get("thumbnail", {})
            src = thumb.get("source", "")
            if src:
                # Increase resolution: replace /320px with /100px for lighter load
                return src.replace("/320px-", "/100px-").replace("/200px-", "/100px-")
    except Exception:
        pass
    return None


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    team_counts = df.groupby("team").size().sort_values(ascending=False)
    teams = team_counts.index.tolist()

    existing = {}
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing.", flush=True)

    logos = dict(existing)

    to_fetch = [t for t in teams if t not in logos]
    print(f"Teams to fetch: {len(to_fetch)}", flush=True)

    for i, team in enumerate(to_fetch):
        # Determine Wikipedia article title
        clean = team.replace("á", "a").replace("é", "e").replace("í", "i") \
                    .replace("ó", "o").replace("ú", "u").replace("ü", "u") \
                    .replace("ñ", "n").replace("è", "e").replace("à", "a") \
                    .replace("â", "a").replace("ô", "o")

        article = WIKI_OVERRIDES.get(team) or WIKI_OVERRIDES.get(clean)
        if not article:
            article = team.replace(" ", "_")

        url = get_wiki_thumbnail(article)
        logos[team] = url or ""
        status = "OK" if url else "NOT FOUND"
        print(f"  [{i+1}/{len(to_fetch)}] {team[:35]:<35} {status}", flush=True)
        time.sleep(0.15)

    found = sum(1 for v in logos.values() if v)
    print(f"\nTotal: {len(logos)} teams, {found} with logos.", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(logos, f, ensure_ascii=False, indent=2)
    print(f"Saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
