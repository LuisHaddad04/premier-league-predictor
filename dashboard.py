
import streamlit as st
from pathlib import Path
import pandas as pd
import textwrap


BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_player_stats():
    file_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "premier_league_player_stats_2026_27.csv"
    )

    return pd.read_csv(file_path)

player_stats_df = load_player_stats()


# Load Premier League 2026/27 fixtures
@st.cache_data
def load_fixtures():
    file_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "premier_league_fixtures_2026_27.csv"
    )

    df = pd.read_csv(file_path)

    df["utc_date"] = pd.to_datetime(
        df["utc_date"],
        utc=True,
        errors="coerce"
    )

    return df


fixtures_df = load_fixtures()

print("Fixtures loaded:", len(fixtures_df))
print(fixtures_df["status"].value_counts())


@st.cache_data
def load_player_photos():
    file_path = BASE_DIR / "data" / "processed" / "premier_league_player_photos_2026_27.csv"
    return pd.read_csv(file_path)


player_photos_df = load_player_photos()

PLAYER_PHOTOS = dict(
    zip(
        player_photos_df["player_name"],
        player_photos_df["photo_url"]
    )
)

print("Player photos loaded:", len(PLAYER_PHOTOS))


# Club crest URLs (public football-data.org image assets)
TEAM_CRESTS = {
    "Arsenal": "https://crests.football-data.org/57.png",
    "Bournemouth": "https://crests.football-data.org/1044.png",
    "Aston Villa": "https://crests.football-data.org/58.png",
    "Chelsea": "https://crests.football-data.org/61.png",
    "Liverpool": "https://crests.football-data.org/64.png",
    "Manchester City": "https://crests.football-data.org/65.png",
    "Manchester United": "https://crests.football-data.org/66.png",
    "Tottenham Hotspur": "https://crests.football-data.org/73.png",
    "Everton": "https://crests.football-data.org/62.png",
    "Newcastle United": "https://crests.football-data.org/67.png",
    "West Ham United": "https://crests.football-data.org/563.png",
    "Brighton & Hove Albion": "https://crests.football-data.org/397.png",
    "Brentford": "https://crests.football-data.org/402.png",
    "Crystal Palace": "https://crests.football-data.org/354.png",
    "Nottingham Forest": "https://crests.football-data.org/351.png",
    "Fulham": "https://crests.football-data.org/63.png",
    "Wolverhampton Wanderers": "https://crests.football-data.org/76.png",
    "Leeds United": "https://crests.football-data.org/341.png",
    "Sunderland": "https://crests.football-data.org/71.png",
    "Ipswich Town": "https://crests.football-data.org/349.png",
    "Coventry City": "https://crests.football-data.org/1076.png",
    "Hull City": "https://crests.football-data.org/322.png",
}

TEAM_ID_TO_NAME = {
    "tm_0256": "Leeds United",
    "tm_0406": "Liverpool",
    "tm_0836": "Nottingham Forest",
    "tm_1002": "Aston Villa",
    "tm_1552": "Hull City",
    "tm_1662": "Brentford",
    "tm_1772": "Bournemouth",
    "tm_2674": "Tottenham Hotspur",
    "tm_2954": "Coventry City",
    "tm_3039": "Manchester City",
    "tm_3498": "Crystal Palace",
    "tm_4643": "Manchester United",
    "tm_5130": "Brighton & Hove Albion",
    "tm_5290": "Fulham",
    "tm_6115": "Sunderland",
    "tm_7938": "Newcastle United",
    "tm_8091": "Everton",
    "tm_8531": "Chelsea",
    "tm_9145": "Arsenal",
    "tm_9825": "Ipswich Town",
}




# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BACKTEST_PATH = PROCESSED_DIR / "rf_v1_2026_27_backtest.csv"

# Load saved historical predictions
backtest = pd.read_csv(BACKTEST_PATH)
backtest["date"] = pd.to_datetime(backtest["date"], utc=True)

# Most recent matches first
backtest = backtest.sort_values("date", ascending=False)



# Load historical match statistics
HISTORICAL_PATH = PROCESSED_DIR / "historical_matches.csv"

historical = pd.read_csv(HISTORICAL_PATH)
historical["date"] = pd.to_datetime(historical["date"], utc=True)

# Only Premier League matches
pl_history = historical[
    historical["competition"] == "premier_league"
].copy()

pl_history = pl_history.sort_values("date")


def get_team_form(team, match_date, last_n=5):
    """Calculate team statistics using matches before the selected date."""

    past = historical[
        (historical["date"] < match_date)
        & (
            (historical["home_team"] == team)
            | (historical["away_team"] == team)
        )
    ].copy()

    past = past.sort_values("date").tail(last_n)

    if past.empty:
        return None

    records = []

    for _, game in past.iterrows():
        is_home = game["home_team"] == team

        prefix = "home" if is_home else "away"

        goals_for = game[f"{prefix}_goals"]
        goals_against = game[
            "away_goals" if is_home else "home_goals"
        ]

        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"

        records.append({
            "result": result,
            "goals": goals_for,
            "xg": game[f"{prefix}_xg"],
            "shots": game[f"{prefix}_shots"],
            "possession": game[f"{prefix}_possession"]
        })

    df = pd.DataFrame(records)

    return {
        "form": df["result"].tolist(),
        "goals": df["goals"].mean(),
        "xg": df["xg"].mean(),
        "shots": df["shots"].mean(),
        "possession": df["possession"].mean(),
        "matches": len(df)
    }



def calculate_standings(matches):
    """Calculate league standings from completed match results."""

    standings = {}

    for _, game in matches.iterrows():
        home = game["home_team"]
        away = game["away_team"]

        home_goals = int(game["home_goals"])
        away_goals = int(game["away_goals"])

        for team in (home, away):
            if team not in standings:
                standings[team] = {
                    "Club": team,
                    "P": 0,
                    "W": 0,
                    "D": 0,
                    "L": 0,
                    "GF": 0,
                    "GA": 0,
                    "GD": 0,
                    "Pts": 0
                }

        h = standings[home]
        a = standings[away]

        h["P"] += 1
        a["P"] += 1

        h["GF"] += home_goals
        h["GA"] += away_goals

        a["GF"] += away_goals
        a["GA"] += home_goals

        if home_goals > away_goals:
            h["W"] += 1
            h["Pts"] += 3
            a["L"] += 1

        elif home_goals < away_goals:
            a["W"] += 1
            a["Pts"] += 3
            h["L"] += 1

        else:
            h["D"] += 1
            a["D"] += 1
            h["Pts"] += 1
            a["Pts"] += 1

    table = pd.DataFrame(standings.values())

    if table.empty:
        return table

    table["GD"] = table["GF"] - table["GA"]

    table = table.sort_values(
        ["Pts", "GD", "GF", "Club"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    table.insert(0, "#", range(1, len(table) + 1))

    return table


# Configure dashboard
st.set_page_config(
    page_title="Premier League Prediction Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


import base64

background_path = BASE_DIR / "assets" / "premier_league_background.png"

with open(background_path, "rb") as image_file:
    background_base64 = base64.b64encode(image_file.read()).decode()

st.markdown(
    f"""
    <style>
    :root {{
        --pl-background-image: url("data:image/png;base64,{background_base64}");
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        bottom: 0;
        left: 19%;
        right: 19%;
        background: #28002E;
        pointer-events: none;
        z-index: 0;
    }}

    [data-testid="stAppViewContainer"] > section {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Dashboard styling
st.markdown("""
<style>
    .stApp {
        background-color: #0B1020;
        color: #F3F4F6;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    h1, h2, h3 {
        color: #FFFFFF;
    }

    /* Model metrics */
    [data-testid="stMetric"] {
        background-color: #172033;
        padding: 14px;
        border-radius: 12px;
    }

    /* Center dashboard and limit width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Reduce excessive vertical spacing */
    [data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }

    /* Rounded dashboard containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }

    /* Modern dashboard typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stApp"] {
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, p, label, input {
        font-family: 'Inter', sans-serif !important;
    }

    /* Navigation buttons */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #29354D;
        background-color: #172033;
    }

    div.stButton > button:hover {
        border-color: #818CF8;
        color: #A5B4FC;
    }

    /* Match statistics expanders */
    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-size: 14px;
        font-weight: 600;
        color: #F3F4F6;
    }

    [data-testid="stExpander"] summary svg {
        flex-shrink: 0;
    }

    /* Standard dashboard buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #172033;
        border: 1px solid #29354D;
        border-radius: 10px;
        min-height: 52px;
        font-size: 13px;
        font-weight: 600;
        transition: border-color 0.2s, background-color 0.2s;
    }

    div.stButton > button[kind="primary"] {
        background-color: #222D4A;
        border: 1px solid #818CF8;
        border-radius: 10px;
        min-height: 52px;
        font-size: 13px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border-color: #818CF8;
        background-color: #222D4A;
    }

    /* FotMob-inspired match card action area */
    div[class*="st-key-team_match_"] {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-top: -1px;
        padding: 0 16px 14px !important;
        background: #172033;
        border: 1px solid #29354D;
        border-top: none;
        border-radius: 0 0 14px 14px;
        box-sizing: border-box;
    }

    /* Center Streamlit's internal button wrapper */
    div[class*="st-key-team_match_"] > div {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    /* Match statistics buttons only */
    div[class*="st-key-team_match_"] button {
        width: auto !important;
        min-height: 36px !important;
        padding: 0 22px !important;
        background: #222D4A !important;
        border: 1px solid #818CF8 !important;
        border-radius: 8px !important;
        color: #A5B4FC !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    div[class*="st-key-team_match_"] button:hover {
        background: #303F63 !important;
        border-color: #A5B4FC !important;
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    [data-testid="stMarkdownContainer"] p {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.2px;
    }

    /* Premier League official-style background */
    .stApp {
        background-color: #28002E !important;
        background-image: var(--pl-background-image) !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: #F8F5FF;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #1B112C !important;
    }

    /* Premier League-inspired leaderboard cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #251A3B !important;
        border-color: #49365E !important;
        border-radius: 12px !important;
    }

    /* Magenta accents for navigation and interactive elements */
    div[data-testid="stButton"] button:hover {
        border-color: #E90083 !important;
        color: #E90083 !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #E90083 !important;
        border-color: #E90083 !important;
        color: white !important;
    }

    /* Magenta top accent on bordered Streamlit containers */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(
            145deg,
            #251A3B 0%,
            #1D142E 100%
        ) !important;

        border: 1px solid #49365E !important;
        border-top: 3px solid #E90083 !important;
        border-radius: 12px !important;
    }

</style>
""", unsafe_allow_html=True)




# ==========================================
# DASHBOARD HEADER & NAVIGATION
# ==========================================

st.title("⚽ Premier League Analytics")
st.caption("Match Predictions • Team & Player Insights • Data-Driven Analysis")

page = st.segmented_control(
    "Dashboard navigation",
    options=[
        "Overview",
        "Match Analysis",
        "Matches & Fixtures",
        "Teams",
        "Players",
        "Model Performance"
    ],
    default="Match Analysis",
    label_visibility="collapsed",
    key="dashboard_page"
)


st.divider()
st.divider()

# ==========================================
# DASHBOARD PAGE ROUTING
# ==========================================

if page == "Match Analysis":

    # Model performance
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Accuracy", "50.7%")

    with col2:
        st.metric("Model Log Loss", "1.0409")

    with col3:
        st.metric("Test Matches", "357")

    st.divider()



    # ==========================================
    # MATCH ANALYSIS DASHBOARD
    # ==========================================

    st.subheader("Match Analysis")
    st.caption("2026/27 Premier League | Historical model backtest")

    # Match selector
    selected_index = st.selectbox(
        "Select a match",
        options=backtest.index.tolist(),
        format_func=lambda i: (
            f"{backtest.loc[i, 'home_team']} vs "
            f"{backtest.loc[i, 'away_team']} — "
            f"{backtest.loc[i, 'date'].strftime('%b %d, %Y')}"
        )
    )

    match = backtest.loc[selected_index]

    home = match["home_team"]
    away = match["away_team"]

    home_prob = float(match["prob_H"])
    draw_prob = float(match["prob_D"])
    away_prob = float(match["prob_A"])


    # ==========================================
    # MATCH HEADER
    # ==========================================

    st.markdown("### Match Overview")

    with st.container(border=True):

        # Teams and final score
        
        col_home, col_score, col_away = st.columns(
            [1, 1, 1],
            vertical_alignment="center"
        )

        with col_home:
            home_crest = TEAM_CRESTS.get(home)

            if home_crest:
                st.image(home_crest, width=70)

            st.markdown(f"### {home}")
            st.caption("HOME")

        with col_score:
            st.markdown(
                f"<h1 style='text-align:center;font-size:42px;'>"
                f"{int(match['home_goals'])} - {int(match['away_goals'])}"
                f"</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align:center;color:#94A3B8;'>"
                "FINAL SCORE</p>",
                unsafe_allow_html=True
            )

        with col_away:
            away_crest = TEAM_CRESTS.get(away)

            if away_crest:
                st.image(away_crest, width=70)

            st.markdown(f"### {away}")
            st.caption("AWAY")



        st.divider()

        st.markdown("**Pre-Match Model Probabilities**")

        # Actual probabilities from the saved backtest
        p_home = max(0.0, min(1.0, home_prob))
        p_draw = max(0.0, min(1.0, draw_prob))
        p_away = max(0.0, min(1.0, away_prob))

        # Compact probability display
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Home Win", f"{p_home:.1%}")

        with c2:
            st.metric("Draw", f"{p_draw:.1%}")

        with c3:
            st.metric("Away Win", f"{p_away:.1%}")

        # Probability bars
        
        # Compact three-color probability distribution
        st.markdown("**Prediction Breakdown**")

        probability_html = (
            '<div style="display:flex;width:100%;height:16px;'
            'border-radius:8px;overflow:hidden;">'
            f'<div style="width:{p_home * 100:.4f}%;background:#818CF8;"></div>'
            f'<div style="width:{p_draw * 100:.4f}%;background:#F59E0B;"></div>'
            f'<div style="width:{p_away * 100:.4f}%;background:#22D3EE;"></div>'
            '</div>'
        )

        st.markdown(probability_html, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.caption(f"🟣 {home}: {p_home:.1%}")

        with c2:
            st.caption(f"🟠 Draw: {p_draw:.1%}")

        with c3:
            st.caption(f"🔵 {away}: {p_away:.1%}")



    # ==========================================
    # TEAM FORM
    # ==========================================

    home_stats = get_team_form(home, match["date"])
    away_stats = get_team_form(away, match["date"])

    if home_stats and away_stats:

        def form_badges(results):
            colors = {
                "W": "#16A34A",
                "D": "#D97706",
                "L": "#DC2626"
            }

            return "".join(
                f'<span style="display:inline-block;'
                f'background:{colors[r]};'
                f'color:white;padding:5px 11px;'
                f'margin-right:5px;border-radius:6px;'
                f'font-weight:700;">{r}</span>'
                for r in results
            )

        left, right = st.columns(2)

        with left:
            st.markdown(
                textwrap.dedent(f"""
                <div style="background:#172033;padding:18px;border-radius:12px;">
                    <div style="font-weight:700;margin-bottom:12px;">
                        {home} | Recent Form
                    </div>
                    {form_badges(home_stats['form'])}
                </div>
                """),
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                textwrap.dedent(f"""
                <div style="background:#172033;padding:18px;border-radius:12px;">
                    <div style="font-weight:700;margin-bottom:12px;">
                        {away} | Recent Form
                    </div>
                    {form_badges(away_stats['form'])}
                </div>
                """),
                unsafe_allow_html=True
            )

        
        # ==========================================
        # TEAM PERFORMANCE COMPARISON
        # ==========================================

        st.markdown("### Pre-Match Team Comparison")
        st.caption("Last 5 available matches across all competitions")

        metrics = [
            ("Average Goals", "goals", 2),
            ("Average xG", "xg", 2),
            ("Average Shots", "shots", 1),
            ("Average Possession (%)", "possession", 1),
        ]

        with st.container(border=True):

            for label, key, decimals in metrics:
                h = float(home_stats[key])
                a = float(away_stats[key])

                total = h + a
                home_share = 0.5 if total == 0 else h / total

                col_home, col_label, col_away = st.columns([1, 3, 1])

                with col_home:
                    st.markdown(f"**{h:.{decimals}f}**")

                with col_label:
                    st.markdown(f"**{label}**")

                with col_away:
                    st.markdown(f"**{a:.{decimals}f}**")

                comparison_bar = (
                    '<div style="display:flex;width:100%;height:10px;'
                    'border-radius:8px;overflow:hidden;margin-bottom:12px;">'
                    f'<div style="width:{home_share * 100:.4f}%;background:#818CF8;"></div>'
                    f'<div style="width:{(1 - home_share) * 100:.4f}%;background:#22D3EE;"></div>'
                    '</div>'
                )

                st.markdown(comparison_bar, unsafe_allow_html=True)

                st.write("")


    else:
        st.warning("Insufficient historical data for team comparison.")




# ==========================================
# PREMIER LEAGUE OVERVIEW
# ==========================================





if page == "Overview":

    # Get the latest completed fixtures
    overview_completed = fixtures_df[
        fixtures_df["status"] == "finished"
    ].copy()

    total_matches = len(overview_completed)

    latest_date = overview_completed["utc_date"].max()

    latest_date_text = (
        latest_date.strftime("%B %d, %Y")
        if pd.notna(latest_date)
        else "No completed matches"
    )

    st.markdown("## Premier League Hub")
    st.caption(
        f"2026/27 Premier League | Data through {latest_date_text}"
    )

    season_matches = pl_history[
        pl_history["season"] == "2026_27"
    ].copy()

    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric("Matches Recorded", total_matches)

    with col2:
        st.metric("Season", "2026/27")

    with col3:
        st.metric("Data Status", "Current Season")

    st.divider()

    left, right = st.columns([2, 1])

    
    
    
    with left:
        st.markdown("### Premier League Standings")

        # Use the latest completed fixtures
        completed_fixtures = fixtures_df[
            (fixtures_df["status"] == "finished")
            & fixtures_df["home_score"].notna()
            & fixtures_df["away_score"].notna()
        ].copy()

        # Match the column names expected by calculate_standings()
        completed_fixtures = completed_fixtures.rename(
            columns={
                "home_score": "home_goals",
                "away_score": "away_goals"
            }
        )

        # Calculate updated league standings
        standings = calculate_standings(completed_fixtures)

        # Display the latest completed match date
        latest_match_date = completed_fixtures["utc_date"].max()

        st.caption(
            f"2026/27 season • Through {latest_match_date.strftime('%B %d, %Y')}"
            if pd.notna(latest_match_date)
            else "2026/27 season"
        )

                # Calculate 2026/27 team xG totals
        import json

        xg_totals = {}
        match_folder = Path("data/raw/matches/premier_league/2026_27")

        completed_ids = set(completed_fixtures["match_id"].astype(str))
        for match_file in match_folder.glob("*.json"):
            with open(match_file, "r", encoding="utf-8") as f:
                match_data = json.load(f)

            details = match_data.get("details", {}).get("data", {})
            stats = match_data.get("statistics", {}).get("data", {})

            if str(details.get("id")) not in completed_ids:
                continue

            home = details.get("home_team", {}).get("name")
            away = details.get("away_team", {}).get("name")

            xg = stats.get("overview", {}).get("expected_goals", {}).get("all", {})

            if not home or not away:
                continue

            for club, value in [(home, xg.get("home")), (away, xg.get("away"))]:
                if value is not None:
                    xg_totals[club] = xg_totals.get(club, 0) + float(value)

        standings["xG"] = standings["Club"].map(xg_totals)

        if not standings.empty:

            from html import escape

            # Football-style standings header
            header_html = (
                '<div style="display:grid;'
                'grid-template-columns:22px 24px minmax(0,1fr) 22px 30px 30px 128px;'
                'gap:5px;align-items:center;padding:12px 8px;'
                'color:#94A3B8;font-size:10px;font-weight:700;'
                'border-bottom:1px solid #334155;">'
                '<span>#</span><span></span><span>CLUB</span>'
                '<span>P</span><span>GD</span><span>PTS</span>'
                '<span>FORM</span></div>'
            )


            rows_html = ""

            for _, team in standings.iterrows():

                position = int(team["#"])
                club = str(team["Club"])
                crest = TEAM_CRESTS.get(club)


                
                # Get this club's five most recent completed league matches
                team_matches = completed_fixtures[
                    (completed_fixtures["home_team"] == club)
                    | (completed_fixtures["away_team"] == club)
                ].sort_values("utc_date").tail(5)

                recent_form = []

                for _, game in team_matches.iterrows():

                    is_home = game["home_team"] == club

                    goals_for = (
                        game["home_goals"] if is_home
                        else game["away_goals"]
                    )

                    goals_against = (
                        game["away_goals"] if is_home
                        else game["home_goals"]
                    )

                    if goals_for > goals_against:
                        recent_form.append("W")
                    elif goals_for < goals_against:
                        recent_form.append("L")
                    else:
                        recent_form.append("D")

                team_stats = {"form": recent_form}

                form_colors = {
                    "W": "#16A34A",
                    "D": "#D97706",
                    "L": "#DC2626"
                }

                form_html = ""

                if team_stats:
                    for result in team_stats["form"]:
                        form_html += (
                            f'<span style="display:inline-flex;'
                            f'align-items:center;justify-content:center;'
                            f'width:19px;height:19px;margin-right:3px;'
                            f'border-radius:4px;background:{form_colors[result]};'
                            f'color:white;font-size:10px;font-weight:700;">'
                            f'{result}</span>'
                        )

                # Position indicators
                if position <= 4:
                    indicator = "#818CF8"
                elif position >= 18:
                    indicator = "#EF4444"
                else:
                    indicator = "transparent"

                # Club crest
                crest_html = (
                    f'<img src="{escape(crest, quote=True)}" '
                    'style="width:23px;height:23px;object-fit:contain;">'
                    if crest else ""
                )

                # Individual team row
                
                rows_html += (
                    '<div style="display:grid;'
                    'grid-template-columns:22px 24px minmax(0,1fr) 22px 30px 30px 128px;'
                    'gap:5px;align-items:center;padding:10px 8px;'
                    'border-bottom:1px solid #263247;'
                    f'border-left:3px solid {indicator};font-size:11px;">'
                    f'<span style="color:#CBD5E1;">{position}</span>'
                    f'<span>{crest_html}</span>'
                    f'<span style="font-weight:600;overflow:hidden;'
                    f'text-overflow:ellipsis;white-space:nowrap;">'
                    f'{escape(club)}</span>'
                    f'<span>{int(team["P"])}</span>'
                    f'<span>{int(team["GD"]):+d}</span>'
                    f'<span style="font-weight:800;color:#FFFFFF;">'
                    f'{int(team["Pts"])}</span>'
                    f'<span style="white-space:nowrap;">{form_html}</span>'
                    '</div>'
                )


            # Combine the table into one HTML component
            standings_html = (
                '<div style="background:#172033;border-radius:14px;'
                'overflow:hidden;padding:6px;">'
                + header_html
                + rows_html
                + '</div>'
            )

            st.markdown(
                standings_html,
                unsafe_allow_html=True
            )

        else:
            st.warning("No completed Premier League matches found.")




    with right:
        st.markdown("### Team Leaders")

        leader_categories = [
            ("⚽ Top Attack", "GF", False, "goals"),
            ("🛡️ Best Defense", "GA", True, "conceded"),
            ("📊 Highest xG", "xG", False, "expected goals"),
        ]

        for title, column, ascending, suffix in leader_categories:
            st.markdown(f"**{title}**")

            leaders = standings.dropna(subset=[column]).sort_values(
                by=column,
                ascending=ascending
            ).head(3)

            for rank, (_, team) in enumerate(leaders.iterrows(), start=1):
                club = str(team["Club"])
                crest = TEAM_CRESTS.get(club, "")
                value = (
                    f'{team[column]:.2f}'
                    if column == "xG"
                    else str(int(team[column]))
                )

                crest_html = (
                    f'<img src="{crest}" style="width:24px;height:24px;object-fit:contain;">'
                    if crest else ""
                )

                row_html = (
                    '<div style="display:flex;align-items:center;gap:10px;'
                    'background:#172033;padding:11px 12px;'
                    'border-radius:9px;margin-bottom:5px;">'
                    f'<span style="color:#94A3B8;width:15px;">{rank}</span>'
                    f'{crest_html}'
                    f'<span style="flex:1;color:#F8FAFC;font-size:13px;font-weight:600;">{club}</span>'
                    f'<span style="color:#A5B4FC;font-weight:700;font-size:15px;">{value}</span>'
                    '</div>'
                )

                st.markdown(row_html, unsafe_allow_html=True)

            st.write("")


if page == "Teams":
    st.markdown("## Premier League Teams")
    st.caption("Explore club performance • 2026/27 season through August 31, 2026")

    team_standings = calculate_standings(
        pl_history[pl_history["season"] == "2026_27"].copy()
    )

    if not team_standings.empty:
        selected_team = st.selectbox(
            "Select a club",
            options=sorted(team_standings["Club"].tolist()),
            key="teams_club_selector"
        )

        team = team_standings[
            team_standings["Club"] == selected_team
        ].iloc[0]

        crest = TEAM_CRESTS.get(selected_team)

        col_crest, col_name = st.columns([1, 7], vertical_alignment="center")

        with col_crest:
            if crest:
                st.image(crest, width=75)

        with col_name:
            st.markdown(f"## {selected_team}")
            st.caption("Premier League · 2026/27")

        st.write("")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("League Position", f'{int(team["#"])}')

        with col2:
            st.metric("Points", int(team["Pts"]))

        with col3:
            st.metric("Matches Played", int(team["P"]))


        import json

        team_xg = 0.0
        xg_matches = 0

        match_folder = Path("data/raw/matches/premier_league/2026_27")

        completed_ids = set(
            pl_history.loc[
                pl_history["season"] == "2026_27", "match_id"
            ].astype(str)
        )

        for match_file in match_folder.glob("*.json"):
            with open(match_file, "r", encoding="utf-8") as f:
                match_data = json.load(f)

            details = match_data.get("details", {}).get("data", {})
            stats = match_data.get("statistics", {}).get("data", {})

            if str(details.get("id")) not in completed_ids:
                continue

            home = details.get("home_team", {}).get("name")
            away = details.get("away_team", {}).get("name")

            xg = stats.get("overview", {}).get("expected_goals", {}).get("all", {})

            if selected_team == home:
                value = xg.get("home")
            elif selected_team == away:
                value = xg.get("away")
            else:
                continue

            if value is not None:
                team_xg += float(value)
                xg_matches += 1


        st.markdown("### Team Performance")

        perf1, perf2, perf3, perf4 = st.columns(4)

        with perf1:
            st.metric("Goals Scored", int(team["GF"]))

        with perf2:
            st.metric("Goals Conceded", int(team["GA"]))

        with perf3:
            st.metric("Goal Difference", f'{int(team["GD"]):+d}')

        with perf4:
            win_rate = (
                int(team["W"]) / int(team["P"]) * 100
                if int(team["P"]) > 0 else 0
            )
            st.metric("Win Rate", f"{win_rate:.0f}%")


        if xg_matches > 0:
            st.markdown("### Expected Goals Analysis")

            xg1, xg2, xg3 = st.columns(3)

            with xg1:
                st.metric("Total xG", f"{team_xg:.2f}")

            with xg2:
                st.metric("xG per Match", f"{team_xg / xg_matches:.2f}")

            with xg3:
                goals_minus_xg = int(team["GF"]) - team_xg
                st.metric("Goals vs. xG", f"{goals_minus_xg:+.2f}")

            


        st.markdown("### Recent Premier League Results")
        st.caption("2026/27 season • Completed matches only")

        team_matches = pl_history[
            (pl_history["season"] == "2026_27")
            & (
                (pl_history["home_team"] == selected_team)
                | (pl_history["away_team"] == selected_team)
            )
        ].sort_values("date", ascending=False).head(5)

        if team_matches.empty:
            st.info("No completed Premier League matches available.")
        else:
            for _, match in team_matches.iterrows():
                home = match["home_team"]
                away = match["away_team"]
                home_goals = int(match["home_goals"])
                away_goals = int(match["away_goals"])

                is_home = home == selected_team
                goals_for = home_goals if is_home else away_goals
                goals_against = away_goals if is_home else home_goals

                result = (
                    "W" if goals_for > goals_against
                    else "D" if goals_for == goals_against
                    else "L"
                )

                result_colors = {
                    "W": "#16A34A",
                    "D": "#D97706",
                    "L": "#DC2626"
                }

                date_text = pd.to_datetime(match["date"]).strftime("%b %d")

                
                # FotMob-inspired match result card
                from html import escape

                match_id = str(match["match_id"])

                is_selected = (
                    st.session_state.get("selected_team_match") == match_id
                )

                home_crest = TEAM_CRESTS.get(home, "")
                away_crest = TEAM_CRESTS.get(away, "")

                result_labels = {
                    "W": "WIN",
                    "D": "DRAW",
                    "L": "LOSS"
                }

                border_color = "#818CF8" if is_selected else "#29354D"

                home_crest_html = (
                    f'<img src="{escape(home_crest, quote=True)}" '
                    'style="width:48px;height:48px;object-fit:contain;">'
                    if home_crest else ""
                )

                away_crest_html = (
                    f'<img src="{escape(away_crest, quote=True)}" '
                    'style="width:48px;height:48px;object-fit:contain;">'
                    if away_crest else ""
                )

                card_html = (
                    f'<div style="background:#172033;'
                    f'border:1px solid {border_color};'
                    'border-radius:14px 14px 0 0;padding:18px 22px;'
                    'margin-top:12px;">'

                    # Date and match status
                    '<div style="display:flex;align-items:center;'
                    'justify-content:space-between;margin-bottom:20px;">'
                    f'<span style="font-size:12px;color:#94A3B8;">'
                    f'{date_text.upper()}, 2026</span>'

                    '<div style="display:flex;align-items:center;gap:10px;">'
                    f'<span style="background:{result_colors[result]};'
                    'color:white;font-size:10px;font-weight:800;'
                    'padding:5px 10px;border-radius:5px;">'
                    f'{result_labels[result]}</span>'
                    '<span style="font-size:11px;color:#94A3B8;">'
                    'FULL TIME</span>'
                    '</div></div>'

                    # Teams and score
                    '<div style="display:grid;'
                    'grid-template-columns:1fr 130px 1fr;'
                    'align-items:center;gap:12px;">'

                    '<div style="display:flex;flex-direction:column;'
                    'align-items:center;gap:10px;text-align:center;">'
                    f'{home_crest_html}'
                    f'<span style="font-size:14px;font-weight:600;">'
                    f'{escape(home)}</span>'
                    '</div>'

                    '<div style="text-align:center;">'
                    '<div style="font-size:34px;font-weight:800;'
                    'color:#FFFFFF;white-space:nowrap;">'
                    f'{home_goals} - {away_goals}</div>'
                    '<div style="font-size:10px;color:#94A3B8;'
                    'margin-top:5px;letter-spacing:1px;">'
                    'FINAL SCORE</div>'
                    '</div>'

                    '<div style="display:flex;flex-direction:column;'
                    'align-items:center;gap:10px;text-align:center;">'
                    f'{away_crest_html}'
                    f'<span style="font-size:14px;font-weight:600;">'
                    f'{escape(away)}</span>'
                    '</div>'

                    '</div></div>'
                )

                
                # Render match card
                st.markdown(card_html, unsafe_allow_html=True)

                # Integrated match statistics action
                

                if st.button(
                    "📊 View Match Statistics"
                    if not is_selected
                    else "📊 Viewing Match Statistics",
                    key=f"team_match_{match_id}",
                    use_container_width=False,
                    type="primary" if is_selected else "secondary"
                ):
                    st.session_state["selected_team_match"] = match_id
                    st.rerun()



                # Detailed view for the selected match
        selected_match_id = st.session_state.get("selected_team_match")

        if selected_match_id:
            selected_rows = team_matches[
                team_matches["match_id"].astype(str) == selected_match_id
            ]

            if not selected_rows.empty:
                selected_match = selected_rows.iloc[0]

                home = selected_match["home_team"]
                away = selected_match["away_team"]
                home_goals = int(selected_match["home_goals"])
                away_goals = int(selected_match["away_goals"])

                st.divider()
                st.markdown("### Match Breakdown")

                st.caption(
                    pd.to_datetime(selected_match["date"]).strftime(
                        "%B %d, %Y"
                    )
                )

                home_col, score_col, away_col = st.columns(
                    [2, 1, 2],
                    vertical_alignment="center"
                )

                with home_col:
                    if TEAM_CRESTS.get(home):
                        st.image(TEAM_CRESTS[home], width=65)
                    st.markdown(f"**{home}**")

                with score_col:
                    st.markdown(
                        f"<h2 style='text-align:center;'>"
                        f"{home_goals} - {away_goals}</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<p style='text-align:center;color:#94A3B8;'>"
                        "Full Time</p>",
                        unsafe_allow_html=True
                    )

                with away_col:
                    if TEAM_CRESTS.get(away):
                        st.image(TEAM_CRESTS[away], width=65)
                    st.markdown(f"**{away}**")

                                # Load detailed statistics for the selected match
                import json

                match_file = (
                    Path("data/raw/matches/premier_league/2026_27")
                    / f"{selected_match_id}.json"
                )

                if match_file.exists():
                    with open(match_file, "r", encoding="utf-8") as f:
                        match_json = json.load(f)

                    stats = match_json.get("statistics", {}).get("data", {})

                    st.markdown("#### Match Statistics")

                    stat_categories = {
                        "Overview": [
                            ("Expected Goals (xG)", "overview", "expected_goals"),
                            ("Possession", "overview", "ball_possession"),
                            ("Big Chances", "overview", "big_chances"),
                            ("Total Shots", "shots", "total_shots"),
                            ("Shots on Target", "shots", "shots_on_target"),
                            ("Corner Kicks", "overview", "corner_kicks"),
                        ],
                        "Attacking": [
                            ("Shots Inside Box", "shots", "shots_inside_box"),
                            ("Shots Outside Box", "shots", "shots_outside_box"),
                            ("Blocked Shots", "shots", "blocked_shots"),
                            ("Big Chances Missed", "attack", "big_chances_missed"),
                            ("Touches in Penalty Area", "attack", "touches_in_penalty_area"),
                            ("Offsides", "attack", "offsides"),
                        ],
                        "Passing": [
                            ("Total Passes", "overview", "passes"),
                            ("Accurate Passes", "overview", "accurate_passes"),
                            ("Accurate Crosses", "passes", "accurate_crosses"),
                            ("Accurate Long Balls", "passes", "accurate_long_balls"),
                            ("Final Third Entries", "passes", "final_third_entries"),
                        ],
                        "Defending": [
                            ("Tackles", "defending", "tackles"),
                            ("Interceptions", "defending", "interceptions"),
                            ("Clearances", "defending", "clearances"),
                            ("Ball Recoveries", "defending", "ball_recoveries"),
                        ],
                        "Goalkeeping": [
                            ("Saves", "goalkeeping", "saves"),
                            ("Goals Prevented", "goalkeeping", "goals_prevented"),
                            ("High Claims", "goalkeeping", "high_claims"),
                        ],
                        "Discipline": [
                            ("Fouls", "overview", "fouls"),
                            ("Yellow Cards", "overview", "yellow_cards"),
                            ("Red Cards", "overview", "red_cards"),
                        ],
                    }

                    for category, fields in stat_categories.items():
                        with st.expander(
                            f"📊 {category}",
                            expanded=(category == "Overview")
                        ):

                            for label, section, field in fields:
                                section_data = stats.get(section) or {}
                                field_data = section_data.get(field) or {}
                                values = field_data.get("all") or {}

                                home_value = values.get("home")
                                away_value = values.get("away")

                                if home_value is None or away_value is None:
                                    continue

                                # Calculate each team's share of the statistic
                                try:
                                    home_num = float(home_value)
                                    away_num = float(away_value)
                                except (ValueError, TypeError):
                                    continue

                                # Only use proportional bars for nonnegative values
                                home_share = max(home_num, 0)
                                away_share = max(away_num, 0)
                                total = home_share + away_share

                                if total > 0:
                                    home_pct = home_share / total * 100
                                    away_pct = away_share / total * 100
                                else:
                                    home_pct = 50
                                    away_pct = 50

                                # Format values
                                if label == "Possession":
                                    home_display = f"{home_value}%"
                                    away_display = f"{away_value}%"
                                else:
                                    home_display = str(home_value)
                                    away_display = str(away_value)

                                # Head-to-head comparison
                                stat_html = (
                                    '<div style="padding:10px 4px 14px;">'

                                    '<div style="display:flex;'
                                    'align-items:center;justify-content:space-between;'
                                    'gap:12px;margin-bottom:9px;">'

                                    f'<span style="color:#A5B4FC;'
                                    f'font-weight:700;font-size:14px;">'
                                    f'{home_display}</span>'

                                    f'<span style="color:#CBD5E1;'
                                    f'font-size:12px;text-align:center;">'
                                    f'{label}</span>'

                                    f'<span style="color:#F8FAFC;'
                                    f'font-weight:700;font-size:14px;">'
                                    f'{away_display}</span>'

                                    '</div>'

                                    '<div style="display:flex;gap:4px;'
                                    'height:7px;width:100%;'
                                    'border-radius:6px;overflow:hidden;">'

                                    f'<div style="width:{home_pct:.2f}%;'
                                    f'background:#818CF8;"></div>'

                                    f'<div style="width:{away_pct:.2f}%;'
                                    f'background:#475569;"></div>'

                                    '</div>'
                                    '</div>'
                                )

                                st.markdown(
                                    stat_html,
                                    unsafe_allow_html=True
                                )

                else:
                    st.warning("Detailed statistics unavailable for this match.")


    else:
        st.warning("No team data available for this season.")



elif page == "Players":

    st.markdown("## ⚽ Player Statistics")
    st.caption("Premier League 2026/27 | Individual Player Performance")

    # League-wide player statistics
    league_players = player_stats_df[
        player_stats_df["minutes_played"].fillna(0) > 0
    ].copy()

    league_summary = (
        league_players.groupby(
            ["player_id", "player_name"],
            as_index=False
        )
        .agg(
            appearances=("match_id", "nunique"),
            goals=("shooting_goals", "sum"),
            assists=("passing_assists", "sum"),
            xg=("shooting_expected_goals", "sum"),
            xa=("shooting_expected_assists", "sum"),
            minutes=("minutes_played", "sum"),
            shots_on_target=("shooting_shots_on_target", "sum"),
            tackles=("defending_tackles", "sum"),
            saves=("goalkeeping_saves", "sum"),
            shots=("shooting_total_shots", "sum"),
            big_chances=("shooting_big_chances_created", "sum"),
            key_passes=("passing_key_passes", "sum"),
            accurate_passes=("passing_accurate_passes", "sum"),
            total_passes=("passing_total_passes", "sum"),
            interceptions=("defending_interceptions", "sum"),
            clearances=("defending_clearances", "sum"),
            ball_recoveries=("defending_ball_recoveries", "sum"),
            fouls=("general_fouls", "sum"),
            fouls_drawn=("general_was_fouled", "sum"),
            yellow_cards=("general_yellow_cards", "sum"),
            red_cards=("general_red_cards", "sum"),
            avg_rating=("rating", "mean"),
        )
    )

    # Identify each player's most recent club in our dataset
    player_clubs = (
        league_players.groupby("player_id", as_index=False)
        .agg(team_id=("team_id", "last"))
    )

    check_names = [
        "Pascal Groß",
        "Antoine Semenyo",
        "Evanilson",
        "Erling Haaland",
    ]

    print("\n=== PLAYER TEAM ASSIGNMENTS ===")

    print(
        league_players.loc[
            league_players["player_name"].isin(check_names),
            ["player_name", "team_id", "match_id"]
        ]
        .drop_duplicates()
        .sort_values(["player_name", "match_id"])
        .to_string(index=False)
    )

    # Attach club IDs without changing leaderboard statistics
    league_summary = league_summary.merge(
        player_clubs,
        on="player_id",
        how="left",
        validate="one_to_one"
    )

    league_summary["club_name"] = (
        league_summary["team_id"].map(TEAM_ID_TO_NAME)
    )

    league_summary["club_crest"] = (
        league_summary["club_name"].map(TEAM_CRESTS)
    )

    print(
        league_summary.loc[
            league_summary["player_name"].isin(
                ["Erling Haaland", "Declan Rice", "Cody Gakpo"]
            ),
            ["player_name", "club_name", "club_crest"]
        ].to_string(index=False)
    )

    print(
        league_summary[
            ["player_name", "team_id"]
        ].head(15).to_string(index=False)
    )

    # Temporary check: inspect player-to-club IDs
    check_players = [
        "Erling Haaland",
        "Alexander Isak",
        "Declan Rice",
        "Bukayo Saka",
        "Cody Gakpo",
        "Cole Palmer",
        "Bruno Fernandes",
        "Alex Iwobi",
    ]

    print("\n=== ALL PREMIER LEAGUE CLUB IDS ===")

    print(
        league_players[
            ["team_id", "player_name"]
        ]
        .dropna(subset=["team_id"])
        .drop_duplicates(subset=["team_id"])
        .sort_values("team_id")
        .to_string(index=False)
    )

    print(
        league_summary.loc[
            league_summary["player_name"].isin(check_players),
            ["player_name", "team_id"]
        ].sort_values("player_name").to_string(index=False)
    )

    # Configure the initial leaderboard categories
    leaderboards = {
        "⚽ Goals": ("goals", "Goals"),
        "🎯 Assists": ("assists", "Assists"),
        "📈 Expected Goals": ("xg", "xG"),
        "🅰️ Expected Assists": ("xa", "xA"),
        "🥅 Shots on Target": ("shots_on_target", "Shots"),
        "💥 Total Shots": ("shots", "Shots"),
        "🎯 Big Chances Created": ("big_chances", "Chances"),
        "🔑 Key Passes": ("key_passes", "Passes"),
        "📬 Accurate Passes": ("accurate_passes", "Passes"),
        "🛡️ Tackles": ("tackles", "Tackles"),
        "🧱 Interceptions": ("interceptions", "Interceptions"),
        "🚧 Clearances": ("clearances", "Clearances"),
        "🔄 Ball Recoveries": ("ball_recoveries", "Recoveries"),
        "🟨 Yellow Cards": ("yellow_cards", "Cards"),
        "🧤 Goalkeeper Saves": ("saves", "Saves"),
        "⏱️ Minutes Played": ("minutes", "Minutes")
    }

    st.markdown("### 🏆 Premier League Leaders")

    left_col, right_col = st.columns(2)

    for i, (title, (stat_column, stat_label)) in enumerate(
        leaderboards.items()
    ):
        target_col = left_col if i % 2 == 0 else right_col

        with target_col:
            with st.container(border=True):

                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        border-left: 4px solid #E90083;
                        padding: 5px 0 5px 12px;
                        margin-bottom: 12px;
                    ">
                        <span style="
                            color: #FFFFFF;
                            font-family: Inter, sans-serif;
                            font-size: 20px;
                            font-weight: 700;
                        ">{title}</span>
                        <span style="
                            color: #E90083;
                            font-size: 11px;
                            font-weight: 700;
                            letter-spacing: 1px;
                        ">TOP 5</span>
                    </div>
                    """.strip(),
                    unsafe_allow_html=True
                )

                top_players = (
                    league_summary
                    .sort_values(
                        by=[stat_column, "player_name"],
                        ascending=[False, True]
                    )
                    .head(5)
                )

                for rank, (_, player) in enumerate(
                    top_players.iterrows(),
                    start=1
                ):

                    # Create three columns: Photo | Player Name | Statistic
                    photo_col, crest_col, name_col, value_col = st.columns(
                        [0.8, 0.6, 4, 0.8],
                        gap="small",
                        vertical_alignment="center"
                    )

                    with crest_col:
                        crest_url = player.get("club_crest")

                        if pd.notna(crest_url) and crest_url:
                            st.image(crest_url, width=28)

                    with photo_col:
                        photo_url = PLAYER_PHOTOS.get(player["player_name"])

                        if photo_url:
                            st.image(photo_url, width=45)
                        else:
                            st.markdown("👤")

                    with name_col:
                        club_name = player.get("club_name")

                        if pd.isna(club_name):
                            club_name = "Club unavailable"

                        st.markdown(
                            f'<div style="font-family: Inter, sans-serif; '
                            f'font-size: 14px; font-weight: 600; '
                            f'color: #F8FAFC; line-height: 1.4;">'
                            f'{rank}. {player["player_name"]}'
                            f'</div>'
                            f'<div style="font-family: Inter, sans-serif; '
                            f'font-size: 12px; font-weight: 400; '
                            f'color: #BFAFD0; margin-top: 3px;">'
                            f'{club_name}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with value_col:
                        value = player[stat_column]

                        formatted_value = (
                            f"{value:.2f}"
                            if stat_column in ["xg", "xa"]
                            else f"{value:.0f}"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                color: {'#52E5C3' if rank == 1 else '#F8F5FF'};
                                font-family: 'Inter', sans-serif;
                                font-size: 17px;
                                font-weight: 700;
                                text-align: center;
                            ">
                                {formatted_value}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    if rank < 5:
                        st.markdown(
                            "<hr style='margin: 6px 0; border: 0; "
                            "border-top: 1px solid #334155;'>",
                            unsafe_allow_html=True
                        )

    # Include players who have appeared in at least one match
    players = player_stats_df[
        player_stats_df["minutes_played"].fillna(0) > 0
    ].copy()

    # Search and filtering controls
    search_col, team_col = st.columns([2, 1])

    with search_col:
        player_search = st.text_input(
            "Search Players",
            placeholder="Search by player name...",
            key="player_search"
        )

    with team_col:
        team_options = sorted(
            players["team_id"].dropna().unique().tolist()
        )

        selected_team = st.selectbox(
            "Filter by Team",
            ["All Teams"] + team_options,
            key="player_team_filter"
        )

    # Apply player name filter
    if player_search:
        players = players[
            players["player_name"].str.contains(
                player_search,
                case=False,
                na=False,
                regex=False
            )
        ]

    # Apply team filter
    if selected_team != "All Teams":
        players = players[
            players["team_id"] == selected_team
        ]

    st.caption(f"{players['player_id'].nunique()} players found")

    # Calculate season statistics for matching players
    player_summary = (
        players.groupby(
            ["player_id", "player_name", "team_id", "position"],
            as_index=False
        )
        .agg(
            appearances=("match_id", "nunique"),
            minutes=("minutes_played", "sum"),
            goals=("shooting_goals", "sum"),
            assists=("passing_assists", "sum"),
            shots=("shooting_total_shots", "sum"),
            shots_on_target=("shooting_shots_on_target", "sum"),
            xg=("shooting_expected_goals", "sum"),
            xa=("shooting_expected_assists", "sum"),
            accurate_passes=("passing_accurate_passes", "sum"),
            total_passes=("passing_total_passes", "sum"),
            avg_rating=("rating", "mean")
        )
    )

    # Calculate season passing accuracy
    player_summary["pass_accuracy"] = (
        player_summary["accurate_passes"]
        / player_summary["total_passes"].replace(0, float("nan"))
        * 100
    )

    # Temporarily display the summary for verification
    # Display an individual player profile when one player matches
    if player_summary["player_id"].nunique() == 1:

        p = player_summary.iloc[0]

        # Use the existing player-photo database
        player_photo = PLAYER_PHOTOS.get(p["player_name"])

        if player_photo:
            st.image(player_photo, width=110)

        st.markdown("---")
        st.markdown(f"### {p['player_name']}")

        st.caption(
            f"Position: {p['position']} | "
            f"Premier League 2026/27"
        )

        # Primary statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Appearances", f"{p['appearances']:.0f}")
            st.metric("Goals", f"{p['goals']:.0f}")

        with col2:
            st.metric("Minutes Played", f"{p['minutes']:.0f}")
            st.metric("Assists", f"{p['assists']:.0f}")

        with col3:
            st.metric("Expected Goals (xG)", f"{p['xg']:.2f}")
            st.metric("Expected Assists (xA)", f"{p['xa']:.2f}")

        st.markdown("#### Shooting & Passing")

        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric("Total Shots", f"{p['shots']:.0f}")

        with col5:
            st.metric("Shots on Target", f"{p['shots_on_target']:.0f}")

        with col6:
            pass_accuracy = p["pass_accuracy"]

            st.metric(
                "Pass Accuracy",
                f"{pass_accuracy:.1f}%"
                if pd.notna(pass_accuracy)
                else "N/A"
            )

    else:
        st.dataframe(
            player_summary[
                [
                    "player_name",
                    "appearances",
                    "goals",
                    "assists",
                    "xg",
                    "xa",
                    "avg_rating"
                ]
            ],
            width="stretch",
            hide_index=True
        )



elif page == "Matches & Fixtures":

    st.markdown("## ⚽ Matches & Fixtures")
    st.caption("Premier League 2026/27 | Results and Upcoming Matches")

    # Matchweek selector
    available_matchweeks = sorted(
        fixtures_df["matchday"].dropna().astype(int).unique().tolist()
    )

    # Identify the next scheduled fixture
    upcoming_fixtures = fixtures_df[
        (fixtures_df["status"] == "scheduled")
        & (fixtures_df["utc_date"] >= pd.Timestamp.now(tz="UTC"))
    ].sort_values("utc_date")

    if not upcoming_fixtures.empty:
        default_matchweek = int(upcoming_fixtures.iloc[0]["matchday"])
    else:
        # Fall back to the latest matchweek with completed results
        finished_fixtures = fixtures_df[
            fixtures_df["status"] == "finished"
        ]

        default_matchweek = (
            int(finished_fixtures["matchday"].max())
            if not finished_fixtures.empty
            else available_matchweeks[0]
        )

    selected_matchweek = st.selectbox(
        "Select Matchweek",
        available_matchweeks,
        index=available_matchweeks.index(default_matchweek),
        key="fixtures_matchweek"
    )

    # Filter fixtures to selected matchweek
    matchweek_fixtures = fixtures_df[
        fixtures_df["matchday"] == selected_matchweek
    ].sort_values("utc_date")

    st.markdown(f"### Matchweek {selected_matchweek}")

    for _, match in matchweek_fixtures.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        home_crest = TEAM_CRESTS.get(home_team)
        away_crest = TEAM_CRESTS.get(away_team)

        status = match["status"]
        match_date = match["utc_date"]

        if status == "finished":
            score_text = (
                f"{int(match['home_score'])} - "
                f"{int(match['away_score'])}"
            )
            status_text = "FT"
            score_color = "#52E5C3"

        elif status == "postponed":
            score_text = "POSTPONED"
            status_text = "PPD"
            score_color = "#E90083"

        else:
            score_text = match_date.strftime("%H:%M")
            status_text = "UPCOMING"
            score_color = "#FFFFFF"

        with st.container(border=True):

            home_col, score_col, away_col = st.columns(
                [4, 1.5, 4],
                vertical_alignment="center"
            )

            with home_col:
                crest_col, name_col = st.columns(
                    [1, 4],
                    vertical_alignment="center",
                    gap="small"
                )

                with crest_col:
                    if home_crest:
                        st.image(home_crest, width=34)

                with name_col:
                    st.markdown(f"**{home_team}**")

            with score_col:
                st.markdown(
                    f'<div style="text-align:center;'
                    f'font-size:19px;font-weight:700;'
                    f'color:{score_color};">'
                    f'{score_text}</div>'
                    f'<div style="text-align:center;'
                    f'font-size:10px;letter-spacing:1px;'
                    f'color:#BFAFD0;margin-top:3px;">'
                    f'{status_text}</div>',
                    unsafe_allow_html=True
                )

            with away_col:
                name_col, crest_col = st.columns(
                    [4, 1],
                    vertical_alignment="center",
                    gap="small"
                )

                with name_col:
                    st.markdown(
                        f'<div style="text-align:right;'
                        f'font-weight:600;">'
                        f'{away_team}</div>',
                        unsafe_allow_html=True
                    )

                with crest_col:
                    if away_crest:
                        st.image(away_crest, width=34)