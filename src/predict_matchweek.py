import pandas as pd
import joblib
import numpy as np
import io
import requests
import os

CONFIDENCE_THRESHOLD = 0.4
MODEL_PATH = "models/xgboost_betting_model.pkl"
FEATURES_PATH = "models/model_features.pkl"
HISTORICAL_DATA_PATH = "data/cleaned_historical_data.csv"

RESULT_MAP = {0: "AWAY WIN", 1: "HOME WIN", 2: "DRAW"}

TEAM_MAPPING = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "PSG": "Paris SG",
    "Bayern Munich": "Bayern Munich",
    "RB Leipzig": "RB Leipzig",
    "Dortmund": "Dortmund",
    "Leverkusen": "Bayer Leverkusen",
    "M'gladbach": "Gladbach",
}


def fetch_fixtures():
    print("🌍 Fetching fixtures...")
    url = "https://www.football-data.co.uk/fixtures.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(response.text), encoding="unicode_escape")
        if "ï»¿Div" in df.columns:
            df.rename(columns={"ï»¿Div": "Div"}, inplace=True)
        return df[df["Div"].isin(["E0", "SP1", "D1", "I1", "F1"])].copy()
    except Exception as e:
        print(f" API Error: {e}")
        return None


def engineer_features(fixtures_df, history_df, expected_features):
    print("⚙️ Engineering features...")

    history_df["home_team"] = history_df["home_team"].str.strip().str.lower()
    history_df["away_team"] = history_df["away_team"].str.strip().str.lower()

    history_df["home_points"] = history_df["match_result"].map({"H": 3, "D": 1, "A": 0})
    history_df["away_points"] = history_df["match_result"].map({"H": 0, "D": 1, "A": 3})
    history_df["home_bal_1h"] = (
        history_df["ht_home_goals"] - history_df["ht_away_goals"]
    )
    history_df["away_bal_1h"] = -history_df["home_bal_1h"]

    processed_rows = []

    for _, match in fixtures_df.iterrows():
        home_t = TEAM_MAPPING.get(match["HomeTeam"], match["HomeTeam"]).strip().lower()
        away_t = TEAM_MAPPING.get(match["AwayTeam"], match["AwayTeam"]).strip().lower()

        past_home = history_df[history_df["home_team"] == home_t].tail(5)
        past_away = history_df[history_df["away_team"] == away_t].tail(5)

        match_stats = {
            "B365H": match.get("B365H", 0),
            "B365D": match.get("B365D", 0),
            "B365A": match.get("B365A", 0),
            "home_team_goals_avg_last_5": past_home["home_goals"].mean(),
            "away_goals_avg_last_5": past_away["away_goals"].mean(),
            "ht_home_goals_avg_last_5": past_home["ht_home_goals"].mean(),
            "ht_away_goals_avg_last_5": past_away["ht_away_goals"].mean(),
            "home_form_1h_last_5": past_home["home_bal_1h"].mean(),
            "away_form_1h_last_5": past_away["away_bal_1h"].mean(),
            "home_points_avg_last_5": past_home["home_points"].mean(),
            "away_points_avg_last_5": past_away["away_points"].mean(),
            "home_overall_points_last_5": past_home["home_points"].mean(),
            "away_overall_points_last_5": past_away["away_points"].mean(),
        }
        processed_rows.append(match_stats)

    df_processed = pd.DataFrame(processed_rows).fillna(0)
    for col in expected_features:
        if col not in df_processed.columns:
            df_processed[col] = 0

    return df_processed[expected_features]


def run_predictions():
    if not os.path.exists(MODEL_PATH):
        print(f" Model not found at {MODEL_PATH}. Run from project root!")
        return

    print("Loading model & data...")
    xgb_model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    history_df = pd.read_csv(HISTORICAL_DATA_PATH)

    upcoming = fetch_fixtures()
    if upcoming is None:
        return

    ready_df = engineer_features(upcoming, history_df, features)

    print("🔮 Running predictions...")
    probs = xgb_model.predict_proba(ready_df)
    upcoming["Confidence"] = np.max(probs, axis=1)
    upcoming["Pick_ID"] = xgb_model.predict(ready_df)
    upcoming["Pick"] = upcoming["Pick_ID"].map(RESULT_MAP)

    picks = upcoming[upcoming["Confidence"] >= CONFIDENCE_THRESHOLD].copy()

    print("\n" + "=" * 95)
    print(f"EXPERT PICKS (Confidence > {CONFIDENCE_THRESHOLD*100}%)")
    print("=" * 95)

    if picks.empty:
        print("No matches met the threshold.")
    else:
        print(
            picks[
                ["Date", "HomeTeam", "AwayTeam", "Pick", "Confidence", "B365H", "B365A"]
            ]
        )


if __name__ == "__main__":
    run_predictions()
