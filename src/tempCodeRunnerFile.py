import pandas as pd
import joblib
import numpy as np
import io
import requests

CONFIDENCE_THRESHOLD = 0.4
MODEL_PATH = 'models/xgboost_betting_model.pkl'
FEATURES_PATH = 'models/model_features.pkl'
HISTORICAL_DATA_PATH = 'data/cleaned_historical_data.csv'

TEAM_MAPPING = {
    'Man Utd': 'Man United',
    'Spurs': 'Tottenham',
    'PSG': 'Paris SG',
    'Bayern Munich': 'Bayern',
    'Union Berlin': 'Union Berlin',
    'RB Leipzig': 'RB Leipzig',
    'Hoffenheim': 'Hoffenheim'
}

def fetch_weekend_fixtures_api():
    print("🌍 Fetching Top 5 leagues fixtures...")
    url = "https://www.football-data.co.uk/fixtures.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df_fixtures = pd.read_csv(io.StringIO(response.text), encoding='unicode_escape')
    except Exception as e:
        print(f"❌ Fetching error: {e}")
        return None

    if 'ï»¿Div' in df_fixtures.columns:
        df_fixtures.rename(columns={'ï»¿Div': 'Div'}, inplace=True)
    
    top_5_leagues = ['E0', 'SP1', 'D1', 'I1', 'F1']
    df_top5 = df_fixtures[df_fixtures['Div'].isin(top_5_leagues)].copy()
    print(f"✅ Found {len(df_top5)} matches.")
    return df_top5

def engineer_features_on_the_fly(fixtures_df, history_df, expected_features):
    print("⚙️ Engineering features...")
    
    # Verify columns in historical data
    print(f"--- DATA CHECK: Historical file columns: {history_df.columns.tolist()[:10]}... ---")

    processed_rows = []
    
    if 'home_balance_1h' not in history_df.columns:
        history_df['home_balance_1h'] = history_df['ht_home_goals'] - history_df['ht_away_goals']
        history_df['home_balance_2h'] = (history_df['home_goals'] - history_df['ht_home_goals']) - (history_df['away_goals'] - history_df['ht_away_goals'])
        history_df['away_balance_1h'] = -history_df['home_balance_1h']
        history_df['away_balance_2h'] = -history_df['home_balance_2h']
        history_df['home_points'] = history_df['match_result'].map({'H': 3, 'D': 1, 'A': 0})
        history_df['away_points'] = history_df['match_result'].map({'H': 0, 'D': 1, 'A': 3})

    for index, match in fixtures_df.iterrows():
        home_t = TEAM_MAPPING.get(match['HomeTeam'], match['HomeTeam'])
        away_t = TEAM_MAPPING.get(match['AwayTeam'], match['AwayTeam'])
        
        past_home = history_df[history_df['home_team'] == home_t].tail(5)
        past_away = history_df[history_df['away_team'] == away_t].tail(5)
        
        # DEBUG: Check if we actually found any rows
        if index < 3:
            print(f"DEBUG [{home_t}]: Found {len(past_home)} past home games in CSV.")
            print(f"DEBUG [{away_t}]: Found {len(past_away)} past away games in CSV.")
        
        past_overall_home = history_df[(history_df['home_team'] == home_t) | (history_df['away_team'] == home_t)].tail(5)
        past_overall_away = history_df[(history_df['home_team'] == away_t) | (history_df['away_team'] == away_t)].tail(5)
        
        home_pts = sum([r['home_points'] if r['home_team'] == home_t else r['away_points'] for _, r in past_overall_home.iterrows()])
        away_pts = sum([r['away_points'] if r['away_team'] == away_t else r['home_points'] for _, r in past_overall_away.iterrows()])
        
        home_overall_avg = home_pts / len(past_overall_home) if len(past_overall_home) > 0 else 0
        away_overall_avg = away_pts / len(past_overall_away) if len(past_overall_away) > 0 else 0

        match_stats = {
            'B365H': match.get('B365H', 0),
            'B365D': match.get('B365D', 0),
            'B365A': match.get('B365A', 0),
            'home_team_goals_avg_last_5': past_home['home_goals'].mean(),
            'away_goals_avg_last_5': past_away['away_goals'].mean(),
            'ht_home_goals_avg_last_5': past_home['ht_home_goals'].mean(),
            'ht_away_goals_avg_last_5': past_away['ht_away_goals'].mean(),
            'home_form_1h_last_5': past_home['home_balance_1h'].mean(),
            'away_form_1h_last_5': past_away['away_balance_1h'].mean(),
            'home_form_2h_last_5': past_home['home_balance_2h'].mean(),
            'away_form_2h_last_5': past_away['away_balance_2h'].mean(),
            'home_shots_avg_last_5': past_home['home_shots'].mean(),
            'away_shots_avg_last_5': past_away['away_shots'].mean(),
            'home_shots_target_avg_last_5': past_home['home_shots_target'].mean(),
            'away_shots_target_avg_last_5': past_away['away_shots_target'].mean(),
            'home_red_cards_avg_last_5': past_home['home_red_cards'].mean(),
            'away_red_cards_avg_last_5': past_away['away_red_cards'].mean(),
            'home_goals_conceded_avg_last_5': past_home['away_goals'].mean(),
            'away_goals_conceded_avg_last_5': past_away['home_goals'].mean(),
            'home_points_avg_last_5': past_home['home_points'].mean(),
            'away_points_avg_last_5': past_away['away_points'].mean(),
            'home_overall_points_last_5': home_overall_avg,
            'away_overall_points_last_5': away_overall_avg
        }
        processed_rows.append(match_stats)
        
    df_processed = pd.DataFrame(processed_rows).fillna(0)
    for col in expected_features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    return df_processed[expected_features]

def run_predictions():
    print("🧠 Loading XGBoost model...")
    xgb_model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    history_df = pd.read_csv(HISTORICAL_DATA_PATH)
    
    upcoming_fixtures = fetch_weekend_fixtures_api()
    if upcoming_fixtures is None: return

    ready_to_predict_df = engineer_features_on_the_fly(upcoming_fixtures, history_df, features)
    
    print("🔮 Running predictions...")
    probabilities = xgb_model.predict_proba(ready_to_predict_df)
    max_probs = np.max(probabilities, axis=1)
    predictions = xgb_model.predict(ready_to_predict_df)
    
    upcoming_fixtures['Model_Pick_Num'] = predictions
    upcoming_fixtures['Confidence'] = max_probs
    
    result_map = {2: 'HOME WIN', 1: 'DRAW', 0: 'AWAY WIN'}
    upcoming_fixtures['Pick'] = upcoming_fixtures['Model_Pick_Num'].map(result_map)
    
    expert_picks = upcoming_fixtures[upcoming_fixtures['Confidence'] >= CONFIDENCE_THRESHOLD].copy()
    
    print("\n" + "="*95)
    print(f"🎯 EXPERT PICKS (Confidence > {CONFIDENCE_THRESHOLD*100}%)")
    print("="*95)
    if expert_picks.empty:
        print("No picks found.")
    else:
        print(expert_picks[['Date', 'HomeTeam', 'AwayTeam', 'Pick', 'Confidence', 'B365H', 'B365D', 'B365A']])

if __name__ == "__main__":
    run_predictions()