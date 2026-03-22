# PitchPredict ML: Football Match Outcome Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-orange)
![Database](https://img.shields.io/badge/Database-Supabase%20(PostgreSQL)-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end Machine Learning pipeline designed to predict football match outcomes across Europe's Top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1). The project focuses on finding high-confidence betting opportunities ("Expert Picks") using historical match data, rolling statistics, and bookmaker odds.

## 🚀 Project Overview & Business Value

Predicting football match results is notoriously difficult due to the high variance of the sport. Instead of trying to predict every match accurately, this project implements a selective prediction strategy. By setting a high confidence threshold, the XGBoost model focuses only on the most predictable fixtures.

*   **Baseline Accuracy:** ~54.11% (Predicting all matches).
*   **Expert Picks Accuracy:** **81.25%** (Predicting only matches with model confidence > 65%).
*   **Actionable Output:** Out of 1656 validation matches, the model successfully identified 240 high-value "Expert" matches.

An automated inference script fetches upcoming fixtures, engineers features on the fly, and outputs only the highest-value betting picks straight to the console.

## 🛠️ Tech Stack & Tools

*   **Data Processing:** `pandas`, `numpy`, `requests`
*   **Machine Learning:** `scikit-learn`, `xgboost`, `joblib`
*   **Database Integration:** `supabase-py`, `python-dotenv`
*   **Data Source:** [football-data.co.uk](https://www.football-data.co.uk)

## 📂 Project Structure

```text
├── data/                       # Raw, cleaned, and combined CSV datasets
├── models/                     # Serialized ML models (.pkl) and feature lists
├── notebooks/                  # Jupyter notebooks for EDA, Feature Engineering, Training
├── src/                        
│   └── predict_matchweek.py    # Main inference script to fetch fixtures and generate picks
├── .env                        # Environment variables (Supabase credentials)
├── .gitignore                  # Ignored files
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## ⚙️ Architecture & Pipeline

1.  **Data Extraction:** Historical data from the last 5 seasons across the Top 5 European Leagues is fetched and aggregated.
2.  **Data Cleaning & Cloud Storage:** The raw data is cleaned, formatted, and pushed to a remote **Supabase** PostgreSQL database for persistent storage (using batch uploads).
3.  **Feature Engineering:** Complex features are calculated, including:
    *   Rolling averages for goals scored/conceded, shots, and shots on target (last 5 matches).
    *   First-half and second-half form balance.
    *   Historical points accumulation and bookmaker odds (B365).
4.  **Model Training:** Logistic Regression, Random Forest, and XGBoost were evaluated using `TimeSeriesSplit` and `GridSearchCV` to prevent data leakage and optimize hyperparameters. XGBoost yielded the best performance and was selected as the final model.
5.  **Inference:** The `predict_matchweek.py` script pulls the upcoming weekend's fixtures, applies the exact same feature engineering pipeline to the new data, and outputs predictions meeting the strict confidence threshold.

## 💻 Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[Your-Username]/[Repository-Name].git
   cd [Repository-Name]
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables. Create a `.env` file in the root directory with your Supabase credentials (optional, for historical data fetching):
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

## 🔮 How to Run Predictions

To generate predictions for the upcoming matchweek, simply run the inference script from the root directory:

```bash
python src/predict_matchweek.py
```
*Note: Ensure the pre-trained models are located in the `models/` directory before running.*

## 📈 Future Work 
- Refactor procedural code into Object-Oriented Programming (OOP) classes.
- Replace `print()` statements with standard Python `logging`.
- Build a simple web UI using **Streamlit** to visualize predictions.
- Implement Scikit-Learn Pipelines for cleaner feature transformations.
