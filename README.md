# ⚽ Premier League Analytics & Match Prediction Dashboard

### Python | Machine Learning | Data Engineering | Streamlit

An interactive football analytics platform combining Premier League statistics, historical performance data, and machine learning to explore team performance, player statistics, and match outcome probabilities.

**Project Status:** Active Development

## Project Overview

This project combines my passion for football with my interest in data science and analytics.

The goal is to build a centralized Premier League analytics platform inspired by applications such as FotMob and Sofascore, with an additional emphasis on predictive modeling.

The platform brings together historical match data, current-season statistics, engineered performance metrics, and machine-learning models through an interactive Streamlit dashboard.

## Dashboard Features

**League Overview**
- Premier League standings and team rankings
- Recent team form and performance statistics
- Goals scored, goals conceded, and expected goals (xG)
- Current-season league insights

**Match Analysis**
- Head-to-head team comparisons
- Historical performance metrics
- Match outcome probability analysis

**Matches & Fixtures**
- Matchweek navigation
- Completed match results and upcoming fixtures
- Club crests, scores, and match statuses

**Player Analytics**
- Individual player statistics
- League-wide statistical leaderboards
- Goals, assists, xG, xA, and additional performance metrics
- Player headshots and club identification

**Model Performance**
- Historical model evaluation
- Comparison against betting-market probability benchmarks
- Analysis of predictive performance

## Machine Learning Approach

The project uses a calibrated Random Forest classification model to estimate three possible match outcomes:

- Home win
- Draw
- Away win

The model uses **51 engineered features**, including:

- Recent team form over rolling match windows
- Goals scored and conceded
- Expected goals (xG)
- Shots, possession, and passing statistics
- Home and away performance
- Rest days
- Elo-based team strength ratings
- Relative performance differences between opponents

### Model Evaluation

| Metric | Result |
|---|---|
| Model | Calibrated Random Forest |
| Features | 51 |
| Test matches | 357 |
| Evaluation period | August 2025 – May 2026 |
| Model log loss | 0.9969 |
| Betting-market benchmark log loss | 0.9871 |

The current model performs slightly below the betting-market benchmark on log loss. Improving predictive performance and evaluating additional modeling approaches remain ongoing priorities.

## Technologies Used

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Random Forest |
| Data Collection | Football statistics APIs |
| Feature Engineering | Rolling statistics, xG, Elo ratings |
| Visualization | Streamlit |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

## Project Structure

```text
premier-league-predictor/
├── assets/
│   └── premier_league_background.png
├── dashboard.py
├── data/
│   └── processed/
├── models/
│   ├── feature_schema.json
│   └── model_metadata.json
├── notebooks/
│   ├── 01_api_exploration.ipynb
│   ├── 02_data_pipeline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 03_feature_engineering_v2.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_api_football_exploration.ipynb
│   ├── 06_reviewing_models.ipynb
│   └── 07_upcoming_predictions.ipynb
├── requirements.txt
└── README.md
```

## Running the Dashboard

The dashboard is developed using Streamlit.

After installing the required dependencies, the application can be launched with:

```bash
python -m streamlit run dashboard.py
```

**Data availability:** Some current-season datasets and trained model artifacts are not included in the public repository. The dashboard currently requires these locally generated files, so cloning the repository alone is not yet sufficient to run every feature.

The project is being prepared for a more reproducible public release.

## Future Development

- Improve match prediction performance
- Expand upcoming fixture predictions
- Introduce additional team and player analytics
- Enhance dashboard interactivity and visualization
- Automate data updates
- Improve reproducibility and deployment readiness

## About This Project

This is an independent project developed to strengthen my experience in sports analytics, machine learning, data engineering, and interactive application development.

It reflects my interest in using data to better understand football performance and build practical tools for analyzing the game.

**Developed by Luis Haddad**

[GitHub Profile](https://github.com/LuisHaddad04)
