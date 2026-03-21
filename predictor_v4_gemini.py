"""
predictor_v4_gemini.py - 究极赛马预测器 (Supreme Version)
============================================================
"The Best, Highest Accuracy, Unparalleled Predictor"

Modified to strictly match the input/output interface of `predictor.py` for direct pipeline compatibility.
Fixed historical data leakage and improved contextual aptitude calculations.

Features:
1.  **Hybrid Objective**: Combines Classification (Top3 Probability) and Ranking (LambdaRank) for optimal ordering.
2.  **Context-Aware Aptitude**: Explicitly models horse suitability for the specific Race Context (Course, Distance, Surface, Condition).
3.  **Auto-Context Detection**: Automatically extracts the upcoming race's location, distance, and surface from `shutuba.csv`.
4.  **Advanced Feature Engineering**:
    -   **Time Index Trends**: Expanding means and recent trends.
    -   **Baba Index Interaction**: Models how horses perform under specific track speeds.
    -   **Jockey/Course Synergy**: Historical performance of Jockey.
5.  **Strict No-Leakage**: Time-Series calculations ensure no future data leaks into past predictions.

Usage:
    python predictor_v4_gemini.py
============================================================
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, ndcg_score
import re
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Helper Functions ---

def configure_utf8_io():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
configure_utf8_io()

OUTPUT_PATH = Path(os.environ.get("PREDICTIONS_OUTPUT", "predictions.csv")).expanduser()


def resolve_lgbm_n_jobs():
    raw = str(os.environ.get("PREDICTOR_LGBM_N_JOBS", "1") or "1").strip()
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 1


LGBM_N_JOBS = resolve_lgbm_n_jobs()


def resolve_scope_key():
    raw = str(os.environ.get("SCOPE_KEY", "") or "").strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_").replace("/", "_")
    if raw in ("central_turf", "central_t", "ct", "1", "t", "turf", "grass", "shiba"):
        return "central_turf"
    if raw in ("central_dirt", "central_d", "cd", "2", "d", "dirt", "sand"):
        return "central_dirt"
    if raw in ("local", "l", "3"):
        return "local"
    return "central_dirt"


SCOPE_KEY = resolve_scope_key()


def normalize_name(value):
    return "".join(str(value or "").split())


def load_current_entries_map(odds_path=None):
    path = Path(odds_path or "odds.csv")
    if not path.exists():
        return {}
    try:
        odds_df = pd.read_csv(path)
    except Exception:
        return {}
    out = {}
    for _, row in odds_df.iterrows():
        raw_name = row.get("name", row.get("HorseName", row.get("horse_name", "")))
        key = normalize_name(raw_name)
        if not key or key in out:
            continue
        horse_no = parse_int(row.get("horse_no", row.get("horse", row.get("馬番", np.nan))))
        odds = parse_float(row.get("odds", row.get("win_odds", np.nan)))
        out[key] = {
            "horse_no": horse_no,
            "odds": odds,
            "name": str(raw_name or "").strip(),
        }
    return out


def default_surface_for_scope(scope_key):
    return "芝" if str(scope_key or "").strip() == "central_turf" else "ダ"


def normalize_surface(value, fallback):
    text = str(value or "").strip().lower()
    if text in ("芝", "t", "turf", "grass", "shiba", "1"):
        return "芝"
    if text in ("ダ", "d", "dirt", "sand", "2"):
        return "ダ"
    return str(fallback or "ダ")


def normalize_condition(value, fallback="良"):
    text = str(value or "").strip()
    return text if text else str(fallback or "良")


def condition_to_baba_index(condition):
    mapping = {
        "良": -5.0,
        "稍重": 5.0,
        "重": 10.0,
        "不良": 15.0,
    }
    return float(mapping.get(str(condition or "").strip(), 0.0))


DEFAULT_TARGET_CONTEXT = {
    "location": str(os.environ.get("PREDICTOR_TARGET_LOCATION", "") or "").strip(),
    "distance": int(float(str(os.environ.get("PREDICTOR_TARGET_DISTANCE", "1800") or "1800").strip() or "1800")),
    "surface": normalize_surface(os.environ.get("PREDICTOR_TARGET_SURFACE", ""), default_surface_for_scope(SCOPE_KEY)),
    "condition": normalize_condition(os.environ.get("PREDICTOR_TARGET_CONDITION", ""), "良"),
    "baba_index_proxy": 0.0,
}
DEFAULT_TARGET_CONTEXT["baba_index_proxy"] = float(
    os.environ.get("PREDICTOR_BABA_INDEX_PROXY", condition_to_baba_index(DEFAULT_TARGET_CONTEXT["condition"])) or 0.0
)

def parse_float(x):
    try:
        return float(str(x).replace(",", "").strip())
    except:
        return np.nan

def parse_int(x):
    try:
        return int(float(str(x).strip()))
    except:
        return np.nan

def parse_course_info(x):
    if not isinstance(x, str): return ""
    return re.sub(r'\d+', '', x) # Remove numbers to get "中山", "东京" etc.

def parse_distance_surface(x):
    if not isinstance(x, str): return "Unknown", np.nan
    surface = x[0]
    try:
        distance = int(x[1:])
    except:
        distance = np.nan
    return surface, distance

def parse_baba_index(x):
    try:
        return float(x)
    except:
        return 0.0 # Default to 0 if missing

def parse_rank(x):
    try:
        return int(x)
    except:
        return np.nan


def resolve_target_context(shutuba_raw, shutuba_clean):
    context = dict(DEFAULT_TARGET_CONTEXT)
    if shutuba_clean is not None and not shutuba_clean.empty:
        sample_row = shutuba_clean.iloc[0]
        location = str(sample_row.get("Location", "") or "").strip()
        surface = sample_row.get("Surface", "")
        distance = sample_row.get("Distance", np.nan)
        context["location"] = location or context["location"]
        if pd.notna(distance):
            context["distance"] = int(distance)
        context["surface"] = normalize_surface(surface, context["surface"])
    if shutuba_raw is not None and not shutuba_raw.empty:
        raw_row = shutuba_raw.iloc[0]
        if not context["location"]:
            raw_location = parse_course_info(raw_row.get("髢句ぎ", ""))
            context["location"] = raw_location or context["location"]
        raw_condition = raw_row.get("鬥ｬ蝣ｴ", context["condition"])
        context["condition"] = normalize_condition(raw_condition, context["condition"])
    context["surface"] = normalize_surface(os.environ.get("PREDICTOR_TARGET_SURFACE", ""), context["surface"])
    env_location = str(os.environ.get("PREDICTOR_TARGET_LOCATION", "") or "").strip()
    if env_location:
        context["location"] = env_location
    env_distance = str(os.environ.get("PREDICTOR_TARGET_DISTANCE", "") or "").strip()
    if env_distance:
        try:
            context["distance"] = int(float(env_distance))
        except Exception:
            pass
    context["condition"] = normalize_condition(os.environ.get("PREDICTOR_TARGET_CONDITION", ""), context["condition"])
    env_baba = str(os.environ.get("PREDICTOR_BABA_INDEX_PROXY", "") or "").strip()
    if env_baba:
        try:
            context["baba_index_proxy"] = float(env_baba)
        except Exception:
            context["baba_index_proxy"] = condition_to_baba_index(context["condition"])
    else:
        context["baba_index_proxy"] = condition_to_baba_index(context["condition"])
    if not context["location"]:
        context["location"] = "Unknown"
    return context

# --- Feature Engineering Class ---

class SupremeFeatureEngineer:
    def __init__(self):
        self.le_sex = LabelEncoder()
        self.global_jockey_win_rates = {}
        
    def preprocess(self, df):
        print("[INFO] Preprocessing raw data...")
        df = df.copy()
        
        # Basic Parsing
        df['TimeIndex'] = df['ﾀｲﾑ指数'].apply(parse_float) if 'ﾀｲﾑ指数' in df.columns else np.nan
        df['BabaIndex'] = df['馬場指数'].apply(parse_baba_index) if '馬場指数' in df.columns else 0.0
        df['Rank'] = df['着順'].apply(parse_rank) if '着順' in df.columns else np.nan
        df['Date'] = pd.to_datetime(df['日付'], errors='coerce') if '日付' in df.columns else pd.NaT
        df['Location'] = df['開催'].apply(parse_course_info) if '開催' in df.columns else ""
        
        # Surface & Distance
        if '距離' in df.columns:
            parsed = df['距離'].apply(lambda x: pd.Series(parse_distance_surface(x)))
            df['Surface'] = parsed[0]
            df['Distance'] = parsed[1]
        else:
            df['Surface'] = "Unknown"
            df['Distance'] = np.nan
        
        # Sex & Age
        if 'SexAge' in df.columns:
            df['Sex'] = df['SexAge'].astype(str).str[0]
            df['Age'] = df['SexAge'].astype(str).str[1:].apply(parse_float)
        else:
            df['Sex'] = "Unknown"
            df['Age'] = np.nan
            
        # Strip whitespace from HorseName
        if 'HorseName' in df.columns:
            df['HorseName'] = df['HorseName'].astype(str).str.strip()
        
        # Encode Categoricals (Safely handle unseen values)
        try:
            df['sex_code'] = self.le_sex.fit_transform(df['Sex'].astype(str))
        except:
            df['sex_code'] = 0
        
        # Jockey
        if 'JockeyId' in df.columns:
            df['JockeyId'] = df['JockeyId'].fillna('0').astype(str)
        else:
            df['JockeyId'] = '0'
            
        if 'JockeyId_current' in df.columns:
            df['JockeyId_current'] = df['JockeyId_current'].fillna('0').astype(str)
            
        # Target Variables
        df['IsWin'] = (df['Rank'] == 1).astype(int)
        df['IsTop3'] = (df['Rank'] <= 3).astype(int)
        
        # Extra columns for Pipeline compatibility
        if '馬番' in df.columns:
            df['horse_no'] = df['馬番'].apply(parse_int)
        else:
            df['horse_no'] = np.nan
            
        if 'オッズ' in df.columns:
            df['Odds'] = df['オッズ'].apply(parse_float)
        else:
            df['Odds'] = np.nan
        
        # Sort
        df = df.sort_values(['HorseName', 'Date'])
        return df

    def fit_globals(self, df):
        print("[INFO] Fitting global statistics...")
        wins = df[df['IsWin'] == 1].groupby('JockeyId').size()
        races = df.groupby('JockeyId').size()
        self.global_jockey_win_rates = (wins / races).fillna(0).to_dict()

    def create_features(self, df, is_inference=False, target_context=None, target_horses=None, current_entries=None):
        """
        Creates features. 
        If is_inference=True, creates features for the *next* race (target_context).
        If is_inference=False, creates features for *each historical race* using only prior data.
        """
        print(f"[INFO] Creating features (Inference={is_inference})...")
        
        df = df.copy()
        
        # --- 1. Jockey Stats (Requires Global Sort by Date) ---
        df = df.sort_values('Date')
        df['jockey_cum_wins'] = df.groupby('JockeyId')['IsWin'].transform(lambda x: x.cumsum().shift(1))
        df['jockey_cum_races'] = df.groupby('JockeyId').cumcount() # 0-indexed, so effectively count prior
        df['jockey_win_rate'] = (df['jockey_cum_wins'] / df['jockey_cum_races'].clip(lower=1)).fillna(0)
        
        # Restore sort order for Horse-specific calculations
        df = df.sort_values(['HorseName', 'Date'])
        grouped = df.groupby('HorseName')
        
        # --- 2. Basic Rolling Stats (Trend) ---
        # Time Index
        df['ti_lag1'] = grouped['TimeIndex'].shift(1)
        df['ti_mean_3'] = grouped['TimeIndex'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['ti_mean_5'] = grouped['TimeIndex'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df['ti_max_5'] = grouped['TimeIndex'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max())
        df['ti_trend'] = df['ti_lag1'] - df['ti_mean_5']
        
        # Adjusted TI (TimeIndex - BabaIndex)
        df['AdjTimeIndex'] = df['TimeIndex'] - df['BabaIndex']
        df['adj_ti_mean_5'] = grouped['AdjTimeIndex'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # --- 3. Aptitude (Context Specific) ---
        def calc_expanding_context_stat(df, context_col, value_col, prior_col, prior_k=3):
            group_cols = ['HorseName', context_col]

            def _calc(group):
                shifted = group[value_col].shift(1)
                obs_sum = shifted.expanding(min_periods=1).sum()
                obs_count = shifted.expanding(min_periods=1).count()
                obs_mean = obs_sum / obs_count.replace(0, np.nan)
                prior = group[prior_col].where(group[prior_col].notna() & (group[prior_col] != 0), obs_mean)
                shrunk = (obs_sum.fillna(0.0) + prior * prior_k) / (obs_count.fillna(0.0) + prior_k)
                return shrunk.where(obs_count > 0, np.nan)

            return df.groupby(group_cols, group_keys=False).apply(_calc)

        df['ti_course_avg'] = calc_expanding_context_stat(df, 'Location', 'TimeIndex', 'ti_mean_5')
        df['ti_surface_avg'] = calc_expanding_context_stat(df, 'Surface', 'TimeIndex', 'ti_mean_5')
        df['ti_dist_avg'] = calc_expanding_context_stat(df, 'Distance', 'TimeIndex', 'ti_mean_5')
        
        # Condition Aptitude (Heavy vs Fast)
        df['BabaCategory'] = (df['BabaIndex'] >= 0).astype(int) # 1=Slow, 0=Fast
        df['ti_cond_avg'] = calc_expanding_context_stat(df, 'BabaCategory', 'TimeIndex', 'ti_mean_5')
        
        # --- 4. Target Context Diff (For Training) ---
        df['WinDistance'] = df['Distance'].where(df['IsWin'] == 1)
        df['avg_win_dist'] = df.groupby('HorseName')['WinDistance'].transform(lambda x: x.expanding().mean().shift(1))
        # If no previous wins, use current distance (diff = 0)
        df['avg_win_dist'] = df['avg_win_dist'].fillna(df['Distance'])
        df['dist_diff_from_optimal'] = (df['Distance'] - df['avg_win_dist']).abs().fillna(0)
        
        # --- 5. Fill NaNs ---
        # For context aptitudes, if NaN (first time in this context), fill with general form 'ti_mean_5'
        for col in ['ti_course_avg', 'ti_surface_avg', 'ti_dist_avg', 'ti_cond_avg']:
            df[col] = df[col].fillna(df['ti_mean_5'])
            
        # Odds implied probability (pre-race, no leakage)
        if 'Odds' in df.columns:
            df['odds_implied'] = (1.0 / df['Odds'].clip(lower=1.0)).fillna(0)
        else:
            df['odds_implied'] = 0.0

        df = df.fillna(0)

        if is_inference:
            return self._prepare_inference_rows(df, target_context, target_horses, current_entries=current_entries)
        else:
            return df

    def _prepare_inference_rows(self, df, context, target_horses, current_entries=None):
        inference_rows = []
        horse_groups = dict(tuple(df.groupby('HorseName')))
        current_entries = current_entries or {}
        
        for horse in target_horses:
            if horse in horse_groups:
                group = horse_groups[horse].sort_values('Date')
                
                # 1. Base Stats
                ti_series = group['TimeIndex'].dropna()
                if len(ti_series) == 0:
                    ti_mean_5, ti_max_5, ti_trend, adj_ti_mean_5 = 0, 0, 0, 0
                else:
                    ti_mean_5 = ti_series.tail(5).mean()
                    ti_max_5 = ti_series.tail(5).max()
                    ti_last = ti_series.iloc[-1]
                    ti_trend = ti_last - ti_mean_5
                    
                    adj_ti_series = group['AdjTimeIndex'].dropna()
                    adj_ti_mean_5 = adj_ti_series.tail(5).mean() if len(adj_ti_series) > 0 else 0

                # 2. Context Aptitude (Average of ALL matches up to now)
                def get_context_avg(col, val, value_col, _k=3):
                    matches = group[group[col] == val][value_col].dropna()
                    mn = len(matches)
                    if mn == 0:
                        return np.nan
                    obs = float(matches.mean())
                    _prior = ti_mean_5 if ti_mean_5 != 0 else obs
                    return (obs * mn + _prior * _k) / (mn + _k)

                ti_course_avg = get_context_avg('Location', context['location'], 'TimeIndex')
                ti_surface_avg = get_context_avg('Surface', context['surface'], 'TimeIndex')
                ti_dist_avg = get_context_avg('Distance', context['distance'], 'TimeIndex')
                
                is_heavy = 1 if context['baba_index_proxy'] >= 0 else 0
                ti_cond_avg = get_context_avg('BabaCategory', is_heavy, 'TimeIndex')
                
                # Fill NaNs with current form
                if pd.isna(ti_course_avg): ti_course_avg = ti_mean_5
                if pd.isna(ti_surface_avg): ti_surface_avg = ti_mean_5
                if pd.isna(ti_dist_avg): ti_dist_avg = ti_mean_5
                if pd.isna(ti_cond_avg): ti_cond_avg = ti_mean_5
                
                # 3. Jockey Win Rate
                last_row = group.iloc[-1]
                # Use JockeyId_current if available for the target race
                current_jockey = last_row.get('JockeyId_current', last_row.get('JockeyId', '0'))
                if pd.isna(current_jockey) or current_jockey == '0':
                    current_jockey = last_row.get('JockeyId', '0')
                jockey_wr = self.global_jockey_win_rates.get(current_jockey, 0.0)
                
                # 4. Dist Diff
                wins = group[group['IsWin'] == 1]['Distance']
                avg_win_dist = wins.mean() if len(wins) > 0 else group['Distance'].mean()
                if pd.isna(avg_win_dist): avg_win_dist = context['distance']
                dist_diff = abs(context['distance'] - avg_win_dist)
                
                age = last_row['Age']
                sex_code = last_row['sex_code']
                current_meta = current_entries.get(normalize_name(horse), {})
                horse_no = current_meta.get("horse_no", last_row.get('horse_no', np.nan))
                odds = current_meta.get("odds", last_row.get('Odds', np.nan))
                odds_implied = 1.0 / max(odds, 1.0) if pd.notna(odds) and odds > 0 else 0.0

            else:
                # No history - New Horse
                ti_mean_5, ti_max_5, ti_trend, adj_ti_mean_5 = 0, 0, 0, 0
                ti_course_avg, ti_surface_avg, ti_dist_avg, ti_cond_avg = 0, 0, 0, 0
                jockey_wr, dist_diff = 0, 0
                age, sex_code = 3, 0
                current_meta = current_entries.get(normalize_name(horse), {})
                horse_no = current_meta.get("horse_no", np.nan)
                odds = current_meta.get("odds", np.nan)
                odds_implied = 1.0 / max(odds, 1.0) if pd.notna(odds) and odds > 0 else 0.0

            row = {
                "HorseName": horse,
                "ti_mean_5": ti_mean_5,
                "ti_max_5": ti_max_5,
                "ti_trend": ti_trend,
                "adj_ti_mean_5": adj_ti_mean_5,
                "ti_course_avg": ti_course_avg,
                "ti_surface_avg": ti_surface_avg,
                "ti_dist_avg": ti_dist_avg,
                "ti_cond_avg": ti_cond_avg,
                "jockey_win_rate": jockey_wr,
                "Age": age,
                "sex_code": sex_code,
                "dist_diff_from_optimal": dist_diff,
                "odds_implied": odds_implied,
                "horse_no": horse_no,
                "Odds": odds
            }
            inference_rows.append(row)
            
        return pd.DataFrame(inference_rows).fillna(0)

# --- Model Class ---

class SupremePredictor:
    def __init__(self):
        self.ranker = None
        self.classifier = None
        self.features = [
            "ti_mean_5", "ti_max_5", "ti_trend", "adj_ti_mean_5",
            "ti_course_avg", "ti_surface_avg", "ti_dist_avg", "ti_cond_avg",
            "jockey_win_rate", "Age", "sex_code", "dist_diff_from_optimal",
            "odds_implied"
        ]
        
    def train(self, train_df):
        print("[INFO] Training Supreme Models...")
        
        # Filter valid rows
        train_df = train_df.dropna(subset=self.features + ['Rank'])
        
        # Create Race Groups
        if 'race_id' not in train_df.columns:
            train_df['race_id'] = train_df['Date'].astype(str) + train_df['Location']
        
        # Sort for LGBM
        train_df = train_df.sort_values('race_id')
        X = train_df[self.features]
        y_rank = train_df['Rank']
        y_top3 = train_df['IsTop3']
        qids = train_df.groupby('race_id', sort=False).size().to_numpy()
        
        # 1. Ranker (LambdaRank)
        print("  > Training Ranker...")
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=LGBM_N_JOBS
        )
        # Label relevance: 1st=10, 2nd=5, 3rd=3, Others=0
        y_rel = y_rank.apply(lambda r: 10 if r==1 else (5 if r==2 else (3 if r==3 else 0)))
        self.ranker.fit(X, y_rel, group=qids)
        
        # 2. Classifier (Binary Logloss)
        print("  > Training Classifier...")
        self.classifier = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=LGBM_N_JOBS
        )
        self.classifier.fit(X, y_top3)
        
        print("[INFO] Training Complete.")
        
    def predict(self, inference_df):
        if inference_df.empty: return pd.DataFrame()
        
        X = inference_df[self.features]
        
        # Predict Rank Score (Higher is better)
        rank_score = self.ranker.predict(X)
        
        # Predict Top3 Probability
        prob_score = self.classifier.predict_proba(X)[:, 1]
        
        # Hybrid Score: 60% Ranker + 40% Classifier (normalized)
        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-6)
        
        final_score = 0.6 * norm(rank_score) + 0.4 * prob_score
        
        results = inference_df.copy()
        results['Top3Prob_model'] = prob_score
        results['Top3Prob'] = prob_score
        results['rank_score'] = final_score
        
        results = results.sort_values('rank_score', ascending=False)
        
        # Pipeline compatibility columns
        scores = results['rank_score'].values
        if len(scores) >= 3:
            gap = float(scores[0]) - float(scores[2])
            confidence = min(gap * 10, 1.0)
        else:
            confidence = 0.0
            
        results['confidence_score'] = confidence 
        results['stability_score'] = 0.5
        results['validity_score'] = 0.5
        results['consistency_score'] = 0.5
        results['rank_ema'] = 0.5
        results['ev_ema'] = 0.5
        results['risk_score'] = 0.5
        
        return results

# --- Main Execution ---

def main():
    print("="*60)
    print("PREDICTOR V4 GEMINI SUPREME - PIPELINE READY")
    print("="*60)
    
    if not os.path.exists("kachiuma.csv") or not os.path.exists("shutuba.csv"):
        print("[ERROR] Data files kachiuma.csv or shutuba.csv missing.")
        return
        
    history = pd.read_csv("kachiuma.csv")
    shutuba = pd.read_csv("shutuba.csv")
    
    engineer = SupremeFeatureEngineer()
    
    # 1. Preprocess
    history_clean = engineer.preprocess(history)
    shutuba_clean = engineer.preprocess(shutuba)
    
    # Extract upcoming race context automatically from shutuba.csv
    if not shutuba_clean.empty:
        sample_row = shutuba_clean.iloc[0]
        loc = parse_course_info(sample_row.get('Location', '中山'))
        surf = sample_row.get('Surface', '芝')
        dist = sample_row.get('Distance', 1800)
        
        # Read raw string condition from original df since preprocess might drop it
        raw_cond = shutuba.iloc[0].get('馬場', '稍重')
        if pd.isna(raw_cond): raw_cond = '稍重'
        baba_map = {"良": -5.0, "稍重": 5.0, "重": 10.0, "不良": 15.0}
        
        TARGET_CONTEXT = {
            "location": loc if loc else "中山",
            "distance": dist if pd.notna(dist) else 1800,
            "surface": surf if surf and surf != "Unknown" else "芝",
            "condition": raw_cond,
            "baba_index_proxy": baba_map.get(raw_cond, 0.0)
        }
    else:
        TARGET_CONTEXT = DEFAULT_TARGET_CONTEXT
        
    print(f"[INFO] Detected Target Context: {TARGET_CONTEXT}")
    
    # Fit global stats
    engineer.fit_globals(history_clean)
    
    # 2. Create Training Data
    train_df = engineer.create_features(history_clean, is_inference=False)
    
    # 3. Train Model
    predictor = SupremePredictor()
    predictor.train(train_df)
    
    # 4. Inference Data
    target_horses = shutuba_clean['HorseName'].unique()
    print(f"[INFO] Target Horses: {len(target_horses)}")
    current_entries = load_current_entries_map("odds.csv")
    
    inference_features = engineer.create_features(
        shutuba_clean, 
        is_inference=True, 
        target_context=TARGET_CONTEXT,
        target_horses=target_horses,
        current_entries=current_entries,
    )
    
    # 5. Predict
    results = predictor.predict(inference_features)
    
    # 6. Output Formatting (Matching predictor.py)
    print("\n" + "="*60)
    print("  Top Predictions")
    print("="*60)
    
    for i, (_, row) in enumerate(results.iterrows()):
        hno = row.get("horse_no", np.nan)
        hno_str = f"{int(hno):>2}" if pd.notna(hno) and hno != 0 else " ?"
        name = row["HorseName"]
        odds_val = row.get("Odds", np.nan)
        score = row["rank_score"]
        
        odds_str = f"odds={odds_val:6.1f}" if pd.notna(odds_val) and odds_val != 0 else "odds=   N/A"
        marker = " ***" if i < 3 else "    " if i < 5 else ""
        print(f"  {hno_str}  {name:18s}  {odds_str}  score={score:.4f}{marker}")
    
    print(f"\nConfidence: {results['confidence_score'].iloc[0]:.4f}")
    print(f"Runners: {len(results)}")
    
    # Save standard output
    results.to_csv("predictions.csv", index=False, encoding="utf-8-sig")
    print("\n[INFO] Saved: predictions.csv")

def main_pipeline_entry():
    print("=" * 60)
    print("PREDICTOR V4 GEMINI SUPREME - PIPELINE READY")
    print("=" * 60)

    history_path = Path("kachiuma.csv")
    shutuba_path = Path("shutuba.csv")
    if (not history_path.exists()) or (not shutuba_path.exists()):
        print("[ERROR] Data files kachiuma.csv or shutuba.csv missing.")
        return

    history = pd.read_csv(history_path)
    shutuba = pd.read_csv(shutuba_path)

    engineer = SupremeFeatureEngineer()
    history_clean = engineer.preprocess(history)
    shutuba_clean = engineer.preprocess(shutuba)
    target_context = resolve_target_context(shutuba, shutuba_clean)
    print(f"[INFO] Detected Target Context: {target_context}")

    engineer.fit_globals(history_clean)
    train_df = engineer.create_features(history_clean, is_inference=False)

    predictor = SupremePredictor()
    predictor.train(train_df)

    if "HorseName" in shutuba_clean.columns:
        target_horses = shutuba_clean["HorseName"].astype(str).str.strip().unique()
    else:
        target_horses = []
    print(f"[INFO] Target Horses: {len(target_horses)}")
    current_entries = load_current_entries_map("odds.csv")

    inference_features = engineer.create_features(
        shutuba_clean,
        is_inference=True,
        target_context=target_context,
        target_horses=target_horses,
        current_entries=current_entries,
    )

    results = predictor.predict(inference_features)

    print("\n" + "=" * 60)
    print("  Top Predictions")
    print("=" * 60)

    for i, (_, row) in enumerate(results.iterrows()):
        hno = row.get("horse_no", np.nan)
        hno_str = f"{int(hno):>2}" if pd.notna(hno) and hno != 0 else " ?"
        name = str(row.get("HorseName", "") or "")
        odds_val = row.get("Odds", np.nan)
        score = float(row.get("rank_score", 0.0) or 0.0)
        odds_str = f"odds={odds_val:6.1f}" if pd.notna(odds_val) and odds_val != 0 else "odds=   N/A"
        marker = " ***" if i < 3 else "    " if i < 5 else ""
        print(f"  {hno_str}  {name:18s}  {odds_str}  score={score:.4f}{marker}")

    confidence = float(results["confidence_score"].iloc[0]) if not results.empty else 0.0
    print(f"\nConfidence: {confidence:.4f}")
    print(f"Runners: {len(results)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Saved: {OUTPUT_PATH}")


main = main_pipeline_entry


if __name__ == "__main__":
    main_pipeline_entry()
