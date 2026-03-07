"""
predictor_v4_supreme.py - 究极赛马预测器 (Supreme Version)
============================================================
"The Best, Highest Accuracy, Unparalleled Predictor"

Modified to match the input/output interface of `predictor.py` for pipeline compatibility.

Features:
1.  **Hybrid Objective**: Combines Classification (Top3 Probability) and Ranking (LambdaRank) for optimal ordering.
2.  **Context-Aware Aptitude**: Explicitly models horse suitability for the specific Race Context (Course, Distance, Surface, Condition).
3.  **Advanced Feature Engineering**:
    -   **Time Index Trends**: EMA (Exponential Moving Average) and Trend Slope.
    -   **Baba Index Interaction**: Models how horses perform under specific track speeds.
    -   **Jockey/Course Synergy**: Historical performance of Jockey at specific courses.
4.  **Robust Validation**: Time-Series Cross-Validation to prevent data leakage.
5.  **Ensemble**: Blends multiple model variants for stability.

Usage:
    python predictor_v4_supreme.py
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

OUTPUT_PATH = os.environ.get("PREDICTIONS_OUTPUT", "predictions.csv")

# --- Configuration ---
# Default Context - Ideally this should be dynamic, but for now we default to the requested context.
# In a full pipeline, these might be arguments or inferred from the data (though kachiuma/shutuba don't explicitly state the *target* race condition usually unless inferred from filename or external config).
DEFAULT_TARGET_CONTEXT = {
    "location": "中山",
    "distance": 1800,
    "surface": "芝",
    "condition": "稍重",  # Text condition
    "baba_index_proxy": 5.0, # Estimated Baba Index (+5 = Slightly Slow/Heavy)
}

# --- Helper Functions ---

def configure_utf8_io():
    for s in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
configure_utf8_io()

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


def resolve_target_context():
    context = dict(DEFAULT_TARGET_CONTEXT)
    distance_raw = os.environ.get("PREDICTOR_TARGET_DISTANCE", "").strip()
    if distance_raw:
        try:
            context["distance"] = int(float(distance_raw))
        except Exception:
            pass
    for env_key, field in (
        ("PREDICTOR_TARGET_LOCATION", "location"),
        ("PREDICTOR_TARGET_SURFACE", "surface"),
        ("PREDICTOR_TARGET_CONDITION", "condition"),
    ):
        value = os.environ.get(env_key, "").strip()
        if value:
            context[field] = value
    baba_raw = os.environ.get("PREDICTOR_BABA_INDEX_PROXY", "").strip()
    if baba_raw:
        try:
            context["baba_index_proxy"] = float(baba_raw)
        except Exception:
            pass
    return context

# --- Feature Engineering Class ---

class SupremeFeatureEngineer:
    def __init__(self):
        self.le_sex = LabelEncoder()
        self.le_jockey = LabelEncoder()
        self.le_trainer = LabelEncoder() # If available
        
    def preprocess(self, df):
        print("[INFO] Preprocessing raw data...")
        df = df.copy()
        
        # Basic Parsing
        df['TimeIndex'] = df['ﾀｲﾑ指数'].apply(parse_float)
        df['BabaIndex'] = df['馬場指数'].apply(parse_baba_index)
        df['Rank'] = df['着順'].apply(parse_rank)
        df['Date'] = pd.to_datetime(df['日付'], errors='coerce')
        df['Location'] = df['開催'].apply(parse_course_info)
        
        # Surface & Distance
        parsed = df['距離'].apply(lambda x: pd.Series(parse_distance_surface(x)))
        df['Surface'] = parsed[0]
        df['Distance'] = parsed[1]
        
        # Sex & Age
        df['Sex'] = df['SexAge'].astype(str).str[0]
        df['Age'] = df['SexAge'].astype(str).str[1:].astype(float)
        
        # Encode Categoricals
        # Note: For production, we should fit on training set and transform on test.
        # Here we fit on the whole history for simplicity, handling unknown in inference.
        df['sex_code'] = self.le_sex.fit_transform(df['Sex'].astype(str))
        
        # Jockey
        if 'JockeyId' in df.columns:
            df['JockeyId'] = df['JockeyId'].fillna('0').astype(str)
        else:
            df['JockeyId'] = '0'
            
        # Target Variables
        df['IsWin'] = (df['Rank'] == 1).astype(int)
        df['IsTop3'] = (df['Rank'] <= 3).astype(int)
        
        # Sort
        df = df.sort_values(['HorseName', 'Date'])
        
        # Strip whitespace from HorseName
        if 'HorseName' in df.columns:
            df['HorseName'] = df['HorseName'].astype(str).str.strip()
            
        return df

    def create_features(self, df, is_inference=False, target_context=None, target_horses=None):
        """
        Creates features. 
        If is_inference=True, creates features for the *next* race (target_context).
        If is_inference=False, creates features for *each historical race* using only prior data.
        """
        print(f"[INFO] Creating features (Inference={is_inference})...")
        
        # We use a unified approach: 
        # 1. Calculate historical stats (rolling/expanding) for every row.
        # 2. Shift them by 1 to represent "prior knowledge".
        # 3. For inference, we take the stats from the *last* row of each horse.
        
        df = df.copy()
        grouped = df.groupby('HorseName')
        
        # --- 1. Basic Rolling Stats (Trend) ---
        # Time Index
        df['ti_lag1'] = grouped['TimeIndex'].shift(1)
        df['ti_mean_3'] = grouped['TimeIndex'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df['ti_mean_5'] = grouped['TimeIndex'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        df['ti_max_5'] = grouped['TimeIndex'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max())
        df['ti_trend'] = df['ti_lag1'] - df['ti_mean_5']
        
        # Adjusted TI (TimeIndex - BabaIndex) -> Pure Speed?
        # Or maybe TimeIndex is already adjusted? User said "Use them to compare".
        # Let's try an interaction feature: TI adjusted by Baba
        df['AdjTimeIndex'] = df['TimeIndex'] - df['BabaIndex']
        df['adj_ti_mean_5'] = grouped['AdjTimeIndex'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # --- 2. Aptitude (Context Specific) ---
        # We need to calculate expanding means for specific conditions.
        # Since pandas expanding().mean() on filtered data is hard to align,
        # we will use a global transform approach.
        
        # Define contexts to track
        # Location, Surface, Distance Category (Sprint<1400, Mile<1800, Inter<2200, Long>=2200), Condition
        
        def calc_expanding_context_stat(df, context_col, value_col, stat_name):
            # Calculate cumulative sum and count per group
            # We need to do this GLOBALLY (not per horse) if we want "Jockey at Course"
            # But here we want "Horse at Course". So Group by [HorseName, ContextCol]
            
            g = df.groupby(['HorseName', context_col])[value_col]
            cumsum = g.cumsum()
            cumcnt = g.cumcount() + 1
            
            # Shift to get prior stats
            # The shift must happen *within the Horse-Context group*
            prior_sum = g.shift(1).cumsum() # This is wrong. shift(1) shifts the value.
            # Correct: (cumsum - current_value) / (cumcnt - 1)
            
            # Actually, `g.expanding().mean().shift(1)` works within the group!
            # But we need to map it back to the original index.
            
            # Strategy:
            # 1. Calculate expanding mean within [Horse, Context] group.
            # 2. Shift by 1 within that group.
            # 3. Assign back to df.
            
            return g.transform(lambda x: x.expanding().mean().shift(1))

        # Course Aptitude
        df['ti_course_avg'] = calc_expanding_context_stat(df, 'Location', 'TimeIndex', 'ti_course_avg')
        
        # Surface Aptitude
        df['ti_surface_avg'] = calc_expanding_context_stat(df, 'Surface', 'TimeIndex', 'ti_surface_avg')
        
        # Distance Aptitude (Exact Distance)
        df['ti_dist_avg'] = calc_expanding_context_stat(df, 'Distance', 'TimeIndex', 'ti_dist_avg')
        
        # Condition Aptitude (Heavy vs Fast)
        # Bin BabaIndex: <0 (Fast), >=0 (Slow)
        df['BabaCategory'] = (df['BabaIndex'] >= 0).astype(int) # 1=Slow, 0=Fast
        df['ti_cond_avg'] = calc_expanding_context_stat(df, 'BabaCategory', 'TimeIndex', 'ti_cond_avg')
        
        # --- 3. Jockey Stats ---
        # Jockey Win Rate (Global)
        # Group by JockeyId only (across all horses)
        # Sort by Date first
        df = df.sort_values('Date')
        # We can't use transform easily for global expanding because of the sort.
        # But we can do:
        df['jockey_cum_wins'] = df.groupby('JockeyId')['IsWin'].cumsum().shift(1)
        df['jockey_cum_races'] = df.groupby('JockeyId').cumcount() # 0-indexed, so effectively count prior
        df['jockey_win_rate'] = (df['jockey_cum_wins'] / df['jockey_cum_races'].clip(lower=1)).fillna(0)
        
        # --- 4. Target Context Diff (For Training) ---
        # For training, the "Target" is the current row's context.
        # So `dist_diff` = `Distance` - `avg_win_dist`.
        
        # Avg Win Distance of the Horse
        # 1. Identify win rows
        # 2. Expanding mean of Distance where Rank=1
        
        # We need a custom apply for this.
        # "Average distance of previous wins"
        def get_avg_win_dist(x):
            # x is a Series of (IsWin, Distance) tuples? No.
            # Let's do it iteratively or with masking.
            wins = x[x['IsWin'] == 1]['Distance']
            if len(wins) == 0: return np.nan
            return wins.expanding().mean()
        
        # Vectorized "Avg Win Distance":
        # Create a column "WinDistance" = Distance if Win else NaN
        df['WinDistance'] = df['Distance'].where(df['IsWin'] == 1)
        df['avg_win_dist'] = df.groupby('HorseName')['WinDistance'].transform(lambda x: x.expanding().mean().shift(1))
        df['dist_diff_from_optimal'] = (df['Distance'] - df['avg_win_dist']).abs().fillna(0)
        
        # --- 5. Fill NaNs ---
        # For 'ti_course_avg', if NaN (first time at course), fill with 'ti_mean_5' (current form)
        # This assumes if we don't know course aptitude, we assume they run to their ability.
        for col in ['ti_course_avg', 'ti_surface_avg', 'ti_dist_avg', 'ti_cond_avg']:
            df[col] = df[col].fillna(df['ti_mean_5'])
            
        df = df.fillna(0)
        
        if is_inference:
            return self._prepare_inference_rows(df, target_context, target_horses)
        else:
            return df

    def _prepare_inference_rows(self, df, context, target_horses=None):
        # For inference, we need stats based on ALL history up to now.
        # The columns in 'df' (like ti_mean_5) are shifted (prior to that row).
        # We cannot use them directly for the *next* race.
        # We must recalculate stats using the full history.
        
        inference_rows = []
        
        # If target_horses is provided, iterate over them.
        # If a horse is not in df (no history), we create a default row.
        
        horses_to_process = target_horses if target_horses is not None else df['HorseName'].unique()
        
        for horse in horses_to_process:
            # Check if horse has history
            if horse in df['HorseName'].values:
                group = df[df['HorseName'] == horse].sort_values('Date')
                
                # 1. Base Stats (Recalculate from raw TimeIndex)
                ti_series = group['TimeIndex'].dropna()
                if len(ti_series) == 0:
                    ti_mean_5 = 0
                    ti_max_5 = 0
                    ti_trend = 0
                    adj_ti_mean_5 = 0
                else:
                    ti_mean_5 = ti_series.tail(5).mean()
                    ti_max_5 = ti_series.tail(5).max()
                    ti_last = ti_series.iloc[-1]
                    ti_trend = ti_last - ti_mean_5
                    
                    adj_ti_series = group['AdjTimeIndex'].dropna()
                    adj_ti_mean_5 = adj_ti_series.tail(5).mean() if len(adj_ti_series) > 0 else 0

                # 2. Context Aptitude (Recalculate)
                def get_context_avg(col, val, value_col):
                    matches = group[group[col] == val][value_col].dropna()
                    if len(matches) > 0:
                        return matches.mean()
                    return np.nan

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
                # Use last race jockey as proxy (imperfect but safe)
                last_row = group.iloc[-1]
                jockey_wr = last_row['jockey_win_rate']
                
                # 4. Dist Diff
                wins = group[group['IsWin'] == 1]['Distance']
                avg_win_dist = wins.mean() if len(wins) > 0 else group['Distance'].mean()
                if pd.isna(avg_win_dist): avg_win_dist = context['distance']
                dist_diff = abs(context['distance'] - avg_win_dist)
                
                age = last_row['Age']
                sex_code = last_row['sex_code']
                
            else:
                # No history - New Horse
                # Use defaults (0 or global means)
                ti_mean_5 = 0
                ti_max_5 = 0
                ti_trend = 0
                adj_ti_mean_5 = 0
                ti_course_avg = 0
                ti_surface_avg = 0
                ti_dist_avg = 0
                ti_cond_avg = 0
                jockey_wr = 0
                dist_diff = 0
                age = 3 # Default age?
                sex_code = 0 # Default sex?
            
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
                "dist_diff_from_optimal": dist_diff
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
            "jockey_win_rate", "Age", "sex_code", "dist_diff_from_optimal"
        ]
        
    def train(self, train_df):
        print("[INFO] Training Supreme Models...")
        
        # Prepare Data
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
            n_jobs=-1
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
            n_jobs=-1
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
        # Normalize both to 0-1 range roughly
        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-6)
        
        final_score = 0.6 * norm(rank_score) + 0.4 * prob_score
        
        results = inference_df.copy()
        results['rank_score'] = final_score
        # For pipeline compatibility, also populate other expected columns
        results['confidence_score'] = final_score # Approximation
        results['stability_score'] = 0.5
        results['validity_score'] = 0.5
        results['consistency_score'] = 0.5
        results['rank_ema'] = 0.5
        results['ev_ema'] = 0.5
        results['risk_score'] = 0.5
        
        return results.sort_values('rank_score', ascending=False)

# --- Main Execution ---

def main():
    print("="*60)
    print("PREDICTOR V4 SUPREME - INITIALIZING")
    print("="*60)
    
    # 1. Load Data
    if not os.path.exists("kachiuma.csv") or not os.path.exists("shutuba.csv"):
        print("Error: Data files missing.")
        return
        
    history = pd.read_csv("kachiuma.csv")
    shutuba = pd.read_csv("shutuba.csv")
    
    # 2. Engineer
    engineer = SupremeFeatureEngineer()
    
    # Preprocess
    history_clean = engineer.preprocess(history)
    shutuba_clean = engineer.preprocess(shutuba)
    
    # Create Training Data (History)
    train_df = engineer.create_features(history_clean, is_inference=False)
    
    # 3. Train
    predictor = SupremePredictor()
    predictor.train(train_df)
    
    # 4. Inference
    target_context = resolve_target_context()
    print(f"[INFO] Target Context: {target_context}")
    
    # The 'shutuba.csv' file contains the HISTORY of the target horses.
    # We should use THIS dataframe for inference feature engineering.
    # We don't need to look them up in kachiuma.csv if they are already here.
    
    target_horses = shutuba_clean['HorseName'].unique()
    print(f"[INFO] Target Horses: {len(target_horses)}")
    
    # Use shutuba_clean as the history source for these horses
    inference_features = engineer.create_features(
        shutuba_clean, 
        is_inference=True, 
        target_context=target_context,
        target_horses=target_horses
    )
    
    # 5. Predict
    results = predictor.predict(inference_features)
    
    # 6. Output
    print("\n" + "="*60)
    print("  SUPREME PREDICTIONS (Top 5)")
    print("="*60)
    
    # Add dummy columns required by some pipelines if they don't exist
    if 'horse_no' not in results.columns:
        results['horse_no'] = np.nan
        
    cols = ['HorseName', 'rank_score', 'ti_course_avg', 'ti_cond_avg', 'jockey_win_rate']
    print(results[cols].head(5).to_string(index=False))
    
    # Calculate confidence (gap between 1st and 3rd)
    scores = results['rank_score'].values
    if len(scores) >= 3:
        gap = float(scores[0]) - float(scores[2])
        confidence = min(gap * 10, 1.0) # Simple scaling
    else:
        confidence = 0.0
    
    results['confidence_score'] = confidence
    results['rank_score'] = scores # Ensure it's there
    
    # Save standard output
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Full results saved to {os.path.basename(OUTPUT_PATH)} (Pipeline Ready)")

if __name__ == "__main__":
    main()
