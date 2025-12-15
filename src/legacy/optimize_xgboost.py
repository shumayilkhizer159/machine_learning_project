"""
XGBoost Hyperparameter Optimization
Performs random search to find better hyperparameters for the XGBoost model.
Optimizes on a subset of frames to speed up the process.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import random
from pathlib import Path
import sys
import itertools

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import NFLDataLoader, FeatureEngineering
from train_gpu import CONFIG, FEATURE_COLS, load_and_preprocess_data

def evaluate_params(X_train, y_x_train, y_y_train, X_val, y_x_val, y_y_val, params):
    """Train and evaluate model with given parameters."""
    
    # Train x model
    model_x = xgb.XGBRegressor(**params)
    model_x.fit(X_train, y_x_train, eval_set=[(X_val, y_x_val)], verbose=False)
    
    # Train y model
    model_y = xgb.XGBRegressor(**params)
    model_y.fit(X_train, y_y_train, eval_set=[(X_val, y_y_val)], verbose=False)
    
    # Predict
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(((pred_x - y_x_val)**2 + (pred_y - y_y_val)**2).mean() / 2)
    return rmse

def main():
    print("Starting XGBoost Hyperparameter Optimization...")
    
    # 1. Load Data
    print("Loading data...")
    input_df, output_df, feature_cols = load_and_preprocess_data()
    
    # 2. Prepare Data for Representative Frames
    # We'll optimize on Frame +15 (mid-range)
    target_frame = 15
    print(f"Preparing data for Frame +{target_frame}...")
    
    from model_xgboost import XGBoostTrajectoryModel
    temp_model = XGBoostTrajectoryModel()
    
    # Use a subset of data for optimization to be fast
    # Group by game/play/nfl_id
    groups = input_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    sample_groups = groups.sample(n=min(2000, len(groups)), random_state=42)
    
    input_sample = input_df.merge(sample_groups, on=['game_id', 'play_id', 'nfl_id'])
    output_sample = output_df.merge(sample_groups, on=['game_id', 'play_id', 'nfl_id'])
    
    # Prepare training data for this frame
    # We need to manually extract it because prepare_training_data does all frames
    # Let's just use the class method but filter the result
    data_dict = temp_model.prepare_training_data(input_sample, output_sample, feature_cols)
    
    if target_frame not in data_dict:
        print(f"Frame {target_frame} not found in data. Using Frame 1.")
        target_frame = 1
        
    X, y_x, y_y = data_dict[target_frame]
    
    # Split
    X_train, X_val, y_x_train, y_x_val = train_test_split(X, y_x, test_size=0.2, random_state=42)
    _, _, y_y_train, y_y_val = train_test_split(X, y_y, test_size=0.2, random_state=42)
    
    print(f"Optimization dataset size: {len(X_train)} train, {len(X_val)} val")
    
    # 3. Define Search Space
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [4, 6, 8, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'n_estimators': [100, 200, 300]
    }
    
    # Random Search
    n_iter = 20
    best_rmse = float('inf')
    best_params = None
    
    print(f"\nRunning {n_iter} iterations of random search...")
    
    for i in range(n_iter):
        # Sample parameters
        params = {
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'tree_method': 'hist',
            'device': 'cuda' if CONFIG['device'] == 'cuda' else 'cpu',
            'random_state': 42
        }
        
        for key, values in param_grid.items():
            params[key] = random.choice(values)
            
        try:
            rmse = evaluate_params(X_train, y_x_train, y_y_train, X_val, y_x_val, y_y_val, params)
            print(f"Iter {i+1}/{n_iter}: RMSE={rmse:.4f} | Params={params}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                print(f"  New Best! RMSE: {best_rmse:.4f}")
                
        except Exception as e:
            print(f"Iter {i+1} failed: {e}")
            
    print("\nOptimization Complete!")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("Best Parameters:")
    print(json.dumps(best_params, indent=2))
    
    # Save best parameters
    save_path = Path(CONFIG['models_dir']) / 'xgboost_params.json'
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Parameters saved to {save_path}")

if __name__ == "__main__":
    main()
