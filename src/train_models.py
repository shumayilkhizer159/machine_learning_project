"""
Model Training and Comparison Script
Trains multiple models and compares their performance.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import NFLDataLoader, FeatureEngineering
from model_baseline import PhysicsBaselineModel
from model_xgboost import XGBoostTrajectoryModel

# Feature columns to use
FEATURE_COLS = [
    'x', 'y', 's', 'a', 'dir', 'o',
    'ball_land_x', 'ball_land_y', 'dist_to_ball_land',
    'v_x', 'v_y', 'frame_offset', 'time_to_frame',
    'predicted_x_cv', 'predicted_y_cv', 'predicted_dist_to_ball',
    'player_position_encoded', 'player_side_encoded', 'player_role_encoded',
    'player_height_inches', 'player_weight', 'player_age',
    'play_direction_binary', 'absolute_yardline_number'
]


def calculate_rmse(predictions: dict, ground_truth: pd.DataFrame) -> float:
    """
    Calculate RMSE for predictions according to competition metric.
    
    Args:
        predictions: Dict mapping (game_id, play_id, nfl_id) to predicted positions
        ground_truth: DataFrame with actual positions
        
    Returns:
        RMSE score
    """
    squared_errors = []
    
    for (game_id, play_id, nfl_id), pred_positions in predictions.items():
        # Get ground truth
        gt = ground_truth[
            (ground_truth['game_id'] == game_id) &
            (ground_truth['play_id'] == play_id) &
            (ground_truth['nfl_id'] == nfl_id)
        ].sort_values('frame_id')
        
        if len(gt) == 0:
            continue
        
        # Calculate errors for each frame
        for i, (_, row) in enumerate(gt.iterrows()):
            if i >= len(pred_positions):
                break
            
            pred_x, pred_y = pred_positions[i]
            true_x, true_y = row['x'], row['y']
            
            error = (pred_x - true_x)**2 + (pred_y - true_y)**2
            squared_errors.append(error)
    
    if len(squared_errors) == 0:
        return float('inf')
    
    # Competition RMSE formula
    rmse = np.sqrt(np.mean(squared_errors) / 2)
    return rmse


def train_baseline_model(input_df, output_df, method='weighted'):
    """Train and evaluate baseline physics model."""
    print("\n" + "="*80)
    print(f"TRAINING BASELINE MODEL (method: {method})")
    print("="*80)
    
    model = PhysicsBaselineModel()
    
    # Get unique plays
    plays = input_df[['game_id', 'play_id']].drop_duplicates()
    print(f"Evaluating on {len(plays)} plays...")
    
    all_predictions = {}
    
    for idx, (_, play) in enumerate(plays.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(plays)}")
        
        game_id = play['game_id']
        play_id = play['play_id']
        
        # Get play data
        play_input = input_df[
            (input_df['game_id'] == game_id) &
            (input_df['play_id'] == play_id)
        ]
        
        # Predict for each player
        for nfl_id in play_input['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            
            # Only predict if flagged
            if not player_input['player_to_predict'].iloc[0]:
                continue
            
            # Get number of frames
            num_frames = int(player_input['num_frames_output'].iloc[0])
            
            # Make prediction
            predictions = model.predict_play(player_input, num_frames, method=method)
            
            if nfl_id in predictions:
                all_predictions[(game_id, play_id, nfl_id)] = predictions[nfl_id]
    
    # Calculate RMSE
    rmse = calculate_rmse(all_predictions, output_df)
    print(f"\nBaseline Model RMSE: {rmse:.4f}")
    
    return model, rmse, all_predictions


def train_xgboost_model(input_df, output_df, feature_cols, max_frames=30):
    """Train and evaluate XGBoost model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    # Preprocess data
    print("Preprocessing data...")
    loader = NFLDataLoader()
    input_processed = loader.preprocess_input_data(input_df)
    
    # Add physics features
    fe = FeatureEngineering()
    input_processed = fe.add_physics_features(input_processed)
    
    # Filter to only available features
    available_features = [col for col in feature_cols if col in input_processed.columns]
    print(f"Using {len(available_features)} features: {available_features[:10]}...")
    
    # Prepare training data
    print("Preparing training data...")
    model = XGBoostTrajectoryModel(max_future_frames=max_frames)
    training_data = model.prepare_training_data(input_processed, output_df, available_features)
    
    # Train models
    model.train(training_data, verbose=True)
    
    # Evaluate
    print("\nEvaluating model...")
    all_predictions = {}
    
    plays = input_processed[['game_id', 'play_id']].drop_duplicates()
    
    for idx, (_, play) in enumerate(plays.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(plays)}")
        
        game_id = play['game_id']
        play_id = play['play_id']
        
        play_input = input_processed[
            (input_processed['game_id'] == game_id) &
            (input_processed['play_id'] == play_id)
        ]
        
        for nfl_id in play_input['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            
            if not player_input['player_to_predict'].iloc[0]:
                continue
            
            num_frames = int(player_input['num_frames_output'].iloc[0])
            
            try:
                predictions = model.predict(player_input, available_features, num_frames)
                all_predictions[(game_id, play_id, nfl_id)] = predictions
            except Exception as e:
                continue
    
    # Calculate RMSE
    rmse = calculate_rmse(all_predictions, output_df)
    print(f"\nXGBoost Model RMSE: {rmse:.4f}")
    
    return model, rmse, all_predictions


def save_results(results, save_path):
    """Save model comparison results."""
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("NFL BIG DATA BOWL 2026 - MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading training data...")
    data_dir = '/home/ubuntu/nfl_project/data'
    loader = NFLDataLoader(data_dir)
    
    # Load first 2 weeks for faster training (use more for final model)
    input_df, output_df = loader.load_all_training_data(weeks=[1, 2])
    
    print(f"Loaded {len(input_df)} input rows, {len(output_df)} output rows")
    
    # Train models
    results = {
        'timestamp': datetime.now().isoformat(),
        'data': {
            'weeks': [1, 2],
            'input_rows': len(input_df),
            'output_rows': len(output_df)
        },
        'models': {}
    }
    
    # 1. Baseline Model
    try:
        baseline_model, baseline_rmse, baseline_preds = train_baseline_model(
            input_df, output_df, method='weighted'
        )
        results['models']['baseline_physics'] = {
            'rmse': float(baseline_rmse),
            'method': 'weighted',
            'description': 'Physics-based model with weighted velocity and ball direction'
        }
    except Exception as e:
        print(f"Error training baseline model: {e}")
        results['models']['baseline_physics'] = {'error': str(e)}
    
    # 2. XGBoost Model
    try:
        xgb_model, xgb_rmse, xgb_preds = train_xgboost_model(
            input_df, output_df, FEATURE_COLS, max_frames=30
        )
        results['models']['xgboost'] = {
            'rmse': float(xgb_rmse),
            'max_frames': 30,
            'description': 'XGBoost with separate models for each future frame'
        }
        
        # Save XGBoost model
        xgb_model.save_models('/home/ubuntu/nfl_project/models/xgboost')
        print("XGBoost model saved!")
        
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        results['models']['xgboost'] = {'error': str(e)}
    
    # Save results
    save_results(results, '/home/ubuntu/nfl_project/outputs/model_comparison.json')
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*80)
    for model_name, model_results in results['models'].items():
        if 'rmse' in model_results:
            print(f"{model_name:30s}: RMSE = {model_results['rmse']:.4f}")
        else:
            print(f"{model_name:30s}: ERROR")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == '__main__':
    results = main()
