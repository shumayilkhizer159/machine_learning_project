"""
Finalize Project Script
Monitors training progress and runs ensemble optimization and submission generation.
"""

import time
import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model_ensemble import ModelEnsemble, optimize_ensemble_weights
from data_loader import NFLDataLoader, FeatureEngineering

CONFIG = {
    'data_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/data',
    'models_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/models',
    'outputs_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/outputs',
    'results_file': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/outputs/gpu_training_results.json'
}

FEATURE_COLS = [
    'x', 'y', 's', 'a', 'dir', 'o',
    'ball_land_x', 'ball_land_y', 'dist_to_ball_land',
    'v_x', 'v_y',
    'player_position_encoded', 'player_side_encoded', 'player_role_encoded',
    'player_height_inches', 'player_weight', 'player_age',
    'play_direction_binary', 'absolute_yardline_number'
]

def wait_for_training_completion():
    """Monitor results file for completion."""
    print("Monitoring training progress...")
    results_path = Path(CONFIG['results_file'])
    
    start_time = time.time()
    
    while True:
        if results_path.exists():
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Check if GRU has a valid RMSE (indicating it finished)
                # Or if XGBoost is done and GRU is done
                models = results.get('models', {})
                xgboost_done = 'rmse' in models.get('xgboost', {})
                gru_done = 'rmse' in models.get('gru', {})
                
                if xgboost_done and gru_done:
                    print("\nTraining detected as COMPLETE!")
                    print(f"XGBoost RMSE: {models['xgboost']['rmse']}")
                    print(f"GRU RMSE: {models['gru']['rmse']}")
                    return True
                
                # Also check for errors
                if 'error' in models.get('gru', {}):
                    print("\nGRU training failed with error!")
                    # If XGBoost is done, we can still proceed with just XGBoost
                    if xgboost_done:
                        print("Proceeding with XGBoost only...")
                        return True
            
            except Exception as e:
                print(f"Error reading results file: {e}")
        
        # Wait 60 seconds
        time.sleep(60)
        elapsed = (time.time() - start_time) / 3600
        print(f"Waiting... ({elapsed:.2f} hours elapsed)")

def run_ensemble_optimization():
    """Run ensemble optimization."""
    print("\n" + "="*80)
    print("RUNNING ENSEMBLE OPTIMIZATION")
    print("="*80)
    
    # Load data for validation
    loader = NFLDataLoader(CONFIG['data_dir'])
    # Load just week 18 for validation to save time/memory
    print("Loading validation data (Week 18)...")
    input_df, output_df = loader.load_all_training_data(weeks=[18])
    
    # Preprocess
    input_df = loader.preprocess_input_data(input_df)
    fe = FeatureEngineering()
    input_df = fe.add_physics_features(input_df)
    input_df = fe.add_temporal_features(input_df)
    
    # NEW: Scale features
    print("Scaling features...")
    scaler_path = Path(CONFIG['models_dir']) / 'scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        features_to_scale = [col for col in FEATURE_COLS if col in input_df.columns]
        input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])
        print("Features scaled successfully.")
    else:
        print("WARNING: Scaler not found! Optimization may be inaccurate.")
    
    # Initialize ensemble
    ensemble = ModelEnsemble(models_dir=CONFIG['models_dir'])
    ensemble.load_models(model_types=['xgboost', 'gru', 'baseline'])
    
    # Optimize
    best_weights, best_rmse = optimize_ensemble_weights(ensemble, input_df, output_df, FEATURE_COLS)
    
    # Save config
    ensemble.save_config(Path(CONFIG['models_dir']) / 'ensemble_config.json')
    
    return best_rmse, ensemble

def generate_submission(ensemble, weights):
    """
    Generate submission file for the competition.
    """
    print("\n" + "="*80)
    print("GENERATING SUBMISSION")
    print("="*80)
    
    data_dir = Path(CONFIG['data_dir'])
    test_input_path = data_dir / 'test_input.csv'
    test_template_path = data_dir / 'test.csv'
    submission_path = Path(CONFIG['outputs_dir']) / 'submission.csv'
    
    if not test_input_path.exists() or not test_template_path.exists():
        print("Test files not found. Skipping submission generation.")
        return
        
    print("Loading test data...")
    test_input = pd.read_csv(test_input_path)
    test_template = pd.read_csv(test_template_path)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test template shape: {test_template.shape}")
    
    # Preprocess
    print("Preprocessing test data...")
    loader = NFLDataLoader(CONFIG['data_dir'])
    test_input = loader.preprocess_input_data(test_input)
    
    fe = FeatureEngineering()
    test_input = fe.add_physics_features(test_input)
    test_input = fe.add_temporal_features(test_input)
    
    # NEW: Scale features
    print("Scaling features...")
    scaler_path = Path(CONFIG['models_dir']) / 'scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        features_to_scale = [col for col in FEATURE_COLS if col in test_input.columns]
        test_input[features_to_scale] = scaler.transform(test_input[features_to_scale])
        print("Features scaled successfully.")
    else:
        print("WARNING: Scaler not found! Predictions may be inaccurate.")
    
    # Generate predictions
    print("Predicting...")
    predictions = {}
    
    # Group by play and player
    # We need to predict for every player in the test input that is marked for prediction
    # But we only need to submit rows present in test_template
    
    # Create a lookup for quick access
    # (game_id, play_id, nfl_id) -> predicted_trajectory (N, 2)
    
    unique_players = test_input[['game_id', 'play_id', 'nfl_id', 'player_to_predict']].drop_duplicates()
    total_players = len(unique_players)
    
    print(f"Predicting for {total_players} players...")
    
    for idx, (_, row) in enumerate(unique_players.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total_players}")
            
        if not row['player_to_predict']:
            continue
            
        game_id, play_id, nfl_id = row['game_id'], row['play_id'], row['nfl_id']
        
        # Get player data
        player_data = test_input[
            (test_input['game_id'] == game_id) & 
            (test_input['play_id'] == play_id) & 
            (test_input['nfl_id'] == nfl_id)
        ].sort_values('frame_id')
        
        if len(player_data) == 0:
            continue
            
        # Determine how many frames to predict
        # We need to look at test_template to see max frame_id for this player
        # But for efficiency, let's just predict a fixed number (e.g. 50) and slice later
        # Or better, find max frame_id from template for this player
        
        # For now, predict 60 frames to be safe
        num_frames = 60
        
        try:
            pred = ensemble.predict_single_player(player_data, FEATURE_COLS, num_frames)
            predictions[(game_id, play_id, nfl_id)] = pred
        except Exception as e:
            print(f"Error predicting for {game_id}-{play_id}-{nfl_id}: {e}")
            
    # Fill submission template
    print("Filling submission template...")
    submission_rows = []
    
    for _, row in test_template.iterrows():
        game_id = row['game_id']
        play_id = row['play_id']
        nfl_id = row['nfl_id']
        frame_id = row['frame_id']
        row_id = row['id'] # Assuming 'id' column exists
        
        key = (game_id, play_id, nfl_id)
        
        x, y = 0.0, 0.0
        
        if key in predictions:
            pred_traj = predictions[key]
            # frame_id is 1-based index
            idx = int(frame_id) - 1
            if 0 <= idx < len(pred_traj):
                x, y = pred_traj[idx]
        
        submission_rows.append({
            'id': row_id,
            'x': x,
            'y': y
        })
        
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

def main():
    print(f"Finalize Project Script started at {datetime.now()}")
    
    # 1. Wait for training
    if wait_for_training_completion():
        
        # 2. Run Ensemble
        try:
            ensemble_rmse, ensemble = run_ensemble_optimization()
            print(f"\nFinal Ensemble RMSE: {ensemble_rmse:.4f}")
            
            # 3. Generate Submission
            generate_submission(ensemble, None)
            
        except Exception as e:
            print(f"Ensemble optimization/submission failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nProject Finalization Complete!")

if __name__ == '__main__':
    main()
