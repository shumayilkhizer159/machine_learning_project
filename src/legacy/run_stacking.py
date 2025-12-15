"""
Run Stacking Ensemble
Orchestrates the training of the stacking meta-model and generation of the submission.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model_stacking import StackingEnsemble
from data_loader import NFLDataLoader, FeatureEngineering
from train_gpu import CONFIG, FEATURE_COLS

def main():
    print(f"Stacking Ensemble Pipeline started at {datetime.now()}")
    
    # 1. Load Validation Data (Week 18)
    print("\nLoading validation data...")
    loader = NFLDataLoader(CONFIG['data_dir'])
    val_input, val_output = loader.load_all_training_data(weeks=[18])
    
    # Sample validation data for faster meta-model training
    print("Sampling validation data (100 plays)...")
    try:
        sample_plays = val_input[['game_id', 'play_id']].drop_duplicates().sample(n=100, random_state=42)
        val_input = val_input.merge(sample_plays, on=['game_id', 'play_id'])
        val_output = val_output.merge(sample_plays, on=['game_id', 'play_id'])
        print(f"Sampled {len(val_input)} input rows")
    except ValueError:
        print("Validation set smaller than sample size, using all data.")
    
    # Preprocess
    val_input = loader.preprocess_input_data(val_input)
    fe = FeatureEngineering()
    val_input = fe.add_physics_features(val_input)
    val_input = fe.add_temporal_features(val_input)
    
    # Scale
    print("Scaling validation features...")
    scaler_path = Path(CONFIG['models_dir']) / 'scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        features_to_scale = [col for col in FEATURE_COLS if col in val_input.columns]
        val_input[features_to_scale] = scaler.transform(val_input[features_to_scale])
    else:
        print("WARNING: Scaler not found!")
        return

    # 2. Train Stacking Model
    stacking = StackingEnsemble(CONFIG['models_dir'])
    stacking.train(val_input, val_output, FEATURE_COLS)
    
    # 3. Generate Submission
    print("\nGenerating Stacking Submission...")
    test_input_path = Path(CONFIG['data_dir']) / 'test_input.csv'
    test_template_path = Path(CONFIG['data_dir']) / 'test.csv'
    submission_path = Path(CONFIG['outputs_dir']) / 'submission_stacking.csv'
    
    test_input = pd.read_csv(test_input_path)
    test_template = pd.read_csv(test_template_path)
    
    # Preprocess Test
    test_input = loader.preprocess_input_data(test_input)
    test_input = fe.add_physics_features(test_input)
    test_input = fe.add_temporal_features(test_input)
    
    # Scale Test
    if scaler_path.exists():
        test_input[features_to_scale] = scaler.transform(test_input[features_to_scale])
        
    # Predict
    predictions = {}
    unique_players = test_input[['game_id', 'play_id', 'nfl_id', 'player_to_predict']].drop_duplicates()
    total = len(unique_players)
    
    print(f"Predicting for {total} players...")
    for idx, (_, row) in enumerate(unique_players.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{total}")
            
        if not row['player_to_predict']:
            continue
            
        game_id, play_id, nfl_id = row['game_id'], row['play_id'], row['nfl_id']
        
        player_data = test_input[
            (test_input['game_id'] == game_id) & 
            (test_input['play_id'] == play_id) & 
            (test_input['nfl_id'] == nfl_id)
        ].sort_values('frame_id')
        
        if len(player_data) == 0:
            continue
            
        # Predict 60 frames
        try:
            pred = stacking.predict(player_data, FEATURE_COLS, num_frames=60)
            predictions[(game_id, play_id, nfl_id)] = pred
        except Exception as e:
            print(f"Error: {e}")
            continue
            
    # Fill Template
    submission_rows = []
    for _, row in test_template.iterrows():
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        x, y = 0.0, 0.0
        if key in predictions:
            pred_traj = predictions[key]
            idx = int(row['frame_id']) - 1
            if 0 <= idx < len(pred_traj):
                x, y = pred_traj[idx]
        
        submission_rows.append({'id': row['id'], 'x': x, 'y': y})
        
    pd.DataFrame(submission_rows).to_csv(submission_path, index=False)
    print(f"Stacking submission saved to {submission_path}")

if __name__ == "__main__":
    main()
