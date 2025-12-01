"""
Generate Submission using Fine-Tuned XGBoost Model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import NFLDataLoader, FeatureEngineering
from model_xgboost import XGBoostTrajectoryModel
from train_gpu import CONFIG, FEATURE_COLS

def generate_submission():
    print("Generating submission with Fine-Tuned XGBoost...")
    
    # 1. Load Data
    print("Loading test data...")
    loader = NFLDataLoader(CONFIG['data_dir'])
    test_df = pd.read_csv(Path(CONFIG['data_dir']) / 'test.csv')
    test_input_df = pd.read_csv(Path(CONFIG['data_dir']) / 'test_input.csv')
    
    print(f"Test input shape: {test_input_df.shape}")
    
    # 2. Preprocess
    print("Preprocessing...")
    # Apply the same preprocessing as training
    test_input_df = loader.preprocess_input_data(test_input_df)
    
    fe = FeatureEngineering()
    test_input_df = fe.add_physics_features(test_input_df)
    test_input_df = fe.add_temporal_features(test_input_df)
    
    # Scale
    print("Scaling features...")
    scaler_path = Path(CONFIG['models_dir']) / 'scaler.pkl'
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        features_to_scale = [col for col in FEATURE_COLS if col in test_input_df.columns]
        test_input_df[features_to_scale] = scaler.transform(test_input_df[features_to_scale])
    else:
        print("Warning: Scaler not found!")
        
    # 3. Load Model
    print("Loading XGBoost model...")
    model = XGBoostTrajectoryModel()
    model.load_models(Path(CONFIG['models_dir']) / 'xgboost')
    
    # 4. Predict
    print("Predicting...")
    predictions = []
    
    # Group by play/player
    # Only predict for players in test.csv
    # test.csv has columns: id, game_id, play_id, nfl_id, frame_id
    # We need to predict 30 frames for each player in test_input where player_to_predict=True?
    # Actually, test.csv defines exactly which frames we need to predict.
    # But our model predicts 30 frames forward.
    # We should predict 30 frames for each target player and then map to test.csv rows.
    
    # Get unique players to predict from test_input
    players_to_predict = test_input_df[test_input_df['player_to_predict'] == 1][
        ['game_id', 'play_id', 'nfl_id']
    ].drop_duplicates()
    
    print(f"Predicting for {len(players_to_predict)} players...")
    
    pred_dict = {}
    
    for idx, row in tqdm(players_to_predict.iterrows(), total=len(players_to_predict)):
        game_id, play_id, nfl_id = row['game_id'], row['play_id'], row['nfl_id']
        
        player_input = test_input_df[
            (test_input_df['game_id'] == game_id) &
            (test_input_df['play_id'] == play_id) &
            (test_input_df['nfl_id'] == nfl_id)
        ]
        
        if len(player_input) == 0:
            continue
            
        # Predict 30 frames (or more if needed, but competition usually asks for specific frames)
        # The model predicts next 30 frames.
        pred_traj = model.predict(player_input, FEATURE_COLS, num_frames=60) # Predict up to 60 just in case
        
        pred_dict[(game_id, play_id, nfl_id)] = pred_traj
        
    # 5. Format Submission
    print("Formatting submission...")
    submission_rows = []
    
    # We need to fill in x, y for each row in test.csv
    # test.csv structure: id, game_id, play_id, nfl_id, frame_id
    # frame_id in test.csv is the absolute frame id.
    # We need to know the start frame of the prediction.
    # The input data (test_input.csv) contains the history up to the point of prediction.
    # The last frame in player_input is the "current" frame (t=0).
    # So frame_id in test.csv corresponds to t=1, t=2, etc. relative to the input?
    # Usually, test.csv frame_id is absolute.
    # We need to find the last frame in input for each player to calculate offset.
    
    # Let's create a map of last_frame for each player
    last_frames = {}
    for idx, row in players_to_predict.iterrows():
        game_id, play_id, nfl_id = row['game_id'], row['play_id'], row['nfl_id']
        player_input = test_input_df[
            (test_input_df['game_id'] == game_id) &
            (test_input_df['play_id'] == play_id) &
            (test_input_df['nfl_id'] == nfl_id)
        ]
        last_frames[(game_id, play_id, nfl_id)] = player_input['frame_id'].max()
        
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        game_id = row['game_id']
        play_id = row['play_id']
        nfl_id = row['nfl_id']
        frame_id = row['frame_id']
        
        key = (game_id, play_id, nfl_id)
        if key in pred_dict and key in last_frames:
            start_frame = last_frames[key]
            offset = frame_id - start_frame
            
            if 1 <= offset <= 60:
                pred_x, pred_y = pred_dict[key][offset-1]
            else:
                # Fallback or error
                pred_x, pred_y = 0, 0 # Should not happen if we predict enough frames
        else:
            pred_x, pred_y = 0, 0
            
        submission_rows.append({
            'id': row['id'],
            'x': pred_x,
            'y': pred_y
        })
        
    submission_df = pd.DataFrame(submission_rows)
    
    # Save
    save_path = Path(CONFIG['outputs_dir']) / 'submission_xgboost_advanced.csv'
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

if __name__ == "__main__":
    generate_submission()
