
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_xgboost import XGBoostTrajectoryModel

def test_predict_batch():
    print("Setting up test data...")
    
    # Create dummy feature columns based on logs
    feature_cols = [
        'x', 'y', 's', 'a', 'dir', 'o',
        'ball_land_x', 'ball_land_y', 'dist_to_ball_land',
        'v_x', 'v_y',
        'player_position_encoded', 'player_side_encoded', 'player_role_encoded',
        'player_height_inches', 'player_weight', 'player_age',
        'play_direction_binary', 'absolute_yardline_number'
    ]
    
    # Create dummy input dataframe (last frame)
    # Based on DEBUG: Input Columns
    input_cols = [
        'game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id', 
        'play_direction', 'absolute_yardline_number', 'player_name', 
        'player_height', 'player_weight', 'player_birth_date', 'player_position', 
        'player_side', 'player_role', 'x', 'y', 's', 'a', 'dir', 'o', 
        'num_frames_output', 'ball_land_x', 'ball_land_y', 
        'v_x', 'v_y' # Added these as they are expected
    ]
    
    # Create 5 dummy players
    data = []
    for i in range(5):
        row = {
            'game_id': 1,
            'play_id': 1,
            'player_to_predict': 1,
            'nfl_id': 100 + i,
            'frame_id': 10,
            'x': 50.0 + i,
            'y': 25.0 + i,
            's': 5.0,
            'a': 1.0,
            'dir': 90.0,
            'o': 90.0,
            'v_x': 5.0,
            'v_y': 0.0,
            'ball_land_x': 60.0,
            'ball_land_y': 25.0,
            # Add other required cols with dummy values
            'player_position_encoded': 0,
            'player_side_encoded': 0,
            'player_role_encoded': 0,
            'player_height_inches': 70,
            'player_weight': 200,
            'player_age': 25,
            'play_direction_binary': 0,
            'absolute_yardline_number': 50,
            'dist_to_ball_land': 10.0
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    print("Initializing model...")
    model = XGBoostTrajectoryModel()
    
    # We don't need trained models for the fallback path (which is where the error likely is or at least the loop structure)
    # But to test the model path, we'd need models. 
    # The error reported "only integers..." suggests it might be in the fallback or the storage logic.
    
    print("Calling predict_batch...")
    try:
        results = model.predict_batch(df, feature_cols, num_frames=60)
        print("✅ predict_batch successful!")
        print(f"Result keys: {list(results.keys())}")
        print(f"Shape of first result: {results[list(results.keys())[0]].shape}")
    except Exception as e:
        print(f"❌ predict_batch FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predict_batch()
