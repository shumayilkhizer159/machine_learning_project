"""
NFL Big Data Bowl 2026 - Kaggle Submission Notebook
This notebook contains the complete solution for the competition.

To use this in Kaggle:
1. Create a new Kaggle Notebook
2. Copy this code
3. Add the competition data as input
4. Run and submit
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Kaggle evaluation API
from kaggle_evaluation.nfl_big_data_bowl_2026_prediction import make_env

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '/kaggle/input/nfl-big-data-bowl-2026-prediction'
TRAIN_DIR = Path(DATA_DIR) / 'train'

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def preprocess_input_data(df):
    """
    Preprocess input tracking data with feature engineering.
    
    Args:
        df: Raw input dataframe
        
    Returns:
        Preprocessed dataframe with engineered features
    """
    df = df.copy()
    
    # Convert play direction to binary
    df['play_direction_binary'] = (df['play_direction'] == 'right').astype(int)
    
    # Calculate velocity components
    if 's' in df.columns and 'dir' in df.columns:
        df['v_x'] = df['s'] * np.cos(np.radians(df['dir']))
        df['v_y'] = df['s'] * np.sin(np.radians(df['dir']))
    else:
        df['v_x'] = 0
        df['v_y'] = 0
    
    # Distance to ball landing location
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        df['dist_to_ball_land'] = np.sqrt(
            (df['x'] - df['ball_land_x'])**2 + 
            (df['y'] - df['ball_land_y'])**2
        )
        
        # Direction to ball
        df['angle_to_ball'] = np.arctan2(
            df['ball_land_y'] - df['y'],
            df['ball_land_x'] - df['x']
        )
    else:
        df['dist_to_ball_land'] = 0
        df['angle_to_ball'] = 0
    
    # Encode categorical variables
    for col in ['player_position', 'player_side', 'player_role']:
        if col in df.columns:
            df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    
    return df

# ============================================================================
# MODEL: ENHANCED PHYSICS-BASED PREDICTOR
# ============================================================================

class EnhancedPhysicsModel:
    """
    Enhanced physics-based model for player trajectory prediction.
    
    This model combines:
    1. Current velocity (momentum)
    2. Attraction to ball landing location
    3. Player role-specific behavior
    4. Field boundary constraints
    """
    
    def __init__(self, fps=10.0):
        self.fps = fps
        self.dt = 1.0 / fps
    
    def predict_trajectory(self, player_data, num_frames):
        """
        Predict future trajectory for a single player.
        
        Args:
            player_data: DataFrame with player's input tracking data
            num_frames: Number of future frames to predict
            
        Returns:
            Array of shape (num_frames, 2) with predicted (x, y) positions
        """
        # Get last known state
        last_row = player_data.sort_values('frame_id').iloc[-1]
        
        # Extract features
        x = last_row['x']
        y = last_row['y']
        v_x = last_row.get('v_x', 0)
        v_y = last_row.get('v_y', 0)
        speed = last_row['s']
        ball_x = last_row['ball_land_x']
        ball_y = last_row['ball_land_y']
        player_role = last_row.get('player_role', 'Unknown')
        
        # Calculate direction to ball
        dx_to_ball = ball_x - x
        dy_to_ball = ball_y - y
        dist_to_ball = np.sqrt(dx_to_ball**2 + dy_to_ball**2)
        
        if dist_to_ball > 0.01:
            dir_x_to_ball = dx_to_ball / dist_to_ball
            dir_y_to_ball = dy_to_ball / dist_to_ball
        else:
            dir_x_to_ball = 0
            dir_y_to_ball = 0
        
        # Role-specific weights
        if 'Targeted Receiver' in player_role:
            # Receivers move strongly towards ball
            weight_momentum = 0.2
            weight_ball = 0.8
            speed_factor = 1.1  # Receivers accelerate
        elif 'Defensive Coverage' in player_role:
            # Defenders move towards ball but maintain coverage
            weight_momentum = 0.3
            weight_ball = 0.7
            speed_factor = 1.05
        elif 'Passer' in player_role:
            # Passer stays relatively stationary
            weight_momentum = 0.9
            weight_ball = 0.1
            speed_factor = 0.5
        else:
            # Other route runners maintain momentum
            weight_momentum = 0.5
            weight_ball = 0.5
            speed_factor = 0.95
        
        # Generate predictions
        predictions = []
        current_x, current_y = x, y
        current_v_x, current_v_y = v_x, v_y
        
        for frame in range(num_frames):
            # Velocity towards ball
            v_ball_x = dir_x_to_ball * speed * speed_factor
            v_ball_y = dir_y_to_ball * speed * speed_factor
            
            # Combined velocity
            combined_v_x = weight_momentum * current_v_x + weight_ball * v_ball_x
            combined_v_y = weight_momentum * current_v_y + weight_ball * v_ball_y
            
            # Update position
            current_x += combined_v_x * self.dt
            current_y += combined_v_y * self.dt
            
            # Apply field boundaries
            current_x = np.clip(current_x, 0, 120)
            current_y = np.clip(current_y, 0, 53.3)
            
            # Slow down as approaching ball
            dist_to_ball = np.sqrt((ball_x - current_x)**2 + (ball_y - current_y)**2)
            if dist_to_ball < 5:  # Within 5 yards of ball
                speed_factor *= 0.95
            
            # Update velocity for next iteration
            current_v_x = combined_v_x * 0.98  # Slight decay
            current_v_y = combined_v_y * 0.98
            
            predictions.append([current_x, current_y])
        
        return np.array(predictions)
    
    def predict_play(self, play_data):
        """
        Predict all players in a play.
        
        Args:
            play_data: DataFrame with input tracking data for the play
            
        Returns:
            Dictionary mapping nfl_id to predicted positions
        """
        predictions = {}
        
        for nfl_id in play_data['nfl_id'].unique():
            player_data = play_data[play_data['nfl_id'] == nfl_id]
            
            # Get number of frames to predict
            num_frames = int(player_data['num_frames_output'].iloc[0])
            
            # Make prediction
            pred = self.predict_trajectory(player_data, num_frames)
            predictions[nfl_id] = pred
        
        return predictions

# ============================================================================
# PREDICTION FUNCTION FOR KAGGLE API
# ============================================================================

# Initialize model
model = EnhancedPhysicsModel(fps=10.0)

def predict(test_input, sample_prediction):
    """
    Make predictions for the test set.
    
    This function is called by the Kaggle evaluation API.
    
    Args:
        test_input: DataFrame with input tracking data
        sample_prediction: DataFrame with the format for predictions
        
    Returns:
        DataFrame with predictions in the required format
    """
    # Preprocess input
    test_input = preprocess_input_data(test_input)
    
    # Get unique plays
    plays = test_input[['game_id', 'play_id']].drop_duplicates()
    
    # Store predictions
    all_predictions = []
    
    for _, play in plays.iterrows():
        game_id = play['game_id']
        play_id = play['play_id']
        
        # Get play data
        play_data = test_input[
            (test_input['game_id'] == game_id) &
            (test_input['play_id'] == play_id)
        ]
        
        # Make predictions
        play_predictions = model.predict_play(play_data)
        
        # Format predictions
        for nfl_id, positions in play_predictions.items():
            for frame_idx, (pred_x, pred_y) in enumerate(positions):
                all_predictions.append({
                    'game_id': game_id,
                    'play_id': play_id,
                    'nfl_id': nfl_id,
                    'frame_id': frame_idx + 1,
                    'x': pred_x,
                    'y': pred_y
                })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Merge with sample_prediction to ensure correct format
    result = sample_prediction.merge(
        predictions_df,
        on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
        how='left',
        suffixes=('', '_pred')
    )
    
    # Use predicted values
    result['x'] = result['x_pred'].fillna(result['x'])
    result['y'] = result['y_pred'].fillna(result['y'])
    
    # Keep only required columns
    result = result[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']]
    
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("NFL BIG DATA BOWL 2026 - SUBMISSION")
    print("="*80)
    
    # Create environment
    print("\nInitializing Kaggle environment...")
    env = make_env()
    iter_test = env.iter_test()
    
    # Process each test batch
    print("Making predictions...")
    for (test_input, sample_prediction) in iter_test:
        print(f"  Processing play: {test_input['game_id'].iloc[0]}, {test_input['play_id'].iloc[0]}")
        
        # Make predictions
        predictions = predict(test_input, sample_prediction)
        
        # Submit predictions
        env.predict(predictions)
    
    print("\n" + "="*80)
    print("SUBMISSION COMPLETE!")
    print("="*80)
