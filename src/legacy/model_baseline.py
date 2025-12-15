"""
Baseline Physics-Based Model for NFL Player Trajectory Prediction
This model uses simple physics (constant velocity) to predict future positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class PhysicsBaselineModel:
    """
    Simple physics-based baseline model.
    
    Predicts future positions using:
    1. Constant velocity assumption
    2. Direction towards ball landing location
    3. Simple linear extrapolation
    """
    
    def __init__(self, name='physics_baseline'):
        self.name = name
        self.predictions = {}
    
    def predict_constant_velocity(self, last_x: float, last_y: float, 
                                  v_x: float, v_y: float, 
                                  num_frames: int, fps: float = 10.0) -> np.ndarray:
        """
        Predict future positions assuming constant velocity.
        
        Args:
            last_x, last_y: Last known position
            v_x, v_y: Velocity components (yards/second)
            num_frames: Number of frames to predict
            fps: Frames per second (default 10)
            
        Returns:
            Array of shape (num_frames, 2) with predicted (x, y) positions
        """
        dt = 1.0 / fps  # Time step in seconds
        
        predictions = []
        for i in range(1, num_frames + 1):
            pred_x = last_x + v_x * dt * i
            pred_y = last_y + v_y * dt * i
            predictions.append([pred_x, pred_y])
        
        return np.array(predictions)
    
    def predict_towards_ball(self, last_x: float, last_y: float,
                            ball_x: float, ball_y: float,
                            speed: float, num_frames: int, 
                            fps: float = 10.0) -> np.ndarray:
        """
        Predict positions assuming player moves towards ball at constant speed.
        
        Args:
            last_x, last_y: Last known position
            ball_x, ball_y: Ball landing position
            speed: Player speed (yards/second)
            num_frames: Number of frames to predict
            fps: Frames per second
            
        Returns:
            Array of shape (num_frames, 2) with predicted positions
        """
        # Calculate direction to ball
        dx = ball_x - last_x
        dy = ball_y - last_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 0.01:  # Already at ball location
            return np.array([[last_x, last_y]] * num_frames)
        
        # Unit direction vector
        dir_x = dx / distance
        dir_y = dy / distance
        
        # Velocity components
        v_x = dir_x * speed
        v_y = dir_y * speed
        
        dt = 1.0 / fps
        predictions = []
        
        for i in range(1, num_frames + 1):
            pred_x = last_x + v_x * dt * i
            pred_y = last_y + v_y * dt * i
            
            # Clip to stay within field boundaries
            pred_x = np.clip(pred_x, 0, 120)
            pred_y = np.clip(pred_y, 0, 53.3)
            
            predictions.append([pred_x, pred_y])
        
        return np.array(predictions)
    
    def predict_weighted_combination(self, last_x: float, last_y: float,
                                    v_x: float, v_y: float,
                                    ball_x: float, ball_y: float,
                                    speed: float, num_frames: int,
                                    player_role: str = 'Unknown',
                                    fps: float = 10.0) -> np.ndarray:
        """
        Predict using weighted combination of velocity and ball direction.
        
        Args:
            last_x, last_y: Last known position
            v_x, v_y: Current velocity components
            ball_x, ball_y: Ball landing position
            speed: Player speed
            num_frames: Number of frames to predict
            player_role: Player's role (affects weighting)
            fps: Frames per second
            
        Returns:
            Array of predicted positions
        """
        # Get predictions from both methods
        vel_pred = self.predict_constant_velocity(last_x, last_y, v_x, v_y, num_frames, fps)
        ball_pred = self.predict_towards_ball(last_x, last_y, ball_x, ball_y, speed, num_frames, fps)
        
        # Weight based on player role
        if 'Targeted Receiver' in player_role:
            # Receivers move strongly towards ball
            weight_ball = 0.7
        elif 'Defensive Coverage' in player_role:
            # Defenders also move towards ball but maintain some momentum
            weight_ball = 0.6
        elif 'Passer' in player_role:
            # Passer doesn't move much after throw
            weight_ball = 0.2
        else:
            # Other route runners maintain momentum
            weight_ball = 0.4
        
        weight_vel = 1.0 - weight_ball
        
        # Weighted combination
        predictions = weight_vel * vel_pred + weight_ball * ball_pred
        
        # Clip to field boundaries
        predictions[:, 0] = np.clip(predictions[:, 0], 0, 120)
        predictions[:, 1] = np.clip(predictions[:, 1], 0, 53.3)
        
        return predictions
    
    def predict_play(self, input_data: pd.DataFrame, 
                    num_frames: int,
                    method: str = 'weighted') -> Dict:
        """
        Predict all players in a play.
        
        Args:
            input_data: DataFrame with input tracking data for one play
            num_frames: Number of frames to predict
            method: Prediction method ('velocity', 'ball', or 'weighted')
            
        Returns:
            Dictionary mapping nfl_id to predicted positions
        """
        predictions = {}
        
        for nfl_id in input_data['nfl_id'].unique():
            player_data = input_data[input_data['nfl_id'] == nfl_id].sort_values('frame_id')
            
            # Get last known state
            last_row = player_data.iloc[-1]
            last_x = last_row['x']
            last_y = last_row['y']
            speed = last_row['s']
            
            # Calculate velocity components if available
            if 'dir' in player_data.columns:
                direction = np.radians(last_row['dir'])
                v_x = speed * np.cos(direction)
                v_y = speed * np.sin(direction)
            else:
                v_x = 0
                v_y = 0
            
            # Get ball landing location
            ball_x = last_row['ball_land_x']
            ball_y = last_row['ball_land_y']
            
            # Get player role
            player_role = last_row.get('player_role', 'Unknown')
            
            # Make prediction based on method
            if method == 'velocity':
                pred = self.predict_constant_velocity(last_x, last_y, v_x, v_y, num_frames)
            elif method == 'ball':
                pred = self.predict_towards_ball(last_x, last_y, ball_x, ball_y, speed, num_frames)
            else:  # weighted
                pred = self.predict_weighted_combination(
                    last_x, last_y, v_x, v_y, ball_x, ball_y, 
                    speed, num_frames, player_role
                )
            
            predictions[nfl_id] = pred
        
        return predictions
    
    def evaluate(self, predictions: Dict, ground_truth: pd.DataFrame) -> float:
        """
        Calculate RMSE for predictions.
        
        Args:
            predictions: Dictionary mapping nfl_id to predicted positions
            ground_truth: DataFrame with actual positions
            
        Returns:
            RMSE score
        """
        squared_errors = []
        
        for nfl_id, pred_positions in predictions.items():
            # Get ground truth for this player
            gt = ground_truth[ground_truth['nfl_id'] == nfl_id].sort_values('frame_id')
            
            if len(gt) == 0:
                continue
            
            # Match predictions to ground truth frames
            for i, (_, row) in enumerate(gt.iterrows()):
                if i >= len(pred_positions):
                    break
                
                pred_x, pred_y = pred_positions[i]
                true_x, true_y = row['x'], row['y']
                
                # Calculate squared error
                error = (pred_x - true_x)**2 + (pred_y - true_y)**2
                squared_errors.append(error)
        
        if len(squared_errors) == 0:
            return float('inf')
        
        # RMSE formula from competition
        rmse = np.sqrt(np.mean(squared_errors) / 2)
        return rmse


def evaluate_baseline_models(data_dir='/home/ubuntu/nfl_project/data'):
    """
    Evaluate different baseline model variants.
    """
    from pathlib import Path
    
    print("=" * 80)
    print("BASELINE MODEL EVALUATION")
    print("=" * 80)
    
    # Load sample data
    train_dir = Path(data_dir) / 'train'
    input_df = pd.read_csv(train_dir / 'input_2023_w01.csv')
    output_df = pd.read_csv(train_dir / 'output_2023_w01.csv')
    
    # Sample a few plays for quick evaluation
    sample_plays = input_df[['game_id', 'play_id']].drop_duplicates().head(50)
    
    model = PhysicsBaselineModel()
    
    methods = ['velocity', 'ball', 'weighted']
    results = {}
    
    for method in methods:
        print(f"\nEvaluating method: {method}")
        rmse_scores = []
        
        for _, play in sample_plays.iterrows():
            game_id = play['game_id']
            play_id = play['play_id']
            
            # Get input and output for this play
            play_input = input_df[(input_df['game_id'] == game_id) & 
                                 (input_df['play_id'] == play_id)]
            play_output = output_df[(output_df['game_id'] == game_id) & 
                                   (output_df['play_id'] == play_id)]
            
            if len(play_output) == 0:
                continue
            
            # Get number of frames to predict
            num_frames = play_input['num_frames_output'].max()
            
            # Make predictions
            predictions = model.predict_play(play_input, num_frames, method=method)
            
            # Evaluate
            rmse = model.evaluate(predictions, play_output)
            rmse_scores.append(rmse)
        
        avg_rmse = np.mean(rmse_scores)
        results[method] = avg_rmse
        print(f"  Average RMSE: {avg_rmse:.4f}")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    for method, rmse in results.items():
        print(f"{method:20s}: {rmse:.4f}")
    
    best_method = min(results, key=results.get)
    print(f"\nBest method: {best_method} (RMSE: {results[best_method]:.4f})")
    
    return results


if __name__ == '__main__':
    results = evaluate_baseline_models()
