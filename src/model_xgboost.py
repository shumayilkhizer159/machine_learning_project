"""
XGBoost-based Model for NFL Player Trajectory Prediction
This model uses gradient boosting to predict future positions frame by frame.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available.")


class XGBoostTrajectoryModel:
    """
    XGBoost model for trajectory prediction.
    
    Strategy: Train separate models for x and y coordinates at each future time step.
    """
    
    def __init__(self, max_future_frames=30, **xgb_params):
        """
        Args:
            max_future_frames: Maximum number of future frames to predict
            xgb_params: Parameters for XGBoost model
        """
        self.max_future_frames = max_future_frames
        self.models_x = {}  # Models for x coordinate at each frame
        self.models_y = {}  # Models for y coordinate at each frame
        
        # Default XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Load optimized parameters if available
        # In notebook environment, we skip this auto-loading.
        # Parameters will be loaded via load_models() or passed explicitly.
        pass

                
        self.xgb_params.update(xgb_params)
    
    def create_features_for_frame(self, input_df: pd.DataFrame, 
                                  frame_offset: int) -> pd.DataFrame:
        """
        Create features for predicting a specific future frame.
        
        Args:
            input_df: Input tracking data
            frame_offset: Which future frame to predict (1, 2, 3, ...)
            
        Returns:
            DataFrame with features
        """
        features = input_df.copy()
        
        # Add frame offset as a feature
        features['frame_offset'] = frame_offset
        
        # Calculate time to target frame (in seconds)
        features['time_to_frame'] = frame_offset / 10.0  # 10 fps
        
        # Predicted position based on constant velocity
        if 'v_x' in features.columns and 'v_y' in features.columns:
            features['predicted_x_cv'] = features['x'] + features['v_x'] * features['time_to_frame']
            features['predicted_y_cv'] = features['y'] + features['v_y'] * features['time_to_frame']
        
        # Distance and direction to ball at predicted time
        if 'ball_land_x' in features.columns:
            features['predicted_dist_to_ball'] = np.sqrt(
                (features['predicted_x_cv'] - features['ball_land_x'])**2 +
                (features['predicted_y_cv'] - features['ball_land_y'])**2
            )
        
        return features
    
    def prepare_training_data(self, input_df: pd.DataFrame, 
                            output_df: pd.DataFrame,
                            feature_cols: List[str]) -> Dict[int, Tuple]:
        """
        Prepare training data for each future frame.
        
        Args:
            input_df: Input tracking data
            output_df: Output tracking data with targets
            feature_cols: List of feature columns
            
        Returns:
            Dictionary mapping frame_offset to (X, y_x, y_y) tuples
        """
        data_by_frame = {}
        
        # Group by play and player
        # Group by play and player
        # Optimize: Pre-group output_df to avoid repeated filtering
        output_groups = dict(list(output_df.groupby(['game_id', 'play_id', 'nfl_id'])))
        
        for (game_id, play_id, nfl_id), input_group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
            # Only include players to predict
            if not input_group['player_to_predict'].iloc[0]:
                continue
            
            # Get last input frame
            last_input = input_group.sort_values('frame_id').iloc[-1]
            
            # Get output frames from pre-grouped dict
            if (game_id, play_id, nfl_id) not in output_groups:
                continue
                
            output_group = output_groups[(game_id, play_id, nfl_id)].sort_values('frame_id')
            
            if len(output_group) == 0:
                continue
            
            # For each output frame
            for i, (_, output_row) in enumerate(output_group.iterrows()):
                frame_offset = i + 1  # 1-indexed
                
                if frame_offset > self.max_future_frames:
                    break
                
                # Create features
                features_df = self.create_features_for_frame(
                    pd.DataFrame([last_input]), frame_offset
                )
                
                # Extract feature values
                try:
                    X_row = features_df[feature_cols].values[0]
                    y_x = output_row['x']
                    y_y = output_row['y']
                    
                    if frame_offset not in data_by_frame:
                        data_by_frame[frame_offset] = {'X': [], 'y_x': [], 'y_y': []}
                    
                    data_by_frame[frame_offset]['X'].append(X_row)
                    data_by_frame[frame_offset]['y_x'].append(y_x)
                    data_by_frame[frame_offset]['y_y'].append(y_y)
                except KeyError as e:
                    continue
        
        # Convert to arrays
        result = {}
        for frame_offset, data in data_by_frame.items():
            result[frame_offset] = (
                np.array(data['X']),
                np.array(data['y_x']),
                np.array(data['y_y'])
            )
        
        return result
    
    def train(self, training_data: Dict[int, Tuple], verbose=True):
        """
        Train XGBoost models for each future frame.
        
        Args:
            training_data: Dictionary from prepare_training_data
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Training XGBoost models for {len(training_data)} future frames...")
        
        for frame_offset, (X, y_x, y_y) in training_data.items():
            if verbose:
                print(f"  Frame +{frame_offset}: {len(X)} samples")
            
            # Split data
            X_train, X_val, y_x_train, y_x_val = train_test_split(
                X, y_x, test_size=0.2, random_state=42
            )
            _, _, y_y_train, y_y_val = train_test_split(
                X, y_y, test_size=0.2, random_state=42
            )
            
            # Train model for x coordinate
            model_x = xgb.XGBRegressor(**self.xgb_params)
            model_x.fit(
                X_train, y_x_train,
                eval_set=[(X_val, y_x_val)],
                verbose=False
            )
            self.models_x[frame_offset] = model_x
            
            # Train model for y coordinate
            model_y = xgb.XGBRegressor(**self.xgb_params)
            model_y.fit(
                X_train, y_y_train,
                eval_set=[(X_val, y_y_val)],
                verbose=False
            )
            self.models_y[frame_offset] = model_y
            
            if verbose:
                # Calculate validation RMSE
                pred_x = model_x.predict(X_val)
                pred_y = model_y.predict(X_val)
                rmse = np.sqrt(((pred_x - y_x_val)**2 + (pred_y - y_y_val)**2).mean() / 2)
                print(f"    Validation RMSE: {rmse:.4f}")
        
        if verbose:
            print("Training complete!")
    
    def predict(self, input_df: pd.DataFrame, feature_cols: List[str],
               num_frames: int) -> np.ndarray:
        """
        Predict future positions (Single Player).
        Wrapper around predict_batch for compatibility.
        """
        # Ensure input is a DataFrame with one row (last frame)
        if len(input_df) > 1:
             last_input = input_df.sort_values('frame_id').iloc[[-1]]
        else:
             last_input = input_df
             
        # Add dummy index for batch processing
        last_input = last_input.copy()
        
        # Predict
        batch_preds = self.predict_batch(last_input, feature_cols, num_frames)
        
        # Extract result
        key = (last_input['game_id'].iloc[0], last_input['play_id'].iloc[0], last_input['nfl_id'].iloc[0])
        return batch_preds.get(key, np.zeros((num_frames, 2)))

    def predict_batch(self, last_input_df: pd.DataFrame, feature_cols: List[str],
                     num_frames: int) -> Dict[Tuple, np.ndarray]:
        """
        Predict future positions for multiple players simultaneously.
        
        Args:
            last_input_df: DataFrame containing the LAST frame for each player.
            feature_cols: List of feature columns
            num_frames: Number of frames to predict
            
        Returns:
            Dictionary mapping (game_id, play_id, nfl_id) -> Array of shape (num_frames, 2)
        """
        # Initialize results dictionary
        results = {}
        keys = []
        for _, row in last_input_df.iterrows():
            key = (row['game_id'], row['play_id'], row['nfl_id'])
            results[key] = np.zeros((num_frames, 2))
            keys.append(key)
            
        # We will update a working DataFrame frame by frame
        current_df = last_input_df.copy()
        
        # Pre-calculate constant velocity components if available
        if 'v_x' in current_df.columns and 'v_y' in current_df.columns:
            v_x = current_df['v_x'].values
            v_y = current_df['v_y'].values
        else:
            v_x = np.zeros(len(current_df))
            v_y = np.zeros(len(current_df))
            
        start_x = current_df['x'].values
        start_y = current_df['y'].values
        
        for frame_offset in range(1, num_frames + 1):
            # Create features for this batch
            # We use the helper but need to ensure it handles batches
            features_df = self.create_features_for_frame(current_df, frame_offset)
            
            X = features_df[feature_cols].values
            
            # Predict
            if frame_offset in self.models_x and frame_offset in self.models_y:
                pred_x = self.models_x[frame_offset].predict(X)
                pred_y = self.models_y[frame_offset].predict(X)
            else:
                # Fallback: Constant Velocity
                time_delta = frame_offset / 10.0
                pred_x = start_x + v_x * time_delta
                pred_y = start_y + v_y * time_delta
            
            # Clip
            pred_x = np.clip(pred_x, 0, 120)
            pred_y = np.clip(pred_y, 0, 53.3)
            
            # Store predictions
            try:
                # Ensure predictions are arrays
                if not isinstance(pred_x, np.ndarray):
                    pred_x = np.array(pred_x)
                if not isinstance(pred_y, np.ndarray):
                    pred_y = np.array(pred_y)
                    
                for i, key in enumerate(keys):
                    # Defensive casting to int
                    idx_frame = int(frame_offset - 1)
                    idx_player = int(i)
                    
                    results[key][idx_frame] = [pred_x[idx_player], pred_y[idx_player]]
            except Exception as e:
                print(f"ERROR in loop: {e}")
                print(f"frame_offset: {frame_offset}, type: {type(frame_offset)}")
                print(f"i: {i}, type: {type(i)}")
                print(f"pred_x type: {type(pred_x)}")
                print(f"pred_x shape: {getattr(pred_x, 'shape', 'N/A')}")
                # Don't raise, just skip this frame/player to keep running
                continue
                
        # Apply smoothing to all trajectories
        for key in results:
            results[key] = self.smooth_trajectory(results[key])
            
        return results
    
    def smooth_trajectory(self, trajectory: np.ndarray, window_size=5) -> np.ndarray:
        """
        Apply smoothing to the predicted trajectory.
        Uses a simple moving average.
        """
        if len(trajectory) < window_size:
            return trajectory
            
        smoothed = trajectory.copy()
        
        # Simple Moving Average for x and y
        for i in range(2):  # x and y
            # Use pandas rolling mean for convenience if available, else numpy
            series = pd.Series(trajectory[:, i])
            # Min_periods=1 ensures we don't get NaNs at the start
            smoothed[:, i] = series.rolling(window=window_size, min_periods=1, center=True).mean().values
            
        return smoothed
    
    def save_models(self, save_dir: str):
        """Save trained models."""
        import pickle
        from pathlib import Path
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save models
        for frame_offset, model in self.models_x.items():
            model.save_model(save_dir / f'xgb_x_frame_{frame_offset}.json')
        
        for frame_offset, model in self.models_y.items():
            model.save_model(save_dir / f'xgb_y_frame_{frame_offset}.json')
        
        # Save metadata
        metadata = {
            'max_future_frames': self.max_future_frames,
            'xgb_params': self.xgb_params,
            'trained_frames': list(self.models_x.keys())
        }
        with open(save_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_models(self, save_dir: str):
        """Load trained models."""
        import pickle
        from pathlib import Path
        
        save_dir = Path(save_dir)
        
        # Load metadata
        with open(save_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.max_future_frames = metadata['max_future_frames']
        self.xgb_params = metadata['xgb_params']
        
        # Load models
        for frame_offset in metadata['trained_frames']:
            model_x = xgb.XGBRegressor()
            model_x.load_model(save_dir / f'xgb_x_frame_{frame_offset}.json')
            self.models_x[frame_offset] = model_x
            
            model_y = xgb.XGBRegressor()
            model_y.load_model(save_dir / f'xgb_y_frame_{frame_offset}.json')
            self.models_y[frame_offset] = model_y


if __name__ == '__main__':
    if not XGBOOST_AVAILABLE:
        print("XGBoost is required. Please install: pip install xgboost")
    else:
        print("XGBoost model module loaded successfully")
