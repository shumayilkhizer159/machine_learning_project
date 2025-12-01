"""
Data Loading and Preprocessing Module for NFL Big Data Bowl 2026
This module handles loading, cleaning, and preprocessing of NFL tracking data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class NFLDataLoader:
    """
    Handles loading and preprocessing of NFL tracking data.
    
    The data consists of:
    - Input files: Player tracking data BEFORE the pass is thrown
    - Output files: Player positions AFTER the pass (targets to predict)
    """
    
    def __init__(self, data_dir: str = '/kaggle/input/nfl-big-data-bowl-2026-prediction'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing the competition data
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / 'train'
        
    def load_week_data(self, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load input and output data for a specific week.
        
        Args:
            week: Week number (1-18)
            
        Returns:
            Tuple of (input_df, output_df)
        """
        input_file = self.train_dir / f'input_2023_w{week:02d}.csv'
        output_file = self.train_dir / f'output_2023_w{week:02d}.csv'
        
        input_df = pd.read_csv(input_file)
        output_df = pd.read_csv(output_file)
        
        return input_df, output_df
    
    def load_all_training_data(self, weeks: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and concatenate data from multiple weeks.
        
        Args:
            weeks: List of week numbers to load. If None, loads all weeks 1-18.
            
        Returns:
            Tuple of (combined_input_df, combined_output_df)
        """
        if weeks is None:
            weeks = range(1, 19)  # Weeks 1-18
        
        input_dfs = []
        output_dfs = []
        
        print(f"Loading data from {len(weeks)} weeks...")
        for week in weeks:
            try:
                input_df, output_df = self.load_week_data(week)
                input_dfs.append(input_df)
                output_dfs.append(output_df)
                print(f"  Week {week}: {len(input_df)} input rows, {len(output_df)} output rows")
            except FileNotFoundError:
                print(f"  Week {week}: Files not found, skipping...")
                continue
        
        combined_input = pd.concat(input_dfs, ignore_index=True)
        combined_output = pd.concat(output_dfs, ignore_index=True)
        
        print(f"\nTotal: {len(combined_input)} input rows, {len(combined_output)} output rows")
        return combined_input, combined_output
    
    def preprocess_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input tracking data.
        
        Args:
            df: Raw input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        # Convert play_direction to binary (left=0, right=1)
        df['play_direction_binary'] = (df['play_direction'] == 'right').astype(int)
        
        # Parse player height to inches
        if 'player_height' in df.columns:
            df['player_height_inches'] = df['player_height'].apply(self._height_to_inches)
        
        # Calculate age from birth date
        if 'player_birth_date' in df.columns:
            df['player_birth_date'] = pd.to_datetime(df['player_birth_date'], errors='coerce')
            df['player_age'] = (pd.Timestamp('2023-01-01') - df['player_birth_date']).dt.days / 365.25
        
        # Encode categorical variables
        df = self._encode_categorical_features(df)
        
        # Calculate distance to ball landing location
        if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
            df['dist_to_ball_land'] = np.sqrt(
                (df['x'] - df['ball_land_x'])**2 + 
                (df['y'] - df['ball_land_y'])**2
            )
        
        # Calculate velocity components
        if 's' in df.columns and 'dir' in df.columns:
            df['v_x'] = df['s'] * np.cos(np.radians(df['dir']))
            df['v_y'] = df['s'] * np.sin(np.radians(df['dir']))
        
        return df
    
    def _height_to_inches(self, height_str: str) -> float:
        """Convert height string (e.g., '6-2') to inches."""
        try:
            if pd.isna(height_str):
                return np.nan
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return np.nan
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        categorical_cols = ['player_position', 'player_side', 'player_role']
        
        for col in categorical_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        
        return df
    
    def create_play_sequences(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> Dict:
        """
        Organize data into sequences for each play and player.
        
        Args:
            input_df: Preprocessed input dataframe
            output_df: Output dataframe with targets
            
        Returns:
            Dictionary mapping (game_id, play_id, nfl_id) to sequence data
        """
        sequences = {}
        
        # Group by play and player
        for (game_id, play_id, nfl_id), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
            # Sort by frame_id to ensure temporal order
            group = group.sort_values('frame_id')
            
            # Get corresponding output data
            output_mask = (
                (output_df['game_id'] == game_id) & 
                (output_df['play_id'] == play_id) & 
                (output_df['nfl_id'] == nfl_id)
            )
            output_group = output_df[output_mask].sort_values('frame_id')
            
            # Only include if we have both input and output
            if len(group) > 0 and len(output_group) > 0:
                sequences[(game_id, play_id, nfl_id)] = {
                    'input': group,
                    'output': output_group,
                    'player_to_predict': group['player_to_predict'].iloc[0] if 'player_to_predict' in group.columns else True,
                    'num_frames_output': group['num_frames_output'].iloc[0] if 'num_frames_output' in group.columns else len(output_group)
                }
        
        return sequences


class FeatureEngineering:
    """
    Advanced feature engineering for NFL tracking data.
    """
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features based on frame sequences.
        
        Args:
            df: Input dataframe with frame_id
            
        Returns:
            Dataframe with additional temporal features
        """
        df = df.copy()
        
        # Group by play and player
        for (game_id, play_id, nfl_id), group in df.groupby(['game_id', 'play_id', 'nfl_id']):
            idx = group.index
            
            # Calculate changes over time (velocity approximation)
            if len(group) > 1:
                df.loc[idx, 'dx'] = group['x'].diff().fillna(0)
                df.loc[idx, 'dy'] = group['y'].diff().fillna(0)
                df.loc[idx, 'ds'] = group['s'].diff().fillna(0)
                
                # Acceleration approximation
                df.loc[idx, 'dv_x'] = group['v_x'].diff().fillna(0) if 'v_x' in group.columns else 0
                df.loc[idx, 'dv_y'] = group['v_y'].diff().fillna(0) if 'v_y' in group.columns else 0
        
        return df
    
    @staticmethod
    def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features representing player interactions.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with interaction features
        """
        df = df.copy()
        
        # For each frame in each play, calculate distances to other players
        for (game_id, play_id, frame_id), group in df.groupby(['game_id', 'play_id', 'frame_id']):
            idx = group.index
            
            # Calculate distance to nearest defender/offensive player
            if 'player_side' in group.columns:
                for i, row in group.iterrows():
                    # Distance to nearest opponent
                    opponents = group[group['player_side'] != row['player_side']]
                    if len(opponents) > 0:
                        distances = np.sqrt(
                            (opponents['x'] - row['x'])**2 + 
                            (opponents['y'] - row['y'])**2
                        )
                        df.loc[i, 'dist_to_nearest_opponent'] = distances.min()
                        df.loc[i, 'avg_dist_to_opponents'] = distances.mean()
        
        return df
    
    @staticmethod
    def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add physics-based features for trajectory prediction.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with physics features
        """
        df = df.copy()
        
        # Time to reach ball landing location (assuming constant velocity)
        if 'dist_to_ball_land' in df.columns and 's' in df.columns:
            df['time_to_ball'] = df['dist_to_ball_land'] / (df['s'] + 1e-6)  # Add small epsilon to avoid division by zero
        
        # Direction towards ball landing location
        if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
            df['angle_to_ball'] = np.arctan2(
                df['ball_land_y'] - df['y'],
                df['ball_land_x'] - df['x']
            )
            
            # Angle difference between current direction and ball direction
            if 'dir' in df.columns:
                df['angle_diff_to_ball'] = np.abs(np.radians(df['dir']) - df['angle_to_ball'])
        
        return df


def prepare_model_data(sequences: Dict, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Prepare data for model training.
    
    Args:
        sequences: Dictionary of play sequences
        feature_cols: List of feature column names to use
        
    Returns:
        Tuple of (X, y, sequence_keys) where:
        - X: Input features array of shape (n_sequences, n_frames, n_features)
        - y: Target array of shape (n_sequences, n_output_frames, 2)
        - sequence_keys: List of (game_id, play_id, nfl_id) tuples
    """
    X_list = []
    y_list = []
    keys_list = []
    
    for key, seq_data in sequences.items():
        # Only include sequences where player should be predicted
        if not seq_data['player_to_predict']:
            continue
        
        input_seq = seq_data['input']
        output_seq = seq_data['output']
        
        # Extract features
        try:
            features = input_seq[feature_cols].values
            targets = output_seq[['x', 'y']].values
            
            X_list.append(features)
            y_list.append(targets)
            keys_list.append(key)
        except KeyError as e:
            print(f"Warning: Missing feature {e} for sequence {key}")
            continue
    
    return X_list, y_list, keys_list
