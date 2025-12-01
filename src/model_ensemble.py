"""
Model Ensemble System for NFL Big Data Bowl 2026
Combines predictions from multiple models for better performance.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple
from scipy.optimize import minimize

from model_baseline import PhysicsBaselineModel
from model_xgboost import XGBoostTrajectoryModel
from model_lstm_advanced import AdvancedLSTMModel, GRUModel
from model_transformer import TransformerTrajectoryModel


class ModelEnsemble:
    """
    Ensemble of multiple trajectory prediction models.
    
    Combines predictions using weighted averaging based on model performance.
    """
    
    def __init__(self, models_dir='/home/ubuntu/nfl_project/models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.weights = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_models(self, model_types=['baseline', 'xgboost', 'lstm', 'gru']):
        """
        Load trained models.
        
        Args:
            model_types: List of model types to load
        """
        print("Loading models...")
        
        for model_type in model_types:
            try:
                if model_type == 'baseline':
                    self.models['baseline'] = PhysicsBaselineModel()
                    print("  ✓ Loaded baseline model")
                
                elif model_type == 'xgboost':
                    model = XGBoostTrajectoryModel()
                    model.load_models(self.models_dir / 'xgboost')
                    self.models['xgboost'] = model
                    print("  ✓ Loaded XGBoost model")
                
                elif model_type == 'lstm':
                    checkpoint_path = self.models_dir / 'lstm' / 'best_lstm.pth'
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        # Need to know input size - load from metadata
                        model = AdvancedLSTMModel(input_size=19, hidden_size=256, num_layers=3)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.to(self.device)
                        model.eval()
                        self.models['lstm'] = model
                        print(f"  ✓ Loaded LSTM model (RMSE: {checkpoint.get('val_rmse', 'N/A')})")
                    else:
                        print(f"  ✗ LSTM checkpoint not found at {checkpoint_path}")
                
                elif model_type == 'gru':
                    checkpoint_path = self.models_dir / 'gru' / 'best_gru.pth'
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        model = GRUModel(input_size=19, hidden_size=256, num_layers=3)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.to(self.device)
                        model.eval()
                        self.models['gru'] = model
                        print(f"  ✓ Loaded GRU model (RMSE: {checkpoint.get('val_rmse', 'N/A')})")
                    else:
                        print(f"  ✗ GRU checkpoint not found at {checkpoint_path}")
                
                elif model_type == 'transformer':
                    checkpoint_path = self.models_dir / 'transformer' / 'best_transformer.pth'
                    if checkpoint_path.exists():
                        checkpoint = torch.load(checkpoint_path, map_location=self.device)
                        model = TransformerTrajectoryModel(input_size=19, d_model=256)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.to(self.device)
                        model.eval()
                        self.models['transformer'] = model
                        print(f"  ✓ Loaded Transformer model")
                    else:
                        print(f"  ✗ Transformer checkpoint not found at {checkpoint_path}")
            
            except Exception as e:
                print(f"  ✗ Error loading {model_type}: {e}")
        
        print(f"\nLoaded {len(self.models)} models: {list(self.models.keys())}")
    
    def set_weights(self, weights: Dict[str, float] = None):
        """
        Set ensemble weights for each model.
        
        Args:
            weights: Dictionary mapping model names to weights.
                    If None, uses equal weights.
        """
        if weights is None:
            # Equal weights
            n_models = len(self.models)
            weights = {name: 1.0 / n_models for name in self.models.keys()}
        
        # Normalize weights
        total = sum(weights.values())
        if total == 0:
            self.weights = {name: 1.0 / len(weights) for name in weights}
        else:
            self.weights = {name: w / total for name, w in weights.items()}
    
    def predict_single_player(self, player_data: pd.DataFrame, feature_cols: List[str],
                             num_frames: int) -> np.ndarray:
        """
        Predict trajectory for a single player using ensemble.
        
        Args:
            player_data: DataFrame with player's input tracking data
            feature_cols: List of feature columns
            num_frames: Number of frames to predict
            
        Returns:
            Array of shape (num_frames, 2) with ensemble predictions
        """
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'baseline':
                    pred = model.predict_play(player_data, num_frames, method='weighted')
                    nfl_id = player_data['nfl_id'].iloc[0]
                    if nfl_id in pred:
                        predictions.append(pred[nfl_id])
                        weights.append(self.weights.get(model_name, 0))
                
                elif model_name == 'xgboost':
                    pred = model.predict(player_data, feature_cols, num_frames)
                    predictions.append(pred)
                    weights.append(self.weights.get(model_name, 0))
                
                elif model_name in ['lstm', 'gru']:
                    # Prepare input for deep learning model
                    X = player_data[feature_cols].values.astype(np.float32)
                    X = torch.FloatTensor(X).unsqueeze(0).to(self.device)
                    
                    ball_pos = player_data[['ball_land_x', 'ball_land_y']].iloc[-1].values
                    ball_pos = torch.FloatTensor(ball_pos).unsqueeze(0).to(self.device)
                    
                    player_role = player_data['player_role_encoded'].iloc[0] if 'player_role_encoded' in player_data.columns else 0
                    player_role = torch.LongTensor([player_role]).to(self.device)
                    
                    with torch.no_grad():
                        pred, _ = model(X, ball_pos, player_role, num_frames)
                        pred = pred.cpu().numpy()[0]
                    
                    predictions.append(pred)
                    weights.append(self.weights.get(model_name, 0))
                
                elif model_name == 'transformer':
                    X = player_data[feature_cols].values.astype(np.float32)
                    X = torch.FloatTensor(X).unsqueeze(0).to(self.device)
                    
                    start_pos = player_data[['x', 'y']].iloc[-1].values
                    start_pos = torch.FloatTensor(start_pos).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        pred = model.predict_autoregressive(X, num_frames, start_pos)
                        pred = pred.cpu().numpy()[0]
                    
                    predictions.append(pred)
                    weights.append(self.weights.get(model_name, 0))
            
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
                continue
        
        if len(predictions) == 0:
            # Fallback to simple constant velocity
            last_row = player_data.iloc[-1]
            x, y = last_row['x'], last_row['y']
            v_x = last_row.get('v_x', 0)
            v_y = last_row.get('v_y', 0)
            
            pred = []
            for i in range(num_frames):
                x += v_x * 0.1
                y += v_y * 0.1
                pred.append([np.clip(x, 0, 120), np.clip(y, 0, 53.3)])
            return np.array(pred)
        
        # Weighted average of predictions
        predictions = np.array(predictions)  # (n_models, num_frames, 2)
        weights = np.array(weights).reshape(-1, 1, 1)  # (n_models, 1, 1)
        
        ensemble_pred = np.sum(predictions * weights, axis=0)  # (num_frames, 2)
        
        # Clip to field boundaries
        ensemble_pred[:, 0] = np.clip(ensemble_pred[:, 0], 0, 120)
        ensemble_pred[:, 1] = np.clip(ensemble_pred[:, 1], 0, 53.3)
        
        return ensemble_pred
    
    def predict_play(self, play_data: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        Predict all players in a play.
        
        Args:
            play_data: DataFrame with input tracking data for the play
            feature_cols: List of feature columns
            
        Returns:
            Dictionary mapping nfl_id to predicted positions
        """
        predictions = {}
        
        for nfl_id in play_data['nfl_id'].unique():
            player_data = play_data[play_data['nfl_id'] == nfl_id]
            
            # Only predict if flagged
            if not player_data['player_to_predict'].iloc[0]:
                continue
            
            num_frames = int(player_data['num_frames_output'].iloc[0])
            
            pred = self.predict_single_player(player_data, feature_cols, num_frames)
            predictions[nfl_id] = pred
        
        return predictions
    
    def save_config(self, path):
        """Save ensemble configuration."""
        config = {
            'models': list(self.models.keys()),
            'weights': self.weights,
            'device': self.device
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Ensemble config saved to {path}")
    
    def load_config(self, path):
        """Load ensemble configuration."""
        with open(path, 'r') as f:
            config = json.load(f)
        
        self.weights = config['weights']
        print(f"Loaded ensemble config from {path}")


def optimize_ensemble_weights(ensemble, val_data, val_output, feature_cols):
    """
    Optimize ensemble weights using validation data.
    Uses scipy.optimize.minimize for precise weight finding.
    """
    print("\nOptimizing ensemble weights...")
    
    # Sample validation plays (increase sample size for better accuracy)
    # Use more plays if available, up to 500
    sample_size = min(500, len(val_data))
    val_plays = val_data[['game_id', 'play_id']].drop_duplicates().sample(sample_size)
    print(f"Using {len(val_plays)} plays for optimization")
    
    # Pre-calculate predictions for all models to avoid re-running inference
    print("Pre-calculating model predictions...")
    model_names = list(ensemble.models.keys())
    n_models = len(model_names)
    
    # Store predictions: list of (game_id, play_id, nfl_id, pred_array)
    # We need to align these with ground truth
    
    # 1. Get all ground truth for these plays
    ground_truth_list = []
    prediction_cache = {name: [] for name in model_names}
    
    valid_entries = 0
    
    for idx, (_, play) in enumerate(val_plays.iterrows()):
        if idx % 50 == 0:
            print(f"  Processing play {idx}/{len(val_plays)}")
            
        game_id, play_id = play['game_id'], play['play_id']
        play_input = val_data[(val_data['game_id'] == game_id) & (val_data['play_id'] == play_id)]
        
        # Get ground truth for this play
        play_gt = val_output[
            (val_output['game_id'] == game_id) &
            (val_output['play_id'] == play_id)
        ]
        
        for nfl_id in play_input['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            
            if not player_input['player_to_predict'].iloc[0]:
                continue
                
            # Get GT for this player
            player_gt = play_gt[play_gt['nfl_id'] == nfl_id].sort_values('frame_id')
            if len(player_gt) == 0:
                continue
                
            num_frames = int(player_input['num_frames_output'].iloc[0])
            gt_coords = player_gt[['x', 'y']].values[:num_frames]
            
            if len(gt_coords) < num_frames:
                # Handle mismatch if GT is shorter
                num_frames = len(gt_coords)
            
            # Get predictions from each model
            valid_models_for_sample = True
            sample_preds = {}
            
            for name in model_names:
                try:
                    # We need to call predict_single_player but force it to return just the array
                    # The current predict_single_player does ensemble averaging, we need raw model preds
                    # So we'll access the models directly here or modify the class
                    # Accessing directly is cleaner for this external function
                    
                    model = ensemble.models[name]
                    pred = None
                    
                    if name == 'baseline':
                        p = model.predict_play(player_input, num_frames, method='weighted')
                        if nfl_id in p:
                            pred = p[nfl_id]
                            
                    elif name == 'xgboost':
                        pred = model.predict(player_input, feature_cols, num_frames)
                        
                    elif name in ['lstm', 'gru']:
                        # Deep learning inference
                        X = player_input[feature_cols].values.astype(np.float32)
                        X = torch.FloatTensor(X).unsqueeze(0).to(ensemble.device)
                        
                        ball_pos = player_input[['ball_land_x', 'ball_land_y']].iloc[-1].values
                        ball_pos = torch.FloatTensor(ball_pos).unsqueeze(0).to(ensemble.device)
                        
                        player_role = player_input['player_role_encoded'].iloc[0] if 'player_role_encoded' in player_input.columns else 0
                        player_role = torch.LongTensor([player_role]).to(ensemble.device)
                        
                        with torch.no_grad():
                            p, _ = model(X, ball_pos, player_role, num_frames)
                            pred = p.cpu().numpy()[0]
                            
                    elif name == 'transformer':
                        X = player_input[feature_cols].values.astype(np.float32)
                        X = torch.FloatTensor(X).unsqueeze(0).to(ensemble.device)
                        
                        start_pos = player_input[['x', 'y']].iloc[-1].values
                        start_pos = torch.FloatTensor(start_pos).unsqueeze(0).to(ensemble.device)
                        
                        with torch.no_grad():
                            p = model.predict_autoregressive(X, num_frames, start_pos)
                            pred = p.cpu().numpy()[0]
                    
                    if pred is not None and len(pred) == num_frames:
                        sample_preds[name] = pred
                    else:
                        valid_models_for_sample = False
                        break
                        
                except Exception:
                    valid_models_for_sample = False
                    break
            
            if valid_models_for_sample:
                ground_truth_list.append(gt_coords)
                for name in model_names:
                    prediction_cache[name].append(sample_preds[name])
                valid_entries += 1
    
    print(f"Collected {valid_entries} valid samples for optimization")
    
    if valid_entries == 0:
        print("Warning: No valid samples found. Returning equal weights.")
        return {name: 1.0/n_models for name in model_names}, float('inf')

    # Convert to numpy arrays for fast calculation
    # y_true: (N_samples, frames, 2) - flattened to (N_total_points, 2)
    # y_preds: (n_models, N_total_points, 2)
    
    y_true_all = np.concatenate(ground_truth_list, axis=0) # (Total_frames, 2)
    
    preds_list = []
    for name in model_names:
        p_arr = np.concatenate(prediction_cache[name], axis=0) # (Total_frames, 2)
        preds_list.append(p_arr)
        
    y_preds_all = np.stack(preds_list, axis=0) # (n_models, Total_frames, 2)
    
    print(f"Optimization data shape: {y_preds_all.shape}")
    
    # Objective function to minimize RMSE
    def objective(weights):
        # weights shape: (n_models,)
        # Normalize weights to sum to 1 (softmax or just divide by sum)
        # Here we rely on constraints, but normalization ensures stability
        w = np.array(weights)
        w = w / np.sum(w)
        
        # Weighted sum of predictions
        # (n_models, 1, 1) * (n_models, Total_frames, 2) -> sum over axis 0
        weighted_pred = np.sum(w.reshape(-1, 1, 1) * y_preds_all, axis=0)
        
        # Calculate MSE
        mse = np.mean((weighted_pred - y_true_all) ** 2)
        return np.sqrt(mse)

    # Constraints and bounds
    # Weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # Weights must be between 0 and 1
    bounds = tuple((0.0, 1.0) for _ in range(n_models))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_models) / n_models
    
    # Optimize
    print("Running optimization...")
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    best_weights_arr = result.x / np.sum(result.x) # Ensure exact sum to 1
    best_rmse = result.fun
    
    best_weights = {name: float(w) for name, w in zip(model_names, best_weights_arr)}
    
    print(f"\nOptimization Success: {result.success}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print("Best weights:")
    for name, weight in best_weights.items():
        print(f"  {name}: {weight:.3f}")
    
    ensemble.set_weights(best_weights)
    return best_weights, best_rmse


def evaluate_ensemble(ensemble, plays, input_data, output_data, feature_cols):
    """Evaluate ensemble on a set of plays."""
    all_predictions = {}
    
    for _, play in plays.iterrows():
        game_id, play_id = play['game_id'], play['play_id']
        play_input = input_data[(input_data['game_id'] == game_id) & (input_data['play_id'] == play_id)]
        
        predictions = ensemble.predict_play(play_input, feature_cols)
        
        for nfl_id, pred in predictions.items():
            all_predictions[(game_id, play_id, nfl_id)] = pred
    
    # Calculate RMSE
    squared_errors = []
    for (game_id, play_id, nfl_id), pred_positions in all_predictions.items():
        gt = output_data[
            (output_data['game_id'] == game_id) &
            (output_data['play_id'] == play_id) &
            (output_data['nfl_id'] == nfl_id)
        ].sort_values('frame_id')
        
        for i, (_, row) in enumerate(gt.iterrows()):
            if i >= len(pred_positions):
                break
            
            pred_x, pred_y = pred_positions[i]
            true_x, true_y = row['x'], row['y']
            
            error = (pred_x - true_x)**2 + (pred_y - true_y)**2
            squared_errors.append(error)
    
    if len(squared_errors) == 0:
        return float('inf')
    
    rmse = np.sqrt(np.mean(squared_errors) / 2)
    return rmse


if __name__ == '__main__':
    print("Model ensemble module loaded")
    print("Use this to combine predictions from multiple models")
