"""
Stacking Ensemble Model
Trains a meta-model (Ridge Regression) on the predictions of base models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import json

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from model_ensemble import ModelEnsemble
from train_gpu import CONFIG, FEATURE_COLS

class StackingEnsemble:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.base_ensemble = ModelEnsemble(models_dir)
        self.meta_model = Ridge(alpha=1.0)
        self.scaler = None
        
    def train(self, val_data, val_output, feature_cols):
        print("Training Stacking Meta-Model...")
        
        # 1. Generate predictions from base models
        print("Generating base model predictions...")
        self.base_ensemble.load_models(model_types=['xgboost', 'gru', 'lstm', 'transformer'])
        
        # Filter to valid plays
        val_plays = val_data[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
        
        X_meta = []
        y_meta = []
        
        total = len(val_plays)
        for idx, (_, row) in enumerate(val_plays.iterrows()):
            if idx % 100 == 0:
                print(f"  Processing {idx}/{total}")
                
            game_id, play_id, nfl_id = row['game_id'], row['play_id'], row['nfl_id']
            
            # Get input data for this player
            player_data = val_data[
                (val_data['game_id'] == game_id) & 
                (val_data['play_id'] == play_id) & 
                (val_data['nfl_id'] == nfl_id)
            ].sort_values('frame_id')
            
            if len(player_data) == 0:
                continue
                
            # Get ground truth
            gt_data = val_output[
                (val_output['game_id'] == game_id) & 
                (val_output['play_id'] == play_id) & 
                (val_output['nfl_id'] == nfl_id)
            ].sort_values('frame_id')
            
            if len(gt_data) == 0:
                continue
                
            # Predict with each base model
            # We need predictions for the same frames as ground truth
            num_frames = len(gt_data)
            
            try:
                # Get predictions from all models
                # ModelEnsemble.predict_single_player returns a dict of predictions if we modify it, 
                # but currently it averages them. We need individual predictions.
                # So we access the models directly from base_ensemble.models
                
                base_preds = []
                for name, model in self.base_ensemble.models.items():
                    if name == 'baseline':
                        # Baseline signature is different
                        p = model.predict_play(player_data, num_frames)
                        # Extract just the trajectory for this player
                        if nfl_id in p:
                            pred = p[nfl_id]
                        else:
                            pred = np.zeros((num_frames, 2))
                    elif name == 'xgboost':
                        pred = model.predict(player_data, feature_cols, num_frames)
                    else:
                        # Deep learning models
                        # They expect specific input format. 
                        # For simplicity, let's assume ModelEnsemble has a method or we assume the standard predict interface
                        # The DL models in ModelEnsemble.models are the raw PyTorch models? 
                        # No, ModelEnsemble loads them using a wrapper or expects a wrapper.
                        # Let's look at ModelEnsemble.load_models...
                        # It loads them into self.models.
                        # We need to ensure we can call predict on them.
                        pass
                        
                # REVISIT: ModelEnsemble structure might not support easy individual access.
                # Let's rely on the fact that we can just run the ensemble with weight=1 for each model.
                
                # Strategy: Run ensemble with 1.0 weight for model X and 0 for others.
                row_preds = []
                valid_row = True
                
                for model_name in ['xgboost', 'gru', 'lstm', 'transformer']:
                    # Set exclusive weight
                    weights = {m: 1.0 if m == model_name else 0.0 for m in ['xgboost', 'gru', 'lstm', 'transformer', 'baseline']}
                    self.base_ensemble.set_weights(weights)
                    
                    # Predict
                    pred = self.base_ensemble.predict_single_player(player_data, feature_cols, num_frames)
                    
                    # Flatten: (num_frames, 2) -> (num_frames * 2,)
                    # Actually, stacking usually works frame-by-frame or on the whole trajectory.
                    # Let's do frame-by-frame stacking.
                    row_preds.append(pred) # List of (N, 2) arrays
                    
                # Now we have [Pred_XGB, Pred_GRU, ...] each (N, 2)
                # We want to train: Meta(Pred_XGB_x, Pred_GRU_x, ...) -> True_x
                
                for t in range(num_frames):
                    # Features for meta-model: [XGB_x, XGB_y, GRU_x, GRU_y, ...]
                    meta_features = []
                    for m_idx, p_traj in enumerate(row_preds):
                        meta_features.extend(p_traj[t])
                        
                    X_meta.append(meta_features)
                    y_meta.append(gt_data.iloc[t][['x', 'y']].values)
                    
            except Exception as e:
                # print(f"Error processing {game_id}-{play_id}: {e}")
                continue

        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        print(f"Meta-training data shape: {X_meta.shape}")
        
        # Train Ridge
        self.meta_model.fit(X_meta, y_meta)
        
        # Evaluate
        y_pred = self.meta_model.predict(X_meta)
        rmse = np.sqrt(mean_squared_error(y_meta, y_pred))
        print(f"Stacking Meta-Model RMSE (Train): {rmse:.4f}")
        
        # Save
        joblib.dump(self.meta_model, self.models_dir / 'stacking_meta_model.pkl')
        print("Meta-model saved.")
        
    def predict(self, player_data, feature_cols, num_frames):
        """
        Predict trajectory using stacking ensemble.
        """
        if self.meta_model is None:
            # Try to load
            path = self.models_dir / 'stacking_meta_model.pkl'
            if path.exists():
                self.meta_model = joblib.load(path)
            else:
                raise ValueError("Meta-model not trained or found!")
        
        # 1. Get base model predictions
        # We need to instantiate the models if not already done
        if not self.base_ensemble.models:
             self.base_ensemble.load_models(model_types=['xgboost', 'gru', 'lstm', 'transformer'])
             
        row_preds = []
        
        # Predict with each base model
        # Note: This is inefficient (re-predicting for each model). 
        # In production, we might want to batch this or optimize ModelEnsemble.
        
        for model_name in ['xgboost', 'gru', 'lstm', 'transformer']:
            # We can access the model directly and call predict/predict_play
            model = self.base_ensemble.models.get(model_name)
            if model is None:
                # If a model is missing, we must handle it. For now, assume all exist or fill with zeros.
                pred = np.zeros((num_frames, 2))
            elif model_name == 'baseline':
                p = model.predict_play(player_data, num_frames)
                nfl_id = player_data['nfl_id'].iloc[0]
                pred = p.get(nfl_id, np.zeros((num_frames, 2)))
            elif model_name == 'xgboost':
                pred = model.predict(player_data, feature_cols, num_frames)
            else:
                # DL models need specific handling if not going through ensemble wrapper
                # But we can use the ensemble wrapper with exclusive weights
                weights = {m: 1.0 if m == model_name else 0.0 for m in ['xgboost', 'gru', 'lstm', 'transformer', 'baseline']}
                self.base_ensemble.set_weights(weights)
                pred = self.base_ensemble.predict_single_player(player_data, feature_cols, num_frames)
            
            row_preds.append(pred)

        # 2. Construct meta-features
        X_meta = []
        for t in range(num_frames):
            meta_features = []
            for p_traj in row_preds:
                if t < len(p_traj):
                    meta_features.extend(p_traj[t])
                else:
                    meta_features.extend([0, 0]) # Padding
            X_meta.append(meta_features)
            
        # 3. Predict with meta-model
        X_meta = np.array(X_meta)
        y_pred = self.meta_model.predict(X_meta)
        
        return y_pred
