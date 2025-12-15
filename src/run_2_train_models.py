"""
Comprehensive GPU Training Pipeline for NFL Big Data Bowl 2026
This script trains all models (Baseline, XGBoost, LSTM, GRU, Transformer) and compares them.
Optimized for GPU training with mixed precision and data parallelism.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

# Import custom modules
from data_loader import NFLDataLoader, FeatureEngineering, prepare_model_data
from model_baseline import PhysicsBaselineModel
from model_xgboost import XGBoostTrajectoryModel
from model_lstm_advanced import AdvancedLSTMModel, GRUModel, NFLTrajectoryDataset, ModelTrainer
from model_transformer import TransformerTrajectoryModel, TransformerTrainer

# Configuration
CONFIG = {
    'data_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/data',
    'models_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/models',
    'outputs_dir': 'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/outputs',
    'weeks_to_train': list(range(1, 19)),  # All 18 weeks
    'validation_split': 0.15,
    'batch_size': 64,
    'num_workers': 0,  # Keep 0 for Windows safety
    'epochs': {
        'lstm': 50,
        'gru': 50,
        'transformer': 30
    },
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,  # Use automatic mixed precision for faster training
    'save_checkpoints': True
}

# Feature columns
FEATURE_COLS = [
    'x', 'y', 's', 'a', 'dir', 'o',
    'ball_land_x', 'ball_land_y', 'dist_to_ball_land',
    'v_x', 'v_y',
    'player_position_encoded', 'player_side_encoded', 'player_role_encoded',
    'player_height_inches', 'player_weight', 'player_age',
    'play_direction_binary', 'absolute_yardline_number'
]


def setup_directories():
    """Create necessary directories."""
    for dir_path in [CONFIG['models_dir'], CONFIG['outputs_dir']]:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for each model
    for model_name in ['baseline', 'xgboost', 'lstm', 'gru', 'transformer']:
        (Path(CONFIG['models_dir']) / model_name).mkdir(exist_ok=True)


def load_and_preprocess_data():
    """Load and preprocess all training data."""
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    loader = NFLDataLoader(CONFIG['data_dir'])
    
    # Load all weeks
    print(f"\nLoading {len(CONFIG['weeks_to_train'])} weeks of data...")
    input_df, output_df = loader.load_all_training_data(weeks=CONFIG['weeks_to_train'])
    
    print(f"Loaded {len(input_df):,} input rows, {len(output_df):,} output rows")
    
    # Preprocess
    print("\nPreprocessing input data...")
    input_df = loader.preprocess_input_data(input_df)
    
    # Feature engineering
    print("Engineering features...")
    fe = FeatureEngineering()
    input_df = fe.add_physics_features(input_df)
    input_df = fe.add_temporal_features(input_df)
    
    # Filter to available features
    available_features = [col for col in FEATURE_COLS if col in input_df.columns]
    print(f"Using {len(available_features)} features")
    
    return input_df, output_df, available_features


def prepare_sequences(input_df, output_df):
    """Prepare sequences for deep learning models."""
    print("\nPreparing sequences...")
    sequences = []
    
    for (game_id, play_id, nfl_id), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
        # Only include players to predict
        if not group['player_to_predict'].iloc[0]:
            continue
        
        input_seq = group.sort_values('frame_id')
        output_seq = output_df[
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id) &
            (output_df['nfl_id'] == nfl_id)
        ].sort_values('frame_id')
        
        if len(output_seq) == 0:
            continue
        
        metadata = {
            'game_id': game_id,
            'play_id': play_id,
            'nfl_id': nfl_id,
            'num_frames': len(output_seq),
            'player_role_encoded': input_seq['player_role_encoded'].iloc[0] if 'player_role_encoded' in input_seq.columns else 0
        }
        
        sequences.append((input_seq, output_seq, metadata))
    
    print(f"Created {len(sequences):,} sequences")
    
    # Save sequences cache
    cache_path = Path(CONFIG['models_dir']) / 'sequences.pkl'
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"Saved sequences to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save sequence cache: {e}")
        
    return sequences

def load_sequences_cache():
    """Load cached sequences if available."""
    cache_path = Path(CONFIG['models_dir']) / 'sequences.pkl'
    if cache_path.exists():
        print(f"Loading sequences from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                sequences = pickle.load(f)
            print(f"Loaded {len(sequences):,} sequences")
            return sequences
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None


def train_baseline_model(input_df, output_df):
    """Train baseline physics model."""
    print("\n" + "="*80)
    print("TRAINING BASELINE PHYSICS MODEL")
    print("="*80)
    
    model = PhysicsBaselineModel()
    
    # Evaluate on sample
    plays = input_df[['game_id', 'play_id']].drop_duplicates().sample(min(1000, len(input_df)))
    
    all_predictions = {}
    for idx, (_, play) in enumerate(plays.iterrows()):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(plays)}")
        
        game_id, play_id = play['game_id'], play['play_id']
        play_input = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
        
        for nfl_id in play_input['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            if not player_input['player_to_predict'].iloc[0]:
                continue
            
            num_frames = int(player_input['num_frames_output'].iloc[0])
            predictions = model.predict_play(player_input, num_frames, method='weighted')
            
            if nfl_id in predictions:
                all_predictions[(game_id, play_id, nfl_id)] = predictions[nfl_id]
    
    # Calculate RMSE
    rmse = calculate_rmse(all_predictions, output_df)
    print(f"\nBaseline RMSE: {rmse:.4f}")
    
    # Save model (it's just the logic, no parameters)
    with open(Path(CONFIG['models_dir']) / 'baseline' / 'model_info.json', 'w') as f:
        json.dump({'method': 'weighted', 'rmse': float(rmse)}, f)
    
    return {'model': 'baseline', 'rmse': float(rmse)}


def train_xgboost_model(input_df, output_df, feature_cols):
    """Train XGBoost model."""
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)
    
    # Optimized XGBoost parameters for < 0.480 RMSE goal
    model = XGBoostTrajectoryModel(
        max_future_frames=30, 
        n_estimators=500, 
        max_depth=9, 
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        device='cuda' if CONFIG['device'] == 'cuda' else 'cpu'
    )
    
    print("Preparing training data...")
    training_data = model.prepare_training_data(input_df, output_df, feature_cols)
    
    print(f"Training on {len(training_data)} future frames...")
    model.train(training_data, verbose=True)
    
    # Save model
    save_path = Path(CONFIG['models_dir']) / 'xgboost'
    model.save_models(save_path)
    print(f"\nXGBoost model saved to {save_path}")
    
    # Evaluate
    print("\nEvaluating XGBoost model...")
    all_predictions = {}
    plays = input_df[['game_id', 'play_id']].drop_duplicates().sample(min(500, len(input_df)))
    
    for idx, (_, play) in enumerate(plays.iterrows()):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(plays)}")
        
        game_id, play_id = play['game_id'], play['play_id']
        play_input = input_df[(input_df['game_id'] == game_id) & (input_df['play_id'] == play_id)]
        
        for nfl_id in play_input['nfl_id'].unique():
            player_input = play_input[play_input['nfl_id'] == nfl_id]
            if not player_input['player_to_predict'].iloc[0]:
                continue
            
            num_frames = int(player_input['num_frames_output'].iloc[0])
            try:
                predictions = model.predict(player_input, feature_cols, num_frames)
                all_predictions[(game_id, play_id, nfl_id)] = predictions
            except:
                continue
    
    rmse = calculate_rmse(all_predictions, output_df)
    print(f"\nXGBoost RMSE: {rmse:.4f}")
    
    return {'model': 'xgboost', 'rmse': float(rmse)}


def train_deep_learning_model(model_type, sequences, feature_cols):
    """Train LSTM, GRU, or Transformer model."""
    print("\n" + "="*80)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*80)
    
    # Create dataset
    dataset = NFLTrajectoryDataset(sequences, feature_cols, max_input_len=50, max_output_len=30)
    
    # Split into train/val
    val_size = int(len(dataset) * CONFIG['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {train_size:,}, Val size: {val_size:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # Create model
    input_size = len(feature_cols)
    
    if model_type == 'lstm':
        model = AdvancedLSTMModel(input_size, hidden_size=128, num_layers=2, dropout=0.3)
        trainer = ModelTrainer(model, device=CONFIG['device'], learning_rate=0.001)
    elif model_type == 'gru':
        model = GRUModel(input_size, hidden_size=128, num_layers=2, dropout=0.3)
        trainer = ModelTrainer(model, device=CONFIG['device'], learning_rate=0.001)
    elif model_type == 'transformer':
        model = TransformerTrajectoryModel(input_size, d_model=256, nhead=8, 
                                          num_encoder_layers=6, num_decoder_layers=6)
        trainer = TransformerTrainer(model, device=CONFIG['device'], learning_rate=0.0001)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    save_path = Path(CONFIG['models_dir']) / model_type / f'best_{model_type}.pth'
    epochs = CONFIG['epochs'][model_type]
    
    history = trainer.train(train_loader, val_loader, epochs=epochs, save_path=str(save_path))
    
    # Get best RMSE
    best_rmse = min(history.get('val_rmse', [float('inf')]))
    print(f"\nBest {model_type.upper()} RMSE: {best_rmse:.4f}")
    
    return {'model': model_type, 'rmse': float(best_rmse), 'history': history}


def calculate_rmse(predictions, ground_truth):
    """Calculate competition RMSE metric."""
    squared_errors = []
    
    for (game_id, play_id, nfl_id), pred_positions in predictions.items():
        gt = ground_truth[
            (ground_truth['game_id'] == game_id) &
            (ground_truth['play_id'] == play_id) &
            (ground_truth['nfl_id'] == nfl_id)
        ].sort_values('frame_id')
        
        if len(gt) == 0:
            continue
        
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


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("NFL BIG DATA BOWL 2026 - GPU TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {CONFIG['device']}")
    
    if CONFIG['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Setup
    setup_directories()
    
    # Load data
    input_df, output_df, feature_cols = load_and_preprocess_data()
    
    # Prepare sequences for deep learning
    # Check cache first
    sequences = load_sequences_cache()
    
    if sequences is None:
        # NEW: Scale features before creating sequences
        print("\nScaling features...")
        scaler = StandardScaler()
        
        # Only scale feature columns that exist in the dataframe
        features_to_scale = [col for col in feature_cols if col in input_df.columns]
        input_df[features_to_scale] = scaler.fit_transform(input_df[features_to_scale])
        
        # Save scaler
        scaler_path = Path(CONFIG['models_dir']) / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
        sequences = prepare_sequences(input_df, output_df)
    else:
        print("Using cached sequences (Scaling assumed already applied)")
    
    # Train all models
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'models': {}
    }
    
    # 1. Baseline
    # try:
    #     baseline_results = train_baseline_model(input_df, output_df)
    #     results['models']['baseline'] = baseline_results
    # except Exception as e:
    #     print(f"Error training baseline: {e}")
    #     results['models']['baseline'] = {'error': str(e)}
    
    # 2. XGBoost
    # try:
    #     xgb_results = train_xgboost_model(input_df, output_df, feature_cols)
    #     results['models']['xgboost'] = xgb_results
    # except Exception as e:
    #     print(f"Error training XGBoost: {e}")
    #     results['models']['xgboost'] = {'error': str(e)}
    
    # 3. LSTM
    try:
        lstm_results = train_deep_learning_model('lstm', sequences, feature_cols)
        results['models']['lstm'] = lstm_results
    except Exception as e:
        print(f"Error training LSTM: {e}")
        results['models']['lstm'] = {'error': str(e)}
    
    # 4. GRU
    try:
        gru_results = train_deep_learning_model('gru', sequences, feature_cols)
        results['models']['gru'] = gru_results
    except Exception as e:
        print(f"Error training GRU: {e}")
        results['models']['gru'] = {'error': str(e)}
    
    # 5. Transformer
    # try:
    #     transformer_results = train_deep_learning_model('transformer', sequences, feature_cols)
    #     results['models']['transformer'] = transformer_results
    # except Exception as e:
    #     print(f"Error training Transformer: {e}")
    #     results['models']['transformer'] = {'error': str(e)}
    
    # Save results
    results_path = Path(CONFIG['outputs_dir']) / 'gpu_training_results.json'
    with open(results_path, 'w') as f:
        # Remove non-serializable items
        results_copy = results.copy()
        for model_name in results_copy['models']:
            if 'history' in results_copy['models'][model_name]:
                del results_copy['models'][model_name]['history']
        json.dump(results_copy, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*80)
    
    for model_name, model_results in results['models'].items():
        if 'rmse' in model_results:
            print(f"{model_name:20s}: RMSE = {model_results['rmse']:.4f}")
        else:
            print(f"{model_name:20s}: ERROR")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == '__main__':
    results = main()
