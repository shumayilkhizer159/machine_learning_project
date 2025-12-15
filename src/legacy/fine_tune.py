"""
Fine-Tuning Script for NFL Trajectory Models
Loads a pre-trained model and continues training with a lower learning rate.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
import sys
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import NFLDataLoader, FeatureEngineering, prepare_model_data
from model_lstm_advanced import AdvancedLSTMModel, GRUModel, NFLTrajectoryDataset, ModelTrainer
from model_transformer import TransformerTrajectoryModel, TransformerTrainer
from train_gpu import CONFIG, FEATURE_COLS, prepare_sequences, load_and_preprocess_data

def fine_tune_model(model_type, sequences, feature_cols, initial_lr=1e-4, epochs=20):
    print(f"\nFine-tuning {model_type.upper()} model...")
    
    # Load scaler
    scaler_path = Path(CONFIG['models_dir']) / 'scaler.pkl'
    if not scaler_path.exists():
        print("Scaler not found! Cannot fine-tune.")
        return
    
    # Create dataset
    dataset = NFLTrajectoryDataset(sequences, feature_cols, max_input_len=50, max_output_len=30)
    
    # Split
    val_size = int(len(dataset) * CONFIG['validation_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, pin_memory=True)
    
    # Initialize Model
    input_size = len(feature_cols)
    device = CONFIG['device']
    
    if model_type == 'lstm':
        model = AdvancedLSTMModel(input_size, hidden_size=256, num_layers=3, dropout=0.3)
        trainer = ModelTrainer(model, device=device, learning_rate=initial_lr)
    elif model_type == 'gru':
        model = GRUModel(input_size, hidden_size=256, num_layers=3, dropout=0.3)
        trainer = ModelTrainer(model, device=device, learning_rate=initial_lr)
    elif model_type == 'transformer':
        model = TransformerTrajectoryModel(input_size, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6)
        trainer = TransformerTrainer(model, device=device, learning_rate=initial_lr)
    else:
        raise ValueError(f"Unknown model: {model_type}")
        
    # Load Pre-trained Weights
    checkpoint_path = Path(CONFIG['models_dir']) / model_type / f'best_{model_type}.pth'
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(str(checkpoint_path))
    else:
        print("No checkpoint found. Starting from scratch (not recommended for fine-tuning).")
        
    # Train
    save_path = Path(CONFIG['models_dir']) / model_type / f'best_{model_type}_finetuned.pth'
    history = trainer.train(train_loader, val_loader, epochs=epochs, save_path=str(save_path))
    
    best_rmse = min(history.get('val_rmse', [float('inf')]))
    print(f"Fine-tuning complete. Best RMSE: {best_rmse:.4f}")
    return best_rmse

if __name__ == "__main__":
    # Load data once
    input_df, output_df, feature_cols = load_and_preprocess_data()
    
    # Scale
    scaler = joblib.load(Path(CONFIG['models_dir']) / 'scaler.pkl')
    features_to_scale = [col for col in feature_cols if col in input_df.columns]
    input_df[features_to_scale] = scaler.transform(input_df[features_to_scale])
    
    sequences = prepare_sequences(input_df, output_df)
    
    # Fine-tune
    # fine_tune_model('lstm', sequences, feature_cols)
    fine_tune_model('gru', sequences, feature_cols)
    # fine_tune_model('transformer', sequences, feature_cols)
