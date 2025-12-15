"""
LSTM-based Model for NFL Player Trajectory Prediction
This model uses Long Short-Term Memory networks to predict future player positions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. LSTM model will not work.")


class NFLTrajectoryDataset(Dataset):
    """Dataset for NFL player trajectory prediction."""
    
    def __init__(self, sequences: List[Tuple], feature_cols: List[str]):
        """
        Args:
            sequences: List of (input_df, output_df, metadata) tuples
            feature_cols: List of feature column names
        """
        self.sequences = sequences
        self.feature_cols = feature_cols
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_df, output_df, metadata = self.sequences[idx]
        
        # Extract features
        X = input_df[self.feature_cols].values.astype(np.float32)
        
        # Extract targets (x, y positions)
        y = output_df[['x', 'y']].values.astype(np.float32)
        
        # Pad sequences if needed
        max_output_len = metadata.get('max_output_len', len(y))
        if len(y) < max_output_len:
            padding = np.zeros((max_output_len - len(y), 2), dtype=np.float32)
            y = np.vstack([y, padding])
        
        return torch.FloatTensor(X), torch.FloatTensor(y), len(output_df)


class LSTMTrajectoryModel(nn.Module):
    """LSTM model for trajectory prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMTrajectoryModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM encoder for input sequence
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder for generating future positions
        self.decoder = nn.LSTM(
            input_size=2,  # Previous (x, y) position
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Predict (x, y)
        )
    
    def forward(self, x, num_future_steps):
        """
        Args:
            x: Input sequence of shape (batch, seq_len, features)
            num_future_steps: Number of future steps to predict
            
        Returns:
            Predicted positions of shape (batch, num_future_steps, 2)
        """
        batch_size = x.size(0)
        
        # Encode input sequence
        _, (hidden, cell) = self.lstm(x)
        
        # Initialize decoder input with last known position
        # Use last two features as x, y (assuming they are in the feature set)
        decoder_input = x[:, -1:, :2]  # (batch, 1, 2)
        
        predictions = []
        
        # Autoregressive decoding
        for _ in range(num_future_steps):
            # Decode one step
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Predict next position
            pred = self.fc(decoder_output)  # (batch, 1, 2)
            predictions.append(pred)
            
            # Use prediction as next input
            decoder_input = pred
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # (batch, num_future_steps, 2)
        
        return predictions


class LSTMTrainer:
    """Trainer for LSTM trajectory model."""
    
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for X, y, lengths in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            num_future_steps = y.size(1)
            predictions = self.model(X, num_future_steps)
            
            # Calculate loss (only for valid positions)
            loss = 0
            for i, length in enumerate(lengths):
                loss += self.criterion(predictions[i, :length], y[i, :length])
            loss = loss / len(lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y, lengths in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                num_future_steps = y.size(1)
                predictions = self.model(X, num_future_steps)
                
                # Calculate loss
                loss = 0
                for i, length in enumerate(lengths):
                    loss += self.criterion(predictions[i, :length], y[i, :length])
                loss = loss / len(lengths)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=10, verbose=True):
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        return self.history
    
    def predict(self, X, num_future_steps):
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            predictions = self.model(X, num_future_steps)
            return predictions.cpu().numpy()[0]
    
    def save_model(self, path):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def prepare_lstm_data(input_df: pd.DataFrame, output_df: pd.DataFrame, 
                     feature_cols: List[str]) -> List[Tuple]:
    """
    Prepare data for LSTM training.
    
    Args:
        input_df: Input tracking data
        output_df: Output tracking data
        feature_cols: List of feature columns to use
        
    Returns:
        List of (input_seq, output_seq, metadata) tuples
    """
    sequences = []
    
    # Group by play and player
    for (game_id, play_id, nfl_id), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
        # Only include players to predict
        if not group['player_to_predict'].iloc[0]:
            continue
        
        # Sort by frame
        input_seq = group.sort_values('frame_id')
        
        # Get corresponding output
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
            'num_frames': len(output_seq)
        }
        
        sequences.append((input_seq, output_seq, metadata))
    
    return sequences


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch is required for LSTM model. Please install: pip install torch")
    else:
        print("LSTM model module loaded successfully")
        print("Use this module to train LSTM models for trajectory prediction")
