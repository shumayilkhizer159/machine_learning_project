"""
Advanced LSTM Model with Attention Mechanism for NFL Player Trajectory Prediction
This model uses bidirectional LSTM with attention for better trajectory prediction.
Optimized for GPU training.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence modeling."""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context, attention_weights.squeeze(-1)


class AdvancedLSTMModel(nn.Module):
    """
    Advanced LSTM model with:
    - Bidirectional LSTM encoder
    - Attention mechanism
    - Autoregressive decoder
    - Residual connections
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(AdvancedLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Context projection
        self.context_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=2 + hidden_size,  # (x, y) + context
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Predict (x, y)
        )
        
        # Additional features for decoder
        self.ball_embedding = nn.Linear(2, 32)  # Ball position embedding
        self.role_embedding = nn.Embedding(10, 32)  # Player role embedding
        
    def forward(self, x, ball_pos, player_role, num_future_steps, teacher_forcing_ratio=0.5):
        """
        Args:
            x: Input sequence (batch, seq_len, features)
            ball_pos: Ball landing position (batch, 2)
            player_role: Player role indices (batch,)
            num_future_steps: Number of steps to predict
            teacher_forcing_ratio: Probability of using ground truth during training
        
        Returns:
            predictions: (batch, num_future_steps, 2)
            attention_weights: (batch, seq_len)
        """
        batch_size = x.size(0)
        
        # Embed input
        x_embedded = self.input_embedding(x)
        
        # Encode input sequence
        encoder_output, (hidden, cell) = self.encoder(x_embedded)
        
        # Apply attention
        context, attention_weights = self.attention(encoder_output)
        context = self.context_proj(context)
        
        # Prepare decoder hidden states (use only forward direction)
        hidden = hidden[:self.num_layers]  # Take forward direction only
        cell = cell[:self.num_layers]
        
        # Initialize decoder input with last known position
        decoder_input = x[:, -1, :2].unsqueeze(1)  # (batch, 1, 2)
        
        # Add context to decoder input
        context_expanded = context.unsqueeze(1)  # (batch, 1, hidden_size)
        
        predictions = []
        
        for t in range(num_future_steps):
            # Concatenate position with context
            decoder_input_with_context = torch.cat([decoder_input, context_expanded], dim=2)
            
            # Decode one step
            decoder_output, (hidden, cell) = self.decoder(decoder_input_with_context, (hidden, cell))
            
            # Predict next position
            pred = self.output_layer(decoder_output)  # (batch, 1, 2)
            predictions.append(pred)
            
            # Use prediction as next input (or ground truth with teacher forcing)
            decoder_input = pred
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # (batch, num_future_steps, 2)
        
        return predictions, attention_weights


class GRUModel(nn.Module):
    """
    GRU-based model as an alternative to LSTM.
    GRUs are faster and sometimes perform better on shorter sequences.
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_size * 2)
        self.context_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=2 + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, ball_pos, player_role, num_future_steps):
        batch_size = x.size(0)
        
        # Embed and encode
        x_embedded = self.input_embedding(x)
        encoder_output, hidden = self.encoder(x_embedded)
        
        # Attention
        context, attention_weights = self.attention(encoder_output)
        context = self.context_proj(context)
        
        # Prepare decoder
        hidden = hidden[:self.num_layers]
        decoder_input = x[:, -1, :2].unsqueeze(1)
        context_expanded = context.unsqueeze(1)
        
        predictions = []
        
        for t in range(num_future_steps):
            decoder_input_with_context = torch.cat([decoder_input, context_expanded], dim=2)
            decoder_output, hidden = self.decoder(decoder_input_with_context, hidden)
            pred = self.output_layer(decoder_output)
            predictions.append(pred)
            decoder_input = pred
        
        predictions = torch.cat(predictions, dim=1)
        return predictions, attention_weights


class NFLTrajectoryDataset(Dataset):
    """Optimized dataset for NFL trajectory prediction."""
    
    def __init__(self, sequences, feature_cols, max_input_len=50, max_output_len=30):
        self.sequences = sequences
        self.feature_cols = feature_cols
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_df, output_df, metadata = self.sequences[idx]
        
        # Extract features
        X = input_df[self.feature_cols].values.astype(np.float32)
        y = output_df[['x', 'y']].values.astype(np.float32)
        
        # Pad/truncate input sequence
        if len(X) < self.max_input_len:
            padding = np.zeros((self.max_input_len - len(X), X.shape[1]), dtype=np.float32)
            X = np.vstack([X, padding])
        else:
            X = X[-self.max_input_len:]
        
        # Pad/truncate output sequence
        actual_output_len = len(y)
        if len(y) < self.max_output_len:
            padding = np.zeros((self.max_output_len - len(y), 2), dtype=np.float32)
            y = np.vstack([y, padding])
        else:
            y = y[:self.max_output_len]
        
        # Get ball position and player role
        ball_pos = input_df[['ball_land_x', 'ball_land_y']].iloc[-1].values.astype(np.float32)
        player_role = metadata.get('player_role_encoded', 0)
        
        return (
            torch.FloatTensor(X),
            torch.FloatTensor(y),
            torch.FloatTensor(ball_pos),
            torch.LongTensor([player_role]),
            actual_output_len
        )


class ModelTrainer:
    """Trainer for LSTM/GRU models with GPU support."""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}
        self.best_val_loss = float('inf')
    
    def calculate_rmse(self, predictions, targets, lengths):
        """Calculate competition RMSE metric."""
        total_error = 0
        total_count = 0
        
        for i, length in enumerate(lengths):
            pred = predictions[i, :length]
            target = targets[i, :length]
            error = ((pred - target) ** 2).sum()
            total_error += error
            total_count += length
        
        rmse = torch.sqrt(total_error / (2 * total_count))
        return rmse
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_rmse = 0
        
        for X, y, ball_pos, player_role, lengths in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            ball_pos = ball_pos.to(self.device)
            player_role = player_role.to(self.device)
            
            # Forward pass
            max_len = y.size(1)
            predictions, _ = self.model(X, ball_pos, player_role, max_len)
            
            # Calculate loss only for valid positions
            loss = 0
            for i, length in enumerate(lengths):
                loss += self.criterion(predictions[i, :length], y[i, :length])
            loss = loss / len(lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate RMSE
            with torch.no_grad():
                rmse = self.calculate_rmse(predictions, y, lengths)
            
            total_loss += loss.item()
            total_rmse += rmse.item()
        
        return total_loss / len(train_loader), total_rmse / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_rmse = 0
        
        with torch.no_grad():
            for X, y, ball_pos, player_role, lengths in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                ball_pos = ball_pos.to(self.device)
                player_role = player_role.to(self.device)
                
                max_len = y.size(1)
                predictions, _ = self.model(X, ball_pos, player_role, max_len)
                
                loss = 0
                for i, length in enumerate(lengths):
                    loss += self.criterion(predictions[i, :length], y[i, :length])
                loss = loss / len(lengths)
                
                rmse = self.calculate_rmse(predictions, y, lengths)
                
                total_loss += loss.item()
                total_rmse += rmse.item()
        
        return total_loss / len(val_loader), total_rmse / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=50, save_path='best_model.pth'):
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            train_loss, train_rmse = self.train_epoch(train_loader)
            val_loss, val_rmse = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_rmse'].append(val_rmse)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'history': self.history
                }, save_path)
                print(f"  ✓ Saved best model (Val RMSE: {val_rmse:.4f})")
        
        return self.history
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best Val RMSE: {checkpoint['val_rmse']:.4f}")


if __name__ == '__main__':
    print("Advanced LSTM model module loaded")
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
