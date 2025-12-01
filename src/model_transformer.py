"""
Transformer-based Model for NFL Player Trajectory Prediction
Uses self-attention mechanisms for better long-range dependencies.
Optimized for GPU training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTrajectoryModel(nn.Module):
    """
    Transformer model for trajectory prediction.
    
    Architecture:
    - Input embedding with positional encoding
    - Transformer encoder for input sequence
    - Transformer decoder for output sequence
    - Multi-head attention for capturing player interactions
    """
    
    def __init__(self, input_size, d_model=256, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TransformerTrajectoryModel, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Output embedding (for decoder input)
        self.output_embedding = nn.Linear(2, d_model)  # (x, y) positions
        self.pos_decoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, src, tgt):
        """
        Args:
            src: Source sequence (batch, src_len, input_size)
            tgt: Target sequence (batch, tgt_len, 2) - for teacher forcing
        
        Returns:
            predictions: (batch, tgt_len, 2)
        """
        # Embed source
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Embed target
        tgt_embedded = self.output_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_decoder(tgt_embedded)
        
        # Generate target mask
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        
        # Transformer forward
        output = self.transformer(
            src=src,
            tgt=tgt_embedded,
            tgt_mask=tgt_mask
        )
        
        # Project to (x, y)
        predictions = self.output_proj(output)
        
        return predictions
    
    def predict_autoregressive(self, src, num_steps, start_pos):
        """
        Autoregressive prediction without teacher forcing.
        
        Args:
            src: Source sequence (batch, src_len, input_size)
            num_steps: Number of future steps to predict
            start_pos: Starting position (batch, 2)
        
        Returns:
            predictions: (batch, num_steps, 2)
        """
        self.eval()
        batch_size = src.size(0)
        
        # Embed source
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Initialize decoder input with start position
        tgt = start_pos.unsqueeze(1)  # (batch, 1, 2)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_steps):
                # Embed current target sequence
                tgt_embedded = self.output_embedding(tgt) * math.sqrt(self.d_model)
                tgt_embedded = self.pos_decoder(tgt_embedded)
                
                # Generate mask
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
                
                # Transformer forward
                output = self.transformer(
                    src=src,
                    tgt=tgt_embedded,
                    tgt_mask=tgt_mask
                )
                
                # Get prediction for last position
                next_pos = self.output_proj(output[:, -1:, :])  # (batch, 1, 2)
                predictions.append(next_pos)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_pos], dim=1)
        
        predictions = torch.cat(predictions, dim=1)
        return predictions


class SpatialTransformerModel(nn.Module):
    """
    Spatial Transformer that models player interactions.
    Treats each player at each timestep as a node in a graph.
    """
    
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(SpatialTransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Player embedding
        self.player_embedding = nn.Linear(input_size, d_model)
        
        # Spatial positional encoding (based on field position)
        self.spatial_encoding = nn.Sequential(
            nn.Linear(2, 64),  # (x, y) position
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
        # Temporal positional encoding
        self.temporal_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * 30)  # Predict 30 future frames at once
        )
    
    def forward(self, x, positions):
        """
        Args:
            x: Player features (batch, seq_len, input_size)
            positions: Player positions (batch, seq_len, 2)
        
        Returns:
            predictions: (batch, 30, 2)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed player features
        x_embedded = self.player_embedding(x)
        
        # Add spatial encoding
        spatial_enc = self.spatial_encoding(positions)
        x_embedded = x_embedded + spatial_enc
        
        # Add temporal encoding
        x_embedded = self.temporal_encoding(x_embedded)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x_embedded)
        
        # Use last timestep for prediction
        last_encoded = encoded[:, -1, :]  # (batch, d_model)
        
        # Predict all future frames
        predictions = self.prediction_head(last_encoded)  # (batch, 60)
        predictions = predictions.view(batch_size, 30, 2)  # (batch, 30, 2)
        
        return predictions


class TransformerTrainer:
    """Trainer for Transformer models."""
    
    def __init__(self, model, device='cuda', learning_rate=0.00001):
        self.model = model.to(device)
        self.device = device
        
        # Use Adam with warmup
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min((step + 1) ** (-0.5), (step + 1) * (4000 ** (-1.5)))
        )
        
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X, y, _, _, lengths in train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Prepare decoder input (shifted right)
            # Start with last known position
            decoder_input = torch.zeros_like(y)
            decoder_input[:, 0, :] = X[:, -1, :2]  # Last known position
            decoder_input[:, 1:, :] = y[:, :-1, :]  # Shifted targets
            
            # Forward pass
            predictions = self.model(X, decoder_input)
            
            # Calculate loss
            loss = 0
            for i, length in enumerate(lengths):
                loss += self.criterion(predictions[i, :length], y[i, :length])
            loss = loss / len(lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y, _, _, lengths in val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Autoregressive prediction
                start_pos = X[:, -1, :2]
                predictions = self.model.predict_autoregressive(X, y.size(1), start_pos)
                
                loss = 0
                for i, length in enumerate(lengths):
                    loss += self.criterion(predictions[i, :length], y[i, :length])
                loss = loss / len(lengths)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=30, save_path='best_transformer.pth'):
        print(f"Training Transformer on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, save_path)
                print(f"  ✓ Saved best model")
        
        return self.history


if __name__ == '__main__':
    print("Transformer model module loaded")
    print("GPU available:", torch.cuda.is_available())
