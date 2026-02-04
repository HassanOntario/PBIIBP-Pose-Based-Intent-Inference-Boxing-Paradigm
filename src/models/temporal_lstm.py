"""
Temporal LSTM model for pose-based intent inference.

This module implements an LSTM-based neural network for processing
sequential pose data from Google MoveNet Lightning and inferring
boxing intent/actions.
"""

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM model for boxing intent inference from pose sequences.
    
    This model processes sequences of pose keypoints (from MoveNet Lightning)
    to predict boxing intent or action categories.
    
    MoveNet Lightning outputs 17 keypoints, each with (x, y, confidence),
    resulting in 51 features per frame (17 * 3 = 51).
    
    Args:
        input_size: Number of input features per timestep (default: 51 for MoveNet).
        hidden_size: Number of LSTM hidden units (default: 128).
        num_layers: Number of stacked LSTM layers (default: 2).
        num_classes: Number of output classes for intent classification (default: 4).
        dropout: Dropout probability between LSTM layers (default: 0.3).
        bidirectional: Whether to use bidirectional LSTM (default: False).
    """
    
    def __init__(
        self,
        input_size: int = 51,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super(TemporalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer for temporal sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Fully connected layers for classification
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
               For MoveNet data: (batch, seq_len, 51).
               
        Returns:
            Output tensor of shape (batch_size, num_classes) containing
            class logits for intent prediction.
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, sequence_length, hidden_size * num_directions)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the output from the last timestep for classification
        # For bidirectional, concatenate the last forward and first backward hidden states
        if self.bidirectional:
            # Concatenate the last hidden states from both directions
            last_hidden = torch.cat(
                (hidden[-2, :, :], hidden[-1, :, :]), dim=1
            )
        else:
            # Use the last hidden state from the final layer
            last_hidden = hidden[-1, :, :]
        
        # Pass through fully connected layers
        output = self.fc(last_hidden)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Tensor of predicted class indices of shape (batch_size,).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities using softmax.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Tensor of class probabilities of shape (batch_size, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
