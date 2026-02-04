"""
Training utilities for the temporal LSTM model.

This module provides functions for training and evaluating the
TemporalLSTM model on pose sequence data.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cpu",
    early_stopping_patience: int = 10,
) -> Dict[str, list]:
    """
    Train the temporal LSTM model.
    
    Args:
        model: The TemporalLSTM model to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.
        epochs: Number of training epochs (default: 100).
        learning_rate: Learning rate for Adam optimizer (default: 0.001).
        device: Device to train on ('cpu' or 'cuda').
        early_stopping_patience: Number of epochs to wait for improvement
                                  before stopping (default: 10).
        
    Returns:
        Dictionary containing training history with keys:
        - 'train_loss': List of training losses per epoch
        - 'train_acc': List of training accuracies per epoch
        - 'val_loss': List of validation losses per epoch (if val_loader provided)
        - 'val_acc': List of validation accuracies per epoch (if val_loader provided)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        
        # Validation phase
        if val_loader is not None:
            val_loss, val_accuracy = evaluate_model(
                model, val_loader, device=device
            )
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
                
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
            )
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The trained model to evaluate.
        data_loader: DataLoader for the evaluation data.
        device: Device to evaluate on ('cpu' or 'cuda').
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy
