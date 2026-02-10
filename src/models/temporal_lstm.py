"""
Temporal LSTM model for pose-based intent inference.

This module implements a unidirectional LSTM-based neural network for
processing sequential pose data and inferring boxing intent/actions.
The model is input-agnostic and accepts any combination of pose keypoints
and engineered kinematic features as input.
"""

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """
    Temporal LSTM model for early intent inference from pose sequences.

    Processes variable-length sequences of per-frame feature vectors (e.g.
    pose keypoints, joint velocities, or other kinematic descriptors) through
    a unidirectional LSTM and classifies the sequence into one of several
    intent categories.

    **Temporal aggregation (mean pooling)** is used instead of taking only the
    last hidden state.  This is critical for *early* intent inference: when a
    sequence is still incomplete (truncated), the last timestep alone may not
    yet carry a clear signal.  Mean-pooling over all observed timesteps lets
    the classifier leverage partial motion cues that appear earlier in the
    sequence, improving robustness at short horizons.

    The architecture is kept deliberately lightweight (no attention, no CNN,
    no transformer) so it can run in real time on CPU or edge hardware.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Number of LSTM hidden units (default: 128).
        num_layers: Number of stacked LSTM layers (default: 2).
        num_classes: Number of output classes for intent classification
                     (default: 4).
        dropout: Dropout probability applied between LSTM layers and in the
                 classification head (default: 0.3).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Unidirectional LSTM — no future-leakage by design.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Classification head applied to the temporally-pooled representation.
        # Structured so that future extensions (e.g. per-timestep logits or
        # early-exit heads) can re-use the LSTM outputs before this block.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal LSTM.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_size)``.

        Returns:
            Logits tensor of shape ``(batch_size, num_classes)``.
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Temporal mean pooling over all timesteps.
        # This aggregates information from the entire observed sequence,
        # which is essential for early intent inference where the sequence
        # may be truncated and the final timestep alone is insufficient.
        pooled = lstm_out.mean(dim=1)  # (batch, hidden_size)

        logits = self.classifier(pooled)
        return logits

    # ------------------------------------------------------------------
    # Convenience inference methods (public API)
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted class indices.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch_size,)`` with predicted class indices.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities via softmax.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch_size, num_classes)`` with class
            probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
