"""
Pose dataset module for loading and processing MoveNet pose data.

This module provides utilities for creating temporal sequences from
pose keypoint data — optionally augmented with engineered kinematic
features — for use with LSTM-based models.
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.kinematic_features import KinematicFeatureExtractor


class PoseDataset(Dataset):
    """
    PyTorch Dataset for pose sequence data.
    
    Handles loading and preprocessing of MoveNet Lightning pose keypoint
    sequences for boxing intent classification.
    
    Args:
        sequences: Array of pose sequences with shape 
                   (num_samples, sequence_length, num_features).
        labels: Array of intent labels with shape (num_samples,).
        transform: Optional transform to apply to each sequence.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
        if len(self.sequences) != len(self.labels):
            raise ValueError(
                f"Number of sequences ({len(self.sequences)}) must match "
                f"number of labels ({len(self.labels)})"
            )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (sequence, label) tensors.
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, label


def create_sequences(
    pose_data: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = 30,
    stride: int = 1,
    feature_extractor: Optional[KinematicFeatureExtractor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from continuous pose data.
    
    This function takes raw pose data, optionally augments each frame
    with engineered kinematic features, and creates overlapping
    sequences suitable for LSTM training.
    
    Args:
        pose_data: Raw pose data of shape (num_frames, num_features).
                   For MoveNet: (num_frames, 51).
        labels: Frame-level labels of shape (num_frames,).
        sequence_length: Number of frames per sequence (default: 30).
        stride: Step size between consecutive sequences (default: 1).
        feature_extractor: Optional ``KinematicFeatureExtractor`` instance.
                           When provided, engineered features are computed
                           from the raw pose data and concatenated along the
                           feature axis before windowing.
        
    Returns:
        Tuple of (sequences, sequence_labels) where:
        - sequences has shape (num_sequences, sequence_length, total_features)
          total_features = num_features (+ num_engineered if extractor given)
        - sequence_labels has shape (num_sequences,)
        
    Example:
        >>> poses = np.random.randn(100, 51)
        >>> labels = np.random.randint(0, 4, 100)
        >>> # Raw only
        >>> seqs, sl = create_sequences(poses, labels, sequence_length=30)
        >>> print(seqs.shape)  # (71, 30, 51)
        >>>
        >>> # Raw + kinematic features (51 + 22 = 73)
        >>> extractor = KinematicFeatureExtractor(fps=30)
        >>> seqs, sl = create_sequences(poses, labels, 30, feature_extractor=extractor)
        >>> print(seqs.shape)  # (71, 30, 73)
    """
    if len(pose_data) != len(labels):
        raise ValueError(
            f"pose_data ({len(pose_data)}) and labels ({len(labels)}) "
            f"must have the same length"
        )
    
    if sequence_length > len(pose_data):
        raise ValueError(
            f"sequence_length ({sequence_length}) cannot be greater than "
            f"data length ({len(pose_data)})"
        )
    
    # Optionally augment with engineered features
    if feature_extractor is not None:
        frame_data = feature_extractor.extract_and_concat(pose_data)
    else:
        frame_data = pose_data
    
    sequences: List[np.ndarray] = []
    sequence_labels: List[int] = []
    
    for i in range(0, len(frame_data) - sequence_length + 1, stride):
        seq = frame_data[i:i + sequence_length]
        sequences.append(seq)
        
        # Use the label of the last frame in the sequence
        # (the intent we're predicting)
        sequence_labels.append(labels[i + sequence_length - 1])
    
    return np.array(sequences), np.array(sequence_labels)


def normalize_poses(pose_data: np.ndarray) -> np.ndarray:
    """
    Normalize pose keypoint data.
    
    Normalizes x, y coordinates to be centered and scaled.
    Assumes MoveNet format with 17 keypoints * 3 values (x, y, confidence).
    
    Args:
        pose_data: Pose data of shape (..., 51) where last dim is
                   [x1, y1, c1, x2, y2, c2, ..., x17, y17, c17].
                   
    Returns:
        Normalized pose data with same shape.
    """
    normalized = pose_data.copy()
    
    # Reshape to separate keypoints: (..., 17, 3)
    original_shape = normalized.shape
    reshaped = normalized.reshape(*original_shape[:-1], 17, 3)
    
    # Normalize x and y coordinates (indices 0 and 1)
    # Keep confidence scores (index 2) unchanged
    xy_coords = reshaped[..., :2]
    
    # Center around mean and scale
    mean = xy_coords.mean(axis=-2, keepdims=True)
    std = xy_coords.std(axis=-2, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    reshaped[..., :2] = (xy_coords - mean) / std
    
    return reshaped.reshape(original_shape)
