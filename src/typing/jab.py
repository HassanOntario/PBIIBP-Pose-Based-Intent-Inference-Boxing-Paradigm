"""
Jab-specific kinematic analysis.

Uses KinematicFeatureExtractor for feature computation and provides
jab-specific thresholds / classification helpers.
"""

import numpy as np

from src.features.kinematic_features import (
    KinematicFeatureExtractor,
    KEYPOINT_INDEX,
    velocity,
    speed,
    acceleration,
    relative_velocity,
    radial_velocity,
    tangential_velocity,
    joint_angle,
    angular_velocity,
)

# MoveNet keypoint indices used for jab analysis
LEFT_WRIST = KEYPOINT_INDEX["left_wrist"]
RIGHT_WRIST = KEYPOINT_INDEX["right_wrist"]
LEFT_ELBOW = KEYPOINT_INDEX["left_elbow"]
RIGHT_ELBOW = KEYPOINT_INDEX["right_elbow"]
LEFT_SHOULDER = KEYPOINT_INDEX["left_shoulder"]
RIGHT_SHOULDER = KEYPOINT_INDEX["right_shoulder"]

# Default frame rate
DT = 1 / 30  # 30 FPS


def compute_jab_features(pose_data: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Compute kinematic features relevant to jab detection.

    This is a convenience wrapper around ``KinematicFeatureExtractor``
    that returns the full set of arm kinematic features for both arms.

    Args:
        pose_data: Raw MoveNet pose data of shape ``(num_frames, 51)``.
        fps: Frame rate of the recording.

    Returns:
        Array of shape ``(num_frames, 22)`` with kinematic features
        (11 per arm × 2 arms).
    """
    extractor = KinematicFeatureExtractor(fps=fps, arms=("left", "right"))
    return extractor.extract(pose_data)