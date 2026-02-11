"""
Kinematic feature extraction from MoveNet pose keypoints.

Computes per-frame engineered features (velocities, accelerations, joint
angles, etc.) from consecutive pose frames.  These features are designed
to capture the biomechanical signatures of boxing strikes and can be
concatenated with raw keypoint data before being fed to the TemporalLSTM.

MoveNet keypoint layout (17 keypoints × 3 values = 51 per frame):
    Index  Name
    -----  ----
    0      nose
    1      left_eye
    2      right_eye
    3      left_ear
    4      right_ear
    5      left_shoulder
    6      right_shoulder
    7      left_elbow
    8      right_elbow
    9      left_wrist
    10     right_wrist
    11     left_hip
    12     right_hip
    13     left_knee
    14     right_knee
    15     left_ankle
    16     right_ankle

Each keypoint stores (y, x, confidence) in the MoveNet output.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# MoveNet keypoint indices
# ---------------------------------------------------------------------------
KEYPOINT_INDEX: Dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Default feature groups for boxing intent inference
DEFAULT_ARMS = ("left", "right")


# ---------------------------------------------------------------------------
# Primitive kinematic functions (stateless, pure NumPy)
# ---------------------------------------------------------------------------

def velocity(p_t: np.ndarray, p_prev: np.ndarray, dt: float) -> np.ndarray:
    """2-D velocity vector between two positions."""
    return (p_t - p_prev) / dt


def speed(v: np.ndarray) -> float:
    """Scalar speed (L2 norm of velocity)."""
    return float(np.linalg.norm(v))


def acceleration(v_t: np.ndarray, v_prev: np.ndarray, dt: float) -> np.ndarray:
    """2-D acceleration vector from consecutive velocities."""
    return (v_t - v_prev) / dt


def relative_velocity(
    v_distal: np.ndarray, v_proximal: np.ndarray
) -> np.ndarray:
    """Velocity of a distal joint relative to a proximal joint."""
    return v_distal - v_proximal


def radial_velocity(
    distal_pos: np.ndarray,
    proximal_pos: np.ndarray,
    v_rel: np.ndarray,
) -> float:
    """Component of relative velocity along the radial direction."""
    r = distal_pos - proximal_pos
    r_hat = r / (np.linalg.norm(r) + 1e-8)
    return float(np.dot(v_rel, r_hat))


def tangential_velocity(v_rel: np.ndarray, v_rad: float, r_hat: np.ndarray) -> float:
    """Component of relative velocity perpendicular to the radial direction."""
    v_rad_vec = v_rad * r_hat
    return float(np.linalg.norm(v_rel - v_rad_vec))


def joint_angle(
    proximal: np.ndarray, joint: np.ndarray, distal: np.ndarray
) -> float:
    """
    Angle at *joint* formed by (proximal → joint → distal), in radians.

    For an elbow angle: proximal=shoulder, joint=elbow, distal=wrist.
    """
    u = proximal - joint
    v = distal - joint
    cos_theta = np.clip(
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8),
        -1.0,
        1.0,
    )
    return float(np.arccos(cos_theta))


def angular_velocity(theta_t: float, theta_prev: float, dt: float) -> float:
    """Scalar angular velocity from consecutive angle measurements."""
    return (theta_t - theta_prev) / dt


# ---------------------------------------------------------------------------
# Helper: extract (y, x) position of a keypoint from a flat 51-d vector
# ---------------------------------------------------------------------------

def _kp_pos(frame: np.ndarray, name: str) -> np.ndarray:
    """Return the (y, x) position of a named keypoint from a flat frame."""
    idx = KEYPOINT_INDEX[name]
    return frame[idx * 3: idx * 3 + 2].astype(np.float64)


# ---------------------------------------------------------------------------
# KinematicFeatureExtractor
# ---------------------------------------------------------------------------

class KinematicFeatureExtractor:
    """
    Compute per-frame kinematic features from raw MoveNet pose data.

    For each arm (left/right) the following features are produced per frame:

    ========  ======  ============================================
    Feature   Dim     Description
    ========  ======  ============================================
    v_wrist   2       Wrist velocity (y, x)
    s_wrist   1       Wrist speed (scalar)
    a_wrist   2       Wrist acceleration (y, x)
    v_rel     2       Wrist velocity relative to shoulder
    v_rad     1       Radial component of v_rel
    v_tan     1       Tangential component of v_rel
    theta     1       Elbow angle (radians)
    omega     1       Elbow angular velocity
    ========  ======  ============================================

    Per arm: **11 features**.  Both arms → **22 features**.

    The first two frames of the output are zero-padded because velocity
    and acceleration require Δt look-back of 1 and 2 frames respectively.

    Args:
        fps: Frame rate of the input data (default: 30).
        arms: Which arms to compute features for
              (default: ``("left", "right")``).
    """

    FEATURES_PER_ARM = 11  # keep in sync with the table above

    def __init__(
        self,
        fps: float = 30.0,
        arms: Tuple[str, ...] = DEFAULT_ARMS,
    ):
        self.dt = 1.0 / fps
        self.arms = arms

    @property
    def num_features(self) -> int:
        """Total number of engineered features per frame."""
        return self.FEATURES_PER_ARM * len(self.arms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Extract kinematic features from a sequence of raw MoveNet frames.

        Args:
            pose_data: Array of shape ``(num_frames, 51)``.

        Returns:
            Array of shape ``(num_frames, num_features)`` with engineered
            kinematic features.  The first two rows are zero because
            acceleration needs two frames of history.
        """
        n_frames = len(pose_data)
        out = np.zeros((n_frames, self.num_features), dtype=np.float64)

        for t in range(n_frames):
            feat_offset = 0
            for arm in self.arms:
                arm_feat = self._arm_features(pose_data, t, arm)
                out[t, feat_offset: feat_offset + self.FEATURES_PER_ARM] = arm_feat
                feat_offset += self.FEATURES_PER_ARM

        return out.astype(np.float32)

    def extract_and_concat(self, pose_data: np.ndarray) -> np.ndarray:
        """
        Concatenate raw pose data with engineered kinematic features.

        Args:
            pose_data: Array of shape ``(num_frames, 51)``.

        Returns:
            Array of shape ``(num_frames, 51 + num_features)``.
        """
        kinematic = self.extract(pose_data)
        return np.concatenate([pose_data, kinematic], axis=-1)

    def feature_names(self) -> List[str]:
        """Return ordered list of engineered feature names."""
        names: List[str] = []
        for arm in self.arms:
            prefix = f"{arm}_"
            names.extend([
                f"{prefix}v_wrist_y",
                f"{prefix}v_wrist_x",
                f"{prefix}s_wrist",
                f"{prefix}a_wrist_y",
                f"{prefix}a_wrist_x",
                f"{prefix}v_rel_y",
                f"{prefix}v_rel_x",
                f"{prefix}v_rad",
                f"{prefix}v_tan",
                f"{prefix}theta",
                f"{prefix}omega",
            ])
        return names

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _arm_features(
        self, pose_data: np.ndarray, t: int, side: str
    ) -> np.ndarray:
        """Compute the 11-d feature vector for one arm at frame *t*."""
        wrist_name = f"{side}_wrist"
        elbow_name = f"{side}_elbow"
        shoulder_name = f"{side}_shoulder"

        wrist_t = _kp_pos(pose_data[t], wrist_name)
        shoulder_t = _kp_pos(pose_data[t], shoulder_name)
        elbow_t = _kp_pos(pose_data[t], elbow_name)

        # --- velocity & speed (need t >= 1) ---
        if t >= 1:
            wrist_prev = _kp_pos(pose_data[t - 1], wrist_name)
            shoulder_prev = _kp_pos(pose_data[t - 1], shoulder_name)
            v_wrist = velocity(wrist_t, wrist_prev, self.dt)
            v_shoulder = velocity(shoulder_t, shoulder_prev, self.dt)
        else:
            v_wrist = np.zeros(2)
            v_shoulder = np.zeros(2)

        s_wrist = speed(v_wrist)

        # --- acceleration (need t >= 2) ---
        if t >= 2:
            wrist_prev2 = _kp_pos(pose_data[t - 2], wrist_name)
            v_wrist_prev = velocity(
                _kp_pos(pose_data[t - 1], wrist_name), wrist_prev2, self.dt
            )
            a_wrist = acceleration(v_wrist, v_wrist_prev, self.dt)
        else:
            a_wrist = np.zeros(2)

        # --- relative, radial, tangential velocity ---
        v_rel = relative_velocity(v_wrist, v_shoulder)

        r = wrist_t - shoulder_t
        r_hat = r / (np.linalg.norm(r) + 1e-8)

        v_rad = radial_velocity(wrist_t, shoulder_t, v_rel)
        v_tan = tangential_velocity(v_rel, v_rad, r_hat)

        # --- elbow angle & angular velocity ---
        theta = joint_angle(shoulder_t, elbow_t, wrist_t)

        if t >= 1:
            elbow_prev = _kp_pos(pose_data[t - 1], elbow_name)
            shoulder_prev_pos = _kp_pos(pose_data[t - 1], shoulder_name)
            wrist_prev_pos = _kp_pos(pose_data[t - 1], wrist_name)
            theta_prev = joint_angle(shoulder_prev_pos, elbow_prev, wrist_prev_pos)
            omega = angular_velocity(theta, theta_prev, self.dt)
        else:
            omega = 0.0

        return np.array([
            v_wrist[0], v_wrist[1],       # 2: wrist velocity (y, x)
            s_wrist,                       # 1: wrist speed
            a_wrist[0], a_wrist[1],        # 2: wrist acceleration (y, x)
            v_rel[0], v_rel[1],            # 2: relative velocity (y, x)
            v_rad,                         # 1: radial velocity
            v_tan,                         # 1: tangential velocity
            theta,                         # 1: elbow angle
            omega,                         # 1: angular velocity
        ])
