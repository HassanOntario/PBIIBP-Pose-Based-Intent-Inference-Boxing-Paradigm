import math
import models
import numpy as np


#pose keypoints
lwrist = models.MoveNet_Vision.get_keypoint_by_name('left_wrist')
rwrist = models.MoveNet_Vision.get_keypoint_by_name('right_wrist')
lelbow = models.MoveNet_Vision.get_keypoint_by_name('left_elbow')
relbow = models.MoveNet_Vision.get_keypoint_by_name('right_elbow')
lshoulder = models.MoveNet_Vision.get_keypoint_by_name('left_shoulder')
rshoulder = models.MoveNet_Vision.get_keypoint_by_name('right_shoulder')

#time step
dt = 1 / 30  # 30 FPS
p_t     = np.array([x_t, y_t])
p_prev  = np.array([x_prev, y_prev])

 
def velocity(p_t, p_prev, dt):
    return (p_t - p_prev) / dt

def speed(v):
    return np.linalg.norm(v)
 
def acceleration(v_t, v_prev, dt):
    return (v_t - v_prev) / dt

 
def relative_velocity(v_wrist, v_shoulder):
    return v_wrist - v_shoulder

def radial_velocity(wrist_pos, shoulder_pos, v_rel):
    r = wrist_pos - shoulder_pos
    r_hat = r / (np.linalg.norm(r) + 1e-8)  # avoid divide by zero
    return np.dot(v_rel, r_hat)

def tangential_velocity(v_rel, v_radial, r_hat):
    v_radial_vec = v_radial * r_hat
    v_tan_vec = v_rel - v_radial_vec
    return np.linalg.norm(v_tan_vec)

def elbow_angle(shoulder, elbow, wrist):
    u = shoulder - elbow
    v = wrist - elbow

    dot = np.dot(u, v)
    norm = np.linalg.norm(u) * np.linalg.norm(v)

    cos_theta = np.clip(dot / (norm + 1e-8), -1.0, 1.0)
    return np.arccos(cos_theta)  # radians

def angular_velocity(theta_t, theta_prev, dt):
    return (theta_t - theta_prev) / dt


v_wrist = velocity(wrist_t, wrist_prev, dt)
v_shoulder = velocity(shoulder_t, shoulder_prev, dt)

v_rel = relative_velocity(v_wrist, v_shoulder)

v_rad = radial_velocity(wrist_t, shoulder_t, v_rel)

theta = elbow_angle(shoulder_t, elbow_t, wrist_t)
omega = angular_velocity(theta, theta_prev, dt)