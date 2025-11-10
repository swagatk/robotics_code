"""
Dobot 4-DOF robot D-H parameters configuration"""
import numpy as np


def get_dobot_dh_params():
    # D-H parameters for Dobot
    L1, L2, L3, L4, L5 = 13.8, 13.5, 14.7, 5.9, 2.0 # lengths in cm
    alpha = [0, np.pi/2, 0, 0, -np.pi/2, 0] # twist
    a = [0, 0, L2, L3, L4, 0] # link length
    d = [L1, 0, 0, 0, 0, -L5] # link offset
    max_joint_angle = [np.deg2rad(135), np.deg2rad(85), np.deg2rad(95), np.deg2rad(10), np.deg2rad(90), 0]
    min_joint_angle = [np.deg2rad(-135), np.deg2rad(0), np.deg2rad(-10), np.deg2rad(-180), np.deg2rad(-90), 0]
    return alpha, a, d, max_joint_angle, min_joint_angle




