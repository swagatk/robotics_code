"""
Solving inverse kinematics for Dobot 4DOF robotic arm
"""

import numpy as np
from dobot_config import get_dobot_dh_params
from inverse_kinematics_dhp import inverse_kinematics_pi, check_joint_limit
from fwd_kinematics_dph import forward_kinematics, plot_robot
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # desired end-effector position
    xd = np.array([10.0, 20.0, 30.0])  # desired position in cm
    alpha, a, d, max_joint_angle, min_joint_angle = get_dobot_dh_params()

    q, qnorm_list = inverse_kinematics_pi(xd, (alpha, a, d)) # compute inverse kinematics

    # check joint limits
    q = check_joint_limit(q, max_joint_angle, min_joint_angle)

    q_deg = np.rad2deg(q)
    print("Computed joint angles (degrees):\n", q_deg)

    # compute forward kinematics to verify
    xhat = forward_kinematics(alpha, a, d, q)  

    print("Desired end-effector position:\n", xd)
    print("Computed end-effector position:\n", xhat[-1])

    plot_robot(xhat, jtp=True, lw=4, labels=True)
    plt.title("Dobot Robot Inverse Kinematics Solution")
    plt.show()