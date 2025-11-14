"""
Solving inverse kinematics for Dobot 4DOF robotic arm
"""

import numpy as np
from dobot_config import get_dobot_dh_params
from inverse_kinematics_dhp import inverse_kinematics_pi, check_joint_limit
from fwd_kinematics_dph import forward_kinematics, plot_robot
import matplotlib.pyplot as plt

show_workspace = False
square_trajectory = False
circular_trajectory = True

if __name__ == "__main__":

    # get robot DH parameters
    alpha, a, d, max_joint_angle, min_joint_angle = get_dobot_dh_params()

    if show_workspace:
        # Visualize workspace
        # desired end-effector position
        xd = np.array([10.0, 20.0, 30.0])  # desired position in cm
        
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

    if square_trajectory:
        ## Inverse kinematics of a square
        X = np.array([[15, -5, 10], [15, 10, 10],
            [30, 10, 10], [30, -5, 10], [15, -5, 10]])
        robpos = []
        for i in range(len(X)):
            eepos = X[i]
            # compute inverse kinematics
            theta, _ = inverse_kinematics_pi(eepos, (alpha, a, d))
            # check joint limits
            #theta = check_joint_limit(theta, max_joint_angle, min_joint_angle)
            # compute forward kinematics to verify
            Y = forward_kinematics(alpha, a, d, theta)
            robpos.append(Y)
            print(Y[5])

        # visualize robot motion
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c='r', s=20)
        ax.plot3D(X[:,0], X[:,1], X[:,2], 'b-', lw=2)
        ax.set_title("Dobot Robot Square Trajectory")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        for i in range(len(robpos)):
            plot_robot(robpos[i], ax=ax, lw=2, jtp=False)
        plt.show()

    if circular_trajectory:
        # inverse kinematics of a circular trajectory
        r = 10 # radius of the circle
        theta = np.linspace(0, np.deg2rad(360), 10)
        x = 20 + r*np.cos(theta)
        y = r*np.sin(theta)
        z = 15 * np.ones_like(theta)
        robpos = []
        for i in range(len(x)):
            eepos = np.array([x[i], y[i], z[i]])
            angles, _ = inverse_kinematics_pi(eepos, (alpha, a, d))
            #angles = check_joint_limit(angles, max_joint_angle, min_joint_angle
            X = forward_kinematics(alpha, a, d, angles)
            robpos.append(X)

        # visualize robot motion
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c='r', s=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(x, y, z, c='r', s=10)
        ax.plot3D(x, y, z, 'b-', lw=2)
        for i in range(len(robpos)):
            ax = plot_robot(robpos[i],  ax=ax, lw=2)
        ax.set_title("Dobot Robot Circular Trajectory")
        plt.show()