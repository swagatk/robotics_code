from fwd_kinematics_dph import forward_kinematics, plot_robot
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":

    L1 = 1.0  # Length of link 1
    L2 = 1.0  # Length of link 2
    L3 = 0.5  # Length of link 3

    # link lengths
    a = [0, L1, L2, L3]  # link length

    # link twist
    alpha = [0, 0, 0, 0]  # twist

    # link offsets
    d = [0, 0, 0, 0]  # link

    # joint angles
    th1 = np.deg2rad(30)
    th2 = np.deg2rad(30)
    th3 = np.deg2rad(30)
    theta = [th1, th2, th3, 0]  # last joint angle is 0 for prismatic joint

    # Compute forward kinematics
    X = forward_kinematics(alpha, a, d, theta)
    print("End effector position:\n", X)

    # Plot the robot
    ax = plot_robot(X, jtp=True, lw=4, labels=True)
    plt.title("3-DOF Planar Robot Forward Kinematics")
    plt.show()