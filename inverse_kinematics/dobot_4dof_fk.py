from fwd_kinematics_dph import forward_kinematics, plot_robot
import numpy as np 
import matplotlib.pyplot as plt
from dobot_config import get_dobot_dh_params


if __name__ == "__main__":
  
    # D-H parameters for Dobot
    alpha, a, d, max_joint_angle, min_joint_angle = get_dobot_dh_params()

    # joint angles in radians
    th1 = np.deg2rad(0)
    th2 = np.deg2rad(0)
    th3 = np.deg2rad(0)
    th4 =  -(th2 + th3)
    th5 = np.deg2rad(0)
    th6 = np.deg2rad(0)

    # joint angle vector
    theta = [th1, th2, th3, th4, th5, th6]

    # Comput Robot joint positions
    X = forward_kinematics(alpha, a, d, theta)
    print("Joint positions:\n", X)

    ax = plot_robot(X, jtp=True, lw=4, labels=True)
    plt.title("Dobot Robot Forward Kinematics")
    

    #############
    # plotting robot workspace
    points = []
    for i in range(10000):
        th1 = np.random.uniform(np.deg2rad(-135), np.deg2rad(135))
        th2 = np.random.uniform(np.deg2rad(0), np.deg2rad(85))
        th3 = np.random.uniform(np.deg2rad(-10), np.deg2rad(95))
        th4 = -(th2 + th3)
        th5 = np.random.uniform(np.deg2rad(-90), np.deg2rad(90))
        th6 = 0

        theta = [th1, th2, th3, th4, th5, th6]
        X = forward_kinematics(alpha, a, d, theta)
        points.append(X[6])

    points = np.asarray(points)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', s=1)
    ax.plot3D(X[:,0], X[:,1], X[:,2], linewidth=4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("Dobot Robot Workspace")
    plt.show()