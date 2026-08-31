import numpy as np
import matplotlib.pyplot as plt

def get_transform_2d(x, y, theta_rad):
    """
    Constructs a 3x3 Homogeneous Transformation Matrix for 2D.
    """
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ])

# 1. Define robot in Homogeneous Coordinates (3 x 4 array)
robot_hom = np.array([
    [ 1.0, -0.5, -0.5,  1.0],  # X coordinates
    [ 0.0, -0.5,  0.5,  0.0],  # Y coordinates
    [ 1.0,  1.0,  1.0,  1.0]   # Homogeneous scale factor
])

# 2. Transformation parameters
tx, ty = 3.0, 2.0
theta = np.radians(45)

T_trans = get_transform_2d(tx, ty, 0.0)
T_rot = get_transform_2d(0.0, 0.0, theta)

# Case A: Translate, then Rotate (with respect to global origin)
# T_A = T_rot @ T_trans
T_A = T_rot @ T_trans
robot_case_A = T_A @ robot_hom

# Case B: Rotate, then Translate (Standard pose composition)
# T_B = T_trans @ T_rot
T_B = T_trans @ T_rot
robot_case_B = T_B @ robot_hom

# 3. Visualization
plt.figure(figsize=(8, 8))
plt.plot(robot_hom[0, :], robot_hom[1, :], 'k--', label='Local Frame (0, 0, 0°)')
plt.plot(robot_case_A[0, :], robot_case_A[1, :], 'r.-', label='Case A: T_rot @ T_trans')
plt.plot(robot_case_B[0, :], robot_case_B[1, :], 'b.-', linewidth=2, label='Case B: T_trans @ T_rot')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.axis('equal')
plt.xlim(-3, 5)
plt.ylim(-2, 5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Task 2: Order of Homogeneous Transformations')
plt.legend()
plt.show()