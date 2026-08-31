import numpy as np
import matplotlib.pyplot as plt

def rotate_2d(points, theta_rad):
    """
    Rotates 2D points (2 x N array) by theta_rad around the origin.
    """
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    return R @ points

def translate_2d(points, tx, ty):
    """
    Translates 2D points (2 x N array) by [tx, ty]^T.
    """
    t = np.array([[tx], [ty]])
    return points + t

# 1. Define triangular robot footprint (2 x 4 matrix for closed shape)
# last column is repeated to close the triangular shape
robot_shape = np.array([
    [ 1.0, -0.5, -0.5,  1.0],  # X coordinates
    [ 0.0, -0.5,  0.5,  0.0]   # Y coordinates
])

# 2. Apply transformations: Rotate 45 deg, then Translate by (3, 2)
theta = np.radians(45)
tx, ty = 3.0, 2.0

rotated_robot = rotate_2d(robot_shape, theta)
transformed_robot = translate_2d(rotated_robot, tx, ty)

# 3. Visualization
plt.figure(figsize=(7, 7))
plt.plot(robot_shape[0, :], robot_shape[1, :], 'k--', label='Original (Local Frame)')
plt.plot(rotated_robot[0, :], rotated_robot[1, :], 'g:', label='Rotated (45°)')
plt.plot(transformed_robot[0, :], transformed_robot[1, :], 'b-', linewidth=2, label='Rotated + Translated (3, 2)')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.axis('equal')
plt.xlim(-2, 5)
plt.ylim(-2, 5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Task 1: Pure 2D Rotation and Translation')
plt.legend()
plt.show()