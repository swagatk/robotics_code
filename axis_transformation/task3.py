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

# 1. Arm Parameters
L1 = 2.0               # Length of Link 1
L2 = 1.5               # Length of Link 2
theta1 = np.radians(30) # Joint 1 angle (relative to Base Frame)
theta2 = np.radians(45) # Joint 2 angle (relative to Link 1 Frame)

# 2. Forward Kinematics via Matrix Composition
# Base to Frame 1 (Joint 1)
T0_1 = get_transform_2d(0, 0, theta1)

# Frame 1 to Frame 2 (Joint 2 / End-Effector Frame)
T1_2 = get_transform_2d(L1, 0, theta2)

# Global End-Effector Transform: T0_2 = T0_1 @ T1_2
T0_2 = T0_1 @ T1_2

# 3. Calculate Joint Locations in Base Frame
p_base = np.array([0, 0, 1])
p_joint1 = T0_1 @ np.array([L1, 0, 1])
p_end_effector = T0_2 @ np.array([L2, 0, 1])

print("--- Kinematics Results ---")
print(f"Base Position:         (x: {p_base[0]:.3f}, y: {p_base[1]:.3f})")
print(f"Joint 1 Position:      (x: {p_joint1[0]:.3f}, y: {p_joint1[1]:.3f})")
print(f"End-Effector Position: (x: {p_end_effector[0]:.3f}, y: {p_end_effector[1]:.3f})")

# 4. Visualization
plt.figure(figsize=(8, 6))

arm_x = [p_base[0], p_joint1[0], p_end_effector[0]]
arm_y = [p_base[1], p_joint1[1], p_end_effector[1]]

# Plot Arm Links and Joints
plt.plot(arm_x, arm_y, 'ro-', linewidth=4, markersize=10, label='Robot Arm Links')
plt.plot(p_base[0], p_base[1], 'ks', markersize=10, label='Base (Origin)')
plt.plot(p_end_effector[0], p_end_effector[1], 'g*', markersize=14, label='End-Effector')

plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True)
plt.axis('equal')
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Task 3: 2-Link Arm Forward Kinematics')
plt.legend()
plt.show()