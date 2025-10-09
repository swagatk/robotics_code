import matplotlib.pyplot as plt
import numpy as np

def draw_axes(x, y, z, ax, length=3):
    """
    Draws a 3D axes reference frame at the given point (x, y, z).

    Args:
        x, y, z: The coordinates of the origin of the axes.
        ax: The matplotlib axes object to draw on.
    """
    # Define the origin
    origin = np.array([x, y, z])

    # Define the unit vectors for each axis
    x_axis = np.array([length, 0, 0])
    y_axis = np.array([0, length, 0])
    z_axis = np.array([0, 0, length])

    # Define the direction for p_axis, normalize it, then scale by length
    p_direction = np.array([1, 1, 1])
    p_norm = p_direction / np.linalg.norm(p_direction)
    p_axis = p_norm * length

    # Draw the axes using quiver
    ax.quiver(*origin, *x_axis, color='red')
    ax.quiver(*origin, *y_axis, color='green')
    ax.quiver(*origin, *z_axis, color='blue')
    ax.quiver(*origin, *p_axis, color='magenta')

# Example usage:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw axes at (0, 0, 0)
draw_axes(0, 0, 0, ax)

# # Draw axes at (2, 2, 2)
# draw_axes(2, 2, 2, ax)

# Set limits for better visualization
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()