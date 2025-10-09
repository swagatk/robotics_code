import numpy as np
import matplotlib.pyplot as plt

def rotate_x(x_vec, y_vec, z_vec, p_vec, angle_deg):
    """
    Rotates the given 3D vectors about the x-axis.

    Args:
        x_vec, y_vec, z_vec, p_vec: The 3D vectors representing the axes.
        angle_deg: The rotation angle in degrees.

    Returns:
        A tuple containing the rotated x, y, z, and p vectors.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])

    rotated_x_vec = np.dot(rotation_matrix_x, x_vec)
    rotated_y_vec = np.dot(rotation_matrix_x, y_vec)
    rotated_z_vec = np.dot(rotation_matrix_x, z_vec)
    rotated_p_vec = np.dot(rotation_matrix_x, p_vec)

    return rotated_x_vec, rotated_y_vec, rotated_z_vec, rotated_p_vec

def rotate_y(x_vec, y_vec, z_vec, p_vec, angle_deg):
    """
    Rotates the given 3D vectors about the y-axis.

    Args:
        x_vec, y_vec, z_vec, p_vec: The 3D vectors representing the axes.
        angle_deg: The rotation angle in degrees.

    Returns:
        A tuple containing the rotated x, y, z, and p vectors.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotation_matrix_y = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])

    rotated_x_vec = np.dot(rotation_matrix_y, x_vec)
    rotated_y_vec = np.dot(rotation_matrix_y, y_vec)
    rotated_z_vec = np.dot(rotation_matrix_y, z_vec)
    rotated_p_vec = np.dot(rotation_matrix_y, p_vec)

    return rotated_x_vec, rotated_y_vec, rotated_z_vec, rotated_p_vec

def rotate_z(x_vec, y_vec, z_vec, p_vec, angle_deg):
    """
    Rotates the given 3D vectors about the z-axis.

    Args:
        x_vec, y_vec, z_vec, p_vec: The 3D vectors representing the axes.
        angle_deg: The rotation angle in degrees.

    Returns:
        A tuple containing the rotated x, y, z, and p vectors.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    rotation_matrix_z = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])

    rotated_x_vec = np.dot(rotation_matrix_z, x_vec)
    rotated_y_vec = np.dot(rotation_matrix_z, y_vec)
    rotated_z_vec = np.dot(rotation_matrix_z, z_vec)
    rotated_p_vec = np.dot(rotation_matrix_z, p_vec)

    return rotated_x_vec, rotated_y_vec, rotated_z_vec, rotated_p_vec



def homogeneous_transform(vector, x_axis, y_axis, z_axis, p_axis, theta_deg, phi_deg, psi_deg, px, py, pz):
    """
    Transforms a 3D vector and its corresponding axes using a homogeneous transformation matrix.

    Args:
        vector: The 3D vector to transform (numpy array).
        x_axis, y_axis, z_axis, p_axis: The 3D vectors representing the axes (numpy arrays).
        theta_deg: Rotation angle about the x-axis in degrees.
        phi_deg: Rotation angle about the y-axis in degrees.
        psi_deg: Rotation angle about the z-axis in degrees.
        px, py, pz: Translation components along the x, y, and z axes.

    Returns:
        A tuple containing the transformed vector, and transformed x, y, z, and p axes vectors.
    """
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    psi_rad = np.deg2rad(psi_deg)

    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)

    # Construct the rotation matrix (R_z * R_y * R_x)
    rotation_matrix = np.array([
        [cos_psi * cos_phi, cos_psi * sin_phi * sin_theta - sin_psi * cos_theta, cos_psi * sin_phi * cos_theta + sin_psi * sin_theta],
        [sin_psi * cos_phi, sin_psi * sin_phi * sin_theta + cos_psi * cos_theta, sin_psi * sin_phi * cos_theta - cos_psi * sin_theta],
        [-sin_phi, cos_phi * sin_theta, cos_phi * cos_theta]
    ])

    # Construct the translation vector
    translation_vector = np.array([px, py, pz])

    # Construct the homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation_vector

    # Convert input vector and axes to homogeneous coordinates
    vector_homogeneous = np.append(vector, 1)
    x_axis_homogeneous = np.append(x_axis, 0) # Direction vector, so translation is 0
    y_axis_homogeneous = np.append(y_axis, 0) # Direction vector, so translation is 0
    z_axis_homogeneous = np.append(z_axis, 0) # Direction vector, so translation is 0
    p_axis_homogeneous = np.append(p_axis, 0) # Direction vector, so translation is 0


    # Transform the vector and axes
    transformed_vector_homogeneous = np.dot(homogeneous_matrix, vector_homogeneous)
    transformed_x_axis_homogeneous = np.dot(homogeneous_matrix, x_axis_homogeneous)
    transformed_y_axis_homogeneous = np.dot(homogeneous_matrix, y_axis_homogeneous)
    transformed_z_axis_homogeneous = np.dot(homogeneous_matrix, z_axis_homogeneous)
    transformed_p_axis_homogeneous = np.dot(homogeneous_matrix, p_axis_homogeneous)


    # Convert back to 3D coordinates
    transformed_vector = transformed_vector_homogeneous[:3]
    transformed_x_axis = transformed_x_axis_homogeneous[:3]
    transformed_y_axis = transformed_y_axis_homogeneous[:3]
    transformed_z_axis = transformed_z_axis_homogeneous[:3]
    transformed_p_axis = transformed_p_axis_homogeneous[:3]


    return transformed_vector, transformed_x_axis, transformed_y_axis, transformed_z_axis, transformed_p_axis



def draw_axes(x, y, z, ax, x_vec, y_vec, z_vec, p_vec, length=1):
    """
    Draws a 3D axes reference frame at the given point (x, y, z) with given vectors.

    Args:
        x, y, z: The coordinates of the origin of the axes.
        ax: The matplotlib axes object to draw on.
        x_vec, y_vec, z_vec, p_vec: The 3D vectors representing the axes.
        color: The color of the quiver arrows.
        length: The length of the quiver arrows.
    """
    origin = np.array([x, y, z])
    ax.quiver(*origin, *x_vec, color='red', length=length)
    ax.quiver(*origin, *y_vec, color='green', length=length)
    ax.quiver(*origin, *z_vec, color='blue', length=length)
    ax.quiver(*origin, *p_vec, color='magenta',length=length)


if __name__ == '__main__':

    # Example usage:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define initial axes vectors
    initial_vector = np.array([2, 2, 2])
    initial_x = np.array([1, 0, 0])
    initial_y = np.array([0, 1, 0])
    initial_z = np.array([0, 0, 1])
    initial_p = np.array([1, 1, 1]) 
    initial_p = initial_p / np.linalg.norm(initial_p)


    # Rotate the axes by 45 degrees about the x-axis
    rotated_x, rotated_y, rotated_z, rotated_p = rotate_x(initial_x, initial_y, initial_z, initial_p, 45)

    # Draw the initial axes at (0, 0, 0)
    draw_axes(0, 0, 0, ax, initial_x, initial_y, initial_z, initial_p)

    # # Draw the rotated axes at (0, 0, 0)
    # draw_axes(1, 1, 1, ax, rotated_x, rotated_y, rotated_z, rotated_p)

    # # Example of rotating about the y-axis (uncomment to use)
    # rotated_x_y, rotated_y_y, rotated_z_y, rotated_p_y = rotate_y(initial_x, initial_y, initial_z, initial_p, 30)
    # draw_axes(2, 2, 2, ax, rotated_x_y, rotated_y_y, rotated_z_y, rotated_p_y)

    # # Example of rotating about the z-axis (uncomment to use)
    # rotated_x_z, rotated_y_z, rotated_z_z, rotated_p_z = rotate_z(initial_x, initial_y, initial_z, initial_p, 60)
    # draw_axes(3, 3, 3, ax, rotated_x_z, rotated_y_z, rotated_z_z, rotated_p_z)


    
    # Define rotation angles and translation vector
    theta = 30  # Rotation about x-axis (degrees)
    phi = 45    # Rotation about y-axis (degrees)
    psi = 60    # Rotation about z-axis (degrees)
    px, py, pz = 2, 3, 4 # Translation vector

    # Perform homogeneous transformation
    transformed_vector, transformed_x, transformed_y, transformed_z, transformed_p = homogeneous_transform(
        initial_vector, initial_x, initial_y, initial_z, initial_p, theta, phi, psi, px, py, pz)
    
    # Draw the transformed axes at the translated origin (px, py, pz)
    draw_axes(px, py, pz, ax, transformed_x, transformed_y, transformed_z, transformed_p)

    # Plot the original and transformed vectors
    ax.quiver(0, 0, 0, *initial_vector, color='black', label='Original Vector')
    ax.quiver(0, 0, 0, *transformed_vector, color='orange', label='Transformed Vector')



    # Set limits for better visualization
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')


    plt.show()
