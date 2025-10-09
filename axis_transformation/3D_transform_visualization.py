import pygame
import numpy as np
import math

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
FONT_COLOR = (50, 50, 50)
AXIS_LENGTH = 150
BASE_AXIS_LENGTH = 40
LABEL_OFFSET = 15
BALL_RADIUS = 10
ROTATION_SENSITIVITY = 0.008
ROTATION_KEY_SENSITIVITY = 0.03
TRANSLATION_SENSITIVITY = 0.5
ZOOM_SENSITIVITY = 10

# --- 3D Projection Setup ---
# This matrix projects our 3D points onto the 2D screen.
# It's a simple perspective projection.
FOV = 90
ASPECT_RATIO = SCREEN_HEIGHT / SCREEN_WIDTH
NEAR = 0.1
FAR = 1000.0
f = 1 / math.tan(math.radians(FOV / 2))

PROJECTION_MATRIX = np.array([
    [ASPECT_RATIO * f, 0, 0, 0],
    [0, f, 0, 0],
    [0, 0, (FAR + NEAR) / (FAR - NEAR), 1],
    [0, 0, (-2 * FAR * NEAR) / (FAR - NEAR), 0]
])


def project_3d_to_2d(point_3d, transform_matrix):
    """
    Applies a transformation matrix to a 3D point and then projects it to 2D.
    Args:
        point_3d (np.ndarray): A 4x1 numpy array representing the point in homogeneous coordinates (x, y, z, 1).
        transform_matrix (np.ndarray): The 4x4 transformation matrix to apply.
    Returns:
        tuple or None: The (x, y) screen coordinates, or None if the point is behind the camera.
    """
    # Apply the rotation and translation
    point_transformed = transform_matrix @ point_3d

    # Project onto the 2D screen
    point_projected = PROJECTION_MATRIX @ point_transformed

    # Perspective division (normalizing)
    w = point_projected[3, 0]
    if w > 0: # Check if the point is in front of the camera
        point_normalized = point_projected / w
    else:
        return None # Avoid division by zero or rendering points behind camera

    # Convert to screen coordinates
    x = int((point_normalized[0, 0] + 1) * 0.5 * SCREEN_WIDTH)
    y = int((1 - (point_normalized[1, 0] + 1) * 0.5) * SCREEN_HEIGHT)

    return (x, y)


def draw_dotted_line(surface, color, start_pos, end_pos, dash_length=5):
    """Draws a dotted line between two points."""
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    if distance == 0:
        return
    dash_count = int(distance / dash_length)
    if dash_count == 0:
        return
    
    for i in range(0, dash_count, 2):
        start = (start_pos[0] + (dx * i / dash_count), start_pos[1] + (dy * i / dash_count))
        end = (start_pos[0] + (dx * (i + 1) / dash_count), start_pos[1] + (dy * (i + 1) / dash_count))
        pygame.draw.line(surface, color, start, end, 1)


def create_rotation_matrix(angle_x, angle_y, angle_z):
    """Creates a combined rotation matrix from Euler angles."""
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x), 0],
        [0, math.sin(angle_x), math.cos(angle_x), 0],
        [0, 0, 0, 1]
    ])
    rot_y = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y), 0],
        [0, 1, 0, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y), 0],
        [0, 0, 0, 1]
    ])
    rot_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0, 0],
        [math.sin(angle_z), math.cos(angle_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # Combine rotations: order matters (e.g., Y -> X -> Z)
    return rot_z @ rot_x @ rot_y


def create_translation_matrix(tx, ty, tz):
    """Creates a translation matrix."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


def main():
    """Main function to run the Pygame application."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive 3D Transformation Matrix")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 14, bold=True)
    axis_font = pygame.font.SysFont("arial", 16, bold=True)


    # --- Initial 3D points for the interactive object ---
    origin = np.array([[0], [0], [0], [1]])
    x_axis_end = np.array([[AXIS_LENGTH], [0], [0], [1]])
    y_axis_end = np.array([[0], [AXIS_LENGTH], [0], [1]])
    z_axis_end = np.array([[0], [0], [AXIS_LENGTH], [1]])
    ball_pos = np.array([[50], [50], [50], [1]])
    x_label = np.array([[AXIS_LENGTH + LABEL_OFFSET], [0], [0], [1]])
    y_label = np.array([[0], [AXIS_LENGTH + LABEL_OFFSET], [0], [1]])
    z_label = np.array([[0], [0], [AXIS_LENGTH + LABEL_OFFSET], [1]])
    
    interactive_points = [origin, x_axis_end, y_axis_end, z_axis_end, ball_pos, x_label, y_label, z_label]

    # --- Points for the fixed base frame at (0,0,0) ---
    base_origin = np.array([[0], [0], [0], [1]])
    base_x_axis = np.array([[BASE_AXIS_LENGTH], [0], [0], [1]])
    base_y_axis = np.array([[0], [BASE_AXIS_LENGTH], [0], [1]])
    base_z_axis = np.array([[0], [0], [BASE_AXIS_LENGTH], [1]])
    base_x_label = np.array([[BASE_AXIS_LENGTH + LABEL_OFFSET], [0], [0], [1]])
    base_y_label = np.array([[0], [BASE_AXIS_LENGTH + LABEL_OFFSET], [0], [1]])
    base_z_label = np.array([[0], [0], [BASE_AXIS_LENGTH + LABEL_OFFSET], [1]])
    base_points = [base_origin, base_x_axis, base_y_axis, base_z_axis, base_x_label, base_y_label, base_z_label]

    # --- Transformation state ---
    # Camera/World State (starts with a perspective view)
    world_angle_x = -math.radians(35)
    world_angle_y = -math.radians(40)
    world_angle_z = 0 # Camera doesn't roll in this setup
    camera_z = -600 # Start away from camera

    # Interactive Object State
    object_angle_x, object_angle_y, object_angle_z = 0, 0, 0
    object_tx, object_ty, object_tz = 0, 0, 0

    running = True
    mouse_dragging_rotate = False
    mouse_dragging_translate = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click for rotation
                    mouse_dragging_rotate = True
                elif event.button == 3:  # Right click for translation
                    mouse_dragging_translate = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_dragging_rotate = False
                elif event.button == 3:
                    mouse_dragging_translate = False
            elif event.type == pygame.MOUSEMOTION:
                keys = pygame.key.get_pressed()
                is_alt_pressed = keys[pygame.K_LALT] or keys[pygame.K_RALT]

                if mouse_dragging_rotate:
                    dx, dy = event.rel
                    if is_alt_pressed:
                        # Rotate Object
                        object_angle_y += dx * ROTATION_SENSITIVITY
                        object_angle_x -= dy * ROTATION_SENSITIVITY
                    else:
                        # Rotate Camera/World
                        world_angle_y += dx * ROTATION_SENSITIVITY
                        world_angle_x -= dy * ROTATION_SENSITIVITY

                if mouse_dragging_translate:
                    dx, dy = event.rel
                    is_shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                    is_ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]

                    if is_shift_pressed:
                        # Translate X/Z
                        object_tx += dx * TRANSLATION_SENSITIVITY
                        object_tz -= dy * TRANSLATION_SENSITIVITY
                    elif is_ctrl_pressed:
                        # Translate Y/Z
                        object_ty += dx * TRANSLATION_SENSITIVITY
                        object_tz -= dy * TRANSLATION_SENSITIVITY
                    else:
                        # Translate X/Y
                        object_tx += dx * TRANSLATION_SENSITIVITY
                        object_ty -= dy * TRANSLATION_SENSITIVITY
            elif event.type == pygame.MOUSEWHEEL:
                camera_z += event.y * ZOOM_SENSITIVITY
        
        # Handle continuous key presses for Object Z rotation (Roll)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            object_angle_z -= ROTATION_KEY_SENSITIVITY
        if keys[pygame.K_e]:
            object_angle_z += ROTATION_KEY_SENSITIVITY


        # --- Update ---
        # 1. Build Model Matrix for the interactive object
        object_rotation_matrix = create_rotation_matrix(object_angle_x, object_angle_y, object_angle_z)
        object_translation_matrix = create_translation_matrix(object_tx, object_ty, object_tz)
        model_matrix = object_translation_matrix @ object_rotation_matrix

        # 2. Build View Matrix for the camera
        view_rotation_matrix = create_rotation_matrix(world_angle_x, world_angle_y, world_angle_z)
        camera_zoom_matrix = create_translation_matrix(0, 0, camera_z)
        view_matrix = camera_zoom_matrix @ view_rotation_matrix
        
        # 3. Create final transformation matrices for rendering
        interactive_transform_matrix = view_matrix @ model_matrix
        base_frame_transform_matrix = view_matrix

        # --- Drawing ---
        screen.fill(WHITE)

        def draw_axis_label(text, pos, color):
            surface = axis_font.render(text, True, color)
            rect = surface.get_rect(center=pos)
            screen.blit(surface, rect)
        
        # Draw the fixed base frame at world origin (0,0,0)
        base_projected_points = []
        for point in base_points:
            projected = project_3d_to_2d(point, base_frame_transform_matrix)
            if projected:
                base_projected_points.append(projected)

        if len(base_projected_points) == 7:
            p_base_origin, p_base_x, p_base_y, p_base_z, p_base_lx, p_base_ly, p_base_lz = base_projected_points
            pygame.draw.line(screen, RED, p_base_origin, p_base_x, 2)
            pygame.draw.line(screen, GREEN, p_base_origin, p_base_y, 2)
            pygame.draw.line(screen, BLUE, p_base_origin, p_base_z, 2)
            draw_axis_label("X", p_base_lx, RED)
            draw_axis_label("Y", p_base_ly, GREEN)
            draw_axis_label("Z", p_base_lz, BLUE)


        # Draw the interactive, transformed frame
        projected_points = []
        for point in interactive_points:
            projected = project_3d_to_2d(point, interactive_transform_matrix)
            if projected:
                projected_points.append(projected)

        if len(projected_points) == 8:
            p_origin, p_x, p_y, p_z, p_ball, p_lx, p_ly, p_lz = projected_points

            # Draw Axes
            pygame.draw.line(screen, RED, p_origin, p_x, 3)
            pygame.draw.line(screen, GREEN, p_origin, p_y, 3)
            pygame.draw.line(screen, BLUE, p_origin, p_z, 3)

            # Draw Labels
            draw_axis_label("X", p_lx, RED)
            draw_axis_label("Y", p_ly, GREEN)
            draw_axis_label("Z", p_lz, BLUE)

            # Draw ball and line to origin
            draw_dotted_line(screen, GRAY, p_origin, p_ball)
            pygame.draw.circle(screen, RED, p_ball, BALL_RADIUS)
            pygame.draw.circle(screen, BLACK, p_ball, BALL_RADIUS, 1) # Outline

        # --- Display Information ---
        info_x = 10
        info_y = 10
        def draw_info_text(text, x, y, color=FONT_COLOR):
            surface = font.render(text, True, color)
            screen.blit(surface, (x, y))

        draw_info_text("CONTROLS:", info_x, info_y); info_y += 18
        draw_info_text("  - Left Click + Drag: Rotate Camera", info_x, info_y); info_y += 15
        draw_info_text("  - ALT + Left Drag:     Rotate Object (Pitch/Yaw)", info_x, info_y); info_y += 15
        draw_info_text("  - Right Click + Drag:  Translate Object (X/Y)", info_x, info_y); info_y += 15
        draw_info_text("  - SHIFT + Right Drag:  Translate Object (X/Z)", info_x, info_y); info_y += 15
        draw_info_text("  - CTRL + Right Drag:   Translate Object (Y/Z)", info_x, info_y); info_y += 15
        draw_info_text("  - Mouse Wheel:         Zoom Camera", info_x, info_y); info_y += 15
        draw_info_text("  - Q/E Keys:            Roll Object", info_x, info_y); info_y += 25
        
        # Display object state
        draw_info_text("OBJECT STATE (relative to base frame):", info_x, info_y); info_y += 18
        draw_info_text(f"  - Translation [X,Y,Z]: [{object_tx:6.1f}, {object_ty:6.1f}, {object_tz:6.1f}]", info_x, info_y); info_y += 18
        draw_info_text("  - ORIENTATION (Degrees):", info_x, info_y); info_y += 15
        draw_info_text(f"    - Pitch (X-axis): {math.degrees(object_angle_x):>8.2f}", info_x, info_y); info_y += 15
        draw_info_text(f"    - Yaw (Y-axis):   {math.degrees(object_angle_y):>8.2f}", info_x, info_y); info_y += 15
        draw_info_text(f"    - Roll (Z-axis):  {math.degrees(object_angle_z):>8.2f}", info_x, info_y); info_y += 25

        # Display camera state
        draw_info_text("CAMERA STATE:", info_x, info_y); info_y += 18
        draw_info_text("  - ORIENTATION (Degrees):", info_x, info_y); info_y += 15
        draw_info_text(f"    - Pitch (X-axis): {math.degrees(world_angle_x):>8.2f}", info_x, info_y); info_y += 15
        draw_info_text(f"    - Yaw (Y-axis):   {math.degrees(world_angle_y):>8.2f}", info_x, info_y); info_y += 15
        draw_info_text(f"    - Roll (Z-axis):  {math.degrees(world_angle_z):>8.2f}", info_x, info_y); info_y += 25

        draw_info_text("MODEL MATRIX (T * R):", info_x, info_y); info_y += 18
        
        for i, row in enumerate(model_matrix):
            row_str = "  [" + " ".join([f"{val:8.2f}" for val in row]) + "]"
            draw_info_text(row_str, info_x, info_y)
            info_y += 15

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()

