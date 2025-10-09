import pygame
import numpy as np
import math

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (150, 150, 150)
AXIS_LENGTH = 100
POINT_RADIUS = 8
FONT_SIZE = 24

# --- Helper Functions ---

def get_transformation_matrix(angle_deg, tx, ty):
    """
    Creates a 3x3 homogeneous transformation matrix for 2D.
    Combines rotation and translation.
    """
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Create a 3x3 homogeneous transformation matrix
    # [ R | t ]
    # [ 0 | 1 ]
    # where R is the 2x2 rotation matrix and t is the 2x1 translation vector.
    return np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [0,      0,      1]
    ])

def world_to_screen(pos, width, height):
    """Converts world coordinates (origin at center) to screen coordinates (origin at top-left)."""
    # World origin is at the center of the screen
    screen_x = int(width / 2 + pos[0])
    # Y-axis is inverted in screen coordinates
    screen_y = int(height / 2 - pos[1])
    return (screen_x, screen_y)

def draw_frame(surface, matrix, color, width, height):
    """Draws a coordinate frame based on its transformation matrix."""
    # The origin of the frame is its translation part
    origin_world = matrix[:2, 2]
    origin_screen = world_to_screen(origin_world, width, height)

    # The direction of the x-axis is the first column of the rotation part
    x_axis_dir_world = matrix[:2, 0] * AXIS_LENGTH
    x_axis_end_world = origin_world + x_axis_dir_world
    x_axis_end_screen = world_to_screen(x_axis_end_world, width, height)

    # The direction of the y-axis is the second column of the rotation part
    y_axis_dir_world = matrix[:2, 1] * AXIS_LENGTH
    y_axis_end_world = origin_world + y_axis_dir_world
    y_axis_end_screen = world_to_screen(y_axis_end_world, width, height)

    # Draw axes
    pygame.draw.line(surface, color, origin_screen, x_axis_end_screen, 3)
    pygame.draw.line(surface, color, origin_screen, y_axis_end_screen, 3)
    
    # Add labels (X, Y)
    font = pygame.font.Font(None, FONT_SIZE)
    x_label = font.render('X', True, color)
    y_label = font.render('Y', True, color)
    surface.blit(x_label, (x_axis_end_screen[0] + 5, x_axis_end_screen[1]))
    surface.blit(y_label, (y_axis_end_screen[0] + 5, y_axis_end_screen[1]))

def main():
    """Main animation loop."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Transformation Matrix Animation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, FONT_SIZE)

    running = True
    angle = 0
    
    # --- Point P defined in Frame {B}'s coordinate system ---
    # We use homogeneous coordinates by adding a '1' at the end.
    p_in_b = np.array([50, 30, 1])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Animate the transformation ---
        angle += 0.5  # degrees per frame
        if angle >= 360:
            angle = 0
        
        # Animate translation in a circular path for more visual interest
        tx = 150 * math.cos(math.radians(angle * 0.5))
        ty = 100 * math.sin(math.radians(angle))

        # --- Core Logic ---
        # 1. Define the transformation of Frame {B} relative to Frame {A}
        T_B_in_A = get_transformation_matrix(angle, tx, ty)

        # 2. Transform the point P from Frame {B} to Frame {A}
        # This is the key calculation: P_A = T_B_A * P_B
        p_in_a = T_B_in_A @ p_in_b

        # --- Drawing ---
        screen.fill(WHITE)
        
        # Draw text info
        angle_text = font.render(f"Rotation (deg): {angle:.1f}", True, BLACK)
        trans_text = font.render(f"Translation (tx, ty): ({tx:.1f}, {ty:.1f})", True, BLACK)
        screen.blit(angle_text, (10, 10))
        screen.blit(trans_text, (10, 30))

        # Draw Frame {A} (the static reference frame at the origin)
        T_A_in_A = get_transformation_matrix(0, 0, 0) # Identity transformation
        draw_frame(screen, T_A_in_A, BLACK, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Draw Frame {B} (the moving frame)
        draw_frame(screen, T_B_in_A, BLUE, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Draw the point P in its final calculated position in Frame {A}
        p_screen_coords = world_to_screen(p_in_a, SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.draw.circle(screen, RED, p_screen_coords, POINT_RADIUS)
        
        # Draw a line from frame B's origin to point P to show their rigid connection
        origin_b_screen = world_to_screen(T_B_in_A[:2, 2], SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.draw.line(screen, GRAY, origin_b_screen, p_screen_coords, 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()