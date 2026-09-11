import numpy as np# Step 1: Update the systemIn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse

# ==============================================================================
# USER TUNING & NOISE PARAMETERS (Adjust these to see EKF behavior!)
# ==============================================================================

# 1. INITIAL POSE UNCERTAINTY & DELIBERATE INITIAL OFFSET
# Introduce an intentional initial error to watch the EKF correct itself:
INITIAL_X_ERROR     = 5.5    # meters offset in X
INITIAL_Y_ERROR     = -5.2   # meters offset in Y
INITIAL_THETA_ERROR = np.radians(45.0)  # heading error in degrees

# Initial Belief Covariance P_0 (How unsure the robot thinks it is initially):
INIT_POS_SIGMA      = 1.5    # meters (std dev)
INIT_THETA_SIGMA    = np.radians(30.0) # radians (std dev)

# 2. ODOMETRY / PROCESS NOISE (Actuator & wheel slip uncertainty: Q matrix)
ODOM_TRANS_NOISE_STD = 0.08   # Linear velocity noise std dev (m/s)
ODOM_ROT_NOISE_STD   = np.radians(3.5) # Rotational velocity noise std dev (rad/s)

# 3. LANDMARK SENSOR NOISE (Range-Bearing sensor uncertainty: R matrix)
SENSOR_RANGE_STD     = 0.20   # Distance measurement noise std dev (meters)
SENSOR_BEARING_STD   = np.radians(2.0) # Angle measurement noise std dev (degrees)

# 4. SIMULATION SETTINGS
SIMULATION_STEPS     = 1000
SENSOR_MAX_RANGE     = 12.0   # Max landmark detection range (m)

# ==============================================================================
# Environment & Map Setup
# ==============================================================================
WALL_THICKNESS = 0.5
OBSTACLES = [
    # Perimeter Walls: [xmin, ymin, xmax, ymax]
    [-10.0, -10.0, 10.0, -9.5],   # Bottom
    [-10.0,  9.5,  10.0, 10.0],   # Top
    [-10.0, -10.0, -9.5, 10.0],   # Left
    [  9.5, -10.0, 10.0, 10.0],   # Right
    # Interior Obstacles
    [-7.0,  3.0, -3.0,  6.0],
    [ 3.0,  2.0,  7.0,  5.0],
    [-6.0, -6.0, -2.0, -3.0],
    [ 4.0, -5.0,  8.0, -2.0],
    [ 0.0, -1.0,  2.0,  1.0]
]

LANDMARKS = np.array([
    [-8.0,  7.8],
    [ 8.0,  7.8],
    [ 8.0, -7.8],
    [-8.0, -7.8]
])

def point_in_box(px, py, box, margin=0.0):
    return (box[0] - margin <= px <= box[2] + margin) and \
           (box[1] - margin <= py <= box[3] + margin)

def is_collision(px, py, margin=0.55):
    for box in OBSTACLES:
        if point_in_box(px, py, box, margin):
            return True
    return False

def cast_ray(x, y, angle, max_range=5.0, step=0.08):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    r = 0.1
    while r < max_range:
        px = x + r * cos_a
        py = y + r * sin_a
        for box in OBSTACLES:
            if point_in_box(px, py, box, margin=0.05):
                return r, px, py, True
        r += step
    return max_range, x + max_range * cos_a, y + max_range * sin_a, False

# ==============================================================================
# Reactive Navigator
# ==============================================================================
class ReactiveNavigator:
    def __init__(self):
        self.state = "NAVIGATING"
        self.recovery_ticks = 0
        self.max_turn = 0.5
        self.preferred_turn = self.max_turn
     

    def compute_control(self, x, y, theta):
        angles = np.linspace(-np.pi/2, np.pi/2, 11)
        distances = [cast_ray(x, y, theta + a, max_range=3.0)[0] for a in angles]
        
        d_right  = np.min(distances[0:4])
        d_center = np.min(distances[4:7])
        d_left   = np.min(distances[7:11])

        CRITICAL_DIST = 0.75
        WARN_DIST     = 1.60

        if self.state == "BACKUP_TURN":
            self.recovery_ticks -= 1
            if self.recovery_ticks <= 0:
                self.state = "NAVIGATING"
            return -0.2, self.preferred_turn * 1.6

        if d_center < CRITICAL_DIST or min(d_left, d_right) < 0.65:
            self.state = "BACKUP_TURN"
            self.recovery_ticks = 15
            self.preferred_turn = self.max_turn if d_left > d_right else -self.max_turn
            return -0.2, self.preferred_turn * 1.6

        v = 0.6
        if d_center < WARN_DIST or d_left < WARN_DIST or d_right < WARN_DIST:
            v = 0.25 * (d_center / WARN_DIST)
            w = self.max_turn if d_left > d_right else -self.max_turn
        else:
            w = 0.15 * self.max_turn

        return max(v, 0.1), w

# ==============================================================================
# EKF with User-Configured Noise Covariances
# ==============================================================================
class RobotEKF:
    def __init__(self, x0, y0, theta0):
        self.mu = np.array([x0, y0, theta0], dtype=float)
        # Initial error covariance P
        self.P = np.diag([INIT_POS_SIGMA**2, INIT_POS_SIGMA**2, INIT_THETA_SIGMA**2])
        # Process noise covariance Q
        self.Q = np.diag([ODOM_TRANS_NOISE_STD**2, ODOM_TRANS_NOISE_STD**2, ODOM_ROT_NOISE_STD**2])
        # Measurement noise covariance R
        self.R = np.diag([SENSOR_RANGE_STD**2, SENSOR_BEARING_STD**2])

    def predict(self, v, w, dt):
        # next robot pose
        th = self.mu[2]
        self.mu[0] += v * np.cos(th) * dt
        self.mu[1] += v * np.sin(th) * dt
        self.mu[2] = (self.mu[2] + w * dt + np.pi) % (2 * np.pi) - np.pi

        # Jacobian of the motion model with respect to the state
        Fx = np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0,  v * np.cos(th) * dt],
            [0.0, 0.0,  1.0]
        ])
        # Update the error covariance based on the motion model
        self.P = Fx @ self.P @ Fx.T + self.Q

    def update(self, z_measurements):
        # Update the state and covariance based on the measurements
        for z, lm in z_measurements:
            dx = lm[0] - self.mu[0]
            dy = lm[1] - self.mu[1]
            q = dx**2 + dy**2
            d = np.sqrt(q)
            # Expected measurement based on the current state estimate
            expected_bearing = (np.arctan2(dy, dx) - self.mu[2] + np.pi) % (2*np.pi) - np.pi
            z_hat = np.array([d, expected_bearing])
            # Innovation (measurement residual)
            y = z - z_hat
            y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
            # Jacobian of the measurement model with respect to the state
            H = np.array([
                [-dx / d, -dy / d,  0.0],
                [ dy / q, -dx / q, -1.0]
            ])
            # Innovation covariance
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            # Kalman gain
            # Update the state estimate and covariance based on the measurement
            self.mu = self.mu + K @ y
            self.mu[2] = (self.mu[2] + np.pi) % (2 * np.pi) - np.pi
            self.P = (np.eye(3) - K @ H) @ self.P

# ==============================================================================
# Occupancy Grid
# ==============================================================================
class OccupancyGrid2D:
    def __init__(self, size=20.0, res=0.2):
        self.res = res
        self.size = size
        self.dim = int(size / res)
        self.grid = np.ones((self.dim, self.dim)) * 0.5

    def to_grid(self, x, y):
        gx = int((x + self.size/2) / self.res)
        gy = int((y + self.size/2) / self.res)
        return gx, gy

    def update_ray(self, x0, y0, x1, y1, hit):
        gx0, gy0 = self.to_grid(x0, y0)
        gx1, gy1 = self.to_grid(x1, y1)
        # Compute the discrete points along the ray using Bresenham's line algorithm approximation
        steps = int(max(abs(gx1 - gx0), abs(gy1 - gy0), 1))
        xs = np.linspace(gx0, gx1, steps + 1, dtype=int)
        ys = np.linspace(gy0, gy1, steps + 1, dtype=int)

        # Determine which grid cells are valid (inside the grid boundaries)
        valid = (xs >= 0) & (xs < self.dim) & (ys >= 0) & (ys < self.dim)
        self.grid[ys[valid][:-1], xs[valid][:-1]] = np.minimum(self.grid[ys[valid][:-1], xs[valid][:-1]] + 0.1, 0.95)

        # Update the occupancy grid based on the ray tracing result
        if hit and valid[-1]:
            self.grid[ys[valid][-1], xs[valid][-1]] = 0.0

# ==============================================================================
# Simulation Execution & Live Error Tracking
# ==============================================================================
def run_simulation():
    dt = 0.1
    true_x, true_y, true_theta = -4.0, -1.0, np.radians(45)

    # Initialize EKF with deliberate user-defined offsets
    ekf_x0 = true_x + INITIAL_X_ERROR
    ekf_y0 = true_y + INITIAL_Y_ERROR
    ekf_th0 = true_theta + INITIAL_THETA_ERROR
    ekf = RobotEKF(ekf_x0, ekf_y0, ekf_th0)

    grid = OccupancyGrid2D()
    nav = ReactiveNavigator()

    true_path_x, true_path_y = [true_x], [true_y]
    ekf_path_x,  ekf_path_y  = [ekf.mu[0]], [ekf.mu[1]]

    # Error tracking lists
    time_steps = []
    pos_errors = []
    sigma_2_bounds = []

    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])  # Spans entire bottom for error convergence plot

    for step in range(SIMULATION_STEPS):
        # 1. Navigation control
        v_cmd, w_cmd = nav.compute_control(true_x, true_y, true_theta)

        # 2. Collision guarded true motion
        next_x = true_x + v_cmd * np.cos(true_theta) * dt
        next_y = true_y + v_cmd * np.sin(true_theta) * dt
        next_th = (true_theta + w_cmd * dt + np.pi) % (2 * np.pi) - np.pi

        if not is_collision(next_x, next_y, margin=0.45):
            true_x, true_y, true_theta = next_x, next_y, next_th
        else:
            nav.state = "BACKUP_TURN"
            nav.recovery_ticks = 10
            true_theta = (true_theta + np.sign(w_cmd or 1.0) * 1.5 * dt + np.pi) % (2 * np.pi) - np.pi

        # 3. EKF Predict with applied odometry noise
        noisy_v = v_cmd + np.random.normal(0, ODOM_TRANS_NOISE_STD)
        noisy_w = w_cmd + np.random.normal(0, ODOM_ROT_NOISE_STD)
        ekf.predict(noisy_v, noisy_w, dt)

        # 4. Landmark updates with applied sensor noise
        measurements = []
        for lm in LANDMARKS:
            dist_true = np.hypot(lm[0] - true_x, lm[1] - true_y)
            bearing_true = (np.arctan2(lm[1] - true_y, lm[0] - true_x) - true_theta + np.pi) % (2*np.pi) - np.pi
            if dist_true < SENSOR_MAX_RANGE:
                z = np.array([
                    dist_true + np.random.normal(0, SENSOR_RANGE_STD),
                    bearing_true + np.random.normal(0, SENSOR_BEARING_STD)
                ])
                measurements.append((z, lm))
        ekf.update(measurements)

        # 5. Ray casting and grid mapping
        scan_angles = np.linspace(-np.pi/2, np.pi/2, 19)
        beams = []
        for sa in scan_angles:
            r, hx, hy, hit = cast_ray(true_x, true_y, true_theta + sa, max_range=4.5)
            beams.append((hx, hy))
            map_ray_ang = ekf.mu[2] + sa
            map_hx = ekf.mu[0] + r * np.cos(map_ray_ang)
            map_hy = ekf.mu[1] + r * np.sin(map_ray_ang)
            grid.update_ray(ekf.mu[0], ekf.mu[1], map_hx, map_hy, hit)

        # Log errors
        true_path_x.append(true_x)
        true_path_y.append(true_y)
        ekf_path_x.append(ekf.mu[0])
        ekf_path_y.append(ekf.mu[1])

        curr_error = np.hypot(ekf.mu[0] - true_x, ekf.mu[1] - true_y)
        pos_cov = ekf.P[0:2, 0:2]
        eigvals = np.linalg.eigvals(pos_cov)
        two_sigma = 2.0 * np.sqrt(np.max(eigvals))

        time_steps.append(step * dt)
        pos_errors.append(curr_error)
        sigma_2_bounds.append(two_sigma)

        # 6. Render
        if step % 3 == 0 or step == SIMULATION_STEPS - 1:
            ax1.cla()
            ax2.cla()
            ax3.cla()

            # --- Panel 1: Physical Ground Truth ---
            ax1.set_title("Ground Truth & EKF Tracking (with 2σ Error Ellipse)", fontsize=10, fontweight='bold')
            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-10, 10)
            ax1.grid(True, linestyle=':')

            for box in OBSTACLES:
                ax1.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], color='black'))
            ax1.plot(LANDMARKS[:, 0], LANDMARKS[:, 1], 'y*', markersize=12, markeredgecolor='black', label='Landmark')

            for bx, by in beams:
                ax1.plot([true_x, bx], [true_y, by], color='cyan', alpha=0.35, linewidth=0.8)

            robot_poly = np.array([[0.6, 0.0], [-0.4, 0.35], [-0.4, -0.35]])
            R = np.array([[np.cos(true_theta), -np.sin(true_theta)], [np.sin(true_theta), np.cos(true_theta)]])
            poly_world = (R @ robot_poly.T).T + np.array([true_x, true_y])
            ax1.add_patch(Polygon(poly_world, facecolor='blue', edgecolor='darkblue', zorder=5, label='True Robot'))
            ax1.plot([true_x, true_x + 0.8*np.cos(true_theta)], [true_y, true_y + 0.8*np.sin(true_theta)], 'r-', lw=2)

            ax1.plot(true_path_x, true_path_y, 'b-', label='True Path')
            ax1.plot(ekf_path_x, ekf_path_y, 'g--', label='EKF Estimated Path')

            # Draw EKF 2σ uncertainty ellipse
            vals, vecs = np.linalg.eigh(pos_cov)
            w_ell, h_ell = 2 * 2 * np.sqrt(np.maximum(vals, 1e-6))
            ell_angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ellipse = Ellipse((ekf.mu[0], ekf.mu[1]), w_ell, h_ell, angle=ell_angle,
                              edgecolor='red', facecolor='none', lw=2, label='2σ Uncertainty')
            ax1.add_patch(ellipse)
            ax1.legend(loc='lower left', fontsize=7)

            # --- Panel 2: Occupancy Grid Map ---
            ax2.set_title("Built Occupancy Grid (Robot Belief)", fontsize=10, fontweight='bold')
            ax2.imshow(grid.grid, cmap='gray', vmin=0.0, vmax=1.0, origin='lower', extent=[-10, 10, -10, 10])

            R_ekf = np.array([[np.cos(ekf.mu[2]), -np.sin(ekf.mu[2])], [np.sin(ekf.mu[2]), np.cos(ekf.mu[2])]])
            poly_ekf = (R_ekf @ robot_poly.T).T + np.array([ekf.mu[0], ekf.mu[1]])
            ax2.add_patch(Polygon(poly_ekf, facecolor='green', edgecolor='darkgreen', zorder=5))
            ax2.plot([ekf.mu[0], ekf.mu[0] + 0.8*np.cos(ekf.mu[2])],
                     [ekf.mu[1], ekf.mu[1] + 0.8*np.sin(ekf.mu[2])], 'yellow', lw=2)

            # --- Panel 3: Live Convergence / Error Graph ---
            ax3.set_title("EKF Estimation Error & Uncertainty Convergence Over Time", fontsize=10, fontweight='bold')
            ax3.plot(time_steps, pos_errors, 'r-', lw=1.8, label='Actual Position Error ||μ - x_true|| (m)')
            ax3.plot(time_steps, sigma_2_bounds, 'k--', lw=1.5, label='EKF 2σ Bound (Theoretical Uncertainty)')
            ax3.set_xlabel("Time (seconds)")
            ax3.set_ylabel("Error / Uncertainty (m)")
            ax3.set_ylim(0, max(max(pos_errors + [2.0]), max(sigma_2_bounds + [2.0])) * 1.15)
            ax3.grid(True, linestyle=':')
            ax3.legend(loc='upper right', fontsize=8)

            plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()