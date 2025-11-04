import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math
from scipy.linalg import block_diag 

# --- Constants ---
DT = 0.1  # Time step
SIM_TIME = 50.0  # Total simulation time
STATE_SIZE = 3  # Robot state size [x, y, theta]
LANDMARK_SIZE = 2  # Landmark state size [x, y]

# Simulation noise (to apply to "truth" - must be non-zero)
# Reduced true noise for a more stable simulation
TRUE_CONTROL_NOISE = np.diag([0.02, np.deg2rad(1.0)])**2
TRUE_SENSOR_NOISE = np.diag([0.1, np.deg2rad(0.5)])**2

# EKF noise parameters
# "Honest" tuning. This is the most stable and robust setting.
R_CONTROL = TRUE_CONTROL_NOISE.copy()
Q_SENSOR = TRUE_SENSOR_NOISE.copy()


# Confidence level for plotting ellipses (95% -> 5.991)
ELLIPSE_CONFIDENCE = 5.991

# --- Ground Truth Landmarks ---
# Define the true positions of landmarks
LANDMARKS_TRUE = np.array([
    [5.0, 10.0],
    [10.0, 5.0],
    [15.0, 15.0],
    [-5.0, 15.0],
    [0.0, 5.0],
    [7.0, -2.0]
])

# --- Helper Functions ---

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

def plot_covariance_ellipse(ax, mu, sigma, color='r'):
    """
    Plot the 2D covariance ellipse for a given mean and covariance.
    """
    # Get 2x2 covariance for position
    cov = sigma[0:2, 0:2]
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Get ellipse angle and dimensions
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2.0 * np.sqrt(ELLIPSE_CONFIDENCE * eigenvalues[0])
    height = 2.0 * np.sqrt(ELLIPSE_CONFIDENCE * eigenvalues[1])
    
    ellipse = Ellipse(xy=mu[0:2], width=width, height=height, 
                      angle=angle, edgecolor=color, facecolor='none', 
                      linestyle='--')
    ax.add_patch(ellipse)


# --- EKF Core Functions ---

def motion_model(mu_robot, u):
    """
    Predict the robot's next state (x, y, theta) given the current state 
    and control input (v, omega). This is the non-linear function f(x, u).
    """
    x, y, theta = mu_robot
    v, omega = u
    
    if abs(omega) < 1e-5:  # Near straight-line motion
        x_new = x + v * DT * np.cos(theta)
        y_new = y + v * DT * np.sin(theta)
        theta_new = theta
    else:  # Circular motion
        radius = v / omega
        x_new = x - radius * np.sin(theta) + radius * np.sin(theta + omega * DT)
        y_new = y + radius * np.cos(theta) - radius * np.cos(theta + omega * DT)
        theta_new = theta + omega * DT
        
    return np.array([x_new, y_new, normalize_angle(theta_new)])

def get_motion_jacobian_G(mu_robot, u):
    """
    Calculate the Jacobian of the motion model w.r.t. the robot state.
    G = df/dx
    """
    x, y, theta = mu_robot
    v, omega = u
    
    G_robot = np.eye(STATE_SIZE) # Initialize as identity
    
    if abs(omega) < 1e-5:
        # Jacobian for straight-line motion
        G_robot[0, 2] = -v * DT * np.sin(theta)
        G_robot[1, 2] = v * DT * np.cos(theta)
    else:
        # Jacobian for circular motion
        radius = v / omega
        G_robot[0, 2] = -radius * np.cos(theta) + radius * np.cos(theta + omega * DT)
        G_robot[1, 2] = -radius * np.sin(theta) + radius * np.sin(theta + omega * DT)

    return G_robot

def get_motion_noise_jacobian_V(mu_robot, u):
    """
    Calculate the Jacobian of the motion model w.r.t. the control noise.
    V = df/dn (where n is noise on v, omega)
    """
    x, y, theta = mu_robot
    v, omega = u
    
    V = np.zeros((STATE_SIZE, 2)) # 3x2 matrix
    
    if abs(omega) < 1e-5:
        # Jacobian for straight-line motion
        V[0, 0] = DT * np.cos(theta)
        V[0, 1] = 0.0 # d(x)/d(omega) is complex, approx 0
        V[1, 0] = DT * np.sin(theta)
        V[1, 1] = 0.0 # d(y)/d(omega) is complex, approx 0
        V[2, 0] = 0.0
        V[2, 1] = 0.0 # d(theta_new)/d(omega) is 0
    else:
        # Jacobian for circular motion
        radius = v / omega
        V[0, 0] = (-np.sin(theta) + np.sin(theta + omega * DT)) / omega
        V[0, 1] = (v * (np.sin(theta) - np.sin(theta + omega * DT)) / omega**2) + \
                  (v * DT * np.cos(theta + omega * DT) / omega)
        V[1, 0] = (np.cos(theta) - np.cos(theta + omega * DT)) / omega
        V[1, 1] = (-v * (np.cos(theta) - np.cos(theta + omega * DT)) / omega**2) + \
                  (v * DT * np.sin(theta + omega * DT) / omega)
        V[2, 0] = 0.0
        V[2, 1] = DT
        
    return V

def ekf_prediction(mu, Sigma, u):
    """
    Perform the EKF prediction step.
    """
    N = len(mu)
    mu_robot = mu[0:STATE_SIZE]
    
    # Predict new robot state
    mu_robot_pred = motion_model(mu_robot, u)
    
    # Calculate Jacobians
    G_robot = get_motion_jacobian_G(mu_robot, u)
    V = get_motion_noise_jacobian_V(mu_robot, u)
    
    # Map control noise to state space
    R_state = V @ R_CONTROL @ V.T
    
    # Construct full G matrix
    G = np.eye(N)
    G[0:STATE_SIZE, 0:STATE_SIZE] = G_robot
    
    # Construct full noise matrix Q_prime
    Q_prime = np.zeros((N, N))
    # Q_prime[0:STATE_SIZE, 0:STATE_STATE] = R_state # OLD - TYPO
    Q_prime[0:STATE_SIZE, 0:STATE_SIZE] = R_state # NEW - FIXED
    
    # Predict state
    mu_pred = mu.copy()
    mu_pred[0:STATE_SIZE] = mu_robot_pred
    
    # Predict covariance
    Sigma_pred = G @ Sigma @ G.T + Q_prime
    
    return mu_pred, Sigma_pred

def get_measurement_model(mu_robot, mu_landmark):
    """
    Predict the sensor measurement (range, bearing) from the robot's
    believed state to the landmark's believed state. This is h(x).
    """
    rx, ry, rtheta = mu_robot
    lx, ly = mu_landmark
    
    delta_x = lx - rx
    delta_y = ly - ry
    
    z_range = np.sqrt(delta_x**2 + delta_y**2)
    z_bearing = np.arctan2(delta_y, delta_x) - rtheta
    
    return np.array([z_range, normalize_angle(z_bearing)])

def get_measurement_jacobian_H(mu, landmark_map_entry):
    """
    Calculate the Jacobian of the measurement model w.r.t. the full state.
    H = dh/dx
    """
    N = len(mu)
    robot_idx = (0, 1, 2)
    landmark_idx = (landmark_map_entry, landmark_map_entry + 1)
    
    rx, ry, rtheta = mu[0:STATE_SIZE]
    lx, ly = mu[landmark_map_entry : landmark_map_entry + LANDMARK_SIZE]
    
    delta_x = lx - rx
    delta_y = ly - ry
    q = delta_x**2 + delta_y**2
    sqrt_q = np.sqrt(q)
    
    # Jacobian w.r.t. robot state [rx, ry, rtheta]
    H_robot = np.zeros((LANDMARK_SIZE, STATE_SIZE))
    H_robot[0, 0] = -delta_x / sqrt_q
    H_robot[0, 1] = -delta_y / sqrt_q
    H_robot[0, 2] = 0.0
    H_robot[1, 0] = delta_y / q
    H_robot[1, 1] = -delta_x / q
    H_robot[1, 2] = -1.0
    
    # Jacobian w.r.t. landmark state [lx, ly]
    H_landmark = np.zeros((LANDMARK_SIZE, LANDMARK_SIZE))
    H_landmark[0, 0] = delta_x / sqrt_q
    H_landmark[0, 1] = delta_y / sqrt_q
    H_landmark[1, 0] = -delta_y / q
    H_landmark[1, 1] = delta_x / q
    
    # Construct full H matrix
    H = np.zeros((LANDMARK_SIZE, N))
    H[:, 0:STATE_SIZE] = H_robot
    H[:, landmark_map_entry : landmark_map_entry + LANDMARK_SIZE] = H_landmark
    
    return H

def augment_state(mu, Sigma, z_obs, landmark_id):
    """
    Add a new landmark to the state vector and covariance matrix.
    This is the "correct" way, handling cross-correlations.
    """
    N_old = len(mu)
    rx, ry, rtheta = mu[0:STATE_SIZE]
    r_obs, b_obs = z_obs
    
    # Calculate initial landmark position estimate
    lx_new = rx + r_obs * np.cos(b_obs + rtheta)
    ly_new = ry + r_obs * np.sin(b_obs + rtheta)
    
    mu_new_landmark = np.array([lx_new, ly_new])
    
    # Append to state vector
    mu_aug = np.concatenate((mu, mu_new_landmark))
    
    # --- Calculate Jacobians for Covariance Augmentation ---
    # G_xr = Jacobian of new landmark position w.r.t. robot state
    G_xr = np.zeros((LANDMARK_SIZE, STATE_SIZE)) # 2x3
    G_xr[0, 0] = 1.0
    G_xr[0, 1] = 0.0
    G_xr[0, 2] = -r_obs * np.sin(rtheta + b_obs)
    G_xr[1, 0] = 0.0
    G_xr[1, 1] = 1.0
    G_xr[1, 2] = r_obs * np.cos(rtheta + b_obs)
    
    # G_z = Jacobian of new landmark position w.r.t. measurement noise
    G_z = np.zeros((LANDMARK_SIZE, LANDMARK_SIZE)) # 2x2
    G_z[0, 0] = np.cos(rtheta + b_obs)
    G_z[0, 1] = -r_obs * np.sin(rtheta + b_obs)
    G_z[1, 0] = np.sin(rtheta + b_obs)
    G_z[1, 1] = r_obs * np.cos(rtheta + b_obs)

    # --- Augment Covariance Matrix Correctly ---
    N_new = N_old + LANDMARK_SIZE
    Sigma_aug = np.zeros((N_new, N_new))
    
    # Top-left block (old covariance)
    Sigma_aug[0:N_old, 0:N_old] = Sigma
    
    # Bottom-right block (new landmark's variance)
    Sigma_RR = Sigma[0:STATE_SIZE, 0:STATE_SIZE] # Robot covariance
    Sigma_NN = G_xr @ Sigma_RR @ G_xr.T + G_z @ Q_SENSOR @ G_z.T
    Sigma_aug[N_old:N_new, N_old:N_new] = Sigma_NN
    
    # Off-diagonal blocks (correlations)
    Sigma_xR = Sigma[0:N_old, 0:STATE_SIZE] # Old state correlation with Robot
    Sigma_xN = Sigma_xR @ G_xr.T
    Sigma_aug[0:N_old, N_old:N_new] = Sigma_xN # Top-right
    Sigma_aug[N_old:N_new, 0:N_old] = Sigma_xN.T # Bottom-left
    
    # Update landmark map
    # The new landmark's x-coordinate is at index N_old
    landmark_map[landmark_id] = N_old 
    
    print(f"--- Initialized Landmark {landmark_id} at index {N_old} ---")
    
    return mu_aug, Sigma_aug, landmark_map


# --- Simulation Functions ---

def get_control_input(time):
    """Generate a circular motion control input."""
    v = 1.0  # constant velocity
    omega = 0.5  # constant angular velocity
    return np.array([v, omega])

def simulate_true_motion(x_true, u):
    """
    Simulate the robot's *true* motion by adding noise
    to the control input *before* applying the motion model.
    """
    # Add noise to control input
    u_noisy = u + np.random.multivariate_normal([0, 0], TRUE_CONTROL_NOISE)
    
    # Apply motion model with noisy controls
    x_true_new = motion_model(x_true, u_noisy)
    return x_true_new

def simulate_sensor(x_true, landmark_map):
    """
    Simulate the robot's *true* sensor measurements by
    calculating perfect measurements and adding noise.
    """
    measurements = []
    landmark_ids = []
    
    rx, ry, rtheta = x_true
    
    for i, (lx, ly) in enumerate(LANDMARKS_TRUE):
        delta_x = lx - rx
        delta_y = ly - ry
        
        true_range = np.sqrt(delta_x**2 + delta_y**2)
        true_bearing = np.arctan2(delta_y, delta_x) - rtheta
        
        # Simulate sensor range limit
        if true_range <= 20.0: # UPDATED: Increased range from 10.0 to 20.0
            # Add sensor noise
            z_noisy = np.array([true_range, normalize_angle(true_bearing)]) + \
                      np.random.multivariate_normal([0, 0], TRUE_SENSOR_NOISE)
            
            # Only normalize the bearing component (index 1) after adding noise
            z_noisy[1] = normalize_angle(z_noisy[1])
            measurements.append(z_noisy)
            
            landmark_ids.append(i) # Use index as landmark ID
            
    return measurements, landmark_ids

# --- Main Simulation ---

if __name__ == "__main__":
    
    # --- Initialization ---
    time = 0.0
    
    # True state
    x_true = np.zeros(STATE_SIZE) # Robot starts at [0, 0, 0]
    
    # EKF state
    # "Known Start" - EKF starts at the true position
    mu = np.zeros(STATE_SIZE) 
    
    # EKF covariance
    # "Known Start" - EKF starts with high confidence (low uncertainty)
    Sigma = np.diag([0.01, 0.01, np.deg2rad(0.1)])**2
    
    # Landmark mapping: {landmark_id -> state_vector_index}
    landmark_map = {} 
    
    # History for plotting
    history_mu = [mu]
    history_x_true = [x_true]
    
    # History for error plots
    history_time = [time]
    pos_err = np.linalg.norm(x_true[0:2] - mu[0:2])
    history_robot_pos_err = [pos_err]
    history_landmark_err = [0.0] # Starts at 0, no landmarks
    
    # --- Matplotlib Setup ---
    plt.ion() # Interactive mode ON
    # Create a 3-plot layout
    fig, (ax_sim, ax_robot_err, ax_landmark_err) = plt.subplots(
        3, 1, 
        figsize=(10, 18), 
        gridspec_kw={'height_ratios': [3, 1, 1]}
    )
    fig.tight_layout(pad=3.0)
    
    # --- Simulation Loop ---
    while time <= SIM_TIME:
        print(f"Time: {time:.2f}s")
        
        # 1. Get control input
        u = get_control_input(time)
        
        # 2. Simulate true robot motion
        x_true = simulate_true_motion(x_true, u)
        
        # 3. Simulate true sensor measurements
        z_observations, z_landmark_ids = simulate_sensor(x_true, landmark_map)
        
        # 4. EKF Prediction
        mu_pred, Sigma_pred = ekf_prediction(mu, Sigma, u)
        
        # 5. EKF Correction (Handle all landmarks at once)
        
        # 5a. Augment state for *new* landmarks
        mu_aug = mu_pred.copy()
        Sigma_aug = Sigma_pred.copy()
        landmark_map_aug = landmark_map.copy() # Use a temporary map for this step
        
        for z_obs, landmark_id in zip(z_observations, z_landmark_ids):
            if landmark_id not in landmark_map_aug:
                # This is a new landmark, augment the *predicted* state
                mu_aug, Sigma_aug, landmark_map_aug = augment_state(
                    mu_aug, Sigma_aug, z_obs, landmark_id
                )
        
        # 5b. Stack all measurement Jacobians (H) and residuals (y)
        H_stack = []
        y_stack = []
        Q_stack_list = []
        
        for z_obs, landmark_id in zip(z_observations, z_landmark_ids):
            # Get landmark index in the *augmented* state vector
            landmark_state_index = landmark_map_aug[landmark_id]
            
            # Calculate expected measurement (z_hat)
            mu_robot = mu_aug[0:STATE_SIZE]
            mu_landmark = mu_aug[landmark_state_index : landmark_state_index + LANDMARK_SIZE]
            z_hat = get_measurement_model(mu_robot, mu_landmark)
            
            # Calculate measurement residual (y)
            y = z_obs - z_hat
            y[1] = normalize_angle(y[1]) # Normalize bearing residual
            
            # Calculate measurement Jacobian (H)
            H = get_measurement_jacobian_H(mu_aug, landmark_state_index)
            
            # Add to stack
            y_stack.append(y)
            H_stack.append(H)
            Q_stack_list.append(Q_SENSOR)

        # 5c. Perform one single "batch" EKF update
        if H_stack: # Only update if we have measurements
            H_all = np.vstack(H_stack)
            y_all = np.concatenate(y_stack)
            
            # Build block-diagonal Q_all matrix
            Q_all = block_diag(*Q_stack_list)
            
            # Calculate Kalman Gain (K)
            S = H_all @ Sigma_aug @ H_all.T + Q_all
            K = Sigma_aug @ H_all.T @ np.linalg.inv(S)
            
            # Update state (mu)
            mu_corr = mu_aug + K @ y_all
            
            # Update covariance (Sigma)
            I = np.eye(len(mu_aug))
            Sigma_corr = (I - K @ H_all) @ Sigma_aug
            
        else: # No measurements, just use prediction
            mu_corr = mu_aug
            Sigma_corr = Sigma_aug
            
        # --- EKF END ---
        
        # Update state for next iteration
        mu = mu_corr
        Sigma = Sigma_corr
        landmark_map = landmark_map_aug # Commit the updated map
        
        # Store history
        history_mu.append(mu)
        history_x_true.append(x_true)
        
        # NEW: Calculate and store errors
        history_time.append(time)
        
        # Robot position error
        pos_err = np.linalg.norm(x_true[0:2] - mu[0:2])
        history_robot_pos_err.append(pos_err)
        
        # Landmark error
        lm_err_sum = 0.0
        num_observed_landmarks = 0
        if landmark_map:
            for landmark_id, state_index in landmark_map.items():
                lm_true = LANDMARKS_TRUE[landmark_id]
                lm_est = mu[state_index : state_index + LANDMARK_SIZE]
                lm_err_sum += np.linalg.norm(lm_true - lm_est)
                num_observed_landmarks += 1
            
            avg_lm_err = (lm_err_sum / num_observed_landmarks) if num_observed_landmarks > 0 else 0
            history_landmark_err.append(avg_lm_err)
        else:
            history_landmark_err.append(0.0)
        
        
        # --- Plotting ---
        ax_sim.cla() # Clear main simulation axis
        
        # Plot landmarks
        ax_sim.plot(LANDMARKS_TRUE[:, 0], LANDMARKS_TRUE[:, 1], "bx", markersize=10, label="True Landmarks")
        
        # Plot paths
        path_true = np.array(history_x_true)
        path_ekf = np.array([m[0:STATE_SIZE] for m in history_mu])
        ax_sim.plot(path_true[:, 0], path_true[:, 1], "b-", label="True Path")
        ax_sim.plot(path_ekf[:, 0], path_ekf[:, 1], "r-", label="EKF Path")
        
        # Plot current robot poses
        ax_sim.plot(x_true[0], x_true[1], "bo", markersize=8, label="True Robot")
        ax_sim.plot(mu[0], mu[1], "ro", markersize=8, label="EKF Robot")
        
        # Plot robot uncertainty
        try:
            plot_covariance_ellipse(ax_sim, mu[0:2], Sigma[0:2, 0:2], color='r')
        except ValueError as e:
            print(f"Warning: Could not plot robot ellipse. {e}")
        
        # Plot landmark estimates and uncertainty
        plot_landmark_legend = True # Flag for legend
        for landmark_id, state_index in landmark_map.items():
            landmark_mu = mu[state_index : state_index + LANDMARK_SIZE]
            landmark_sigma = Sigma[state_index : state_index + LANDMARK_SIZE, 
                                   state_index : state_index + LANDMARK_SIZE]
            
            # UPDATED: Add label only once for the legend
            if plot_landmark_legend:
                ax_sim.plot(landmark_mu[0], landmark_mu[1], "r+", markersize=10, label="EKF Landmark")
                plot_landmark_legend = False
            else:
                ax_sim.plot(landmark_mu[0], landmark_mu[1], "r+", markersize=10)

            try:
                plot_covariance_ellipse(ax_sim, landmark_mu, landmark_sigma, color='r')
            except ValueError as e:
                print(f"Warning: Could not plot landmark {landmark_id} ellipse. {e}")

        ax_sim.set_title(f"EKF SLAM Simulation (Time: {time:.1f}s)")
        ax_sim.set_xlabel("X (meters)")
        ax_sim.set_ylabel("Y (meters)")
        ax_sim.legend()
        ax_sim.axis("equal")
        ax_sim.grid(True)
        
        # NEW: Plot Robot Position Error
        ax_robot_err.cla()
        ax_robot_err.plot(history_time, history_robot_pos_err, 'r-')
        ax_robot_err.set_title("Robot Position Error vs. Time")
        ax_robot_err.set_xlabel("Time (s)")
        ax_robot_err.set_ylabel("Position Error (m)")
        ax_robot_err.grid(True)
        
        # NEW: Plot Landmark Error
        ax_landmark_err.cla()
        ax_landmark_err.plot(history_time, history_landmark_err, 'b-')
        ax_landmark_err.set_title("Average Landmark Position Error vs. Time")
        ax_landmark_err.set_xlabel("Time (s)")
        ax_landmark_err.set_ylabel("Avg. Error (m)")
        ax_landmark_err.grid(True)
        
        
        plt.pause(0.01) # Pause for animation
        
        # Update time
        time += DT
    
    # End of simulation
    plt.ioff()
    ax_sim.set_title("EKF SLAM Simulation (Final)")
    fig.tight_layout(pad=3.0) # Adjust layout
    print("Simulation finished.")
    plt.show()


