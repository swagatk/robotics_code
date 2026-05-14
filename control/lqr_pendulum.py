"""
This script implements an LQR (Linear-Quadratic Regulator) controller
for the OpenAI Gym "Pendulum-v1" environment.

The script performs the following steps:
1.  Sets up the "Pendulum-v1" environment.
2.  Linearizes the pendulum's non-linear dynamics around its upright
    equilibrium point.
3.  Discretizes the continuous-time linear system.
4.  Designs a discrete-time LQR controller using the 'control' library.
5.  Runs a simulation loop where the LQR controller is used to stabilize
    the pendulum.
6.  Plots the pendulum's angle (theta), angular velocity (theta_dot), and
    the control input over time.
"""
import argparse
import gymnasium as gym
import numpy as np

# Workaround for older 'control' package versions with SciPy >= 1.9.0
import scipy
if not hasattr(scipy, 'polyval'):
    scipy.polyval = np.polyval

import control
import matplotlib.pyplot as plt

def run_lqr_control(swingup=False, sim_time=None):
    """
    Main function to run the LQR control simulation for the Pendulum-v1 env.
    """
    # 1. Environment Setup
    # Using render_mode='human' will display the pendulum simulation
    # env = gym.make('Pendulum-v1', render_mode='human')
    env = gym.make('Pendulum-v1')

    # 2. System Linearization and Discretization
    # Get system parameters from the environment
    g = env.unwrapped.g
    m = env.unwrapped.m
    l = env.unwrapped.l
    dt = env.unwrapped.dt  # Timestep

    # The pendulum's dynamics are non-linear:
    # theta_ddot = (3*g / (2*l)) * sin(theta) + (3 / (m*l**2)) * u
    # We linearize these dynamics around the upright equilibrium point (theta=0).
    # At this point, sin(theta) ≈ theta.
    # The linearized state-space representation (x_dot = A*x + B*u) is:
    # x = [theta, theta_dot]
    A_c = np.array([[0, 1],
                    [3 * g / (2 * l), 0]])

    B_c = np.array([[0],
                    [3 / (m * l**2)]])

    # The controller will be digital, so we discretize the continuous system.
    # x[k+1] = A_d*x[k] + B_d*u[k]
    from scipy.signal import cont2discrete
    sys_d = cont2discrete((A_c, B_c, np.eye(2), np.zeros((2,1))), dt, method='zoh')
    A_d = sys_d[0]
    B_d = sys_d[1]

    # 3. LQR Controller Design
    # The LQR cost function is J = sum(x'Qx + u'Ru).
    # Q penalizes state deviation, R penalizes control effort.
    # We penalize angle deviation more than angular velocity.
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])

    # Solve the discrete-time Algebraic Riccati Equation to find the optimal
    # feedback gain matrix K.
    import scipy.linalg
    P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
    print(f"LQR Gain Matrix (K): {K}")

    # 4. Simulation Loop
    # Reset environment to a random initial state
    obs, info = env.reset()

    # Set the initial state based on whether swing-up control is active
    if swingup:
        initial_angle =  np.pi  # Start at the bottom
        initial_velocity = 0.0
    else:
        # LQR is a local controller. Start near the upright position (theta=0).
        initial_angle = 0.2  # radians (about 11.5 degrees)
        initial_velocity = 0.1
        
    env.unwrapped.state = np.array([initial_angle, initial_velocity])
    obs = np.array([np.cos(initial_angle), np.sin(initial_angle), initial_velocity])

    # Simulation parameters
    if sim_time is not None:
        if sim_time <= 0:
            raise ValueError('sim_time must be greater than 0 seconds')
        n_steps = max(1, int(np.ceil(sim_time / dt)))
    else:
        n_steps = 400 if not swingup else 600
    history = {
        'time': [],
        'theta': [],
        'theta_dot': [],
        'control_input': []
    }
    target_state = np.array([0.0, 0.0])  # Target: upright and still

    for i in range(n_steps):
        # Convert observation [cos(th), sin(th), th_dot] to state [th, th_dot]
        cos_th, sin_th, th_dot = obs
        theta = np.arctan2(sin_th, cos_th)
        current_state = np.array([theta, th_dot])

        if swingup and abs(theta) > 0.3:
            # Energy-based swing-up controller
            E = 0.5 * th_dot**2 + (3 * g / (2 * l)) * (np.cos(theta) - 1)
            if E < 0:
                u_swing = 2.0 if abs(th_dot) < 0.01 else 2.0 * np.sign(th_dot)
            else:
                u_swing = -2.0 * np.sign(th_dot)
            u = np.array([u_swing])
        else:
            # Calculate state error
            state_error = current_state - target_state
    
            # Calculate control input: u = -K * x_error
            # The action needs to be a 1-element array for env.step
            u = -K @ state_error

        # Clip the control input to the environment's action space limits
        u_clipped = np.clip(u, env.action_space.low, env.action_space.high)

        # Apply the action to the environment
        obs, reward, terminated, truncated, info = env.step(u_clipped)

        # Store data for plotting
        history['time'].append(i * dt)
        history['theta'].append(theta)
        history['theta_dot'].append(th_dot)
        history['control_input'].append(u_clipped[0])

        # Optional: Render the environment
        # env.render()

    env.close()

    # 5. Plotting Results
    plot_results(history, swingup)

def plot_results(history, swingup=False):
    """
    Plots the simulation results: angle, angular velocity, and control input.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Pendulum Angle (theta)
    axs[0].plot(history['time'], np.rad2deg(history['theta']), label='Theta (θ)', color='b')
    axs[0].axhline(0, color='r', linestyle='--', label='Target')
    axs[0].set_ylabel('Angle (degrees)')
    axs[0].set_title('Pendulum State and Control Input vs. Time')
    axs[0].legend(loc='upper right')

    # Plot 2: Pendulum Angular Velocity (theta_dot)
    axs[1].plot(history['time'], history['theta_dot'], label='Theta_dot (ὡ)', color='g')
    axs[1].axhline(0, color='r', linestyle='--', label='Target')
    axs[1].set_ylabel('Angular Velocity (rad/s)')
    axs[1].legend(loc='upper right')

    # Plot 3: Control Input (Torque)
    axs[2].plot(history['time'], history['control_input'], label='Control Input (u)', color='purple')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Torque (Nm)')
    axs[2].legend(loc='upper right')

    plt.tight_layout()
    filename = 'response_plots_swingup.png' if swingup else 'response_plots.png'
    plt.savefig(filename)
    print(f"Plots saved to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LQR and Swing-up Control for Pendulum-v1')
    parser.add_argument('--swingup', action='store_true', help='Enable swing-up control from bottom position')
    parser.add_argument('--sim_time', type=float, default=None, help='Simulation time in seconds (e.g., --sim_time 5)')
    args = parser.parse_args()
    run_lqr_control(swingup=args.swingup, sim_time=args.sim_time)
