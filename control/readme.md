# Pendulum-v1 LQR and Swing-up Control

This directory contains an implementation of a Linear-Quadratic Regulator (LQR) and an energy-based swing-up controller for the Gymnasium `Pendulum-v1` environment.

## The Pendulum-v1 Model

The `Pendulum-v1` environment is a classic control problem where a pendulum is attached to an unactuated joint. The goal is to apply torque on the free end to swing it into an upright position and keep it there.

- **State Space**: The state consists of the angle $\theta$ (in radians) and the angular velocity $\dot{\theta}$ (in radians/second). In the environment observations, this is represented as `[cos(theta), sin(theta), theta_dot]`.
- **Action Space**: The control input is the torque applied to the joint, which is a continuous value bounded by the environment limits.
- **Non-linear Dynamics**: The system is described by the non-linear differential equation:
  $$ \ddot{\theta} = \frac{3g}{2l} \sin(\theta) + \frac{3}{ml^2} u $$
  where $g$ is gravity, $m$ is mass, $l$ is length, and $u$ is the control input (torque).

## Control Methods

### 1. LQR (Linear-Quadratic Regulator) Control
The LQR controller is used to stabilize the pendulum when it is near the upright equilibrium point ($\theta \approx 0$).
- **Linearization**: The non-linear dynamics are linearized around the upright position, where $\sin(\theta) \approx \theta$.
- **Discretization**: The continuous-time linear system is discretized using zero-order hold (ZOH) to match the discrete time-steps of the simulation.
- **Controller Design**: The LQR cost function $J = \sum (x^T Q x + u^T R u)$ is minimized by solving the discrete-time Algebraic Riccati Equation (ARE) to find the optimal feedback gain matrix $K$. The control law is then $u = -Kx$, where $x$ is the state error.

### 2. Energy-Based Swing-up Control
Since LQR is a local controller, it cannot swing the pendulum up from the bottom position on its own. To address this, an energy-based swing-up controller is used when the pendulum is far from the upright target.
- The controller calculates the current energy $E$ of the pendulum relative to the upright position.
- If the energy is insufficient, it applies maximum torque in the direction of the angular velocity to "pump" energy into the system.
- Once the pendulum approaches the upright position (e.g., $|\theta| \le 0.3$ rad), the system switches smoothly to the LQR controller for final stabilization.

## Installation

Ensure you have Python 3.x installed. It is recommended to use the provided virtual environment `.venv/` or create your own. Install the required dependencies:

```bash
pip install gymnasium numpy scipy control matplotlib
```

## Usage

You can run the main simulation script directly:

```bash
python lqr_pendulum.py
```

### Command-line Arguments

- `--swingup`: Enables the energy-based swing-up control from a bottom position. If not provided, the simulation starts near the upright position and uses only LQR.
- `--sim_time <seconds>`: Specifies the duration of the simulation in seconds.

**Examples:**

1. **Local LQR Stabilization (Starts near upright)**:
   ```bash
   python lqr_pendulum.py --sim_time 5
   ```

2. **Swing-up and LQR Stabilization (Starts near bottom)**:
   ```bash
   python lqr_pendulum.py --swingup --sim_time 10
   ```

### Output

The script runs the simulation and generates a plot (`response_plots.png` or `response_plots_swingup.png`) showing:
- Pendulum Angle ($\theta$) vs. Time
- Angular Velocity ($\dot{\theta}$) vs. Time
- Control Input (Torque) vs. Time

## Testing

The project includes several test scripts to verify the controller's behavior:
- `test_lqr_convergence.py`: Checks if the LQR converges to the target.
- `test_lqr_gains.py`: Validates the calculated LQR gain matrix.
- `test_lqr_zero.py`: Tests the response starting exactly at zero.
- `test_swingup.py`: Verifies the swing-up logic.

You can run any test with `pytest` or directly via Python:
```bash
python test_lqr_convergence.py
```