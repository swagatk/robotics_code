import gymnasium as gym
import numpy as np
import scipy
if not hasattr(scipy, 'polyval'):
    scipy.polyval = np.polyval
import control
from scipy.signal import cont2discrete

env = gym.make('Pendulum-v1')
g = env.unwrapped.g
m = env.unwrapped.m
l = env.unwrapped.l
dt = env.unwrapped.dt

A_c = np.array([[0, 1], [3 * g / (2 * l), 0]])
B_c = np.array([[0], [3 / (m * l**2)]])

sys_d = cont2discrete((A_c, B_c, np.eye(2), np.zeros((2,1))), dt, method='zoh')
A_d, B_d = sys_d[0], sys_d[1]

Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])

sys_d_ss = control.StateSpace(A_d, B_d, np.eye(2), np.zeros((2, 1)))
lqr_solver = control.LQR(sys_d_ss, Q, R, N=200)
K = -lqr_solver.solve()
print("K matrix:", K)

obs, _ = env.reset()
env.unwrapped.state = np.array([0.0, 0.0])
obs = np.array([1.0, 0.0, 0.0])

for i in range(10):
    cos_th, sin_th, th_dot = obs
    theta = np.arctan2(sin_th, cos_th)
    current_state = np.array([theta, th_dot])
    u = -K @ current_state
    print(f"Step {i}: theta={theta}, th_dot={th_dot}, u={u}")
    u_clipped = np.clip(u, env.action_space.low, env.action_space.high)
    obs, reward, terminated, truncated, info = env.step(u_clipped)
