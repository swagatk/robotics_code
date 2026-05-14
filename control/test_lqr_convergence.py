import gymnasium as gym
import numpy as np
import scipy.linalg
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

P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)
K = np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d

obs, _ = env.reset()
env.unwrapped.state = np.array([0.2, 0.1])
obs = np.array([np.cos(0.2), np.sin(0.2), 0.1])

for i in range(200):
    cos_th, sin_th, th_dot = obs
    theta = np.arctan2(sin_th, cos_th)
    current_state = np.array([theta, th_dot])
    u = -K @ current_state
    u_clipped = np.clip(u, env.action_space.low, env.action_space.high)
    obs, reward, terminated, truncated, info = env.step(u_clipped)

print("Final theta:", theta)
print("Final th_dot:", th_dot)
print("Final u:", u_clipped[0])
