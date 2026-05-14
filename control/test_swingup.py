import gymnasium as gym
import numpy as np

env = gym.make('Pendulum-v1')
g = env.unwrapped.g
m = env.unwrapped.m
l = env.unwrapped.l
dt = env.unwrapped.dt

obs, info = env.reset()
env.unwrapped.state = np.array([np.pi, 0.0])
obs = np.array([np.cos(np.pi), np.sin(np.pi), 0.0])

for i in range(400):
    cos_th, sin_th, th_dot = obs
    theta = np.arctan2(sin_th, cos_th)

    if abs(theta) < 0.3:
        print(f"Reached LQR region at step {i} with theta={theta:.2f}, th_dot={th_dot:.2f}")
        break

    # Mechanical energy (upright = 0)
    # Actually, E_mech = 0.5 * J * th_dot**2 + m * g * l * (cos(th) - 1). Wait, J = m * l**2 / 3.
    # Pendulum-v1 dynamics: thddot = 3g/(2l) * sin(th) + 3/(ml^2) u
    # Multiply by ml^2/3: (ml^2/3) thddot = mgl/2 * sin(th) + u
    # Equivalent to I thddot = mgl/2 sin(th) + u with I = ml^2/3, effective mass = m/2
    # So E = 0.5 * (ml^2/3) th_dot^2 + (mgl/2) * (cos(th) - 1)
    # Let's just use the energy function: E = 0.5 * th_dot**2 + (3 * g / (2 * l)) * (np.cos(theta) - 1)
    E = 0.5 * th_dot**2 + (3 * g / (2 * l)) * (np.cos(theta) - 1)
    
    # Simple bang-bang energy controller
    # When E < 0, we want to pump energy, so u should be in the direction of th_dot
    # Since u adds directly to thddot (with positive sign), u*th_dot > 0 increases energy.
    if E < 0:
        if abs(th_dot) < 0.01:
            u_swing = 2.0
        else:
            u_swing = 2.0 * np.sign(th_dot)
    else:
        # E >= 0, we want to remove energy
        u_swing = -2.0 * np.sign(th_dot)

    u_clipped = np.clip([u_swing], env.action_space.low, env.action_space.high)
    obs, reward, terminated, truncated, info = env.step(u_clipped)
else:
    print("Failed to reach LQR region.")
