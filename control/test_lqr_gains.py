import numpy as np
import scipy.linalg
from scipy.signal import cont2discrete

g, l, m, dt = 10.0, 1.0, 1.0, 0.05
A_c = np.array([[0, 1], [3 * g / (2 * l), 0]])
B_c = np.array([[0], [3 / (m * l**2)]])
sys_d = cont2discrete((A_c, B_c, np.eye(2), np.zeros((2,1))), dt, method='zoh')
A_d, B_d = sys_d[0], sys_d[1]

Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])

# Solve DARE using scipy instead of control-toolbox
P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)
K = np.linalg.inv(R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
print("Scipy K:", K)

import control
import scipy as sp
if not hasattr(sp, 'polyval'): sp.polyval = np.polyval
sys_d_ss = control.StateSpace(A_d, B_d, np.eye(2), np.zeros((2, 1)))
lqr_solver = control.LQR(sys_d_ss, Q, R, N=200)
K_toolbox = -lqr_solver.solve()
print("Control toolbox K:", K_toolbox)
