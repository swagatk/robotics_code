from dobot_config import get_dobot_dh_params
from inverse_kinematics_dhp import inverse_kinematics_pi, check_joint_limit, symbolic_fwd_kinematics, theta
from fwd_kinematics_dph import forward_kinematics, plot_robot
import matplotlib.pyplot as plt

import sympy as sym
from sympy import sin, cos, pprint, latex, init_printing
init_printing()

if __name__ == "__main__":
    # get robot DH parameters
    alpha, a, d, max_joint_angle, min_joint_angle = get_dobot_dh_params()

    x = symbolic_fwd_kinematics(theta, alpha, a, d)
    pprint(latex(x[0]))