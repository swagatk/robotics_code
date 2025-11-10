from numpy.core.multiarray import ndarray
from sympy.utilities.lambdify import lambdify
import numpy as np
import sys
import sympy as sym
from sympy import sin, cos, pprint, latex, init_printing

def deg2rad(deg):
  rad = deg * np.pi / 180.0
  return rad

def rad2deg(rad):
  deg = rad * 180.0 / np.pi
  return deg

def check_joint_limit(theta, theta_max, theta_min):
  '''
  limit joint angles to valid range.
  Input: angles in radians
  Output: valid angles in radians
  '''
  assert len(theta) == len(theta_max) == len(theta_min), 'theta, theta_max, and theta_min should have the same length'

  for i in range(len(theta)):
    if theta[i] > theta_max[i]:
      theta[i] = theta_max[i]

    if theta[i] < theta_min[i]:
      theta[i] = theta_min[i]
  return theta



init_printing()

theta1, theta2, theta3, theta4, theta5, theta6 = sym.symbols('theta1, theta2, theta3, theta4, theta5, theta6')
x, y, z = sym.symbols('x, y, z')
theta = [theta1, theta2, theta3, theta4, theta5, theta6]


def symbolic_transformation_matrix(a, d, alpha, theta):
    """
    It follow Craig's convention for DH parameters
    """
    # transformation matrix
    T = sym.Matrix(([ cos(theta), -sin(theta), 0, a],
        [sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -sin(alpha)*d],
        [sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), cos(alpha)*d],
            [0, 0, 0, 1]))
    return T

def symbolic_fwd_kinematics(theta, alpha, a, d):
    assert len(theta) == 6, 'angle vector must have 6 elements'
    num_joints = len(theta) # number of joints
    base = sym.Matrix((0, 0, 0, 1)) # robot base coordinates
    #print(base.shape)
    T = sym.eye(4)
    for i in range(1, num_joints+1): # start from index 1
        T_new = symbolic_transformation_matrix(a[i-1], d[i-1], alpha[i-1], theta[i-1])
        T = T * T_new    # matrix dot product

    # the end effector position [x y z 1]
    x = T * base    # matrix - vector dot product
    #print(type(x))
    return x[:-1]



def inverse_kinematics_pi(xd, dh_params:tuple, num_steps=100000, gain=10, stat=False):
    '''
    Inverse kinematics using pseudo-inverse of Jacobian matrix
    Input:
      xd: desired end effector position (3 x 1) numpy array
      dh_params: tuple containing DH parameters (alpha, a, d)
      num_steps: number of iteration steps
      gain: gain for the control law
      stat: if True, print iteration status
    Output:
      q: joint angles (6 x 1) numpy array
      qnorm_list: list of norm of joint angles at each iteration step
    '''
    
    if isinstance(xd, ndarray):
        if xd.ndim == 1:
            xd = np.array(xd)
            xd = np.expand_dims(xd, axis=1)
        elif xd.ndim > 2:
            raise ValueError('xd must have two dimensions')
    else:
        raise TypeError('xd must be in numpy array format')
    assert xd.shape == (3, 1), 'xd must be of shape (3, 1)'

    # Symbolic end effector position
    alpha, a, d = dh_params
    x = symbolic_fwd_kinematics(theta, alpha, a, d) # theta is a list of symbolic variables
    X = sym.Matrix(x)

    # Symbolic Jacobian matrix
    J = X.jacobian(theta)

    # Lambdify functions for numerical computation
    array2mat = [{'sympy.matrices.dense.MutableDenseMatrix': np.array}, 'numpy']
    end_effector_position = lambdify(theta, X, modules=array2mat)
    jacobian_matrix = lambdify(theta, J, modules=array2mat)

    # Initial joint angles
    q = [0, 0, 0, 0, 0, 0]
    dt = 0.0001
    K = gain
    qnorm_list = []
    for i in range(num_steps):
        x = end_effector_position(*q).astype(np.float64)
        e = xd - x
        J = jacobian_matrix(*q).astype(np.float64)
        Jinv = np.linalg.pinv(J)
        qdot = K *  np.dot(Jinv, e)
        qdot = qdot.squeeze()
        q_prev = q
        q = q + qdot * dt
        qnorm = np.linalg.norm(q)
        qnorm_list.append(qnorm)

        if np.allclose(q, q_prev):
            print('solution converges in {} steps.'.format(i))
            break
        if stat:
            if i % 1000 == 0:
                print('\riteration step: {}, qnorm: {}'.format(i, qnorm), end="")
                sys.stdout.flush()
        
    return q, qnorm_list
