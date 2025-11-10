"""
Forward kinematics using Denavit-Hartenberg parameters
It uses Craig's convention for DH parameters
"""

import numpy as np


def homogeneous_transformation_matrix(a, d, alpha, theta):  
  """
  Computes the 4x4 homogeneous transformation matrix from joint i-1 to joint i
  """
  T = np.asarray([
    [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
    [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
    [0, np.sin(alpha), np.cos(alpha), d],
    [0, 0, 0, 1]
  ])
  return T



def transformation_matrix(a, d, alpha, theta):
  """
  It follow Craig's convention for DH parameters
  Computes the 4x4 transformation matrix from joint i-1 to joint i
  input:
    theta: angle of joint i
    alpha: twist between joint i and i-1
    d: link offset
    a: link length
  output:
    4x4 transformation matrix
  """
  # rotation matrix
  R = np.asarray([[ np.cos(theta), -np.sin(theta), 0],
       [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha)],
       [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha)]])
  # translation vector
  P = np.asarray([[a, -np.sin(alpha) * d, np.cos(alpha) * d]])
  U = np.asarray([[0, 0, 0, 1]])
  # homogenous transformation matrix
  T = np.concatenate((np.concatenate((R, np.transpose(P)), axis=1), U), axis=0)
  return T


def forward_kinematics(alpha, a, d, theta):
  assert len(alpha) == len(a) == len(d) == len(theta), 'DH parameter vectors must have the same length'
  num_joints = len(theta) # number of joints
  
  base = np.array([0, 0, 0, 1]) # robot base coordinates
  jt_pos = np.zeros((num_joints+1, 3))
  T = np.identity(4)
  for i in range(1, num_joints+1): # start from index 1
    T_new = transformation_matrix(a[i-1], d[i-1], alpha[i-1], theta[i-1])
    T = np.dot(T, T_new)
    x = np.dot(T, np.transpose(base))
    jt_pos[i] = x[:-1]
  return jt_pos


## Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def plot_robot(X, view=(25, -45), ax=None, labels=False, jtp=False, lw=4):
  if ax is None:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
  ax.plot3D(X[:,0], X[:,1], X[:,2], lw=lw)
  if jtp:
    ax.scatter(X[:,0], X[:,1], X[:,2], c='r', s=20)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')

  # Change the view
  elev, azim = view
  ax.view_init(elev=elev, azim=azim)

  if labels:
    for i in range(len(X)):
      label = '$J_{}$'.format(i)
      ax.text(X[i][0], X[i][1], X[i][2], label, size=15)
  return ax

