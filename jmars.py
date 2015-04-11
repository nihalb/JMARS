import numpy as np
import scipy as sp

# Number of aspects
A = 5

# Number of latent factors
K = 5

# Number of users and movies
U = 1000
M = 1000
I = 100

# Hidden variables

# Switching variable y
y = np.zeros((U, M, I))

# Topic variable z
z = np.zeros((U, M, I))

# Sentiment variable s
s = np.zeros((U, M, I))

# User
v_u = np.zeros((U, K))      # Latent factor vector
b_u = np.zeros((U, 1))      # Common bias vector
theta_u = np.zeros((U, A))  # Aspect specific vector

# Movie
v_m = np.zeros((M, K))      # Latent factor vector
b_m = np.zeros((M, 1))      # Common bias vector
theta_m = np.zeros((M, A))  # Aspect specific vector

# Common bias
b_o = 0

# Scaling Matrix
M_a = np.zeros((A, K))

# Joint aspect distribution
def joint_aspect(u, m):
    """
    """
    u_a = theta_u[u].T
    m_a = theta_m[m].T
    um_a = np.exp(np.add(u_a, m_a))
    theta_um = um_a / np.sum(um_a)
    return theta_um

def predicted_rating(u, m):
    """
    """
    theta_um = joint_aspect(u, m)
    temp = np.diag((np.dot(M_a.T, theta_um)).reshape(K))
    r = v_u[u].dot(temp).dot(v_m[m].T) + b_o + b_u[u] + b_m[m]
    return r.sum()

