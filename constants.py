import numpy as np

# Prior parameters
eta = 0.01

# TODO: Get appropriate value for following
gamma = 0.1
sigma_b0 = 0.1
sigma_u = 0.1
sigma_m = 0.1
sigma_bu = 0.1
sigma_bm = 0.1
sigma_Ma = 0.1
c = 1
b = 1

# Number of aspects
A = 5

# Number of latent factors
K = 5

#Aspect Sigma
sigma_ua = 0.1
sigma_ma = 1.0

# Number of users and movies
U = 1000
M = 1000
I = 100

# Hidden variables

# Switching variable y
y = np.random.multinomial(1000,[1.0/I]*I,(U,M)) / 1000
#y = np.zeros((U, M, I))

# Topic variable z
z = np.random.multinomial(1000,[1.0/I]*I,(U,M)) / 1000
#z = np.zeros((U, M, I))

# Sentiment variable s
s = np.random.multinomial(1000,[1.0/I]*I,(U,M)) / 1000
#s = np.zeros((U, M, I))

# User
v_u = np.random.normal(0,sigma_u,(U, K))      # Latent factor vector
b_u = np.random.normal(0,sigma_bu,(U, 1))      # Common bias vector
theta_u = np.random.normal(0,sigma_ua,(U, A))  # Aspect specific vector

# Movie
v_m = np.random.normal(0,sigma_m,(M, K))      # Latent factor vector
b_m = np.random.normal(0,sigma_bm,(M, 1))      # Common bias vector
theta_m = np.random.normal(0,sigma_ma,(M, A))  # Aspect specific vector

# Common bias
b_o = np.random.normal(0,sigma_b0) 

# Scaling Matrix
M_a = np.random.normal(0,sigma_Ma,(A, K))

#Matrices N
Nums = np.zeros((U,M,2))
Numas = np.zeros((U,M,A,2))
Numa = np.zeros((U,M,A))

#epsilon
epsilon = 5

#Counter
counter = 1

#rating matrix
rating_matrix = np.zeros((U,M))


