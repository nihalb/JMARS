# Prior parameters
eta = 0.01

# TODO: Get appropriate value for gamma
gamma = 0.1

# Number of aspects
A = 5

# Number of latent factors
K = 5

# Number of users and movies
U = 1#1000
M = 1#1000
I = 1#100

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

#Matrices N
Nums = np.zeros((U,M,2))
Numas = np.zeros((U,M,A,2))
Numa = np.zeros((U,M,A))

#epsilon
epsilon = 5

#Sentiment
c = 1
b = 1

#Counter
counter = 1

#rating matrix
rating_matrix = np.zeros((U,M))


