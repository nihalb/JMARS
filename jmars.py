import numpy as np
import scipy as sp

# Prior parameters
eta = 0.01

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

def sample_multiple_indices(p):
    """
    """
    # TODO: Function to sample a joint distribution
    pass

def word_indices(vec):
    """
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

class GibbsSampler:
    """
    """
    def __init__(Y=5, Z=A, S=2):
        """
        """
        self.Y = Y
        self.Z = Z
        self.S = S
        self.M = M

    def _initialize(self, matrix):
        """
        """
        (self.n_reviews, self.vocab_size) = matrix.shape

        # Number of times y occurs
        self.cy = np.zeros(self.Y)
        self.c = 0
        # Number of times y occurs with w
        self.cyw = np.zeros((self.Y, self.vocab_size))
        # Number of times y occurs with s and w
        self.cysw = np.zeros((self.Y, self.S, self.vocab_size))
        # Number of times y occurs with s
        self.cys = np.zeros((self.Y, self.S))
        # Number of times y occurs with z and w
        self.cyzw = np.zeros((self.Y, self.Z, self.vocab_size))
        # Number of times y occurs with z
        self.cyz = np.zeros((self.Y, self.Z))
        # Number of times y occurs with m and w
        self.cymw = np.zeros((self.Y, self.M, self.vocab_size))
        # Number of times y occurs with m
        self.cym = np.zeros((self.Y, self.M))
        self.topics = {}

        for r in xrange(self.n_reviews):
            for i, w in enumerate(word_indices(matrix[r, :])):
                # Choose a random assignment of y, z, w
                (y, z, s) = (np.random.randint(self.Y), np.random.randint(self.Z), np.random.randint(self.S))
                # Assign new values
                self.cy[y] += 1
                self.c += 1
                self.cyw[y][w] += 1
                self.cy[y][w] += 1
                self.cysw[y][s][w] += 1
                self.cys[y][s] += 1
                self.cyzw[y][z][w] += 1
                self.cyz[y][z] += 1
                # TODO: Define m
                self.cymw[y][m][w] += 1
                self.cym[y][m] += 1
                self.topics[(r, i)] = (y, z, w)

    def _conditional_distribution(self, u, m, w):
        """
        """
        # TODO: Function to return the conditional distribution for (y, z, s)
        pass

    def run(self, matrix, max_iter=50):
        """
        """
        self._initialize(matrix)

        for it in xrange(max_iter):
            for r in xrange(self.n_reviews):
                for i, w in enumerate(word_indices(matrix[r, :])):
                    (y, z, s) = self.topics[(r, i)]
                    # Exclude current assignment
                    self.cy[y] -= 1
                    self.c -= 1
                    self.cyw[y][w] -= 1
                    self.cy[y][w] -= 1
                    self.cysw[y][s][w] -= 1
                    self.cys[y][s] -= 1
                    self.cyzw[y][z][w] -= 1
                    self.cyz[y][z] -= 1
                    # TODO: Define m
                    self.cymw[y][m][w] -= 1
                    self.cym[y][m] -= 1

                    # Get next distribution
                    # TODO: Define u
                    p_z = self._conditional_distribution(u, m, w)
                    (y, z, w) = sample_multiple_indices(p_z)

                    # Assign new values
                    self.cy[y] += 1
                    self.c += 1
                    self.cyw[y][w] += 1
                    self.cy[y][w] += 1
                    self.cysw[y][s][w] += 1
                    self.cys[y][s] += 1
                    self.cyzw[y][z][w] += 1
                    self.cyz[y][z] += 1
                    # TODO: Define m
                    self.cymw[y][m][w] += 1
                    self.cym[y][m] += 1
                    self.topics[(r, i)] = (y, z, w)

