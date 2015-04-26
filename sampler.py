from constants import *
import numpy as np

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

def predicted_aspect_rating(u, m, a):
    """
    """
    temp = np.diag(M_a[a])
    r = v_u[u].dot(temp).dot(v_m[m].T) + b_o + b_u[u] + b_m[m]
    return r.sum()

def aspect_sentiment_probability(s, u, m, a):
    """
    """
    ruma = predicted_aspect_rating(u,m,a)
    prob_suma = 1.0 / (1.0 + np.exp(-s*(c*ruma - b)))
    return prob_suma

def aggregate_sentiment_probability(s, u, m):
    """
    """
    rum = predicted_rating(u,m)
    prob_sum = 1.0 / (1.0 + np.exp(-s*(c*rum - b)))
    return prob_sum

def sample_multinomial(w):
    """
    """
    x = np.random.uniform(0,1)
    for i,v in enumerate(np.cumsum(w)):
        if x < v: return i
    return len(w)-1

def sample_multiple_indices(p):
    """
    """
    (Y, Z, S) = p.shape
    dist = list()
    for y in xrange(Y):
        for z in xrange(Z):
            for s in xrange(S):
                dist.append(p[y,z,s])
    index = sample_multinomial(dist)
    y = index / (Z * S)
    rem = index % (Z * S)
    z = rem / S
    s = rem % S
    return (y, z, s)

def word_indices(vec):
    """
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

class GibbsSampler:
    """
    """
    def __init__(self, Y, Z, S):
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
                self.cyw[y,w] += 1
                self.cy[y] += 1
                self.cysw[y,s,w] += 1
                self.cys[y,s] += 1
                self.cyzw[y,z,w] += 1
                self.cyz[y,z] += 1
                # TODO: Define m
                m = np.random.randint(self.M)
                self.cymw[y,m,w] += 1
                self.cym[y,m] += 1
                self.topics[(r, i)] = (y, z, s)

    def _conditional_distribution(self, u, m, w):
        """
        """
        p_z = np.zeros((self.Y, self.Z, self.S))
        # y = 0
        for z in xrange(self.Z):
            for s in xrange(self.S):
                p_z[0,z,s] = (self.cy[0] + gamma) / (self.c + 5 * gamma)
                p_z[0,z,s] = (p_z[0,z,s] * (self.cyw[0,w] + eta)) / (self.cy[0] + eta)

        # y = 1
        for z in xrange(self.Z):
            for s in xrange(self.S):
                p_z[1,z,s] = (self.cy[1] + gamma) / (self.c + 5 * gamma)
                p_z[1,z,s] = (p_z[1,z,s] * (self.cysw[1,s,w] + eta)) / (self.cys[1,s] + eta)
                p_z[1,z,s] = p_z[1,z,s] * aggregate_sentiment_probability(s,u,m)

        # y = 2
        for z in xrange(self.Z):
            for s in xrange(self.S):
                p_z[2,z,s] = (self.cy[2] + gamma) / (self.c + 5 * gamma)
                p_z[2,z,s] = (p_z[2,z,s] * (self.cyzw[2,z,w] + eta)) / (self.cyz[2,z] + eta)
                p_z[2,z,s] = p_z[2,z,s] * (joint_aspect(u, m)[z])
                p_z[2,z,s] = p_z[2,z,s] * aspect_sentiment_probability(s,u,m,z)

        # y = 3
        for z in xrange(self.Z):
            for s in xrange(self.S):
                p_z[3,z,s] = (self.cy[3] + gamma) / (self.c + 5 * gamma)
                p_z[3,z,s] = (p_z[3,z,s] * (self.cyzw[3,z,w] + eta)) / (self.cyz[3,z] + eta)
                p_z[3,z,s] = p_z[3,z,s] * (joint_aspect(u,m)[z])

        # y = 4
        for z in xrange(self.Z):
            for s in xrange(self.S):
                p_z[4,z,s] = (self.cy[4] + gamma) / (self.c + 5 * gamma)
                p_z[4,z,s] = (p_z[4,z,s] * (self.cymw[4,m,w] + eta)) / (self.cym[4,m] + eta)

        # Normalize
        p_z = p_z / p_z.sum()

        return p_z

    def run(self, matrix, max_iter=20):
        """
        """
        self._initialize(matrix)

        for it in xrange(max_iter):
            print 'Gibbs Sampling Iteration: %d' % it
            for r in xrange(self.n_reviews):
                for i, w in enumerate(word_indices(matrix[r, :])):
                    (y, z, s) = self.topics[(r, i)]
                    # Exclude current assignment
                    self.cy[y] -= 1
                    self.c -= 1
                    self.cyw[y,w] -= 1
                    self.cy[y] -= 1
                    self.cysw[y,s,w] -= 1
                    self.cys[y,s] -= 1
                    self.cyzw[y,z,w] -= 1
                    self.cyz[y,z] -= 1
                    # TODO: Define m
                    m = np.random.randint(self.M)
                    self.cymw[y,m,w] -= 1
                    self.cym[y,m] -= 1

                    # Get next distribution
                    # TODO: Define u
                    u = np.random.randint(1000)
                    p_z = self._conditional_distribution(u, m, w)
                    (y, z, s) = sample_multiple_indices(p_z)

                    # Assign new values
                    self.cy[y] += 1
                    self.c += 1
                    self.cyw[y,w] += 1
                    self.cy[y] += 1
                    self.cysw[y,s,w] += 1
                    self.cys[y,s] += 1
                    self.cyzw[y,z,w] += 1
                    self.cyz[y,z] += 1
                    # TODO: Define m
                    m = np.random.randint(self.M)
                    self.cymw[y,m,w] += 1
                    self.cym[y,m] += 1
                    self.topics[(r, i)] = (y, z, s)


