import logging
from constants import *
from optimize import optimizer
from sampler import GibbsSampler
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib
from indexer import Indexer

# Constants
MAX_ITER = 500
MAX_OPT_ITER = 50

def main():
    """
    """
    # Download data for NLTK if not already done
    #nltk.download('all')

    # Read 
    imdb = Indexer()
    imdb_file = 'data/data_t.json'
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info('Reading file %s' % imdb_file)
    imdb.read_file(imdb_file)
    logging.info('File %s read' % imdb_file)
    (vocab_size, user_list, movie_list, \
    rating_matrix, review_matrix, review_map) = imdb.get_mappings()

    # Get number of users and movies
    U = len(user_list)
    M = len(movie_list)
    logging.info('No. of users U = %d' % U)
    logging.info('No. of movies M = %d' % M)

    # Run Gibbs EM
    for it in xrange(1,MAX_ITER+1):
        logging.info('Running iteration %d of Gibbs EM' % it)
        logging.info('Running E-Step - Gibbs Sampling')
        gibbs_sampler = GibbsSampler(len(y),len(z),len(s))
        gibbs_sampler.run(rating_matrix)
        logging.info('Running M-Step - Gradient Descent')
        for i in xrange(1,MAX_OPT_ITER+1):
            optimizer()

if __name__ == "__main__":
    main()
