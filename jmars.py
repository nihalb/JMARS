import logging
from constants import *
from optimize import optimizer
from sampler import GibbsSampler, predicted_rating
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib
from indexer import Indexer

# Constants
MAX_ITER = 500
MAX_OPT_ITER = 10

def main():
    """
    Main function
    """
    # Download data for NLTK if not already done
    #nltk.download('all')

    # Read 
    imdb = Indexer()
    imdb_file = 'data/data.json'
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.info('Reading file %s' % imdb_file)
    imdb.read_file(imdb_file)
    logging.info('File %s read' % imdb_file)
    (vocab_size, user_list, movie_list, \
    rating_matrix, review_matrix, review_map) = imdb.get_mappings()

    # Get number of users and movies
    Users = len(user_list)
    Movies = len(movie_list)
    logging.info('No. of users U = %d' % Users)
    logging.info('No. of movies M = %d' % Movies)

    # Run Gibbs EM
    for it in xrange(1,MAX_ITER+1):
        logging.info('Running iteration %d of Gibbs EM' % it)
        logging.info('Running E-Step - Gibbs Sampling')
        gibbs_sampler = GibbsSampler(5,A,2)
        gibbs_sampler.run(rating_matrix)
        logging.info('Running M-Step - Gradient Descent')
        for i in xrange(1,MAX_OPT_ITER+1):
            optimizer()

    # Output Predicted Ratings
    for u in range(U):
        for m in range(M):
            pred_rate = predicted_rating(u, m)
            print "Predicted Rating of user " + str(u) + " and movie " + str(m) + ": " + str(pred_rate)

if __name__ == "__main__":
    main()
