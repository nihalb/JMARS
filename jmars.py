from constants import *
from optimize import *
from sampler import GibbsSampler
import numpy as np
import scipy as sp
from scipy.optimize import fmin_l_bfgs_b
from numpy import linalg as LA
import numpy.matlib
from indexer import Indexer

def main():
    """
    """
    # Download data for NLTK if not already done
    #nltk.download('all')

    # Read 
    imdb = Indexer()
    imdb.read_file('data/data_t.json')
    x = imdb.get_mappings()

if __name__ == "main":
    main()
