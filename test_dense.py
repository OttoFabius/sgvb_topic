import numpy as np
from helpfuncs import *
from analysis import plot_lowerbound, plot_used_dims
from vae_1l_indices import topic_model_1layer
from vae_2l import topic_model_2layer
from vae_lin import topic_model_linear
import time
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import cPickle as pickle
import scipy.sparse as sp
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt




def test_dense(x):

    start = time.time()  
    a = x.todense()
    print time.time()