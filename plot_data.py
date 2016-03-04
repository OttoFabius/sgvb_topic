import numpy as np
from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset
from vae_1l import topic_model_1layer
from vae_2l import topic_model_2layer
import time
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import cPickle as pickle
import scipy.sparse as sp
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt



if __name__=="__main__":


    argdict = parse_config(sys.argv[1])

    x = load_dataset(argdict)
    x_dense = x.todense()
    x_sum = np.sum(x_dense, axis=0)

    plt.hist(x_sum.T, bins=np.linspace(0,1000,50))
    plt.title('#words in dataset, per word')
    plt.xlabel('#of instances of word')
    plt.ylabel('unique words')
    plt.show()