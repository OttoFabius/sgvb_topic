from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset, perplexity_during_train
from vae_1l import topic_model_1layer
from vae_2l import topic_model_2layer
from vae_lin import topic_model_linear
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np
import sys
import matplotlib.pyplot as plt


if __name__=="__main__":

    argdict = parse_config(sys.argv[1])
    x = load_dataset(argdict)
    x_csc = csc_matrix(x)
    n_total, empty = x_csc.shape
    x_train = x_csc[:argdict['trainset_size'],:]
    x_test = x_csc[n_total-1-argdict['trainset_size']:n_total-1,:] #always same test set
    argdict['samples'] = 10
    if argdict['minfreq'] == 0:
        selected_features=None


    n_test, voc_size = x_test.shape
    argdict['voc_size'] = voc_size

    if argdict['HUe2']==0:
        model = topic_model_1layer(argdict)
    else:
        model = topic_model_2layer(argdict)
    load_parameters(model, 'results/vae_own/' + sys.argv[1])

    print 'evaluating with', argdict['[samples'], 'samples per datapoint'
    perp_mean, perp_std = perplexity_during_train(model, x_test, argdict)

    print perp_mean, perp_std