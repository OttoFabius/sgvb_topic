import numpy as np
from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset
from vae_1l import topic_model_1layer
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



if __name__=="__main__":

    # THEANO_FLAGS=optimizer=None

    # import warnings
    # warnings.filterwarnings("ignore")

    argdict = parse_config(sys.argv[1])

    # f = gzip.open('data/KOS/docwordkos_matrix.pklz','rb')
    # x = pickle.load(f)
    # f.close()

    x = load_dataset(argdict)
    x_csc = csc_matrix(x)


    n_total, empty = x_csc.shape

    x_train = x_csc[:argdict['trainset_size'],:]
    x_test = x_csc[n_total-1-argdict['trainset_size']:n_total-1,:] #always same test set
    n, argdict['voc_size'] = x_train.shape
    n_test = x_test.shape[0]


    print "initializing model + graph..."
    if argdict['HUe2']==0:
        model = topic_model_1layer(argdict)
    else:
        model = topic_model_2layer(argdict)
    
    #	----------------		optional: load parameters           --------------------

    if len(sys.argv) > 2 and sys.argv[2] == "--load":
        print "loading params for restart"
    	load_parameters(model, 'results/vae_own/' + sys.argv[1])
    	lowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound.npy')
        testlowerbound_list = []
        testlowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound_test.npy')
    	epoch = lowerbound_list.shape[0]
    	print "Restarting at epoch: " + str(epoch)
    else:
    	lowerbound_list = []
        testlowerbound_list = []
    	epoch = 0

    #	----------------				iterate      			     --------------------
    print 'iterating'
    while True:
        start = time.time()
        epoch += 1
        x_train = shuffle(x_train)
        lowerbound, recon_err, KLD, KLD_train = model.iterate(x_train, epoch)
        testlowerbound = model.getLowerBound(x_test, epoch)
        print 'epoch ', epoch, 'with objectives =', lowerbound/n, "testlowerbound =", testlowerbound/n_test, ",and {0} seconds".format(time.time() - start)
        print 'kld: ', KLD/n, 'kld train: ', KLD_train/n, 'recon_err', recon_err/n
        lowerbound_list = np.append(lowerbound_list, lowerbound/n)
        testlowerbound_list = np.append(testlowerbound_list,testlowerbound/n_test)

        if epoch % 10 == 0:            
            print "saving lowerbound, testlowerbound, params"
            np.save('results/vae_own/' + sys.argv[1] + '/lowerbound.npy', lowerbound_list)
            np.save('results/vae_own/' + sys.argv[1] + '/lowerbound_test.npy', testlowerbound_list)
            save_parameters(model, "results/vae_own/" + sys.argv[1])
            print "done"