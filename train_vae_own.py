import numpy as np
import theano as th
from loadsave import load_pickle_list, load_pickle_matrix, create_pickle_matrix, create_pickle_list # , load_pickle
from vae_own import topic_model
import sys
import ConfigParser
import time
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import cPickle as pickle
import scipy.sparse as sp
from sklearn.utils import shuffle

def parse_config():
    config = ConfigParser.ConfigParser()
    config.read("results/vae_own/" + sys.argv[1] + "/parameters.cfg")

    latent_variables = config.getint('parameters','latent_variables')
    HUe1 = config.getint('parameters','HUe1')
    HUe2 = config.getint('parameters','HUe2')
    HUd1 = config.getint('parameters','HUd1')
    HUd2 = config.getint('parameters','HUd2')
    learning_rate = config.getfloat('parameters','learning_rate')
    sigmaInit = config.getfloat('parameters','sigmaInit')
    batch_size = config.getint('parameters','batch_size')
    only_trainset = config.getboolean('parameters','only_trainset')


    return latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, only_trainset




if __name__=="__main__":


    THEANO_FLAGS=optimizer=None

    import warnings
    warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

    latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, only_trainset = parse_config()


    #	----------------				load dataset & create model 	   --------------------
    print "loading dataset"
    f = gzip.open('data/NY/docwordny_matrix_10000.pklz','rb')
    x = pickle.load(f)
    f.close()
    print "converting to csr"
    x_train = csr_matrix(x)
    n, voc_size = x_train.get_shape()
    print n, "datapoints and", voc_size, "features"

    print "initializing model + graph..."
    model = topic_model(voc_size, latent_variables, HUe1, HUd1, learning_rate, sigmaInit, batch_size, only_trainset, HUe2=HUe2, HUd2=HUd2)
    
    #	----------------		optional: load parameters           --------------------

    if len(sys.argv) > 2 and sys.argv[2] == "--load":
        print "loading params for restart"
    	model.load_parameters('results/vae_own/' + sys.argv[1])
    	lowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound.npy')
    	epoch = lowerbound_list.shape[0]
    	print "Restarting at epoch: " + str(epoch)
    else:
    	lowerbound_list = []
    	epoch = 0

    #	----------------				iterate      			     --------------------
    print 'iterating'
    while True:
        start = time.time()
        epoch += 1
        x_train = shuffle(x_train)
        lowerbound, recon_err, KLD = model.iterate(x_train, epoch)
        print 'epoch ', epoch, 'with objectives = ', lowerbound/n, recon_err/n, KLD/n, "and {0} seconds".format(time.time() - start)
        lowerbound_list = np.append(lowerbound_list, lowerbound/n)
        if epoch % 5 == 0:
        	print "saving lowerbound, params"
        	np.save('results/vae_own/' + sys.argv[1] + '/lowerbound.npy', lowerbound_list)
        	model.save_parameters("results/vae_own/" + sys.argv[1])