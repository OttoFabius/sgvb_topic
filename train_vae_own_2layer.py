import numpy as np
import theano as th
from loadsave import load_pickle_list, load_pickle_matrix, create_pickle_matrix, create_pickle_list # , load_pickle
from vae_own_2layer import topic_model
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
    trainset_size = config.getint('parameters','trainset_size')
    dataset_num = config.getint('parameters','dataset')
    if dataset_num == 0:
        dataset = 'kos'
    elif dataset_num == 1:
        dataset = 'ny'
    minfreq = config.getint('parameters','minfreq')
    entselect = config.getint('parameters','entselect')


    return latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, trainset_size, dataset, minfreq, entselect




if __name__=="__main__":

    THEANO_FLAGS=optimizer=None

    import warnings
    warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

    latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, trainset_size, dataset, minfreq, entselect = parse_config()


    #	----------------				load dataset & create model 	   --------------------
    
    if dataset=='kos':
        if minfreq>0:
            print "loading KOS dataset with minimum", minfreq, 'word frequency'
            f = gzip.open('data/KOS/docwordkos_matrix_'+str(minfreq)+'.pklz','rb')
        elif entselect>0:
            f = gzip.open('data/KOS/docwordkos_matrix_'+str(entselect)+'_ent.pklz','rb')
            print "loading KOS dataset with", entselect, 'features selected on entropy'
        else:
            print 'loading KOS dataset full vocabulary'
            f = gzip.open('data/KOS/docwordkos_matrix.pklz','rb')

    if dataset=='ny':
        if minfreq>0:
            print "loading NY dataset with minimum", minfreq, 'word frequency'
            f = gzip.open('data/NY/docwordny_matrix_'+str(minfreq)+'.pklz','rb')
        elif entselect>0:
            f = gzip.open('data/NY/docwordny_matrix_'+str(entselect)+'_ent.pklz','rb')
            print "loading NY dataset with", entselect, 'features selected on entropy'
        else:
            print 'loading NY dataset full vocabulary'
            f = gzip.open('data/NY/docwordny_matrix.pklz','rb')

    x = pickle.load(f)
    f.close()
    print "converting to csr"
    x_csc = csc_matrix(x)
    x_train = csc_matrix(x_csc[:trainset_size,:])
    x_test = csc_matrix(x_csc[trainset_size:,:])
    n, voc_size = x_train.get_shape()
    print n, "datapoints and", voc_size, "features"

    print "initializing model + graph..."
    model = topic_model(voc_size, latent_variables, HUe1, HUd1, learning_rate, sigmaInit, batch_size, HUe2=HUe2, HUd2=HUd2)
    
    #	----------------		optional: load parameters           --------------------

    if len(sys.argv) > 2 and sys.argv[2] == "--load":
        print "loading params for restart"
    	model.load_parameters('results/vae_own/' + sys.argv[1])
    	lowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound.npy')
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
        lowerbound, recon_err, KLD = model.iterate(x_train, epoch)
        testlowerbound = model.getLowerBound(x_test)
        print 'epoch ', epoch, 'with objectives = ', lowerbound/n, "testlowerbound = ", testlowerbound, ",and {0} seconds".format(time.time() - start)

        lowerbound_list = np.append(lowerbound_list, lowerbound/n)
        testlowerbound_list = np.append(testlowerbound_list,testlowerbound)

        if epoch % 5 == 0:            
        	print "saving lowerbound, testlowerbound, params"
        	np.save('results/vae_own/' + sys.argv[1] + '/lowerbound.npy', lowerbound_list)
        	np.save('results/vae_own/' + sys.argv[1] + '/lowerbound_test.npy', testlowerbound_list)
        	model.save_parameters("results/vae_own/" + sys.argv[1])