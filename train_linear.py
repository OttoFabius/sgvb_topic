import numpy as np
import theano as th
from loadsave import load_pickle_list, load_pickle_matrix, create_pickle_matrix, create_pickle_list # , load_pickle
from linear_model_theano import topic_model_matrix
import sys
import ConfigParser
import time
from scipy.sparse import csr_matrix

def parse_config():
    config = ConfigParser.ConfigParser()
    config.read("results/" + sys.argv[1] + "/parameters.cfg")

    latent_variables = config.getint('parameters','latent_variables')
    hidden_units_pzd = config.getint('parameters','hidden_units_pzd')
    hidden_units_qx = config.getint('parameters','hidden_units_qx')
    hidden_units_qd = config.getint('parameters','hidden_units_qd')
    learning_rate = config.getfloat('parameters','learning_rate')
    sigmaInit = config.getfloat('parameters','sigmaInit')
    doc_per_doc = config.getboolean('parameters','doc_per_doc')
    batch_size = config.getint('parameters','batch_size')
    only_trainset = config.getboolean('parameters','only_trainset')


    return latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, learning_rate, sigmaInit, doc_per_doc, batch_size, only_trainset






if __name__=="__main__":
    THEANO_FLAGS=optimizer=None

    import warnings
    warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

    latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, \
    learning_rate, sigmaInit, doc_per_doc, batch_size, only_trainset = parse_config()

    #	----------------				load dataset & create model 	   --------------------

    if doc_per_doc:
        print 'using per-doc batch model, loading data'
        x, d = load_pickle_list() #no argument uses KOS dataset
        voc_size = d[1].size
        print "initializing model + graph..."
        model = topic_model(voc_size, latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, hidden_units_zx, learning_rate, sigmaInit)
    else:
        if not only_trainset:
            print 'using randomized batch model, loading all data'
            x,d,d_nrs = load_pickle_matrix() #no argument uses KOS dataset
            d_nrs = np.ndarray.flatten(d_nrs)
        else:
            print 'using randomized batch model, loading train and test data'
            x,d,d_nrs = load_pickle_matrix(filename='_train_matrix.pklz') #no argument uses KOS dataset
            d_nrs = np.ndarray.flatten(d_nrs)
            x_test,d_test,d_nrs_test = load_pickle_matrix(filename='_test_matrix.pklz') #no argument uses KOS dataset
            d_nrs_test = np.ndarray.flatten(d_nrs_test)
        voc_size = d.shape[1]    
        print "initializing model + graph..."
        model = topic_model_matrix(voc_size, latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, learning_rate, sigmaInit, batch_size, only_trainset)
    print 'done'

    #	----------------		optional: load parameters           --------------------

    if len(sys.argv) > 2 and sys.argv[2] == "--load":
    	model.load_parameters('results/' + sys.argv[1])
    	lowerbound_list = np.load('results/' + sys.argv[1] + '/lowerbound.npy')
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

        if doc_per_doc:
            lowerbound = model.iterate(x, d, epoch)
        else:
            lowerbound = model.iterate(x, d, d_nrs, epoch)
        print 'epoch ', epoch, 'with lowerbound = ', lowerbound/csr_matrix.sum(x), "and {0} seconds".format(time.time() - start)
        lowerbound_list = np.append(lowerbound_list, lowerbound/csr_matrix.sum(x))
        if epoch % 5 == 0:
        	print "saving lowerbound, params"
        	np.save('results/' + sys.argv[1] + '/lowerbound.npy', lowerbound_list)
        	model.save_parameters("results/" + sys.argv[1])

        

