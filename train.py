import numpy as np
import theano as th
from loadsave import load_pickle_list, load_pickle_matrix, create_pickle_matrix, create_pickle_list # , load_pickle
from model_list import topic_model
from model_matrix import topic_model_matrix
import sys
import ConfigParser
import time
from scipy.sparse import csr_matrix

#-------------------       		 parse config file       		--------------------
import warnings
warnings.filterwarnings("ignore")

config = ConfigParser.ConfigParser()
config.read('results/' + sys.argv[1] + "/parameters.cfg")

latent_variables = config.getint('parameters','latent_variables')
hidden_units_pzd = config.getint('parameters','hidden_units_pzd')
hidden_units_qx = config.getint('parameters','hidden_units_qx')
hidden_units_qd = config.getint('parameters','hidden_units_qd')
learning_rate = config.getfloat('parameters','learning_rate')
sigmaInit = config.getfloat('parameters','sigmaInit')
doc_per_doc = config.getboolean('parameters','doc_per_doc')

#	----------------				load dataset & create model 	   --------------------


# create_pickle_list(filename = 'data/KOS/docwordkos_matrix.npy')

if doc_per_doc:
    print 'using per-doc batch model'
    x, d = load_pickle_list() #no argument uses KOS dataset
    voc_size = d[1].size
    print "initializing model + graph..."
    model = topic_model(voc_size, latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, learning_rate, sigmaInit, L=10)
    print 'done'
else:
    print 'using randomized batch model'
    x,d,d_nrs = load_pickle_matrix() #no argument uses KOS dataset
    d_nrs = np.ndarray.flatten(d_nrs)
    voc_size = d.shape[1]    
    print "initializing model + graph..."
    model = topic_model_matrix(voc_size, latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd, learning_rate, sigmaInit)
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
    print 'epoch ', epoch, 'with lowerbound = ', lowerbound/np.sum(d), "and {0} seconds".format(time.time() - start)
    lowerbound_list = np.append(lowerbound_list, lowerbound)
    if epoch % 5 == 0:
    	print "saving lowerbound, params"
    	np.save('results/' + sys.argv[1] + '/lowerbound.npy', lowerbound_list)
    	model.save_parameters("results/" + sys.argv[1])
        print "done"
    
