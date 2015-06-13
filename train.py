import numpy as np
import theano as th
from loadsave import load_pickle_list # , load_pickle
from model_list import topic_model
import sys
import ConfigParser
import time



#-------------------       		 parse config file       		--------------------
import warnings
warnings.filterwarnings("ignore")

config = ConfigParser.ConfigParser()
config.read('results/' + sys.argv[1] + "/parameters.cfg")

latent_variables = config.getint('parameters','latent_variables')
hidden_units_pzd = config.getint('parameters','hidden_units_pzd')
hidden_units_qx = config.getint('parameters','hidden_units_qx')
hidden_units_qd = config.getint('parameters','hidden_units_qd')

#	----------------				load dataset       	 	   --------------------

x, d = load_pickle_list() #no argument uses KOS dataset
voc_size = d[1].size

#-------------------       		 initialize model       		--------------------

print "initializing model + graph..."
model = topic_model(voc_size, latent_variables, hidden_units_pzd, hidden_units_qx, hidden_units_qd)

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
	
while True:
    start = time.time()
    epoch += 1
    lowerbound = model.iterate(x, d, epoch)
    print 'epoch ', epoch, 'with lowerbound = ', lowerbound/np.sum(d), "and {0} seconds".format(time.time() - start)
    lowerbound_list = np.append(lowerbound_list, lowerbound)
    if epoch % 10 == 0:
    	print "saving lowerbound, params"
    	np.save('results/' + sys.argv[1] + 'lowerbound.npy', lowerbound_list)
    	model.save_parameters("results")
    
