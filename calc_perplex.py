from train_vae_own import parse_config
import vae_own
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np

if __name__=="__main__":


    THEANO_FLAGS=optimizer=None

    import warnings
    warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

    latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, only_trainset = parse_config()


    #	----------------				load dataset & create model 	   --------------------
    print "loading dataset"
    f = gzip.open('data/NY/docwordny_matrix.pklz','rb')
    x = pickle.load(f)
    f.close()



    print "converting to csr"
    x_train = csr_matrix(x)
    n, voc_size = x_train.get_shape()

    # x_train = x_train.T #for kos dataset
    model = vae_own.topic_model(voc_size, latent_variables, HUe1, HUd1, learning_rate, sigmaInit, batch_size, only_trainset, HUe2=HUe2, HUd2=HUd2)

    selected_features = None
    docnrs = range(3)
    runs = 1
    ndocs_test = len(docnrs)
   	
    log_perplexity = 0
    for i in range(runs):
	    log_perplexity_run = 0
	    for docnr in docnrs:
			doc = x_train[:,docnr]
			log_perplexity_run += model.calculate_perplexity(doc, selected_features=selected_features)
	    log_perplexity += log_perplexity_run
    print log_perplexity/runs

