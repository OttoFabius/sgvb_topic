from train_vae_own import parse_config
import vae_own
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp

if __name__=="__main__":


    THEANO_FLAGS=optimizer=None

    import warnings
    warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

    latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, only_trainset = parse_config()


    #	----------------				load dataset & create model 	   --------------------
    print "loading dataset"
    f = gzip.open('data/KOS/docwordkos_matrix.pklz','rb')
    x = pickle.load(f)
    f.close()



    print "converting to csr"
    x_train = csr_matrix(x)
    n, voc_size = x_train.get_shape()

    x_train = x_train.T
    model = vae_own.topic_model(voc_size, latent_variables, HUe1, HUd1, learning_rate, sigmaInit, batch_size, only_trainset, HUe2=HUe2, HUd2=HUd2)

    selected_features = None
    docnrs = range(3)

    ndocs_test = len(docnrs)
   

	perplexity = 0
	for docnr in docnrs:
		doc = x_train[:,docnr]
		perplexity += model.calculate_perplexity(doc, selected_features=selected_features)/ndocs_test 
	print perplexity

