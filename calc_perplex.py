from train_vae_own_2layer import parse_config
import vae_own_2layer
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np
import sys

if __name__=="__main__":


	THEANO_FLAGS=optimizer=None

	import warnings
	warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

	latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, trainset_size, validationset_size, dataset, minfreq, entselect = parse_config()


    #	----------------				load dataset & create model 	   --------------------
	print "loading dataset"
	if dataset == 'ny':
		f = gzip.open('data/NY/docwordny_matrix.pklz','rb')
	elif dataset == 'kos':
		f = gzip.open('data/KOS/docwordkos_matrix.pklz','rb')
	x = pickle.load(f)
	f.close()

	if len(sys.argv) > 2 and sys.argv[2] == "--selected_features":
		print "selected features"
		if dataset == 'ny':
			print 'ny'
	 		f = gzip.open('data/NY/docwordny_'+str(minfreq) +'selected.pklz','rb')
		elif dataset=='kos':
			print 'kos'
			f = gzip.open('data/KOS/docwordkos_'+str(minfreq) +'_selected.pklz','rb')
		selected_features = pickle.load(f)
		f.close()
	else:
		selected_features = None
	
	if dataset == 'ny':
		f = gzip.open('data/NY/docwordny_means.pklz','rb')
	elif dataset=='kos':
		f = gzip.open('data/KOS/docwordkos_means.pklz','rb')
	word_means = pickle.load(f)
	f.close()

	x_csr = csr_matrix(x)

	# x_csr = x_csr.T #for kos dataset with all features
	n, voc_size = x_csr.shape

	print n, voc_size


	if selected_features==None:
		n_features = voc_size
	else:
		n_features = selected_features.shape[0]

	model = vae_own_2layer.topic_model(n_features, latent_variables, HUe1, HUd1, learning_rate, sigmaInit, batch_size, HUe2=HUe2, HUd2=HUd2)
	model.load_parameters('results/vae_own/' + sys.argv[1])
	docnrs = np.arange(trainset_size, n, 1)
	runs = 1
	ndocs_test = len(docnrs)
   	
	log_perplexity = 0
	for i in range(runs):
	    log_perplexity_run = 0
	    for docnr in docnrs:
			doc = x_csr[docnr,:]
			# word_means = np.zeros_like(doc.toarray())
			# print word_means
			log_perplexity_run += model.calculate_perplexity(doc.T, selected_features=selected_features, means=word_means.T)
	    log_perplexity += log_perplexity_run
	print log_perplexity/runs


