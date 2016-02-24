from train_vae_own import parse_config
import vae_own as vae
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np
import sys
import matplotlib.pyplot as plt

if __name__=="__main__":


	THEANO_FLAGS=optimizer=None

	import warnings
	warnings.filterwarnings("ignore")

    #-------------------       		 parse config file       		--------------------

	latent_variables, HUe1, HUe2, HUd1, HUd2, learning_rate, sigmaInit, batch_size, trainset_size, validationset_size, dataset, minfreq, entselect = parse_config()
	runs = 5

    #	----------------				load dataset & create model 	   --------------------
	print "loading dataset"
	if dataset == 'ny':
		f = gzip.open('data/NY/docwordny_matrix.pklz','rb')
	elif dataset == 'kos':
		f = gzip.open('data/KOS/docwordkos_matrix.pklz','rb')
	elif dataset == 'enron':
		f = gzip.open('data/enron/docwordenron_matrix.pklz','rb')
	x = pickle.load(f)
	f.close()
	x_test = csc_matrix(x[trainset_size:trainset_size+validationset_size,:])
	
	# -------------------- selected features: not great for evaluating (?) ----------------
	if len(sys.argv) > 2 and sys.argv[2] == "--selected_features":
		print "selected features"
		if dataset == '':
			print 'ny'
	 		f = gzip.open('data/NY/docwordny_'+str(minfreq) +'selected.pklz','rb')
		elif dataset=='kos':
			print 'kos'
			f = gzip.open('data/KOS/docwordkos_'+str(minfreq) +'_selected.pklz','rb')
		selected_features = pickle.load(f)
		f.close()

		if dataset == 'ny':
			f = gzip.open('data/NY/docwordny_means.pklz','rb')
		elif dataset=='kos':
			f = gzip.open('data/KOS/docwordkos_means.pklz','rb')
		word_means = pickle.load(f)
		f.close()
	else:
		selected_features = None
	
	n, voc_size = x_test.shape


	if selected_features==None:
		n_features = voc_size
	else:
		n_features = selected_features.shape[0]

	model = vae.topic_model(n_features, latent_variables, HUe1, HUd1, learning_rate, sigmaInit, batch_size, HUe2=HUe2, HUd2=HUd2)
	model.load_parameters('results/vae_own/' + sys.argv[1])
	
	docnrs = np.arange(1, validationset_size, 1)
	log_perplexity = 0
	n_words=0

	means = np.zeros((2,430))
	# j=0

	print 'evaluating with', runs, 'runs'

	for docnr in docnrs:
		doc = x_test[docnr,:]

		# mu, logvar = model.encode(np.array(doc.T.todense()))
		# means[:,j]=mu[(5,32),0]
		# j+=1


		log_perplexity_doc, n_words_doc = model.calculate_perplexity(doc.T, selected_features=selected_features, runs=runs)
		log_perplexity += log_perplexity_doc
		n_words += n_words_doc

	# plt.scatter(means[1,:],means[0,:])
	# plt.show()
	print log_perplexity/n_words


