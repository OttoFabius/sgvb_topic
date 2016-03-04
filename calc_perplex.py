from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset
from vae_1l import topic_model_1layer
from vae_2l import topic_model_2layer
from vae_lin import topic_model_linear
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

    argdict = parse_config(sys.argv[1])
    samples = 1

    #	----------------				load dataset & create model 	   --------------------
    print "loading dataset"
    x = load_dataset(argdict)

    x_test = csc_matrix(x[argdict['trainset_size']:argdict['trainset_size']+argdict['testset_size'],:])
	
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
	
    n_test, voc_size = x_test.shape
    argdict['voc_size'] = voc_size

    testlowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound_test.npy')
    lowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound.npy')
    print 'train lb=', lowerbound_list[-1]
    print 'test lb=', testlowerbound_list[-1]/n_test

    if selected_features==None:
		n_features = voc_size
    else:
		n_features = selected_features.shape[0]

    if argdict['HUe2']==0:
        model = topic_model_1layer(argdict)
    else:
        model = topic_model_2layer(argdict)
    load_parameters(model, 'results/vae_own/' + sys.argv[1])
	
    docnrs = np.arange(1, argdict['testset_size'], 1)


    print 'evaluating with', samples, 'samples per datapoint'
    perplexity = []
    for i in xrange(samples):
    	print i
        log_perplexity = 0
        n_words=0
        for docnr in docnrs:
            doc = x_test[docnr,:]
            log_perplexity_doc, n_words_doc = model.calculate_perplexity(doc.T, selected_features=selected_features)
            log_perplexity += log_perplexity_doc
            n_words += n_words_doc

    	perplexity.append(np.exp(log_perplexity/n_words))
    print perplexity
    plt.hist(perplexity)
    plt.show()
    print np.mean(perplexity)