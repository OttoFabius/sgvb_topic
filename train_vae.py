import numpy as np
from helpfuncs import *
from analysis import plot_stats, plot_used_dims
from vae_1l import topic_model_1layer
from vae_2l import topic_model_2layer
from vae_21l import topic_model_21layer
from vae_20l import topic_model_20layer
from vae_lin import topic_model_linear
import time
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import cPickle as pickle
import scipy.sparse as sp
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt



if __name__=="__main__":

    # THEANO_FLAGS=optimizer=None

    # import warnings
    # warnings.filterwarnings("ignore")

    argdict = parse_config(sys.argv[1])

    x = load_dataset(argdict)

    x_csc = csc_matrix(x)

    n_total, empty = x_csc.shape

    x_train = x_csc[:argdict['trainset_size'],:]
    x_test = x_csc[n_total-1-argdict['testset_size']:n_total-1,:] #always same test set
    n_train, argdict['voc_size'] = x_train.shape
    n_test = argdict['testset_size']

    print 'voc size:', argdict['voc_size'], "n_total:", n_total, "n_train:", n_train, "n_test:", n_test
    used_features = load_used_features(argdict)
    print "initializing model + graph..."
    if argdict['HUe2']==0:
        model = topic_model_1layer(argdict)
    elif argdict['HUd2']!=0:
        model = topic_model_2layer(argdict)
    elif argdict['HUd1']!=0:
        model = topic_model_21layer(argdict)    
    elif argdict['HUd1']==0:
        model = topic_model_20layer(argdict)

    else:
        print 'no model selected :('



    if len(sys.argv) > 2 and sys.argv[2] == "--load":
        print "loading params for restart"
    	load_parameters(model, 'results/vae_own/' + sys.argv[1])
        lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, recon_train_list, \
        recon_test_list, perplexity_list, perp_sem_list, epoch = load_stats('results/vae_own/' + sys.argv[1])
    	print "Restarting at epoch: " + str(epoch) + ' with lowerbounds ', lowerbound_list[-1], testlowerbound_list[-1]
    else:
    	lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, \
        recon_train_list, recon_test_list, perplexity_list, perp_sem_list = ([] for i in range(8))
        epoch = 0

        print "estimating perplexity on test set with", argdict['samples'], "samples"
        perplexity, perp_sem = perplexity_during_train(model, x_test, argdict)
        perplexity_list = np.append(perplexity_list, perplexity)
        perp_sem_list = np.append(perp_sem_list, perp_sem)
        print "perplexity =", perplexity, 'with', perp_sem, 'sem'



    print 'iterating' 

    while epoch < argdict['max_epochs']:

        epoch += 1      
        if epoch % argdict['save_every'] == 0:    

            print "saving stats, params at epoch", epoch
            save_stats(            'results/vae_own/'+sys.argv[1], lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, \
                                                                    recon_train_list, recon_test_list, perplexity_list, perp_sem_list)
            save_parameters(model, 'results/vae_own/'+sys.argv[1])

            print "estimating perplexity on test set with", argdict['samples'], "samples"
            perplexity, perp_sem = perplexity_during_train(model, x_test, argdict)
            perplexity_list = np.append(perplexity_list, perplexity)
            perp_sem_list = np.append(perp_sem_list, perp_sem)
            print "perplexity =", perplexity, 'with', perp_sem, 'sem'

        start = time.time()  
        x_train = shuffle(x_train)
        lowerbound, recon_train, KLD, KLD_used = model.iterate(x_train, epoch)
        testlowerbound, recon_test = model.getLowerBound(x_test, epoch)

        if epoch == 1:
            print time.time() - start, 'seconds for first epoch'
        print 'epoch ', epoch, 'lb: ', lowerbound/n_train, 'lb test', testlowerbound/(n_test-n_test%argdict['batch_size']), 'recon test', recon_test/(n_test-n_test%argdict['batch_size'])

        lowerbound_list     = np.append(lowerbound_list     , lowerbound    /n_train)
        KLD_list            = np.append(KLD_list            , KLD           /n_train)
        KLD_used_list       = np.append(KLD_used_list       , KLD_used      /n_train)
        recon_train_list    = np.append(recon_train_list    , recon_train   /n_train)

        recon_test_list     = np.append(recon_test_list     , recon_test    /(n_test-n_test%argdict['batch_size']))
        testlowerbound_list = np.append(testlowerbound_list , testlowerbound/(n_test-n_test%argdict['batch_size']))
 
    print "done, saving stats, params"
    save_stats(            'results/vae_own/'+sys.argv[1], lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, \
                                                                    recon_train_list, recon_test_list, perplexity_list, perp_sem_list)
    save_parameters(model, 'results/vae_own/'+sys.argv[1])

plot_stats(lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, perplexity_list, sys.argv[1])
plot_used_dims(model, x_test, sys.argv[1]) 