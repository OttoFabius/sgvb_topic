import numpy as np
from random import seed
from helpfuncs import *
from analysis import plot_stats, plot_used_dims
import time
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import cPickle as pickle
import scipy.sparse as sp
from sklearn.utils import shuffle
import sys
import matplotlib.pyplot as plt
from vae import topic_model


if __name__=="__main__":
    seed(5)
    # THEANO_FLAGS=optimizer=None

    # import warnings
    # warnings.filterwarnings("ignore")

    argdict = parse_config(sys.argv[1])

    x = load_dataset(argdict)

    # if sys.argv[2]=='GCE':
    #     x = precompute_GCE(x, argdict)
    x = csc_matrix(x)



    # TODO put code below in load_dataset, include train/test splits

    if argdict['rp']==1:        
        print 'using random projection of rest in encoder'
        rest = np.load('data/'+argdict['dataset']+'/data_proj_'+str(argdict['minfreq'])+'.npy')
   

    n_total, empty = x.shape
    n_test = argdict['testset_size']
    n_train = argdict['trainset_size']


    # always use same test set
    x_test = x[:n_test,:]
    x_train = x[n_total-n_train:n_total,:]

    # normalize per document NB should optimize this method for use on large datasets
    if argdict['normalize_input']==1:
        dl_train = np.ndarray.flatten(np.array(csc_matrix.sum(x_train, axis=1)))
        dl_test  = np.ndarray.flatten(np.array(csc_matrix.sum(x_test , axis=1)))
        x_test_notnorm = x_test
        x_test = x_test/csc_matrix.sum(x_test, 1)
        x_train = x_train/csc_matrix.sum(x_train, 1)

        total_words_train = np.sum(dl_train)

    if argdict['rp']==1:
        rest_train = rest[:n_test,:]
        rest_test = rest[n_total-1-n_train:n_total,:]
    else:
        rest_train = None
        rest_test = None

    unused_sum = get_unused_sum(argdict)

    argdict['voc_size'] = x_train.shape[1]
    print 'vocabulary size:', argdict['voc_size']
    raw_input()
    print argdict

    used_features = load_used_features(argdict)
    model = topic_model(argdict)

    if len(sys.argv) > 2 and sys.argv[2] == "--load":
        print "loading params for restart"
        load_parameters(model, 'results/vae_own/' + sys.argv[1])
        lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, recon_train_list, \
                    recon_test_list, perplexity_list, perp_sem_list, epoch = load_stats('results/vae_own/' + sys.argv[1])
        print "Restarting at epoch: " + str(epoch) + ' with lowerbounds ', lowerbound_list[-1], testlowerbound_list[-1]
    else:
        lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, recon_train_list, \
                    recon_test_list, perplexity_list, perp_sem_list = ([] for i in range(8))
        epoch = 0

        testlowerbound, recon_test, KLD_test = model.getLowerBound(x_test, dl_test, unused_sum, epoch, rest=rest_test)
        print 'lb test', testlowerbound

        perplexity_est = list()
        for i in xrange(argdict['samples']):
            perplexity_est.append(model.calc_perplexity(argdict, x_test_notnorm, unused_sum))
        perplexity_list.append(np.mean(perplexity_est))
        perp_sem_list.append(np.std(perplexity_est)/np.sqrt(argdict['samples']))
        print 'perplexity is', perplexity_list[-1], 'with sem', perp_sem_list[-1]

    print 'iterating' 
    idx = np.arange(n_train)
    print epoch, argdict['max_epochs']
    while epoch < argdict['max_epochs']:

        epoch += 1      
        if epoch % argdict['save_every'] == 0:    
            perplexity_est = list()
            for i in xrange(argdict['samples']):
                perplexity_est.append(model.calc_perplexity(argdict, x_test_notnorm, unused_sum))
            perplexity_list.append(np.mean(perplexity_est))
            perp_sem_list.append(np.std(perplexity_est)/np.sqrt(argdict['samples']))
            print 'perplexity is', perplexity_list[-1], 'with sem', perp_sem_list[-1]

        start = time.time()  
        

        np.random.shuffle(idx)
        x_train = x_train[idx,:]
        dl_train = dl_train[idx]

        if argdict['rp']==1:
            rest_train = rest_train[idx,:]

        lowerbound, recon_train, KLD, KLD_used = model.iterate(x_train, dl_train, unused_sum, epoch, rest=rest_train)
        testlowerbound, recon_test, KLD_test = model.getLowerBound(x_test, dl_test, unused_sum, epoch, rest=rest_test) 

        if epoch == 1:
            print time.time() - start, 'seconds for first epoch'
        if epoch % argdict['save_every'] == 0: 
            print 'epoch ', epoch, 'lb: ', lowerbound, 'lb test', \
                    testlowerbound, \
                    'recon test', recon_test, \
                    'KLD test', KLD_test
                    
        lowerbound_list     = np.append(lowerbound_list     , lowerbound )
        KLD_list            = np.append(KLD_list            , KLD        )
        KLD_used_list       = np.append(KLD_used_list       , KLD_used   )
        recon_train_list    = np.append(recon_train_list    , recon_train)

        recon_test_list     = np.append(recon_test_list     , recon_test    )
        testlowerbound_list = np.append(testlowerbound_list , testlowerbound)
 
    print "done, skipping saving stats, params"
    save_stats(            'results/vae_own/'+sys.argv[1], lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, \
                                                                    recon_train_list, recon_test_list, perplexity_list, perp_sem_list)
    save_parameters(model, 'results/vae_own/'+sys.argv[1])

plot_stats(lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, perplexity_list, perp_sem_list, sys.argv[1], argdict['save_every'])
plot_used_dims(model, x_test, sys.argv[1]) 
