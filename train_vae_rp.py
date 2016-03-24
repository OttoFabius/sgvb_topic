import numpy as np
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



if __name__=="__main__":

    # THEANO_FLAGS=optimizer=None

    # import warnings
    # warnings.filterwarnings("ignore")

    argdict = parse_config(sys.argv[1])

    x = load_dataset(argdict)

    x_csc = csc_matrix(x)
    
    if argdict['rp']==1:        
        print 'using random projection of rest in encoder'
        f = gzip.open('data/'+argdict['dataset']+'/data_proj_'+str(argdict['minfreq'])+'.pklz','rb')
        rest = pickle.load(f)
        f.close()


    n_total, empty = x_csc.shape
    n_test = argdict['testset_size']
    n_train = argdict['trainset_size']

    x_train = x_csc[:n_train,:]
    x_test = x_csc[n_total-1-n_test:n_total,:] 
    if argdict['rp']==1:
        rest_train = rest[:n_train,:]
        rest_test = rest[n_total-1-n_test:n_total,:]
    else:
        rest_train = None
        rest_test = None

    argdict['voc_size'] = x_train.shape[1]
    print 'voc size:', argdict['voc_size'], "n_total:", n_total, "n_train:", n_train, "n_test:", n_test

    used_features = load_used_features(argdict)

    model = initialize_model(argdict)

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

        print "estimating perplexity on test set with", argdict['samples'], "samples"
        perplexity, perp_sem = perplexity_during_train(model, x_test, argdict, rest=rest_test)
        perplexity_list = np.append(perplexity_list, perplexity)
        perp_sem_list = np.append(perp_sem_list, perp_sem)
        print "perplexity =", perplexity, 'with', perp_sem, 'sem'



    print 'iterating' 
    idx = np.arange(n_train)
    while epoch < argdict['max_epochs']:

        epoch += 1      
        if epoch % argdict['save_every'] == 0:    

            print "saving stats, params at epoch", epoch
            # save_stats(            'results/vae_own/'+sys.argv[1], lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, \
            #                                                         recon_train_list, recon_test_list, perplexity_list, perp_sem_list)
            # save_parameters(model, 'results/vae_own/'+sys.argv[1])

            print "estimating perplexity on test set with", argdict['samples'], "samples"
            perplexity, perp_sem = perplexity_during_train(model, x_test, argdict, rest=rest_test)
            perplexity_list = np.append(perplexity_list, perplexity)
            perp_sem_list = np.append(perp_sem_list, perp_sem)
            print "perplexity =", perplexity, 'with', perp_sem, 'sem'

        start = time.time()  
        
        np.random.shuffle(idx)
        x_train = x_train[idx,:]
        if argdict['rp']==1:
            rest_train = rest_train[idx,:]

        lowerbound, recon_train, KLD, KLD_used = model.iterate(x_train, epoch, rest=rest_train)
        testlowerbound, recon_test = model.getLowerBound(x_test, epoch, rest=rest_test)

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

plot_stats(lowerbound_list, testlowerbound_list, KLD_list, KLD_used_list, perplexity_list, perp_sem_list, sys.argv[1], argdict['save_every'])
plot_used_dims(model, x_test, sys.argv[1]) 