
from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset, load_stats
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np
import sys
import matplotlib.pyplot as plt
from vae import topic_model

def plot_stats(lb, lb_test, KLD, KLDtrain, perplex, sem, model_title, save_every):
    plt.plot(KLD)
    plt.plot(KLDtrain)
    plt.gca().set_xscale("log")
    plt.xlabel('Epochs')
    plt.ylabel('Log KLD')
    plt.ylim((0, 0.5))

    plt.title('KL Divergence')
    plt.legend(['KLD', 'KLD used'])
    plt.savefig("results/vae_own/" + model_title + '/kld')
    plt.close()

    plt.plot(lb)
    plt.plot(lb_test)
    plt.gca().set_xscale("log")
    plt.ylim((-9, -6))
    plt.xlabel('Epochs')
    plt.ylabel('Lower Bound')
    plt.title('Lower Bound of Log Likelihood')
    plt.legend(['Train', 'Test'])
    plt.savefig("results/vae_own/" + model_title + '/lb')
    plt.close()

    xaxis = np.arange(np.size(perplex))*(save_every)
    upbound = [(x + y) for (x, y) in zip(perplex, sem)]
    lowbound = [(x - y) for (x, y) in zip(perplex, sem)]
    plt.plot(xaxis, perplex, 'b')
    plt.plot(xaxis, upbound, 'b--')
    plt.plot(xaxis, lowbound, 'b--')
    plt.gca().set_xscale("log")

    plt.ylim((1400, 2500))
    # plt.legend(['Perplexity', 'upper confidence', 'lower confidence'])
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')

    plt.title('Test Perplexity During Training')
    plt.savefig("results/vae_own/" + model_title + '/perplex')
    plt.close()
    print "best stats:", np.max(lb), np.max(lb_test), np.min(perplex) 
    print "end stats:", lb[-1], lb_test[-1], perplex[-1]

def plot_used_dims(model, x_test, model_title):
    plt.figure()
    mu, logvar = model.encode(x_test.T)
    plt.hist(np.var(mu, 1), bins=np.linspace(0,6,10))
    plt.title('#used latent dimensions')
    plt.xlim( 0, 3 )
    plt.xlabel('variance over means of encoded datapoint')
    plt.ylabel('#latent dimensions')
    plt.savefig("results/vae_own/" + model_title + '/used_latent_dims')
    plt.close()

if __name__=="__main__":

    argdict = parse_config(sys.argv[1])
    # x = load_dataset(argdict)
    # x_csc = csc_matrix(x)

    # n_total, empty = x_csc.shape
    # x_train = x_csc[:argdict['trainset_size'],:]
    # x_test = x_csc[n_total-1-argdict['trainset_size']:n_total-1,:] #always same test set
    # argdict['samples'] = 10
    # if argdict['minfreq'] == 0:
    #     selected_features=None


    # n_test, voc_size = x_test.shape
    # argdict['voc_size'] = voc_size

    # model = topic_model(argdict)

    # load_parameters(model, 'results/vae_own/' + sys.argv[1])
    lb, lb_test, KLD, KLDtrain, recon_train, recon_test, perplexity, perp_sem, epoch = load_stats('results/vae_own/'+ sys.argv[1])
    plot_stats(lb, lb_test, KLD, KLDtrain, perplexity, perp_sem, sys.argv[1], argdict['save_every'])
    # plot_used_dims(model, x_test, sys.argv[1]) 

