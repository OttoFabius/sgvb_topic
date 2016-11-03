from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np
import sys
import matplotlib.pyplot as plt


def plot_stats(lb, lb_test, KLD, KLDtrain, perplex, sem, model_title, save_every):
    plt.plot(KLD)
    plt.plot(KLDtrain)
    plt.gca().set_xscale("log")
    plt.xlabel('epochs')
    plt.ylabel('log KLD')
    plt.title('KL Divergence')
    plt.legend(['KLD', 'KLDused'])
    plt.savefig("results/vae_own/" + model_title + '/kld')
    plt.close()

    plt.plot(-lb)
    plt.plot(-lb_test)
    plt.gca().set_xscale("log")
    plt.xlabel('epochs')
    plt.ylabel('lowerbound of log likelihood')
    plt.title('stats')
    plt.legend(['train', 'test'])
    plt.savefig("results/vae_own/" + model_title + '/lb')
    plt.close()

    xaxis = np.arange(np.size(perplex))*(save_every)
    plt.plot(xaxis, perplex)
    plt.plot(xaxis, perplex+sem, '-')
    plt.plot(xaxis, perplex-sem, '-')
    plt.gca().set_xscale("log")
    plt.ylim((600, 1200))
    plt.legend(['perplexity', 'upper confidence', 'lower confidence'])
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.title('perplexity per 10 epochs with min perplexity ' + str(np.min(perplex)))
    plt.savefig("results/vae_own/" + model_title + '/perplex')

def plot_used_dims(model, x_test, model_title):
    plt.figure()
    mu, logvar = model.encode(x_test.todense().T)
    plt.hist(np.var(mu, 1), bins=np.linspace(0,3,15))
    plt.title('#used latent dimensions')
    plt.xlim( 0, 14 )
    plt.xlabel('variance over means of encoded datapoint')
    plt.ylabel('#latent dimensions')
    plt.savefig("results/vae_own/" + model_title + '/used_latent_dims')

if __name__=="__main__":

    argdict = parse_config(sys.argv[1])
    x = load_dataset(argdict)
    x_csc = csc_matrix(x)
    n_total, empty = x_csc.shape
    x_train = x_csc[:argdict['trainset_size'],:]
    x_test = x_csc[n_total-1-argdict['trainset_size']:n_total-1,:] #always same test set
    argdict['samples'] = 10
    if argdict['minfreq'] == 0:
        selected_features=None


    n_test, voc_size = x_test.shape
    argdict['voc_size'] = voc_size

    if argdict['HUe2']==0:
        model = topic_model_1layer(argdict)
    else:
        model = topic_model_2layer(argdict)
    load_parameters(model, 'results/vae_own/' + sys.argv[1])
    lb_test = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound_test.npy')
    lb = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound.npy')

    plot_lowerbound(lb, lb_test, sys.argv[1])
    plot_used_dims(model, x_test, sys.argv[1]) 

