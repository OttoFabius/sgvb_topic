from helpfuncs import load_parameters, save_parameters, parse_config, load_dataset, perplexity_during_train, perplexity_rest
import gzip
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    perplexities = np.array([])
    n = np.array([])
    n_feats = np.array([])
    dataset='kos'
    freqs = [10, 12, 15, 20, 30, 50, 70, 100, 200, 300]
    for freq in freqs:
        print 'frequency', freq
        if freq == 1:
            f = gzip.open('data/'+dataset+'/docword_matrix.pklz','rb')
        else:
            f = gzip.open('data/'+dataset+'/docword_rest_matrix_'+str(freq)+'.pklz','rb')
        x = pickle.load(f)
        x_csc = csc_matrix(x)
        perplexities = np.append(perplexities, perplexity_rest(x_csc))
        n = np.append(n, csc_matrix.sum(x_csc))
        n_feats = np.append(n_feats, x_csc.shape[1])

    plt.plot(freqs, n/1e6)
    plt.title('Total number of unused words per word frequency')
    plt.xlabel('word frequency')
    plt.ylabel('#unused words (10^6)')
    plt.savefig("results/freqsn")
    plt.close()

    plt.plot(freqs, n_feats)
    plt.title('Number of unique unused words per word frequency')
    plt.xlabel('word frequency')
    plt.ylabel('#unique words')
    plt.savefig("results/freqsfeats")   
    plt.close()

    plt.plot(n_feats, perplexities)
    plt.title('Contribution of left out unique words to naive perplexity')
    plt.xlabel('#unique left out words')
    plt.ylabel('contribution towards perplexity')
    # plt.savefig("results/perplexrest")