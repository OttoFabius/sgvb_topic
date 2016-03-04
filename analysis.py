import numpy as np
from vae_1l import topic_model_1layer
from vae_2l import topic_model_2layer
from vae_lin import topic_model_linear
import gzip
import cPickle as pickle
import scipy.sparse as sp
from helpfuncs import load_parameters, save_parameters, parse_config
import sys
from scipy.sparse import csr_matrix, csc_matrix
import matplotlib.pyplot as plt


argdict = parse_config(sys.argv[1])

# load data
f = gzip.open('data/KOS/docwordkos_matrix.pklz','rb')
x = pickle.load(f)
f.close()
x_csc = csc_matrix(x)
x_train = csc_matrix(x_csc[:argdict['trainset_size'],:])
x_test = csc_matrix(x_csc[argdict['trainset_size']:argdict['trainset_size']+argdict['testset_size'],:])
n, argdict['voc_size'] = x_train.shape
n_test = x_test.shape[0]

#create model
if argdict['HUe2']==0:
    model = topic_model_linear(argdict)
else:
    model = topic_model_2layer(argdict)

load_parameters(model, 'results/vae_own/' + sys.argv[1])
testlowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound_test.npy')
lowerbound_list = np.load('results/vae_own/' + sys.argv[1] + '/lowerbound.npy')

# plt.plot(-lowerbound_list)
# plt.gca().set_xscale("log")
# plt.gca().set_yscale("log")
# plt.show()

mu, logvar = model.encode(x_test.todense().T)
plt.hist(np.var(mu, 1), bins=np.linspace(0,3,15))
plt.title('#used latent dimensions')
plt.xlabel('variance over means of encoded datapoint')
plt.ylabel('#latent dimensions')
plt.show()
