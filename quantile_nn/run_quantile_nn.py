import numpy as np
from random import seed
import matplotlib.pyplot as plt
from quantile_nn import quantile_nn
import cPickle as pickle
from scipy.special import gdtr

x = np.load('quantile_data_x.npy')
y = np.load('quantile_data_y.npy')


# x[:,0] = x[:,0] - np.mean(x[:,0])
# x[:,0] = x[:,0]/np.sqrt(np.var(x[:,0]))

meany = np.mean(y)
stdy = np.sqrt(np.var(y))

# y = y-meany
# y = y/stdy

#load nnet
nnet = quantile_nn()

names_qnn = pickle.load(open("qnn_names.pkl", "rb"))
params_qnn = pickle.load(open("qnn_params.pkl", "rb"))
for name, param in zip(names_qnn, params_qnn): 
	nnet.params[name] = param


y_k_approx = nnet.forward(x.T)



