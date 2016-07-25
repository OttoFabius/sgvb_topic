import numpy as np
from random import seed
import matplotlib.pyplot as plt
from quantile_nn import quantile_nn


x = np.load('quantile_data_x.npy')
y = np.load('quantile_data_y.npy')

seed(5)
nnet = quantile_nn()

import cPickle as pickle
n_iter = 1000
for i in xrange(n_iter):
	MSE = nnet.iterate(x,y)
	print MSE

pickle.dump([name for name in nnet.params.keys()], open("qnn_names.pkl", "wb"))
pickle.dump([p.get_value() for p in nnet.params.values()], open("qnn_params.pkl", "wb"))