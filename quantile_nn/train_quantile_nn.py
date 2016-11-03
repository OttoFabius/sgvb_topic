import numpy as np
from random import seed
import matplotlib.pyplot as plt
from quantile_nn import quantile_nn
import cPickle as pickle

x = np.load('quantile_data_x.npy')
y_raw = np.load('quantile_data_y.npy')

x_small = np.load('quantile_data_small_x.npy')
y_small = np.load('quantile_data_small_y.npy')

meany = np.mean(y_raw)
stdy = np.sqrt(np.var(y_raw))

y = y_raw-meany
y = y/stdy

y_small = y_small-meany
y_small = y_small/stdy

nnet = quantile_nn()

RMSE=0

n_iter = 10000
for i in xrange(n_iter):
	y_approx = (nnet.forward(x_small)*stdy)+meany

	plt.plot(x_small[:,1], y_approx[:].T, 'r.', ms=3)
	plt.plot(x_small[:,1], y_small[:]*stdy+meany,'b.', ms=3)
	plt.xlabel('Epsilon', fontsize=16)
	plt.ylabel('Y', fontsize=16)
	plt.legend(['Output','True Values'], loc=2)
	plt.title('QNN during training. 0.01 < alpha < 2')
	plt.savefig('qnn_train_development_epoch_'+str(i)+'_RMSE_'+str(RMSE)+'.png')
	plt.close()

	RMSE, y_approx1 = nnet.iterate(x,y)

	print 'Saving, epoch', i, 'with RMSE', RMSE
	pickle.dump([name for name in nnet.params.keys()], open("qnn_names.pkl", "wb"))
	pickle.dump([p.get_value() for p in nnet.params.values()], open("qnn_params.pkl", "wb"))



