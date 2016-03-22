from scipy.linalg import orth
from scipy.sparse import csc_matrix
import numpy as np
from helpfuncs import *
from analysis import plot_stats, plot_used_dims
from scipy.sparse import csr_matrix, csc_matrix
import gzip
import cPickle as pickle
import sys
import matplotlib.pyplot as plt


argdict = parse_config(sys.argv[1])

x = load_dataset(argdict)
x_csc = csc_matrix(x)
N, D = x_csc.shape

K = 50
print 'creating data'
data = np.random.rand(N, D)
print 'creating R'
R = np.random.normal(0, 1/np.sqrt(D), [D, K])
print R.shape
print 'orthogonalize'
R_o = orth(R)
print R_o.shape
print 'check orth'
dot = np.dot(R_o.T, R_o)
print 'max is ', np.max(np.dot(R_o.T, R_o))
print 'projecting data'
data_proj = np.dot(data, R)
print data_proj.shape
plt.hist(np.sum(R**2, axis=1))
plt.show()
plt.hist(np.sqrt(np.sum(data_proj**2, axis=0)))
plt.show()
