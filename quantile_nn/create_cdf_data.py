import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gdtr


''' Gamma distribution cumulative density function.

Returns the integral from zero to x of the gamma probability density function:

a**b / gamma(b) * integral(t**(b-1) exp(-at), t=0..x).
The arguments a and b are used differently here than in other definitions. 

It seems to match the wikipedia definition of shape alpha and rate beta (need to test)'''

# beta is one, alpha_k varies, y_k varies, compute epsilon
n_alpha = 202
n_alpha_max = 101
n_yk = 101
yk_max = 10

alpha_k = np.linspace(0,n_alpha_max,n_alpha) #this range seems more than enough since we are likely to have priors below 1.
y_k = np.linspace(0,yk_max,n_yk) 
#this way, we have +- 2*10^4 data points. 
#We have numerical instability as eps-->1. perhaps just always cutoff eps at 1-1e-5?

#might need to add more data for eps near 1 as function grows increasingly nonlinear.

eps = np.zeros((n_alpha, n_yk))

for i in xrange(n_alpha):
	for j in xrange(n_yk):
		eps[i,j] = gdtr(alpha_k[i],1,y_k[j])


x = np.zeros((n_alpha*n_yk, 2))
x[:,0] = np.tile(alpha_k, n_yk)
x[:,1] = np.tile(y_k, n_alpha)

y = np.reshape(eps,(n_alpha*n_yk,1))

np.save('quantile_data_x.npy', x)
np.save('quantile_data_y.npy', y)



