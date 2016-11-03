import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma, gdtr

''' Gamma distribution cumulative density function.

Returns the integral from zero to x of the gamma probability density function:

a**b / gamma(b) * integral(t**(b-1) exp(-at), t=0..x). In our case, with beta = 1, we can write this as:

1/gamma(alpha) * integral(t**(alpha-1) exp-t), t=0..x).

The arguments a and b are used differently here than in other definitions. 

It matches the wikipedia definition, with alpha and beta flipped.'''

def cdf(alpha, t, eps_max):
	'''t must be linspaced'''
	cumsum = np.array(gdtr(1, alpha, t[0]))
	i = 0
	while (np.max(cumsum)<eps_max) and (i<len(t)-1):
		i+=1
		cumsum = np.append(cumsum, gdtr(1, alpha, t[i]))
	return cumsum

# beta is one, alpha_k varies, y_k varies, compute epsilon
n_alpha = 22
alpha_min = 1e-2
alpha_max = 2
n_yk = 404
yk_min = 3e-3

alpha_k = np.logspace(np.log10(alpha_min), np.log10(alpha_max),n_alpha) #this range seems more than enough since we are likely to have priors below 1.
y_k_general = np.linspace(yk_min,500,1e4) 
 

eps = np.zeros((n_alpha* n_yk))
eps_max = 1-1e-3

for i in xrange(n_alpha):
	print i
	cumsum = cdf(alpha_k[i], y_k_general, eps_max) 
	yk_max = y_k_general[len(cumsum)] 
	
	yk = np.linspace(yk_min, yk_max, n_yk)
	eps[i*n_yk:(i+1)*n_yk] = cdf(alpha_k[i], yk, 1)

x = np.zeros((n_alpha*n_yk, 2))
x[:,0] = np.repeat(alpha_k, n_yk)
x[:,1] = eps

y = np.tile(yk, n_alpha)


np.save('quantile_data_small_x.npy', x)
np.save('quantile_data_small_y.npy', y)


plt.plot(y[:], x[:,1],'.', linewidth = 0.4)
plt.show()


