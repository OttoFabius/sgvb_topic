import numpy as np

fname = 'results/vae_own/ny/increase/50k/'
lb 		= np.load(fname+'lowerbound.npy')
lb_test = np.load(fname+'lowerbound_test.npy')
perp	= np.load(fname+'perplexity.npy')

print np.max(lb), np.max(lb_test), lb_test[-1], np.min(perp), perp[-1]