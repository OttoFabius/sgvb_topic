import numpy as np
import matplotlib.pyplot as plt

n_words = 467714
fname = 'simple_1'
fname2 = 'simple_2'
lb = np.load('results/lowerbounds/'+fname+'/lowerbound.npy')
lb2= np.load('results/lowerbounds/'+fname2+'/lowerbound.npy')
plt.plot(np.log(xrange(len(lb))),lb/n_words)
plt.plot(np.log(xrange(len(lb2))),lb2/n_words)
plt.legend(('1','2'))
plt.show()