import numpy as np
import theano as th
from loadsave import load_pickle_list # , load_pickle
from model_list import model

x, d = load_pickle_list()
voc_size = d[1].size


print "initializing model + graph..."
model = model(voc_size)

epoch = 0

while True:
    epoch += 1
    lowerbound = model.iterate(x, d, epoch)
    print 'epoch ', epoch, 'with lowerbound = ', lowerbound/np.sum(d)
    
