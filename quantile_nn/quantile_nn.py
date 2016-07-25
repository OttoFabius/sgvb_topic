import numpy as np
import theano as th
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import OrderedDict

class quantile_nn:
    def __init__(self):


        self.learning_rate = 0.001
        self.batch_size = 10

        #structure
        self.nin = 2
        self.nout = 1
        self.nhidden = 100

        # --------------    Define model-specific dimensions    --------------------
        W1 = th.shared(np.random.normal(0,1./np.sqrt(self.nin)/5	,(self.nhidden, self.nin 	 )).astype(th.config.floatX), name = 'W1')
        b1 = th.shared(np.random.normal(0,1./np.sqrt(self.nin)/5 	,(self.nhidden, 1            )).astype(th.config.floatX), name = 'b1', broadcastable=(False,True))
		
        W2 = th.shared(np.random.normal(0,1./np.sqrt(self.nhidden)/5,(self.nout, self.nhidden )).astype(th.config.floatX), name = 'W2') 
        b2 = th.shared(np.random.normal(0,1./np.sqrt(self.nhidden)/5,(self.nout, 1            )).astype(th.config.floatX), name = 'b2',  broadcastable=(False,True))


        self.params = dict([('W1', W1), ('b1', b1),('W2', W2), ('b2', b2)])


        self.createGradientFunctions()

    def createGradientFunctions(self):
    	"""Symbolic definition of model, currently one hidden layer"""
        x = T.matrix(name='x')
        y = T.matrix(name='y')

        H1 = T.tanh(T.dot(self.params['W1'], x) + self.params['b1'])
        out = T.nnet.relu(T.dot(self.params['W2'], H1) + self.params['b2'])

        MSE = T.mean((out-y)**2)

        gradients = T.grad(MSE, self.params.values())

        updates = OrderedDict()

        for parameter, gradient in zip(self.params.values(), gradients):

            updates[parameter] = parameter - self.learning_rate * gradient
        
        self.update = th.function([x,y], [MSE], updates=updates, on_unused_input='ignore')

    def iterate(self, X,y):
        """Main method, slices data in minibatches and performs a training epoch. """

        N = X.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        X = X[idx,:]
        y = y[idx,:]

        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        MSE = 0
        for i in xrange(0,len(batches)-1):
            
            X_batch = X[batches[i]:batches[i+1]]
            y_batch = y[batches[i]:batches[i+1]]
            MSE_batch = self.update(X_batch.T, y_batch.T)
            MSE += MSE_batch[0]
            if MSE_batch[0]>1:
                print MSE_batch[0]


        return MSE/batches.shape[0]
