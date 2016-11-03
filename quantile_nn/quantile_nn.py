import numpy as np
import theano as th
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import OrderedDict

class quantile_nn:
    def __init__(self):


        self.learning_rate = 0.005
        self.batch_size = 1000

        #structure
        self.nin = 2
        self.nout = 1
        self.nhidden1 = 50
        self.nhidden2 = 50
    
        #r50 and 50 seems very good, 0.31 RMSE after 4 epochs

        # --------------    Define model-specific dimensions    --------------------
        W1 = th.shared(np.random.normal(0,1./np.sqrt(self.nin)	,(self.nhidden1, self.nin 	 )).astype(th.config.floatX), name = 'W1')
        b1 = th.shared(np.random.normal(0,1./np.sqrt(self.nin) 	,(self.nhidden1, 1            )).astype(th.config.floatX), name = 'b1', broadcastable=(False,True))
		
        W2 = th.shared(np.random.normal(0,1./np.sqrt(self.nhidden1),(self.nhidden2, self.nhidden1 )).astype(th.config.floatX), name = 'W2') 
        b2 = th.shared(np.random.normal(0,1./np.sqrt(self.nhidden1),(self.nhidden2, 1            )).astype(th.config.floatX), name = 'b2',  broadcastable=(False,True))

        W3 = th.shared(np.random.normal(0,1./np.sqrt(self.nhidden2),(self.nout, self.nhidden2 )).astype(th.config.floatX), name = 'W3') 
        b3 = th.shared(np.random.normal(0,1./np.sqrt(self.nhidden2),(self.nout, 1            )).astype(th.config.floatX), name = 'b3',  broadcastable=(False,True))

        self.params = dict([('W1', W1), ('b1', b1),('W2', W2), ('b2', b2), ('W3', W3), ('b3', b3)])

        self.createGradientFunctions()

    def createGradientFunctions(self):
    	"""Symbolic definition of model, currently two hidden layers"""

        x = T.matrix(name='x')
        y = T.dvector(name='y')

        H1 = T.nnet.relu(T.dot(self.params['W1'], x) + self.params['b1'])
        H2 = T.nnet.relu(T.dot(self.params['W2'], H1) + self.params['b2'])
        out = T.dot(self.params['W3'], H2) + self.params['b3']


        RMSE = T.mean(T.sqrt((out-y)**2))

        gradients = T.grad(RMSE, self.params.values())

        updates = OrderedDict()

        for parameter, gradient in zip(self.params.values(), gradients):

            updates[parameter] = parameter - self.learning_rate * gradient
        
        self.update = th.function([x,y], [RMSE, out], updates=updates, on_unused_input='ignore')

    def iterate(self, X, y):
        """Main method, slices data in minibatches and performs a training epoch. """

        N = X.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        X = X[idx,:]
        y = y[idx]

        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)
        
        out = np.array([])
        RMSE = 0
        for i in xrange(0,len(batches)-1):
            X_batch = X[batches[i]:batches[i+1]]
            y_batch = y[batches[i]:batches[i+1]]
            RMSE_batch, out_batch = self.update(X_batch.T, y_batch.T)
            RMSE += RMSE_batch
            out = np.append(out,out_batch)


        return RMSE/batches.shape[0], out

    def forward(self, x):
        """create output y from input X"""

        H1 = np.dot(self.params['W1'].get_value(), x.T) + self.params['b1'].get_value()
        H1[H1<0] = 0
        H2 = np.dot(self.params['W2'].get_value(), H1) + self.params['b2'].get_value()
        H2[H2<0] = 0
        y = np.dot(self.params['W3'].get_value(), H2) + self.params['b3'].get_value()
        return y
