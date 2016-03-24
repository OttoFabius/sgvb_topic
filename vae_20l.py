import numpy as np
import theano as th
import theano.tensor as T
import theano.sparse
import scipy as sp
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import OrderedDict
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
from scipy.special import gammaln


def relu(x):
    return T.switch(x<0,0,x)
        
class topic_model_20layer:
    def __init__(self, argdict):

        self.dimZ = argdict['dimZ']
        self.learning_rate = th.shared(argdict['learning_rate'])
        self.batch_size = argdict['batch_size']
        sigmaInit = argdict['sigmaInit']
        self.voc_size = argdict['voc_size']
        self.e = 1e-4

        We1 = th.shared(np.random.normal(0,1./np.sqrt(self.voc_size),(argdict['HUe1'], self.voc_size)).astype(th.config.floatX), name = 'We1')
        be1 = th.shared(np.random.normal(0,1,(argdict['HUe1'],1)).astype(th.config.floatX), name = 'be1', broadcastable=(False,True))

        We2 = th.shared(np.random.normal(0,1./np.sqrt(float(argdict['HUe1'])),(argdict['HUe2'], argdict['HUe1'])).astype(th.config.floatX), name = 'We1')
        be2 = th.shared(np.random.normal(0,1,(argdict['HUe2'],1)).astype(th.config.floatX), name = 'be1', broadcastable=(False,True))

        We_mu = th.shared(np.random.normal(0,1./np.sqrt(float(argdict['HUe2'])),(self.dimZ,argdict['HUe2'])).astype(th.config.floatX), name = 'We_mu')
        be_mu = th.shared(np.random.normal(0,1,(self.dimZ,1)).astype(th.config.floatX), name = 'be_mu', broadcastable=(False,True))

        We_var = th.shared(np.random.normal(0,1./np.sqrt(float(argdict['HUe2'])),(self.dimZ, argdict['HUe2'])).astype(th.config.floatX), name = 'We_var')
        be_var = th.shared(np.random.normal(0,1,(self.dimZ,1)).astype(th.config.floatX), name = 'be_var', broadcastable=(False,True))

        Wd1 = th.shared(np.random.normal(0,1./np.sqrt(self.dimZ),(self.voc_size, self.dimZ)).astype(th.config.floatX), name = 'Wd1')
        bd1 = th.shared(np.random.normal(0,1,(self.voc_size,1)).astype(th.config.floatX), name = 'bd1', broadcastable=(False,True))


        self.params = OrderedDict([('We1', We1), ('be1', be1), ('We2', We2), ('be2', be2), ('We_mu', We_mu), ('be_mu', be_mu),  \
            ('We_var', We_var), ('be_var', be_var), ('Wd1', Wd1), ('bd1', bd1)])

        # Adam
        self.b1 = 0.1
        self.b2 = 0.001
        self.m = OrderedDict()
        self.v = OrderedDict()

        self.KLD_free = argdict['KLD_free']
        self.KLD_burnin = argdict['KLD_burnin']



        for key,value in self.params.items():
            if 'b' in key:
                self.m[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='m_' + key, broadcastable=(False,True))
                self.v[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='v_' + key, broadcastable=(False,True))
            else:
                self.m[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='m_' + key)
                self.v[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='v_' + key)


        self.createGradientFunctions()


    def createGradientFunctions(self):

        x = th.sparse.csc_matrix(name='x', dtype=th.config.floatX)
        epoch = T.iscalar('epoch')

        srng = T.shared_randomstreams.RandomStreams()

        H1_lin = th.sparse.dot(self.params['We1'], x) + self.params['be1']
        H1 = relu(H1_lin)

        H2_lin = T.dot(self.params['We2'], H1) + self.params['be2']
        H2 = relu(H2_lin)

        mu  = T.dot(self.params['We_mu'], H2)  + self.params['be_mu']
        logvar = T.dot(self.params['We_var'], H2) + self.params['be_var']


        eps = srng.normal((self.dimZ, self.batch_size), avg=0.0, std=1.0, dtype=theano.config.floatX)
        z = mu + T.exp(0.5*logvar)*eps

        y_notnorm = T.exp(-T.dot(self.params['Wd1'], z)  - self.params['bd1'])
        y = y_notnorm/T.sum(y_notnorm, axis=0)

        KLD_factor = T.minimum(1,T.maximum(0, (epoch - self.KLD_free)/self.KLD_burnin))
        KLD      =  -T.sum(T.sum(1 + logvar - mu**2 - T.exp(logvar), axis=0)/(theano.sparse.basic.sp_sum(x, axis=0)+self.e))
        KLD_train = KLD*KLD_factor


        recon_err =  T.sum(theano.sparse.basic.sp_sum(x*T.log(y), axis=0)/theano.sparse.basic.sp_sum(x, axis=0))

        lowerbound_train = recon_err - KLD_train
        lowerbound = recon_err - KLD

        gradients = T.grad(lowerbound_train, self.params.values())

        ###################################################
        # Weight updates
        ###################################################

        updates = OrderedDict()

        gamma = T.sqrt(1 - (1 - self.b2) ** epoch)/(1 - (1 - self.b1)**epoch)

        i=0
        # Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            i+=1
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v
            
            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + 1e-20)

            updates[m] = new_m
            updates[v] = new_v


        self.update = th.function([x, epoch], [lowerbound, recon_err, KLD, KLD_train], updates=updates)
        self.lowerbound  = th.function([x, epoch], [lowerbound, recon_err], on_unused_input='ignore')

    def encode(self, x):
        """Helper function to compute the encoding of a datapoint or minibatch to z"""


        We1 = self.params["We1"].get_value() 
        be1 = self.params["be1"].get_value()      

        We2 = self.params["We2"].get_value() 
        be2 = self.params["be2"].get_value()  

        We_mu = self.params["We_mu"].get_value()
        be_mu = self.params["be_mu"].get_value()

        We_var = self.params["We_var"].get_value()
        be_var = self.params["be_var"].get_value()

        H1 = np.dot(We1, x) + be1
        H1[H1<0] = 0

        H = np.dot(We2, H1) + be2
        H[H<0] = 0

        mu  = np.dot(We_mu, H)  + be_mu
        logvar = np.dot(We_var, H) + be_var

        return mu, logvar

    def decode(self, mu, logvar):
        """Helper function to compute the decoding of a datapoint from z to x"""

        Wd1 = self.params["Wd1"].get_value()
        bd1 = self.params["bd1"].get_value()

        z = np.random.normal(mu, np.exp(logvar))

        y_lin = np.dot(Wd1, z)  + bd1
        y_notnorm = np.exp(-y_lin)


        return y_notnorm

    def calculate_perplexity(self, doc, selected_features=None, means=None, seen_words=0.5, samples=1):

        doc = np.array(doc.todense())
        
        if selected_features!=None:
            doc_selected = doc[selected_features] #new
        else:
            doc_selected = doc

        doc_seen = np.zeros_like(doc)

        cs = np.cumsum(doc)

        samp = np.random.choice(np.arange(cs[-1]), np.floor(cs[-1]*seen_words), replace=False)
        for word_no in samp:
            word_index = np.argmax(cs>word_no)
            doc_seen[word_index]+=1
            
        log_perplexity_doc_vec = 0
        total_lambda = 0

        doc_unseen = doc - doc_seen
        mu, logvar = self.encode(doc_seen)
        y_notnorm = self.decode(mu, logvar)

        if selected_features!=None:
            if means!=None:
                mult_params_naive = means
            else:
                mult_params = np.zeros_like(doc)
                mult_params[selected_features] = y_notnorm/np.sum(y_notnorm, axis=0)
        else:
            mult_params = y_notnorm/np.sum(y_notnorm, axis=0)
        n_words = np.sum(doc_unseen)
        t3 = np.sum(doc_unseen*np.log(mult_params))
        t4 = gammaln(n_words+1)
        t6 = -np.sum(gammaln(doc_unseen+1))
        log_perplexity_doc = t3# + t4 + t6

        return log_perplexity_doc, n_words

    def iterate(self, X, epoch):
        """Main method, slices data in minibatches and performs a training epoch. Returns LB for whole dataset
            added a progress print during an epoch (comment/uncomment line 164)"""

        lowerbound = 0
        recon_err = 0
        KLD = 0
        KLD_train = 0
        progress = -1

        [N,dimX] = X.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            

            X_batch = X[batches[i]:batches[i+1]]

            lowerbound_batch, recon_err_batch, KLD_batch, KLD_train_batch = self.update(X_batch.T, epoch)

            lowerbound += lowerbound_batch
            recon_err += recon_err_batch
            KLD += KLD_batch
            KLD_train += KLD_train_batch

        return lowerbound, recon_err, KLD, KLD_train

    def getLowerBound(self,data,epoch):
        """Use this method for example to compute lower bound on testset"""
        lowerbound = 0
        recon_err = 0
        [N,dimX] = data.shape

        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-1):
            if batches[i+1]<N:
                miniBatch = data[batches[i]:batches[i+1]]
                lb_batch, recon_batch = self.lowerbound(miniBatch.T, epoch)
            else:
                lb_batch, recon_batch = (0, 0) #function doesnt work for non-batch_size :(

            lowerbound += lb_batch
            recon_err += recon_batch

        return lowerbound, recon_err