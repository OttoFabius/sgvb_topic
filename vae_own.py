import numpy as np
import theano as th
import theano.tensor as T
import theano.sparse
import scipy as sp
from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict
import cPickle as pickle


class topic_model:
    def __init__(self, voc_size, dimZ, HUe1, HUd1, learning_rate, sigmaInit, batch_size, only_trainset, HUe2=0, HUd2=0):
        """NB dimensions of HU_qx and HU_qd have to match if they merge"""

        self.dimZ = dimZ
        self.learning_rate = th.shared(learning_rate)
        self.batch_size = batch_size


        # initialize weights, biases

        We1 = th.shared(np.random.normal(0,sigmaInit,(HUe1, voc_size)).astype(th.config.floatX), name = 'We1')
        be1 = th.shared(np.random.normal(0,sigmaInit,(HUe1,1)).astype(th.config.floatX), name = 'be1', broadcastable=(False,True))

        We_mu = th.shared(np.random.normal(0,sigmaInit,(dimZ,HUe1)).astype(th.config.floatX), name = 'We_mu')
        be_mu = th.shared(np.random.normal(0,sigmaInit,(dimZ,1)).astype(th.config.floatX), name = 'be_mu', broadcastable=(False,True))

        We_var = th.shared(np.random.normal(0,sigmaInit,(dimZ, HUe1)).astype(th.config.floatX), name = 'We_var')
        be_var = th.shared(np.random.normal(0,sigmaInit,(dimZ,1)).astype(th.config.floatX), name = 'be_var', broadcastable=(False,True))

        Wd1 = th.shared(np.random.normal(0,sigmaInit,(HUd1, dimZ)).astype(th.config.floatX), name = 'Wd1')
        bd1 = th.shared(np.random.normal(0,sigmaInit,(HUd1,1)).astype(th.config.floatX), name = 'bd1', broadcastable=(False,True))

        Wd2 = th.shared(np.random.normal(0,sigmaInit,(voc_size, HUd1)).astype(th.config.floatX), name = 'Wd2')
        bd2 = th.shared(np.random.normal(0,sigmaInit,(voc_size,1)).astype(th.config.floatX), name = 'bd2', broadcastable=(False,True))


        self.params = OrderedDict([('We1', We1), ('be1', be1), ('We_mu', We_mu), ('be_mu', be_mu),  \
            ('We_var', We_var), ('be_var', be_var), ('Wd1', Wd1), ('bd1', bd1), ('Wd2', Wd2), ('bd2', bd2)])

        # Adam
        self.b1 = 0.1
        self.b2 = 0.001
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key,value in self.params.items():
            if 'b' in key:
                self.m[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='m_' + key, broadcastable=(False,True))
                self.v[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='v_' + key, broadcastable=(False,True))
            else:
                self.m[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='m_' + key)
                self.v[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='v_' + key)

        self.createGradientFunctions()
        print "compiled Gradient functions!"

    def createGradientFunctions(self):
        """ Defines optimization criterion and creates symbolic gradient function"""
        # voc x batch
        x = th.sparse.csc_matrix(name='x', dtype=th.config.floatX)


        srng = T.shared_randomstreams.RandomStreams()

        H_lin = th.sparse.dot(self.params['We1'], x) + self.params['be1']
        H = T.nnet.softplus(H_lin)


        mu  = T.dot(self.params['We_mu'], H)  + self.params['be_mu']
        logvar = T.dot(self.params['We_var'], H) + self.params['be_var']


        eps = srng.normal((self.dimZ, self.batch_size), avg=0.0, std=1.0, dtype=theano.config.floatX)
        z = mu + T.exp(0.5*logvar)*eps

        H_d = T.nnet.softplus(T.dot(self.params['Wd1'], z)  + self.params['bd1'])
        # y=lambda of Poisson
        y = T.nnet.softplus(T.dot(self.params['Wd2'], H_d)  + self.params['bd2'])

        # define lowerbound 
        KLD = - T.sum(T.sum(1 + logvar - mu**2 - T.exp(logvar), axis=0))

        recon_err = T.sum(T.sum(-y + x * T.log(y), axis=0))

        lowerbound = recon_err - KLD

        gradients = T.grad(lowerbound, self.params.values())

        ###################################################
        # Weight updates
        ###################################################

        updates = OrderedDict()

        epoch = T.iscalar('epoch')

        gamma = T.sqrt(1 - (1 - self.b2) ** epoch)/(1 - (1 - self.b1)**epoch)

        i=0
        # Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            i+=1
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v
            
            if i%2 == 0:
                updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + 1e-20) 
            else:
                parameter_norm = parameter / T.sqrt(T.sum(T.sqr(parameter), axis=1, keepdims=True))
                updates[parameter] = parameter_norm + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + 1e-20)
                

            updates[m] = new_m
            updates[v] = new_v


        self.update = th.function([x, epoch], [lowerbound, recon_err, KLD, y], updates=updates)
        self.lowerbound  = th.function([x], lowerbound)



    def iterate(self, X, epoch):
        """Main method, slices data in minibatches and performs a training epoch. Returns LB for whole dataset
            added a progress print during an epoch (comment/uncomment line 164)"""

        lowerbound = 0
        recon_err = 0
        KLD = 0
        progress = -1

        [N,dimX] = X.shape
        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        for i in xrange(0,len(batches)-2):
            

            X_batch = X[batches[i]:batches[i+1]]
           
            lowerbound_doc, recon_err_doc, KLD_doc, y = self.update(X_batch.T, epoch)
            lowerbound += lowerbound_doc
            recon_err += recon_err_doc
            KLD += KLD_doc
            # if progress != int(50.*i/len(data_x)):
            #     print '='*int(50.*i/len(data_x))+'>'
            #     progress = int(50.*i/len(data_x))

        return lowerbound, recon_err, KLD

    def encode(self, x):
        """Helper function to compute the encoding of a datapoint to z"""
        h = np.zeros((self.hidden_units_encoder,1))

        We1 = self.params["We1"].get_value() 
        Wb1 = self.params["Wb1"].get_value()      

        We_mu = self.params["We_mu"].get_value()
        be_mu = self.params["be_mu"].get_value()

        We_var = self.params["We_var"].get_value()
        be_var = self.params["be_var"].get_value()


        H_lin = np.dot(We1, x) + be1
        H = np.log(1 + np.exp(H_lin)) #softplus

        mu  = np.dot(We_mu, H)  + be_mu
        logvar = T.dot(We_var, H) + be_var


        return mu, logvar

    def decode(self, mu, logvar):
        """Helper function to compute the decoding of a datapoint from z to x"""

        Wd1 = self.params["Wd1"].get_value()
        bd1 = self.params["bd1"].get_value()

        Wd2 = self.params["Wd2"].get_value()
        bd2 = self.params["bd2"].get_value()

        z = np.random.normal(mu, np.exp(logvar))

        H_d_lin = np.dot(Wd1, z)  + bd1

        y = T.nnet.softplus(T.dot(self.params['Wd2'], H_d)  + self.params['bd2'])

        x = np.zeros((t_steps+1,self.features))

        W_zh = self.params['W_zh'].get_value()
        b_zh = self.params['b_zh'].get_value()

        W_hhd = self.params['W_hhd'].get_value()
        b_hhd = self.params['b_hhd'].get_value()

        W_xhd = self.params['W_xhd'].get_value()
        b_xhd = self.params['b_xhd'].get_value()

        W_hx = self.params['W_hx'].get_value()
        b_hx = self.params['b_hx'].get_value()

        h = W_zh.dot(z) + b_zh

        for t in xrange(t_steps):
            h = np.tanh(W_hhd.dot(h) + b_hhd + W_xhd.dot(x[t,:,np.newaxis]) + b_xhd)
            x[t+1,:] = np.squeeze(1 /(1 + np.exp(-(W_hx.dot(h) + b_hx))))
        
        return x[1:,:]

    def save_parameters(self, path):
        """Saves parameters"""
        pickle.dump([name for name in self.params.keys()], open(path + "/names.pkl", "wb"))
        pickle.dump([p.get_value() for p in self.params.values()], open(path + "/params.pkl", "wb"))
        pickle.dump([m.get_value() for m in self.m.values()], open(path + "/m.pkl", "wb"))
        pickle.dump([v.get_value() for v in self.v.values()], open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Loads parameters, restarting training is possible afterwards"""
        names = pickle.load(open(path + "/names.pkl", "rb"))
        params = pickle.load(open(path + "/params.pkl", "rb"))

        for name,param in zip(names,params): 
            self.params[name].set_value(param)
        
        m_list = pickle.load(open(path + "/m.pkl", "rb"))
        v_list = pickle.load(open(path + "/v.pkl", "rb"))

        for name,m in zip(names,m_list): 
            self.m[name].set_value(m)

        for name,v in zip(names,v_list): 
            self.v[name].set_value(v)
