import numpy as np
import theano as th
import theano.tensor as T
import theano.sparse
from theano import ProfileMode, pp, printing
import scipy as sp
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt
from collections import OrderedDict
import cPickle as pickle
from scipy.sparse import csr_matrix, csc_matrix
from scipy.special import gammaln
from helpfuncs import select_half

def psi(x):
    "fifith order Euler-Maclaurin formula of psi function as gradient is not supported by theano"
    return T.log(x) - 1/(2*x) - 1/(12*x**2) + 1/(120*x**4) - 1/(256*x**6)

def beta_func(x,y):
    return (T.gamma(x)*T.gamma(y))/T.gamma(x+y)

def relu(x, alpha=0.01):
    return T.switch(x > 0, x, alpha * x) #leaky relu in 1-layer encoder for stability

def cap_logvar(logvar,cap):
    return T.switch(logvar<cap, logvar, cap)

def print_t(tensor, label):
    tensor.name = label
    tensor = theano.printing.Print(tensor.name, attrs=('__str__', 'shape'))(tensor)
    return tensor

class topic_model:
    def __init__(self, argdict):


        self.learning_rate = th.shared(argdict['learning_rate'])
        self.batch_size = argdict['batch_size']
        self.rp = argdict['rp']
        self.full_vocab = argdict['full_vocab']
        self.KLD_weight = argdict['kld_weight']
        self.ignore_logvar = argdict['ignore_logvar']

        self.logvar_cap = 3.
        self.lam = 0.
        self.N = argdict['trainset_size']
        self.prior_alpha = 1. #creates dirichlet process prior
        self.prior_beta = 1. #concentration parameter alpha0

        #structure
        self.dimZ = argdict['dimZ']
        self.HUe1 = argdict['HUe1']
        self.HUe2 = argdict['HUe2']
        self.HUe3 = argdict['HUe3']
        self.HUd1 = argdict['HUd1']
        self.HUd2 = argdict['HUd2']
        self.voc_size = argdict['voc_size']
        
        self.stickbreak = argdict['stickbreak']
        self.normalize_input = argdict['normalize_input']

        #Adam
        self.b1 = 0.1
        self.b2 = 0.001
        self.m = OrderedDict()
        self.v = OrderedDict()

        self.KLD_free = argdict['KLD_free']
        self.KLD_burnin = argdict['KLD_burnin']

        # np std is important, relates to normalization and such        ./np.sqrt(self.voc_size)       
        # --------------    Define model-specific dimensions    --------------------

        We1 = th.shared(np.random.normal(0,1,(argdict['HUe1'], self.voc_size)).astype(th.config.floatX), name = 'We1')
        be1 = th.shared(np.random.normal(0,1,(argdict['HUe1'],1)).astype(th.config.floatX), name = 'be1', broadcastable=(False,True))

        if self.HUe2==0:
            H_e_last = argdict['HUe1']

        elif self.HUe2!=0:
            We2 = th.shared(np.random.normal(0,1./np.sqrt(float(argdict['HUe1'])),(argdict['HUe2'], argdict['HUe1'])).astype(th.config.floatX), name = 'We2')
            be2 = th.shared(np.zeros((argdict['HUe2'],1)).astype(th.config.floatX), name = 'be2', broadcastable=(False,True))
           
            if self.HUe3==0:
                H_e_last = argdict['HUe2']
            elif self.HUe3!=0:

                We3 = th.shared(np.random.normal(0,1./np.sqrt(float(argdict['HUe2'])),(argdict['HUe3'], argdict['HUe2'])).astype(th.config.floatX), name = 'We2')
                be3 = th.shared(np.zeros((argdict['HUe3'],1)).astype(th.config.floatX), name = 'be2', broadcastable=(False,True))

                H_e_last = argdict['HUe3']

                
        We_mu = th.shared(np.random.normal(0,0.01/np.sqrt(float(H_e_last)),(self.dimZ,H_e_last)).astype(th.config.floatX), name = 'We_mu')
        be_mu = th.shared(np.zeros((self.dimZ,1)).astype(th.config.floatX), name = 'be_mu', broadcastable=(False,True))

        We_var = th.shared(np.random.normal(0,0.01/np.sqrt(float(H_e_last)),(self.dimZ, H_e_last)).astype(th.config.floatX), name = 'We_var')
        be_var = th.shared(np.zeros((self.dimZ,1)).astype(th.config.floatX), name = 'be_var', broadcastable=(False,True))


        
        if self.HUd1==0:
            Wd1 = th.shared(np.random.normal(0,0.01/np.sqrt(self.dimZ),(self.voc_size, self.dimZ)).astype(th.config.floatX), name = 'Wd1')
            bd1 = th.shared(np.zeros((self.voc_size,1)).astype(th.config.floatX), name = 'bd1', broadcastable=(False,True))
        elif self.HUd1!=0:        
            Wd1 = th.shared(np.random.normal(0,0.01/np.sqrt(self.dimZ),(argdict['HUd1'], self.dimZ)).astype(th.config.floatX), name = 'Wd1')
            bd1 = th.shared(np.zeros((argdict['HUd1'],1)).astype(th.config.floatX), name = 'bd1', broadcastable=(False,True))

            Wd2 = th.shared(np.random.normal(0,0.01/np.sqrt(float(argdict['HUd1'])),(self.voc_size, argdict['HUd1'])).astype(th.config.floatX), name = 'Wd2')
            bd2 = th.shared(np.zeros((self.voc_size,1)).astype(th.config.floatX), name = 'bd2', broadcastable=(False,True))

        if self.HUd2!=0:
            Wd2 = th.shared(np.random.normal(0,0.01/np.sqrt(float(argdict['HUd1'])),(argdict['HUd2'], argdict['HUd1'])).astype(th.config.floatX), name = 'Wd2')
            bd2 = th.shared(np.zeros((argdict['HUd2'],1)).astype(th.config.floatX), name = 'bd2', broadcastable=(False,True))

            Wd3 = th.shared(np.random.normal(0,0.01/np.sqrt(float(argdict['HUd2'])),(argdict['voc_size'], argdict['HUd2'])).astype(th.config.floatX), name = 'Wd3')
            bd3 = th.shared(np.zeros((argdict['voc_size'],1)).astype(th.config.floatX), name = 'bd3', broadcastable=(False,True))

        #initialize for stickbreaking
        self.z1                   = th.shared(np.zeros((self.batch_size)).astype(th.config.floatX))
        self.initial_stick_length = th.shared(np.zeros((self.batch_size)).astype(th.config.floatX))
        self.v_last               = th.shared(np.ones ((self.batch_size, 1)).T.astype(th.config.floatX))
        self.params = dict([('We1', We1), ('be1', be1), ('We_mu', We_mu), ('be_mu', be_mu), ('We_var', We_var), ('be_var', be_var), ('Wd1', Wd1), ('bd1', bd1)])
       


        if self.HUe2!=0:
            self.params.update(dict([('We2', We2), ('be2', be2)]))
        if self.HUe3!=0:
            self.params.update(dict([('We3', We3), ('be3', be3)]))
        if self.HUd1!=0:
            self.params.update(dict([('Wd2', Wd2), ('bd2', bd2)]))
        if self.HUd2!=0:
            self.params.update(dict([('Wd3', Wd3), ('bd3', bd3)]))


        if self.stickbreak==1:
            self.params_qnn = dict()
            names_qnn = pickle.load(open("qnn_names.pkl", "rb"))
            params_qnn = pickle.load(open("qnn_params.pkl", "rb"))

            for name, param in zip(names_qnn,params_qnn): 
                self.params_qnn[name] = param

        for key, value in self.params.items():

        # Init m and v for adam, and TODO: init gamma and beta for batch normalization for each layer
            if 'b' in key:
                self.m[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='m_' + key, broadcastable=(False,True))
                self.v[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='v_' + key, broadcastable=(False,True))
            else:
                self.m[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='m_' + key)
                self.v[key] = th.shared(np.zeros_like(value.get_value()).astype(th.config.floatX), name='v_' + key)

        self.createGradientFunctions()

    def createGradientFunctions(self):

        x       = th.sparse.csc_matrix(name='x', dtype=th.config.floatX)
        x_test  = th.sparse.csc_matrix(name='x_test', dtype=th.config.floatX)

        dl = T.dvector(name='dl')
        rest = T.matrix(name='rest')
        epoch = T.iscalar('epoch')
        unused_sum = T.scalar('unused_sum')

        srng = T.shared_randomstreams.RandomStreams()

        H_lin = th.sparse.dot(self.params['We1'], x) + self.params['be1']

        if self.rp==1:
            H_lin += rest

        H = relu(H_lin)

        if self.HUe2!=0:
            H2_lin = T.dot(self.params['We2'], H) + self.params['be2']
            H = relu(H2_lin)
        if self.HUe3!=0:
            H3_lin = T.dot(self.params['We3'], H) + self.params['be3']
            H = relu(H3_lin) 


        KLD_factor = T.maximum(0, T.minimum(1, (epoch-self.KLD_free)/self.KLD_burnin))

        if self.stickbreak==0:
            mu  = T.dot(self.params['We_mu'], H)  + self.params['be_mu']
            logvar = T.dot(self.params['We_var'], H) + self.params['be_var']
            logvar = cap_logvar(logvar, self.logvar_cap)

            eps = srng.normal((self.dimZ, self.batch_size), avg=0.0, std=1.0, dtype=theano.config.floatX)
            z = mu + T.exp(0.5*logvar)*eps
            if self.ignore_logvar==1:
                z = mu

            KLD   =  -self.KLD_weight*T.sum(T.sum(1 + logvar - mu**2 - T.exp(logvar), axis=0))

        if self.stickbreak==1:

            ############################       QNN Approach        #############################

            #                   NB need to remove We_var and be_var

            # alpha = T.nnet.relu(T.dot(self.params['We_mu'], H)  + self.params['be_mu']) #encoding of dirichlet distributed latent variables
            # eps = srng.uniform(size=(1, self.dimZ* self.batch_size), dtype=theano.config.floatX) #one uniformly distributed val for each Gamma distribution
            
            # alpha_vec = T.reshape(alpha, [1,self.dimZ*self.batch_size]) #not sure if this reshape works correctly yet!

            # qnn_input = T.concatenate([alpha_vec, eps], axis=0)

            # H1_qnn = T.nnet.relu(T.dot(self.params_qnn["W1"], qnn_input) + self.params_qnn["b1"])
            # H2_qnn = T.nnet.relu(T.dot(self.params_qnn["W2"], H1_qnn) + self.params_qnn["b2"])
            # gammavars = T.dot(self.params_qnn["W3"], H2_qnn) + self.params_qnn["b3"] #these are the gamma variables

            # z = gammavars/T.sum(gammavars,axis=1) #now we have the dirichlet distributed latent variables

            # z = T.reshape(z, (self.dimZ, self.batch_size))


            # summing over axis 0 is summing over latent variables, summing over 1 is summing over batch)
            # alpha_0 = self.dimZ * self.alpha_prior #calculate alpha_0 by summing over dir.lat.vars
            # beta_0 = T.sum(z, axis=0) #sum over priors for latent vars
            

            # KLD      =  T.sum(      T.gammaln(beta_0)           \
            #                       - T.sum(T.gammaln(z), axis=0) \
            #                       - T.gammaln(alpha_0)          \
            #                       + self.dimZ * (T.gammaln(self.alpha_prior))   \
            #                       + T.sum(    (z - self.alpha_prior) * (psi(z) - psi(beta_0))     , axis=0)       \
            #                       )




            #######################       Kumaraswami        ########################  


            a = T.nnet.softplus(T.dot(self.params['We_mu'], H)  + self.params['be_mu']) + 1e-5
            b = T.nnet.softplus(T.dot(self.params['We_var'], H)  + self.params['be_var']) + 1e-5
            eps = srng.uniform(size=(self.dimZ, self.batch_size), low=0.01, high=0.99, dtype=theano.config.floatX) 

            v = (1 - eps**(1/b))**(1/a)
            T.concatenate([v, self.v_last]) #Truncation. NB this makes one more latent dimension. 

            # Stick-breaking
            def stickbreak(v, z_prev, sl_used):
                z = v*(1-sl_used)
                new_sl_used = sl_used+z
                return z, new_sl_used
            
            out, updates = theano.scan(fn=stickbreak,
                                  outputs_info=[self.z1, self.initial_stick_length],
                                  sequences=[v] 
                                  )

            z, new_sl_used = out
            
            # KL Divergance of Beta and Kumaraswami

            KLD =  (1./(1+a*b))*beta_func(1./a, b)                       
            KLD += (1./(2+a*b))*beta_func(2./a, b)                              
            KLD += (1./(3+a*b))*beta_func(3./a, b) 
            KLD += (1./(4+a*b))*beta_func(4./a, b) 
            KLD += (1./(5+a*b))*beta_func(5./a, b) 
            KLD += (1./(6+a*b))*beta_func(6./a, b) 
            KLD += (1./(7+a*b))*beta_func(7./a, b) 
            KLD += (1./(8+a*b))*beta_func(8./a, b) 
            KLD += (1./(9+a*b))*beta_func(9./a, b)
            KLD *= (self.prior_beta-1)*b

            KLD += ((a - self.prior_alpha)/a) * (-0.57721 - psi(b)- 1/b)
            KLD += T.log(a*b+1e-5) + T.log(beta_func(self.prior_alpha, self.prior_beta)+1e-5)
            KLD += -(b-1)/(b+1e-5)

            KLD = T.sum(KLD, axis=0)
            KLD = KLD
            KLD = T.sum(KLD)
            KLD *= self.KLD_weight
            #################################################################################### 

        KLD_train = KLD*KLD_factor

        if self.HUd1==0:
            y_notnorm = T.exp(T.dot(self.params['Wd1'], z)  + self.params['bd1'])
        elif self.HUd1!=0:  
            H_d_lin = T.dot(self.params['Wd1'], z)  + self.params['bd1']
            H_d = relu(H_d_lin)
            if self.HUd2==0:
                y_notnorm = T.exp(T.dot(self.params['Wd2'], H_d)  + self.params['bd2'])
            elif self.HUd2!=0:
                H_d_lin = T.dot(self.params['Wd2'], H_d)  + self.params['bd2']
                H_d = relu(H_d_lin)
                y_notnorm = T.exp(T.dot(self.params['Wd3'], H_d)  + self.params['bd3'])


        y = y_notnorm/T.sum(y_notnorm, axis=0, keepdims=True)*(1-unused_sum)

        

        logy = T.log(y+1e-10)

        recon_err = T.sum(th.sparse.sp_sum(th.sparse.basic.mul(x, logy), axis=0)*dl)

        lowerbound_train = recon_err - KLD_train
        lowerbound = recon_err - KLD
        perplexity = th.sparse.sp_sum(th.sparse.basic.mul(x_test,logy))

        gradients = T.grad(lowerbound_train, self.params.values())

        ###################################################
        # Weight updates
        ###################################################

        updates = OrderedDict()

        gamma = T.sqrt(1 - (1 - self.b2) ** epoch)/(1 - (1 - self.b1)**epoch)

        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):

            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v            

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v + 1e-10))
            if 'Wd' in str(parameter):
                # MAP on weights (same as L2 regularization)
                updates[parameter] -= self.learning_rate * self.lam * (parameter * np.float32(self.batch_size / self.N))
            
            updates[m] = new_m
            updates[v] = new_v

        
        self.update      =  th.function([x, dl, rest, unused_sum, epoch], [lowerbound, recon_err, KLD, KLD_train], updates=updates, on_unused_input='ignore')#, mode=profmode)
        self.lowerbound  =  th.function([x, dl, rest, unused_sum, epoch], [lowerbound, recon_err, KLD], on_unused_input='ignore')
        self.perplexity  =  th.function([x, x_test, unused_sum]     , perplexity, on_unused_input='ignore')

    def encode(self, x, rest=None):
        """Helper function to compute the encoding of a datapoint or minibatch to z"""

        We1 = self.params["We1"].get_value() 
        be1 = self.params["be1"].get_value()      

        We_mu = self.params["We_mu"].get_value()
        be_mu = self.params["be_mu"].get_value()

        We_var = self.params["We_var"].get_value()
        be_var = self.params["be_var"].get_value()

        H = np.dot(We1, x) + be1 

        if type(rest)==np.ndarray:
            H = H+rest

        H[H<0] = 0

        if self.HUe2!=0:
            We2 = self.params["We2"].get_value() 
            be2 = self.params["be2"].get_value() 

            H =  np.dot(We2, H) + be2
            H[H<0] = 0

        if self.HUe3!=0:
            We3 = self.params["We3"].get_value() 
            be3 = self.params["be3"].get_value() 

            H =  np.dot(We3, H) + be3
            H[H<0] = 0

        mu  = np.dot(We_mu, H)  + be_mu
        logvar = np.dot(We_var, H) + be_var

        logvar[logvar>self.logvar_cap] = self.logvar_cap

        return mu, logvar

    def decode(self, mu, logvar):
        """Helper function to compute the decoding of a datapoint from z to x"""   
        z = np.random.normal(mu, (np.exp(0.5*logvar)))

        Wd1 = self.params["Wd1"].get_value()
        bd1 = self.params["bd1"].get_value()

        if self.HUd1==0:
            y_lin = np.dot(Wd1, z) + bd1 
        elif self.HUd1!=0:
            Wd2 = self.params["Wd2"].get_value()
            bd2 = self.params["bd2"].get_value()  

            H_d = np.dot(Wd1, z) + bd1 
            H_d[H_d<0] = 0

            if self.HUd2==0:
                y_lin = np.dot(Wd2, H_d)  + bd2

            elif self.HUd2!=0:
                Wd3 = self.params["Wd3"].get_value()
                bd3 = self.params["bd3"].get_value() 

                H_d = np.dot(Wd2, H_d) + bd2
                H_d[H_d<0] = 0.

                y_lin = np.dot(Wd3, H_d)  + bd3


        # y_lin[y_lin<-3]=-3
        # y_lin[y_lin> 3]= 3



        y_notnorm = np.exp(y_lin)


        return y_notnorm

    def iterate(self, X, dl, unused_sum, epoch, rest=None):
        """Main method, slices data in minibatches and performs a training epoch. """

        lowerbound = 0
        recon_err = 0
        KLD = 0
        KLD_train = 0
        progress = -1

        [N,dimX] = X.shape
        batches = np.arange(0, N, self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)
        rest_batch = np.zeros((self.batch_size, self.HUe1)) #K=HUe1 for now

        for i in xrange(0,len(batches)-2):
            X_batch = X[batches[i]:batches[i+1]]
            dl_batch = dl[batches[i]:batches[i+1]]
            if type(rest)==np.ndarray:
                rest_batch = rest[batches[i]:batches[i+1]].T

            lowerbound_batch, recon_err_batch, KLD_batch, KLD_train_batch  = self.update(X_batch.T, dl_batch, rest_batch, unused_sum, epoch)
               
            lowerbound += lowerbound_batch
            recon_err += recon_err_batch
            KLD += KLD_batch
            KLD_train += KLD_train_batch
            n_words = np.sum(dl)
        return lowerbound/n_words, recon_err/n_words, KLD/n_words, KLD_train/n_words

    def getLowerBound(self, data, dl, unused_sum, epoch, rest=None):
        """Use this method for example to compute lower bound on testset"""
        lowerbound = 0
        recon_err = 0
        KLD = 0

        [N,dimX] = data.shape

        batches = np.arange(0,N,self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        rest_batch = np.zeros((self.batch_size, self.HUe1))
        for i in xrange(0,len(batches)-1):
            if batches[i+1]<=N:
                X_batch = data[batches[i]:batches[i+1]]
                dl_batch = dl[batches[i]:batches[i+1]]
                if type(rest)==np.ndarray:
                    rest_batch = rest[batches[i]:batches[i+1]].T
                lb_batch, recon_batch, KLD_batch = self.lowerbound(X_batch.T, dl_batch, rest_batch, unused_sum, epoch)
            else:
                lb_batch, recon_batch = (0, 0) # doesnt work for non-batch_size :(
                                          
            lowerbound += lb_batch
            recon_err += recon_batch
            KLD += KLD_batch

        return lowerbound/np.sum(dl), recon_err/np.sum(dl), KLD/np.sum(dl)

    def calc_perplexity(self, argdict, data, unused_sum):

        data_seen, data_unseen = select_half(data)
        if argdict['normalize_input']==1:
            data_seen   = csc_matrix(data_seen  /csc_matrix.sum(data_seen  , 1))

        [N,dimX] = data.shape
        batches = np.arange(0, N, self.batch_size)
        if batches[-1] != N:
            batches = np.append(batches,N)

        rest_batch = np.zeros((self.batch_size, self.HUe1)) 

        perplexity = 0
        n_words = 0

        for i in xrange(0,len(batches)-1):
            batch_seen   = data_seen  [batches[i]:batches[i+1]]
            batch_unseen = data_unseen[batches[i]:batches[i+1]]

            perplexity_batch = self.perplexity(batch_seen.T, batch_unseen.T, unused_sum)
            n_words += csc_matrix.sum(batch_unseen)

            perplexity += perplexity_batch


        perp_per_word = perplexity/n_words

        return np.exp(-perp_per_word)