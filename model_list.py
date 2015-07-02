import numpy as np
import theano as th
import theano.tensor as T
import theano.sparse
import scipy as sp
from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict
import cPickle as pickle


class topic_model:
    def __init__(self, voc_size, dimZ, HU_dz, HU_qx, HU_qd, learning_rate, sigmaInit, L=1):
        """NB dimensions of HU_qx and HU_qd have to match if they merge"""

        self.dimZ = dimZ
        self.learning_rate = th.shared(learning_rate)
        self.L = L

        # define number of hidden units in each hidden layer if hidden layers exist

        # define vocabulary size here?

        # initialize weights, biases

        W_dh = th.shared(np.random.normal(0,sigmaInit,(HU_dz, voc_size)).astype(th.config.floatX), name = 'W_dh')
        b_dh = th.shared(np.random.normal(0,sigmaInit,(HU_dz,1)).astype(th.config.floatX), name = 'b_dh', broadcastable=(False,True))

        W_d_mu = th.shared(np.random.normal(0,sigmaInit,(dimZ,HU_dz)).astype(th.config.floatX), name = 'W_d_mu')
        b_d_mu = th.shared(np.random.normal(0,sigmaInit,(dimZ,1)).astype(th.config.floatX), name = 'b_d_mu', broadcastable=(False,True))

        W_d_var = th.shared(np.random.normal(0,sigmaInit,(dimZ, HU_dz)).astype(th.config.floatX), name = 'W_d_var')
        b_d_var = th.shared(np.random.normal(0,sigmaInit,(dimZ,1)).astype(th.config.floatX), name = 'b_d_var', broadcastable=(False,True))

        W_zx = th.shared(np.random.normal(0,sigmaInit,(voc_size, dimZ)).astype(th.config.floatX), name = 'W_zx')
        b_zx = th.shared(np.random.normal(0,sigmaInit,(voc_size,1)).astype(th.config.floatX), name = 'b_zx', broadcastable=(False,True))

        W_xz_q = th.shared(np.random.normal(0,sigmaInit,(HU_qx, voc_size)).astype(th.config.floatX), name = 'W_xz_q')
        b_xz_q = th.shared(np.random.normal(0,sigmaInit,(HU_qx,1)).astype(th.config.floatX), name = 'b_xz_q', broadcastable=(False,True))

        W_dz_q = th.shared(np.random.normal(0,sigmaInit,(HU_qd, voc_size)).astype(th.config.floatX), name = 'W_dz_q')
        b_dz_q = th.shared(np.random.normal(0,sigmaInit,(HU_qd,1)).astype(th.config.floatX), name = 'b_dz_q', broadcastable=(False,True))

        W_q_mu = th.shared(np.random.normal(0,sigmaInit,(dimZ,HU_qd)).astype(th.config.floatX), name = 'W_q_mu')
        b_q_mu = th.shared(np.random.normal(0,sigmaInit,(dimZ,1)).astype(th.config.floatX), name = 'b_q_mu', broadcastable=(False,True))

        W_q_var = th.shared(np.random.normal(0,sigmaInit,(dimZ, HU_qd)).astype(th.config.floatX), name = 'W_q_var')
        b_q_var = th.shared(np.random.normal(0,sigmaInit,(dimZ,1)).astype(th.config.floatX), name = 'b_q_var', broadcastable=(False,True))

        self.params = OrderedDict([('W_dh', W_dh), ('b_dh', b_dh), ('W_d_mu', W_d_mu), ('b_d_mu', b_d_mu),  \
            ('W_d_var', W_d_var), ('b_d_var', b_d_var), ('W_zx', W_zx), ('b_zx', b_zx),  \
            ('W_xz_q', W_xz_q), ('b_xz_q', b_xz_q), ('W_dz_q', W_dz_q), ('b_dz_q', b_dz_q), \
            ('W_q_mu', W_q_mu), ('b_q_mu', b_q_mu), ('W_q_var', W_q_var), ('b_q_var', b_q_var)])

        # Adam
        self.b1 = 0.05
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

        doc_size = T.iscalar("doc_size")
        x = th.sparse.csr_matrix(name='x', dtype=th.config.floatX)
        d = T.dcol("d")     #dimensions (?,1), broadcastable in 2nd dimension. Optionally a sparse matrix of (V x 1)?

        seed = 20
        srng = T.shared_randomstreams.RandomStreams(seed=seed)

        # log p(z|d). One layer to hidden should be fine as we have few documents
        H_lin = T.dot(self.params['W_dh'], d) + self.params['b_dh']
        H = H_lin * (H_lin>0.)

        mu_pzd  = T.dot(self.params['W_d_mu'], H)  + self.params['b_d_mu']
        logvar_pzd = T.dot(self.params['W_d_var'], H) + self.params['b_d_var']

        # Encoder
        # Should probably add extra hidden layer for x->z at some point because of the large amount of data x
        # 1xhidden
        H_dz_lin = T.dot(self.params['W_dz_q'], d) + self.params['b_dz_q']
        # batchxhidden
        H_xz_lin = th.sparse.dot(self.params['W_xz_q'], x) + self.params['b_xz_q']
        # H_xz_lin = T.dot(self.params['W_xz_q'], x) + self.params['b_xz_q']

        # batch x hidden
        H_q_lin = H_dz_lin + H_xz_lin
        H_q = H_q_lin * (H_q_lin > 0) 

        # batch x dimZ
        mu_q = T.dot(self.params['W_q_mu'], H_q) + self.params['b_q_mu']
        logvar_q = T.dot(self.params['W_q_var'], H_q) + self.params['b_q_var']

        eps = srng.normal((self.dimZ, doc_size), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        z = mu_q+ T.exp(0.5*logvar_q)*eps

        # decoder. NB only one layer now
        # if self.L != 1:
        #     for i in xrange(1,self.L):
        #         eps = srng.normal((self.dimZ, doc_size), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        #         z += mu_q + T.exp(0.5*logvar_q)*eps
        #     z = z/self.L
            
        y_lin = T.dot(self.params['W_zx'], z) + self.params['b_zx']
        y = T.nnet.softmax(y_lin.T).T # use custom version if the dimensions are flipped?

        # define lowerbound 
        # NB need to account for including doc specific prior for every word, can in part be done by broadcasting
        KLD = - 0.5 * self.dimZ * doc_size                          \
            + 0.5 * T.sum(                                          \
              T.exp(logvar_q - logvar_pzd)                          \
            + T.pow((mu_q - mu_pzd), 2) / (T.exp(logvar_q))         \
            + logvar_pzd  - logvar_q) #broadcast logvar_pzd everywhere


        recon_err_mat_sparse = x * T.log(y)
        recon_err_mat = th.sparse.dense_from_sparse(recon_err_mat_sparse)
        recon_err = T.sum(recon_err_mat)
        lowerbound = (recon_err - KLD)

        gradients = T.grad(lowerbound, self.params.values())

        ###################################################
        # Weight updates
        ###################################################

        updates = OrderedDict()

        epoch = T.iscalar('epoch')

        gamma = T.sqrt(1 - (1 - self.b2) ** epoch)/(1 - (1 - self.b1)**epoch)

        # Adam
        for parameter, gradient, m, v in zip(self.params.values(), gradients, self.m.values(), self.v.values()):
            new_m = self.b1 * gradient + (1 - self.b1) * m
            new_v = self.b2 * (gradient**2) + (1 - self.b2) * v
       
            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + 1e-100) 

            updates[m] = new_m
            updates[v] = new_v


        self.update = th.function([x, d, doc_size, epoch], [lowerbound, KLD, recon_err_mat], updates=updates)
        self.lowerbound  = th.function([x, d, doc_size]  , lowerbound)
        self.encode = th.function([x, d, doc_size], mu_q, on_unused_input = 'ignore')


    def iterate(self, data_x, data_d, epoch):
        """Main method, slices data in minibatches and performs a training epoch. Returns LB for whole dataset
            added a progress print during an epoch (comment/uncomment line 164)"""

        lowerbound = 0
        progress = -1
        for i in xrange(len(data_x)):
            x = data_x[i] #sparse
            d = data_d[i]
            doc_size = x.shape[0]
            lowerbound_document, KLD, errs = self.update(x.T, d, doc_size, epoch)
            lowerbound += lowerbound_document
            # print y
            # print '--------------------------------------------------'
            # print np.sum(y, axis = 0)
            # print y.shape
            # raw_input()
            # print x
            error_terms = errs[errs !=0]
            print '---------------------------------------------------------------------------------------------------'
            print "doc nr:", i, "doc lb: ", lowerbound_document/doc_size, "doc size: ", doc_size
            print "max error: ", np.min(errs), "min err: ", np.max(errs[errs !=0]), "pct_bad: ", int([-8.84>=i for i in error_terms].count(True))/int(doc_size)
            if lowerbound_document/doc_size<= (-9.0):
                print error_terms

            # raw_input()
            # if progress != int(50.*i/len(data_x)):
            #     print '='*int(50.*i/len(data_x))+'>'
            #     progress = int(50.*i/len(data_x))



        return lowerbound #NB need to divide by number of words
  
    def save_parameters(self, path):
        """Saves all the parameters in a way they can be retrieved later - not adapted for current model yet!"""
        pickle.dump([name for name in self.params.keys()], open(path + "/names.pkl", "wb"))
        pickle.dump([p.get_value() for p in self.params.values()], open(path + "/params.pkl", "wb"))
        pickle.dump([m.get_value() for m in self.m.values()], open(path + "/m.pkl", "wb"))
        pickle.dump([v.get_value() for v in self.v.values()], open(path + "/v.pkl", "wb"))

    def load_parameters(self, path):
        """Load the variables in a shared variable safe way - not adapted for current model yet!"""
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

