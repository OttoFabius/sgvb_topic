import theano
import theano.tensor as T
import numpy as np
import theano.sparse
import scipy as sp


"""
Contains various Encoders and Decoders (objects that manipulate Theano functions)
that can be used for model building
"""

srng = T.shared_randomstreams.RandomStreams(seed=12345)
low = 3
high = 8


def change_random_seed(seed):
    global srng
    srng = T.shared_randomstreams.RandomStreams(seed=seed)

'''
Nonlinear functions
'''
relu = lambda x: T.switch(x < 0, 0, x)
relu2 = lambda x: T.switch(x < 0, 0, x) + 0.01 * x
logcosh = lambda x: T.log(T.cosh(x))
cubic = lambda x: x**3
softsign = lambda x: x / (1. + np.abs(x))
linear = lambda x: x
randrelu = lambda x: T.switch(
    x < 0, x / srng.uniform(size=(1,), low=low, high=high), x)
randreluinf = lambda x: T.switch(x < 0, 2 * x / (low + high), x)
randtanh = lambda x: T.tanh(x * srng.uniform(size=x.shape, low=low, high=high))
randtanhinf = lambda x: T.tanh(2 * x / (low + high))
nonlinearities = {'tanh': T.tanh, 'sigmoid': T.nnet.sigmoid, 'softmax': T.nnet.softmax, 'softplus': T.nnet.softplus,
                  'relu': relu, 'relu2': relu2, 'linear': linear, 'logcosh': logcosh, 'cubic': cubic, 'softsign': softsign,
                  'randrelu': randrelu, 'randreluinf': randreluinf, 'randtanh': randtanh, 'randtanhinf': randtanhinf}

'''
Weight normalization functions
'''
l2norm = lambda W: W / T.sqrt(T.sum(W**2, axis=0, keepdims=True))
l1norm = lambda W: W / T.sum(np.abs(W), axis=0, keepdims=True)
infinitynorm = lambda W: W / T.max(np.abs(W), axis=0, keepdims=True)
normalizations = {'l2': l2norm, 'l1': l1norm, 'infinity': infinitynorm}


class MLP(object):

    def __init__(self, input_params, hidden_layers, nonlin='softplus', normalization='l2', dropout_rate=0.):
        """
        Deterministic MLP that takes a number of inputs and returns the last hidden layer

        inputs: List of inputs
            [x0, x1, ...]
        input_params : List of parameters for the inputs
            [[W0, s0], [W1, s1], ...., [b]]
        hidden_layers : List of hidden layer params
            [[W_h1, s_h1, b_h1], [W_h2, s_h2, b_h2], ...]
        nonlin : nonlinearity to use
        """
        self.input_params = input_params
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.nonlin = nonlin

        # get the corresponding activation
        self.activation = nonlinearities[nonlin]
        if 'rand' in nonlin:
            self.activation_inf = nonlinearities[nonlin + 'inf']
        else:
            self.activation_inf = nonlinearities[nonlin]
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def ff_inputs(self, inputs, input_params=None, inference=False, sparse=False, i=0):
        """
        Feedforward just the inputs
        """
        if input_params is None:
            input_params = self.input_params
        # do the dot product of the inputs
        lin_dot = T.as_tensor_variable(0)
        drop_rate = max(0., self.dropout_rate - 0.3)
        for inp, params in zip(inputs, input_params[:-1]):
            i +=1
            W = self.normalize(
                params[0]) * T.exp(params[1]) if len(params) > 1 else params[0]
            if drop_rate > 0:
                if not inference:
                    inp *= srng.binomial(size=inp.shape, n=1,
                                        p=(1. - drop_rate)).astype(theano.config.floatX)
                else:
                    inp *= (1. - drop_rate)

            if sparse & i==1:
                lin_dot += theano.sparse.basic.dot(inp, W)
            else:
                lin_dot +=T.dot(inp, W)

        # add the bias
        lin_dot += input_params[-1][0]
        if inference:
            h = [self.activation_inf(lin_dot)]
        else:
            h = [self.activation(lin_dot)]
        # return output
        return h

    def ff_hidden_layers(self, h, start=0, inference=False):
        """
        Continue to the remaining hidden layers
        """
        # feedforward from the remaining layers
        if len(self.hidden_layers[start:]) > 0:
            for i, layer in enumerate(self.hidden_layers):
                W = self.normalize(
                    layer[0]) * T.exp(layer[1]) if len(layer) > 2 else layer[0]
                hin = h[-1]
                if inference:
                    if self.dropout_rate > 0:
                        hin *= (1. - self.dropout_rate)
                    h_ = self.activation_inf(T.dot(hin, W) + layer[-1])
                else:
                    if self.dropout_rate > 0:
                        hin *= srng.binomial(size=hin.shape, n=1, p=(1. - self.dropout_rate)).astype(
                            theano.config.floatX)
                    h_ = self.activation(T.dot(hin, W) + layer[-1])
                h.append(h_)
        # return output
        return h

    def ff(self, inputs, inference=False, sparse=False):
        """
        Wrapper for the full feedforward process
        """
        h = self.ff_inputs(inputs, inference=inference, sparse=sparse)
        h2 = self.ff_hidden_layers(h, inference=inference)
        return h2


class DiagGaussianEncoder(MLP):

    """
    Encoder that maps inputs to a latent Gaussian distribution
    """

    def __init__(self, input_params, hidden_layers, latent_layer, batch_size=1, nonlin='softplus',
                 normalization='l2', L=1, dropout_rate=0., prior_mu=0., prior_sg=1.):

        super(DiagGaussianEncoder, self).__init__(input_params, hidden_layers,
                                                  nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.latent_layer = latent_layer
        self.batch_size = batch_size
        if normalization is not None:
            self.normalize = normalizations[normalization]
        self.prior_mu = prior_mu
        self.prior_sg = prior_sg

    def transform(self, inputs, constrain_means=False, inference=False, sparse=False):
        # ff from the deterministic MLP
        hf = self.ff(inputs, inference=inference, sparse=sparse)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if inference:
                hout *= (1. - self.dropout_rate)
            else:
                hout *= srng.binomial(size=hout.shape, n=1,
                                      p=(1. - self.dropout_rate)).astype(theano.config.floatX)
        # get the weights for the mean and logvariance
        Wmu = self.normalize(self.latent_layer[0][
                             0]) * T.exp(self.latent_layer[0][1]) if len(self.latent_layer[0]) > 2 else self.latent_layer[0][0]
        Wstd = self.normalize(self.latent_layer[1][
                              0]) * T.exp(self.latent_layer[1][1]) if len(self.latent_layer[1]) > 2 else self.latent_layer[1][0]

        # estimate the parameters
        mu_z = T.dot(hout, Wmu) + self.latent_layer[0][-1]
        if constrain_means:
            mu_z = T.nnet.sigmoid(mu_z)
        std_z = T.nnet.softplus(T.dot(hout, Wstd) + self.latent_layer[1][-1])
        return mu_z, std_z

    def sample(self, mu_z, std_z):
        return mu_z + std_z * srng.normal(mu_z.shape).astype(theano.config.floatX)

    def kldivergence(self, mu_z, std_z):
        return self.kldivergence_per_x(mu_z, std_z).sum()

    def kldivergence_per_x(self, mu_z, std_z):
        return 0.5 * T.sum(1 - T.log(self.prior_sg**2) + T.log(std_z**2) -
                           ((mu_z - self.prior_mu)**2 + std_z**2) / self.prior_sg**2, axis=1)

    def kldivergence_givenp(self, mu_zq, std_zq, mu_zp, std_zp):
        return self.kldivergence_givenp_per_x(mu_zq, std_zq, mu_zp, std_zp).sum()

    def kldivergence_givenp_per_x(self, mu_zq, std_zq, mu_zp, std_zp):
        return 0.5 * T.sum(1 - T.log(std_zp**2) + T.log(std_zq**2) -
                           ((mu_zq - mu_zp)**2 + std_zq**2) / std_zp**2, axis=1)

    def logp_perx(self, sample):
        return -.5 * (T.log(2 * np.pi) + T.log(self.prior_sg**2) + (sample - self.prior_mu)**2 / self.prior_sg**2).sum(axis=1)

    def logp(self, sample):
        return T.sum(self.logp_perx(sample))

    def logq_perx(self, sample, mu_z, std_z):
        return -.5 * (T.log(2 * np.pi) + T.log(std_z**2) + ((sample - mu_z)**2) / (std_z**2)).sum(axis=1)

    def logq(self, sample, mu_z, std_z):
        return T.sum(self.logq_perx(sample, mu_z, std_z))


class DiagGaussianDecoder(MLP):

    """
    Stochastic decoder that maps inputs to a Gaussian distribution with diagonal covariance
    """

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus',
                 normalization='l2', L=1, dropout_rate=0.):

        super(DiagGaussianDecoder, self).__init__(input_params, hidden_layers,
                                                  nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, constrain_means=False, inference=False, mult=1., add=0.):
        # ff from the deterministic mlp
        if len(self.input_params) > 0:
            hf = self.ff(inputs, inference=inference)
            hout = hf[-1]
            if self.dropout_rate > 0:
                if inference:
                    hout *= (1. - self.dropout_rate)
                else:
                    hout *= srng.binomial(size=hout.shape, n=1,
                                          p=(1. - self.dropout_rate)).astype(theano.config.floatX)
        else:
            hout = inputs[0]
            if self.dropout_rate > 0:
                drop_rate = 0.  # max(0., self.dropout_rate - 0.3)
                if inference:
                    hout *= (1. - drop_rate)
                else:
                    hout *= srng.binomial(size=hout.shape, n=1,
                                          p=(1. - drop_rate)).astype(theano.config.floatX)

        # get the weights for the mean and logvariance
        Wmu = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[
            0][1]) if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        Wstd = self.normalize(self.reconstruction_layer[1][0]) * T.exp(self.reconstruction_layer[
            1][1]) if len(self.reconstruction_layer[1]) > 2 else self.reconstruction_layer[1][0]
        # get the mean and variances of logp(x | inputs)
        mu_x = T.dot(hout, Wmu) + self.reconstruction_layer[0][-1]
        if constrain_means:
            mu_x = mult * T.nnet.sigmoid(mu_x) + add
        std_x = T.nnet.softplus(
            T.dot(hout, Wstd) + self.reconstruction_layer[1][-1])

        return mu_x, std_x

    def logp(self, x, mu_x, std_x):
        return T.sum(-.5 * (T.log(2 * np.pi) + T.log(std_x**2) + ((x - mu_x)**2) / (std_x**2)).sum(axis=1))

    def logp_per_x(self, x, mu_x, std_x):
        return -.5 * (T.log(2 * np.pi) + T.log(std_x**2) + ((x - mu_x)**2) / (std_x**2)).sum(axis=1)

    def kldivergence(self, mu_x, std_x, prior_mu_x, prior_std_x):
        return T.sum(self.kldivergence_perx(mu_x, std_x, prior_mu_x, prior_std_x))

    def kldivergence_perx(self, mu_x, std_x, prior_mu_x, prior_std_x):
        return 0.5 * T.sum(1 - T.log(prior_std_x**2) + T.log(std_x**2) -
                        ((mu_x - prior_mu_x)**2 + std_x**2) / prior_std_x**2, axis=1)

    def sample(self, mu_x, std_x):
        return mu_x + std_x * srng.normal(mu_x.shape).astype(theano.config.floatX)


class BernoulliDecoder(MLP):

    """
    Stochastic decoder that maps inputs to a Bernoulli distribution
    """

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus',
                 normalization='l2', dropout_rate=0.):
        super(BernoulliDecoder, self).__init__(input_params, hidden_layers,
                                               nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, inference=False):
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if inference:
                hout *= (1. - self.dropout_rate)
            else:
                hout *= srng.binomial(size=hout.shape, n=1,
                                      p=(1. - self.dropout_rate)).astype(theano.config.floatX)

        # get the mean of the bernoulli
        Wp = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[
            0][1]) if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        p = T.nnet.sigmoid(T.dot(hout, Wp) + self.reconstruction_layer[0][-1])
        return [p]

    def logp(self, x, p):
        return -T.nnet.binary_crossentropy(p, x).sum()

    def logp_per_x(self, x, p):
        return -T.nnet.binary_crossentropy(p, x).sum(axis=1)

    def sample(self, p):
        return srng.binomial(size=p.shape, n=1, p=p).astype(theano.config.floatX)


class PoissonDecoder(MLP):

    """
    Stochastic decoder that maps inputs to a Poisson distribution
    """

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus',
                 normalization='l2', dropout_rate=0.):
        super(PoissonDecoder, self).__init__(input_params, hidden_layers,
                                             nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, inference=False):
        # ff through the MLP
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if inference:
                hout *= (1. - self.dropout_rate)
            else:
                hout *= srng.binomial(size=hout.shape, n=1,
                                      p=(1. - self.dropout_rate)).astype(theano.config.floatX)

        # get the log(mean/variance) of the Poisson
        Wloglambda = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[
            0][1]) if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        lamba = T.nnet.softplus(
            T.dot(hout, Wloglambda) + self.reconstruction_layer[0][-1])
        return [lamba]

    def logp(self, x, lamba):
        # return T.sum((-lamba + x*T.log(lamba) - T.gammaln(x +
        # 1)).sum(axis=1))
        # logGamma(x+1) is constant w.r.t. the optimization
        return T.sum((-lamba + x * T.log(lamba)).sum(axis=1))

    def logp_per_x(self, x, lamba):
        # return T.sum((-lamba + x*T.log(lamba) - T.gammaln(x +
        # 1)).sum(axis=1))
        # logGamma(x+1) is constant w.r.t. the optimization,
        # do not compute for faster evaluations
        return (-lamba + x * T.log(lamba)).sum(axis=1)

    def sample(self, lamba):
        return srng.poisson(size=lamba.shape, lam=lamba, dtype='int64')


class NegativeBinomialDecoder(MLP):

    """
    Stochastic decoder that maps inputs to a Negative-Binomial distribution
        (more appropriate for overdispersion)
    """

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus',
                 normalization='l2', dropout_rate=0.):
        super(NegativeBinomialDecoder, self).__init__(input_params, hidden_layers,
                                                      nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, inference=False):
        # ff through the MLP
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if inference:
                hout *= (1. - self.dropout_rate)
            else:
                hout *= srng.binomial(size=hout.shape, n=1,
                                      p=(1. - self.dropout_rate)).astype(theano.config.floatX)

        # get the p and logr of the binomial
        Wp = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[
            0][1]) if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        Wlogr = self.normalize(self.reconstruction_layer[1][0]) * T.exp(self.reconstruction_layer[
            1][1]) if len(self.reconstruction_layer[1]) > 2 else self.reconstruction_layer[1][0]

        p = T.nnet.sigmoid(T.dot(hout, Wp) + self.reconstruction_layer[0][-1])
        r = T.nnet.softplus(
            T.dot(hout, Wlogr) + self.reconstruction_layer[1][-1])

        return p, r

    def logp(self, x, p, r):
        # return T.sum(T.gammaln(x + r) - T.gammaln(x + 1) - T.gammaln(r) +
        # x*T.log(p) + r*T.log(1. - p), axis=1).sum()
        # logGamma(x+1) is constant
        return T.sum(T.gammaln(x + r) - T.gammaln(r) + x * T.log(p) + r * T.log(1. - p), axis=1).sum()

    def sample(self, p, r):
        pass


class DirichletDecoder(MLP):

    """
    Stochastic decoder that maps inputs to a Dirichlet distribution
    """

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus',
                 normalization='l2', dropout_rate=0.):
        super(DirichletDecoder, self).__init__(input_params, hidden_layers,
                                               nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]

    def transform(self, inputs, inference=False):
        hf = self.ff(inputs, inference=inference)
        hout = hf[-1]
        if self.dropout_rate > 0:
            if inference:
                hout *= (1. - self.dropout_rate)
            else:
                hout *= srng.binomial(size=hout.shape, n=1,
                                      p=(1. - self.dropout_rate)).astype(theano.config.floatX)

        # get the alphas of the Dirichlet
        Wa = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[
            0][1]) if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        a_ = T.nnet.softplus(
            T.dot(hout, Wa) + self.reconstruction_layer[0][-1])
        a = T.clip(a_, 1e-8, 1. - 1e-8)
        return [a]

    def logp(self, x, a):
        return self.logp_per_x(x, a).sum()

    def logp_per_x(self, x, a):
        return T.gammaln(a.sum(axis=1)) - T.sum(T.gammaln(a), axis=1) + T.sum((a - 1.) * T.log(x), axis=1)

    def sample(self, p):
        pass


class CategoricalDecoder(MLP):

    """
    Stochastic decoder that maps inputs to a Categorical distribution
    """

    def __init__(self, input_params, hidden_layers, reconstruction_layer, nonlin='softplus',
                 normalization='l2', dropout_rate=0., inference=False):
        super(CategoricalDecoder, self).__init__(input_params, hidden_layers,
                                                 nonlin=nonlin, normalization=normalization, dropout_rate=dropout_rate)
        self.reconstruction_layer = reconstruction_layer
        if normalization is not None:
            self.normalize = normalizations[normalization]
        self.softmax = nonlinearities['softmax']

    def transform(self, inputs, inference=False):
        # ff from the deterministic mlp
        if len(self.input_params) > 0:
            hf = self.ff(inputs, inference=inference)
            hout = hf[-1]
            if self.dropout_rate > 0:
                if inference:
                    hout *= (1. - self.dropout_rate)
                else:
                    hout *= srng.binomial(size=hout.shape, n=1,
                                          p=(1. - self.dropout_rate)).astype(theano.config.floatX)
        else:
            hout = inputs[0]
            if self.dropout_rate > 0:
                drop_rate = 0.  # max(0., self.dropout_rate - 0.3)
                if inference:
                    hout *= (1. - drop_rate)
                else:
                    hout *= srng.binomial(size=hout.shape, n=1,
                                          p=(1. - drop_rate)).astype(theano.config.floatX)

        W = self.normalize(self.reconstruction_layer[0][0]) * T.exp(self.reconstruction_layer[
            0][1]) if len(self.reconstruction_layer[0]) > 2 else self.reconstruction_layer[0][0]
        # get the probabilities of each category
        ps = (
            self.softmax(T.dot(hout, W) + self.reconstruction_layer[0][-1]) + 1e-10)

        return [ps]

    def logp(self, x, ps):
        # idx = T.eq(x, 1).nonzero()[1]
        return T.sum(T.log(ps)[T.arange(x.shape[0]), x])

    def logp_per_x(self, x, ps):
        idx = T.eq(x, 1).nonzero()[1]
        return T.log(ps)[T.arange(x.shape[0]), idx]

    def entr(self, ps):
        return - T.sum(ps * T.log(ps), axis=1).sum()

    def kldivergence(self, prior, ps):
        return - T.sum(ps * (T.log(prior) - T.log(ps)), axis=1).sum()

    def most_probable(self, ps):
        return T.argmax(ps, axis=1)

    def sample(self, ps):
        return srng.multinomial(size=ps.shape[0], n=1, pvals=ps).astype(theano.config.floatX)
