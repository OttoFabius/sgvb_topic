import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import sys


"""
Contains various accelerated SGD methods for optimization
"""


class BaseOptimizer(object):

    """
    Base optimizer class
    """

    def __init__(self, objectives, objectives_eval, inputs, params, gradients=None, regularization='l2', normalization=None, weight_decay=0.,
                 batch_size=128):
        self.normalization = normalization
        self.regularization = regularization
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        # np.random.seed(12345)
        self.batches_test = 5

        if gradients is None:
            self.gradients = T.grad(
                T.sum(objectives), params, disconnected_inputs='warn', add_names=True)
        else:
            self.gradients = gradients
            if regularization is not None:
                print 'Warning! You already passed gradients and a regularizer. Make sure that the regulizer is \
                    not already precomputed.'

    def train(self, data, verbose=True, debug_funcs=[]):
        batches = np.arange(0, data[0].shape[0], self.batch_size)
        lb = 0
        for j in xrange(len(batches) - 1):
            print "start", time.time()
            inp = [d[batches[j]:batches[j + 1]].toarray()
            print "end", time.time()
                   if d.shape[0] > 0 else d for d in data]
            objectives = np.array(self.ascent(*inp))
            if np.isnan(objectives).any():
                print lb
                print objectives
                raise Exception('NaN objective!')
            lb += objectives
            if verbose:
                sys.stdout.write("\rBatch:{0}, Objectives:{1}, Total:{2}".format(
                    str(j + 1) + '/' + str(len(batches) - 1), str((objectives / self.batch_size).tolist()), str(lb.tolist())))
                sys.stdout.flush()
        if verbose:
            print

        return lb

    def evaluate(self, data):
        if self.batches_test == 1:
            return np.array(self.eval(*data))

        b_t = data[0].shape[0] / self.batches_test
        batches = np.arange(0, data[0].shape[0], b_t)
        lb = 0
        for j in xrange(len(batches) - 1):
            inp = [d[batches[j]:batches[j + 1]]
                   if d.shape[0] > 0 else d for d in data]
            objectives = np.array(self.eval(*inp))
            lb += objectives
        return lb

    def normalize_param(self, param, w_):
        if 'W' in param.name and self.normalization is not None:
            # print self.norm, 'normalization on', params[i].name
            if w_.ndim == 2:
                sum_over = [0]
            elif w_.ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
                sum_over = list(xrange(1, w_.ndim))

            if self.normalization == 'l2':
                w_new = w_ / \
                    T.sqrt(T.sum(T.sqr(w_), axis=sum_over, keepdims=True))
            elif self.normalization == 'l1':
                w_new = w_ / T.sum(np.abs(w_), axis=sum_over, keepdims=True)
            elif self.normalization == 'infinity':
                w_new = w_ / T.max(np.abs(w_), axis=sum_over, keepdims=True)
            elif self.normalization == 'nonorm':
                w_new = w_
            else:
                raise NotImplementedError()
        else:
            w_new = w_

        return w_new

    def get_updates_eval(self, params_inf, params, ma=True):
        """
        Keep a moving average of the parameters that will be used for evaluation
        """
        updates_eval = OrderedDict()

        itinf = theano.shared(0., name='itinf')
        updates_eval[itinf] = itinf + 1.
        fix3 = 1. - self.beta3**(itinf + 1.)

        for i in xrange(len(params)):
            avg = theano.shared(
                params_inf[i].get_value() * 0., name=params_inf[i].name + '_avg',
                broadcastable=params_inf[i].broadcastable)

            avg_new = self.beta3 * avg + (1. - self.beta3) * params[i]
            updates_eval[avg] = T.cast(avg_new, theano.config.floatX)
            updates_eval[params_inf[i]] = T.cast(avg_new / fix3, theano.config.floatX)

        return updates_eval


class AdaM(BaseOptimizer):

    """
    AdaM optimizer for an objective function
    """

    def __init__(self, objectives, objectives_eval, inputs, params, params_inf, gradients=None, regularization='l2', normalization=None, weight_decay=0.,
                 alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=128, polyak=True, beta3=0.5, **kwargs):

        super(AdaM, self).__init__(objectives, objectives_eval, inputs, params, gradients=gradients,
                                   regularization=regularization, normalization=normalization,
                                   weight_decay=weight_decay, batch_size=batch_size)

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.polyak = polyak
        self.beta3 = beta3
        self.epsilon = epsilon
        updates = self.get_updates(params, self.gradients)
        updates_eval = self.get_updates_eval(params_inf, params)

        # evaluate all the objectives and update parameters
        self.ascent = theano.function(
            inputs, objectives, updates=updates, on_unused_input='ignore', mode='FAST_RUN')  # , mode=run_mode)
        # evaluate all the objectives and (optionally) use a moving average for the parameters
        self.eval = theano.function(
            inputs, objectives_eval, updates=updates_eval, on_unused_input='ignore', mode='FAST_RUN')  # mode=run_mode)
        print 'AdaM', 'alpha:', alpha, 'beta1:', beta1, 'beta2:', beta2, 'epsilon:', self.epsilon, 'batch_size:', self.batch_size, \
            'normalization:', normalization, 'regularization:', regularization, 'weight_decay:', weight_decay, 'polyak:', polyak, 'beta3:', beta3

    def get_updates(self, params, grads):
        updates = OrderedDict()

        it = theano.shared(0., name='it')
        updates[it] = it + 1.

        fix1 = 1. - self.beta1**(it + 1.)  # To make estimates unbiased
        fix2 = 1. - self.beta2**(it + 1.)  # To make estimates unbiased

        for i in xrange(len(grads)):
            gi = grads[i]
            if self.regularization is not None:
                if self.regularization == 'l1':
                    gi -= self.weight_decay * T.sgn(params[i])
                elif self.regularization == 'l2':
                    gi -= self.weight_decay * params[i]

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = theano.shared(
                params[i].get_value() * 0., broadcastable=params[i].broadcastable)
            mom2 = theano.shared(
                params[i].get_value() * 0., broadcastable=params[i].broadcastable)

            # Update moments
            mom1_new = self.beta1 * mom1 + (1. - self.beta1) * gi
            mom2_new = self.beta2 * mom2 + (1. - self.beta2) * T.sqr(gi)

            # Compute the effective gradient
            corr_mom1 = mom1_new / fix1
            corr_mom2 = mom2_new / fix2
            effgrad = corr_mom1 / (T.sqrt(corr_mom2) + self.epsilon)

            # Do update
            w_ = params[i] + self.alpha * effgrad
            # Apply normalization
            w_new = self.normalize_param(params[i], w_)

            # Apply update
            updates[params[i]] = T.cast(w_new, theano.config.floatX)
            updates[mom1] = T.cast(mom1_new, theano.config.floatX)
            updates[mom2] = T.cast(mom2_new, theano.config.floatX)

        return updates


class AdaMax(BaseOptimizer):

    """
    AdaMax optimizer for an objective function (variant of Adam based on infinity norm)
    """

    def __init__(self, objectives, objectives_eval, inputs, params, params_inf, gradients=None, regularization='l2', normalization='l2', weight_decay=0.,
                 alpha=0.002, beta1=0.9, beta2=0.999, batch_size=128, polyak=True, beta3=0.9, **kwargs):

        super(AdaMax, self).__init__(objectives, objectives_eval, inputs, params, gradients=gradients,
                                     regularization=regularization, normalization=normalization,
                                     weight_decay=weight_decay, batch_size=batch_size)

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.polyak = polyak
        self.beta3 = beta3
        updates = self.get_updates(params, self.gradients)
        updates_eval = self.get_updates_eval(params_inf, params)

        # evaluate all the objectives and update parameters
        self.ascent = theano.function(
            inputs, objectives, updates=updates, on_unused_input='ignore', mode='FAST_RUN')  # , mode=run_mode)
        # evaluate all the objectives and do not update the parameters
        self.eval = theano.function(
            inputs, objectives_eval, updates=updates_eval, on_unused_input='ignore', mode='FAST_RUN')  # mode=run_mode)
        print 'AdaMax', 'alpha:', alpha, 'beta1:', beta1, 'beta2:', beta2, 'batch_size:', self.batch_size, \
            'normalization:', normalization, 'regularization:', regularization, 'weight_decay:', weight_decay, 'polyak:', polyak, 'beta3:', beta3

    def get_updates(self, params, grads):
        updates = OrderedDict()

        it = theano.shared(0., name='it')
        updates[it] = it + 1.

        for i in xrange(len(grads)):
            gi = grads[i]
            if self.regularization is not None:
                if self.regularization == 'l1':
                    gi -= self.weight_decay * T.sgn(params[i])
                elif self.regularization == 'l2':
                    gi -= self.weight_decay * params[i]

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = theano.shared(
                params[i].get_value() * 0., broadcastable=params[i].broadcastable)
            mom2 = theano.shared(
                params[i].get_value() * 0., broadcastable=params[i].broadcastable)

            # Update moments
            mom1_new = self.beta1 * mom1 + (1. - self.beta1) * gi
            mom2_new = T.maximum(self.beta2 * mom2, np.abs(gi))

            # Compute the effective gradient
            # To make estimates unbiased
            corr_mom1 = mom1_new / (1. - self.beta1 ** (it + 1.))
            effgrad = corr_mom1 / mom2_new

            # Do update
            w_ = params[i] + self.alpha * effgrad
            # Apply normalization
            w_new = self.normalize_param(params[i], w_)

            # Apply update
            updates[params[i]] = T.cast(w_new, theano.config.floatX)
            updates[mom1] = T.cast(mom1_new, theano.config.floatX)
            updates[mom2] = T.cast(mom2_new, theano.config.floatX)

        return updates


class NesterovMomentum(BaseOptimizer):

    """
    SGD updates with Nesterov Momentum
    """

    def __init__(self, objectives, objectives_eval, inputs, params, params_inf, gradients=None, regularization='l2', normalization='l2', weight_decay=0.,
                 alpha=0.001, momentum=0.9, batch_size=128, polyak=True, beta3=0.9, **kwargs):

        super(NesterovMomentum, self).__init__(objectives, objectives_eval, inputs, params, gradients=gradients,
                                               regularization=regularization, normalization=normalization,
                                               weight_decay=weight_decay, batch_size=batch_size)
        self.alpha = alpha
        self.momentum = momentum
        self.polyak = polyak
        self.beta3 = beta3
        updates = self.get_updates(params, self.gradients)
        updates_eval = self.get_updates_eval(params_inf, params)

        # evaluate all the objectives and update parameters
        self.ascent = theano.function(
            inputs, objectives, updates=updates, on_unused_input='ignore', mode='FAST_RUN')  # , mode=run_mode)
        # evaluate all the objectives and do not update the parameters
        self.eval = theano.function(
            inputs, objectives_eval, updates=updates_eval, on_unused_input='ignore', mode='FAST_RUN')  # mode=run_mode)
        print 'NesterovMomentum', 'alpha:', alpha, 'momentum:', momentum, 'batch_size:', self.batch_size, \
            'normalization:', normalization, 'regularization:', regularization, 'weight_decay:', weight_decay, 'polyak:', polyak, 'beta3:', beta3

    def get_updates(self, params, grads):
        updates = OrderedDict()

        for i in xrange(len(grads)):
            gi = grads[i]
            if self.regularization is not None:
                if self.regularization == 'l1':
                    gi -= self.weight_decay * T.sgn(params[i])
                elif self.regularization == 'l2':
                    gi -= self.weight_decay * params[i]

            new_param = params[i] - self.alpha * gi
            value = params[i].get_value(borrow=True)
            velocity = theano.shared(
                np.zeros(value.shape, dtype=value.dtype), broadcastable=params[i].broadcastable)

            x = self.momentum * velocity + new_param - params[i]

            # Do update
            w_ = self.momentum * x + new_param
            # Apply normalization
            w_new = self.normalize_param(params[i], w_)

            # Apply update
            updates[params[i]] = w_new
            updates[velocity] = x

        return updates
