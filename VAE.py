import theano
import theano.tensor as T
import numpy as np
import blocks as blk
import generate_params as gpm
from optimizer import AdaM, AdaMax, NesterovMomentum
import time

"""
Object that contains a variational auto-encoder as example

Possible parameter values:
   nonlinearity: tanh, sigmoid, softplus (default),
              relu, relu2 (leaky ReLU), linear, logcosh, cubic, softsign,
              randrelu, randtanh
   normalization: l2 (default), l1, nonorm (no normalization), infinity (unstable)
   regularization: l2 (default), l1, uniform (no regularization)
   L: number of samples, 1 is enough for most applications
   type_rec: binary, diag_gaussian, poisson, negative-binomial
   type_latent: diag_gaussian
   iterations: number of epochs
   dropout_rate: 0. (values in [0, 1) are acceptable)
   algo_optim (optimization algorithm): adam (default), adamax, nesterov
   polyak: Boolean for Polyak-Ruppert like averaging
   beta3: weight for recent updates for Polyak Ruppert averaging (values in [0,1])
   name_log: location that will be used as a log file
   seed: seed for the random number generators
"""


def log_f(file_, string):
    print string
    with open(file_, "a") as myfile:
        myfile.write("\n" + string)


class VAE(object):
    """
    Variational Autoencoder that implements a simple latent variable model
        p(x, z) = p(z)p(x|z)
    with q(z|x) being the variational posterior.

    The final objective function is
        L = E_q(z|x)[logp(x|z)] - KL(q(z|x)||p(z))
    """
    def __init__(self, N, dim_x, dim_h_en_z=[50], dim_h_de_x=[50], dim_z=50, batch_size=100,
                nonlinearity='softplus', normalization='l2', regularization='l2', L=1,
                type_rec='poisson', type_latent='gaussian', iterations=100, constrain_means=False,
                dropout_rate=0., weight_decay=None, learningRate=0.01, mode='FAST_RUN',
                algo_optim='adam', polyak=False, beta3=0.1, name_log='log_vae.txt', seed=12345, save_params=False):
        self.dim_x = dim_x
        self.dim_h_en_z = dim_h_en_z
        self.dim_h_de_x = dim_h_de_x
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.nonlinearity = nonlinearity
        self.normalization = normalization
        self.regularization = regularization
        self.learningRate = learningRate
        self.normed = False
        self.save_params = save_params
        if self.normalization is not None:
            self.normed = True
        if weight_decay is None:
            self.weight_decay = float(batch_size) / N
        else:
            self.weight_decay = weight_decay
        self.L = L

        self.algo_optim = algo_optim
        if self.algo_optim == 'adam':
            self._optimizer = AdaM
        elif self.algo_optim == 'adamax':
            self._optimizer = AdaMax
        elif self.algo_optim == 'nesterov':
            self._optimizer = NesterovMomentum
        else:
            raise Exception()
        self.polyak = polyak
        self.beta3 = beta3

        self.type_rec = type_rec
        if self.type_rec == 'binary':
            self.decoder = blk.BernoulliDecoder
        elif self.type_rec == 'diag_gaussian':
            self.decoder = blk.DiagGaussianDecoder
        elif self.type_rec == 'poisson':
            self.decoder = blk.PoissonDecoder
        elif self.type_rec == 'negative-binomial':
            self.decoder = blk.NegativeBinomialDecoder
        else:
            raise NotImplementedError()

        self.type_latent = type_latent
        if self.type_latent == 'gaussian':
            self.encoder = blk.DiagGaussianEncoder
        else:
            raise NotImplementedError()

        # initialize random seeds
        blk.change_random_seed(seed)
        gpm.change_random_seed(seed)

        self.iterations = iterations
        self.constrain_means = constrain_means
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.name_log = name_log

        log_f(self.name_log, 'VAE, dim_h_en_z: ' + str(dim_h_en_z) + ', dim_h_de_x: ' + str(dim_h_de_x) +
            ', dim_z: ' + str(dim_z) + ', nonlinearity: ' + str(nonlinearity) + ', type_rec: ' + str(type_rec) +
            ', type_latent: ' + str(type_latent) + ', dropout: ' + str(dropout_rate) + ', normalization: ' + str(normalization) +
            ', L: ' + str(self.L) + ', weight_decay: ' + str(self.weight_decay) + ', algo_optim: ' + algo_optim + ', random_seed:' + str(seed))

        self._build_blocks()
        self._create_model()

    def copy_params(self, params1, params2, params3):
        """
        Copy the parameters so as to later use a form of Polyak averaging for the evaluation model
        """
        params1_inf = [[theano.shared(param.get_value(), name=param.name + '_avg') for param in params] for params in params1]
        params2_inf = [[theano.shared(param.get_value(), name=param.name + '_avg') for param in params] for params in params2]
        params3_inf = [[theano.shared(param.get_value(), name=param.name + '_avg') for param in params] for params in params3]

        return params1_inf, params2_inf, params3_inf

    def _build_blocks(self):
        """
        Create the encoders and decoders according to a specific topology of the network
        """
        # create the encoder for z q(z|x)
        in_encoder_params_z = gpm.create_input_layer(dim_in=[self.dim_x], dim_h0=self.dim_h_en_z[0], normed=self.normed, appended_name='_en_z')
        if len(self.dim_h_en_z) > 1:
            hidden_layers_en_z = gpm.append_hidden_layers(dim_in=self.dim_h_en_z[0], dim_h=self.dim_h_en_z[1:], normed=self.normed, appended_name='_en_z')
        else:
            hidden_layers_en_z = []
        z_layer = gpm.create_output_layer(dim_in=self.dim_h_en_z[-1], dim_out=[self.dim_z, self.dim_z], normed=self.normed, appended_name='_en_z')
        self.encoder_z = self.encoder(in_encoder_params_z, hidden_layers_en_z, z_layer, nonlin=self.nonlinearity, normalization=self.normalization,
                    dropout_rate=self.dropout_rate)
        in_encoder_params_z_inf, hidden_layers_en_z_inf, z_layer_inf = self.copy_params(in_encoder_params_z, hidden_layers_en_z, z_layer)
        self.encoder_z_inf = self.encoder(in_encoder_params_z_inf, hidden_layers_en_z_inf, z_layer_inf, nonlin=self.nonlinearity, normalization=self.normalization,
                                dropout_rate=self.dropout_rate)

        # create the decoder for p(x|z)
        in_decoder_params_x = gpm.create_input_layer(dim_in=[self.dim_z], dim_h0=self.dim_h_de_x[0], normed=self.normed, appended_name='_de_x')
        if len(self.dim_h_de_x) > 1:
            hidden_layers_de_x = gpm.append_hidden_layers(dim_in=self.dim_h_de_x[0], dim_h=self.dim_h_de_x[1:], normed=self.normed, appended_name='_de_x')
        else:
            hidden_layers_de_x = []
        if self.type_rec in ['diag_gaussian', 'negative-binomial']:
            reconstruction_layer_x = gpm.create_output_layer(dim_in=self.dim_h_de_x[-1], dim_out=[self.dim_x, self.dim_x], normed=self.normed, appended_name='_de_x')
        else:
            reconstruction_layer_x = gpm.create_output_layer(dim_in=self.dim_h_de_x[-1], dim_out=[self.dim_x], normed=self.normed, appended_name='_de_x')
        self.decoder_x = self.decoder(in_decoder_params_x, hidden_layers_de_x, reconstruction_layer_x, nonlin=self.nonlinearity, normalization=self.normalization,
                                    dropout_rate=self.dropout_rate)
        in_decoder_params_x_inf, hidden_layers_de_x_inf, reconstruction_layer_x_inf = self.copy_params(in_decoder_params_x, hidden_layers_de_x, reconstruction_layer_x)
        self.decoder_x_inf = self.decoder(in_decoder_params_x_inf, hidden_layers_de_x_inf, reconstruction_layer_x_inf, nonlin=self.nonlinearity, normalization=self.normalization,
                                        dropout_rate=self.dropout_rate)

        params = in_encoder_params_z + hidden_layers_en_z + z_layer + in_decoder_params_x + hidden_layers_de_x + reconstruction_layer_x
        params_inf = in_encoder_params_z_inf + hidden_layers_en_z_inf + z_layer_inf + in_decoder_params_x_inf + hidden_layers_de_x_inf + reconstruction_layer_x_inf
        self.params = [item for sublist in params for item in sublist]
        self.params_inf = [item for sublist in params_inf for item in sublist]

    def _create_model(self):
        """
        Specify the objective function
        """
        # for the encoder
        x = T.matrix('x')

        # construct the objective function
        enp_z = self.encoder_z.transform([x])
        enpinf_z = self.encoder_z_inf.transform([x], inference=True)  # for test time

        rec_x, rec_x_inf = 0, 0
        for i in xrange(self.L):
            # recognition part
            # sample from q(z|x)
            z = self.encoder_z.sample(*enp_z)
            zinf = self.encoder_z_inf.sample(*enpinf_z)

            # generative part
            # p(x|z)
            pxparams_x = self.decoder_x.transform([z])
            pxparamsinf_x = self.decoder_x_inf.transform([zinf], inference=True)

            # get the reconstruction loss for p(x|z)
            rec_x += self.decoder_x.logp(x, *pxparams_x) / float(self.L)
            rec_x_inf += self.decoder_x_inf.logp(x, *pxparamsinf_x) / float(self.L)

        # kl-divergence for z
        kldiv_z = self.encoder_z.kldivergence(*enp_z)
        kldiv_zinf = self.encoder_z_inf.kldivergence(*enpinf_z)

        # stack the objectives
        objectives = [rec_x, kldiv_z]
        # stack the evaluation objectives
        objectives_inference = [rec_x_inf, kldiv_zinf]

        self.optimizer = self._optimizer(objectives, objectives_inference, [x], self.params, self.params_inf, mode=self.mode, alpha=self.learningRate,
                            regularization=self.regularization, normalization=self.normalization, weight_decay=self.weight_decay,
                            batch_size=self.batch_size, polyak=self.polyak, beta3=self.beta3)
        self.transform = theano.function([x], enpinf_z[0])  # return the mean of the latent as the new representation

    def fit(self, x, xvalid=None, verbose=False, print_every=10):
        """
        Fit the model to data
        """
        indices_t = np.arange(x.shape[0])
        gpm.prng.shuffle(indices_t)
        rounding = lambda x: ['%.3f' % i for i in x]

        stats_train, stats_valid = [], []
        for i in xrange(self.iterations + 1):
            t = time.time()
            objectives = self.optimizer.train([x[indices_t]], verbose=verbose).tolist()
            objectives_v = self.optimizer.evaluate([xvalid]).tolist()

            objectives[0] /= x.shape[0]  # E[logp(x|z)]
            objectives[1] /= x.shape[0]  # KL(q(z|x) || p(z))
            objectives_v[0] /= xvalid.shape[0]
            objectives_v[1] /= xvalid.shape[0]
            gpm.prng.shuffle(indices_t)

            stats_train.append(objectives)
            stats_valid.append(objectives_v)
            if i % print_every == 0:
                dt = time.time() - t
                objectives += [np.sum(objectives)]  # lower bound
                objectives_v += [np.sum(objectives_v)]
                log_f(self.name_log, 'Epoch: ' + str(i) + '/' + str(self.iterations) + ', train: ' + str(map(float, rounding(objectives))) + ', valid: ' + str(map(float, rounding(objectives_v))) + ', dt: ' + str(dt))

        return stats_train, stats_valid
