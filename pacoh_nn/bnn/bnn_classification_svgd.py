import tensorflow as tf
import tensorflow_probability as tfp
import math

from pacoh_nn.bnn.classification_algo import ClassificationModel
from pacoh_nn.modules.neural_network import BatchedFullyConnectedNN
from pacoh_nn.modules.prior_posterior import GaussianPrior
from pacoh_nn.modules.classification_likelihood import Categorical_softmax_Likelihood

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class BayesianNeuralNetwork_classification_SVGD(ClassificationModel):

    def __init__(self, x_train, y_train, num_classes, hidden_layer_sizes=(32, 32, 32, 32), activation='relu',
                 likelihood_prior_mean=tf.math.log(0.1), likelihood_prior_std=1.0, learn_likelihood=True, 
                 sqrt_mode=False,
                 prior_std=0.1, prior_weight=1e-4, n_particles=10, batch_size=8, bandwidth=100., lr=1e-3,
                 meta_learned_prior=None, normalization_stats=None):

        self.prior_weight = prior_weight
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.sqrt_mode = sqrt_mode

        self.num_classes = num_classes

        # data handling
        self._process_train_data(x_train, y_train, normalization_stats)

        # setup nn
        self.nn = BatchedFullyConnectedNN(n_particles, self.output_dim, hidden_layer_sizes, activation)
        self.nn.build((None, self.input_dim))

        # setup prior
        self.nn_param_size = self.nn.get_variables_stacked_per_model().shape[-1]
        if learn_likelihood:
            self.likelihood_param_size = self.output_dim
        else:
            self.likelihood_param_size = 0

        if meta_learned_prior is None:
            self.prior = GaussianPrior(self.nn_param_size, nn_prior_std=prior_std,
                                       likelihood_param_size=self.likelihood_param_size,
                                       likelihood_prior_mean=likelihood_prior_mean,
                                       likelihood_prior_std=likelihood_prior_std)
            self.meta_learned_prior_mode = False
        else:
            self.prior = meta_learned_prior
            assert meta_learned_prior.get_variables_stacked_per_model().shape[-1] == \
                   2 * (self.nn_param_size + self.likelihood_param_size)
            assert n_particles % self.prior.n_batched_priors == 0, "n_particles must be multiple of n_batched_priors"
            self.meta_learned_prior_mode = True

        # Likelihood
        self.likelihood = Categorical_softmax_Likelihood(self.output_dim, n_particles, self.num_classes)

        # setup particles
        if self.meta_learned_prior_mode:
            # initialize posterior particles from meta-learned prior
            params = tf.reshape(self.prior.sample(n_particles // self.prior.n_batched_priors), (n_particles, -1))
            self.particles = tf.Variable(params)
        else:
            # initialize posterior particles from model initialization
            nn_params = self.nn.get_variables_stacked_per_model()
            likelihood_params = tf.ones((self.n_particles, self.likelihood_param_size)) * likelihood_prior_mean
            self.particles = tf.Variable(tf.concat([nn_params, likelihood_params], axis=-1))

        # setup kernel and optimizer
        self.kernel = tfk.ExponentiatedQuadratic(length_scale=bandwidth)
        self.optim = tf.keras.optimizers.Adam(lr)


    def predict(self, x):
        # data handling
        x = self._handle_input_data(x, convert_to_tensor=True)
        x = self._normalize_data(x)

        # nn prediction

        #nn_params, _ = _split_into_nn_params_and_likelihood_std)self.particles
        nn_params = self.particles
        y_pred = self.nn.call_parametrized(x, nn_params)

        # form mixture of predictive distributions
        pred_dist = self.likelihood.get_pred_mixture_dist(y_pred)

        # unnormalize preds
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    


    @tf.function
    def step(self, x_batch, y_batch):

        # compute posterior score (gradient of log prob)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.particles)
            #nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(self.particles)
            nn_params = self.particles

            # compute likelihood
            y_pred = self.nn.call_parametrized(x_batch, nn_params)  # (k, b, d)
            avg_log_likelihood = self.likelihood.log_prob(y_pred, y_batch, self.num_classes)

            if self.meta_learned_prior_mode:
                particles_reshaped = tf.reshape(self.particles, (self.prior.n_batched_priors,
                                                self.n_particles // self.prior.n_batched_priors, -1))
                prior_prob = tf.reshape(self.prior.log_prob(particles_reshaped, model_params_prior_weight=self.prior_weight), (self.n_particles,))
            else:
                prior_prob = self.prior.log_prob(self.particles, model_params_prior_weight=self.prior_weight)

            # compute posterior log_prob
            prior_pre_factor = 1 / math.sqrt(self.num_train_samples) if self.sqrt_mode else 1 / self.num_train_samples
            post_log_prob = avg_log_likelihood + prior_pre_factor * prior_prob # (k,)
        score = tape.gradient(post_log_prob, self.particles)  # (k, n)

        # compute kernel matrix and grads
        particles_copy = tf.identity(self.particles)  # (k, n)
        with tf.GradientTape() as tape:
            tape.watch(self.particles)
            k_xx = self.kernel.matrix(self.particles, particles_copy)  # (k, k)
        k_grad = tape.gradient(k_xx, self.particles)
        svgd_grads_stacked = k_xx @ score - k_grad / self.n_particles  # (k, n)

        # apply SVGD gradients
        self.optim.apply_gradients([(- svgd_grads_stacked, self.particles)])
        return - post_log_prob