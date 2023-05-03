import tensorflow as tf
import tensorflow_probability as tfp
import math

tfd = tfp.distributions


class Categorical_softmax_Likelihood(tf.Module):

    #maybe remove std
    def __init__(self, output_dim, n_batched_models, n_classes, trainable=True, name='categorical_softmax_likelihood'):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.n_batched_models = n_batched_models
        self.n_classes = n_classes

        #if trainable:
          #  self.log_std = tf.Variable(tf.ones(n_batched_models, output_dim) * tf.math.log(std), trainable=True)
        #else:
         #   self.log_std = tf.constant(tf.ones(n_batched_models, output_dim) * tf.math.log(std))

    """
    @property
    def std(self):
        return tf.exp(self.log_std)
    

    def log_prob(self, y_pred, y_true, std=None):
        #if std is None:
          #  std = self.std
        assert len(y_pred.shape) == 3
        assert y_pred.shape[2] == self.output_dim
        batch_size = y_pred.shape[1]


        #change to categorical softmax
        likelihood = tfd.Independent(tfd.OneHotCategorical(logits=y_pred), reinterpreted_batch_ndims=1)
        log_likelihood = likelihood.log_prob(y_true)
        avg_log_likelihood = tf.reduce_mean(log_likelihood, axis=-1) # average over batch
        return avg_log_likelihood
    """

    def log_prob(self, y_pred, y_true, num_classes):
        assert len(y_pred.shape) == 3
        assert y_pred.shape[2] == num_classes

        #likelihood = tfd.Independent(tfd.Normal(y_pred, likelihood_std), reinterpreted_batch_ndims=1)
        likelihood = tfd.OneHotCategorical(logits=y_pred) #, reinterpreted_batch_ndims=1)
        log_likelihood = likelihood.log_prob(y_true)
        avg_log_likelihood = tf.reduce_mean(log_likelihood, axis=-1) # average over batch
        #print(avg_log_likelihood)
        return avg_log_likelihood



    """
      likelihood = tfd.Independent(tfd.Normal(y_pred, likelihood_std), reinterpreted_batch_ndims=1)
        log_likelihood = likelihood.log_prob(y_true)
        avg_log_likelihood = tf.reduce_mean(log_likelihood, axis=-1) # average over batch
        return avg_log_likelihood

    def calculate_eval_metrics(self, pred_dist, y_true):
        eval_results = {'avg_ll': tf.reduce_mean(pred_dist.log_prob(y_true)).numpy(),
                        'avg_rmse': self.rmse(pred_dist.mean(), y_true).numpy(),
                        }

        if self.output_dim == 1:
            eval_results['cal_err'] = self.calib_error(pred_dist, y_true).numpy()

        return eval_results
    """
      

    def calculate_eval_metrics(self, pred_dist, y_true):
        eval_results = {#'accuracy': tf.reduce_mean(tf.cast(tf.argmax(pred_dist.mean(), axis=-1) == y_true, tf.float32)).numpy()
                        'avg_accuracy' : self.accuracy(pred_dist.mean(), y_true).numpy(),
                        'avg_ll': tf.reduce_mean(pred_dist.log_prob(y_true)).numpy(),
                        #'avg_rmse': self.rmse(pred_dist.mean(), y_true).numpy(),
                        }

        return eval_results

    @staticmethod
    def calib_error(pred_dist, y_true, use_circular_region=False):
        if y_true.ndim == 3:
            y_true = y_true[0]

        if use_circular_region or y_true.shape[-1] > 1:
            cdf_vals = pred_dist.cdf(y_true, circular=True)
        else:
            cdf_vals = pred_dist.cdf(y_true)
        cdf_vals = tf.reshape(cdf_vals, (-1, 1))

        num_points = tf.cast(tf.size(cdf_vals), tf.float32)
        conf_levels = tf.linspace(0.05, 0.95, 20)
        emp_freq_per_conf_level = tf.reduce_sum(tf.cast(cdf_vals <= conf_levels, tf.float32), axis=0) / num_points

        calib_err = tf.reduce_mean(tf.abs((emp_freq_per_conf_level - conf_levels)))
        return calib_err

    """
    def get_pred_mixture_dist(self, y_pred, std=None):
        #if std is None:
         #   std = self.std

        # check shapes
        tf.assert_equal(len(y_pred.shape), 3)
        tf.assert_equal(y_pred.shape[0], std.shape[0])
        tf.assert_equal(y_pred.shape[-1], std.shape[-1])

        num_mixture_components = y_pred.shape[0]

        components = [tfd.Independent(tfd.Normal(y_pred[i], std[i]), reinterpreted_batch_ndims=1)
                      for i in range(num_mixture_components)]
        categorical = tfd.Categorical(logits=tf.transpose(tf.zeros(y_pred.shape[:2])))
        return tfp.distributions.Mixture(categorical, components, name='predictive_mixture')
    """

    def get_pred_mixture_dist(self, y_pred, std=None):
        #if std is None:
        #   std = self.std

        # check shapes
        tf.assert_equal(len(y_pred.shape), 3)
        #print(y_pred.shape)
        #print(y_pred)

        #num_classes = y_pred.shape[-1]
        #y_pred.shape[0] = K * L (number of particles hyper-posterior 3 * number o fpartils psoterior 5)
        #y_pred.shape[1] number of samples
        #y_pred.shape[2] = number of classes


        num_mixture_components = y_pred.shape[0]

        components = [tfd.Independent(tfd.OneHotCategorical(logits=y_pred[i]), reinterpreted_batch_ndims=1)
                    for i in range(num_mixture_components)]
        
        categorical = tfd.Categorical(logits=tf.transpose(tf.zeros(y_pred.shape[:2])))
        print(categorical)
        print(components)
        print(categorical)
        return tfp.distributions.Mixture(categorical, components, name='predictive_mixture')



    """
    def rmse(self, y_pred_mean, y_true):
        
        Args:
            y_pred_mean (tf.Tensor): mean prediction
            y_true (tf.Tensor): true target variable

        Returns: (tf.Tensor) Root mean squared error (RMSE)

        
        tf.assert_equal(y_pred_mean.shape, y_true.shape)
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred_mean - y_true)))
    """
    
    def accuracy(y_pred, y_true):
        """
        Args:
            y_pred (tf.Tensor): predicted classes
            y_true (tf.Tensor): true target classes

        Returns: (tf.Tensor) Accuracy

        """
        tf.assert_equal(y_pred.shape, y_true.shape)
        correct_predictions = tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y_true, axis=-1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



""" helper function """

def _split(array, n_splits):
    """
        splits array into n_splits of potentially unequal sizes
    """
    assert array.ndim == 1
    n_elements = array.shape[0]

    remainder = n_elements % n_splits
    split_sizes = []
    for i in range(n_splits):
        if i < remainder:
            split_sizes.append(n_elements //  n_splits + 1)
        else:
            split_sizes.append(n_elements // n_splits)
    return tf.split(array, split_sizes)