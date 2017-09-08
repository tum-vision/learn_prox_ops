"""TensorFlow Denoising CNNs"""

import numpy as np
import tensorflow as tf
from utilities import tf_mse, tf_psnr


class DNCNN(object):
    """
    A reimplementation of the denoising CNN originally designed in [Beyond a
    Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising]
    (https://arxiv.org/abs/1608.03981)
    """

    name = 'DNCNN'
    depth = 17

    def __init__(self, pipelines=None, data=None):
        """
        Class constructor.

        :param pipelines:  Used for training.
        :type pipelines: Pipelines
        :param data: Used for deployment.
        :type data: tf.placeholder
        """
        assert not (pipelines is None and data is None), "At least one network input must not be None."

        with tf.variable_scope(self.name):

            def return_and_enqueue(return_arg, enqueue_arg):

                return return_arg
            if pipelines is not None:
                self.is_train = pipelines.is_train  # tf.placeholder(tf.bool, name='is_train')
                data = tf.cond(self.is_train,
                               lambda: pipelines.train.data,
                               lambda: pipelines.test.data)
                labels = tf.cond(self.is_train,
                                 lambda: pipelines.train.labels,
                                 lambda: pipelines.test.labels)
                self.labels = tf.to_float(labels)
                self.channels = pipelines.input_shape[2]
            else:
                self.is_train = tf.constant(False)
                self.channels = data.get_shape()[3]

            self.data = tf.to_float(data)
            # Output is the residual and clipped output is the denoised image.
            self.output = self.init_network()
            # TODO: try scale and offset instead clipping
            self.output_clipped = tf.clip_by_value(tf.subtract(self.data, self.output), 0., 1., name='output_clipped')

            # pipeline train variables
            if pipelines is not None:
                self.loss = self.init_loss()
                self.total_num_steps = pipelines.train.epochs * \
                    pipelines.train.num / pipelines.train.batch_size
                self.train_opt, self.learning_rate = self.init_train_opt_and_learning_rate()

                self.metrics = self.init_metrics()
                # self.init_summaries()

    def init_train_opt_and_learning_rate(self):
        """
        Initialize all the necessary training operations and a stepwise
        decreasing learning rate. Only necessary for training i.e. if data
        pipelines are provided.
        """
        global_step = tf.Variable(0, trainable=False, name="global_step")
        start_learning_rate = 0.001
        decay_steps = self.total_num_steps
        learning_rate = tf.train.exponential_decay(start_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   0.1,
                                                   staircase=False)

        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

        grads = optimizer.compute_gradients(self.loss)

        # with tf.variable_scope('gradient_clipping'):
        #     grads = [(tf.clip_by_value(grad, -0.005, 0.005), var) for grad, var in grads]

        train_opt = optimizer.apply_gradients(grads, global_step=global_step)
        return train_opt, learning_rate

    def init_loss(self):
        """
        Initialize a residual loss with normalized batches. Only necessary for
        training i.e. if data pipelines are provided.
        """
        with tf.variable_scope('batch_normed_residual_loss'):
            residual = tf.subtract(self.data, self.labels)
            error_squared = tf.square(tf.subtract(self.output, residual))
            error_sum = tf.reduce_sum(error_squared)

            dynamic_batch_size = tf.shape(self.data)[0]
            return tf.div(error_sum, tf.to_float(tf.multiply(dynamic_batch_size, 2)))

    def init_metrics(self):
        """
        Initialize data-labels and output-labels metrics (PSNR).
        """
        metrics = {'psnr_data': tf_psnr(self.data, self.labels, name='psnr_data'),
                   'psnr_output': tf_psnr(self.output_clipped, self.labels, name='psnr_output')}
        return metrics

    def init_summaries(self):
        """
        Initialize summaries for TensorBoard.
        """
        # train
        # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        #    tf.histogram_summary(v.name, v, collections=['train'], name='variables')
        tf.summary.scalar('LOSS/batch_train_loss', self.loss, collections=['train'])
        if hasattr(self, 'learning_rate'):
            tf.summary.scalar('learning_rate', self.learning_rate, collections=['train'])

        # test
        for v in tf.get_collection('moving_avgs'):
            tf.summary.histogram(v.name, v, collections=['test'], name='moving_avgs')

        # images
        nb_imgs = 3
        tf.summary.image('data', self.data, max_outputs=nb_imgs, collections=['images'])
        tf.summary.image('output', self.output_clipped, max_outputs=nb_imgs, collections=['images'])
        tf.summary.image('label', self.labels, max_outputs=nb_imgs, collections=['images'])

    def init_network(self):
        """
        Initialize entire network graph. The self.depth attribute specifies
        the number of convolution layers.

        Batch normalization is disabled.
        """
        # CNN weights
        weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                             mode='FAN_IN',
                                                                             uniform=False)
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

        # Batch normalization
        # Disabled because network was not trainable.
        # factor 9 division to calculate stddev with 9*64 (paper) instead of 64 (tensorflow) incoming nodes.
        # gamma_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0 / 9.0,
        #                                                                    mode='FAN_IN',
        #                                                                    uniform=False)
        # batch_norm_initializers = {'gamma': gamma_initializer,
        #                            'beta': tf.zeros_initializer,
        #                            'moving_mean': tf.zeros_initializer,
        #                            'moving_variance': tf.constant_initializer(0.01)}

        # batch_norm_params = {'initializers': batch_norm_initializers,
        #                      'scale': True,
        #                      'is_training': self.is_train,
        #                      'variables_collections': {'moving_mean': ['moving_avgs'], 'moving_variance': ['moving_avgs']},
        #                      'updates_collections': ['update_ops']}

        # Network architecture
        net = self.data
        net = self.convolution('conv1',
                               net, 64, [3, 3],
                               biases_initializer=tf.zeros_initializer,
                               weights_initializer=weights_initializer,
                               weights_regularizer=weights_regularizer,
                               activation_fn=tf.nn.relu)

        for i in range(2, self.depth):
            net = self.convolution('conv' + str(i),
                                   net, 64, [3, 3],
                                   # normalizer_fn=tf.contrib.layers.batch_norm,
                                   # normalizer_params=batch_norm_params,
                                   biases_initializer=tf.zeros_initializer,
                                   weights_initializer=weights_initializer,
                                   weights_regularizer=weights_regularizer,
                                   activation_fn=tf.nn.relu)

        net = self.convolution('conv' + str(self.depth),
                               net, self.channels, [3, 3],
                               biases_initializer=tf.zeros_initializer,
                               weights_initializer=weights_initializer,
                               weights_regularizer=weights_regularizer,
                               activation_fn=None)
        return net

    @staticmethod
    def convolution(name, *args, **params):
        """
        Helper function to initialize namescoped convolution layers.

        :param name:
        type: name: String
        :param *args: Args for tf.contrib.layers.convolution2d
        :type *args: List or Tuple
        :param **params: Kwargs for tf.contrib.layers.convolution2d, except scope
        :type **params: Dict

        :returns: Tensor with convolution applied to it.
        :rtype: tf.Tensor
        """
        with tf.variable_scope(name) as scope:
            out = tf.contrib.layers.convolution2d(scope=scope, *args, **params)
        return out


class DNCNNB(DNCNN):
    """
    DNCNN network with more layers for blind denoising.
    """
    name = 'DNCNNB'
    depth = 20


class DNCNNTV(DNCNN):
    """
    DNCNN network with modified cost function to take TV norm into account.
    The idea is to penalize a small TV norm since TV norm should already be
    minimized by the TV prior in the Primal-Dual minimization.

    Attention: The negative term in the loss makes training highly instable!
    """
    name = 'DNCNNTV'

    def init_loss(self):
        """
        Initialize a residual loss with normalized batches. This loss also
        penalizes a low TV norm. Only necessary for training i.e. if data
        pipelines are provided.
        """
        with tf.variable_scope('batch_normed_residual_loss'):
            residual = tf.subtract(self.data, self.labels)
            error_squared = tf.square(tf.subtract(self.output, residual))
            error_sum = tf.reduce_sum(error_squared)

            dynamic_batch_size = tf.shape(self.data)[0]
            x_y_grad_kernel = np.array([[[[1.0, 1.0]], [[-1.0, 0.0]]], [[[0.0, -1.0]], [[0.0, 0.0]]]],
                                       dtype=np.float32)
            grad_kernel = np.repeat(x_y_grad_kernel, self.channels, axis=2)
            grad = tf.nn.depthwise_conv2d(tf.subtract(self.data, self.output),
                                          tf.constant(grad_kernel),
                                          strides=[1, 1, 1, 1],
                                          padding="SAME")
            tv_norm = tf.multiply(tf.reduce_sum(tf.square(grad)), 0.05)
            return tf.div(tf.subtract(error_sum, tv_norm),
                          tf.to_float(tf.multiply(dynamic_batch_size, 2)))


NETWORKS = {'DNCNN': DNCNN,
            'DNCNNB': DNCNNB,
            'DNCNNTV': DNCNNTV}
