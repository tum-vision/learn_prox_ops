"""TensorFlow Solver to train denoising CNNs."""

from __future__ import division

import os
import time
import math
from collections import deque
import numpy as np

from tf_networks import NETWORKS
from utilities import tf_print_trainable_variables
from data import PIPELINES

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data/tf_models',
                           """Base directory where to write event logs. Default: 'data/tf_models'.""")
tf.app.flags.DEFINE_string('device_name', '/cpu:0',
                           """Name of GPU to run model on. Default: '/cpu:0'.""")
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Train and test mini batch size. Default: 100.""")
tf.app.flags.DEFINE_integer('train_epochs', 10,
                            """Training epochs. Default: 10.""")
tf.app.flags.DEFINE_integer('average_train_loss_size', 50,
                            """Mini batch losses to use for average loss. Default: 50.""")
tf.app.flags.DEFINE_string('noise_type', 'gaussian',
                           """Noise type. Default: 'gaussian'.""")
tf.app.flags.DEFINE_float('sigma_noise', 0.08,
                          """Sigma for Gaussian noise generation. Default: 0.08.""")
tf.app.flags.DEFINE_string('network', 'DNCNN',
                           """Network architecture. Default: 'DNCNN'.""")
tf.app.flags.DEFINE_string('pipelines', 'bsds500',
                           """Data pipeline. Default: 'bsds500'.""")
tf.app.flags.DEFINE_integer('channels', 3,
                            """Number of color channels. Default: 3.""")
tf.app.flags.DEFINE_string('model_path', '',
                           """Model file path. Default: ''.""")
tf.app.flags.DEFINE_boolean('fill_gpu', False,
                            """Whether to fill entire gpu. Default: False""")


class Deployer(object):
    """Deploy a trained network."""

    def __init__(self, shape, opt=FLAGS):
        """Class constructor.

        :param shape: Runtime shape of the input image
        :type shape: 1D tf.Tensor or Python array
        :param opt: Option flags.
        :type opt: tf.app.flags.FLAGS
        """
        tf.reset_default_graph()
        data = tf.placeholder(tf.float32,
                              (None, None, None, opt.channels),
                              name='data')
        opt.network = os.path.split(os.path.dirname(opt.model_path))[1].split('__')[0]

        with tf.device(opt.device_name):
            self.net = NETWORKS[opt.network](data=data)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = not opt.fill_gpu
        sess_config.allow_soft_placement = True
        # sess_config.log_device_placement = True

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_name[-1]
        self.sess = tf.Session(config=sess_config)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if opt.model_path:
            saver = tf.train.Saver()
            saver.restore(self.sess, opt.model_path)

    def deploy(self, images):
        """
        Deploy and denoise an array of images.

        :param images: Input images
        :type images: List of np.ndarray

        :returns: Output (residual) and clipped output (denoised image) of the network.
        :rtype: Dict of List
        """
        images = np.array(images)

        feed_dict = {self.net.data: images}
        op_dict = {'output': self.net.output,
                   'output_clipped': self.net.output_clipped}
        res = self.sess.run(op_dict, feed_dict)
        return res

    def print_trainable_variables(self):
        """
        Prints info name and shapes of trainable model variables which are
        available in the current session.
        """
        self.sess.run(tf.global_variables_initializer())

        tf_print_trainable_variables()


class Solver(object):
    """Train a given neural network mode."""

    def __init__(self, opt=FLAGS):
        """
        Class constructor.

        :param opt: Option flags.
        :type opt: tf.app.flags.FLAGS
        """
        tf.reset_default_graph()

        with tf.device(opt.device_name):
            pipelines = PIPELINES[opt.pipelines](opt, test_epochs=None)
            self.net = NETWORKS[opt.network](pipelines)
            self.net.init_summaries()

        noise_str = opt.noise_type
        if opt.noise_type in ["gaussian", "local_gaussian"]:
            noise_str += "_" + str(opt.sigma_noise)
        opt.data_dir = os.path.join(opt.data_dir, '__'.join([self.net.name,
                                                             noise_str,
                                                             '-'.join([str(s) for s in pipelines.input_shape]),
                                                             str(opt.batch_size),
                                                             time.strftime('%Y_%m_%d_%H_%M')]))

        self.train_summaries = tf.summary.merge_all('train')
        self.test_summaries = tf.summary.merge_all('test')
        self.image_summaries = tf.summary.merge_all('images')
        self.queue_summaries = tf.summary.merge_all()

        self.min_test_loss = None
        self.pipelines = pipelines
        self.train_loss_queue = deque([], opt.average_train_loss_size)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = not opt.fill_gpu
        sess_config.allow_soft_placement = True
        # sess_config.log_device_placement = True

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_name[-1]
        self.sess = tf.Session(config=sess_config)
        self.writer = tf.summary.FileWriter(opt.data_dir, self.sess.graph)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        tf_print_trainable_variables()

        if not opt.model_path == '':
            self.saver.restore(self.sess, opt.model_path)
        self.opt = opt

    def train_step(self, step):
        """
        Execute a training step including parameter updates and logging.

        :param step: Global iteration step.
        :type step: Int
        """
        op_dict = {'loss': self.net.loss,
                   'learning_rate': self.net.learning_rate,
                   'train_summaries': self.train_summaries,
                   'queue_summaries': self.queue_summaries}
        op_dict.update(self.net.metrics)

        feed_dict = {self.net.is_train: True}
        if step != 0:
            op_dict['update_ops'] = tf.get_collection('update_ops')
            op_dict['train_op'] = self.net.train_opt

        res = self.sess.run(op_dict, feed_dict)
        self.writer.add_summary(res['train_summaries'], step)
        self.writer.add_summary(res['queue_summaries'], step)

        # avg_train_loss summary
        self.train_loss_queue.append(res['loss'])
        avg_train_loss = sum(self.train_loss_queue) / len(self.train_loss_queue)
        summary = tf.Summary()
        summary.value.add(tag='LOSS/running_batch_train', simple_value=avg_train_loss)
        self.writer.add_summary(summary, step)

        # train log
        if step % 50 == 0 or step in range(1, 11):
            print('Iteration: %i Batch training loss: %f Batch PSNR: %f Learning rate: %f' %
                  (step, res['loss'], res['psnr_output'], res['learning_rate']))

    def start_test(self, global_step=None):
        """
        Start a testing step over the entire test set and log the results.
        If the model yields superior results it is saved.

        :param global_step: Global iteration step.
        :type global_step: Int

        :returns: Average test loss.
        :rtype: Float
        """
        print('Start testing.')

        avg_loss = avg_psnr_output = avg_psnr_data = 0

        steps = int(math.ceil(self.pipelines.test.num / self.opt.batch_size))
        feed_dict = {self.net.is_train: False}
        for step in range(steps):
            op_dict = {'loss': self.net.loss,
                       # 'test_summaries': self.test_summaries,
                       'image_summaries': self.image_summaries}
            op_dict.update(self.net.metrics)
            res = self.sess.run(op_dict, feed_dict)

            avg_loss += res['loss']
            avg_psnr_data += res['psnr_data']
            avg_psnr_output += res['psnr_output']

        # steps = 0
        # try:
        #     while not test_coord.should_stop():
        #         steps += 1
        #         op_list = [self.net.loss, self.psnr_data, self.psnr_output, self.image_summaries]
        #         test_loss, psnr_data, psnr_output, summaries = self.sess.run(op_list, feed_dict={self.net.is_train: False})
        #         avg_test_loss += test_loss
        #         avg_psnr_data += psnr_data
        #         avg_psnr_output += psnr_output
        #
        # except tf.errors.OutOfRangeError:
        #     print 'Done testing.'
        # finally:
        #     test_coord.request_stop()

        avg_loss /= steps
        avg_psnr_data /= steps
        avg_psnr_output /= steps

        summary = tf.Summary()
        summary.value.add(tag='LOSS/avg_test', simple_value=avg_loss)
        summary.value.add(tag='PSNR/avg_data', simple_value=avg_psnr_data)
        summary.value.add(tag='PSNR/avg_output', simple_value=avg_psnr_output)
        self.writer.add_summary(summary, global_step)
        # self.writer.add_summary(res['test_summaries'], global_step)

        # save model with lowest test loss
        if self.min_test_loss is None or avg_loss < self.min_test_loss:
            self.min_test_loss = avg_loss
            self.writer.add_summary(res['image_summaries'], global_step)
            if global_step > 0:
                self.save_model(global_step)

        # test log
        self.print_info(global_step)

        print('Average test loss: %g' % avg_loss)
        print('Average data PSNR: %g' % avg_psnr_data)
        print('Average output PSNR: %g' % avg_psnr_output)

        return avg_loss

    def start_train(self):
        """
        Start the training of a model including epoch-wise testing.
        """
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=self.sess, coord=coord)

        step = 0
        self.start_test(step)
        try:
            print('Start training %i epochs.' % self.opt.train_epochs)
            while not coord.should_stop():
                step += 1
                self.train_step(step)
                num_epochs = (self.opt.batch_size * step) / self.pipelines.train.num
                # TODO Make work for any batch_size step combination
                if num_epochs % 0.25 == 0.0:
                    self.start_test(step)
        except Exception as e:
            coord.request_stop(e)
            print('Done training (train_epochs reached).')
        finally:
            coord.request_stop()

        self.writer.close()

    def save_model(self, step):
        """
        Saves graph model in self.opt.data_dir

        :param step: Global iteration step.
        :type step: Int
        """
        model_path = os.path.join(self.opt.data_dir,
                                  "model_" + str(step) + ".ckpt")
        save_path = self.saver.save(self.sess, model_path)
        print('Model saved in file: %s' % save_path)
        # save best model in non step size related format
        self.saver.save(self.sess, os.path.join(self.opt.data_dir, "model.ckpt"))


    def print_info(self, step):
        """
        Prints training related info on how many samples are already processed.

        :param step: Global iteration step.
        :type step: Int
        """
        num_images = self.opt.batch_size * step
        num_epochs = num_images / self.pipelines.train.num
        print('Train data: %g/%g images and therefore %g/%g epochs.' %
              (num_images, self.pipelines.train.num, num_epochs, self.opt.train_epochs))
        print('Test data: %g images with batch size %g.' %
              (self.pipelines.test.num, self.opt.batch_size))


def main():
    """Main wrapper function."""
    solver = Solver()
    solver.start_train()

if __name__ == "__main__":
    main()
