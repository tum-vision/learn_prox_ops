"""Datasets"""

import os
import random
from glob import glob
from collections import namedtuple
import multiprocessing
import numpy as np

import tensorflow as tf
import skimage.data
from utilities import (unpickle,
                       tf_add_gaussian_noise,
                       tf_add_local_gaussian_noise,
                       tf_add_poisson_noise,
                       tf_add_gaussian_noise_and_random_blur,
                       tf_shuffle_batch_join)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Dataset = namedtuple('Dataset', ['images'])
Datasets = namedtuple('Datasets', ['train', 'test', 'shape'])


def load_image(path):
    """Open, load and normalize an image.

    :param path: Image path.
    :type path: String

    :returns: Normalized image.
    :rtype: np.ndarray
    """
    img = skimage.data.imread(path)

    if img.dtype == np.uint8:
        normalizer = 255.
    else:
        normalizer = 65535.

    img = img / normalizer
    return img.astype(np.float32)


def load_deblurring_grey_data(experiment_name=None, image_name=None):
    """
    Load the data for the grayscale deblurring experiments on 11 standard test
    images first conducted in [A machine learning approach for non-blind image deconvolution](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Schuler_A_Machine_Learning_2013_CVPR_paper.pdf).

    :param experiment_name: Name of the experiment a-e: experiment_*
    :type experiment_name: String
    :param image_name: Name of the image
    :type image_name: String

    :returns: Experiment data as Dict or single Tuple
    :rtype: Tuple
    """
    crop = 12
    data_dir = os.path.join(ROOT_DIR, 'data/deblurring_grey')
    experiments_data = {os.path.basename(experiment_dir):
                        {os.path.basename(image_dir):
                         {'f': load_image(os.path.join(image_dir, 'blurred_observation.png')),
                          'img': load_image(os.path.join(image_dir, 'original.png'))}
                         for image_dir in glob(experiment_dir + '/*')
                         if os.path.isdir(image_dir)}
                        for experiment_dir in glob(data_dir + '/*')}

    for experiment, experiment_images in experiments_data.items():
        kernel_img = load_image(os.path.join(data_dir, experiment + '/kernel.png'))
        kernel_img /= kernel_img.sum()
        experiment_images['kernel_img'] = kernel_img

    if experiment_name is not None:
        experiment_data = experiments_data[experiment_name]
        if image_name is not None:
            image_data = experiment_data[image_name]
            return (image_data['f'], image_data['img'], experiment_data['kernel_img'], crop)
        return experiment_data, crop
    else:
        if image_name is not None:
            print("Specifying only an image is not possible.")
        return experiments_data, crop


def load_demosaicking_data(image_name=None, dataset="mc_master"):
    """
    Load McMaster or Kodak demosaicking data.

    :param image_name: Name of a particular image
    :type image_name: String

    :return test_images: Experiment data as Dict or single Tuple
    :rtype test_images: Tuple
    """
    crop = 5
    image_paths = glob(os.path.join(ROOT_DIR,
                                    "data/demosaicking",
                                    dataset.lower(),
                                    "*"))

    def sort_key(d): return os.path.splitext(os.path.basename(d))[0]
    data = {os.path.splitext(os.path.basename(d))[0]:
            {'img': load_image(d)}
             for d in sorted(image_paths, key=sort_key)}

    if image_name is not None:
        return (data[str(image_name).lower()]['img'],
                crop)
    else:
        return data, crop


class Pipelines(object):
    """Base class for Tensorflow denoising pipelines."""
    patch_size = 40
    train_set_multiplier = 1
    test_set_multiplier = 1
    grayscale = False

    def __init__(self, opt, test_epochs):
        """
        Class constructor.

        :param opt: Option flags.
        :type opt: tf.app.flags.FLAGS
        :param test_epochs: Number of test_epochs. Usually None for 1 entire epoch.
        :type test_epochs: Int
        """
        self.input_shape = (self.patch_size, self.patch_size, opt.channels)
        self.train_shape = (self.patch_size, self.patch_size, opt.channels)
        self.test_shape = (self.patch_size, self.patch_size, opt.channels)
        self.sigma_noise = opt.sigma_noise
        self.noise_type = opt.noise_type
        self.batch_size = opt.batch_size
        self.img_decoder = tf.image.decode_jpeg
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        train_files, test_files = self.init_file_lists()
        train_pipe = self.tf_data_pipeline(train_files * self.train_set_multiplier,
                                           self.train_shape,
                                           'train_pipeline',
                                           opt.train_epochs)
        test_pipe = self.tf_data_pipeline(test_files * self.test_set_multiplier,
                                          self.test_shape,
                                          'test_pipeline',
                                          test_epochs,
                                          train=False)

        Pipeline = namedtuple('Pipeline',
                              ['data', 'labels', 'num', 'epochs', 'batch_size'])
        self.train = Pipeline(*train_pipe,
                              epochs=opt.train_epochs,
                              batch_size=opt.batch_size)
        self.test = Pipeline(*test_pipe,
                             epochs=test_epochs,
                             batch_size=opt.batch_size)

    def init_file_lists(self):
        """
        Implement to inject a specific dataset.

        :returns: Two lists of train and test file paths.
        :rtype: Tuple
        """
        raise NotImplementedError

    def tf_data_pipeline(self, file_paths, input_shape, name,
                         num_epochs=None, train=True):
        """
        Build data pipeline for denoising data and label pairs. The pipeline
        includes batch-sampling, multiple noise models, data augmentation and
        data preprocessing.

        :param file_paths: File paths
        :type file_paths: List of String
        :param input_shape: Runtime shape of the input image
        :type input_shape: 1D tf.Tensor or Python array
        :param name: variable scope name
        :type name: String
        :param num_epochs: Number of epochs the pipeline should provide
        :type num_epochs: Int
        :param train: If it is a training pipeline i.e. with data augmentation
        :type train: Bool

        :returns: Data batch, label batch and number of total files
        :rtype: Tuple including tf.Tensor and Int
        """
        n_jobs = multiprocessing.cpu_count()

        with tf.variable_scope(name):
            filename_queue = tf.train.string_input_producer(file_paths,
                                                            num_epochs=num_epochs)

            def prepare_data():
                reader = tf.WholeFileReader()
                _, value = reader.read(filename_queue)
                datum = self.img_decoder(value, channels=input_shape[2])
                datum = tf.to_float(datum)
                datum = tf.div(datum, 255.)
                if train:
                    datum = tf.image.random_flip_up_down(datum)
                    datum = tf.image.random_flip_left_right(datum)
                    datum = tf.image.rot90(datum, k=random.randint(0, 3))
                datum = tf.random_crop(datum, input_shape)
                label = tf.identity(datum)

                if self.noise_type == 'gaussian':
                    datum = tf_add_gaussian_noise(datum, input_shape, self.sigma_noise)
                elif self.noise_type == 'local_gaussian':
                    datum = tf_add_local_gaussian_noise(datum, input_shape, self.sigma_noise, 2)
                elif self.noise_type == 'random_sigma_gaussian':
                    datum = tf_add_random_sigma_gaussian_noise(datum, input_shape, 0.0, 0.22)
                elif self.noise_type == 'poisson':
                    datum = tf_add_poisson_noise(datum, input_shape)
                elif self.noise_type == 'gaussian_random_blur':
                    datum = tf_add_gaussian_noise_and_random_blur(datum,
                                                                  input_shape,
                                                                  self.sigma_noise,
                                                                  1.0,
                                                                  4.0,
                                                                  25)
                return datum, label

            data_list = [prepare_data() for _ in range(n_jobs)]

            if train:
                do_dequeue = self.is_train
                min_after_dequeue = 10 * self.batch_size
            else:
                do_dequeue = tf.logical_not(self.is_train)
                min_after_dequeue = 0
            capacity = min_after_dequeue + 3 * self.batch_size
            data_batch, labels_batch = tf_shuffle_batch_join(data_list,
                                                             batch_size=self.batch_size,
                                                             capacity=capacity,
                                                             do_dequeue=do_dequeue,
                                                             allow_smaller_final_batch=False,
                                                             min_after_dequeue=min_after_dequeue)

            return data_batch, labels_batch, len(file_paths)


class BSDS500Pipelines(Pipelines):
    """
    Data pipelines for denoising training of the BSDS500 dataset.
    """
    patch_size = 40
    train_set_multiplier = 512  # to have 128 * 1600 patches in one epoch
    test_set_multiplier = 192  # to have 13056 patches in one epoch

    def init_file_lists(self):
        """
        Load and split the BSDS500 dataset.

        :returns: Two lists of train and test file paths.
        :rtype: Tuple
        """
        if self.noise_type == 'gaussian_random_sigma':
            self.patch_size = 50
            self.train_set_multiplier = 960  # to have 128 * 3000 patches in one epoche

        data_path = os.path.join(ROOT_DIR, 'data/bsds_500')
        if self.grayscale:
            train_files = glob(os.path.join(data_path, "greyscale_images/train/*.png"))
            test_files = glob(os.path.join(data_path, "greyscale_images/test/*.png"))
            self.img_decoder = tf.image.decode_png
        else:
            train_files = glob(os.path.join(data_path, 'color_images/train/*.jpg')) + \
                          glob(os.path.join(data_path, 'color_images/test/*.jpg'))
            train_files = train_files[:400]
            test_files = glob(os.path.join(data_path, 'data/color_images/val/*.jpg'))[:68]

        return train_files, test_files


class BSDS500PipelinesGray(BSDS500Pipelines):
    """
    Data pipelines for denoising training of the BSDS500 dataset. All images
    are converted to grayscale.
    """
    grayscale = True


PIPELINES = {'bsds500': BSDS500Pipelines,
             'bsbds500gray': BSDS500PipelinesGray}
