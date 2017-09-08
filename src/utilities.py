"""Utility functions"""

from functools import reduce
import random
import math
import pickle

import numpy as np
import scipy.stats as st
import cv2
import tensorflow as tf
from scipy import sparse

from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import data_flow_ops, math_ops
from tensorflow.python import summary
from tensorflow.python.training.input import (_as_tensor_list_list,
                                              _dtypes,
                                              _shapes,
                                              _enqueue_join,
                                              _flatten,
                                              _validate_join,
                                              _restore_sparse_tensors,
                                              _as_original_type,
                                              _store_sparse_tensors_join)


def tf_log_10(x):
    """
    Common logarithm with base 10 for TensorFlow.

    :param x: Input tensor
    :type x: tf.Tensor

    :returns: Result of logarithm with base 10
    :rtype: tf.Tensor
    """
    return tf.div(tf.log(x), tf.log(10.0))


def tf_psnr(image_i, image_k, reduce_mean=True, name='psnr'):
    """
    Peak signal-to-noise ratio for TensorFlow.

    :param image_i: Noise free reference image
    :type image_i: tf.Tensor
    :param image_k: Noisy image version of image_i
    :type image_k: tf.Tensor
    :param reduce_mean: If tensor with PSNRs is reduced by mean
    :type reduce_mean: Bool
    :param name: variable scope name
    :type name: String

    :returns: PSNR between noisy and reference image
    :rtype: tf.Tensor
    """
    if image_i.dtype is not image_k.dtype:
        raise TypeError('PSNR images must have same data type.')

    with tf.variable_scope(name):
        # TODO: make more robust
        if image_i.dtype is tf.uint8:
            max_pos_2 = tf.pow(255.0, 2)
        else:
            max_pos_2 = 1.0

        mse_i_k = tf_mse(image_i, image_k, [1, 2, 3])

        psnr_tensor = tf.mul(10.0, tf_log_10(tf.div(max_pos_2, mse_i_k)))

        if reduce_mean:
            return tf.reduce_mean(psnr_tensor)
        else:
            return psnr_tensor


def tf_mse(a, b, reduction_indices=None, name='mse'):
    """
    Mean squared error for TensorFlow.

    :param a: First input tensor
    :type b: tf.Tensor
    :param a: Second input tensor
    :type b: tf.Tensor
    :param reduction_indices: Dimensions to reduce. If None all dimensions are reduced.
    :type reduction_indices: List or None
    :param name: Variable scope name
    :type reduction_indices: String

    :returns: MSE between a and b
    :rtype: tf.Tensor
    """
    with tf.variable_scope(name):
        return tf.reduce_mean(tf.pow(tf.sub(a, b), 2),
                              reduction_indices=reduction_indices)


def tf_add_gaussian_noise(image, shape, stddev):
    """
    Add randomly distributed Gaussian noise to an image.

    :param image: Input image
    :type image: tf.Tensor
    :param shape: Runtime shape of the input image
    :type shape: 1D tf.Tensor or Python array
    :param stddev: Standard deviation of the noise.
    :type stddev: 0D tf.Tensor or Python array (float32)

    :returns: Image with Gaussian noise
    :rtype: tf.Tensor
    """
    image = tf.to_float(image)
    noise = tf.random_normal(shape, mean=0.0, stddev=stddev)
    image = tf.clip_by_value(tf.add(image, noise), 0, 1.)
    return image


def tf_add_local_gaussian_noise(image, shape, stddev, locality):
    """
    Add randomly distributed Gaussian noise with local resemblance to an image

    :param image: Input image
    :type image: tf.Tensor
    :param shape: Runtime shape of the input image
    :type shape: 1D tf.Tensor or Python array
    :param stddev: Standard deviation of the noise.
    :type stddev: 0D tf.Tensor or Python array (float32)
    :param locality: Resemblance factor
    :type locality: Int

    :returns: Image with Gaussian noise
    :rtype: tf.Tensor
    """
    if shape[0] % locality != 0 or shape[1] % locality != 0:
        print("For local Gaussian noise division of the image dimensions by "
              "the locality must be without rest.")
        exit()

    image = tf.to_float(image)
    base_shape = list(shape)
    base_shape[0] //= locality
    base_shape[1] //= locality
    noise = tf.random_normal(base_shape, mean=0.0, stddev=stddev)
    image = tf.clip_by_value(tf.add(image, tf.image.resize_images(noise, (shape[0], shape[1]))),
                             0.0,
                             1.0)
    return image


def tf_add_random_sigma_gaussian_noise(image, shape, min_sigma, max_sigma):
    """
    Add randomly distributed Gaussian noise with random standard deviation
    (between min_sigma and max_sigma) to an image.

    :param image: Input image
    :type image: tf.Tensor
    :param shape: Runtime shape of the input image
    :type shape: 1D tf.Tensor or Python array
    :param min_sigma: Minimum standard deviation of the noise.
    :type min_sigma: 0D tf.Tensor or Python array (float32)
    :param max_sigma: Maximum standard deviation of the noise.
    :type max_sigma: 0D tf.Tensor or Python array (float32)

    :returns: Image with Gaussian noise
    :rtype: tf.Tensor
    """
    sigma_noise = tf.random_uniform((1,), minval=min_sigma, maxval=max_sigma)
    return tf_add_gaussian_noise(datum, shape, sigma_noise)


def tf_add_poisson_noise(image, shape):
    """
    Add randomly distributed (fake) Poisson noise to image.

    :param image: Input image
    :type image: tf.Tensor
    :param shape: Runtime shape of the input image
    :type shape: 1D tf.Tensor or Python array

    :returns: Image with Gaussian noise
    :rtype: tf.Tensor
    """
    image = tf.to_float(image)
    noise = tf.random_normal(shape, mean=0.0, stddev=1.0)
    tf.mul(noise, tf.sqrt(image))
    tf.add(noise, image)
    image = tf.clip_by_value(tf.add(image, noise), 0, 1.)
    return image


def tf_add_gaussian_noise_and_random_blur(datum, shape, sigma_noise,
                                          min_sigma_blur, max_sigma_blur,
                                          kernel_len):
    """
    Add randomly distributed Gaussian noise (fixed standard deviation) and
    blur with a random standard deviation (between min_sigma_blur and
    max_sigma_blur) to an image.

    :param datum: Input datum (image)
    :type datum: tf.Tensor
    :param shape: Runtime shape of the input datum
    :type shape: 1D tf.Tensor or Python array
    :param sigma_noise: Standard deviation of the noise.
    :type sigma_noise: 0D tf.Tensor or Python array (float32)
    :param min_sigma_blur: Minimum standard deviation of the blur.
    :type min_sigma_blur: 0D tf.Tensor or Python array (float32)
    :param max_sigma_blur: Maximum standard deviation of the blur.
    :type max_sigma_blur: 0D tf.Tensor or Python array (float32)
    :param kernel_len: Length of the squared kernel.
    :type kernel_len: Int

    :returns: Datum (image) with Gaussian noise and blur
    :rtype: tf.Tensor
    """
    sigma_blur = tf.random_uniform((1,),
                                   minval=min_sigma_blur,
                                   maxval=max_sigma_blur)

    # no random blur for now
    sigma_blur = 2.0
    # gaussian blur kernel
    radius = kernel_len / 2.0 #int(truncate * sigma_blur + 0.5)
    x, y = np.mgrid[-radius:radius, -radius:radius]
    k = 2*np.exp(-0.5 * (x**2 + y**2) / sigma_blur**2)
    blur_kernel = k / np.sum(k)
    blur_kernel = blur_kernel.astype(np.float32)
    blur_filter = np.array(blur_kernel, dtype=np.float32)[..., None, None]
    blur_filter = np.repeat(blur_filter, shape[-1], axis=2)
    tf_blur_filter = tf.Variable(tf.convert_to_tensor(blur_filter), name="blur_kernel")

    convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1], padding='SAME')

    datum = tf.expand_dims(datum, 0)
    datum = convolve(datum, tf_blur_filter)
    datum = tf.squeeze(datum, [0])
    datum = tf_add_gaussian_noise(datum, shape, sigma_noise)
    return datum


def tf_print_trainable_variables():
    """
    Prints info name and shapes of trainable model variables which are
    available in the current session.
    """
    trainable_vars = tf.trainable_variables()
    print('Trainable variables:')
    for var in trainable_vars:
        print('Variable: %s with shape %s' % (var.name, var.get_shape()))
    total_num_vars = reduce(lambda x, y: x + y, [reduce(lambda x, y: x*y, var.get_shape()) for var in trainable_vars])
    print('Total number of variables: %s' % total_num_vars)


def tf_shuffle_batch_join(tensors_list, batch_size, capacity, do_dequeue,
                          min_after_dequeue, seed=None, enqueue_many=False,
                          shapes=None, allow_smaller_final_batch=False,
                          shared_name=None, name=None):
    """
    Custom version of tf.train.tf_shuffle_batch which correctly queues and
    dequeues data from the given pipeline depending on a tf.cond switch.

    :param tensors_list: Data pipeline tensors.
    :type tensors_list: List of Dict
    :param batch_size: Train and test batch size.
    :type batch_size: Int
    :param capacity: The maximum number of elements in the queue.
    :type capacity: Int
    :param do_dequeue: Switch for dequeuing
    :type do_dequeue: tf.Bool
    :param min_after_dequeue: Minimum number elements in the queue after a dequeue.
    :type min_after_dequeue: Int
    :param seed: Seed for the random shuffling within the queue.
    :type seed: Int
    :param enqueue_many: Whether each tensor in tensor_list is a single example.
    :type enqueue_many: Bool
    :param shapes:  The shapes for each example. Defaults to the inferred shapes for tensor_list.
    :type shapes: List
    :param allow_smaller_final_batch: Allow the final batch to be smaller if there are insufficient items left in the queue.
    :type allow_smaller_final_batch: Bool
    :param shared_name: If set, this queue will be shared under the given name across multiple sessions.
    :type shared_name: String
    :param name: A name for the operations.
    :type name: String

    :returns: A list or dictionary of tensors with the types as tensors_list
    :rtype: List or Dict
    """
    tensor_list_list = _as_tensor_list_list(tensors_list)
    with ops.name_scope(name, "shuffle_batch_join", _flatten(tensor_list_list)) as name:
        tensor_list_list = _validate_join(tensor_list_list)
        tensor_list_list, sparse_info = _store_sparse_tensors_join(
            tensor_list_list, enqueue_many)
        types = _dtypes(tensor_list_list)
        shapes = _shapes(tensor_list_list, shapes, enqueue_many)
        queue = data_flow_ops.RandomShuffleQueue(
            capacity=capacity, min_after_dequeue=min_after_dequeue, seed=seed,
            dtypes=types, shapes=shapes, shared_name=shared_name)
        _enqueue_join(queue, tensor_list_list, enqueue_many)
        full = (math_ops.cast(math_ops.maximum(0, queue.size() - min_after_dequeue),
                dtypes.float32) * (1. / (capacity - min_after_dequeue)))
        summary_name = (
            "queue/%sfraction_over_%d_of_%d_full" %
            (name, min_after_dequeue, capacity - min_after_dequeue))
        summary.scalar(summary_name, full)

        def do_dequeue_func():
            if allow_smaller_final_batch:
                dequeued = queue.dequeue_up_to(batch_size)
            else:
                dequeued = queue.dequeue_many(batch_size, name=name)
            dequeued = _restore_sparse_tensors(dequeued, sparse_info)
            return _as_original_type(tensors_list[0], dequeued)

        def do_not_dequeue_func():
            # dequeued = queue.dequeue_up_to(batch_size)
            # queue.enqueue_many(dequeued)
            if allow_smaller_final_batch:
                queue_size = queue.size()
                batch_size_tensor = tf.constant(batch_size)
                dequeued_batch_size = tf.select(tf.less(queue_size, batch_size_tensor),
                                                queue_size,
                                                batch_size_tensor)
                # return [tf.ones() for t in tensors_list[0]]
            else:
                return [tf.ones(shape=[batch_size] + t.get_shape().as_list())
                        for t in tensors_list[0]]

        dequeued = tf.cond(do_dequeue,
                           do_dequeue_func,
                           do_not_dequeue_func)

        return dequeued


def unpickle(file_path):
    """
    Unpickle binary file at file_path.

    :param file_path: Path to binary pickle file.
    :type file_path: String

    :returns: Unpickled content of file.
    :rtype: Dict
    """
    file_out = open(file_path, 'rb')
    data_dict = pickle.load(file_out)
    file_out.close()
    return data_dict


def pickle_load_all_to_list(f):
    """
    Load everything of a pickled file and return as list of dictionaries.

    :param f: Already opened file.
    :type f: file

    :returns: Content of the file.
    :rtype: List
    """
    content = []

    while True:
        try:
            l = pickle.load(f)
            if isinstance(l, list):
                content.extend(l)
            else:
                content.append(l)
        except EOFError:
            break

    return content
