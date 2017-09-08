"""
Proximal pipeline for grayscale image deblurring
"""
from experiment_ingredients import *

import sacred
import collections
from tabulate import tabulate
import numpy as np

import proximal as px

from data import load_deblurring_grey_data


ex = sacred.Experiment('deblurring',
                       ingredients=[elemental_ingredient, grid_ingredient])


##
## Elemental ingredient
##

@elemental_ingredient.named_config
def optimal_BM3D_experiment_a():
    """
    Updated experimental configuration.
    Corresponds to FlexISP*.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'BM3D'
    sigma = 20.0
    sigma_scale = 1.0

    alpha_data = 200.0
    alpha_tv = 0.3


@elemental_ingredient.named_config
def optimal_BM3D_experiment_b():
    """
    Updated experimental configuration.
    Corresponds to FlexISP*.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'BM3D'
    sigma = 5.
    sigma_scale = 20.0

    alpha_data = 1400.0
    alpha_tv = 0.1


@elemental_ingredient.named_config
def optimal_BM3D_experiment_c():
    """
    Updated experimental configuration.
    Corresponds to FlexISP*.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'BM3D'
    sigma = 1.0
    sigma_scale = 1.0

    alpha_data = 40.0
    alpha_tv = 0.0


@elemental_ingredient.named_config
def optimal_BM3D_experiment_d():
    """
    Updated experimental configuration.
    Corresponds to FlexISP*.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'BM3D'
    sigma = 10.0
    sigma_scale = 15.0

    alpha_data = 600.0
    alpha_tv = 0.0


@elemental_ingredient.named_config
def optimal_BM3D_experiment_e():
    """
    Updated experimental configuration.
    Corresponds to FlexISP*.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'BM3D'
    sigma = 5.
    sigma_scale = 1.0

    alpha_data = 150.0
    alpha_tv = 0.1


@elemental_ingredient.named_config
def optimal_DNCNN_experiment_a():
    """
    Updated experimental configuration.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'CNN'
    cnn_model_path = 'data/tf_models/DNCNN__gaussian_0.02__40-40-1__128/model.ckpt'
    sigma = 1.0
    alpha_data = 2.0
    alpha_tv = 0.0


@elemental_ingredient.named_config
def optimal_DNCNN_experiment_b():
    """
    Updated experimental configuration.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'CNN'
    cnn_model_path = 'data/tf_models/DNCNN__gaussian_0.02__40-40-1__128/model.ckpt'
    sigma = 1.0
    alpha_data = 75.0
    alpha_tv = 0.0


@elemental_ingredient.named_config
def optimal_DNCNN_experiment_c():
    """
    Updated experimental configuration.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'CNN'
    cnn_model_path = 'data/tf_models/DNCNN__gaussian_0.02__40-40-1__128/model.ckpt'
    sigma = 1.0
    alpha_data = 4.0
    alpha_tv = 0.0


@elemental_ingredient.named_config
def optimal_DNCNN_experiment_d():
    """
    Updated experimental configuration.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'CNN'
    cnn_model_path = 'data/tf_models/DNCNN__gaussian_0.02__40-40-1__128/model.ckpt'
    sigma = 1.0
    alpha_data = 73.0
    alpha_tv = 0.0


@elemental_ingredient.named_config
def optimal_DNCNN_experiment_e():
    """
    Updated experimental configuration.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'CNN'
    cnn_model_path = 'data/tf_models/DNCNN__gaussian_0.02__40-40-1__128/model.ckpt'
    sigma = 1.0
    alpha_data = 23.0
    alpha_tv = 0.0


##
## Grid search
##

def experiment_wrapper(experiment_name, image_name, cnn_func, grid_params, elemental):
    """
    Wrapper function which is passed to start_grid_search and handles a grid
    search for a single a single set of grid_params.

    :param experiment_name: Name of the experiment a-e: experiment_*
    :type experiment_name: String
    :param image_name: Name of the image
    :type image_name: String
    :param cnn_func: Preinitialized deployment CNN
    :type cnn_func: function
    :param grid_params: Update parameters for specific grid configuration
    :type grid_params: Dict
    :param elemental: General experiment configuration parameters
    :type elemental: Dict

    :returns: Average (image- and experiment-wise) PSNR result
    :rtype: Float
    """
    # pylint:disable=no-value-for-parameter
    elemental_copy = elemental.copy()
    elemental_copy.update(grid_params)
    data, crop = load_deblurring_grey_data()

    def evaluate(image_name, images, kernel_img):
        metric = init_metric(images['img'], pad=(crop, crop))

        u = solver(images['f'], kernel_img, metric, cnn_func, elemental_copy)
        return metric.eval(u)

    if experiment_name is not None:
        data = {experiment_name: data[experiment_name]}
        if image_name is not None:
            data[experiment_name] = {image_name: data[experiment_name][image_name],
                                     'kernel_img': data[experiment_name]['kernel_img']}

    averages = []
    for experiment_images in data.values():
        kernel_img = experiment_images['kernel_img']
        results = collections.OrderedDict((name, evaluate(name, images, kernel_img))
                                          for name, images in experiment_images.items()
                                          if not name == 'kernel_img')

        average = np.mean(list(results.values()), axis=0)
        averages.append(average)

    average = np.squeeze(np.mean(averages, axis=0))
    return average


@ex.command(unobserved=True)
def grid_search(experiment_name, image_name, elemental, _log):
    """
    CML command which starts a grid search. If image_name can not be set without
    experiment_name.

    :param experiment_name: Name of the experiment a-e: experiment_*
    :type experiment_name: String
    :param image_name: Name of the image
    :type image_name: String
    :param elemental: General experiment configuration parameters
    :type elemental: Dict
    :param _log: Sacred logging module
    :type _log: Logger
    """
    # pylint:disable=no-value-for-parameter
    # pylint:disable=unused-variable
    grid_params = {}
    if elemental['denoising_prior'] == 'NLM':
        grid_params['sigma'] = [1, 5] + list(np.arange(10, 110, 10))
        grid_params['sigma_scale'] = [1] + list(np.arange(5, 25, 5))
        grid_params['alpha_data'] = [20, 60] + list(np.arange(100, 900, 100)) + [1200, 1600, 2000, 4000, 6000]#, 8000, 10000]
        grid_params['alpha_tv'] = [0.0] + list(np.arange(0.1, 0.6, 0.1))
    if elemental['denoising_prior'] == 'BM3D':
        grid_params['sigma'] = [1, 5] + list(np.arange(10, 110, 10))
        grid_params['sigma_scale'] = [1] + list(np.arange(5, 20, 5))
        grid_params['alpha_data'] = [20, 60] + list(np.arange(100, 900, 100)) + [1200, 1600, 2000, 4000, 6000]#, 8000, 10000]
        grid_params['alpha_tv'] = [0.0] + list(np.arange(0.1, 0.8, 0.1))
    elif elemental['denoising_prior'] == 'CNN':
        grid_params['sigma'] = [1]#, 2, 3, 4] + list(np.arange(5, 105, 5))
        grid_params['alpha_data'] = list(np.arange(1, 201, 1)) #+ list(np.arange(20, 10020, 20)) #[20, 60] + list(np.arange(100, 900, 100)) + [1200, 1600, 2000, 4000, 6000, 8000, 10000]
        grid_params['alpha_tv'] = [0.0] + list(np.arange(0.1, 1.1, 0.1)) #, 0.05] + list(np.arange(0.1, 0.6, 0.1))

    # image-wise searches only for specific experiments
    if experiment_name is None and image_name is not None:
        _log.error("Specify experiment_name for image_name.")
        exit()

    # if no experiment is specified search for all experiments (individually)
    if experiment_name is None:
        experiment_names = ["experiment_" + end for end in ['a', 'b', 'c', 'd', 'e']]
    else:
        experiment_names = [experiment_name]

    psnrs = []
    # TODO: grid search on all experiments and images jointly.
    for experiment_name in experiment_names:
        wrapper_args = [experiment_name, image_name]
        experiment_info = "__".join(list(filter(None.__ne__, [ex.path,
                                                experiment_name,
                                                image_name])))
        psnrs.append(start_grid_search(experiment_info,
                                       experiment_wrapper,
                                       wrapper_args,
                                       grid_params))

    # TODO: move experiment wise search and final printing to wrapper.
    # no average of averages for single experiment searches
    if len(psnrs) > 1:
        msg = "AVG of AVGS: %f" % np.mean(psnrs)
        print(msg)


##
## Experiment
##

@ex.config
def config():
    """
    Default experiment configuration.
    """
    # pylint:disable=unused-variable
    experiment_name = None  # ["experiment_a", ..., "experiment_e"]
    image_name = None  # ['barbara', 'boat', 'cameraman', 'couple', 'fingerprint', 'hill', 'house', 'lena', 'man', 'montage', 'peppers']
    mlp_psnrs = {"experiment_a": 24.76,
                 "experiment_b": 27.23,
                 "experiment_c": 22.20,
                 "experiment_d": 22.75,
                 "experiment_e": 29.42}

@ex.capture
def solver(f, kernel_img, metric, cnn_func, elemental):
    """
    Solves the deblurring problem for the given input and kernel image.

    :param f: Corrupted input image
    :type f: np.ndarray
    :param kernel_img: Blur kernel
    :type kernel_img: np.ndarray
    :param metric: Preinitialized metric
    :type metric: proximal.utils.metrics
    :param cnn_func: Preinitialized deployment CNN
    :type cnn_func: function
    :param elemental: General experiment configuration parameters
    :type elemental: Dict

    :returns: Reconstructed output image
    :rtype: np.ndarray
    """
    # pylint:disable=no-value-for-parameter
    options = px.cg_options(tol=1e-4, num_iters=100, verbose=True)

    u = px.Variable(f.shape)

    alpha_sumsquare = elemental['alpha_data'] / 2.0
    A_u = px.conv(kernel_img, u)

    prox_fns = px.sum_squares(A_u - f, alpha=alpha_sumsquare)
    if elemental['alpha_tv'] > 0.0:
        prox_fns += px.norm1(elemental['alpha_tv'] * px.grad(u))
    prox_fns += init_denoising_prior(u,
                                     cnn_func,
                                     sigma=elemental['sigma'],
                                     sigma_scale=elemental['sigma_scale'])

    prob = init_problem(prox_fns)
    solve_problem(prob,
                  x0=f.copy(),
                  metric=metric,
                  sigma=elemental['sigma'],
                  lin_solver_options=options)

    return np.clip(u.value, 0.0, 1.0)


@ex.automain
def main(experiment_name, image_name, elemental, _log):
    """
    Default command which solves the given deblurring experiment for image_name.

    :param experiment_name: Name of the experiment a-e: experiment_*
    :type experiment_name: String
    :param image_name: Name of the image
    :type image_name: String
    :param elemental: General experiment configuration parameters
    :type elemental: Dict
    :param _log: Sacred logging module
    :type _log: Logger
    """
    # pylint:disable=no-value-for-parameter
    if experiment_name is None and image_name is not None:
        _log.error("Specify experiment_name for image_name.")
        exit()
    f, img, kernel_img, crop = load_deblurring_grey_data(experiment_name,
                                                         image_name)

    cnn_func = init_cnn_func() if elemental['denoising_prior'] == 'CNN' else None
    metric = init_metric(img, pad=(crop, crop))
    u = solver(f, kernel_img, metric, cnn_func)

    print_config()

    _log.info("Input PSNR: %f" % metric.eval(f))
    _log.info("Final PSNR: %f" % metric.eval(u))

    if elemental['show_plot']:
        fig = plt.figure(figsize=(20, 5))
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax1.set_title('image')
        ax2.set_title('f')
        ax3.set_title('our')
        ax1.imshow(img, cmap='gray')
        ax2.imshow(f, cmap='gray')
        ax3.imshow(u, cmap='gray')
        plt.show()
