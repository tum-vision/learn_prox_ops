"""
Proximal pipeline
"""
from experiment_ingredients import *

import sacred
from tabulate import tabulate
import numpy as np

import proximal as px
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

from data import load_demosaicking_data


ex = sacred.Experiment('demosaicking',
                       ingredients=[elemental_ingredient, grid_ingredient],
                       interactive=True)


##
## Elemental ingredient
##

@elemental_ingredient.config
def elemental_config():
    """
    Automatically updates the default elemental configuration.
    """
    # pylint:disable=unused-variable
    channels = 3
    scale = False


@elemental_ingredient.named_config
def optimal_BM3D():
    """
    Updated experimental configuration.
    Corresponds to FlexISP*.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'BM3D'

    sigma = 15.0
    sigma_scale = 1.0

    alpha_data = 8000.0
    alpha_tv = 0.1
    alpha_cross = 0.0



@elemental_ingredient.named_config
def optimal_DNCNN():
    """
    Updated experimental configuration.
    """
    # pylint:disable=unused-variable
    denoising_prior = 'CNN'
    cnn_model_path = 'models/DNCNN__gaussian_0.02__40-40-3__128/model.ckpt'

    sigma = 1.0

    alpha_data = 90.0
    alpha_tv = 0.01
    alpha_cross = 0.0


##
## Grid search
##

@ex.capture
def init_grid_params(elemental):
    """
    Returns the prior-specific hyper parameters for a grid search optimization.
    The parameters are given as ranges.

    :param elemental: General experiment configuration parameters
    :type elemental: Dict of List

    :returns: Update parameters for all grid configurations
    :rtype: Dict
    """
    grid_params = {}
    if elemental['denoising_prior'] == 'NLM':
        grid_params['sigma'] = np.arange(5, 105, 5)
        grid_params['sigma_scale'] = np.arange(1, 11, 1)
        grid_params['alpha_data'] = np.arange(4800, 12800, 800)
        grid_params['alpha_tv'] = np.arange(0.1, 1.1, 0.1)
        # grid_params['alpha_cross'] = [10]
    elif elemental['denoising_prior'] == 'BM3D':
        grid_params['sigma'] = [1, 5, 10, 15] + list(np.arange(20, 120, 20))
        grid_params['sigma_scale'] = [1] + list(np.arange(5, 15, 5))
        grid_params['alpha_data'] = [400] + list(np.arange(800, 4800, 800)) + [6000, 8000, 12000, 20000]
        grid_params['alpha_tv'] = np.arange(0.0, 0.6, 0.1)
        grid_params['alpha_cross'] = [0.0, 0.1]  # + list(np.arange(0.1, 1.1, 0.1))
    elif elemental['denoising_prior'] == 'CNN':
        grid_params['sigma'] = [1.0]  #[1, 5, 10] + list(np.arange(20, 120, 20))
        grid_params['alpha_data'] = list(np.arange(1, 151, 1))  #[200] + list(np.arange(600, 575, 4600)) + [10000]
        grid_params['alpha_tv'] = [0.0]  # + list(np.arange(0.1, 0.6, 0.1))
        grid_params['alpha_cross'] = [0.0]  # [0.0, 0.3]  # + list(np.arange(0.1, 1.1, 0.1))

    return grid_params


def channelwise_metric(ref, u, crop):
    """
    Computes the channel-wise PSNR between the given reference and input images.

    :param ref: Uncorrupted reference image
    :type ref: np.ndarray
    :param u: Input image
    :type u: np.ndarray
    :param crop: Amount of pixels that will be cropped at each side.
    :type crop: Int

    :returns: PSNR values for each image channel
    :rtype: List
    """
    def metric_c(c):
        metric = init_metric(ref[..., c], pad=(crop, crop))
        return metric.eval(u[..., c])

    return [metric_c(c) for c in range(ref.shape[-1])]


def experiment_wrapper(img, crop, cnn_func, grid_params, elemental):
    """
    Wrapper function which is passed to start_grid_search and handles a grid
    search for a single image and set of grid_params.

    :param img: Uncorrupted input image
    :type img: np.ndarray
    :param crop: Amount of pixels that will be cropped at each side.
    :type crop: Int
    :param cnn_func: Preinitialized deployment CNN
    :type cnn_func: function
    :param grid_params: Update parameters for specific grid configuration
    :type grid_params: Dict
    :param elemental: General experiment configuration parameters
    :type elemental: Dict

    :returns: Average (over all channels) PSNR result
    :rtype: Float
    """
    # pylint:disable=no-value-for-parameter
    elemental_copy = elemental.copy()
    elemental_copy.update(grid_params)

    f_bayer = img * bayer_mask(img.shape)
    x0 = x0_demosaic(f_bayer)

    metric = init_metric(img, pad=(crop, crop))
    u = solver(f_bayer, x0, metric, cnn_func, elemental_copy)

    psnrs = channelwise_metric(img, u, crop)
    # keep channel-wise values
    return np.mean(psnrs, axis=0)


@ex.command(unobserved=True)
def grid_search(image_name, dataset, elemental):
    """
    CML command which starts a grid search for a specific image.

    :param image_name: Image name from dataset
    :type image_name: String
    :param dataset: Dataset name
    :type dataset: String
    :param elemental: General experiment configuration parameters
    :type elemental: Dict
    """
    # pylint:disable=no-value-for-parameter
    # pylint:disable=unused-variable
    img, _, crop = load_demosaicking_data(image_name, dataset)
    wrapper_args = img, crop
    grid_params = init_grid_params()

    experiment_info = '__'.join([ex.path, image_name])
    start_grid_search(experiment_info,
                      experiment_wrapper,
                      wrapper_args,
                      grid_params)


def experiment_all_images_wrapper(dataset, cnn_func, grid_params, elemental):
    """
    Wrapper function which is passed to start_grid_search and handles a grid
    search for all images and a single a single set of grid_params.

    :param cnn_func: Preinitialized deployment CNN
    :type cnn_func: function
    :param grid_params: Update parameters for specific grid configuration
    :type grid_params: Dict
    :param elemental: General experiment configuration parameters
    :type elemental: Dict

    :returns: Average (over all images) PSNR result
    :rtype: Float
    """
    # pylint:disable=no-value-for-parameter
    elemental_copy = elemental.copy()
    elemental_copy.update(grid_params)
    data, crop = load_demosaicking_data(dataset=dataset)

    def evaluate(img):
        f_bayer = img.astype(np.float32) * bayer_mask(img.shape)
        x0 = x0_demosaic(f_bayer)
        metric = init_metric(img, pad=(crop, crop))
        u = solver(f_bayer, x0, metric, cnn_func, elemental_copy)

        return channelwise_metric(img, u, crop)

    results = [evaluate(datum['img']) for datum in data.values()]

    #average_rgb = np.mean(results, axis=0)
    return np.mean(results)


@ex.command(unobserved=True)
def grid_search_all_images(dataset, elemental):
    """
    CML command which starts a grid search for a all images of the dataset.

    :param dataset: Dataset name
    :type dataset: String
    :param elemental: General experiment configuration parameters
    :type elemental: Dict
    """
    # pylint:disable=no-value-for-parameter
    # pylint:disable=unused-variable
    grid_params = init_grid_params()
    start_grid_search(ex.path, experiment_all_images_wrapper, [dataset], grid_params)


##
## Experiment
##

@ex.config
def config():
    """
    Default experiment configuration.
    """
    # pylint:disable=unused-variable
    image_name = '1' # ["1", ..., "18"]
    dataset = 'mc_master' # ["mc_master", "kodak"]
    bayer_pattern ='rggb'


@ex.capture
def solver(f, x0, metric, cnn_func, elemental):
    """
    Solves the demosaicking problem for the given input.

    :param f: Corrupted input image
    :type f: np.ndarray
    :param x0: Predemosaicked initialization image
    :type x0: np.ndarray
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
    A = bayer_mask(f.shape)
    A_u = px.mul_elemwise(A, u)

    alpha_sumsquare = elemental['alpha_data'] / 2.0
    data = px.sum_squares(A_u - f, alpha=alpha_sumsquare)

    prox_fns = data
    if elemental['alpha_tv'] > 0.0:
        prox_fns += px.norm1(elemental['alpha_tv'] * px.grad(u, dims=2))

    if elemental['alpha_cross'] > 0.0:
        grad_u = px.grad(u, dims=2)
        grad_x0 = px.grad(x0, dims=2).value
        x0_stacked = np.array([x0, x0]).reshape(x0.shape + (2,))
        u_stacked = px.reshape(px.hstack([u, u]), x0.shape + (2,))
        cross_1 = px.vstack([px.mul_elemwise(np.roll(x0_stacked, 1, 2), grad_u),
                             px.mul_elemwise(np.roll(x0_stacked, 2, 2), grad_u)])
        cross_2 = px.vstack([px.mul_elemwise(np.roll(grad_x0, 1, 2), u_stacked),
                             px.mul_elemwise(np.roll(grad_x0, 2, 2), u_stacked)])

        prox_fns += px.norm1(0.5 * elemental['alpha_cross'] * (cross_1 - cross_2))

    prox_fns += init_denoising_prior(u,
                                     cnn_func,
                                     sigma=elemental['sigma'],
                                     sigma_scale=elemental['sigma_scale'])

    prob = init_problem(prox_fns)
    solve_problem(prob,
                  x0=x0,
                  metric=metric,
                  sigma=elemental['sigma'],
                  lin_solver_options=options)
    return np.clip(u.value, 0.0, 1.0)


@ex.capture
def bayer_mask(shape, bayer_pattern, _log):
    """
    Compute a Bayer mask which multiplied with an image produces a Bayer image
    with the given pattern.

    :param shape: Shape of the resulting mask
    :type shape: Tuple
    :param bayer_pattern: Pattern of Bayer mask
    :type bayer_pattern: String
    :param _log: Sacred logging module
    :type _log: Logger

    :returns: Bayer mask
    :rtype: np.ndarray
    """
    red_mask = np.zeros((shape[0], shape[1]))
    green_mask = np.zeros((shape[0], shape[1]))
    blue_mask = np.zeros((shape[0], shape[1]))

    if bayer_pattern == 'rggb':
        red_mask[::2, ::2] = 1
        green_mask[1::2, ::2] = 1
        green_mask[::2, 1::2] = 1
        blue_mask[1::2, 1::2] = 1
    else:
        _log.error('No valid bayer pattern selected.')
        exit()

    return np.array([red_mask, green_mask, blue_mask]).transpose((1, 2, 0))


@ex.capture
def x0_demosaic(f_bayer, bayer_pattern):
    """
    Applies a simple demosaicking step to

    :param f_bayer: Input bayer image
    :type f_bayer: np.ndarray
    :param bayer_pattern: Pattern of the f_bayer image
    :type bayer_pattern: String

    :returns: Demosaicked output
    :rtype: np.ndarray
    """
    x0 = np.clip(demosaicing_CFA_Bayer_Malvar2004(np.sum(f_bayer, axis=2),
                                          pattern=bayer_pattern), 0.0, 1.0)
    return x0


@ex.automain
def main(image_name, dataset, elemental, _log):
    """
    Default command which solves the demosaicking problem for the given dataset
    and image_name.

    :param image_name: Name of the image
    :type image_name: String
    :param dataset: Dataset name
    :type dataset: String
    :param elemental: General experiment configuration parameters
    :type elemental: Dict
    :param _log: Sacred logging module
    :type _log: Logger
    """
    # pylint:disable=no-value-for-parameter
    img, crop = load_demosaicking_data(image_name, dataset)
    f_bayer = img.astype(np.float32) * bayer_mask(img.shape)
    x0 = x0_demosaic(f_bayer)

    cnn_func = init_cnn_func() if elemental['denoising_prior'] == 'CNN' else None
    metric = init_metric(img, pad=(crop, crop))

    # TODO: refactor
    # edgetaper to better handle circular boundary condition
    # f_bayer += 1
    # x0 += 1.0
    u = solver(f_bayer, x0, metric, cnn_func)
    # f_bayer -= 1
    # x0 -= 1.0
    # u -= 1.0

    print_config()

    _log.info("Input f_bayer PSNR: %f" % metric.eval(f_bayer))
    _log.info("Input x0 PSNR: %f" % metric.eval(x0))
    _log.info("Final PSNR: %f" % metric.eval(u))

    psnrs = channelwise_metric(img, u, crop)
    for c, psnr in enumerate(psnrs):
        _log.info("Final PSNR Channel %i: %f" % (c, psnr))

    if elemental['show_plot']:
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(img)
        ax2.imshow(f_bayer)
        ax3.imshow(u)
        plt.show()
