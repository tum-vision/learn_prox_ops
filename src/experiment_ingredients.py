"""
Proximal pipeline
"""
import os
import itertools
import multiprocessing
import pickle
from joblib import Parallel, delayed

# if no graphics support (e.g. on server) use non-graphic backend
import matplotlib
try:
    import tkinter
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

import sacred
import numpy as np
import proximal as px
from proximal.utils.utils import CUDA_AVAILABLE, Impl
from proximal.utils.metrics import psnr_metric, ssim_metric

from tf_solver import Deployer, FLAGS
from utilities import pickle_load_all_to_list


__all__ = ('elemental_ingredient',
           'print_config',
           'init_denoising_prior',
           'grid_ingredient',
           'init_cnn_func',
           'init_metric',
           'init_problem',
           'solve_problem',
           'start_grid_search',
           'plt')

#
# This is the "elemental" ingredient to all our experiments. It provides a
# default configuration and several helper functions.
#
elemental_ingredient = sacred.Ingredient('elemental')


@elemental_ingredient.config
def elemental_config():
    """
    Default experiment configuration.

    Remark: The notation of the alpha_* is different than in our paper but in
    consistence with the interface of the ProxImal framework.
    """
    # pylint:disable=unused-variable
    sigma = 1.0
    sigma_scale = 6.0 # default proximal value
    tau = None
    theta = None
    alpha_data = 1.0
    alpha_tv = 0.0
    alpha_cross = 0.0
    alpha_denoising = 1.0

    max_iters = 30
    conv_check = 1 # check convergence each conv_check iteration
    eps_rel = 1e-3
    eps_abs = 1e-3
    scale = True
    try_split = True
    absorb = True
    merge = True
    metric_type = "PSNR" # ["PSNR", "SSIM"]
    metric_decimals = 4
    conv_mode = "metric" # ["residual", "metric", "metric_all_iter"]
    verbose = 0 # [0, 1, 2]
    print_config = True
    implem = 'numpy' # ["halide", "numpy"]
    show_plot = False
    denoising_prior = 'CNN' # ["CNN", "NLM", "BM3D"]
    cnn_model_path = "" # "XXX/model.ckpt"

    patch_size = 4
    nlm_search_window_size = 15
    channels = 1 # [1, 3]

    if CUDA_AVAILABLE:
        device_name = '/gpu:0' # if cuda is available "/cpu:0" else "/gpu:0"
    else:
        device_name = '/cpu:0'
    img_log_dir = "data/experiment_log" # path for logging intermediate img results if verbose > 1


@elemental_ingredient.command(unobserved=True)
def print_config(print_config, _run, _log):
    """
    Prints configuration parameters.

    :param _run: Sacred Run object
    :type _run: Sacred.Run
    :param _log: Sacred logging module
    :type _log: Logger
    """
    # print config only for levels lower than warning
    if print_config and _log.getEffectiveLevel() < 30:
        sacred.commands.print_config(_run)


@elemental_ingredient.capture
def init_denoising_prior(u, cnn_func, denoising_prior, sigma,
                         sigma_scale, patch_size, nlm_search_window_size,
                         alpha_denoising, _log):
    """
    Initializes the specified denoising prior which then can be integrated into
    the problem.

    :param u: Input variable that later contains the solution
    :type u: proximal.Variable
    :param cnn_func: Preinitialized deployment CNN
    :type cnn_func: function
    :param denoising_prior: Specified the type of denoising prior i.e. denoising operation
    :type denoising_prior: String
    :param sigma: Expected noise standard deviation
    :type sigma: Float
    :param sigma_scale: Scale for the noise standard deviation
    :type sigma_scale: Float
    :param patch_size: NLM and BM3D denoising patch size
    :type patch_size: Int
    :param nlm_search_window_size: Size of the NLM search window
    :type nlm_search_window_size: Int
    :param alpha_denoising: Weight coefficient for the denoising prior
    :type alpha_denoising: Float
    :param _log: Sacred logging module
    :type _log: Logger

    :returns: Prior in form of a proxable function.
    :rtype: proximal.ProxFn
    """
    if denoising_prior == 'CNN':
        return px.prox_black_box(u, cnn_func, alpha=alpha_denoising)
    elif denoising_prior == 'NLM':
        return px.patch_NLM(u,
                            alpha=alpha_denoising,
                            sigma_fixed=np.sqrt(1. / sigma),
                            sigma_scale=sigma_scale,
                            templateWindowSizeNLM=patch_size,
                            searchWindowSizeNLM=nlm_search_window_size,
                            prior=0)
    elif denoising_prior == 'BM3D':
        return px.patch_BM3D(u,
                             sigma_fixed=np.sqrt(1. / sigma),
                             sigma_scale=sigma_scale,
                             patch_size=patch_size,
                             alpha=alpha_denoising)
    else:
        _log.error('No valid denoising_prior selected.')
        exit()


@elemental_ingredient.capture
def init_cnn_func(cnn_model_path, device_name, channels, _log):
    """
    Initializes the denoising CNN function which reduces the overhead during
    grid searches.

    :param cnn_model_path: File path to the trained denoising model
    :type cnn_model_path: String
    :param device_name: Name of the device the model is loaded to.
    :type device_name: String
    :param channels: Number of image channels. Must be given for the comp graph.
    :type channels: Int
    :param _log: Sacred logging module
    :type _log: Logger

    :returns: Preinitialized deployment CNN
    :rtype: function
    """
    if not cnn_model_path:
        _log.error('No cnn_model_path given.')
        exit()
    FLAGS.device_name = device_name
    FLAGS.model_path = cnn_model_path
    FLAGS.channels = channels
    nn_deployer = Deployer(FLAGS)

    def cnn(x):
        if len(x.shape) == 2:
            x = x[np.newaxis, ..., np.newaxis]
        elif len(x.shape) == 3:
            # x = x[np.newaxis, ...].transpose((3, 1, 2, 0))
            # res = nn_deployer.deploy(x)['output_clipped']
            # res = np.squeeze(res.transpose((3, 1, 2, 0)))
            x = x[np.newaxis, ...]
        else:
            _log.error('No valid data dimension.')
            exit()

        res = np.squeeze(nn_deployer.deploy(x)['output_clipped'])
        return res

    return cnn


@elemental_ingredient.capture
def init_metric(ref, pad, metric_decimals, metric_type, _log):
    """
    Initializes a metric function.

    :param ref: Uncorrupted reference image
    :type ref: np.ndarray
    :param pad: Right-Left and Up-Down padding
    :type pad: Tuple
    :param metric_decimals: Number of printed decimals
    :type metric_decimals: Int
    :param metric_type: Specified metric type ("PSNR" or "SSIM")
    :type metric_type: String
    :param _log: Sacred logging module
    :type _log: Logger

    :returns: Preinitialized metric
    :rtype: proximal.utils.metrics
    """
    if metric_type == "PSNR":
        return psnr_metric(ref, pad=pad, decimals=metric_decimals)
    elif metric_type == "SSIM":
        return ssim_metric(ref, pad=pad, decimals=metric_decimals)
    else:
        _log.error('No valid metric selected.')
        exit()



@elemental_ingredient.capture
def init_problem(prox_fns, implem, scale, try_split, merge, absorb):
    """
    Initializes a problem.

    :param prox_fns: Proxable functions which define the problem.
    :type prox_fns: List
    :param implem: Implementation modus ("halide" or "numpy")
    :type implem: String
    :param scale: If the problem should be scaled
    :type scale: Bool
    :param try_split: If a problem split should be tried
    :type try_split: Bool
    :param merge: If problem should be merged
    :type merge: Bool
    :param absorb: If linear operators should be absorbed
    :type absorb: Bool

    :returns: Initialized problem
    :rtype: proximal.Problem
    """
    prob = px.Problem(prox_fns,
                      implem=Impl[implem],
                      scale=scale,
                      try_split=try_split,
                      merge=merge,
                      absorb=absorb)

    return prob


@elemental_ingredient.capture
def solve_problem(problem, x0, lin_solver_options, metric, conv_mode, tau, sigma,
                  theta, max_iters, conv_check, eps_abs, eps_rel, verbose, img_log_dir):
    """
    Solve a previously initialized problem.

    :param problem: Previously initialized problem
    :type problem: proximal.Problem
    :param x0: Initialization for the first iterate
    :type x0: np.ndarray
    :param metric: Preinitialized metric
    :type metric: proximal.utils.metrics
    :param conv_mode: Convergence mode for the solver. ("residual", "metric" or "metric_all_iter")
    :type conv_mode: String
    :param tau: Step size for the dual variable
    :type tau: Float
    :param sigma:  Step size for the primal variable
    :type sigma: Float
    :param theta: Step size for the interpolation step
    :param theta: Float
    :param max_iters: Maximum number of iterations
    :type max_iters: Int
    :param conv_check: Check convergence each conv_check iteration
    :type conv_check: Int
    :param eps_abs: Absolute error epsilon for residual convergence
    :type eps_abs: Float
    :param eps_rel: Relative error epsilon for residual convergence
    :type eps_rel: Float
    :param verbose: Logging verbosity (0,1 or 2)
    :type verbose: Int
    :param img_log_dir: Path to directory for logging intermediate iteration results.
    :type img_log_dir: String
    """
    problem.solve(solver="pc",
                  tau=tau,
                  sigma=sigma,
                  theta=theta,
                  max_iters=max_iters,
                  conv_check=conv_check,
                  eps_abs=eps_abs,
                  eps_rel=eps_rel,
                  x0=x0,
                  lin_solver_options=lin_solver_options,
                  metric=metric,
                  verbose=verbose,
                  img_log_dir=img_log_dir,
                  conv_mode=conv_mode)


grid_ingredient = sacred.Ingredient('grid_search',
                                    ingredients=[elemental_ingredient])


@grid_ingredient.config
def grid_config():
    """
    Grid search configuration.
    """
    # pylint:disable=unused-variable
    log_path = None
    param_dicts_file_path = None # plot or continue previous grid search
    info_msg = ''
    n_jobs = None


@grid_ingredient.command(unobserved=True)
def plot(param_dicts_file_path, _log):
    """
    Plot a previous grid search.

    :param param_dicts_file_path: Path to pickled file with previous parameters and results
    :type param_dicts_file_path: String
    :param _log: Sacred logging module
    :type _log: Logger
    """
    if param_dicts_file_path is None:
        _log.error('No grid search param_dicts_file_path specified.')
        exit()

    with open(param_dicts_file_path, 'rb') as f:
        data = pickle_load_all_to_list(f)
    psnrs = [d['psnr'] for d in data]

    keys = [k for k in data[0].keys() if k != 'psnr']
    changed_keys_order = []
    for d, next_d in zip(data, data[1:]):
        for key in keys:
            if d[key] != next_d[key]:
                if key not in changed_keys_order:
                    changed_keys_order.append(key)
                    break

    print("Order of changed grid params first to last: %s" % changed_keys_order)
    max_id = np.argmax(psnrs)
    print("Max id: %i" % max_id)
    print("Max params: %s" % data[max_id])

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(psnrs)
    # for x, y in enumerate(psnrs):
    #     circl = plt.Circle((x, y), 1.0)
    #     ax.add_patch(circl)
    plt.show()


def grid_search_wrapper(experiment_wrapper, experiment_wrapper_args, cnn_func,
                        grid_params, elemental, param_dicts_save_path):
    """
    Wrapper for each call of the experiment_wrapper in start_grid_search which
    executes the grid search for a particular parameter set and saves the result
    in the file at param_dicts_save_path.

    :param experiment_wrapper: Experiment specific wrapper functions
    :type experiment_wrapper: function
    :param experiment_wrapper_args: Arguments for experiment_wrapper
    :type experiment_wrapper_args: Tuple or List
    :param cnn_func: Preinitialized deployment CNN
    :type cnn_func: function
    :param grid_params: Update parameters for specific grid configuration
    :type grid_params: Dict
    :param elemental: General experiment configuration parameters
    :type elemental: Dict
    :param param_dicts_save_path: Path to the current file with parameters and results
    :type param_dicts_save_path: String

    :returns: PSNR result
    :rtype: Float
    """
    psnr = experiment_wrapper(*experiment_wrapper_args,
                              cnn_func=cnn_func,
                              grid_params=grid_params,
                              elemental=elemental)
    with open(param_dicts_save_path, 'a+b') as f:
        pickle.dump(dict(params, **{'psnr': psnr}), f)
    return psnr


@grid_ingredient.capture
def start_grid_search(experiment_info, experiment_wrapper, experiment_wrapper_args,
                      grid_params, info_msg, log_path, param_dicts_file_path,
                      n_jobs, _seed, _log, _config):
    """
    Starts a grid search.

    :param experiment_info: Info about the experiment which will be logged.
    :type experiment_info: String
    :param experiment_wrapper: Experiment specific wrapper functions
    :type experiment_wrapper: function
    :param experiment_wrapper_args: Arguments for experiment_wrapper
    :type experiment_wrapper_args: Tuple or List
    :param grid_params: Update parameters for specific grid configuration
    :type grid_params: Dict
    :param info_msg: Additional info message which will be logged.
    :type info_msg: String
    :param log_path: Path to text file where final results will be logged.
    :type log_path: String
    :param param_dicts_file_path: Path to pickled file with previous parameters and results
    :type param_dicts_file_path: String
    :param n_jobs: Number of jobs (threads) for parallel grid search.
    :param _seed: Run specific random integer.
    :type _seed: Int
    :param _log: Sacred logging module
    :type _log: Logger
    :param _config: Entire configuration space.
    :type _config: Dict

    :returns: PSNR result
    :rtype: Float
    """
    # pylint:disable=no-value-for-parameter
    # TODO: refactor
    elemental = _config['elemental']
    n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs
    cnn_func = init_cnn_func() if elemental['denoising_prior'] == 'CNN' else None
    param_dicts_save_path = os.path.join('data/grid_search/param_dicts',
                                            str(_seed) + '.p')

    if log_path is None:
        log_path = os.path.join("data/grid_search",
                                experiment_info.split("_")[0] + \
                                "_" + \
                                elemental['denoising_prior'].lower() + \
                                ".txt")

    # uniquify grid param sets
    grid_params = {k: sorted(list(set(v))) for k, v in grid_params.items()}
    params_iter = [{k: p[i] for i, k in enumerate(grid_params.keys())}
                   for p in itertools.product(*grid_params.values())]

    #
    # logging
    #
    info_str = ("\n\tExperiment: %s\n"
                "\tDenoising prior: %s\n"
                "\tinfo_msg: %s\n"
                "\tseed: %i\n"
                "\tElemental config: %s\n"
                "\tNumber of parameter sets to iterate: %i\n") % \
                (experiment_info,
                 elemental['denoising_prior'],
                 info_msg,
                 _seed,
                 elemental,
                 len(params_iter))
    for name, values in grid_params.items():
        info_str += "\t%s: [%f, ..., %f]\n" % (name, values[0], values[-1])

    _log.info(info_str)
    with open(log_path, 'a+') as f:
        f.write(info_str.replace('\t', ''))

    #
    # search
    #

    # continue previous search
    if param_dicts_file_path is not None:

        def is_dict_in(dict, dict_list):
            for d in dict_list:
                if False not in [dict[k] == v for k, v in d.items()]:
                    return True
            return False

        with open(param_dicts_file_path, 'rb') as f:
            prev_results = pickle_load_all_to_list(f)
        prev_psnrs = [res_dict['psnr'] for res_dict in prev_results]
        prev_params_iter = [{k: v for k, v in res_dict.items() if k != 'psnr'}
                            for res_dict in prev_results]

        #for param in prev_params_iter:
        #    param["alpha_data"] *= 4

        if False in [k in prev_params_iter[0].keys() for k in grid_params.keys()]:
            _log.error("Previous and current param keys are not equal.")
            exit()

        params_iter = [params for params in params_iter
                       if not is_dict_in(params, prev_params_iter)]
        _log.info("Number of parameter sets without previous: %i" % len(params_iter))

        # save previous results in new param dicts file
        with open(param_dicts_save_path, 'a+b') as f:
            for r_dict in prev_results:
                pickle.dump(r_dict, f)
    else:
        prev_psnrs = []
        prev_params_iter = []

    jobs = [delayed(grid_search_wrapper)(experiment_wrapper, experiment_wrapper_args, cnn_func, params, elemental, param_dicts_save_path)
            for params in params_iter]
    psnrs = Parallel(backend="threading", n_jobs=n_jobs, verbose=11)(jobs)

    psnrs += prev_psnrs
    params_iter += prev_params_iter
    max_idx = np.argmax(psnrs)

    #
    # logging
    #
    result_str = 'PSNR: %f PARAMS: %s' % (psnrs[max_idx], params_iter[max_idx])
    _log.info(result_str)

    with open(log_path, 'a+') as f:
        if elemental['denoising_prior'] != "CNN" and "sigma_scale" in params_iter[max_idx]:
            effective_noise_sigma = np.sqrt(1 / params_iter[max_idx]['sigma']) / 30 \
                                            * params_iter[max_idx]['sigma_scale']
            result_str += " effective_noise_sigma: %s\n" % str(effective_noise_sigma)
        f.write(result_str + '\n')
    return psnrs[max_idx]
