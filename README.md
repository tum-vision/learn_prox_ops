Learning Proximal Operators
================
This repository provides the implementation of our paper **Learning Proximal Operators: Using Denoising Networks for Regularizing Inverse Imaging Problems** (Tim Meinhardt, Michael Möller, Caner Hazirbas, Daniel Cremers, ICCV 2017) [https://arxiv.org/abs/1704.03488]. All results presented in our work were produced with this code.

Additionally we provide a TensorFlow implementation of the denoising convolutional neural network (_DNCNN_) introduced in **Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising** [https://arxiv.org/abs/1608.03981].

Installation
-------------------
1. Install the following packages for Python 3.6:
    1. `pip3 install -r requirements.txt`
    2. ProxImaL: `pip3 install git+https://github.com/timmeinhardt/ProxImaL@learn_prox_ops`
    3. PyBM3D:
        1. with CUDA: `pip3 install git+https://github.com/timmeinhardt/pybm3d_gpu`
        2. or without CUDA: `pip3 install git+https://github.com/timmeinhardt/pybm3d@learn_prox_ops`
    3. TensorFlow:
        1. with CUDA: `pip3 install tensorflow-gpu==1.3.0`
        2. or without CUDA: `pip3 install tensorflow==1.3.0`
    4. OpenCV:
        1. `pip3 install opencv-python==3.3.0.10`
        2. or for faster _NLM_ denoising compile OpenCV 3.3.0 manually with CUDA support and Python 3.6 bindings
2. Download the demosaicking (_McMaster_ and _Kodak_) and the greyscale deblurring datasets with `data/download_datasets.sh`.
3. Download pretrained _DNCNN_ models with `data/download_tf_models.sh`
4. (**Optional**, for faster computation and training _DNCNN_ models) Install CUDA and set the CUDA_HOME environment variable.
5. (**Optional**, for optimal results and faster computation) Install [Halide](http://halide-lang.org/) and set the HALIDE_PATH environment variable.
6. (**Optional**, for training a _DNCNN_) Download BSDS500 training data with `data/download_cnn_training_datasets.sh`.

Run an Experiment
-------------------
The evaluation of our method included two exemplary linear inverse problems, namely Bayer color demosaicking and grayscale deblurring. In order to configure, organize, log and reproduce our computational experiments we structured the problems with the [Sacred](http://sacred.readthedocs.io/en/latest/index.html) framework.

For a detailed explanation on a typical Sacred interface please read its documentation. We implemented two Sacred _ingredients_ (`elemental_ingredient, grid_ingredient`) which are both injected into our experiments. Among other things each of the experiments consists of multiple command line executable Sacred _commands_.

If everything is setup correctly the `print_config` command for example prints the current configuration scope by executing:

`python src/experiment_deblurring.py print_config`

A typical run with a preset configuration scope for optimal _DNCNN_ parameters is executed with (`automain` command):

`python src/experiment_deblurring.py with experiment_name=experiment_a image_name=barbara elemental.optimal_DNCNN_experiment_a`


Hyperparameter Grid Search
-------------------
We conducted multiple exhaustive grid searches to establish the optimal hyper parameters for both experiments. The set of searchable `grid_params` has to be set in the respective experiment file. A search for the optimal demosaicking parameters for all images and the BM3D denoising prior is started by executing:

`python src/experiment_demosaicking.py grid_search_all_images with elemental.denoising_prior=BM3D`

The `grid_search.param_dicts_file_path` configuration parameter can be used to continue a previous search.


Training a _DNCNN_
-------------------

The training of the denoising convolutional neural network which we applied as a learned denoising prior was implemented with TensorFlow. With the help of command line `tf.app.flags` we provide full control over the training procedure. The single channel model provided with this repository was trained by executing:

`python src/tf_solver.py --sigma_noise 0.02 --batch_size 128 --network DNCNN --channels 1 --pipeline bsds500 --device_name /gpu:0 --train_epochs 100`


Publication
-------------------
If you use this software in your research, please cite our publication:

```
@article{DBLP:journals/corr/Meinhardt0HC17,
    author    = {Tim Meinhardt and
                 Michael Moeller and
                 Caner Hazirbas and
                 Daniel Cremers},
    title     = {Learning Proximal Operators: Using Denoising Networks for Regularizing Inverse Imaging Problems},
    journal   = {CoRR},
    volume    = {abs/1704.03488},
    year      = {2017},
    url       = {http://arxiv.org/abs/1704.03488},
    timestamp = {Wed, 07 Jun 2017 14:40:59 +0200},
    biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/Meinhardt0HC17},
    bibsource = {dblp computer science bibliography, http://dblp.org}}
```
