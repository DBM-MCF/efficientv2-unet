[![License](https://img.shields.io/pypi/l/efficientv2-unet.svg?color=green)](https://github.com/DBM-MCF/efficientv2-unet/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/efficientv2-unet.svg?color=green)](https://pypi.org/project/efficientv2-unet)
[![Python Version](https://img.shields.io/pypi/pyversions/efficientv2-unet.svg?color=green)](https://python.org)
[![CI](https://github.com/DBM-MCF/efficientv2-unet/actions/workflows/ci.yml/badge.svg)](https://github.com/DBM-MCF/efficientv2-unet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DBM-MCF/efficientv2-unet/branch/main/graph/badge.svg)](https://codecov.io/gh/DBM-MCF/efficientv2-unet)

A U-Net implementation of the EfficientNetV2.

# EfficientV2-UNet
This package is a U-Net implementation of the [EfficientNetV2](https://arxiv.org/abs/2104.00298), using TensorFlow.

EfficientNetV2 improves speed and parameter efficiency. This implementation also uses the ImageNet weights for training new models.

It is intended for segmentation of histological images (RGB) that are **not** saved in pyramidal file format (WSI).

The output segmentation are foreground / background. Multi-class segmentation is not (yet) possible.

It works on TIF images (and probably also PNG).

# Installation

1. Create a python environment (e.g. with conda, python=3.9 and 3.10 work), in a CLI:

    `conda create --name ev2unet_env python=3.9`

2. Activate environment:

    `conda activate ev2unet_env`

3. GPU support for **Windows** *(Non GPU installations not extensively tested)*

    a. Install the cudatoolkit and cudnn, e.g. with conda:

    `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
    
      - Windows requires a specific version of TensorFlow (i.e. v2.10.1, higher versions are not supported on Windows), which will be installed by this package. 

    - Linux GPU support and Apple Silicon support will be resolved by installing this library.

4. Install this library

    `pip install efficientv2-unet`
     
5. Verify the GPU-support:

    `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` >> lists your active GPU

    or

    `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())"` >> prints `true` if GPU enabled

    or 

    `ev2unet --version` >> prints the versions and whether GPU support is on.


# Data preparation
Mask should have background values of 0, and foreground values of 1.

At least 3 image/mask TIF pairs are required to train a model, and should be located in separate folders.

Folder Structure:
```
├── images
   ├── image1.tif
   ├── image2.tif
   ├── image3.tif
   └── ...
└── masks
   ├── image1.tif
   ├── image2.tif
   ├── image3.tif
   └── ...
```
Training a model will split the data into train, validation and test images (by default 70%, 15%, 15%, respectively).
And the images will be moved to corresponding sub-folders.

Training is performed not on the full images but on tiles (with no overlap), which will be saved into corresponding sub-folders.

# Usage
### Command-line:
```
ev2unet --help

# train example:
ev2unet --train --images path/to/images --masks path/to/masks --basedir . --name myUNetName --basemodel b2 --epochs 50 --train_batch_size 32

# predict example:
ev2unet --predict --dir path/to/images --model ./models/myUnetName/myUNetName.h5 --resolution 1 --threshold 0.5
```

### Jupyter notebooks 
Examples are also available from this [repository](notebooks/).
### QuPath extension
Get the [qupath-extension-efficientv2unet](https://github.com/DBM-MCF/qupath-extension-efficientv2unet)!

With this QuPath extension you can easily create training data and train a model via the QuPath GUI (or script). And you can also use the GUI or a script to predict.

<!--
## NOTES:
<span style="color:yellow">
- !!DONE: remove all "src" from import of this package... !! i.e. refactor the folder structure !!
- !!DONE: remove the temp deactivation in data_generation line 436
- !!DONE: remove temp return in efficientv2_unet line 613
- TODO: check that resolution for image scaling (e.g. in predict) is always an int and not a float
- TODO: make a notebook, where model is loaded and images are predicted one by one (so not all images need to be loaded into memory at once)
</span>

## Info to self:
On Windows, I have a working env 'test', and a new one to test the installation 'ev2unet'.

-->
