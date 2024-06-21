# Attention
This library is under development.

# EfficientV2-UNet
This package is a U-Net implementation of the [EfficientNetV2](https://arxiv.org/abs/2104.00298), using TensorFlow.

EfficientNetV2 improves speed and parameter efficiency. This implementation also uses the ImageNet weights for training new models.

It is intended for segmentation of histological images (RGB) that are **not** saved in pyramidal file format (WSI).

The output segmentation are foreground / background. Multi-class segmentation is not (yet) possible.

It works on TIF images (and probably also PNG).

# Installation

1. Create a python environment (e.g. with conda, requires python>=3.9), in a CLI:

    `conda create --name myenv python=3.9`

2. Activate environment:

    `conda activate myenv`

3. GPU support

    a. GPU support for **Windows** (example with conda):

    `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

    b. GPU support for **Linux** -- not tested:

    `python3 -m pip install tensorflow[and-cuda]`

    c. **Apple Silicon** support (requires Xcode command-line tools) -- Apple Intel not tested:

    ` xcode-select --install`

4. Install this library
    - clone/download the repository
    - open a CLI, activate your environment with tensorflow (see above)
    - (TensorFlow will be installed for Windows and MacOS platforms)
    ```
    cd path/to/repository
    pip install -e .
    ````
5. Verify the GPU-support:

    `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

    or

    `python -c "import tensorflow as tf; print(tf.test.is_gpu_available())`


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
ev2unet --train --images path/to/images --masks path/to/masks --basedir . --name myUNetName --basemodel b2 --epochs 50

# predict example:
ev2unet --predict --dir path/to/images --model ./models/myUnetName/myUNetName.h5 --resolution 1 --threshold 0.5
```

### Jupyter notebooks 
Examples are also available from this [repository](notebooks/).
### QuPath extension
A [QuPath](https://qupath.github.io/) implementation is on the way...

<!--
## NOTES:
<span style="color:yellow">
- !!DONE: remove all "src" from import of this package... !! i.e. refactor the folder structure !!
- !!DONE: remove the temp deactivation in data_generation line 436
- !!DONE: remove temp return in efficient_v2_unet line 613
- TODO: check that resolution for image scaling (e.g. in predict) is always an int and not a float
- TODO: make a notebook, where model is loaded and images are predicted one by one (so not all images need to be loaded into memory at once)
</span>

### Data preparation:
The raw images and corresponding masks, should be in separate folders,
and the file names must be the same.

Training will split the images into train, validation, 
and test sets (default is 70%, 15%, 15%, respectively). Eventually, 
the input images will be tiled (with no overlap) for training purposes (except 
the test images).

### DataGeneration for training:
There is a resolution parameter for the data generator,
usually at 1. But it will generate in addition crops for training
at resolutions +1 and +2. Hence, generally/at the moment, training
is done at multiple resolutions.

For training/validation data, crops of the images are generated. The crops
do not have any overlap, and the image is padded (reflecting at bottom and 
right boarders), to accommodate crops.

### Best model for Martin
currently it is the B3-best-checkpoint.


### Prediction
somehow prediction works better if the input image is downscaled.
-->
