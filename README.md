# Installation

Create a python environment (e.g. with conda), in a CLI:

`conda create --name myenv python=3.7`

Activate environment:

`conda activate myenv`

Install the cudatoolkit, cudnn (for GPU support), and tensorflow:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow<2.11"
```
Verify the GPU-support:

`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

Install this library
- download the repository
- open a CLI, activate your environment with tensorflow (see above)

```
cd path/to/repository
pip install -e .
````

## NOTES:
<span style="color:yellow">
- TODO: remove all "src" from import of this package... !! i.e. refactor the folder structure !!
- TODO: remove the temp deactivation in data_generation line 436
- TODO: remove temp return in efficient_v2_unet line 613
- TODO: check that resolution for image scaling (e.g. in predict) is always an int and not a float
- TODO: make a notebook, where model is loaded and images are predicted one by one (so not all images need to be loaded into memory at once)

</span>

### Data preparation:
The raw images and corresponding masks, should be in separate folderes,
and the file names must be the same.

Running the predict function, will split the images into train, validation, 
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

Training currently takes an input folder and splits it in **train and validation datasets**

### Prediction
somehow prediction works better if the input image is downscaled.

# TODOs
- create also test dataset
- find a way to **evaluate** on test data (and write results into the metadata), directly after training
- create function to predict on multiple images (or adjust current predict function)
- to predict function:
  - add saving
  - add threshold?
  - or/and optional probability mask saving
- once evaluation is there, test for
  - minimal epochs
  - minimal input images