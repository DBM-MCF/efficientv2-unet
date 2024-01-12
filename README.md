
## NOTES:

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