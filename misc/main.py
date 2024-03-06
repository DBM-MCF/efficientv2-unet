import os

import cv2
import keras.models
import numpy as np
from skimage.util import montage
from skimage.io import imread

from efficient_v2_unet.model.efficient_v2_unet import create_and_train
from efficient_v2_unet.utils.data_generation import create_tiles
from efficient_v2_unet.utils.visualize import show_2images


IMG_SIZE = 256

# Train/Val data generation
img_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/train/images"
mask_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/train/masks"

val_im_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/test/images"
val_mask_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/test/masks"


all_img_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images"
all_mask_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks"
all_train_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/train"
all_train_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/train"
all_val_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/val"
all_val_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/val"
#DGenInMem
#DataSetGenerator

# FIXME
    # [] create prediction methods (that will tile image with given size, resize for DL, resize back to original size, ect.)
    # [/] i should create a new subfolder per model, and put the tensorboard logs in there

do_train = False
load_best = True
model_type = 'l'
# on all images, with 200 epochs
# not too bad models: b0-best, b1-best, (b1), ! b3-best !, (s), (s-best, bit less than s), m
# not good models: b0 (unusual), (b2, and b2-best->similar), m-best, l, l-best
# bad models: b3
final_name = 'my_efficientUNet-' + model_type.upper() + '_allIMGs'
# Model creation / loading
if do_train:

    model = create_and_train(
        name=final_name,
        basedir='.',
        train_img_dir=all_train_img_split,  #all_img_dir,
        train_mask_dir=all_train_mask_split,  #all_mask_dir,
        val_img_dir=all_val_img_split,  #None,
        val_mask_dir=all_val_mask_split,  #None,
        efficientnet=model_type,
        epochs=200
    )

    # FIXME: B0 model, does not seem to predict well when training with the additional images (no masks) i created
    # so i renamed the images in train/images to *.tiff


    """
    # previous version
    print('************  creating train and val data...')
    train_gen = DataSetGenerator(train_im_path=img_dir, train_mask_path=mask_dir, augmentations=True, resolution=1, batch_size=32)
    val_gen = DataSetGenerator(train_im_path=val_im_dir, train_mask_path=val_mask_dir, augmentations=False, resolution=1, batch_size=32)

    # build model
    model = build_efficient_unet(efficient_model=model_type, input_shape=(None, None, 3))
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(),
        metrics=[
            BinaryAccuracy(),
            BinaryIoU()
        ]
    )
    callbacks = get_callbacks(
        best_only=final_name + '_best-ckp.h5',
        early_stop=False,
        tensor_board=False
    )

    print('***********  Start fitting...')
    history = model.fit(
        x=train_gen,
        y=None,  # would be the target data, but must be none if X is a dataset/generator
        batch_size=None,  # (default = 32) should be none for generators (as they specify the batch size)
        epochs=300,
        verbose=2,  # 0=silent, 1=progressbar, 2=one-line per epoch
        callbacks=callbacks,
        validation_split=0.0,  # splits the train data, overwritten by next parameter
        validation_data=val_gen, # for datasets, validation_steps "could" be provided
        shuffle=True,  # shuffles train data before each epoch
        class_weight=None,  # dict with key=ints, val=floats to weight loss function on underrepresented labels
        sample_weight=None,  # not really relevant
        initial_epoch=0,  # used for resuming a training
        steps_per_epoch=None,  # when none, one epoch is over when all samples processed
        validation_steps=None,  # when none, all val data is used to evaluate epoch (only when validation_data is supplied)
        validation_freq=1,  # 1 = validation at each epoch  (only when validation_data is supplied)
        max_queue_size=10,  # only for generators, default=10
        workers=1,  # processed-based threading for generators, default=1
        use_multiprocessing=False  # similar to workers... but also not relevant here(?)
    )

    print('***********  End fitting...')
    model.save(final_name + '.h5')
    """
else:
    if load_best:
        model_path = os.path.join('../models', final_name, (final_name + '_best-ckp.h5'))
    else:
        model_path = os.path.join('../models', final_name, (final_name + '.h5'))
    print('****  loading model:', os.path.basename(model_path))
    model = keras.models.load_model(model_path)
    # FYI best-checkpoint seems bad... (old version)
    #model = keras.models.load_model(final_name + '.h5')
    #model = keras.models.load_model(final_name + '_best-ckp.h5')

print()
print('----predict unseen image')
path = 'G:/20231006_Martin/images/Slide2-26_ChannelBrightfield_Seq0007_XY1.ome.tif'
img = imread(path)
y = (img.shape[0] // 256) * 256
x = (img.shape[1] // 256) * 256
#img = cv2.resize(img, (y, x))
img = cv2.resize(img, (5120, 5120))
#img_crop = img[6000:10048, 5500:9548, :]
#img_crop = cv2.resize(img_crop, (4096, 4096))

print(img.shape)
#img = cv2.resize(img_crop, (2048, 2048))
#pred = model(np.asarray([img]))  # for me 5120x5120 seems to be the biggest size i can predict

#show_2images(img, pred)
#show_3images(img, pred, thresh=0.5)

tiles = create_tiles(img, 1024)
pred_tiles = model.predict(np.asarray(tiles))
pred_tiles = np.squeeze(pred_tiles)

row_col = int(len(tiles)**0.5)
print('img', np.asarray(tiles).shape)
print('mask', np.asarray(pred_tiles).shape)
img_montage = montage(tiles, grid_shape=(row_col, row_col), channel_axis=-1)
mask_montage = montage(pred_tiles, grid_shape=(row_col, row_col))

#show_3images(img_montage, mask_montage, thresh=0.5)
show_2images(img_montage, mask_montage, thresh=0.5)


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    print('name = main')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
