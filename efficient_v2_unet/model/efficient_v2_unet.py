import glob
import json
import os
from time import sleep
from typing import Union

from skimage.io import imread, imsave

import keras.models
from keras import Model
from keras import callbacks
from keras.applications.efficientnet_v2 import (
    EfficientNetV2B0, EfficientNetV2B3, EfficientNetV2B1, EfficientNetV2B2,
    EfficientNetV2L, EfficientNetV2M, EfficientNetV2S
)
from keras.layers import (
    Conv2D, BatchNormalization, Conv2DTranspose,
    Activation, Input, Concatenate,
    # LeakyReLU, Add, MaxPooling2D, Dropout, concatenate
)
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, BinaryIoU
from keras.optimizers import Adam

from efficient_v2_unet.model.metrics import (
    calc_metrics, calc_metrics_average, create_metrics_graph)
from efficient_v2_unet.model.predict import predict
from efficient_v2_unet.utils.data_generation import (
    DataSetGenerator, split_folder_files_to_train_val_test
)

# Based on EfficientNetV2: https://arxiv.org/abs/2104.00298
# EfficientNetV2 improves speed and parameter efficiency

MODELS = {
    'efficientnetv2-b0': 'b0',
    'efficientnetv2-b1': "b1",
    'efficientnetv2-b2': "b2",
    'efficientnetv2-b3': "b3",
    'efficientnetv2-l': "l",
    'efficientnetv2-m': "m",
    'efficientnetv2-s': "s"
}


def _get_base_encoder(short_model_name, inputs) -> Model:
    """
    Creates and returns an EfficientNet model
    :param short_model_name: short
    :param inputs: Input object of the shape tuple?
    :return:
    """
    # create model and return
    if short_model_name == "b0":
        return EfficientNetV2B0(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    if short_model_name == "b1":
        return EfficientNetV2B1(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    if short_model_name == "b2":
        return EfficientNetV2B2(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    if short_model_name == "b3":
        return EfficientNetV2B3(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    if short_model_name == "l":
        return EfficientNetV2L(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    if short_model_name == "m":
        return EfficientNetV2M(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
    if short_model_name == "s":
        return EfficientNetV2S(
            include_top=False, weights="imagenet", input_tensor=inputs
        )


ENCODER_LAYER_NAMES = {
    'b0_s1': 'input_1',                     # Activation layer with size 256
    'b0_s2': 'block1a_project_activation',  # Activation layer with size 128
    'b0_s3': 'block2b_expand_activation',   # Activation layer with size 64
    'b0_s4': 'block4a_expand_activation',   # Activation layer with size 32
    # Activation layer with size 16 ('Bottleneck')
    'b0_b1': 'block6a_expand_activation',
    'b1_s1': 'input_1',
    'b1_s2': 'block1b_project_activation',
    'b1_s3': 'block2c_expand_activation',
    'b1_s4': 'block4a_expand_activation',
    'b1_b1': 'block6a_expand_activation',
    'b2_s1': 'input_1',
    'b2_s2': 'block1b_project_activation',
    'b2_s3': 'block2c_expand_activation',
    'b2_s4': 'block4a_expand_activation',
    'b2_b1': 'block6a_expand_activation',
    'b3_s1': 'input_1',                     # 256
    'b3_s2': 'block1b_project_activation',  # 128, ?block1a_project_activation?
    'b3_s3': 'block2c_expand_activation',   # 64, or block2a* or block2b??
    'b3_s4': 'block4a_expand_activation',   # 32
    'b3_b1': 'block6a_expand_activation',   # 16
    'l_s1': 'input_1',
    'l_s2': 'block1d_project_activation',
    'l_s3': 'block2g_expand_activation',
    'l_s4': 'block4a_expand_activation',
    'l_b1': 'block6a_expand_activation',
    'm_s1': 'input_1',
    'm_s2': 'block1c_project_activation',
    'm_s3': 'block2e_expand_activation',
    'm_s4': 'block4a_expand_activation',
    'm_b1': 'block6a_expand_activation',
    's_s1': 'input_1',
    's_s2': 'block1b_project_activation',
    's_s3': 'block2d_expand_activation',
    's_s4': 'block4a_expand_activation',
    's_b1': 'block6a_expand_activation',
}


def conv_block(inputs, num_filters):
    """
    Convolutional block
    :param inputs: Tensor?
    :param num_filters: number of filters
    :return:
    """
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def decoder_block(inputs, skip, num_filters):
    """
    Decoder block
    :param inputs: Tensor
    :param skip: bool
    :param num_filters: number of filters
    :return:
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x


def build_efficient_v2_unet(efficient_model, input_shape) -> Model:
    """
    Build an efficient UNet model, takes imagenet weights.

    to convert the efficientNet ot a UNET this video was helpful:
    https://www.youtube.com/watch?v=2uIq1FvVfp8

    :param efficient_model: short 'version' of desired efficientNet backbone
    :param input_shape: tuple (usually (IMG_SIZE, IMG_SIZE, CHANNELS))
    :return: Model
    """
    inputs = Input(input_shape)

    # check if model type is supported
    efficient_model = efficient_model.lower()
    if efficient_model not in MODELS.values():
        if efficient_model in MODELS.keys():
            short_model_name = MODELS.get(efficient_model)
            print(short_model_name)
        else:
            raise NotImplementedError(
                f'{efficient_model} not recognised EfficientNetV2 model type.')

    # print('-*-*-*-*-*-    Encoder for:', efficient_model)
    # get the efficientNet encoder
    encoder = _get_base_encoder(efficient_model, inputs)

    # encoder summary will give info about the different layers,
    # for the decoder I have to find the last activation layer type
    # that matches the IMG_SIZE
    # encoder.summary()

    # get the convolution output layers of the encoder
    s1 = encoder.get_layer(
        ENCODER_LAYER_NAMES.get(efficient_model + '_s1')
    ).output  # 256
    s2 = encoder.get_layer(
        ENCODER_LAYER_NAMES.get(efficient_model + '_s2')
    ).output  # 128
    s3 = encoder.get_layer(
        ENCODER_LAYER_NAMES.get(efficient_model + '_s3')
    ).output  # 64
    s4 = encoder.get_layer(
        ENCODER_LAYER_NAMES.get(efficient_model + '_s4')
    ).output  # 32

    # Bottleneck
    b1 = encoder.get_layer(
        ENCODER_LAYER_NAMES.get(efficient_model + '_b1')
    ).output  # 16

    # Decoder  # FIXME ?? I actually think these are skip-connections ??
    d1 = decoder_block(b1, s4, 512)  # 32
    d2 = decoder_block(d1, s3, 256)  # 64
    d3 = decoder_block(d2, s2, 128)  # 128
    d4 = decoder_block(d3, s1, 64)  # 256

    # output layer
    # for binary output
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    # for categorical output
    #outputs = Conv2D('n_classes', 1, padding='same', activation='softmax')(d4) # not sure if this would work
    model = Model(inputs, outputs, name="efficient_unet_v2-" + efficient_model)
    return model


def get_callbacks(best_only='my_efficientUNet_model_best_ckp.h5',
                  monitor: str = 'val_binary_accuracy',
                  early_stop=False,
                  tensor_board_logdir=None):
    """
    Creates a list of callbacks for model training.
    :param best_only: (str) save model checkpoints, only best
    :param monitor: (str) metric to monitor, e.g.:
                            - "val_binary_accuracy", or
                            - "val_binary_io_u"
    :param early_stop: (bool) early stopping, based on val_loss
    :param tensor_board_logdir: (str) tensorboard logging path (or None
                                      for no logging)
    :return: list of callback objects
    """
    callbks = [
        callbacks.ModelCheckpoint(
            best_only,
            save_best_only=True,
            monitor=monitor,
            mode='max',
            verbose=1
        )
    ]
    if early_stop:
        callbks.append(callbacks.EarlyStopping(
            patience=7, monitor=monitor, verbose=1
        ))
    if tensor_board_logdir is not None:
        callbks.append(
            (callbacks.TensorBoard(log_dir=tensor_board_logdir,
                                   write_images=True))
        )
        print('Start tensorboard with: "tensorboard --logdir=."')
        print('Access tensorboard in browser: http://localhost:6006/')
    return callbks


def create_and_train(
    name: str = None,
    basedir: str = '.',
    train_img_dir: str = None,
    train_mask_dir: str = None,
    val_img_dir: str = None,
    val_mask_dir: str = None,
    test_img_dir: str = None,
    test_mask_dir: str = None,
    efficientnet: str = 'b0',
    epochs: int = 100,
    early_stopping: bool = False,
    batch_size: int = 64,
    img_size: int = 256,
    file_ext: str = '.tif'
) -> Model:
    # FIXME adjust description (i.e. train/val/test split and metadata saving)
    """
    Creates a model and trains it.
    Validation paths and test paths are optional.
    By default the data will be split into 70/15/15% train, validation and
    test data respectively.
    Provide paths to validation and test data if other splitting is desired.
    Splitting images with custom percentages can be done, using the function
    utils.data_generation.split_folder_files_to_train_val_test()

    Saves the model to:
        basedir / models / name / name.h5
    It will overwrite existing models.
    :param name: (str) name for the model and folder the model will be
                placed in. If None, default name will be given.
    :param basedir: (str) path to where the 'models' folder is.
                    Default = '.'
    :param train_img_dir: (str) path to folder with the training images (tif).
                          Required.
    :param train_mask_dir: (str) path to the folder with the training masks.
                           Required.
    :param val_img_dir: (str) optional: path to folder with validation images.
    :param val_mask_dir: (str) optional: path to folder with validation masks.
    :param test_mask_dir: (str) optional: path to folder with test images.
    :param test_img_dir: (str) optional: path to folder with test masks.
    :param efficientnet: (str) base EfficientNet backbone
                         (see MODELS dict for options)
    :param epochs: (int) number of epochs to train the model for (default = 100)
    :param early_stopping: (Bool) for early stopping callback during training.
    :param batch_size: (int) default is 64, should be 2**x
    :param img_size: (int) crop image size. Default is 256.
                     I don't suggest changing that.
    :param file_ext: (str) image file extension TODO !only '.tif' tested!
    :return: keras.model
    """
    # sanity checks         --------------------------------------------------
    if name is None:
        name = 'myEfficientUNet_' + efficientnet
    elif name.endswith('.h5'):
        name = name.replace('.h5', '')
    if train_img_dir is None or train_mask_dir is None:
        raise RuntimeError("No training and/or mask paths provided.")
    if not os.path.exists(train_img_dir):
        raise IOError(f'The training image path does not exist: '
                      f'<{train_img_dir}>')
    if not os.path.exists(train_mask_dir):
        raise IOError(f'The training mask path does not exist: '
                      f'<{train_mask_dir}>')

    # Split input images into train, validation and test images     ----------
    if val_img_dir is not None:
        if not os.path.exists(val_img_dir):
            raise IOError(f'The validation image path does not exist: '
                          f'<{val_img_dir}>')
        if val_mask_dir is None or not os.path.exists(val_mask_dir):
            raise IOError(f'The validation mask is missing or wrong: '
                          f'<{val_mask_dir}>')

        if test_img_dir is None:
            # split into 85% train and 15% test images
            if isinstance(test_mask_dir, str):
                print(f'Warning: No test image path provided, but a test mask '
                      f'path <{test_mask_dir}>, which will be ignored.')
            (
                train_img_dir, train_mask_dir, _, _,
                test_img_dir, test_mask_dir
            ) = split_folder_files_to_train_val_test(
                train_img_dir, train_mask_dir, do_val=False, do_test=True)
        else:
            # all paths were given
            if not os.path.exists(test_img_dir):
                raise IOError(f'The test image path does not exist: '
                              f'<{test_img_dir}>')
            if not os.path.exists(test_mask_dir):
                raise IOError(f'The test mask path does not exist: '
                              f'<{test_mask_dir}>')

    else:  # validation is None
        if test_img_dir is None:
            # split into 85% train, 15% val and 15% test images
            if isinstance(test_mask_dir, str):
                print(f'Warning: No test image path provided, but a test mask '
                      f'path <{test_mask_dir}>, which will be ignored.')
            (
                train_img_dir, train_mask_dir,
                val_img_dir, val_mask_dir,
                test_img_dir, test_mask_dir
            ) = split_folder_files_to_train_val_test(
                train_img_dir, train_mask_dir, do_val=True, do_test=True)
        else:
            # split into 85% train and 15% val images
            if not os.path.exists(test_img_dir):
                raise IOError(f'The test image path does not exist: '
                              f'<{test_img_dir}>')
            if not os.path.exists(test_mask_dir):
                raise IOError(f'The test mask path does not exist: '
                              f'<{test_mask_dir}>')
            (
                train_img_dir, train_mask_dir,
                val_img_dir, val_mask_dir, _, _
            ) = split_folder_files_to_train_val_test(
                train_img_dir, train_mask_dir, do_val=True, do_test=False)

    # sanity checks (probably not necessary here...)    ----------------------
    n_train_img = len(glob.glob(train_img_dir + '/*' + file_ext))
    n_train_mask = len(glob.glob(train_mask_dir + '/*' + file_ext))
    n_val_img = len(glob.glob(val_img_dir + '/*' + file_ext))
    n_val_mask = len(glob.glob(val_mask_dir + '/*' + file_ext))
    n_test_img = len(glob.glob(test_img_dir + '/*' + file_ext))
    n_test_mask = len(glob.glob(test_mask_dir + '/*' + file_ext))
    assert n_train_img == n_train_mask, f'Number of training images ' \
                                        f'({n_train_img}) is not the same as '\
                                        f'the number of masks ({n_train_mask})'
    assert n_val_img == n_val_mask, f'Number of validation images ' \
                                    f'({n_val_img}) is not the same as the ' \
                                    f'number of masks ({n_val_mask})'
    assert n_test_img == n_test_mask, f'Number of test images ({n_test_img}) '\
                                      f'is not the same as the ' \
                                      f'number of masks ({n_test_mask})'
    n_total_img = n_train_img + n_val_img + n_test_img
    print(
        f'The data ({n_total_img} images) was split into:\n'
        f'- {round(100 * n_train_img / n_total_img)}% = {n_train_img} '
        f'training images\n'
        f'- {round(100 * n_val_img / n_total_img)}% = {n_val_img} '
        f'validation images\n'
        f'- {round(100 * n_test_img / n_total_img)}% = {n_test_img} '
        f'test images.'
    )
    # Warn if there is too little images for a certain category     ----------
    # i.e. when images were split manually...
    if n_train_img / n_total_img < 0.6:
        print('!!! WARNING: you have less than 60% of training images. '
              'Consider aborting...')
    if n_val_img / n_total_img < 0.1:
        if n_val_img == 0:
            raise RuntimeError('You need validation images but you have none.')
        else:
            print('!!! WARNING: you have less than 10% of validation images. '
                  'Consider aborting...')
    if n_test_img / n_total_img < 0.1:
        if n_test_img == 0:
            print("!!! WARNING: you have 0 test images. You won't be able to "
                  "evaluate the trained model...")
        else:
            print("!!! WARNING: you have less than 10% of test images...")

    # train the model       --------------------------------------------------
    # create model directory
    path = os.path.join(basedir, 'models', name)
    if os.path.exists(path):
        print(f'Warning: the model already exists and will be overwritten '
              f'(in: {path})')
        print('Continue in ', end='')
        for i in range(3, 0, -1):
            print(i, end='...')
            sleep(1)
        print('0')
    os.makedirs(path, exist_ok=True)

    # create data generators
    print('Creating training data')
    train_gen = DataSetGenerator(
        train_im_path=train_img_dir,
        train_mask_path=train_mask_dir,
        augmentations=True,
        batch_size=batch_size,
        img_size=img_size
    )
    val_gen = DataSetGenerator(
        train_im_path=val_img_dir,
        train_mask_path=val_mask_dir,
        augmentations=False,
        batch_size=batch_size,
        img_size=img_size
    )
    print("Number of training images (crops) =", len(train_gen.crop_paths))
    print("Number of validation images (crops) =", len(val_gen.crop_paths))

    # check that the masks do not contain more than one label
    if train_gen.get_classes() > 1:
        raise NotImplementedError("Training models for more than one class "
                                  "is not yet implemented.")
    if val_gen is not None and val_gen.get_classes() > 1:
        raise NotImplementedError("Training models for more than one class "
                                  "is not yet implemented.")

    # create model
    print('Creating the model')
    model = build_efficient_v2_unet(
        efficient_model=efficientnet,
        input_shape=(None, None, 3)
    )

    # create callbacks
    best_model_path = os.path.join(path, (name + '_best-ckp.h5'))
    call_backs = get_callbacks(
        best_only=best_model_path,
        monitor='val_binary_io_u',  # or e.g. default = 'val_binary_accuracy'
        early_stop=early_stopping,
        tensor_board_logdir=os.path.join(path, 'logs')
    )

    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(),
        metrics=[
            BinaryAccuracy(threshold=0.5),  # uses default threshold = 0.5
            BinaryIoU(
                target_class_ids=[1],  # just the IoU for class 1
                # (if None it calculates for (0, 1))
                threshold=0.5  # default threshold = 0.5
            )
        ]
    )

    # train model
    print('Starting training...')
    history = model.fit(
        x=train_gen,
        y=None,  # target data, but must be none if X is a dataset/generator
        batch_size=None,  # (default = 32) should be none for generators
        epochs=epochs,
        verbose=2,  # 0=silent, 1=progressbar, 2=one-line per epoch
        callbacks=call_backs,
        validation_split=None,  # splits the train data (but only if
        # X and Y are given, i.e. no datasets)
        validation_data=val_gen,  # for datasets, validation_steps "could" be provided
        shuffle=True,  # shuffles train data before each epoch
        class_weight=None,  # dict with key=ints, val=floats for weight loss
        # function on underrepresented labels
        sample_weight=None,  # not really relevant
        initial_epoch=0,  # used for resuming a training
        steps_per_epoch=None,  # when none, one epoch = process all samples
        validation_steps=None,  # none = all val data is used to evaluate epoch
        validation_freq=1,  # 1 = validation at each epoch
        max_queue_size=10,  # only for generators, default=10
        workers=1,  # processed-based threading for generators, default=1
        use_multiprocessing=False  # not relevant here(?)
    )

    # save the model
    model_path = os.path.join(path, (name + '.h5'))
    model.save(model_path)
    print('Saved final model to:', model_path)

    # Evaluate models and calculate metrics      -----------------------------
    print('Evaluate model:', name + '.h5', 'on test data set')
    metrics = evaluate_model(
        model=model,
        img_dir_path=test_img_dir,
        mask_dir_path=test_mask_dir,
        file_extension=file_ext
    )
    # evaluate also best model
    metrics_best = evaluate_model(
        model=best_model_path,
        img_dir_path=test_img_dir,
        mask_dir_path=test_mask_dir,
        file_extension=file_ext
    )
    test_metrics = {
            name + '.h5': metrics,
            os.path.basename(best_model_path): metrics_best
        }
    # calculate the metric averages
    test_metrics = calc_metrics_average(test_metrics)
    # create metric graphs and calculate the best model parameters
    best_bin_ious = create_metrics_graph(
        test_metrics=test_metrics,
        save_dir_path=path  # is the path to the model folder
    )
    # add the best_bin_ious to the test_metrics dict
    for _model, _metrics in best_bin_ious.items():
        for _best_iou, _values in _metrics.items():
            test_metrics[_model][_best_iou] = _values

    # create metadata       --------------------------------------------------
    # create test image metadata
    if n_test_img == 0:
        test_image_metadata = None
    else:
        test_image_metadata = {}
        for i, (img_path, mask_path) in enumerate(
            zip(
                glob.glob(test_img_dir + '/*' + file_ext),
                glob.glob(test_mask_dir + '/*' + file_ext)
            )
        ):
            test_image_metadata['test_image_' + str(i)] = {
                'image_path': img_path,
                'mask_path': mask_path
            }

    # create model metadata
    metadata = {
        'callbacks': {
            # unfortunately I couldn't access the best_model.history
            'best_model_checkpoint': {
                'best_model_name': os.path.basename(call_backs[0].filepath),
                'monitor_metric': call_backs[0].monitor
            },
            'early_stopping': early_stopping,
        },
        'model': {
            'model_name': name + '.h5',
            'model_parameters': history.params,
            # e.g. = {'verbose': 2, 'epochs': 2, 'steps': 135}
            'model_history': history.history,
            # i.e. all monitored metrics for each epoch
            'other_parameters': {
                'base_model': list(MODELS.keys())[
                    list(MODELS.values()).index(efficientnet)
                ],
                'train_image_size': (img_size, img_size),
                'batch_size': batch_size,
                'number_of_train_images': n_train_img,
                'number_of_train_tiles': len(train_gen.crop_paths),
                'number_of_validation_images': n_val_img,
                'number_of_validation_tiles': len(val_gen.crop_paths),
                'number_of_test_images': n_test_img,
            }
        },
        'train_image_data': train_gen.metadata,
        'validation_image_data': val_gen.metadata,
        'test_image_data': test_image_metadata,
        'test_metrics': test_metrics,
    }

    # save metadata
    json_path = os.path.join(path, (name + '.json'))
    with open(json_path, "w") as json_file:
        json_file.write(json.dumps(metadata, indent=4))
    print()
    print('Saved model training metadata to:', json_path)

    # FIXME, temporarily return also test_metrics
    return model, test_metrics


def evaluate_model(
    model: Union[Model, str],
    img_dir_path: str,
    mask_dir_path: str,
    file_extension: str = '.tif'
):
    # TODO / FIXME -> Describe!
    """
    Evaluate model on test data.
    :param model:
    :param img_dir_path:
    :param mask_dir_path:
    :param file_extension:
    :return:
    """

    # sanity checks
    if not os.path.exists(img_dir_path):
        raise IOError(f'The image folder does not exist: <{img_dir_path}>')
    if not os.path.exists(mask_dir_path):
        raise IOError(f'The mask folder does not exist: <{mask_dir_path}>')
    img_paths = glob.glob(img_dir_path + '/*' + file_extension)
    if len(img_paths) == 0:
        raise RuntimeError(f'No image files ({file_extension}) found in: '
                           f'{img_dir_path}')
    mask_paths = []
    for path in img_paths:
        mask_path = path.replace(img_dir_path, mask_dir_path)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f'The file <{os.path.basename(str(mask_paths))}> was not '
                f'found in the mask folder: <{mask_dir_path}>.'
            )
        mask_paths.append(mask_path)

    # remember if it is the 'best-ckp-model', for output file renaming
    best = False
    # check if model path exist and load model
    if isinstance(model, str):
        if not os.path.exists(model):
            raise FileNotFoundError(f'The model does not exist: {model}')
        if '_best-ckp' in model:
            best = True
        model = keras.models.load_model(model)

    # Load test images and the ground truth             ----------------------
    x = [imread(path) for path in img_paths]
    gt = [imread(path) for path in mask_paths]

    # Predict test images at different resolutions          ------------------
    metrics = {}
    for i in range(1, 4):  # Resolutions: 1/1, 1/2, 1/3
        print(f'Predicting test dataset at resolution = 1/{i}')
        predictions = predict(
            images=x, model=model,
            factor=i,  # Resolution1 = full (2 = half resolution)
            tile_size=512,  # default for this function
            overlap=0,  # default for this function
            batch_size=16  # a quarter of the training batch size (hard-coded)
        )
        # prediction function scales the images back to full resolution

        # save test predictions to file             --------------------------
        test_path = os.path.dirname(mask_dir_path)
        test_path = os.path.join(test_path, 'test_predictions')
        if os.path.exists(test_path):
            print(f'!!!Warning the output directory for the test predictions '
                  f'already exists: <{test_path}>\n'
                  f'Images in this folder will be overwritten.')
        os.makedirs(test_path, exist_ok=True)

        if not isinstance(predictions, list):
            predictions = [predictions]
        for img_path, pred_img in zip(img_paths, predictions):
            path = os.path.join(test_path, os.path.basename(img_path))
            # add at which resolution the image was predicted
            res_info = f'_resolution1-{i}' + file_extension
            path = path.replace(file_extension, res_info)
            if best:
                # rename the output file, if it is the best checkpoint model
                path = path.replace(file_extension,
                                    '_best-ckp' + file_extension)
            imsave(path, pred_img)
        print(f'The {len(predictions)} image(s) were saved to: <{test_path}>')

        # calculate metrics         ------------------------------------------
        # init list of metrics per image (row = images, cols = thresholds)
        image_acc = [[] for x in range(len(mask_paths))]
        image_iou = [[] for x in range(len(mask_paths))]
        thresholds = [x / 10 for x in
                      range(0, 10)]  # threshold of 1.0 is useless
        for t in thresholds:
            (acc, iou) = calc_metrics(y_true=gt, y_pred=predictions,
                                      threshold=t)
            for j in range(len(acc)):
                image_acc[j].append(acc[j])
                image_iou[j].append(iou[j])

        # create metrics dictionary
        accuracy = {'thresholds': thresholds}
        bin_iou = {'thresholds': thresholds}
        # add the metrics per image_name to the corresponding dicts
        for j in range(len(mask_paths)):
            image_name = os.path.basename(mask_paths[j])
            accuracy[image_name] = image_acc[j]
            bin_iou[image_name] = image_iou[j]

        metrics[f'@resolution=1/{i}'] = {
            'binary_accuracy': accuracy,
            'binary_iou': bin_iou
        }
    return metrics


# For testing
if __name__ == "__main__":
    dir_img_unsplit = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images"
    dir_mask_unsplit = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks"
    all_train_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/train"
    all_train_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/train"
    all_val_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/val"
    all_val_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/val"
    all_test_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/test"
    all_test_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/test"
    model_type = 'b0'
    final_name = 'test_history_model.h5'
    base_dir = '../../models_test/'

    do_train = True

    if do_train:
        model_test = create_and_train(
            name=final_name,
            basedir=base_dir,
            train_img_dir=all_train_img_split,  # all_img_dir,
            train_mask_dir=all_train_mask_split,  # all_mask_dir,
            val_img_dir=all_val_img_split,  # all_val_img_split, None,
            val_mask_dir=all_val_mask_split,  # all_val_mask_split, None,
            test_img_dir=all_test_img_split,
            test_mask_dir=all_test_mask_split,
            efficientnet=model_type,
            epochs=2
        )
    # evaluate
    '''model_path = '../../../models_test/models/test_history_model/test_history_model.h5'
    model_path ='../../../models/my_efficientUNet-B3_allIMGs/my_efficientUNet-B3_allIMGs_best-ckp.h5'
    evaluate_model(model_path, all_test_img_split, all_test_mask_split)'''

    '''
    # load already saved model to check the history
    #  does not exist after loading model from file
    best_model = 'G:/20231006_Martin/EfficientUNet/models/my_efficientUNet-B3_allIMGs/my_efficientUNet-B3_allIMGs.h5'
    best_model = keras.models.load_model(best_model)
    #print(best_model.history.params)
    for k, v in best_model.history.history.items():
        print(k,v)
    '''
    '''
    cal = keras.callbacks.ModelCheckpoint(base_dir)
    best = cal.best
    print(type(best))
    '''
