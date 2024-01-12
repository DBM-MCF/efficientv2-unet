import json
import os

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

from src.efficientUNet.utils.data_generation import DataSetGenerator, \
    split_folder_files_to_train_val

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


def build_efficient_unet(efficient_model, input_shape) -> Model:
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
                  early_stop=False,
                  tensor_board_logdir=None):
    """
    Creates a list of callbacks for model training.
    :param best_only: (str) save model checkpoints, only best
    :param early_stop: (bool) early stopping, based on val_loss
    :param tensor_board_logdir: (str) tensorboard logging path (or None
                                      for no logging)
    :return: list of callback objects
    """
    callbks = [
        callbacks.ModelCheckpoint(
            best_only,
            save_best_only=True,
            monitor='val_binary_accuracy',
            verbose=1
        )
    ]
    if early_stop:
        callbks.append(callbacks.EarlyStopping(
            patience=7, monitor='binary_accuracy', verbose=1  # FIXME maybe rather use binary_io_u??
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
    name=None,
    basedir='.',
    train_img_dir=None,
    train_mask_dir=None,
    val_img_dir=None,
    val_mask_dir=None,
    efficientnet='b0',
    epochs=300,
    early_stopping=False,
    batch_size=64,
    img_size=256
) -> Model:
    """
    Creates a model and trains it.
    Validation data are optional parameters: if None it will split the
    train data to 80/20%, training/validation data respectively.
    (does not check if a model with the same name and path already exists,
    would just overwrite it).
    Saves the model to:
        basedir / models / name / name.h5
    :param name: (str) name for the model and folder the model will be
                placed in. If None, default name will be given.
    :param basedir: (str) path to where the 'models' folder is.
                    Default = '.'
    :param train_img_dir: (str) path to folder with the training images (tif).
    :param train_mask_dir: (str) path to the folder with the training masks.
    :param val_img_dir: (str) optional: path to folder with validation images.
    :param val_mask_dir: (str) optional: path to folder with validation masks.
    :param efficientnet: (str) base EfficientNet backbone
                         (see MODELS dict for options)
    :param epochs: (int) number of epochs to train the model for
    :param early_stopping: (Bool) for early stopping callback during training.
    :param batch_size: (int) default is 64, should be 2**x
    :param img_size: (int) crop image size. Default is 256.
                     I don't suggest changing that.
    :return: keras.model
    """

    # sanity checks
    if name is None:
        name = 'myEfficientUNnet_' + efficientnet
    elif name.endswith('.h5'):
        name = name.replace('.h5', '')
    if train_img_dir is None or train_mask_dir is None:
        raise RuntimeError("No training and/or mask paths provided.")

    # TODO check if folder for folder already exists and print warning that it will overwrite it...

    # If no validation dataset provided, split them manually
    if val_img_dir is None:
        if val_mask_dir is None:
            # shuffle and split into train and validation folders
            (train_img_dir, train_mask_dir,
             val_img_dir, val_mask_dir) = split_folder_files_to_train_val(
                image_dir=train_img_dir,
                mask_dir=train_mask_dir
            )
        else:
            print('Ignoring the path to the validation masks directory, '
                  'since no validation image directory path was provided.')
    # sanity check
    if val_mask_dir is None:
        raise RuntimeError("No path to the validation mask data provided.")

    # create model directory
    path = os.path.join(basedir, 'models', name)
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
    model = build_efficient_unet(
        efficient_model=efficientnet,
        input_shape=(None, None, 3)
    )

    # create callbacks
    call_backs = get_callbacks(
        best_only=os.path.join(path, (name + '_best-ckp.h5')),
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
            }
        },
        'train_image_data': train_gen.metadata,
        'validation_image_data': val_gen.metadata,
        'test_image_data': None,  # FIXME needs implementation
        'test_metrics': None,  # FIXME needs implementation
    }

    json_path = os.path.join(path, (name + '.json'))
    with open(json_path, "w") as json_file:
        json_file.write(json.dumps(metadata, indent=4))
    print()
    print('Saved model training metadata to:', json_path)

    # save the model
    model_path = os.path.join(path, (name + '.h5'))
    model.save(model_path)
    print('Saved final model to:', model_path)
    return model


# For testing
if __name__ == "__main__":
    all_train_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/train"
    all_train_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/train"
    all_val_img_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_images/val"
    all_val_mask_split = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/all_masks/val"
    model_type = 'b0'
    final_name = 'test_history_model.h5'
    base_dir = '../../../models_test/'

    model_test = create_and_train(
        name=final_name,
        basedir=base_dir,
        train_img_dir=all_train_img_split,  # all_img_dir,
        train_mask_dir=all_train_mask_split,  # all_mask_dir,
        val_img_dir=all_val_img_split,  # None,
        val_mask_dir=all_val_mask_split,  # None,
        efficientnet=model_type,
        epochs=2
    )

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
