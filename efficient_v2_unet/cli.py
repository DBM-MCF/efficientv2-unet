import argparse


def get_arg_parser():
    """
    command line argument parser for the main function
    :return: ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="EfficientNetV2 UNet Command Line Parameters"
    )

    parser.add_argument("-v", "--version", action="store_true",
                        help="Show version info.")

    # train or predict arguments
    modality_args = parser.add_argument_group("Modality Arguments")
    modality_args.add_argument("--train", action="store_true",
                               help="Train model, using images in "
                                    "'images' and masks in 'masks'.")
    modality_args.add_argument("--predict", action="store_true",
                               help="Predict model, using images in 'dir'.")

    # model training arguments
    model_train_args = parser.add_argument_group("Model Training Arguments")
    model_train_args.add_argument("--images", type=str,
                                  help="Folder containing images.")
    model_train_args.add_argument("--masks",  type=str,
                                  help="Folder containing masks.")
    # TODO add additional path arguments
    model_train_args.add_argument("--basedir", default='.', type=str,
                                  help="Path to model saving location. "
                                       "Default: current working directory.")
    model_train_args.add_argument("--name", type=str,
                                  help="Name of your model (optional).")
    model_train_args.add_argument("--basemodel", default="b0", type=str,
                                  help="EfficientNetV2 base model type. "
                                       "Default is %(default)s.")
    model_train_args.add_argument("--epochs", default=10, type=int,
                                  help="Number of training epochs.")
    model_train_args.add_argument("--train_batch_size", default=64, type=int,
                                  help="Batch size (should be a power of 2). "
                                       "Default is %(default)s.")

    # predict arguments
    predict_args = parser.add_argument_group("Predict Arguments")
    predict_args.add_argument("--dir", type=str,
                              help="Folder containing images to predict.")
    predict_args.add_argument("--model", type=str,
                              help="Model file path (.h5) to use "
                                   "for prediction.")
    predict_args.add_argument("--resolution", default=1, type=int,
                              help="Resolution for prediction. Use e.g. 2 "
                                   "for downscaling input images to half the "
                                   "resolution.")
    predict_args.add_argument("--threshold", type=float,
                              help="Value for thresholding prediction maps. "
                                   "Default: None, which saves the "
                                   "the prediction maps.")
    predict_args.add_argument("--savedir", type=str,
                              help="Output directory. "
                                   "Default: 'dir/predicted'.")
    predict_args.add_argument("--use_less_memory", action="store_true",
                              help="Use less memory by processing one image "
                                   "by one.")

    return parser
