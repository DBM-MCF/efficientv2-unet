import glob
import os.path

from efficient_v2_unet.cli import get_arg_parser

from skimage.io import imread
from tifffile import imwrite
from zipfile import ZIP_DEFLATED


def main():
    """
    Run package from commandline
    """
    args = get_arg_parser().parse_args(
    )

    # If version was requested, print it and end
    if args.version:
        from efficient_v2_unet.version import version_summary
        print(version_summary)
        return

    if args.train and args.predict:
        print('Error: --train and --predict cannot be used together')
        return

    #   Train        ---------------------------------------------------------
    if args.train:
        # Make sure that image path and mask path are not None and exist
        if args.images is None and args.masks is None:
            print('Error: --train requires --images and --masks')
            return
        if args.images is None:
            print('Error: --train requires --images')
            return
        if args.masks is None:
            print('Error: --train requires --masks')
            return
        # TODO implement val and test paths (and batch size)

        from efficient_v2_unet.model.efficient_v2_unet import create_and_train
        # Start training
        try:
            create_and_train(
                name=args.name,
                basedir=args.basedir,
                train_img_dir=args.images,
                train_mask_dir=args.masks,
                val_img_dir=None,  # TODO implement
                val_mask_dir=None,  # TODO implement
                test_img_dir=None,  # TODO implement
                test_mask_dir=None,  # TODO implement
                efficientnet=args.basemodel,
                epochs=args.epochs,
                batch_size=64,  # TODO implement
                img_size=256,
                file_ext='.tif'
            )
        except FileExistsError:
            print(f'Error: it seems like the input images have already, '
                  f'been split into train, validation and test data. '
                  f'Please move the images back into <{args.images}>, the '
                  f'masks back into <{args.masks}>, and the sub-folders.')
            return

    #   Predict        -------------------------------------------------------
    elif args.predict:
        if args.dir is None:
            print('Error: --predict requires --dir')
            return
        # check that dir exists
        if not os.path.exists(args.dir):
            print(f'Error: the directory does not exist: <{args.dir}>')
            return
        # check the output path
        if args.savedir is None:
            args.savedir = os.path.join(args.dir, 'prediction')
            os.makedirs(args.savedir, exist_ok=True)
            print('Created savedir:', args.savedir)
        else:
            if not os.path.exists(args.savedir):
                print(f'Error: The save dir <{args.savedir}> does not exist.')
                return

        # Start prediction      ----------------------------------------------
        # Get image paths
        img_paths = glob.glob(os.path.join(args.dir, '*.tif'))
        if len(img_paths) == 0:
            print(f'Error: No *.tif file found in <{args.dir}>')
            return

        # Load model
        model = None
        if args.model is None:
            print('Error: --predict requires --model')
            return
        else:
            from efficient_v2_unet.model.predict import predict
            import keras
            model = keras.models.load_model(args.model)

        # process image by image
        if args.use_less_memory:
            for path in img_paths:
                img = imread(path)
                prediction = predict(
                    images=[img],
                    model=model,
                    threshold=args.threshold,
                    factor=args.resolution,
                    # TODO implement batch_size, but keep overlap and tile_size at defaults
                )
                # Save predictions
                image_name = os.path.basename(path)
                if args.threshold is None:
                    # save img without compression
                    imwrite(os.path.join(args.savedir, image_name), prediction)
                else:
                    # save image with compression
                    imwrite(os.path.join(args.savedir, image_name), prediction,
                            compression=ZIP_DEFLATED)
                print(f'Saved {image_name} to: {args.savedir}')
        # Process all images at once
        else:
            # read images
            images = [imread(path) for path in img_paths]
            print(len(images), type(images))
            predictions = predict(
                images=images,
                model=model,
                threshold=args.threshold,
                factor=args.resolution,
                # TODO implement batch_size, but keep overlap and tile_size at defaults
            )
            # Save predictions
            for img, path in zip(predictions, img_paths):
                image_name = os.path.basename(path)
                if args.threshold is None:
                    # save img without compression
                    imwrite(os.path.join(args.savedir, image_name), img)
                else:
                    # save image with compression
                    imwrite(os.path.join(args.savedir, image_name), img,
                            compression=ZIP_DEFLATED)
                print(f'Saved {image_name} to: {args.savedir}')

    else:
        print('Not train / no predict')

    # Final print that process is done
    print('Done!')


if __name__ == '__main__':
    main()
