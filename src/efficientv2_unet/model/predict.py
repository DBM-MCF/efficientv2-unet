from math import ceil
from typing import Optional, Union

import cv2
import keras
import numpy
import numpy as np
from tifffile import imread

from efficientv2_unet.utils.tile_utils import blend_tiles, tile_with_overlap

# from efficientv2_unet.utils.visualize import show_3images, show_all_images


def predict(
    images: Union[list, numpy.ndarray],
    model: Union[keras.Model, str],
    threshold: Optional[float] = None,
    factor: int = 1,
    tile_size: int = 512,
    overlap: int = 0,
    batch_size: int = 1,
):
    """
    Predict a list of images with tiling.

    The input images will be tiled (shape = (tile_size, tile_size)), with
    overlaps. The prediction is performed on the tiles, one image after
    the other.
    Eventually, the predicted tiles are fused back together into the final
    output image, and resized to the full (original) resolution.
    :param images: input image (array of 3 channels)
    :param model: either a loaded keras model,
                  or a (str) path to the .h5 model file
    :param threshold: (float) threshold for prediction. If None (or 0),
                      the function will return the probability map, otherwise
                      a binary image.
    :param factor: (int) downscaling factor for input image, should be a
                   multiple of 2 (I think).
    :param tile_size: (int) YX tile size in pixels, (must be 2**x)
                      I think the tile size should be maximized to what
                      the GPU can take
    :param overlap: (int) tile overlap in pixels, if 0 (default) then 15% of
                    the tile size is chosen as overlap
    :param batch_size: (int) number of tiles to be predicted simultaneously,
                       (max. number depends also on the tile size).
                       If set to 0, it defaults to 32.
    :return: predicted image or list of predicted images
    """
    # sanity check
    if isinstance(images, list):
        # only check first item in list
        if not isinstance(images[0], np.ndarray):
            raise TypeError(
                f"The list of images does not contain numpy "
                f"arrays. The first item is of type: "
                f"{type(images[0])}"
            )
    elif not isinstance(images, np.ndarray):
        raise TypeError(
            f"The predict function only works with a numpy array or a list "
            f"of numpy arrays. Got: {type(images)}"
        )

    # load model already, to avoid multiple reloads
    if isinstance(model, str):
        print("Loading model...")
        try:
            model = keras.models.load_model(model)
        except OSError as e:
            raise OSError(
                f"The path to the EfficientUNet-V2 model is wrong. "
                f'Check the path (it should be an ".h5" file), yours: '
                f"<{model}>"
            ) from e

    # do the prediction for a single image
    if not isinstance(images, list):
        img = predict_single_image(
            img=images,
            model=model,
            factor=factor,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
        )
        if threshold is None or threshold == 0:
            return img
        else:
            return (img > threshold).astype(np.uint8)
    else:
        predictions = []
        for i in images:
            img = predict_single_image(
                img=i,
                model=model,
                factor=factor,
                tile_size=tile_size,
                overlap=overlap,
                batch_size=batch_size,
            )
            if threshold is None or threshold == 0:
                predictions.append(img)
            else:
                predictions.append((img > threshold).astype(np.uint8))
        return predictions


def predict_single_image(
    img,
    model: Union[keras.Model, str],
    factor: int = 1,
    tile_size: int = 512,
    overlap: int = 0,
    batch_size: int = 1,
):
    """
    Predict an image with tiling.

    The input image will be tiled (shape = (tile_size, tile_size)), with
    overlaps. The prediction is performed on the tiles.
    Eventually, the predicted tiles are fused back together into the final
    output image, and resized to the full (original) resolution.
    :param img: input image (array of 3 channels)
    :param model: either a loaded keras model,
                  or a (str) path to the .h5 model file
    :param factor: (int) downscaling factor for input image, should be a
                   multiple of 2 (I think).
    :param tile_size: (int) YX tile size in pixels, (must be 2**x)
                      I think the tile size should be maximized to what
                      the GPU can take
    :param overlap: (int) tile overlap in pixels, if 0 (default) then 15% of
                    the tile size is chosen as overlap
    :param batch_size: (int) number of tiles to be predicted simultaneously,
                       (max. number depends also on the tile size).
                       If set to 0, it defaults to 32.
    :return: predicted image
    """
    # sanity checks
    # fixme tile size must be a multiple of /256/ -> must be 2**X
    if overlap == 0:
        overlap = int(tile_size * 0.15)
        # print(f'-- Info: tile size is <{tile_size}x{tile_size}> and has an '
        #       f'overlap of {overlap}px.')
    if overlap < tile_size * 0.1:
        raise RuntimeError(
            f"The tiling overlap should be more than 10% of the tile size. "
            f"You set {overlap}px, but should be more than "
            f"{int(tile_size * 0.1)}px."
        )
    if not len(img.shape) == 3:
        raise NotImplementedError(
            f"Only 3 channel images are supported. Your image has "
            f"{len(img.shape)} dimensions."
        )
    if not img.shape[2] == 3:
        raise NotImplementedError(
            f"Only 3 channel images are supported. Your image has "
            f"{img.shape[2]} channels."
        )
    ori_size_y = img.shape[0]
    ori_size_x = img.shape[1]

    # Downscaling image
    if factor > 1:
        img = cv2.resize(img, (ori_size_x // factor, ori_size_y // factor))
    size_y = img.shape[0]
    size_x = img.shape[1]

    # Input image tiling        ----------------------------------------------
    # calculate the (N)umber of tiles --> (tile_size - overlap) * N + tile_size
    if size_y <= tile_size:
        tiles_y = 0
    else:
        tiles_y = size_y - tile_size
        tiles_y = ceil(tiles_y / (tile_size - overlap))

    if size_x <= tile_size:
        tiles_x = 0
    else:
        tiles_x = size_x - tile_size
        tiles_x = ceil(tiles_x / (tile_size - overlap))

    # resize image by reflecting the border, so it fits the tiles
    pad_top = 0
    pad_bottom = ((tile_size - overlap) * tiles_y + tile_size) - size_y
    pad_left = 0
    pad_right = ((tile_size - overlap) * tiles_x + tile_size) - size_x
    img_resized = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
    )
    # print('Resized img shape:', img_resized.shape)

    # tile the image
    print("Tiling input image...        ", end="")
    tiles = tile_with_overlap(img_resized, tile_size=tile_size, overlap=overlap)
    print(f"into {len(tiles)} tiles.")

    # predict the tiles         ----------------------------------------------
    if isinstance(model, str):
        print("Loading model...")
        try:
            model = keras.models.load_model(model)
        except OSError as e:
            raise OSError(
                f"The path to the EfficientUNet-V2 model is wrong. "
                f'Check the path (it should be an ".h5" file), yours: '
                f"<{model}>"
            ) from e
    print("Predicting...")
    if len(tiles) > 1:
        pred_tiles = model.predict(np.asarray(tiles), batch_size=batch_size)
    else:
        pred_tiles = model(np.asarray(tiles))

    # 'Stitch' predicted tiles    --------------------------------------------
    # via blending
    print("Fusing predicted tiles...")
    if len(pred_tiles) > 1:
        blended_image = blend_tiles(
            imgs=pred_tiles, n_y=tiles_y + 1, n_x=tiles_x + 1, overlap=overlap
        )
    else:
        blended_image = np.asarray(pred_tiles[0])

    # Crop back to original size    ------------------------------------------
    blended_image = blended_image[:size_y, :size_x]
    # resize back to original shape
    return cv2.resize(blended_image, (ori_size_x, ori_size_y))


# ------ testing -----
if __name__ == "__main__":
    # predict(np.random.random((500, 500)))
    model_path = (
        "/models/my_efficientUNet-B3_allIMGs/my_efficientUNet-B3_allIMGs_best-ckp.h5"
    )
    img_real = (
        "G:/20231006_Martin/images/Slide2-26_ChannelBrightfield_Seq0007_XY1.ome.tif"
    )
    img_real = imread(img_real)

    # model = model = keras.models.load_model(model_path)
    img_predicted = predict_single_image(
        img=img_real, model=model_path, tile_size=512, overlap=0, batch_size=0, factor=4
    )
    # _show_temp([img_in, img_predicted])
    # show_3images(img_real, img_predicted, axes_on=True, thresh=0.5)
