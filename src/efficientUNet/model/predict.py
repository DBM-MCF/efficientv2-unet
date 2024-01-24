from math import ceil
from typing import Union

import keras
import numpy
import numpy as np
import cv2


from tifffile import imread

from src.efficientUNet.utils.visualize import show_3images, show_all_images


def predict(images: Union[list, numpy.ndarray],
            model: Union[keras.Model, str],
            factor: int = 1,
            tile_size: int = 512,
            overlap: int = 0,
            batch_size: int = 1
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
            raise TypeError(f'The list of images does not contain numpy '
                            f'arrays. The first item is of type: '
                            f'{type(images[0])}')
    elif not isinstance(images, np.ndarray):
        raise TypeError(
            f'The predict function only works with a numpy array or a list '
            f'of numpy arrays. Got: {type(images)}'
        )

    # load model already, to avoid multiple reloads
    if isinstance(model, str):
        print('Loading model...')
        try:
            model = keras.models.load_model(model)
        except OSError:
            raise IOError(
                f'The path to the EfficientUNet-V2 model is wrong. '
                f'Check the path (it should be an ".h5" file), yours: '
                f'<{model}>')

    # do the prediction for a single image
    if not isinstance(images, list):
        return predict_single_image(
            img=images, model=model,
            factor=factor, tile_size=tile_size,
            overlap=overlap, batch_size=batch_size
        )
    else:
        predictions = []
        # Fixme, would be nice to use tqdm, but im too lazy to install it
        # i.e.:     for i in tqdm(images):
        for i in images:
            predictions.append(
                predict_single_image(
                    img=i, model=model,
                    factor=factor, tile_size=tile_size,
                    overlap=overlap, batch_size=batch_size
                )
            )
        return predictions


def predict_single_image(img,
                         model: Union[keras.Model, str],
                         factor: int = 1,
                         tile_size: int = 512,
                         overlap: int = 0,
                         batch_size: int = 1
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
            f'The tiling overlap should be more than 10% of the tile size. '
            f'You set {overlap}px, but should be more than '
            f'{int(tile_size * 0.1)}px.'
        )
    if not len(img.shape) == 3:
        raise NotImplementedError(
            f'Only 3 channel images are supported. Your image has '
            f'{len(img.shape)} dimensions.'
        )
    if not img.shape[2] == 3:
        raise NotImplementedError(
            f'Only 3 channel images are supported. Your image has '
            f'{img.shape[2]} channels.'
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
    img_resized = cv2.copyMakeBorder(img, pad_top, pad_bottom,
                                     pad_left, pad_right, cv2.BORDER_REFLECT)
    # print('Resized img shape:', img_resized.shape)

    # tile the image
    print('Tiling input image...        ', end='')
    tiles = tile_with_overlap(img_resized,
                              tile_size=tile_size, overlap=overlap)
    print(f'into {len(tiles)} tiles.')

    # predict the tiles         ----------------------------------------------
    if isinstance(model, str):
        print('Loading model...')
        try:
            model = keras.models.load_model(model)
        except OSError:
            raise IOError(
                f'The path to the EfficientUNet-V2 model is wrong. '
                f'Check the path (it should be an ".h5" file), yours: '
                f'<{model}>')
    print('Predicting...')
    if len(tiles) > 1:
        pred_tiles = model.predict(
            np.asarray(tiles),
            batch_size=batch_size
        )
    else:
        pred_tiles = model(np.asarray(tiles))

    # 'Stitch' predicted tiles    --------------------------------------------
    # via blending
    print('Fusing predicted tiles...')
    if len(pred_tiles) > 1:
        blended_image = blend_tiles(
            imgs=pred_tiles,
            n_y=tiles_y + 1,
            n_x=tiles_x + 1,
            overlap=overlap
        )
    else:
        blended_image = pred_tiles[0]

    # Crop back to original size    ------------------------------------------
    blended_image = blended_image[:size_y, :size_x]
    # resize back to original shape
    return cv2.resize(blended_image, (ori_size_x, ori_size_y))


def tile_with_overlap(img, tile_size, overlap):
    """
    Tile an input image with overlapping tiles.
    The image shape must be exact, to fit the overlapping tiles.
    :param img: 3 channel image
    :param tile_size: (int) tile size, e.g. 512
    :param overlap:  (int) overlap between tiles, e.g. 100
    :return: list of tiles (row by row)
    """
    size_y = img.shape[0]
    size_x = img.shape[1]
    if not size_y == tile_size:
        assert size_y == ceil((size_y - tile_size) / (tile_size - overlap)) * \
               (tile_size - overlap) + tile_size, \
               f'The Y dimension of the image ({size_y}), ' \
               f'does not allow proper image tiling.'
    if not size_x == tile_size:
        assert size_x == ceil((size_x - tile_size) / (tile_size - overlap)) * \
               (tile_size - overlap) + tile_size, \
               f'The X dimension of the image ({size_x}), ' \
               f'does not allow proper image tiling.'

    # create tiles
    tiles = []
    for y in range(
        0, img.shape[0] - (tile_size - overlap), tile_size - overlap
    ):
        # print(y, '-', y + tile_size)
        # print('----')
        for x in range(
            0, img.shape[1] - (tile_size - overlap), tile_size - overlap
        ):
            # print(x, '-', x + tile_size)
            crop = img[y:y + tile_size, x:x + tile_size]
            tiles.append(crop)
    return tiles


def blend_tiles(imgs, n_y: int, n_x: int, overlap: int):
    """
    Will 'stitch' tiles into single image.
    :param imgs: list of tiles
    :param n_y: (int) number of Y-tiles
    :param n_x: (int) number of X-tiles
    :param overlap: (int) number of overlapping pixels between tiles
    :return: (np.array) of 'stitched' image
    """
    # sanity checks
    assert len(imgs) == n_y * n_x, f'{len(imgs), n_y, n_x}'
    shape = imgs[0].shape
    for img in imgs:
        assert img.shape == shape
    if len(shape) < 2 or len(shape) > 3:
        raise NotImplementedError(f'Blending tiles is implemented only for YX '
                                  f'and YXC dimensions. You tried to blend an '
                                  f'image with {len(shape)} dimension.')
    dtype_in = imgs[0].dtype

    # create a gradient ramp for blending the overlapping row tile parts    --
    alpha_h = create_gradient(
        width=overlap,
        height=shape[0],
        is_horizontal=True
    )
    # for multichannel images, match the number of channels to alpha
    if len(shape) > 2:
        alpha_h = alpha_h[:, :, None] * np.ones(
            shape[2],
            dtype=int)[None, None, :]

    # 'blend' tiles per row                 ----------------------------------
    blended_rows = []
    for row in range(n_y):
        # initialise row image
        row_img = None
        for tile in range(n_x - 1):
            # get the overlapping tile parts
            middle_left = imgs[tile + (row * n_x)][:, -overlap:]
            middle_right = imgs[tile + (row * n_x) + 1][:, :overlap]

            # blend them together
            blend = (alpha_h * middle_left) + (np.flip(alpha_h) * middle_right)

            # concatenate the row parts
            if tile == 0:  # left most tile
                # Special case, if there is only 2 x-tiles
                if n_x == 2:
                    row_img = np.concatenate(
                        (
                            # left most tile
                            imgs[tile + (row * n_x)][:, :-overlap, ],
                            # blended overlap part
                            blend.astype(dtype_in),
                            # remainder of next tile minus overlap on both ends
                            imgs[tile + (row * n_x) + 1][:, overlap:]
                        ),
                        axis=1
                    )
                else:
                    row_img = np.concatenate(
                        (
                            # left most tile
                            imgs[tile + (row * n_x)][:, :-overlap, ],
                            # blended overlap part
                            blend.astype(dtype_in),
                            # remainder of next tile minus overlap on both ends
                            imgs[tile + (row * n_x) + 1][:, overlap:-overlap]
                        ),
                        axis=1
                    )
            elif tile == n_x - 2:  # right most tile
                row_img = np.concatenate(
                    (
                        # left part
                        row_img,
                        # blended overlap part
                        blend.astype(dtype_in),
                        # remainder of the very last til
                        imgs[tile + (row * n_x) + 1][:, overlap:]
                    ),
                    axis=1
                )
            else:  # tiles in the middle
                row_img = np.concatenate(
                    (
                        # left part
                        row_img,
                        # blended overlap part
                        blend.astype(dtype_in),
                        # remainder of the next tile minus overlap on both ends
                        imgs[tile + (row * n_x) + 1][:, overlap:-overlap]
                    ),
                    axis=1
                )
        blended_rows.append(row_img)

    # create a gradient ramp for blending the overlapping rows      ----------
    alpha_v = create_gradient(
        width=blended_rows[0].shape[1],
        height=overlap,
        is_horizontal=False
    )
    # for multichannel images, match the number of channels to alpha
    if len(shape) > 2:
        alpha_v = alpha_v[:, :, None] * np.ones(
            shape[2],
            dtype=int)[None, None, :]

    # Blend rows into the full image        ----------------------------------
    img_out = None
    for row in range(len(blended_rows) - 1):
        # get the overlapping row parts
        middle_top = blended_rows[row][-overlap:, :]
        middle_bottom = blended_rows[row + 1][:overlap, :]

        # blend them together
        blend = (alpha_v * middle_top) + (np.flip(alpha_v) * middle_bottom)

        # concatenate the rows
        if row == 0:  # first row
            # Special case, if there is only 2 rows
            if len(blended_rows) == 2:
                img_out = np.concatenate(
                    (
                        # very top row
                        blended_rows[row][:-overlap, :],
                        # blended overlap part
                        blend.astype(dtype_in),
                        # very bottom row
                        blended_rows[row + 1][overlap:, :]
                    ),
                    axis=0
                )
            else:
                img_out = np.concatenate(
                    (
                        # very top row
                        blended_rows[row][:-overlap, :],
                        # blended overlap part
                        blend.astype(dtype_in),
                        # remained of next row minus overlap on both ends
                        blended_rows[row + 1][overlap:-overlap, :]
                    ),
                    axis=0
                )
        elif row == len(blended_rows) - 2:  # last row
            img_out = np.concatenate(
                (
                    # previous rows
                    img_out,
                    # blended overlap part
                    blend.astype(dtype_in),
                    # remainder of the next row
                    blended_rows[row + 1][overlap:, :]
                ),
                axis=0
            )
        else:  # rows in the middle
            img_out = np.concatenate(
                (
                    # previous rows
                    img_out,
                    # blended overlap part
                    blend.astype(dtype_in),
                    # remained of next row minus overlap on both ends
                    blended_rows[row + 1][overlap:-overlap, :]
                ),
                axis=0
            )
    return img_out


def create_gradient(width, height, is_horizontal=True, start=1, stop=0):
    """
    Create a one-channel gradient/ramp image with desired size.
    The gradient values are evenly spaced between start and stop.
    :param width: (int) desired image width
    :param height: (int) desired image height
    :param is_horizontal: (bool) if the gradient should flow horizontally
    :param start: start intensity value
    :param stop: end intensity value
    :return: (np.array) of shape = (height, width)
    """
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


# ------ testing -----
if __name__ == '__main__':
    # predict(np.random.random((500, 500)))
    model_path = 'G:/20231006_Martin/EfficientUNet/models/my_efficientUNet-B3_allIMGs/my_efficientUNet-B3_allIMGs_best-ckp.h5'
    img_real = 'G:/20231006_Martin/images/Slide2-26_ChannelBrightfield_Seq0007_XY1.ome.tif'
    img_real = imread(img_real)


    # model = model = keras.models.load_model(model_path)
    img_predicted = predict_single_image(img=img_real, model=model_path, tile_size=512, overlap=0, batch_size=0, factor=4)
    # _show_temp([img_in, img_predicted])
    show_3images(img_real, img_predicted, axes_on=True, thresh=0.5)
