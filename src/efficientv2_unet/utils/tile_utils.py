from math import ceil

import numpy as np


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
        assert (
            size_y
            == ceil((size_y - tile_size) / (tile_size - overlap))
            * (tile_size - overlap)
            + tile_size
        ), (
            f"The Y dimension of the image ({size_y}), "
            f"does not allow proper image tiling."
        )
    if not size_x == tile_size:
        assert (
            size_x
            == ceil((size_x - tile_size) / (tile_size - overlap))
            * (tile_size - overlap)
            + tile_size
        ), (
            f"The X dimension of the image ({size_x}), "
            f"does not allow proper image tiling."
        )

    # create tiles
    tiles = []
    for y in range(0, img.shape[0] - (tile_size - overlap), tile_size - overlap):
        # print(y, '-', y + tile_size)
        # print('----')
        for x in range(0, img.shape[1] - (tile_size - overlap), tile_size - overlap):
            # print(x, '-', x + tile_size)
            crop = img[y : y + tile_size, x : x + tile_size]
            tiles.append(crop)
    return tiles


def blend_tiles(imgs, n_y: int, n_x: int, overlap: int):
    """
    'Stitch' tiles into single image.

    :param imgs: list of tiles
    :param n_y: (int) number of Y-tiles
    :param n_x: (int) number of X-tiles
    :param overlap: (int) number of overlapping pixels between tiles
    :return: (np.array) of 'stitched' image
    """
    # sanity checks
    assert len(imgs) == n_y * n_x, f"{len(imgs), n_y, n_x}"
    shape = imgs[0].shape
    for img in imgs:
        assert img.shape == shape
    if len(shape) < 2 or len(shape) > 3:
        raise NotImplementedError(
            f"Blending tiles is implemented only for YX "
            f"and YXC dimensions. You tried to blend an "
            f"image with {len(shape)} dimension."
        )
    dtype_in = imgs[0].dtype

    # if there is only one tile, return it
    if len(imgs) == 1:
        return imgs[0]

    # create a gradient ramp for blending the overlapping row tile parts    --
    alpha_h = create_gradient(width=overlap, height=shape[0], is_horizontal=True)
    # for multichannel images, match the number of channels to alpha
    if len(shape) > 2:
        alpha_h = alpha_h[:, :, None] * np.ones(shape[2], dtype=int)[None, None, :]

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
                            imgs[tile + (row * n_x)][
                                :,
                                :-overlap,
                            ],
                            # blended overlap part
                            blend.astype(dtype_in),
                            # remainder of next tile minus overlap on both ends
                            imgs[tile + (row * n_x) + 1][:, overlap:],
                        ),
                        axis=1,
                    )
                else:
                    row_img = np.concatenate(
                        (
                            # left most tile
                            imgs[tile + (row * n_x)][
                                :,
                                :-overlap,
                            ],
                            # blended overlap part
                            blend.astype(dtype_in),
                            # remainder of next tile minus overlap on both ends
                            imgs[tile + (row * n_x) + 1][:, overlap:-overlap],
                        ),
                        axis=1,
                    )
            elif tile == n_x - 2:  # right most tile
                row_img = np.concatenate(
                    (
                        # left part
                        row_img,
                        # blended overlap part
                        blend.astype(dtype_in),
                        # remainder of the very last til
                        imgs[tile + (row * n_x) + 1][:, overlap:],
                    ),
                    axis=1,
                )
            else:  # tiles in the middle
                row_img = np.concatenate(
                    (
                        # left part
                        row_img,
                        # blended overlap part
                        blend.astype(dtype_in),
                        # remainder of the next tile minus overlap on both ends
                        imgs[tile + (row * n_x) + 1][:, overlap:-overlap],
                    ),
                    axis=1,
                )
        blended_rows.append(row_img)

    # create a gradient ramp for blending the overlapping rows      ----------
    alpha_v = create_gradient(
        width=blended_rows[0].shape[1], height=overlap, is_horizontal=False
    )
    # for multichannel images, match the number of channels to alpha
    if len(shape) > 2:
        alpha_v = alpha_v[:, :, None] * np.ones(shape[2], dtype=int)[None, None, :]

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
                        blended_rows[row + 1][overlap:, :],
                    ),
                    axis=0,
                )
            else:
                img_out = np.concatenate(
                    (
                        # very top row
                        blended_rows[row][:-overlap, :],
                        # blended overlap part
                        blend.astype(dtype_in),
                        # remained of next row minus overlap on both ends
                        blended_rows[row + 1][overlap:-overlap, :],
                    ),
                    axis=0,
                )
        elif row == len(blended_rows) - 2:  # last row
            img_out = np.concatenate(
                (
                    # previous rows
                    img_out,
                    # blended overlap part
                    blend.astype(dtype_in),
                    # remainder of the next row
                    blended_rows[row + 1][overlap:, :],
                ),
                axis=0,
            )
        else:  # rows in the middle
            img_out = np.concatenate(
                (
                    # previous rows
                    img_out,
                    # blended overlap part
                    blend.astype(dtype_in),
                    # remained of next row minus overlap on both ends
                    blended_rows[row + 1][overlap:-overlap, :],
                ),
                axis=0,
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
