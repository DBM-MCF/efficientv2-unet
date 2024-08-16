import math
import random

import numpy as np
from efficient_v2_unet.model.predict import (
    tile_with_overlap, blend_tiles
)
import pytest


def random_rgb_image(size: tuple):
    """
    Create a random RGB image (YXC)
    :param size: tuple (int, int, int) for YXC
    :return: numpy.array
    """
    assert len(size) == 3
    assert size[2] == 3
    img = np.random.random(size) * 255
    return img.astype(np.uint8)


def test_tile_with_overlap():
    img = random_rgb_image((1024, 1024, 3))
    with pytest.raises(AssertionError):
        tile_with_overlap(img, 512, 100)
    img = random_rgb_image((924, 924, 3))
    tiles = tile_with_overlap(img, 512, 100)
    assert len(tiles) == 4
    for tile in tiles:
        assert tile.shape == (512, 512, 3)


def test_blend_tiles():
    """
    Test blending tiles for an 2x2 tiled image
    """
    img = random_rgb_image((924, 924, 3))
    tiles = tile_with_overlap(img, 512, 100)
    blended = blend_tiles(tiles, 2, 2, 100)
    assert blended.shape == img.shape


def test_more_stuff():
    """
    Test tiling and blending for an 1x1 tiled image.
    """
    img = random_rgb_image((512, 512, 3))
    tiles = tile_with_overlap(img, 512, 100)
    assert len(tiles) == 1
    blended = blend_tiles(tiles, 1, 1, 100)
    assert blended.shape == img.shape


def test_random_size_images():
    """
    Test tiling and blending images of "random" sizes
    """
    # create random sized image an assert that it is no valid input
    y = random.randint(513, 4000)
    x = random.randint(513, 4000)
    tile_size = 512
    overlap = 100
    img = random_rgb_image((y, x, 3))
    with pytest.raises(AssertionError):
        tile_with_overlap(img, tile_size, overlap)

    # create random image XY, according to tile size and overlap
    for i in range(4):
        tile_size = random.randint(200, 1000)
        overlap = int(tile_size / 4)
        y = random.randint(1, 10)
        y = y * (tile_size - overlap) + tile_size
        x = random.randint(1, 10)
        x = x * (tile_size - overlap) + tile_size

        img = random_rgb_image((y, x, 3))

        tiles = tile_with_overlap(img, tile_size=tile_size, overlap=overlap)
        # calculate the theoretical number of tiles
        y_tiles = math.ceil((y - tile_size) / (tile_size - overlap)) + 1
        x_tiles = math.ceil((x - tile_size) / (tile_size - overlap)) + 1
        assert len(tiles) == y_tiles * x_tiles

        # blend the tiles
        blended = blend_tiles(tiles, y_tiles, x_tiles, overlap)
        assert blended.shape == img.shape


# ------ run testing functions -----
if __name__ == '__main__':
    pass
