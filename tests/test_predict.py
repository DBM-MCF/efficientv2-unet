import matplotlib.pyplot as plt
import numpy as np
from src.efficient_v2_unet.model.predict import (
    tile_with_overlap, blend_tiles
)
import pytest
from src.efficient_v2_unet.utils.visualize import show_all_images


def random_rgb_image(size: tuple):
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
    img = random_rgb_image((924, 924, 3))
    tiles = tile_with_overlap(img, 512, 100)
    blended = blend_tiles(tiles, 2, 2, 100)
    assert blended.shape == img.shape

def test_more_stuff():
    img = random_rgb_image((512, 512, 3))
    tiles = tile_with_overlap(img, 512, 100)
    blended = blend_tiles(tiles, 1, 1, 100)
    assert blended.shape == img.shape


# ------ run testing functions -----
if __name__ == '__main__':
    pass
