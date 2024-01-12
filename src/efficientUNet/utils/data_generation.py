import glob
import math
import os
import shutil

import cv2
from cv2 import resize
import numpy as np
from skimage.io import imread, imsave
from keras.utils import Sequence
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, RandomGamma, OneOf,
    GridDistortion, ElasticTransform, OpticalDistortion, RandomSizedCrop,
    RandomRotate90,
    # CLAHE, HueSaturationValue, RandomContrast, ToFloat, ShiftScaleRotate,
    # JpegCompression, RGBShift, RandomBrightness, Blur, MotionBlur,
    # MedianBlur, GaussNoise, CenterCrop, IAAAdditiveGaussianNoise,
)

# import matplotlib.pyplot as plt
# from src.efficientUNet.utils.visualize import show_2images, show_all_images


IMG_SIZE = 256
SEED = 42

"""
INFO: Stain normalization might be good to do, 
to eventually do some color augmentations
"""


class DataSetGenerator(Sequence):

    """
    Generate a dataset of matching raw images and masks.
    Crops the input images to tiles and saves them to corresponding subfolders.
     # modified from:
     https://www.kaggle.com/code/meaninglesslives/
     unet-with-efficientnet-encoder-in-keras
    """
    def __init__(self,
                 train_im_path=None,
                 train_mask_path=None,
                 augmentations: bool = False,
                 batch_size=16,
                 img_size=256,
                 n_channels=3,
                 file_ext='.tif',
                 shuffle=True,
                 resolution: int = 1  # 1= full, 2= half, ect.
                 ):
        """"""
        # setting the random seed, for shuffling images
        np.random.seed(SEED)

        # Initialise
        self.batch_size = batch_size
        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path

        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.file_ext = file_ext
        self.resolution = int(resolution)
        # make sure that the requested
        if self.resolution < 1:
            print("*** INFO: Using full resolution, not as requested:",
                  self.resolution)
            self.resolution = 1
        self.indexes = None
        self.n_classes = 1  # number of max labels in mask

        # list of all crops saved to file
        self.crop_paths = []

        # create augmentations
        if self.augment:
            self.augment = self.create_augmentations()
        else:
            self.augment = None
        # metadata for the dataset
        self.metadata = {}

        # load images
        self.__save_crops__()
        # generate crops at different resolutions
        self.resolution += 1
        self.__save_crops__()
        self.resolution += 1
        self.__save_crops__()
        self.on_epoch_end()

        print(f'Created {len(self.crop_paths)} crops for the dataset.')

        '''
        # check the metadata temp fixme
        for k, v in self.metadata.items():
            if isinstance(v, dict):
                print(k, '=')
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        print('\t', kk, '=')
                        for kkk, vvv in vv.items():
                            print('\t\t', kkk, vvv)
                    else:
                        print('\t', kk, vv)
            else:
                print(k, v)
        '''

    def __save_crops__(self):
        """
        Loads images and saves crops to corresponding subfolders
        :return: None (but populates the self.crop_paths)
        """
        # get images
        _imgs = []
        _masks = []
        crop_count = len(self.crop_paths)  # count the crops
        for count, path in enumerate(
            glob.glob(self.train_im_path + '/*' + self.file_ext)
        ):
            img = imread(path)
            path = path.replace(self.train_im_path, self.train_mask_path)
            mask = imread(path)
            height = img.shape[0]
            width = img.shape[1]

            assert height == mask.shape[0]
            assert width == mask.shape[1]
            assert img.shape[2] == 3, 'Raw image must have 3 channels'
            assert len(mask.shape) == 2, 'No channels in mask image allowed, ' \
                                         f'problematic image: {path}'

            # resize image to resolution requested
            if self.resolution > 1:
                img = resize(img,
                             (width // self.resolution,
                              height // self.resolution)
                             )
                mask = resize(mask,
                              (mask.shape[1] // self.resolution,
                               mask.shape[0] // self.resolution)
                              )
            # pad image so YX dimensions match multiple of crop size
            pad_y = math.ceil(img.shape[0] / self.img_size)
            pad_y = (pad_y * self.img_size) - img.shape[0]
            pad_x = math.ceil(img.shape[1] / self.img_size)
            pad_x = (pad_x * self.img_size) - img.shape[1]
            img = cv2.copyMakeBorder(
                img,
                0,  # top padding
                pad_y,  # bottom padding
                0,  # left padding
                pad_x,  # right padding
                cv2.BORDER_REFLECT
            )
            # same for mask
            mask = cv2.copyMakeBorder(
                mask, 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT
            )

            # check how many labels there are in the mask
            if np.max(mask) > self.n_classes:
                self.n_classes = np.max(mask)

            _imgs.append(img)
            _masks.append(mask)

            # create metadata for image
            img_name = str(os.path.basename(path))
            if count in self.metadata.keys():
                # if image already in metadata, add new resolution key
                img_metadata = self.metadata[count]
                img_metadata.update({f'Resolution@1/{self.resolution}': {
                    'resized_image_shape': (
                        img.shape[0], img.shape[1], img.shape[2]
                    ),
                    'number_of_img_crop': 0}
                })
            else:
                # add image metadata
                img_metadata = {
                    'img_name': img_name,
                    'img_path': os.path.abspath(path.replace(
                        self.train_mask_path,
                        self.train_im_path
                    )),
                    'mask_path': os.path.abspath(path),
                    'original_image_shape': (height, width, img.shape[2]),
                    f'Resolution@1/{self.resolution}': {
                        'resized_image_shape': (
                            img.shape[0], img.shape[1], img.shape[2]
                        ),
                        'number_of_img_crop': 0,
                    }

                }
                self.metadata[count] = img_metadata

        # tile images and masks
        count = 0  # to count the processed images
        for (img, mask) in zip(_imgs, _masks):
            y_tiles = (img.shape[0] // self.img_size) - 1
            x_tiles = (img.shape[1] // self.img_size) - 1
            for y in range(0, y_tiles * self.img_size + 1, self.img_size):
                for x in range(0, x_tiles * self.img_size + 1, self.img_size):
                    img_crop = img[y:y+self.img_size, x:x+self.img_size, :]
                    mask_crop = mask[y:y+self.img_size, x:x+self.img_size]

                    # save crops to file
                    crop_path = os.path.join(self.train_im_path, 'crops')
                    file_name = f'crop{crop_count:04d}.tif'
                    file_path = os.path.join(crop_path, file_name)
                    os.makedirs(crop_path, exist_ok=True)
                    self.crop_paths.append(file_path)
                    imsave(file_path, img_crop, check_contrast=False)
                    crop_path = crop_path.replace(self.train_im_path,
                                                  self.train_mask_path)
                    file_path = file_path.replace(self.train_im_path,
                                                  self.train_mask_path)
                    os.makedirs(crop_path, exist_ok=True)
                    imsave(file_path, mask_crop, check_contrast=False)
                    crop_count += 1

                    # add crop info to this image's metadata
                    d = self.metadata[count][f'Resolution@1/{self.resolution}']
                    d['number_of_img_crop'] += 1
                    d[file_name.replace('.tif', '')] = {
                        'crop_image_path': os.path.abspath(file_path.replace(
                            self.train_mask_path,
                            self.train_im_path
                        )),
                        'crop_mask_path': os.path.abspath(file_path),
                        'start_Y_on_full_image': y,
                        'start_X_on_full_image': x,
                        'end_Y_on_full_image': y + self.img_size,
                        'end_X_on_full_image': x + self.img_size,
                    }
            count += 1

    def get_classes(self):
        """
        Get the number of classes / labels within a dataset.
        I.e. the highest intensity found in any of the mask images.
        :return: (int)
        """
        return self.n_classes

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.crop_paths) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: (int) in range (0 to __len__())
        :return: X, Y (batch for raw and mask)
        """
        indexes = self.indexes[
                  index * self.batch_size:min((index + 1) * self.batch_size,
                                              len(self.crop_paths))
                  ]

        # Find list of IDs
        list_IDs_im = [self.crop_paths[k] for k in indexes]

        # Generate data
        x, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return x, y  # np.array(y)
        else:
            im, mask = [], []
            for _x, _y in zip(x, y):
                augmented = self.augment(image=_x, mask=_y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask)

    def create_augmentations(self):
        return Compose([
            # Augmentation for train images,
            # (augmentations are only applied to the masks when they should)
            OneOf([
                HorizontalFlip(p=0.5),
                RandomRotate90(p=0.5)
            ], p=0.5),
            OneOf([
                RandomGamma(),
                RandomBrightnessContrast(),
            ], p=0.5),
            OneOf([
                # I dont like those transformations, but maybe they do help
                ElasticTransform(alpha=120, sigma=120 * 0.05,
                                 alpha_affine=120 * 0.03),
                GridDistortion(),
                OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
            RandomSizedCrop(min_max_height=(self.img_size // 2, self.img_size),
                            height=self.img_size,
                            width=self.img_size,
                            p=0.5),
            # don't convert to float as efficientNet does it
            # ToFloat(max_value=1)
        ], p=1)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.crop_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        """
        'Generates data containing batch_size samples'
        #X : (n_samples, *dim, n_channels)
        :param list_IDs_im: array of (randomly) selected image paths
        :return: list of images, list of masks (see shape of x & y below)
        """
        # Initialization
        x = np.empty(
            (len(list_IDs_im), self.img_size, self.img_size, self.n_channels)
        )
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):
            x[i, ] = imread(im_path)
            mask_path = im_path.replace(self.train_im_path,
                                        self.train_mask_path)
            y[i, ] = imread(mask_path)
        return np.uint8(x), np.uint8(y)


def split_folder_files_to_train_val(
    image_dir: str,
    mask_dir: str,
    split: float = 0.2,
    file_ext: str = '.tif'
):
    """
    Check if files in image_dir match the mask names.
    Create new sub-folders in the image_dir and mask_dir, for
    randomly selected train and validation images.
    This function will move (not copy) the images.

    :param image_dir: (str) path to raw image directory
    :param mask_dir: (str) path to mask image directory
    :param split: (float) percent for validation data split
    :param file_ext: (str) optional image file extension
    :return: tuple(str) of the paths: train_img, train_mask, val_img, val_mask
    """
    # check if the final folders already exist
    train_img = os.path.join(image_dir, 'train')
    train_mask = os.path.join(mask_dir, 'train')
    val_img = os.path.join(image_dir, 'val')
    val_mask = os.path.join(mask_dir, 'val')
    for folder in [train_img, train_mask, val_img, val_mask]:
        if os.path.exists(folder):
            raise FileExistsError(f'The folder <{folder}> already exists. '
                                  f'Looks like the training data has '
                                  f'already been split...')
    # create those folders
    for folder in [train_img, train_mask, val_img, val_mask]:
        os.makedirs(folder)
        print('Created folder:', folder)

    # check that the files exist
    img_paths = glob.glob(image_dir + "/*" + file_ext)
    # sanity check
    if len(img_paths) == 0:
        raise RuntimeError(f'No <{file_ext}> files found in {image_dir}.')
    missing_masks = []
    for p in img_paths:
        mask_path = p.replace(image_dir, mask_dir)
        if not os.path.exists(mask_path):
            missing_masks.append(mask_path)
    if len(missing_masks) > 0:
        for p in missing_masks:
            print("Missing masks:", p)
        raise FileNotFoundError(f'Could not find {len(missing_masks)} '
                                f'masks. See above.')

    # randomise and split into train and validation images
    np.random.seed(SEED)
    img_paths = np.asarray(img_paths)
    np.random.shuffle(np.asarray(img_paths))
    val_paths = img_paths[:int(len(img_paths)*split)]
    train_paths = img_paths[int(len(img_paths)*split):]

    # move train images
    for path in train_paths:
        file_name = os.path.basename(path)
        new_img_path = os.path.join(train_img, file_name)
        mask_path = path.replace(image_dir, mask_dir)
        new_mask_path = os.path.join(train_mask, file_name)

        # move the train images
        shutil.move(path, new_img_path)
        shutil.move(mask_path, new_mask_path)
    # move validation images
    for path in val_paths:
        file_name = os.path.basename(path)
        new_img_path = os.path.join(val_img, file_name)
        mask_path = path.replace(image_dir, mask_dir)
        new_mask_path = os.path.join(val_mask, file_name)

        # move the train images
        shutil.move(path, new_img_path)
        shutil.move(mask_path, new_mask_path)

    print(f'Split {len(img_paths)} images into {len(train_paths)} train '
          f'images and {len(val_paths)} validation images.')
    return train_img, train_mask, val_img, val_mask


def create_tiles(img, tile_size: int):
    """
    Tile an input image into tile_size'ed tiles.
    If the image shape is not a multiple of tile_size, bottom and right most
    parts will be ignored.
    :param img: (np.array) input image
    :param tile_size: (int) of tile size
    :return: list of tiles (np.arrays)
    """
    y_tiles = (img.shape[0] // tile_size) - 1
    x_tiles = (img.shape[1] // tile_size) - 1
    tiles = []
    for y in range(0, y_tiles * tile_size + 1, tile_size):
        for x in range(0, x_tiles * tile_size + 1, tile_size):
            img_crop = img[y:y + tile_size, x:x + tile_size, :]
            tiles.append(img_crop)
    return tiles


# For testing
if __name__ == '__main__':
    """
    # test DGenInMem
    img_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/train/images"
    mask_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/train/masks"
    a = DGenInMem(train_im_path=img_dir, train_mask_path=mask_dir, augmentations=None, resolution=1, batch_size=32, shuffle=False)
    b = DGenInMem(train_im_path=img_dir, train_mask_path=mask_dir, augmentations=AUGMENTATIONS_TRAIN, resolution=1, batch_size=32, shuffle=False)
    
    aimage, amasks = a.__getitem__(0)
    bimage, bmasks = b.__getitem__(0)
    
    max_imgs = 32
    grid_w = 16
    grid_h = int(max_imgs / grid_w)
    fig, axs = plt.subplots(grid_h*2, grid_w, figsize=(grid_w, grid_h*2))
    
    for i, (im, mask) in enumerate(zip(aimage, amasks)):
        ax = axs[int(i/grid_w), i % grid_w]
        ax.imshow(im)
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.axis('off')
    
    for i, (im, mask) in enumerate(zip(bimage, bmasks)):
        ax = axs[int((i+max_imgs)/grid_w), (i+max_imgs) % grid_w]
        ax.imshow(im)
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.axis('off')
    plt.suptitle("test")
    plt.show()
    
    for i in range(len(aimage)):
        imsave(f'{i}_raw_noAug.tif', aimage[i])
        imsave(f'{i}_raw_yesAug.tif', bimage[i])
        imsave(f'{i}_noAug.tif', amasks[i])
        imsave(f'{i}_yesAug.tif', bmasks[i])
    print('done')
    """

    '''
    # Train/Val data generation
    img_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/train/images"
    mask_dir = "G:/20231006_Martin/Plaque-Size-Samples_annotations_photoshop/train/masks"
    a = DataSetGenerator(train_im_path=img_dir, train_mask_path=mask_dir, augmentations=False, resolution=1, batch_size=32, shuffle=False, img_size=1024)
    b = DataSetGenerator(train_im_path=img_dir, train_mask_path=mask_dir, augmentations=True, resolution=1, batch_size=32, shuffle=False, img_size=1024)
    aimage, amasks = a.__getitem__(0)
    bimage, bmasks = b.__getitem__(0)
    """
    for i in range(len(amasks)):
        imsave(f'{i}notAug.tif', amasks[i])
        imsave(f'{i}yesAug.tif', bmasks[i])
        imsave(f'{i}_rawNotAug.tif', aimage[i])
        imsave(f'{i}_rawYestAug.tif', bimage[i])
    
    """

    print('done')
    images, masks = a.__getitem__(0)
    print('number of batches per epoch', a.__len__())
    print('number of images in one batch', len(images))
    
    max_imgs = len(images)
    grid_w = len(images) // 2
    grid_h = int(max_imgs / grid_w)
    fig, axs = plt.subplots(grid_h*2, grid_w, figsize=(grid_w, grid_h*2))
    
    for i, (im, mask) in enumerate(zip(images, masks)):
        ax = axs[int(i/grid_w), i % grid_w]
        ax.imshow(im)
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.axis('off')
    b_imgs, b_masks = b.__getitem__(0)
    for i, (im, mask) in enumerate(zip(b_imgs, b_masks)):
        ax = axs[int((i+max_imgs)/grid_w), (i+max_imgs) % grid_w]
        ax.imshow(im)
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.axis('off')
    plt.suptitle("test")
    plt.show()
    
    #show_2images(images[5], b_imgs[5])
    #imsave('z_notAug.tif', images[5])
    #imsave('z_yesAut.tif', b_imgs[5])
    '''
