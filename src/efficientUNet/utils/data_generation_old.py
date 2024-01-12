import glob

import cv2
import numpy as np
from skimage.io import imread
import keras
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)

IMG_SIZE = 256
SEED = 42


class DataGenerator(keras.utils.Sequence):
    # from: https://www.kaggle.com/code/meaninglesslives/unet-with-efficientnet-encoder-in-keras
    def __init__(self,
                 train_im_path=None,
                 train_mask_path=None,
                 augmentations=None,
                 batch_size=16,
                 img_size=IMG_SIZE,
                 n_channels=3,
                 file_ext='.tif',
                 shuffle=True,
                 tile=None # or 4, 8, 16 etc. tiles
                 ):
        np.random.seed(42)
        self.batch_size = batch_size
        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path

        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.tile = tile
        self.file_ext = file_ext
        self.__load_img__()
        self.on_epoch_end()

        self.train_im_paths = glob.glob(train_im_path + '/*' + file_ext)
        # setting the random seed, for shuffling images

    def __load_img__(self):
        _imgs = []
        _masks = []

        for path in glob.glob(self.train_im_path + '/*' + self.file_ext):
            _imgs.append(imread(path))
            _masks.append((imread(path.replace(
                self.train_im_path, self.train_mask_path
            ))))


        self.imgs = []
        self.masks = []

        if self.tile is None:
            self.imgs = _imgs
            self.masks = _masks
        elif self.tile != 4:
            print('Tiling only in 4 tiles supported, aborting...')
            end
        else:
            for img in _imgs:
                if img.shape[0] % 2 != 0:
                    print("image with shape =", img.shape,
                          "cannot be tiled evenly, aborting...")
                    end
                y = img.shape[0]
                x = img.shape[1]
                self.imgs.append(img[:y // 2, :x // 2, :])
                self.imgs.append(img[y // 2:, :x // 2, :])
                self.imgs.append(img[:y // 2, x // 2:, :])
                self.imgs.append(img[y // 2:, x // 2:, :])
            for img in _masks:
                y = img.shape[0]
                x = img.shape[1]
                self.masks.append(img[:y // 2, :x // 2])
                self.masks.append(img[y // 2:, :x // 2])
                self.masks.append(img[:y // 2, x // 2:])
                self.masks.append(img[y // 2:, x // 2:])


    def __len__(self):
        return int(np.ceil(len(self.imgs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:min((index + 1) * self.batch_size,
                                              len(self.imgs))]

        # Find list of IDs
        list_IDs_im = [self.imgs[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y)# / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask)# / 255

    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty(
            (len(list_IDs_im), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):

            im = self.imgs[i]
            mask = self.masks[i]
            #im = np.array(imread(im_path))
            #mask_path = im_path.replace(self.train_im_path,
            #                            self.train_mask_path)

            #mask = np.array(imread(mask_path))

            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            # Store class
            y[i,] = cv2.resize(mask, (self.img_size, self.img_size))[
                ..., np.newaxis]
            y[y > 0] = 255

        return np.uint8(X), np.uint8(y)



AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(128, 256), height=IMG_SIZE, width=IMG_SIZE, p=0.5),
    ToFloat(max_value=1)
], p=1)


AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=1)
], p=1)
