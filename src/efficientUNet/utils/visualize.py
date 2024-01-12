import matplotlib.pyplot as plt
import numpy as np


def show_2images(img1, img2, thresh=0, axes_on=False):
    """
    show raw image next to probability mask
    :param img1: raw image
    :param img2: probability image
    :param thresh: (float) if threshold >0 and <=1, probability maks will be
                    shown as overlay on img1
    :param axes_on: whether to show the axes or not (on image)
    :return: none - shows image in window
    """
    axes_on = 'on' if axes_on else 'off'

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(img1)
    if thresh > 0 and thresh <= 1:
        axs[0].imshow(np.squeeze(img2) > thresh, alpha=0.2, cmap='Greens')
    axs[1].imshow(np.squeeze(img2))
    axs[0].axis(axes_on)
    axs[1].axis(axes_on)
    plt.show()


def show_3images(img1, img2, thresh=0.9, axes_on=False):
    """
    show raw image next to probability,
    next to thresholded probability mask.
    :param img1: raw image
    :param img2: probability image
    :param thresh: threshold value for probability maks
    :param axes_on: whether to show the axes or not (on image)
    :return: none - shows image in window
    """
    axes_on = 'on' if axes_on else 'off'

    _, axs = plt.subplots(1, 3)
    axs[0].imshow(img1)
    axs[0].axis(axes_on)
    axs[1].imshow(np.squeeze(img2))
    axs[1].axis(axes_on)
    axs[2].imshow(img1)
    axs[2].imshow(np.squeeze(img2 > thresh), alpha=0.2, cmap='Greens')
    axs[2].axis(axes_on)
    plt.show()


def show_all_images(images, axes_on=False):
    """
    Show all images in input array in a row.
    :param images: list of images
    :param axes_on: whether to show the axes or not (on image)
    :return: None, shows images in viewer
    """
    axes_on = 'on' if axes_on else 'off'

    _, axs = plt.subplots(1, len(images))
    for i in range(len(images)):
        axs[i].imshow(images[i])
        axs[i].axis(axes_on)
    plt.show()
