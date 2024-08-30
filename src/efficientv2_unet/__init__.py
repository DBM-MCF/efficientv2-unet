"""A U-Net implementation of the EfficientNetV2."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("efficient-v2-unet")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Loïc Sauteur"
__email__ = "loic.sauteur@unibas.ch"
