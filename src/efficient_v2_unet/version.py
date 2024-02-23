import tensorflow as tf
import sys
from platform import python_version

from importlib.metadata import PackageNotFoundError, version

try:
    version = version("efficient_v2_unet")
except PackageNotFoundError:
    version = 'unknown'


version_summary = f"""
efficient_v2_unet version:  \t{version}
Platfrom:                   \t{sys.platform}
Python version:             \t{python_version()}
Tensorflow version:         \t{tf.__version__}
Tensorflow GPU support:     \t{tf.test.is_gpu_available()}
Tensorflow GPU with CUDA:   \t{tf.test.is_gpu_available(cuda_only=True)}
"""

if __name__ == "__main__":
    print(version_summary)
