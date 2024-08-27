import sys
from importlib.metadata import PackageNotFoundError, version
from platform import python_version

import tensorflow as tf

try:
    version = version("efficient_v2_unet")
except PackageNotFoundError:
    version = "unknown"


version_summary = f"""
efficient_v2_unet version:  \t{version}
Platform:                   \t{sys.platform}
Python version:             \t{python_version()}
Tensorflow version:         \t{tf.__version__}
Tensorflow GPU support:     \t{tf.test.is_gpu_available()}
Tensorflow GPU with CUDA:   \t{tf.test.is_gpu_available(cuda_only=True)}
"""

if __name__ == "__main__":
    print(version_summary)
