import pathlib
import sys
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

install_requirements = [
    "numpy>1.26",
    "scikit-image",
    "tifffile",
    "matplotlib",
    "albumentations",
    "opencv-python",
    "pandas",  # not sure if this is really used...
    "notebook",
    # previously...
    # "tensorflow==2.10.1;platform_system=='Windows'",
    # "tensorflow-macos==2.11.0;platform_system=='Darwin'",
    # "tensorflow-metal==0.7.0;platform_system=='Darwin'",
    # "tensorflow>=2.10.0,<2.12.0;platform_system!='Windows'",
    # "keras>=2.10.0,<2.12.0;platform_system!='Windows'",
    # "keras==2.10.0;platform_system=='Windows'",
    # "keras==2.11.0;platform_system=='Darwin'",
]

# OS specific tensorflow install
if sys.platform == 'darwin':
    # MacOS (Apple Silicon) specific (Intel may not work)
    install_requirements.append("tensorflow-macos==2.11.0")
    install_requirements.append("tensorflow-metal==0.7.0")
    install_requirements.append("keras==2.11.0")

else:
    # other platforms (only Win tested)
    install_requirements.append("tensorflow==2.10.1")
    install_requirements.append("keras==2.10.0")

setup(
    name="efficient_v2_unet",
    version="0.0.1",
    description="EfficientNetV2 UNet implementation",
    long_description=long_description,
    longdescription_content_type="text/markdown",
    url="",  # TODO
    author="Loïc Sauteur",
    author_email="loic.sauteur@unibas.ch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Image Processing :: Deep Learning",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(),
    python_requires=">=3.9, <4",
    install_requires=install_requirements,
    extras_require={
        "dev": ["pydot"],
        "test": ["pytest"],
    },
    entry_points={
        # this is how the argument parser can be called
        'console_scripts': ['ev2unet = efficient_v2_unet.__main__:main']
    },
    # package_data= # data files to be included into distribution
    # entry_points={
    #    "console_scripts": ["efficient_v2_unet=efficient_v2_unet.XYZ:main"]}
    #    # provide a command called however you want which executes the
    #    # function 'main' from this package when invoked

    # project_url={
    #    "Bug Reports": "https://github.com/loic-sauteur/EfficientUNet/issues",
    #    "Documentation": "https://github.com/loic-sauteur/EfficientUNet",
    #    "Source Code": "https://github.com/loic-sauteur/EfficientUNet",
    # },
)
