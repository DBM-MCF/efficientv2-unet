import pathlib

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4",
    install_requires=[
        "tensorflow==2.10.1",
        "numpy>1.26",
        "scikit-image",
        "tifffile",
        "matplotlib",
        "albumentations",
        "opencv-python",
        "pandas",
        "keras==2.10.0",
        "notebook",
    ],
    extras_require={
        "dev": ["pydot"],
        "test": ["pytest"],
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