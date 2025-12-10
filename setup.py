#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="spnf",
    version="0.1.0",
    description="Spectral Prefiltering of Neural Fields",
    packages=find_packages(exclude=("tests", "Jupyter")),
    install_requires=[
        "torch>=2.0",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "tqdm>=4.65",
        "imageio>=2.31",
        "numpy>=1.24",
        "scipy>=1.10",
    ],
)
