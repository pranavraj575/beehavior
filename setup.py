from distutils.core import setup
from setuptools import find_packages

setup(
    name='beehavior',
    version='6.9.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'msgpack-rpc-python',
                      'matplotlib',
                      'airsim',
                      'opencv-python',
                      ],
    license='Liscence to Krill',
)
