from distutils.core import setup
from setuptools import find_packages

setup(
    name='beehavior',
    version='6.9.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'msgpack-rpc-python',
                      'matplotlib',
                      'opencv-python',
                      'gymnasium',
                      # 'airsim', TODO: NEED TO INSTALL THIS AFTER, it needs numpy and msgpack to already be installed
                      ],
    license='Liscence to Krill',
)
