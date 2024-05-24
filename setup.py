# setup.py

import os
from setuptools import setup, find_packages

VERSION = '0.1dev' # TODO: implement version tracking

setup(
    name='pcisa',
    version=VERSION,
    description='SP24 CSE185 Project - PCA implementation in Python',
    author='Isabella Garcia',
    author_email='i4garcia@ucsd.edu',
    packages=find_packages(),
    install_requires=['numpy', 'anndata', 'matplotlib', 'pandas'],
    entry_points={
        'console_scripts': [
            'pcisa=pcisa.__main__:main'
        ]
    },
)