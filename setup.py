import os
from setuptools import setup, find_packages

setup(
    name='pcisa',
    version=VERSION,
    description='SP24 CSE185 Project - PCA implementation in Python',
    author='Isabella Garcia',
    author_email='i4garcia@ucsd.edu',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'pandas'],
    entry_points={
        'console_scripts': [
            'pcisa=pcisa.__main__:main'
        ]
    },
)