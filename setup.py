# setup.py

import os
from setuptools import setup, find_packages

VERSION = '0.1dev' # TODO: implement version tracking


# # version-keeping code based on pybedtools & gymreklab demo project (https://github.com/gymreklab/cse185-demo-project/tree/main)
# curdir = os.path.abspath(os.path.dirname(__file__))
# MAJ = 0
# MIN = 0
# REV = 0
# VERSION = '%d.%d.%d' % (MAJ, MIN, REV)
# with open(os.path.join(curdir, 'pcisa/version.py'), 'w') as fout:
#         fout.write(
#             "\n".join(["",
#                        "# THIS FILE IS GENERATED FROM SETUP.PY",
#                        "version = '{version}'",
#                        "__version__ = version"]).format(version=VERSION)
#         )

setup(
    name='pcisa',
    version=VERSION,
    description='SP24 CSE185 Project - PCA implementation in Python',
    author='Isabella Garcia',
    author_email='i4garcia@ucsd.edu',
    packages=find_packages(),
    install_requires=['numpy', 'anndata', 'matplotlib', 'pandas', 'scipy'],
    entry_points={
        'console_scripts': [
            'pcisa=pcisa.pcisa:main'
        ]
    },
)