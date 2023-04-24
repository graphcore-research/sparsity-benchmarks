# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from setuptools import setup, find_packages

with open("README.md", "r") as readme:
    long_description = readme.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='reptil',
    version='0.3.1',
    author='Graphcore',
    install_requires=requirements,
    python_requires='>=3.6',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    package_data={"": ["requirements.txt"]},
    description='REPTIL - (pva) REPort uTILs',
    long_description=long_description,
)
