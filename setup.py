# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
basedir = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(basedir, 'README.org'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = 'antope',
    version = '0.1.0',
    description = 'Demo library',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    # url="https://medium-multiply.readthedocs.io/",
    author = 'Florian Pouthier',
    author_email = 'florian.pouthier@grenoble-inp.fr',
    # license = "MIT", # To be discussed
    # classifiers=[
    #     "Intended Audience :: Developers",
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.6",
    #     "Programming Language :: Python :: 3.7",
    #     "Programming Language :: Python :: 3.8",
    #     "Programming Language :: Python :: 3.9",
    #     "Operating System :: OS Independent"
    # ],
    packages = find_packages(include = ['antope']),
    include_package_data = True,
    install_requires = ['numpy']
)
