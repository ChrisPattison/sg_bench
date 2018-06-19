#!/bin/env python3
from setuptools import setup, find_packages
setup(
    name='quit_bench', 
    packages=['quit_bench'], 
    version='1.0', 
    description='Benchmarking suite for Quantum Inspired Tempering',
    author='Chris Pattison',
    author_email='chpattison@gmail.com',
    install_requires=['numpy', 'scipy', 'pandas', 'paramiko'],
    py_modules=find_packages())
