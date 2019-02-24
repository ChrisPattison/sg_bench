#!/bin/env python3
from setuptools import setup, find_packages
setup(
    name='sg_bench', 
    packages=find_packages('src'),
    package_dir={'': 'src'}, 
    version='3.0', 
    description='Benchmarking suite for Heuristic Solvers',
    author='Chris Pattison',
    author_email='chpattison@gmail.com',
    install_requires=['numpy', 'scipy', 'pandas', 'paramiko', 'psutil'],
    entry_points={
        'console_scripts':[
            'pt_bench=pt_bench.__main__:main',
            'psqa_bench=psqa_bench.__main__:main',
            'quit_bench=quit_bench.__main__:main',
            'siman_bench=siman_bench.__main__:main',
        ]
    }
    )
