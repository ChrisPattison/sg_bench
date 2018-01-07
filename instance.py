import numpy as np
import pandas as pd
import os
import bondfile
import pathlib

def read_instance(filename):
    instance = {}
    instance['bonds'] = bondfile.read_bondfile(filename)
    instance['bondscale'] = instance['bonds']['J_ij'].abs().max()
    instance['ground_energy'] = np.NaN
    instance['size'] = np.NaN
    return instance

def get_instance_set(filename, size = np.NaN, bondscale = None):
    set_path = pathlib.Path(filename)
    ground_states = pd.read_csv(str(set_path.resolve()), names=['bondfile', 'gse'], delim_whitespace=True, header=None)
    instances = []
    for index, row in ground_states.iterrows():
        instances.append(read_instance(str((set_path.parents[0] / row['bondfile']).resolve())))

        if bondscale is not None:
            instances[-1]['bondscale'] = bondscale
        instances[-1]['ground_energy'] = row['gse']
        instances[-1]['size'] = size
    return instances