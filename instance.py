import numpy as np
import pandas as pd
import os
import bondfile

def read_instance(filename):
    instance = {}
    instance['bonds'] = bondfile.read_bondfile(filename)
    instance['bondscale'] = instance['bonds']['J_ij'].abs().max()
    instance['ground_energy'] = np.NaN
    instance['size'] = np.NaN
    return instance

def get_size(filename, size = np.NaN, bondscale = None):
    ground_states = pd.read_csv(filename, names=['bondfile', 'gse'], delim_whitespace=True, header=None)
    instances = []
    for index, row in ground_states.iterrows():
        instances.append(read_instance(os.path.join(os.path.dirname(filename), row['bondfile'])))

        if bondscale is not None:
            instances[-1]['bondscale'] = bondscale
        instances[-1]['ground_energy'] = row['gse']
        instances[-1]['size'] = size
    return instances