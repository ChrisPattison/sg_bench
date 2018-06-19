import numpy as np
import pandas as pd
import os
import io
import pathlib
from quit_bench import bondfile

def read_instance(filename):
    instance = {}
    instance['bonds'] = bondfile.read_bondfile(filename)
    instance['bondscale'] = instance['bonds']['J_ij'].abs().max()
    instance['ground_energy'] = np.NaN
    instance['size'] = np.NaN
    return instance

def read_instance_list(filename):
    set_path = pathlib.Path(filename)
    inst_list = pd.read_csv(str(set_path.resolve()), names=['bondfile', 'gse'], delim_whitespace=True, header=None)
    inst_list['bondfile'] = [str(bondfile_path.resolve()) if bondfile_path.is_absolute() else str((set_path.parents[0] / bondfile_path).resolve()) for bondfile_path in np.vectorize(pathlib.Path)(inst_list['bondfile'])]
    return inst_list

def write_instance_list(filename, instances):
    set_path = pathlib.Path(filename)
    with set_path.open('w') as f:
        f.write('\n'.join([r['bondfile'] + ' ' + str(r['gse']) for i, r in instances.iterrows()]))

def get_instance_set(filename, size = np.NaN, bondscale = None):
    set_path = pathlib.Path(filename)
    ground_states = read_instance_list(filename)
    instances = []
    for index, row in ground_states.iterrows():
        bondfile_path = pathlib.Path(row['bondfile'])
        instances.append(read_instance(bondfile_path))

        if bondscale is not None:
            instances[-1]['bondscale'] = bondscale
        instances[-1]['ground_energy'] = row['gse']
        instances[-1]['size'] = size
    return instances
