import pandas as pd
import numpy as np
import json
import io
from quit_bench import propanelib


def extract_data(output):
    data = propanelib.extract_data(output)
    # Bin samples
    bins = []
    for name, group in data.groupby(['Gamma', 'Beta']):
        binned = group.sort_values('Samples').reset_index()
        del binned['index']
        binned['Bin'] = binned['Samples']
        binned['Binned'] = False
        pre_binned = binned.copy()
        if(len(group) > 1):
            upper = binned
            lower = binned.shift(1)
            binned_values = (upper.mul(upper['Samples'], axis='index') - lower.mul(lower['Samples'], axis='index')).div(upper['Samples'] - lower['Samples'], axis='index')
            binned_values['Samples'] = upper['Samples'] - lower['Samples']
            binned.iloc[1:] = binned_values.iloc[1:]
            
            binned.loc[binned.index[1:], 'Binned'] = True
            binned['E_MIN'] = pre_binned['E_MIN']
            assert(binned.iloc[-1]['E_MIN'] == binned['E_MIN'].min())
            binned['Total_Sweeps'] = pre_binned['Total_Sweeps']
            binned['Total_Walltime'] = pre_binned['Total_Walltime']
            binned['Bin'] = pre_binned['Bin']
        bins.append(binned)
    data = pd.concat(bins)
    return data

def make_schedule(sweeps, param_set, mc_sweeps, hit_criteria, bins=None):
    assert(len(param_set['driver']) == len(param_set['beta']))
    assert(len(param_set['problem']) == len(param_set['beta']))
    schedule = {'sweeps':int(sweeps), 'solver_mode':True, 'uniform_init':False, 'hit_criteria':hit_criteria,
        'schedule':[{ 
            'beta':param_set['beta'][i], 
            'gamma':param_set['driver'][i], 
            'lambda':param_set['problem'][i], 
            'heatbath':1, 
            'microcanonical':mc_sweeps } for i in range(len(param_set['beta']))],
        'bin_set':([int(sweeps)//2**i for i in range(8)] if bins is None else bins)}
    return json.dumps(schedule, indent=1)
