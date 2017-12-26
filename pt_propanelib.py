import propanelib
import pandas as pd
import numpy as np
import json
import io


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

def make_schedule(sweeps, steps, bondscale, field_strength = 1.0, bins=None, beta = 10):
    beta /= bondscale
    field_strength *= bondscale

    mc_sweeps = 10
    schedule = {'sweeps':int(sweeps), 'solver_mode':True, 'uniform_init':False, \
        'schedule':[{ 'beta':beta, 'gamma':s*field_strength, 'lambda':1.-s, 'heatbath':1, 'microcanonical':mc_sweeps } for s in steps],\
        'bin_set':([int(sweeps)/2**i for i in range(8)] if bins is None else bins)}
    return json.dumps(schedule, indent=1)
