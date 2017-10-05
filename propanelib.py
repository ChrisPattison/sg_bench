import pandas as pd
import numpy as np
import json
import io

magic = '%%%---%%%'
comment = '#'
histstart = '|'

def extract_data(output):
    split = 0
    # find magic string
    for num, l in enumerate(output, 0):
        split = num
        if magic in l:
            break
    # get lines to write back later
    lines = output
    
    # remove histograms
    lines = lines[0:split]
    # strip out whitespacee and comments
    lines = [l.split(comment)[0].strip() for l in lines]
    # remove empty lines
    lines = [l for l in lines if len(l)]
    # reassemble in buffer
    buff = io.StringIO(unicode('\n'.join(lines)))
    buff.seek(0)
    data = pd.read_csv(buff, delim_whitespace=True)
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
        bins.append(binned)
    data = pd.concat(bins)
    return data

def make_schedule(sweeps, steps, bondscale, bins=None):
    beta = 0.51/bondscale
    mc_sweeps = 1

    schedule = {'sweeps':int(sweeps), 'solver_mode':True, 'uniform_init':True, \
        'schedule':[{ 'beta':beta, 'gamma':s, 'metropolis':1, 'microcanonical':mc_sweeps } for s in steps],\
        'bin_set':([int(sweeps)/2**i for i in range(8)] if bins is None else bins)}
    return json.dumps(schedule, indent=1)
