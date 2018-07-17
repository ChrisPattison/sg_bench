import subprocess
import pandas as pd
import numpy as np
import json
import io
from psqa_bench import propanelib

def extract_data(output):
    return propanelib.extract_data(output)

def make_schedule(population, param_set, wolff_sweeps, precool, bins=None):
    anneal_schedule = ([{ 
            'beta':param_set['beta'][i], 
            'gamma':param_set['gamma'][i],
            'lambda':param_set['lambda'][i]} 
        for i in range(len(param_set['beta']))])
    precool_schedule = ([{ 
            'beta':beta, 
            'gamma':param_set['gamma'][0],
            'lambda':param_set['lambda'][0]} 
        for beta in np.linspace(0, param_set['beta'][0], precool)])
    
    schedule = {'default_wolff':wolff_sweeps, 'solver_mode':True,
        'schedule':(precool_schedule + anneal_schedule),
        'population':population}
    return json.dumps(schedule, indent=1)
