import subprocess
import propanelib
import pandas as pd
import numpy as np
import json
import io

def run_restart(schedule_file, instance_file): # schedule, instance
    command = ['propane', '-m', '1', schedule_file, instance_file]
    output = subprocess.check_output(command, universal_newlines=True)
    output = output.split('\n')
    restart_data = propanelib.extract_data(output)
    return restart_data

def extract_data(output):
    return propanelib.extract_data(output)

def make_schedule(population, sweeps, beta_stop = 10):
    steps = np.linspace(0.0, beta_stop, sweeps)
    schedule = {'population':population, 'solver_mode':True, 'default_sweeps':1, 'schedule':[{ 'beta':s } for s in steps]}
    schedule['schedule'][-1]['compute_observables'] = True
    return json.dumps(schedule, indent=1)
