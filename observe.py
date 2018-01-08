#!/usr/bin/python3
import solve
import pinput
import numpy as np
import pandas as pd

def observe(optimize_temp = True):
    config, instances = pinput.get_input('Compute observables for instance class')
    beta_set = np.linspace(config['beta']['min'], config['beta']['max'], config['beta']['count'])

    run_data = solve.get_observable(instances, '<E>', \
        beta_set = beta_set, \
        profile = config['profile'], \
        field_strength = config.get('field_strength', 1.0))
    disorder_avg = pd.concat(run_data).groupby(['Beta']).mean().reset_index()
    print(disorder_avg)
if __name__ == "__main__":
    observe()