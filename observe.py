#!/usr/bin/python3
import solve
import pinput
import numpy as np
import pandas as pd

def observe(optimize_temp = True):
    config, instances = pinput.get_input('Compute observables for instance class')
    beta_min = config['beta']['min']
    beta_max = config['beta']['max']
    beta_count = config['beta']['count']
    field_strength = config.get('field_strength', 1.0)
    profile = config['profile']
    beta_set = np.linspace(beta_min, beta_max, beta_count)

    if optimize_temp:
        print("Getting initial energy data...")
        initial_data = solve.get_observable(instances, '<E>', beta_set = beta_set, profile = profile, field_strength = field_strength)
        disorder_avg = pd.concat(initial_data).groupby(['Beta']).mean().reset_index()

        beta_set = solve.get_optimized_temps(disorder_avg, beta_min, beta_max, beta_count)

    print("Getting observables...")
    run_data = solve.get_observable(instances, '<E>', beta_set = beta_set, profile = profile, field_strength = field_strength)
    disorder_avg = pd.concat(run_data).groupby(['Beta']).mean().reset_index()
    print(disorder_avg)
if __name__ == "__main__":
    observe()