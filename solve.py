import numpy as np
import scipy as sp
import pandas as pd
import warnings
import tempfile
import bondfile
import propanelib
import localrun

def run_instances(schedule, instances, restarts = 100):
    with tempfile.NamedTemporaryFile('w') as sch_file:
        # write schedule
        sch_file.write(schedule)
        sch_file.flush()

        for i in instances:
            with tempfile.NamedTemporaryFile('w') as bonds_file:
                # write bond file
                bondfile.write_bondfile(i['bonds'], bonds_file)
                bonds_file.flush()
                
                # run restarts
                i['results'] = localrun.get_data(sch_file.name, bonds_file.name, restarts = restarts)
    return instances

# Get TTS given a temperature set and sweep count
def get_tts(instances, temp_set, sweeps):
    results = []
    
    # make schedule
    schedule = propanelib.make_schedule(sweeps, temp_set, instances[0]['bondscale'])
    instances = run_instances(schedule, instances)
    
    tts = []
    for i in instances:
        success_prob = np.mean(np.isclose(i['ground_energy'], i['results']['E_MIN']))
        tts.append(np.mean(i['results']['Total_Sweeps'])*np.log(1-.99)/np.log(1. - success_prob))
    
    return tts

# Find optimal TTS given a temperature set
# Should this cost function be bootstrapped for the fit?
# Double sweeps until the minimum TTS is included in the range
# Fit polynomial to TTS to find optimum sweep count
def get_opt_tts(instances, temp_set, init_sweeps=128, cost=np.median):
    sweeps = init_sweeps
    trials = []
    trials.append({'tts':cost(get_tts(instances, temp_set, sweeps)), 'sweeps':sweeps})
    sweeps *= 2

    while True:
        trials.append({'tts':cost(get_tts(instances, temp_set, sweeps)), 'sweeps':sweeps})
        # Upper bound on minimum TTS given by maximum of range
        if trials[-2] < trials[-1]:
            break
        sweeps *= 2

    if len(trials) <=2:
        warnings.warn('Minimum TTS found in less than 2 iterations')

    # fit to find optimal sweep count
    trials = pd.DataFrame.from_records(trials)
    opt_sweeps = int(np.exp(np.polyfit(trials['sweeps'].log(), trials['tts'].log(), 2)[2]))
    # Return TTS at optimal sweep count
    return get_tts(instances, opt_sweeps)

# Check whether the results are thermalized based on residual from last bin
def check_thermalized(data, obs, threshold=.99):
    for name, group in data.groupby(['Gamma']):
        sorted_group = group.sort_values(['Bin'])
        if(np.abs(sorted_group.iloc[-1][obs] - sorted_group.iloc[-2][obs])/np.mean(sorted_group.iloc[:-2][obs]) < threshold):
            return False
    return True

# Return observables with thermalization based on observable obs
def get_observable(instances, obs, temp_set, max_iterations = 4):
    sweeps = 4096
    for i in range(max_iterations):
        schedule = propanelib.make_schedule(sweeps*4**i, temp_set, instances[0]['bondscale'])
        instances = run_instances(schedule, instances, restarts = 1)
        # check equillibriation
        if np.all(np.vectorize(lambda i, obs: check_thermalized(i['results'], obs))(instances, obs)):
            break
        
        if i == max_iterations-1:
            warnings.warn('Maximum iterations in get_observable reached')
    return [i['results'][i['results']['Bin']==i['results']['Bin'].max()] for i in instances]

# Disorder average <E>(Gamma)
# Fit temperatures to make dEd1/T constant
# Optimize MC move count (NOT IMPLEMENTED)
# Get optimal TTS
def bench_tempering(instances):
    temp_set = np.linspace(3, 0, 32)
    # fit to disorder averaged E(Gamma)
    disorder_avg = pd.concat(get_observable(instances, '<E>', temp_set)).groupby(['Gamma']).mean()
    energy = sp.interpolate.interp1d(disorder_avg['Gamma'], disorder_avg['<E>'], kind='quadratic')
    fixed = [temp_set[0], temp_set[-1]]
    del temp_set[0]
    del temp_set[-1]
    # new temperature set has constant dE*d1/T
    # List comprehensions to do binary operations
    residual = lambda x: [cost - np.mean(cost) for cost in [np.ediff1d(sorted) * np.ediff1d(energy(reciprocal(sorted))) for sorted in [np.sort(np.reciprocal(np.concatenate((x, fixed))))]]][0]
    temperatures = sp.optimize.minimize(residual, temp_set,  bounds=[(fixed[-1], fixed[0]) for t in temp_set])
    temperatures = np.flip(np.sort(np.concatenate((temperatures, temp_set))))
    print(temperatures)
    return get_opt_tts(instances, temperatures)


