import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
import tempfile
import bondfile
import backend
import pt_propanelib as propanelib

# Get TTS given a temperature set and sweep count
def get_tts(instances, beta_set, profile, sweeps, field_strength, restarts):
    results = []
    
    # make schedule
    schedule = propanelib.make_schedule( \
            sweeps = sweeps, \
            beta_set = beta_set, \
            profile = profile, \
            field_strength = field_strength)
    instances = backend.get_backend().run_instances(schedule, instances, restarts, statistics=False)

    tts = []
    for i in instances:
        success_prob = np.mean(np.isclose(i['ground_energy'], i['results'].groupby('restart').min()['E_MIN']))
        if not np.isclose(success_prob, 1.0):
            warnings.warn('TTS run timed out. Success probability: '+str(success_prob))

        runtimes = np.sort(np.apply_along_axis(np.asscalar, 1, i['results'].groupby('restart')['Total_Sweeps'].unique().reset_index()['Total_Sweeps'].tolist()))
        success = np.linspace(0., 1., len(runtimes))
        # make this shorter
        unique_runtimes = []
        unique_success = []
        for i in range(len(runtimes)-1):
            if runtimes[i] < runtimes[i+1]:
                unique_runtimes.append(runtimes[i])
                unique_success.append(success[i])

        prob = sp.interpolate.interp1d(unique_runtimes, unique_success, kind='linear', bounds_error=True)
        clipped_prob = lambda x: np.clip(prob(x), 0.0, 1.0)
        instance_tts = lambda t: t * np.log(1.-.99)/np.log(1.-clipped_prob(t))

        optimized = sp.optimize.minimize(instance_tts, np.median(unique_runtimes), method='TNC', bounds=[(unique_runtimes[1]+1e-4, unique_runtimes[-1]-1e-4)])
        if optimized.success:
            optimal_runtime = optimized['x'][0]
            optimal_tts = instance_tts(optimal_runtime)
            tts.append(optimal_tts)
        else:
            warnings.warn('Optimization for TTS failed.')
    
    return tts

def fit_opt_sweeps(trials):
    trials = pd.DataFrame.from_records(trials)
    trials = trials.sort_values('sweeps').tail(4)
    trials['log_sweeps'] = np.log(trials['sweeps'])
    trials['log_tts'] = np.log(trials['tts'])
    fit = sp.optimize.least_squares(lambda x: ((x[0] * (trials['log_sweeps'] - x[1])**2 + x[2])-trials['log_tts']), \
        [np.mean(trials['log_tts']), trials['log_sweeps'].iloc[-2], np.mean(trials['log_tts'])])['x']
    opt_sweeps = int(np.exp(fit[1]))
    if opt_sweeps > trials['sweeps'].max():
        warnings.warn('Optimal sweep count more than maximum sweep count tested. Got: '+str(opt_sweeps))
        assert(opt_sweeps < 2*trials['sweeps'].max())
    return opt_sweeps

# Find optimal TTS given a temperature set
# Should this cost function be bootstrapped for the fit?
# Double sweeps until the minimum TTS is included in the range
# Fit polynomial to TTS to find optimum sweep count
def get_opt_tts(instances, beta_set, profile, field_strength, restarts, cost=np.median):
    return get_tts(instances, beta_set, profile, 65536, field_strength, restarts=restarts)

# Check whether the results are thermalized based on residual from last bin
def check_thermalized(data, obs, threshold=.001):
    for name, group in data.groupby(['Beta']):
        sorted_group = group.sort_values(['Samples'])
        residual = np.abs(sorted_group.iloc[-1][obs] - sorted_group.iloc[-2][obs])/np.mean(sorted_group.iloc[-2:][obs])
        if(residual > threshold):
            print(str(obs) + ' not thermalized. Residual: '+str(residual))
            return False
    return True

# Return observables with thermalization based on observable obs
def get_observable(instances, obs, beta_set, profile, field_strength, max_iterations = 3):
    sweeps = 4096
    solver = backend.get_backend()
    for i in range(max_iterations):
        schedule = propanelib.make_schedule( \
            sweeps = sweeps, \
            beta_set = beta_set, \
            profile = profile, \
            field_strength = field_strength)
        instances = solver.run_instances(schedule, instances, restarts = 1)
        # check equillibriation
        if np.all(np.vectorize(lambda i, obs: check_thermalized(i['results'], obs))(instances, obs)):
            break
        
        if i == max_iterations-1:
            warnings.warn('Maximum iterations in get_observable reached')
        sweeps *= 4
        print('Using '+str(sweeps)+' sweeps')
    return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

# Given a particular step and a starting temperature, uniformly place temperatures
def get_beta_set(distance, low, count, energy):
    betas = [low]
    for i in range(count-1):
        cost = lambda x: ((betas[-1] - x)*np.abs(energy(betas[-1]) - energy(x)) - distance)
        for i in range(5):
            # Bias in starting value to get the positive incremen
            next_temp = sp.optimize.root(cost, betas[-1]+np.random.uniform(0,2) , tol=1e-7)
            if next_temp['success']:
                break
            else:
                print(betas, next_temp['x'][0])
        assert(next_temp['success'])
        betas.append(next_temp['x'][0])
        assert(betas[-1] > betas[-2])
    return betas

# Selects a dB*dE step such that the final temperature is the one desired
def get_optimized_temps(disorder_avg, beta_min, beta_max, temp_count):
    print('Computing temp set...')
    # fit to disorder averaged E(Beta)
    energy = sp.interpolate.interp1d(disorder_avg['Beta'], disorder_avg['<E>'], kind='linear', bounds_error=False, fill_value='extrapolate')
    residual = lambda step: get_beta_set(step, beta_min, temp_count, energy)[-1] - beta_max
    init_step = -(disorder_avg['<E>'].max() - disorder_avg['<E>'].min())*(disorder_avg['Beta'].max() - disorder_avg['Beta'].min())
    step = sp.optimize.bisect(residual, init_step*1e-5, init_step)
    beta_set = get_beta_set(step, beta_min, temp_count, energy)
    return beta_set

# Disorder average <E>(Beta)
# Fit temperatures to make dEd1/T constant
# Optimize MC move count (NOT IMPLEMENTED)
# Optimize temperature count
# Get optimal TTS
def bench_tempering(instances, beta, temp_count, field_strength, profile, optimize_temp = True, restarts = 400):
    beta_min = np.min(beta)
    beta_max = np.max(beta)
    print('Computing observables...')
    beta_set = np.linspace(beta_min, beta_max, temp_count)
    disorder_avg = pd.concat(get_observable(instances, '<E>', beta_set, profile, field_strength = field_strength)).groupby(['Beta']).mean().reset_index()
    time_per_sweep = np.median(disorder_avg['Total_Walltime']/disorder_avg['Total_Sweeps'])
    if optimize_temp:
        beta_set = get_optimized_temps(disorder_avg, beta_min, beta_max, temp_count)
        print(beta_set)
    print('Benchmarking...')
    return get_opt_tts(instances, beta_set, profile, field_strength, restarts=restarts), time_per_sweep



