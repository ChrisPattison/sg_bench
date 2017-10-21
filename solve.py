import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
import tempfile
import bondfile
import pt_propanelib as propanelib
import localrun

def run_instances(schedule, instances, restarts = 400):
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

def fit_opt_sweeps(trials):
    trials = pd.DataFrame.from_records(trials)
    trials = trials.sort_values('sweeps').tail(4)
    trials['log_sweeps'] = np.log(trials['sweeps'])
    trials['log_tts'] = np.log(trials['tts'])
    fit = sp.optimize.least_squares(lambda x: ((x[0] * (trials['log_sweeps'] - x[1])**2 + x[2])-trials['log_tts']), [np.mean(trials['log_tts']), trials['log_sweeps'].iloc[-2], np.mean(trials['log_tts'])])['x']
    opt_sweeps = int(np.exp(fit[1]))
    if opt_sweeps > trials['sweeps'].max():
        warnings.warn('Optimal sweep count more than maximum sweep count tested. Got: '+str(opt_sweeps))
        assert(opt_sweeps < 2*trials['sweeps'].max())
    return opt_sweeps

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
        if np.isinf(trials[-2]['tts']):
            trials = [trials[-1]]
        else:
            # Upper bound on minimum TTS given by maximum of range
            if trials[-2]['tts'] < trials[-1]['tts']:
                break
        sweeps *= 2
	print(trials)
    if len(trials) <=2:
        warnings.warn('Minimum TTS found in less than 2 iterations')

    # fit to find optimal sweep count

    opt_sweeps = fit_opt_sweeps(trials)
    # Return TTS at optimal sweep count
    return get_tts(instances, temp_set, opt_sweeps)

# Check whether the results are thermalized based on residual from last bin
def check_thermalized(data, obs, threshold=.001):
    for name, group in data.groupby(['Gamma']):
        sorted_group = group.sort_values(['Samples'])
        residual = np.abs(sorted_group.iloc[-1][obs] - sorted_group.iloc[-2][obs])/np.mean(sorted_group.iloc[-2:][obs])
        if(residual > threshold):
            print(str(obs) + ' not thermalized. Residual: '+str(residual))
            return False
    return True

# Return observables with thermalization based on observable obs
def get_observable(instances, obs, temp_set, max_iterations = 3):
    sweeps = 4096
    for i in range(max_iterations):
        schedule = propanelib.make_schedule(sweeps, temp_set, instances[0]['bondscale'])
        instances = run_instances(schedule, instances, restarts = 1)
        # check equillibriation
        if np.all(np.vectorize(lambda i, obs: check_thermalized(i['results'], obs))(instances, obs)):
            break
        
        if i == max_iterations-1:
            warnings.warn('Maximum iterations in get_observable reached')
        sweeps *= 4
        print('Using '+str(sweeps)+' sweeps')
    return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

# Disorder average <E>(Gamma)
# Fit temperatures to make dEd1/T constant
# Optimize MC move count (NOT IMPLEMENTED)
# Optimize temperature count
# Get optimal TTS
def bench_tempering(instances):
    print('Getting temperature set...')
    temp_set = np.linspace(3, 0.1, 32)*instances[0]['bondscale']
    # fit to disorder averaged E(Gamma)
    disorder_avg = pd.concat(get_observable(instances, '<E>', temp_set)).groupby(['Gamma']).mean().reset_index()
    energy = sp.interpolate.interp1d(disorder_avg['Gamma'], disorder_avg['<E>'], kind='quadratic')
    fixed = [temp_set[0], temp_set[-1]]
    temp_set = temp_set[1:-1]
    # List comprehensions to do binary operations
    # new temperature set has constant dE*dT
    residual = lambda x: np.linalg.norm([cost - np.mean(cost) for cost in [np.ediff1d(sorted) * np.ediff1d(energy(sorted)) for sorted in [np.sort(np.concatenate((x, fixed)))]]][0])
    temperatures = sp.optimize.minimize(residual, temp_set,  bounds=[(fixed[-1]+1e-6, fixed[0]-1e-6) for t in temp_set])
    temperatures = np.sort(np.concatenate((temperatures['x'], temp_set)))
    print(temperatures)
    print('Benchmarking...')
    return get_opt_tts(instances, temperatures)


