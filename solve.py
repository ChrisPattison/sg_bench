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

def run_instances(schedule, instances, restarts = 100, statistics=True):
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
                ground_energy = None
                if not statistics:
                    ground_energy = i['ground_energy']
                i['results'] = localrun.get_data(sch_file.name, bonds_file.name, restarts = restarts, ground_energy = ground_energy)
                if not statistics:
                    i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)
    return instances

# Get TTS given a temperature set and sweep count
def get_tts(instances, field_set, sweeps, restarts = 100):
    results = []
    
    # make schedule
    schedule = propanelib.make_schedule(sweeps, field_set, instances[0]['bondscale'])
    instances = run_instances(schedule, instances, restarts, statistics=False)
    
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

        prob = sp.interpolate.interp1d(unique_runtimes, unique_success, kind='quadratic', bounds_error=True)
        clipped_prob = lambda x: np.clip(prob(x), 0.0, 1.0)
        instance_tts = lambda t: t * np.log(1.-.99)/np.log(1.-clipped_prob(t))

        optimized = sp.optimize.minimize(instance_tts, unique_runtimes[1], method='TNC', bounds=[(unique_runtimes[1]+1e-4, unique_runtimes[-5])])
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
def get_opt_tts(instances, field_set, init_sweeps=128, cost=np.median):
    return get_tts(instances, field_set, 65536, restarts=400)

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
def get_observable(instances, obs, field_set, max_iterations = 3):
    sweeps = 4096
    for i in range(max_iterations):
        schedule = propanelib.make_schedule(sweeps, field_set, instances[0]['bondscale'])
        instances = run_instances(schedule, instances, restarts = 1)
        # check equillibriation
        if np.all(np.vectorize(lambda i, obs: check_thermalized(i['results'], obs))(instances, obs)):
            break
        
        if i == max_iterations-1:
            warnings.warn('Maximum iterations in get_observable reached')
        sweeps *= 4
        print('Using '+str(sweeps)+' sweeps')
    return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

def get_field_set(distance, low, count, energy):
    fields = [low]
    for i in range(count-1):
        cost = lambda x: ((fields[-1] - x)*(energy(fields[-1]) - energy(x)) - distance)**2
        # Bias in starting value to get the positive increment
        next_field = sp.optimize.minimize(cost, [fields[-1]+0.1], options={'gtol':1e-10})
        fields.append(next_field['x'][0])
    assert(np.all((np.ediff1d(fields) * np.ediff1d(energy(fields)) - distance) < (distance * 1e-5))) # residual check
    return fields

# Disorder average <E>(Gamma)
# Fit temperatures to make dEd1/T constant
# Optimize MC move count (NOT IMPLEMENTED)
# Optimize temperature count
# Get optimal TTS
def bench_tempering(instances, field = (3, 0.1), field_count = 32, optimize_fields = True):
    print('Computing observables...')
    field_set = np.linspace(field[0], field[1], field_count)*instances[0]['bondscale']
    # fit to disorder averaged E(Gamma)
    disorder_avg = pd.concat(get_observable(instances, '<E>', field_set)).groupby(['Gamma']).mean().reset_index()
    time_per_sweep = np.median(disorder_avg['Total_Walltime']/disorder_avg['Total_Sweeps'])
    if optimize_fields:
        energy = sp.interpolate.interp1d(disorder_avg['Gamma'], disorder_avg['<E>'], kind='quadratic', bounds_error=False, fill_value='extrapolate')

        print('Computing field set...')
        residual = lambda k: (np.max(get_field_set(k[0], field_set[-1], field_count, energy)) - field_set[0])**2
        temp_seperation = sp.optimize.minimize(residual, [(np.max(field_set) - np.min(field_set))/field_count], method='CG', options={'gtol':1e-4, 'eps':1e-6})['x'][0]
        field_set = get_field_set(temp_seperation, field_set[-1], field_count, energy)
        print(field_set)
    print('Benchmarking...')
    return get_opt_tts(instances, field_set), time_per_sweep



