import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
import tempfile
import bondfile
import backend
import pt_propanelib

# Note: the notation in this class is that temp = B = 1/T
class solve:
    def __init__(self, config):
        self._machine_readable = config['machine_readable']

        self._beta_max = config['beta']['max']
        self._beta_min = config['beta']['min']
        self._temp_count = config['beta']['count']

        self._profile = config['profile']
        self._field_strength = config.get('field_strength', 1.0)

        self._restarts = config.get('bench_restarts', 100)
        self._sweep_timeout = config.get('sweep_timeout', 65536)
        self._optimize_temps = config.get('optimize_temps', False)
        self._observable_sweeps = config.get('observable_sweeps', 4096)
        self._observable_timeout = config.get('observable_timeout', 3)
        self._thermalize_threshold = config.get('thermalize_threshold', 1e-3)

        self._detailed_log = {'field_strength':self._field_strength, 'profile':self._profile}

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def _get_initial_temp_set(self):
        return np.linspace(self._beta_min, self._beta_max, self._temp_count)

    def _make_schedule(self, sweeps, beta_set = None):
        return pt_propanelib.make_schedule( \
                sweeps = sweeps, \
                beta_set = beta_set if beta_set else self._get_initial_temp_set(), \
                profile = self._profile, \
                field_strength = self._field_strength)

    # Get TTS given a temperature set and sweep count
    def _get_tts(self, instances, beta_set, cost = np.median):
        results = []
        
        schedule = self._make_schedule(sweeps = self._sweep_timeout, beta_set = beta_set)
        instances = backend.get_backend().run_instances(schedule, instances, self._restarts, statistics=False)

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

            optimized = sp.optimize.minimize(instance_tts, cost(unique_runtimes), method='TNC', bounds=[(unique_runtimes[1]+1e-4, unique_runtimes[-1]-1e-4)])
            if optimized.success:
                optimal_runtime = optimized['x'][0]
                optimal_tts = instance_tts(optimal_runtime)
                tts.append(optimal_tts)
            else:
                warnings.warn('Optimization for TTS failed.')
        
        return tts

    def _fit_opt_sweeps(self, trials):
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

    # Check whether the results are thermalized based on residual from last bin
    def _check_thermalized(self, data, obs):
        for name, group in data.groupby(['Beta']):
            sorted_group = group.sort_values(['Samples'])
            residual = np.abs(sorted_group.iloc[-1][obs] - sorted_group.iloc[-2][obs])/np.mean(sorted_group.iloc[-2:][obs])
            if(residual > self._thermalize_threshold):
                self._output(str(obs) + ' not thermalized. Residual: '+str(residual))
                return False
        return True

    # Return observables with thermalization based on observable obs
    def _get_observable(self, instances, obs, beta_set):
        solver = backend.get_backend()
        for i in range(self._observable_timeout):
            schedule = self._make_schedule(sweeps = self._observable_sweeps, beta_set = beta_set)
            instances = solver.run_instances(schedule, instances, restarts = 1)
            # check equillibriation
            if np.all(np.vectorize(lambda i, obs: self._check_thermalized(i['results'], obs))(instances, obs)):
                break
            
            if i == self._observable_timeout-1:
                warnings.warn('Maximum iterations in get_observable reached')
            sweeps *= 4
            self._output('Using '+str(sweeps)+' sweeps')
        return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

    def _get_disorder_avg(self, instances, obs, beta_set):
        return pd.concat(self._get_observable(instances, '<E>', beta_set)).groupby(['Beta']).mean().reset_index()


    # Given a particular step and a starting temperature, uniformly place temperatures
    def _get_temp_set(self, distance, energy):
        betas = [self._beta_min]
        for i in range(self._temp_count-1):
            cost = lambda x: ((betas[-1] - x)*np.abs(energy(betas[-1]) - energy(x)) - distance)
            for i in range(5):
                # Bias in starting value to get the positive incremen
                next_temp = sp.optimize.root(cost, betas[-1]+np.random.uniform(0,2) , tol=1e-7)
                if next_temp['success']:
                    break
            assert(next_temp['success'])
            betas.append(next_temp['x'][0])
            assert(betas[-1] > betas[-2])
        return betas

    # Selects a dB*dE step such that the final temperature is the one desired
    def _get_optimized_temps(self, disorder_avg):
        _output('Computing temp set...')
        # fit to disorder averaged E(Beta)
        energy = sp.interpolate.interp1d(disorder_avg['Beta'], disorder_avg['<E>'], kind='linear', bounds_error=False, fill_value='extrapolate')
        residual = lambda step: self._get_temp_set(step, energy)[-1] - self._beta_max
        init_step = -(disorder_avg['<E>'].max() - disorder_avg['<E>'].min())*(disorder_avg['Beta'].max() - disorder_avg['Beta'].min())
        step = sp.optimize.bisect(residual, init_step*1e-5, init_step)
        beta_set = _get_temp_set(step, energy)

        self._detailed_log['optimized_temps'] = beta_set
        return beta_set

    def observe(self, instances):
        self._output('Initial run...')
        beta_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', beta_set)
        if self._optimize_temps:
            beta_set = self._get_optimized_temps(disorder_avg)
            disorder_avg = self._get_disorder_avg(instances, '<E>', beta_set)
        return disorder_avg
        

    # Disorder average <E>(Beta)
    # Fit temperatures to make dEdB constant
    # Optimize temperature count
    # Get optimal TTS
    def bench_tempering(self, instances):
        self._output('Computing observables...')
        beta_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', beta_set)
        time_per_sweep = np.median(disorder_avg['Total_Walltime']/disorder_avg['Total_Sweeps'])
        if self._optimize_temps:
            beta_set = self._get_optimized_temps(disorder_avg)
            self._output(beta_set)

        self._output('Benchmarking...')
        tts = self._get_tts(instances, beta_set)
        self._detailed_log['time_per_sweep'] = time_per_sweep
        self._detailed_log['tts'] = tts
        return tts, time_per_sweep

    def get_full_data(self):
        return self._detailed_log


