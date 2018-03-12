import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
import bondfile
import backend
import pt_propanelib

class solve:
    def __init__(self, config):
        self._success_prob = 0.99

        self._machine_readable = config['machine_readable']

        self._replica_count = config['replica_count']

        self._driver = self._get_param_set_values(config['driver'])
        self._problem = self._get_param_set_values(config['problem'])
        self._beta = self._get_param_set_values(config['beta'])

        self._mc_sweeps = config.get('mc_sweeps', 10)

        self._restarts = config.get('bench_restarts', 100)
        self._sweep_timeout = config.get('sweep_timeout', 65536)
        self._optimize_set = config.get('optimize_set', False)
        self._observable_sweeps = config.get('observable_sweeps', 4096)
        self._observable_timeout = config.get('observable_timeout', 3)
        self._thermalize_threshold = config.get('thermalize_threshold', 1e-3)

        self._gse_target = config.get('gse_target', 1.00)
        
        if self._optimize_set and (
            self._driver['set'] 
            or self._problem['set'] 
            or self._beta['set'] ):
            warnings.warn('Optimize parameter set true but parameter set provided')

        self._detailed_log = {'beta':self._beta}

    def _get_param_set_values(self, dictionary):
        param_set = {}
        param_set['max'] = dictionary['max']
        param_set['min'] = dictionary['min']
        param_set['set'] = dictionary.get('set', None)
        param_set['distr'] = dictionary.get('distr', 'linear')
        return param_set

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def _get_initial_set(self, set_params):
        if set_params['distr'] == 'linear':
            return np.linspace(set_params['min'], set_params['max'], self._replica_count)
        else:
            return np.exp(np.linspace(np.log(set_params['min']), np.log(set_params['max']), self._replica_count))

    def _make_schedule(self, sweeps, param_set = None):
        if not param_set:
            param_set = {}
            param_set['driver'] = self._driver['set'] if self._driver['set'] else self._get_initial_set(self._driver)
            param_set['problem'] = self._problem['set'] if self._problem['set'] else self._get_initial_set(self._problem)
            param_set['beta'] = self._beta['set'] if self._beta['set'] else self._get_initial_set(self._beta)
        return pt_propanelib.make_schedule( \
                sweeps = sweeps, \
                param_set = param_set, \
                mc_sweeps = self._mc_sweeps)

    # Get TTS given a field set and sweep count
    def _get_tts(self, instances, param_set, cost = np.median):
        results = []
        for i in range(len(instances)):
            instances[i]['target_energy'] = instances[i]['ground_energy'] * self._gse_target
        
        schedule = self._make_schedule(sweeps = self._sweep_timeout, param_set = param_set, mc_sweeps = self._mc_sweeps)
        instances = backend.get_backend().run_instances(schedule, instances, self._restarts, statistics=False)

        p_s = []
        tts = []
        p99_tts = []
        for i in instances:
            min_energy = i['results'].groupby('restart').min()['E_MIN']
            success_prob = np.mean(np.logical_or(np.isclose(i['target_energy'], min_energy), i['target_energy'] > min_energy))
            if not np.isclose(success_prob, 1.0):
                warnings.warn('TTS run timed out. Success probability: '+str(success_prob))

            runtimes = np.sort(np.apply_along_axis(np.asscalar, 1, i['results'].groupby('restart')['Total_Sweeps'].unique().reset_index()['Total_Sweeps'].tolist()))
            p99_tts.append(np.percentile(runtimes, 99))
            runtimes = np.insert(runtimes, 0, 0)
            success = np.linspace(0., 1, len(runtimes))
            unique_runtimes, unique_indices = np.unique(runtimes, return_index=True)
            unique_success = [success[i] for i in unique_indices]
            # Last values are timeout and not successes
            if success_prob != 1.0:
                unique_success = unique_success[:-1]
                unique_runtimes = unique_runtimes[:-1]
            if len(unique_success) < 2:
                tts.append(np.inf)
                p_s.append(success_prob)
                continue

            prob = sp.interpolate.interp1d(unique_runtimes, unique_success, kind='linear', bounds_error=False, fill_value='extrapolate')
            max_runtime = np.max(runtimes)
            clipped_prob = lambda x: np.clip(prob(x), 0.0, min(self._success_prob, success_prob))
            instance_tts = lambda t: t * np.log(1.-self._success_prob)/np.log(1.-clipped_prob(t))

            # CG methods fail due to cusp in TTS
            optimized = sp.optimize.minimize(instance_tts, np.percentile(runtimes, 99), method='Nelder-Mead')
            if optimized.success:
                optimal_runtime = optimized['x'][0]
                optimal_tts = instance_tts(optimal_runtime)
                tts.append(optimal_tts)
                p_s.append(success_prob)
            else:
                self._output(optimized)
                warnings.warn('Optimization for TTS failed.')

        return tts, p_s, p99_tts

    # Check whether the results are thermalized based on residual from last bin
    def _check_thermalized(self, data, obs):
        for name, group in data.groupby(['Gamma']):
            sorted_group = group.sort_values(['Samples'])
            residual = np.abs(sorted_group.iloc[-1][obs] - sorted_group.iloc[-2][obs])/np.mean(sorted_group.iloc[-2:][obs])
            if(residual > self._thermalize_threshold):
                self._output(str(obs) + ' not thermalized. Residual: '+str(residual))
                return False
        return True

    # Return observables with thermalization based on observable obs
    def _get_observable(self, instances, obs, param_set):
        solver = backend.get_backend()
        for i in range(self._observable_timeout):
            sweeps = self._observable_sweeps
            schedule = self._make_schedule(sweeps = sweeps, param_set = param_set)
            instances = solver.run_instances(schedule, instances, restarts = 1)
            # check equillibriation
            if np.all(np.vectorize(lambda i, obs: self._check_thermalized(i['results'], obs))(instances, obs)):
                break
            
            if i == self._observable_timeout-1:
                warnings.warn('Maximum iterations in get_observable reached')
            sweeps *= 4
            self._output('Using '+str(sweeps)+' sweeps')
        return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

    def _get_disorder_avg(self, instances, obs, param_set):
        return pd.concat(self._get_observable(instances, '<E>', param_set)).groupby('Gamma').apply(np.mean).drop(columns=['Gamma']).reset_index()


    # Given a particular step and a starting field, uniformly place fields
    def _get_field_set(self, distance, energy, min_field):
        fields = [min_field]
        for i in range(self._replica_count-1):
            cost = lambda x: ((fields[-1] - x)*np.abs(energy(fields[-1]) - energy(x)) - distance)
            for i in range(5):
                # Bias in starting value to get the positive incremen
                next_field = sp.optimize.root(cost, fields[-1]+np.random.uniform(0,2) , tol=1e-7)
                if next_field['success']:
                    break
            assert(next_field['success'])
            fields.append(next_field['x'][0])
            assert(fields[-1] > fields[-2])
        return fields

    def _interpolate_energy(self, field, energy):
        linear_energy = sp.interpolate.interp1d(field, energy, kind='linear', bounds_error=False, fill_value='extrapolate')
        cubic_energy = sp.interpolate.interp1d(field, energy, kind='cubic')
        return (lambda f, linear_energy=linear_energy, cubic_energy=cubic_energy, bounds=(np.min(field),np.max(field)):
            cubic_energy(f) if bounds[0] < f and f < bounds[1] else linear_energy(f))

    # Selects a dT*dE step such that the final field is the one desired
    # Broken
    def _get_optimized_fields(self, disorder_avg):
        self._output('Computing field set...')
        field_norm = disorder_avg['Gamma'].max()
        energy_norm = disorder_avg['<E>'].max()
        disorder_avg['norm_Gamma'] = disorder_avg['Gamma'] / field_norm
        disorder_avg['norm_<E>'] = disorder_avg['<E>'] / energy_norm
        # fit to disorder averaged E(field)
        energy = self._interpolate_energy(disorder_avg['norm_Gamma'], disorder_avg['norm_<E>'])
        residual = lambda step: (self._get_field_set(step, energy, self._driver_min/field_norm)[-1] - self._driver_max/field_norm)
        init_step = -(disorder_avg['norm_<E>'].max() - disorder_avg['norm_<E>'].min())*(disorder_avg['norm_Gamma'].max() - disorder_avg['norm_Gamma'].min())
        step = sp.optimize.bisect(residual, init_step*1e-5, init_step)
        field_set = list(np.array(self._get_field_set(step, energy, self._driver_min/field_norm)) * field_norm)

        self._detailed_log['optimized_fields'] = field_set
        return field_set

    def observe(self, instances):
        self._output('Initial run...')
        param_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
        if self._optimize_set:
            param_set = self._get_optimized_fields(disorder_avg)
            disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
        return disorder_avg
        

    # Disorder average <E>(field)
    # Fit fields to make dEdT constant
    # Optimize field count
    # Get optimal TTS
    def bench_tempering(self, instances):
        self._output('Computing observables...')
        param_set = {}
        disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
        time_per_sweep = np.median(disorder_avg['Total_Walltime']/disorder_avg['Total_Sweeps'])

        param_set['driver'] = self._driver['set']
        param_set['problem'] = self._problem['set']
        param_set['beta'] = self._beta['set']

        if self._optimize_set:
            param_set = self._get_optimized_params(disorder_avg)
            self._output(param_set)

        self._output('Benchmarking...')
        tts, success_prob, p99_tts = self._get_tts(instances, param_set)
        self._detailed_log['time_per_sweep'] = time_per_sweep
        self._detailed_log['tts'] = tts
        self._detailed_log['p99_tts'] = p99_tts
        self._detailed_log['p_s'] = success_prob
        self._detailed_log['set'] = param_set
        return tts, time_per_sweep

    def get_full_data(self):
        return self._detailed_log


