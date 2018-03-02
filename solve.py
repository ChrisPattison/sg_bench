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

        self._field_max = config['field']['max']
        self._field_min = config['field']['min']
        self._field_count = config['field']['count']

        self._field_set = config['field'].get('set', None)

        self._beta = config.get('beta', 10.0)
        self._mc_sweeps = config.get('mc_sweeps', 10)

        self._restarts = config.get('bench_restarts', 100)
        self._sweep_timeout = config.get('sweep_timeout', 65536)
        self._optimize_fields = config.get('optimize_fields', False)
        self._observable_sweeps = config.get('observable_sweeps', 4096)
        self._observable_timeout = config.get('observable_timeout', 3)
        self._thermalize_threshold = config.get('thermalize_threshold', 1e-3)

        if self._optimize_fields and self._field_set:
            warnings.warn('Optimize fields true but field set provided')

        self._detailed_log = {'beta':self._beta}

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def _get_initial_field_set(self):
        return np.exp(np.linspace(np.log(self._field_min), np.log(self._field_max), self._field_count))

    def _make_schedule(self, sweeps, field_set = None):
        return pt_propanelib.make_schedule( \
                sweeps = sweeps, \
                field_set = field_set if field_set else self._get_initial_field_set(), \
                beta = self._beta, \
                mc_sweeps = self._mc_sweeps)

    # Get TTS given a field set and sweep count
    def _get_tts(self, instances, field_set, cost = np.median):
        results = []
        
        schedule = self._make_schedule(sweeps = self._sweep_timeout, field_set = field_set)
        instances = backend.get_backend().run_instances(schedule, instances, self._restarts, statistics=False)

        p_s = []
        tts = []
        for i in instances:
            success_prob = np.mean(np.isclose(i['ground_energy'], i['results'].groupby('restart').min()['E_MIN']))
            if not np.isclose(success_prob, 1.0):
                warnings.warn('TTS run timed out. Success probability: '+str(success_prob))

            runtimes = np.sort(np.apply_along_axis(np.asscalar, 1, i['results'].groupby('restart')['Total_Sweeps'].unique().reset_index()['Total_Sweeps'].tolist()))
            runtimes = np.insert(runtimes, 0, 0)
            success = np.linspace(0., 1, len(runtimes))
            unique_runtimes, unique_indices = np.unique(runtimes, return_index=True)
            unique_success = [success[i] for i in unique_indices]
            # Last values are timeout and not successes
            if success_prob != 1.0:
                unique_success = unique_success[:-1]
                unique_runtimes = unique_runtimes[:-1]

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

        return tts, p_s

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
    def _get_observable(self, instances, obs, field_set):
        solver = backend.get_backend()
        for i in range(self._observable_timeout):
            sweeps = self._observable_sweeps
            schedule = self._make_schedule(sweeps = sweeps, field_set = field_set)
            instances = solver.run_instances(schedule, instances, restarts = 1)
            # check equillibriation
            if np.all(np.vectorize(lambda i, obs: self._check_thermalized(i['results'], obs))(instances, obs)):
                break
            
            if i == self._observable_timeout-1:
                warnings.warn('Maximum iterations in get_observable reached')
            sweeps *= 4
            self._output('Using '+str(sweeps)+' sweeps')
        return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

    def _get_disorder_avg(self, instances, obs, field_set):
        return pd.concat(self._get_observable(instances, '<E>', field_set)).groupby('Gamma').apply(np.mean).drop(columns=['Gamma']).reset_index()
        # return pd.concat(self._get_observable(instances, '<E>', field_set)).groupby(['Gamma']).mean().reset_index()


    # Given a particular step and a starting field, uniformly place fields
    def _get_field_set(self, distance, energy, min_field):
        fields = [min_field]
        for i in range(self._field_count-1):
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

    # Selects a dT*dE step such that the final field is the one desired
    def _get_optimized_fields(self, disorder_avg):
        self._output('Computing field set...')
        field_norm = disorder_avg['Gamma'].max()
        energy_norm = disorder_avg['<E>'].max()
        disorder_avg['norm_Gamma'] = disorder_avg['Gamma'] / field_norm
        disorder_avg['norm_<E>'] = disorder_avg['<E>'] / energy_norm
        # fit to disorder averaged E(field)
        energy = sp.interpolate.interp1d(disorder_avg['norm_Gamma'], disorder_avg['norm_<E>'], kind='linear', bounds_error=False, fill_value='extrapolate')
        residual = lambda step: (self._get_field_set(step, energy, self._field_min/field_norm)[-1] - self._field_max/field_norm)
        init_step = -(disorder_avg['norm_<E>'].max() - disorder_avg['norm_<E>'].min())*(disorder_avg['norm_Gamma'].max() - disorder_avg['norm_Gamma'].min())
        step = sp.optimize.bisect(residual, init_step*1e-5, init_step)
        field_set = list(np.array(self._get_field_set(step, energy, self._field_min/field_norm)) * field_norm)

        self._detailed_log['optimized_fields'] = field_set
        return field_set

    def observe(self, instances):
        self._output('Initial run...')
        field_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', field_set)
        if self._optimize_fields:
            field_set = self._get_optimized_fields(disorder_avg)
            disorder_avg = self._get_disorder_avg(instances, '<E>', field_set)
        return disorder_avg
        

    # Disorder average <E>(field)
    # Fit fields to make dEdT constant
    # Optimize field count
    # Get optimal TTS
    def bench_tempering(self, instances):
        self._output('Computing observables...')
        field_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', field_set)
        time_per_sweep = np.median(disorder_avg['Total_Walltime']/disorder_avg['Total_Sweeps'])
        if self._field_set:
            field_set = self._field_set
        elif self._optimize_fields:
            field_set = self._get_optimized_fields(disorder_avg)
            self._output(field_set)

        self._output('Benchmarking...')
        tts, success_prob = self._get_tts(instances, field_set)
        self._detailed_log['time_per_sweep'] = time_per_sweep
        self._detailed_log['tts'] = tts
        self._detailed_log['p_s'] = success_prob
        self._detailed_log['field_set'] = self._field_set
        return tts, time_per_sweep

    def get_full_data(self):
        return self._detailed_log


