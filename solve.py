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
        self._machine_readable = config['machine_readable']

        self._field_max = config['field']['max']
        self._field_min = config['field']['min']
        self._field_count = config['field']['count']

        self._beta = config.get('beta', 10.0)
        self._mc_sweeps = config.get('mc_sweeps', 10)

        self._restarts = config.get('bench_restarts', 100)
        self._sweep_timeout = config.get('sweep_timeout', 65536)
        self._optimize_fields = config.get('optimize_fields', False)
        self._observable_sweeps = config.get('observable_sweeps', 4096)
        self._observable_timeout = config.get('observable_timeout', 3)
        self._thermalize_threshold = config.get('thermalize_threshold', 1e-3)

        self._detailed_log = {'beta':self._beta}

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def _get_initial_field_set(self):
        return np.linspace(self._field_min, self._field_max, self._field_count)

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
        return pd.concat(self._get_observable(instances, '<E>', field_set)).groupby(['Gamma']).mean().reset_index()


    # Given a particular step and a starting field, uniformly place fields
    def _get_field_set(self, distance, energy):
        fields = [self._field_min]
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
        # fit to disorder averaged E(field)
        energy = sp.interpolate.interp1d(disorder_avg['Gamma'], disorder_avg['<E>'], kind='linear', bounds_error=False, fill_value='extrapolate')
        residual = lambda step: self._get_field_set(step, energy)[-1] - self._field_max
        init_step = -(disorder_avg['<E>'].max() - disorder_avg['<E>'].min())*(disorder_avg['Gamma'].max() - disorder_avg['Gamma'].min())
        step = sp.optimize.bisect(residual, init_step*1e-5, init_step)
        field_set = self._get_field_set(step, energy)

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
        if self._optimize_fields:
            field_set = self._get_optimized_fields(disorder_avg)
            self._output(field_set)

        self._output('Benchmarking...')
        tts = self._get_tts(instances, field_set)
        self._detailed_log['time_per_sweep'] = time_per_sweep
        self._detailed_log['tts'] = tts
        return tts, time_per_sweep

    def get_full_data(self):
        return self._detailed_log


