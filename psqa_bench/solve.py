import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
from sg_bench import bondfile, backend
from psqa_bench import pa_propanelib

class solve:
    def __init__(self, config, borrowed_backend = None):
        self._success_prob = 0.99

        self._machine_readable = config['machine_readable']

        self._traj = self._get_param_set_values(config['traj'])
        self._beta = self._get_param_set_values(config['beta'])
        self._gamma = self._get_param_set_values(config['gamma'])
        self._lambda = self._get_param_set_values(config['lambda'])

        self._sweeps = config.get('sweeps')
        self._population = config.get('population')
        self._restarts = config.get('bench_restarts', 100)

        self._gse_target = config.get('gse_target', 1.00)
        
        self._detailed_log = {'beta':self._beta, 'gamma':self._gamma, 'lambda':self._lambda, 'traj':self._traj}

        self._slurm = config.get('slurm', None)
        self._backend = (borrowed_backend if borrowed_backend else 
            backend.get_backend('python3 -m psqa_bench.launch_restarts', slurmconf = self._slurm))

        self._wolff_sweeps = config.get('wolff_sweeps', 1)
        self._precool = config.get('precool', 0)

        self._detailed_log['gse_target'] = self._gse_target
        self._detailed_log['sweeps'] = self._sweeps
        self._detailed_log['population'] = self._population
        self._detailed_log['wolff_sweeps'] = self._wolff_sweeps
        self._detailed_log['precool'] = self._precool

    def _get_param_set_values(self, dictionary):
        param_set = {}
        param_set['points'] = dictionary['points']
        return param_set

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def get_backend(self):
        return self._backend

    def _make_schedule(self, sweeps = None, population = None, param_set = None, replica_count = None):
        #if not param_set:
        #    param_set = self._get_initial_set(replica_count)
        if not sweeps:
            sweeps = self._sweeps
        if not population:
            population = self._population
        if not param_set:
            relation = self._get_linear_relation()
            traj = np.linspace(0.0, 1.0, sweeps)
            param_set = {}
            param_set['gamma'] = relation['gamma'](traj)
            param_set['lambda'] = relation['lambda'](traj)
            param_set['beta'] = relation['beta'](traj)

        return pa_propanelib.make_schedule(
                population = population,
                wolff_sweeps = self._wolff_sweeps,
                precool=self._precool,
                param_set = param_set)

    # Get TTS given a field set and sweep count
    def _get_tts(self, instances, param_set, cost = np.median):
        results = []
        for i in range(len(instances)):
            instances[i]['target_energy'] = instances[i]['ground_energy'] * self._gse_target
        
        schedule = self._make_schedule(param_set = param_set)
        instances = self._backend.run_instances(schedule, instances, self._restarts, statistics=False)

        p_s = []
        tts = []
        for i in instances:
            min_energy = i['results'].groupby('restart').min()['E_MIN']
            success_prob = np.mean(np.logical_or(np.isclose(i['target_energy'], min_energy), i['target_energy'] > min_energy))
            if np.isclose(success_prob, 0.0):
                warnings.warn('Success probability is 0')
            run_time = i['results'].groupby('restart').max()['Total_Walltime'].mean()
            
            tts.append(run_time * np.log1p(-self._success_prob)/np.log1p(-np.clip(success_prob, 1e-12, self._success_prob)))
            p_s.append(success_prob)

        return tts, p_s

    # Check whether the results are thermalized based on residual from last bin
    def _check_thermalized(self, data, obs):
        for name, group in data.groupby(['Beta', 'Gamma', 'Lambda']):
            sorted_group = group.sort_values(['Samples'])
            residual = np.abs(sorted_group.iloc[-1][obs] - sorted_group.iloc[-2][obs])/np.mean(sorted_group.iloc[-2:][obs])
            if(residual > self._thermalize_threshold):
                self._output(str(obs) + ' not thermalized. Residual: '+str(residual))
                return False
        return True

    # Return observables with thermalization based on observable obs
    def _get_observable(self, instances, obs, param_set, replica_count):
        for i in range(self._observable_timeout):
            sweeps = self._observable_sweeps
            schedule = self._make_schedule(sweeps = sweeps, param_set = param_set, replica_count = replica_count)
            instances = self._backend.run_instances(schedule, instances, restarts = 1)
            # check equillibriation
            if np.all(np.vectorize(lambda i, obs: self._check_thermalized(i['results'], obs))(instances, obs)):
                break
            
            if i == self._observable_timeout-1:
                warnings.warn('Maximum iterations in get_observable reached')
            sweeps *= 4
            self._output('Using '+str(sweeps)+' sweeps')
        return [i['results'][i['results']['Samples']==i['results']['Samples'].max()] for i in instances]

    def _get_disorder_avg(self, instances, obs, param_set, replica_count = None):
        if not replica_count:
            replica_count = self._obs_replica_count
        return (pd.concat(self._get_observable(instances, '<E>', param_set, replica_count = replica_count))
            .groupby(['Beta', 'Gamma', 'Lambda']).apply(np.mean).drop(columns=['Beta', 'Gamma', 'Lambda']).reset_index())

    # Returns a linear relationship between gamma/beta
    def _get_linear_relation(self):
        relation = {}
        relation['beta'] = sp.interpolate.interp1d(
            self._traj['points'], self._beta['points'], 
            bounds_error=False, fill_value = (self._beta['points'][0], self._beta['points'][-1]))
        relation['gamma'] = sp.interpolate.interp1d(
            self._traj['points'], self._gamma['points'], 
            bounds_error=False, fill_value = (self._gamma['points'][0], self._gamma['points'][-1]))
        relation['lambda'] = sp.interpolate.interp1d(
            self._traj['points'], self._lambda['points'], 
            bounds_error=False, fill_value = (self._lambda['points'][0], self._lambda['points'][-1]))
        return relation

    # Get TTS given parameters
    def bench(self, instances):
        param_set = None
        
        self._output('Benchmarking...')
        tts, success_prob = self._get_tts(instances, param_set)
        self._detailed_log['tts'] = tts
        self._detailed_log['p_s'] = success_prob
        self._detailed_log['set'] = param_set
        return tts

    def get_full_data(self):
        return self._detailed_log


