import numpy as np
import pandas as pd
import warnings
from sg_bench import backend
from sg_bench.solve_base import solve_base

class sequential_solve_base(solve_base):
    def __init__(self, config, borrowed_backend = None):
        super().__init__(config, borrowed_backend = borrowed_backend)

    # Get TTS given a field set and sweep count
    def _get_tts(self, instances, param_set, cost = np.median):
        results = []
        for i in range(len(instances)):
            instances[i]['target_energy'] = instances[i]['ground_energy'] * self._gse_target
        
        schedule = self._make_schedule(param_set = param_set)
        instances = self._backend.run_instances(self._launcher_command, schedule, instances, self._restarts, statistics=False)

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
        for name, group in data.groupby(self._var_set):
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
            instances = self._backend.run_instances(self._launcher_command, schedule, instances, restarts = 1)
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
            .groupby(self._var_set).apply(np.mean).drop(columns=self._var_set).reset_index())

    # Get TTS given parameters
    def bench(self, instances):
        param_set = None
        
        self._output('Benchmarking...')
        tts, success_prob = self._get_tts(instances, param_set)
        self._detailed_log['tts'] = tts
        self._detailed_log['p_s'] = success_prob
        self._detailed_log['set'] = param_set
        return tts
