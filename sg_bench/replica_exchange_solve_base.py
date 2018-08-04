import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
from sg_bench import backend

class replica_exchange_solve_base:
    def __init__(self, config, borrowed_backend = None):
        self._success_prob = 0.99

        self._machine_readable = config['machine_readable']

        self._replica_count = config['replica_count']
        self._obs_replica_count = config.get('obs_replica_count', self._replica_count)

        self._hit_criteria = config.get('hit_criteria', 1e-12)

        self._restarts = config.get('bench_restarts', 100)
        self._sweep_timeout = config.get('sweep_timeout', 65536)
        self._optimize_set = config.get('optimize_set', False)
        self._observable_sweeps = config.get('observable_sweeps', 4096)
        self._observable_timeout = config.get('observable_timeout', 3)
        self._thermalize_threshold = config.get('thermalize_threshold', 1e-3)

        self._gse_target = config.get('gse_target', 1.00)
        
        self._detailed_log['replica_count'] = self._replica_count
        self._detailed_log['obs_replica_count'] = self._obs_replica_count

        self._slurm = config.get('slurm', None)
        self._backend = (borrowed_backend if borrowed_backend else 
            backend.get_backend(self._launcher_command, slurmconf = self._slurm))

    def _get_param_set_values(self, dictionary):
        param_set = {}
        param_set['points'] = dictionary['points']
        param_set['set'] = dictionary.get('set', None)
        param_set['distr'] = dictionary.get('distr', 'linear')
        return param_set

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    # Get TTS given a field set and sweep count
    def _get_tts(self, instances, param_set, cost = np.median):
        results = []
        for i in range(len(instances)):
            instances[i]['target_energy'] = instances[i]['ground_energy'] * self._gse_target
        
        schedule = self._make_schedule(sweeps = self._sweep_timeout, param_set = param_set)
        instances = self._backend.run_instances(schedule, instances, self._restarts, statistics=False)

        p_s = []
        tts = []
        p99_tts = []
        p_xchg = []
        for i in instances:
            avg_xchg_p = i['results'].groupby(self._var_set + ['Total_Sweeps'], as_index=False).mean()
            p_xchg.append(list(avg_xchg_p.loc[avg_xchg_p['Total_Sweeps'] == avg_xchg_p['Total_Sweeps'].max()].sort_values(self._var_set)['P_XCHG']))
            min_energy = i['results'].groupby('restart').min()['E_MIN']
            success_prob = np.mean(i['results'].groupby('restart').max()['Total_Sweeps'] < self._sweep_timeout)

            if not np.isclose(success_prob, 1.0):
                warnings.warn('TTS run timed out. Success probability: '+str(success_prob))
            runtimes = np.sort(i['results'].groupby('restart')['Total_Walltime'].max().reset_index()['Total_Walltime'].tolist()).astype(float)
            p99_tts.append(np.percentile(i['results'].groupby('restart')['Total_Sweeps'].max().reset_index()['Total_Sweeps'], 99))
            runtimes = np.insert(runtimes, 0, 0)
            success = np.linspace(0., success_prob, len(runtimes))
            if success_prob > 0:
                prob = sp.interpolate.interp1d(runtimes, success, kind='linear', bounds_error=False, fill_value='extrapolate')
                clipped_prob = lambda x: np.clip(prob(x), 0.0, min(self._success_prob, success_prob))
                instance_tts = lambda t: t * np.log1p(-self._success_prob)/np.log1p(-clipped_prob(t))

                # CG methods fail due to cusp in TTS
                optimized = sp.optimize.minimize(instance_tts, [np.percentile(runtimes, 50)], 
                    method='Nelder-Mead', tol=1e-5, options={'maxiter':1000, 'adaptive':True})
                if optimized.success:
                    optimal_runtime = optimized['x'][0]
                    optimal_tts = instance_tts(optimal_runtime)
                    tts.append(optimal_tts)
                else:
                    self._output(optimized)
                    warnings.warn('Optimization for TTS failed.')
            else:
                tts.append(float('inf'))

            p_s.append(success_prob)

        self._detailed_log['p_xchg'] = list(np.mean(np.stack(p_xchg).astype(float), axis=0))
        return tts, p_s, p99_tts

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
            .groupby(self._var_set).apply(np.mean).drop(columns=self._var_set).reset_index())
    
    def observe(self, instances):
        self._output('Initial run...')
        param_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
        if self._optimize_set:
            param_set = self._get_optimized_param_set(disorder_avg)
            disorder_avg = self._get_disorder_avg(instances, '<E>', param_set, replica_count=self._replica_count)
        return disorder_avg
        

    # Disorder average <E>(field)
    # Fit fields to make dEdT constant
    # Optimize field count
    # Get optimal TTS
    def bench(self, instances):
        self._output('Computing observables...')
        param_set = None
        time_per_sweep = -1.0

        if self._optimize_set:
            disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
            param_set = self._get_optimized_param_set(disorder_avg)
            self._output(param_set)

        self._output('Benchmarking...')
        tts, success_prob, p99_tts = self._get_tts(instances, param_set)
        self._detailed_log['time_per_sweep'] = time_per_sweep
        self._detailed_log['tts'] = tts
        self._detailed_log['p99_tts'] = p99_tts
        self._detailed_log['p_s'] = success_prob
        self._detailed_log['set'] = param_set
        return tts, time_per_sweep * float(self._replica_count)/self._obs_replica_count

    def get_full_data(self):
        return self._detailed_log