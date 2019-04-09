import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
from sg_bench.solve_base import solve_base

class replica_exchange_solve_base(solve_base):
    def __init__(self, config, borrowed_backend = None):
        super().__init__(config, borrowed_backend = borrowed_backend)

        self._replica_count = config['replica_count']
        self._obs_replica_count = config.get('obs_replica_count', self._replica_count)

        self._sweep_timeout = config['sweep_timeout']
        self._optimize_set = config.get('optimize_set', False)
        self._observable_sweeps = config.get('observable_sweeps', 4096)
        self._observable_timeout = config.get('observable_timeout', 3)
        self._thermalize_threshold = config.get('thermalize_threshold', 1e-3)
        
        self._detailed_log['replica_count'] = self._replica_count
        self._detailed_log['obs_replica_count'] = self._obs_replica_count

    # Get TTS given a field set and sweep count
    def _get_tts(self, instances, param_set, cost = np.median):
        results = []
        for i in range(len(instances)):
            instances[i]['target_energy'] = instances[i]['ground_energy'] * self._gse_target
        
        schedule = self._make_schedule(sweeps = self._sweep_timeout, param_set = param_set)
        instances = self._backend.run_instances(self._launcher_command, schedule, instances, self._restarts, statistics=False)

        p_s = []
        runtime_list = []
        p99_tts = []
        p_xchg = []
        for i in instances:
            if i['results'] is not None:
                avg_xchg_p = i['results'].groupby(self._var_set + ['Total_Sweeps'], as_index=False).mean()
                p_xchg.append((avg_xchg_p.loc[avg_xchg_p['Total_Sweeps'] == avg_xchg_p['Total_Sweeps'].max()].sort_values(self._var_set)['P_XCHG']).tolist())
                restart_group = i['results'].groupby('restart')
                min_energy = restart_group.min()['E_MIN']

                timeouts = restart_group.max()['Total_Sweeps'] < self._sweep_timeout
                success_prob = np.mean(timeouts)
                p_s.append(float(success_prob))
                runtime_list.append(np.where(timeouts, restart_group.max()['Total_Walltime'], float('inf')))
            
                if not np.isclose(success_prob, 1.0):
                    warnings.warn('TTS run timed out (sweeps). Success probability: '+str(success_prob))
                else:
                    runtimes = np.sort(restart_group['Total_Walltime'].max().reset_index()['Total_Walltime']).astype(float).tolist()
                    runtimes = np.insert(runtimes, 0, 0)

                    p99_tts.append(float(np.percentile(restart_group['Total_Sweeps'].max().reset_index()['Total_Sweeps'], 99)))
                    success = np.linspace(0., success_prob, len(runtimes))
            else:
                runtime_list.append([float('inf') for i in range(self._restarts)])
                warnings.warn('TTS run timed out (wallclock)')


        optimal_runtime, optimal_tts = self._optimize_runtime(tts)
        self._detailed_log['runtime'] = runtime_list
        self._detailed_log['p_xchg'] = np.mean(np.stack(p_xchg).astype(float), axis=0).tolist()
        return optimal_tts, p_s, p99_tts

    # Creates a function that will give the median TTS as a function of runtime
    def _time_to_solution(self, runtime_list):
        # Assemble p(R)
        tts = []
        for runtimes in runtime_list:
            prob = sp.interpolate.interp1d(runtimes, np.linspace(0,1,num=len(runtimes)), kind='linear', bounds_error=False, fill_value='extrapolate')
            clipped_prob = lambda x: np.clip(prob(x), 0.0, self._success_prob)
            instance_tts = lambda t: t * np.log1p(-self._success_prob)/np.log1p(-clipped_prob(t))
            tts.append(instance_tts)

        def smooth_function(f, points = 20, width = 1, support = 6):
            '''Smooth out a function using a convolution with a gaussian'''
            def convolved_function(x):
                eval_points = np.linspace(-width*support, width*support, points)
                weights = np.exp(-1/2 * (eval_points / width)**2) / (width*np.sqrt(2 * np.pi))
                return np.sum(np.vectorize(f)(eval_points+x) * weights)
            return convolved_function
        median_tts = lambda test_runtime: np.median([(inst_tts(test_runtime) if hasattr(inst_tts, '__call__') else inst_tts) for inst_tts in tts])
        return smooth_function(median_tts)

    # Given a list of runtimes, return the optimal median TTS
    def _optimize_runtime(self, runtime_list):
        optimized = sp.optimize.minimize(self._time_to_solution(runtime_list), [np.percentile(p99_tts, 50)], 
            method='Nelder-Mead', tol=1e-5, options={'maxiter':1000, 'adaptive':True})
        if optimized.success:
            optimal_runtime = optimized['x'][0]
            optimal_tts = median_tts(optimal_runtime)
        else:
            self._output(optimized)
            warnings.warn('Optimization for TTS failed.')
            optimal_runtime = None
            optimal_tts = None
        return optimal_runtime, optimal_tts

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
