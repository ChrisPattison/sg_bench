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
        self._obs_replica_count = config.get('obs_replica_count', self._replica_count)

        self._driver = self._get_param_set_values(config['driver'])
        self._problem = self._get_param_set_values(config['problem'])
        self._beta = self._get_param_set_values(config['beta'])

        self._beta_distr = config.get('beta_power', 1.0)

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

        self._detailed_log = {'beta':self._beta, 'driver':self._driver, 'problem':self._problem}

        self._detailed_log['replica_count'] = self._replica_count
        self._detailed_log['obs_replica_count'] = self._obs_replica_count

        self._slurm = config.get('slurm', None)
        self._backend = backend.get_backend(self._slurm)

    def _get_param_set_values(self, dictionary):
        param_set = {}
        param_set['points'] = dictionary['points']
        param_set['set'] = dictionary.get('set', None)
        param_set['distr'] = dictionary.get('distr', 'linear')
        return param_set

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def _get_initial_set(self, count):
        param_set = {}
        param_set['beta'] = np.linspace((self._beta['points'][0])**(self._beta_distr), (self._beta['points'][-1])**(self._beta_distr), count)**(1./self._beta_distr)
        linear_relation = self._get_linear_relation()
        param_set['driver'] = linear_relation['driver'](param_set['beta'])
        param_set['problem'] = linear_relation['problem'](param_set['beta'])
        return param_set


    def _make_schedule(self, sweeps, param_set = None, replica_count = None):
        if not replica_count:
            replica_count = self._replica_count
            
        if not param_set:
            param_set = self._get_initial_set(replica_count)
        return pt_propanelib.make_schedule( \
                sweeps = sweeps, \
                param_set = param_set, \
                mc_sweeps = self._mc_sweeps)

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
        for i in instances:
            min_energy = i['results'].groupby('restart').min()['E_MIN']
            success_prob = np.mean(i['results'].groupby('restart').max()['Total_Sweeps'] < self._sweep_timeout)
            if not np.isclose(success_prob, 1.0):
                warnings.warn('TTS run timed out. Success probability: '+str(success_prob))

            runtimes = np.sort(np.apply_along_axis(np.asscalar, 1, i['results'].groupby('restart')['Total_Walltime'].unique().reset_index()['Total_Walltime'].tolist()))
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
        return pd.concat(self._get_observable(instances, '<E>', param_set, replica_count = replica_count)).groupby(['Beta', 'Gamma', 'Lambda']).apply(np.mean).drop(columns=['Beta', 'Gamma', 'Lambda']).reset_index()


    # Given a particular step and a starting field, uniformly place temperatures
    def _get_beta_set(self, distance, energy, min_beta, relation):
        temps = [min_beta]
        for i in range(self._replica_count-1):
            cost = np.vectorize(lambda x: ( 
                (temps[-1]*relation['driver'](temps[-1]) - x*relation['driver'](x)) 
                * (energy['driver'](temps[-1]) - energy['driver'](x)) 
                + (temps[-1]*relation['problem'](temps[-1]) - x*relation['problem'](x)) 
                * (energy['problem'](temps[-1]) - energy['problem'](x)) 
                - distance))

            next_value = sp.optimize.root(cost, temps[-1]+0.1)
            assert(next_value['success'])
            assert(next_value['x'] - temps[-1] > 0)
            temps.append(next_value['x'])
        return temps

    # energy['problem'] and energy['driver'] are the problem and driver energies as a function of beta
    def _interpolate_energy(self, field, energy):
        linear_energy = sp.interpolate.interp1d(field, energy, kind='linear', bounds_error=False, fill_value='extrapolate')
        cubic_energy = sp.interpolate.interp1d(field, energy, kind='cubic')
        return (lambda f, linear_energy=linear_energy, cubic_energy=cubic_energy, bounds=(np.min(field),np.max(field)):
            cubic_energy(f) if bounds[0] < f and f < bounds[1] else linear_energy(f))

    # Selects a dT*dE step such that the final field is the one desired
    # relation['driver'] and relation['problem'] are functions that return the driver and problem values as a function of beta
    def _get_optimized_param_set(self, disorder_avg, relation):
        self._output('Computing field set...')
        energy_norm = disorder_avg['<E>'].max()
        disorder_avg['norm_<E_P>'] = disorder_avg['<E_P>'] / energy_norm
        disorder_avg['norm_<E_D>'] = disorder_avg['<E_D>'] / energy_norm
        
        # fit to disorder averaged E(field)
        energy = {}
        energy['problem'] = self._interpolate_energy(disorder_avg['Beta'], disorder_avg['norm_<E_P>'])
        energy['driver'] = self._interpolate_energy(disorder_avg['Beta'], disorder_avg['norm_<E_D>'])
        residual = lambda step: (self._get_beta_set(step, energy, self._beta['points'][0], relation)[-1] - self._beta['points'][-1])

        sdiff = lambda x: x.iloc[-1] - x.iloc[0]
        step = sp.optimize.bisect(residual, -np.log(.001), -np.log(.99))
        beta_set = list(np.array(self._get_beta_set(step, energy, self._beta['points'][0], relation)))
        
        param_set = {}
        param_set['beta'] = beta_set
        param_set['driver'] = list(relation['driver'](beta_set))
        param_set['problem'] = list(relation['problem'](beta_set))
        self._detailed_log['optimized_param_set'] = param_set
        return param_set

    # Returns a linear relationship between driver/beta and problem/beta
    def _get_linear_relation(self):
        relation = {}
        relation['driver'] = sp.interpolate.interp1d(
            self._beta['points'], 
            self._driver['points'], 
            bounds_error=False, fill_value=(self._driver['points'][0], self._driver['points'][-1]))
        relation['problem'] = sp.interpolate.interp1d(
            self._beta['points'], 
            self._problem['points'], 
            bounds_error=False, fill_value=(self._problem['points'][0], self._problem['points'][-1]))
        return relation
    
    def observe(self, instances):
        self._output('Initial run...')
        param_set = None
        disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
        if self._optimize_set:
            param_set = self._get_optimized_param_set(disorder_avg, self._get_linear_relation())
            disorder_avg = self._get_disorder_avg(instances, '<E>', param_set, replica_count=self._replica_count)
        return disorder_avg
        

    # Disorder average <E>(field)
    # Fit fields to make dEdT constant
    # Optimize field count
    # Get optimal TTS
    def bench_tempering(self, instances):
        self._output('Computing observables...')
        param_set = None
        time_per_sweep = -1.0

        if self._optimize_set:
            disorder_avg = self._get_disorder_avg(instances, '<E>', param_set)
            param_set = self._get_optimized_param_set(disorder_avg, self._get_linear_relation())
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


