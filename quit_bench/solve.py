import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
from sg_bench import backend, bondfile, replica_exchange_solve_base
from quit_bench import quit_propanelib

class solve(replica_exchange_solve_base):
    def __init__(self, config, borrowed_backend = None):
        
        self._driver = self._get_param_set_values(config['driver'])
        self._problem = self._get_param_set_values(config['problem'])
        self._beta = self._get_param_set_values(config['beta'])

        self._beta_distr = config.get('beta_power', 1.0)

        self._mc_sweeps = config.get('mc_sweeps', 10)
        
        if self._optimize_set and (
            self._driver['set'] 
            or self._problem['set'] 
            or self._beta['set'] ):
            warnings.warn('Optimize parameter set true but parameter set provided')

        self._detailed_log = {'beta':self._beta, 'driver':self._driver, 'problem':self._problem}
        self._var_set = ['Beta', 'Gamma', 'Lambda']
        super().__init__(config, borrowed_backend)

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
        return quit_propanelib.make_schedule(
                sweeps = sweeps,
                param_set = param_set,
                mc_sweeps = self._mc_sweeps,
                hit_criteria = self._hit_criteria)

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

