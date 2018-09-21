import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
from sg_bench import backend, bondfile, replica_exchange_solve_base
from pt_bench import pt_propanelib

class solve(replica_exchange_solve_base.replica_exchange_solve_base):
    def __init__(self, config, borrowed_backend = None):
        self._var_set = ['Beta']
        self._launcher_command = 'python3 -m pt_bench.launch_restarts'
        super().__init__(config, borrowed_backend)

        self._beta = self._get_param_set_values(config['beta'])
        self._beta_distr = config.get('beta_power', 1.0)
        
        self._detailed_log = {'beta':self._beta }

    def _get_initial_set(self, count):
        param_set = {}
        param_set['beta'] = np.linspace((self._beta['points'][0])**(self._beta_distr), (self._beta['points'][-1])**(self._beta_distr), count)**(1./self._beta_distr)
        return param_set

    def _make_schedule(self, sweeps, param_set = None, replica_count = None):
        if not replica_count:
            replica_count = self._replica_count
            
        if not param_set:
            param_set = self._get_initial_set(replica_count)
        return pt_propanelib.make_schedule(
                sweeps = sweeps,
                param_set = param_set,
                hit_criteria = self._hit_criteria)

    # Given a particular step and a starting field, uniformly place temperatures
    def _get_beta_set(self, distance, energy, min_beta, relation):
        temps = [min_beta]
        for i in range(self._replica_count-1):
            cost = np.vectorize(lambda x: ((temps[-1] - x) * (energy(temps[-1]) - energy(x)) - distance))

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
    def _get_optimized_param_set(self, disorder_avg):
        self._output('Computing field set...')

        # fit to disorder averaged E(field)
        energy = self._interpolate_energy(disorder_avg['Beta'], disorder_avg['<E>'])
        residual = lambda step: (self._get_beta_set(step, energy, self._beta['points'][0])[-1] - self._beta['points'][-1])

        sdiff = lambda x: x.iloc[-1] - x.iloc[0]
        step = sp.optimize.bisect(residual, -np.log(.001), -np.log(.99))
        beta_set = list(np.array(self._get_beta_set(step, energy, self._beta['points'][0])))
        
        param_set = {}
        param_set['beta'] = beta_set
        self._detailed_log['optimized_param_set'] = param_set
        return param_set

