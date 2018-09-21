import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
from sg_bench import bondfile, backend, sequential_solve_base
from psqa_bench import pa_propanelib

class solve(sequential_solve_base.sequential_solve_base):
    def __init__(self, config, borrowed_backend = None):
        self._var_set = ['Beta', 'Gamma', 'Lambda']
        self._launcher_command = 'python3 -m psqa_bench.launch_restarts'
        super().__init__(config, borrowed_backend = borrowed_backend)

        self._traj = self._get_param_set_values(config['traj'])
        self._beta = self._get_param_set_values(config['beta'])
        self._gamma = self._get_param_set_values(config['gamma'])
        self._lambda = self._get_param_set_values(config['lambda'])

        self._sweeps = config.get('sweeps')
        self._population = config.get('population')
        
        self._detailed_log = {'beta':self._beta, 'gamma':self._gamma, 'lambda':self._lambda, 'traj':self._traj}

        self._wolff_sweeps = config.get('wolff_sweeps', 1)
        self._precool = config.get('precool', 0)

        self._detailed_log['sweeps'] = self._sweeps
        self._detailed_log['population'] = self._population
        self._detailed_log['wolff_sweeps'] = self._wolff_sweeps
        self._detailed_log['precool'] = self._precool

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

