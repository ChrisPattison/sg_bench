import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.optimize
import pandas as pd
import warnings
import json
from sg_bench import bondfile, backend, sequential_solve_base

class solve(sequential_solve_base.sequential_solve_base):
    def __init__(self, config, borrowed_backend = None):
        self._var_set = ['Beta']
        self._launcher_command = 'python3 -m siman_bench.launch_restarts'
        super().__init__(config, borrowed_backend = borrowed_backend)

        self._beta = self._get_param_set_values(config['beta'])
        if len(self._beta) > 2:
            warnings.warn('siman_bench does not support non-linear schedules')

        self._solver = config['solver']

        self._sweeps = config['sweeps']
        self._detailed_log = {'beta':self._beta}

        self._detailed_log['sweeps'] = self._sweeps
        self._detailed_log['solver'] = self._solver

    def _make_schedule(self, sweeps = None, param_set = None):
        if not sweeps:
            sweeps = self._sweeps
        if not param_set:
            param_set = {}
            param_set['beta'] = self._beta
        config = {}
        config['param_set'] = {'beta':param_set['beta']}
        config['sweeps'] = sweeps
        config['solver'] = self._solver
        return json.dumps(config, indent=1)
        

