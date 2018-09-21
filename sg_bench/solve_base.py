from sg_bench import backend

class solve_base:
    def __init__(self, config, borrowed_backend):
        if not hasattr(self, '_detailed_log'):
            self._detailed_log = {}

        self._success_prob = 0.99

        self._machine_readable = config['machine_readable']
        self._hit_criteria = config.get('hit_criteria', 1e-12)

        self._restarts = config.get('bench_restarts', 100)
        self._gse_target = config.get('gse_target', 1.00)

        self._slurm = config.get('slurm', None)
        self._backend = (borrowed_backend if borrowed_backend else 
            backend.get_backend(self._launcher_command, slurmconf = self._slurm))

    def _get_param_set_values(self, dictionary):
        param_set = {}
        param_set['points'] = dictionary['points']
        return param_set

    def _output(self, string):
        if not self._machine_readable:
            print(string)

    def get_full_data(self):
        return self._detailed_log
    
    def get_backend(self):
        return self._backend