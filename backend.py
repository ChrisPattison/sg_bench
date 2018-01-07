import pandas as pd
import numpy as np
import tempfile
import copy
import subprocess
import multiprocessing
import dispy
import bondfile
import propanelib
import pt_propanelib
# psutil

def run_restart(schedule, instance, ground_energy = None): # schedule, instance
    import bondfile
    import pt_propanelib

    with tempfile.NamedTemporaryFile('w') as schedule_file:
        # Write schedule
        schedule_file.write(schedule)
        schedule_file.flush()
        with tempfile.NamedTemporaryFile('w') as bonds_file:
            # Write bondfile
            bondfile.write_bondfile(instance['bonds'], bonds_file)
            bonds_file.flush()

            # Run solver
            restart_data = []
            command = ['propane_ptsvmc', '-m', 'pt', schedule_file.name, bonds_file.name]
            # Optional ground state energy specification
            if ground_energy is not None:
                command.extend(['-p', str(ground_energy)])

            output = subprocess.check_output(command, universal_newlines=True)
            if 'returned non-zero exit status' in output:
                raise RuntimeException(output)
            output = output.split('\n')

            restart_data = pt_propanelib.extract_data(output)
    return restart_data

def get_backend(dispyconf = None):
    if dispyconf:
        return remoterun(dispyconf)
    else:
        return localrun()

class localrun:
    def __init__(self):
        cores = -1
        # Attempt to find out the number of physical cores using psutil
        try:
            import psutil
            cores = psutil.cpu_count(logical=False)
        except ImportError:
            cores = multiprocessing.cpu_count()
        self._cores = cores

    def run_instances(self, schedule, instances, restarts, statistics=True):

        pool = multiprocessing.Pool(self._cores)
        for i in instances:
            ground_energy = None if statistics else i['ground_energy']

            # deepcopy required since i is not picklable
            i_copy = copy.deepcopy(i)
            i['results'] = []
            for r in range(restarts):
                i['results'].append(pool.apply_async(run_restart, args=(schedule, i_copy, ground_energy)))

        for i in instances:
            i['results'] = [job.get() for job in i['results']]
            # print(i['results'])
            for index, df in enumerate(i['results']):
                df['restart'] = index
            i['results'] = pd.concat(i['results'])

            if not statistics:
                i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)
        return instances

class remoterun:
    def __init__(self, conf):
        self._dispyconf = conf
        self._cluster = dispy.JobCluster(run_restart, depends=[bondfile, propanelib, pt_propanelib], \
            loglevel=dispy.logger.CRITICAL, pulse_interval=2, reentrant=True, ping_interval=1,
            ext_ip_addr=self._dispyconf['ext_ip_addr'], nodes=self._dispyconf['nodes'])
    
    def run_instances(self, schedule, instances, restarts, statistics=True):
        
        for i in instances:
            ground_energy = None if statistics else i['ground_energy']

            # deepcopy required since i is not picklable
            i_copy = copy.deepcopy(i)
            i['results'] = []
            for r in range(restarts):
                i['results'].append(self._cluster.submit(schedule, i_copy, ground_energy = ground_energy))

        _cluster.wait()
        for i in instances:
            for job in i['results']:
                if job.exception is not None:
                    print(job.exception)
            i['results'] = [job() for job in i['results']]
            # print(i['results'])
            for index, df in enumerate(i['results']):
                df['restart'] = index
            i['results'] = pd.concat(i['results'])

            if not statistics:
                i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)
        return instances