import pandas as pd
import numpy as np
import os
import tempfile
import copy
import io
import subprocess
import multiprocessing
import bondfile
import propanelib
import pt_propanelib
import slurm
import ssh
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
            command = ['quit_solve', schedule_file.name, bonds_file.name]
            # Optional ground state energy specification
            if ground_energy is not None:
                command.extend(['-p', str(ground_energy)])

            output = subprocess.check_output(command, universal_newlines=True)
            if 'returned non-zero exit status' in output:
                raise RuntimeException(output)
            output = output.split('\n')

            restart_data = pt_propanelib.extract_data(output)
    return restart_data

def get_backend(slurmconf = None):
    if slurmconf:
        return slurmrun(slurmconf)
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
            ground_energy = None if statistics else i['target_energy']

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

    def close():
        return None


class slurmrun:
    def __init__(self, slurmconf):
        self._ssh = ssh.ssh_wrapper(slurmconf)
        self._slurmconf = slurmconf

    def run_instances(self, schedule, instances, restarts, statistics=True):
        with slurm.slurm(self._slurmconf, ssh=self._ssh) as slurm_wrapper:
            commands = []

            schedule_path = slurm_wrapper.put_temp_file(schedule)
            for i in instances:
                bonds = io.StringIO()
                bondfile.write_bondfile(i['bonds'], bonds)
                instance_path = slurm_wrapper.put_temp_file(bonds.getvalue())
                output_path = instance_path + '.out'
                slurm_wrapper.reg_temp_file(output_path)
                commands.append('launch_restarts.py {} {} {} > {}'.format(schedule_path, instance_path, restarts, output_path))
                i['output_file'] = output_path

            slurm_wrapper.submit_job_array(commands)

            for i in instances:
                i['results'] = pd.read_csv(io.StringIO(self._ssh.get_string(i['output_file'])))
                if not statistics:
                    i['results'] = i['results'].groupby(['restart']).apply(lambda d: d[d['Total_Sweeps'] == d['Total_Sweeps'].max()]).reset_index(drop=True)

        return instances

    def close():
        self._ssh.close()